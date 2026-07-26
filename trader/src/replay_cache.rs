use crate::broker::{Bar, BarKind, BarType, BrokerKind, CandleMode, ReplayDownloadCacheTarget};
use crate::config::TradingEnvironment;
use crate::replay_download::{
    DownloadWindow, HistoricalDownloadTelemetry, split_download_window,
};
use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ConvertedType, Type as PhysicalType};
use parquet::errors::Result as ParquetResult;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{ChunkReader, Length};
use parquet::file::statistics::Statistics;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::AsRawFd;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::FileExt;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const MANIFEST_FILE_NAME: &str = "manifest.json";
pub const MANIFEST_VERSION: u32 = 1;
pub const SERVER_BARS_SCHEMA_VERSION: u32 = 1;
pub const RAW_TICKS_SCHEMA_VERSION: u32 = 2;
const RAW_TICKS_LEGACY_SCHEMA_VERSION: u32 = 1;
/// Maximum rows retained by the Arrow writer before it flushes a Parquet row group.
pub const PARQUET_ROW_GROUP_ROWS: usize = 65_536;
/// Maximum source rows materialized into one Arrow `RecordBatch` while writing.
pub const PARQUET_WRITE_BATCH_ROWS: usize = 8_192;
/// Maximum rows decoded into one Arrow `RecordBatch` while streaming a cache file.
pub const PARQUET_READ_BATCH_ROWS: usize = 8_192;
const PARQUET_COMPRESSION_LABEL: &str = "snappy";
static CACHE_FILE_VERSION: AtomicU64 = AtomicU64::new(0);
const MANIFEST_LOCK_FILE_NAME: &str = ".manifest.lock";
const MANIFEST_LOCK_WAIT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawTickReadPolicy {
    LegacyV1SequenceIds,
    StrictV2ProviderIds,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug, Clone)]
struct PositionIndependentFile {
    file: Arc<File>,
    len: u64,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl PositionIndependentFile {
    fn new(file: Arc<File>) -> Result<Self> {
        let len = file
            .metadata()
            .context("inspect leased replay cache file")?
            .len();
        Ok(Self { file, len })
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Length for PositionIndependentFile {
    fn len(&self) -> u64 {
        self.len
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug)]
struct PositionIndependentRead {
    file: Arc<File>,
    offset: u64,
    len: u64,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Read for PositionIndependentRead {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() || self.offset >= self.len {
            return Ok(0);
        }
        let remaining = usize::try_from((self.len - self.offset).min(buffer.len() as u64))
            .unwrap_or(buffer.len());
        loop {
            match self.file.read_at(&mut buffer[..remaining], self.offset) {
                Ok(read) => {
                    self.offset = self.offset.saturating_add(read as u64);
                    return Ok(read);
                }
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ChunkReader for PositionIndependentFile {
    type T = PositionIndependentRead;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        if start > self.len {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Parquet read starts beyond end of file: {start} > {}",
                    self.len
                ),
            )
            .into());
        }
        Ok(PositionIndependentRead {
            file: self.file.clone(),
            offset: start,
            len: self.len,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let end = start.checked_add(length as u64).ok_or_else(|| {
            std::io::Error::new(ErrorKind::InvalidInput, "Parquet byte range overflow")
        })?;
        if end > self.len {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Parquet byte range exceeds file length: {start}..{end} > {}",
                    self.len
                ),
            )
            .into());
        }

        let mut bytes = vec![0_u8; length];
        let mut read = 0usize;
        while read < length {
            match self.file.read_at(&mut bytes[read..], start + read as u64) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        ErrorKind::UnexpectedEof,
                        format!(
                            "short position-independent read at byte {}",
                            start + read as u64
                        ),
                    )
                    .into());
                }
                Ok(count) => read += count,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err.into()),
            }
        }
        Ok(Bytes::from(bytes))
    }
}

struct ReplayManifestLock {
    file: File,
}

impl ReplayManifestLock {
    fn acquire(dataset_dir: &Path) -> Result<Self> {
        Self::acquire_with_timeout(dataset_dir, MANIFEST_LOCK_WAIT)
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn acquire_with_timeout(dataset_dir: &Path, timeout: Duration) -> Result<Self> {
        fs::create_dir_all(dataset_dir)
            .with_context(|| format!("create {}", dataset_dir.display()))?;
        let lock_path = dataset_dir.join(MANIFEST_LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&lock_path)
            .with_context(|| format!("open replay cache manifest lock {}", lock_path.display()))?;
        let started = Instant::now();

        loop {
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                return Ok(Self { file });
            }

            let err = std::io::Error::last_os_error();
            let raw_error = err.raw_os_error();
            if raw_error != Some(libc::EWOULDBLOCK)
                && raw_error != Some(libc::EAGAIN)
                && raw_error != Some(libc::EINTR)
            {
                return Err(err).with_context(|| {
                    format!("lock replay cache manifest {}", lock_path.display())
                });
            }
            if started.elapsed() >= timeout {
                bail!(
                    "timed out waiting for replay cache manifest lock {} after {} ms",
                    lock_path.display(),
                    timeout.as_millis()
                );
            }
            let retry_delay =
                Duration::from_millis(20).min(timeout.saturating_sub(started.elapsed()));
            std::thread::sleep(retry_delay);
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn acquire_with_timeout(dataset_dir: &Path, timeout: Duration) -> Result<Self> {
        let _ = (dataset_dir, timeout);
        bail!(
            "replay cache manifest locking is unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Drop for ReplayManifestLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheSourceKind {
    ServerBars,
    RawTicks,
    Mixed,
    DerivedBars,
    DomStream,
    LocalText,
}

impl ReplayCacheSourceKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ServerBars => "server bars",
            Self::RawTicks => "raw ticks",
            Self::Mixed => "mixed sources",
            Self::DerivedBars => "derived bars",
            Self::DomStream => "L2/DOM",
            Self::LocalText => "local text",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::ServerBars => "server-bars",
            Self::RawTicks => "raw-ticks",
            Self::Mixed => "mixed",
            Self::DerivedBars => "derived-bars",
            Self::DomStream => "l2-dom",
            Self::LocalText => "local-text",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheFileFormat {
    Parquet,
    Jsonl,
    Csv,
    Text,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheInstrument {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchange: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheContract {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiration: Option<NaiveDate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheTickSpecs {
    pub tick_size: f64,
    pub value_per_point: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheMetadataContext {
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,
    #[serde(default)]
    pub accounts: Vec<ReplayCacheMetadataAccount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accounts_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accounts_fetched_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheMetadataAccount {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheMetadataSnapshot {
    pub endpoint: String,
    pub fetched_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_timestamp: Option<DateTime<Utc>>,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheSuggestedCoverage {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub basis: String,
    pub estimated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheContractMetadata {
    pub context: ReplayCacheMetadataContext,
    pub contract: ReplayCacheMetadataSnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maturity: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maturity_chain: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_sessions: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_margins: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_margins: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fee_params: Option<ReplayCacheMetadataSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_coverage: Option<ReplayCacheSuggestedCoverage>,
}

#[derive(Debug, PartialEq, Eq)]
struct ReplayCacheMetadataIdentity {
    provider: BrokerKind,
    env: TradingEnvironment,
    user_id: i64,
    account_ids: Vec<i64>,
    contract_id: i64,
    product_id: i64,
}

fn contract_metadata_identity(
    metadata: &ReplayCacheContractMetadata,
) -> Option<ReplayCacheMetadataIdentity> {
    let user_id = metadata.context.user_id?;
    let mut account_ids = metadata
        .context
        .accounts
        .iter()
        .map(|account| account.id)
        .collect::<Vec<_>>();
    account_ids.sort_unstable();
    account_ids.dedup();
    if account_ids.is_empty() {
        return None;
    }
    let contract_id = metadata.contract.payload.get("id")?.as_i64()?;
    let product_id = metadata
        .product
        .as_ref()
        .and_then(|snapshot| snapshot.payload.get("id"))
        .and_then(Value::as_i64)
        .or_else(|| {
            metadata
                .maturity
                .as_ref()
                .and_then(|snapshot| snapshot.payload.get("productId"))
                .and_then(Value::as_i64)
        })?;
    Some(ReplayCacheMetadataIdentity {
        provider: metadata.context.provider,
        env: metadata.context.env,
        user_id,
        account_ids,
        contract_id,
        product_id,
    })
}

fn merge_contract_metadata(
    existing: Option<ReplayCacheContractMetadata>,
    incoming: Option<ReplayCacheContractMetadata>,
) -> Option<ReplayCacheContractMetadata> {
    let Some(mut incoming) = incoming else {
        return None;
    };
    let Some(existing) = existing else {
        return Some(incoming);
    };
    let identities_match = contract_metadata_identity(&existing)
        .zip(contract_metadata_identity(&incoming))
        .is_some_and(|(existing, incoming)| existing == incoming);
    if !identities_match {
        return Some(incoming);
    }
    incoming.maturity = incoming.maturity.or(existing.maturity);
    incoming.maturity_chain = incoming.maturity_chain.or(existing.maturity_chain);
    incoming.product = incoming.product.or(existing.product);
    incoming.product_sessions = incoming.product_sessions.or(existing.product_sessions);
    incoming.product_margins = incoming.product_margins.or(existing.product_margins);
    incoming.contract_margins = incoming.contract_margins.or(existing.contract_margins);
    incoming.fee_params = incoming.fee_params.or(existing.fee_params);
    incoming.suggested_coverage = incoming.suggested_coverage.or(existing.suggested_coverage);
    Some(incoming)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheCoverage {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trading_date: Option<NaiveDate>,
}

impl ReplayCacheCoverage {
    pub fn contains(&self, requested: &Self) -> bool {
        self.start <= requested.start && self.end >= requested.end
    }

    pub fn label(&self) -> String {
        match self.trading_date {
            Some(date) => format!(
                "{} {} to {}",
                date,
                self.start.format("%H:%M:%S UTC"),
                self.end.format("%H:%M:%S UTC")
            ),
            None => format!(
                "{} to {}",
                self.start.format("%Y-%m-%d %H:%M:%S UTC"),
                self.end.format("%Y-%m-%d %H:%M:%S UTC")
            ),
        }
    }
}

/// Exact replay-time filter with half-open UTC semantics: `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayCacheTimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl ReplayCacheTimeRange {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Self> {
        if start >= end {
            bail!("replay cache timestamp range requires start before end");
        }
        start
            .timestamp_nanos_opt()
            .context("replay cache range start is outside supported nanosecond range")?;
        end.timestamp_nanos_opt()
            .context("replay cache range end is outside supported nanosecond range")?;
        Ok(Self { start, end })
    }

    fn bounds_ns(self) -> Result<(i64, i64)> {
        Ok((
            self.start
                .timestamp_nanos_opt()
                .context("replay cache range start is outside supported nanosecond range")?,
            self.end
                .timestamp_nanos_opt()
                .context("replay cache range end is outside supported nanosecond range")?,
        ))
    }

    fn contains_ns(self, ts_ns: i64) -> Result<bool> {
        let (start_ns, end_ns) = self.bounds_ns()?;
        Ok(ts_ns >= start_ns && ts_ns < end_ns)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayCacheParquetReadStats {
    pub total_row_groups: usize,
    pub selected_row_groups: usize,
    pub pruned_row_groups: usize,
    pub record_batches: usize,
    pub max_record_batch_rows: usize,
    pub decoded_rows: u64,
    pub emitted_rows: u64,
    pub first_timestamp_ns: Option<i64>,
    pub last_timestamp_ns: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheMarketShape {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bar_type: Option<BarType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_mode: Option<CandleMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_template: Option<String>,
}

impl ReplayCacheMarketShape {
    pub fn supports(&self, bar_type: BarType, candle_mode: CandleMode) -> bool {
        let bar_matches = self.bar_type.is_none_or(|cached| cached == bar_type);
        let candle_matches = self
            .chart_mode
            .is_none_or(|cached| cached == bar_type.effective_candle_mode(candle_mode));
        bar_matches && candle_matches
    }

    pub fn label(&self) -> String {
        match (self.bar_type, self.chart_mode) {
            (Some(bar_type), Some(candle_mode)) => bar_type.mode_label(candle_mode),
            (Some(bar_type), None) => bar_type.label(),
            (None, Some(candle_mode)) => candle_mode.label().to_string(),
            (None, None) => "unshaped market data".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheDataHash {
    pub algorithm: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheDataFile {
    pub relative_path: PathBuf,
    pub source_kind: ReplayCacheSourceKind,
    pub format: ReplayCacheFileFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<String>,
    pub market_shape: ReplayCacheMarketShape,
    pub row_count: u64,
    pub first_timestamp: DateTime<Utc>,
    pub last_timestamp: DateTime<Utc>,
    /// Provider request coverage for immutable chunk files. Legacy one-file
    /// entries omit these fields and continue to use manifest coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_start: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_end: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_hash: Option<ReplayCacheDataHash>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheAppMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheManifest {
    pub manifest_version: u32,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub display_name: String,
    pub coverage: ReplayCacheCoverage,
    /// Fully completed half-open raw-tick request coverage. This is published
    /// only after every checkpoint leaf has explicit provider completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_raw_tick_coverage: Option<ReplayCacheCoverage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_raw_tick_windows: Vec<DownloadWindow>,
    pub source_kind: ReplayCacheSourceKind,
    #[serde(default)]
    pub download_request: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_metadata: Option<ReplayCacheContractMetadata>,
    #[serde(default)]
    pub files: Vec<ReplayCacheDataFile>,
    #[serde(default)]
    pub app: Option<ReplayCacheAppMetadata>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub badges: Vec<String>,
    #[serde(default)]
    pub available_bar_shapes: Vec<BarType>,
    #[serde(default)]
    pub available_chart_modes: Vec<CandleMode>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl ReplayCacheManifest {
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut manifest: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        manifest.normalize_derived_fields();
        Ok(manifest)
    }

    pub fn supports_replay(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        if !self.errors.is_empty() {
            return false;
        }
        if requested_coverage.is_some_and(|requested| !self.coverage.contains(requested)) {
            return false;
        }
        self.files
            .iter()
            .any(|file| replay_file_can_serve(file, bar_type, candle_mode))
    }

    pub fn normalize_derived_fields(&mut self) {
        self.source_kind = manifest_source_kind_from_files(&self.files, self.source_kind);
        if self.available_bar_shapes.is_empty() {
            let mut shapes = Vec::new();
            for file in &self.files {
                if let Some(bar_type) = file.market_shape.bar_type
                    && !shapes.contains(&bar_type)
                {
                    shapes.push(bar_type);
                }
            }
            self.available_bar_shapes = shapes;
        }

        if self.available_chart_modes.is_empty() {
            let mut modes = Vec::new();
            for file in &self.files {
                if let Some(mode) = file.market_shape.chart_mode
                    && !modes.contains(&mode)
                {
                    modes.push(mode);
                }
            }
            self.available_chart_modes = modes;
        }

        if self.badges.is_empty() {
            self.badges = self.derived_badges();
        }
    }

    pub fn derived_badges(&self) -> Vec<String> {
        let mut badges = BTreeSet::new();
        if self.source_kind != ReplayCacheSourceKind::Mixed {
            badges.insert(self.source_kind.badge().to_string());
        }
        for file in &self.files {
            badges.insert(file.source_kind.badge().to_string());
            if file.market_shape.bar_type.is_some_and(|bar_type| {
                matches!(
                    bar_type.kind(),
                    BarKind::Tick | BarKind::Volume | BarKind::Range
                )
            }) {
                badges.insert(file.market_shape.label().to_ascii_lowercase());
            }
            if let Some(chart_mode) = file.market_shape.chart_mode {
                badges.insert(chart_mode.label().to_ascii_lowercase());
            }
        }
        badges.into_iter().collect()
    }

    #[cfg(test)]
    fn row_count_total(&self) -> u64 {
        self.files.iter().map(|file| file.row_count).sum()
    }

    pub fn preferred_row_count_total(&self) -> u64 {
        self.files
            .iter()
            .filter(|file| {
                !(file.format == ReplayCacheFileFormat::Jsonl
                    && self.files.iter().any(|candidate| {
                        candidate.source_kind == file.source_kind
                            && candidate.format == ReplayCacheFileFormat::Parquet
                            && candidate.market_shape == file.market_shape
                    }))
            })
            .map(|file| file.row_count)
            .sum()
    }

    pub fn has_source_kind(&self, source_kind: ReplayCacheSourceKind) -> bool {
        self.files
            .iter()
            .any(|file| file.source_kind == source_kind)
    }

    pub fn downloadable_source_kinds(&self) -> Vec<ReplayCacheSourceKind> {
        [
            ReplayCacheSourceKind::ServerBars,
            ReplayCacheSourceKind::RawTicks,
        ]
        .into_iter()
        .filter(|source_kind| self.has_source_kind(*source_kind))
        .collect()
    }

    pub fn available_shapes_label(&self) -> String {
        if self.available_bar_shapes.is_empty() {
            return "none listed".to_string();
        }
        self.available_bar_shapes
            .iter()
            .map(|bar_type| bar_type.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn available_chart_modes_label(&self) -> String {
        if self.available_chart_modes.is_empty() {
            return "none listed".to_string();
        }
        self.available_chart_modes
            .iter()
            .map(|mode| mode.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn badges_label(&self) -> String {
        if self.badges.is_empty() {
            return "none".to_string();
        }
        self.badges.join(", ")
    }
}

fn replay_file_can_serve(
    file: &ReplayCacheDataFile,
    bar_type: BarType,
    candle_mode: CandleMode,
) -> bool {
    file.errors.is_empty()
        && matches!(
            file.source_kind,
            ReplayCacheSourceKind::ServerBars
                | ReplayCacheSourceKind::DerivedBars
                | ReplayCacheSourceKind::LocalText
        )
        && file.market_shape.supports(bar_type, candle_mode)
}

fn manifest_source_kind_from_files(
    files: &[ReplayCacheDataFile],
    fallback: ReplayCacheSourceKind,
) -> ReplayCacheSourceKind {
    let Some(first) = files.first().map(|file| file.source_kind) else {
        return fallback;
    };
    if files.iter().all(|file| file.source_kind == first) {
        first
    } else {
        ReplayCacheSourceKind::Mixed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheServerBarRow {
    pub timestamp: DateTime<Utc>,
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
}

impl ReplayCacheServerBarRow {
    pub fn from_bar(bar: &Bar) -> Self {
        Self {
            timestamp: DateTime::<Utc>::from_timestamp_nanos(bar.ts_ns),
            ts_ns: bar.ts_ns,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheRawTickRow {
    pub timestamp: DateTime<Utc>,
    pub ts_ns: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick_id: Option<i64>,
    pub price: f64,
    pub size: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trade_date: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_ts_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_price_ticks: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ReplayCacheServerBarsWrite {
    pub cache_root: PathBuf,
    pub target: Option<ReplayDownloadCacheTarget>,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub request_start: DateTime<Utc>,
    pub request_end: DateTime<Utc>,
    pub source_kind: ReplayCacheSourceKind,
    pub download_request: Value,
    pub bar_type: BarType,
    pub tick_specs: ReplayCacheTickSpecs,
    pub contract_metadata: Option<ReplayCacheContractMetadata>,
    pub session_template: Option<String>,
    pub bars: Vec<Bar>,
    pub warnings: Vec<String>,
    pub display_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReplayCacheRawTicksWrite {
    pub cache_root: PathBuf,
    pub target: Option<ReplayDownloadCacheTarget>,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub request_start: DateTime<Utc>,
    pub request_end: DateTime<Utc>,
    pub download_request: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub contract_metadata: Option<ReplayCacheContractMetadata>,
    pub session_template: Option<String>,
    pub ticks: Vec<ReplayCacheRawTickRow>,
    pub warnings: Vec<String>,
    pub display_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCacheWriteOutcome {
    pub dataset_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub data_path: PathBuf,
    pub row_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheRawTicksNormalizeOutcome {
    pub rows: Vec<ReplayCacheRawTickRow>,
    pub duplicate_tick_ids: usize,
    pub dropped_rows: usize,
}

pub const RAW_TICK_CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheRawTickCheckpointIdentity {
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub request: DownloadWindow,
    pub chunk_seconds: i64,
}

impl ReplayCacheRawTickCheckpointIdentity {
    pub fn stable_key(&self) -> String {
        format!(
            "{}|{:?}|{}|{}|{}|{}|{}|{}",
            self.provider.label(),
            self.env,
            self.instrument.symbol.trim().to_ascii_uppercase(),
            self.contract.symbol.trim().to_ascii_uppercase(),
            self.contract.id.unwrap_or_default(),
            self.request.start.timestamp_millis(),
            self.request.end.timestamp_millis(),
            self.chunk_seconds,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheRawTickChunkStatus {
    Pending,
    Split {
        left: DownloadWindow,
        right: DownloadWindow,
    },
    Completed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file: Option<ReplayCacheDataFile>,
        telemetry: HistoricalDownloadTelemetry,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheRawTickCheckpointChunk {
    pub window: DownloadWindow,
    pub status: ReplayCacheRawTickChunkStatus,
    #[serde(default)]
    pub attempts: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheRawTickCheckpoint {
    pub checkpoint_version: u32,
    pub identity_key: String,
    pub identity: ReplayCacheRawTickCheckpointIdentity,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub chunks: Vec<ReplayCacheRawTickCheckpointChunk>,
}

impl ReplayCacheRawTickCheckpoint {
    pub fn pending_windows(&self) -> Vec<DownloadWindow> {
        let mut windows = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                matches!(chunk.status, ReplayCacheRawTickChunkStatus::Pending)
                    .then_some(chunk.window)
            })
            .collect::<Vec<_>>();
        windows.sort_by_key(|window| window.start);
        windows
    }

    pub fn completed_leaf_count(&self) -> usize {
        self.chunks
            .iter()
            .filter(|chunk| {
                matches!(chunk.status, ReplayCacheRawTickChunkStatus::Completed { .. })
            })
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct ReplayCacheRawTickChunkPlanWrite {
    pub cache_root: PathBuf,
    pub target: Option<ReplayDownloadCacheTarget>,
    pub identity: ReplayCacheRawTickCheckpointIdentity,
    pub download_request: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub contract_metadata: Option<ReplayCacheContractMetadata>,
    pub session_template: Option<String>,
    pub warnings: Vec<String>,
    pub display_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheRawTickCheckpointState {
    pub dataset_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub checkpoint: ReplayCacheRawTickCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCacheRawTickChunkWriteOutcome {
    pub dataset_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub data_path: Option<PathBuf>,
    pub row_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCacheChunkedWriteOutcome {
    pub dataset_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub data_paths: Vec<PathBuf>,
    pub row_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheResolvedServerBarsFile {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheLoadedServerBars {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
    pub bars: Vec<Bar>,
}

#[derive(Debug, Clone)]
pub struct ReplayCacheResolvedRawTicksFile {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    /// Open lease on the immutable versioned file. On Unix this keeps active replay
    /// readable after a cache refresh unlinks the superseded pathname.
    pub data_file: Arc<File>,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
}

#[derive(Debug, Clone)]
pub struct ReplayCacheResolvedRawTicks {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub files: Vec<ReplayCacheResolvedRawTicksFile>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheLoadedRawTicks {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
    pub ticks: Vec<ReplayCacheRawTickRow>,
}

#[allow(dead_code)]
pub fn write_server_bars_jsonl_cache(
    write: ReplayCacheServerBarsWrite,
) -> Result<ReplayCacheWriteOutcome> {
    if write.source_kind != ReplayCacheSourceKind::ServerBars {
        bail!("JSONL server-bar cache writer only accepts server-bars source data");
    }

    let rows = normalize_server_bar_rows(write.bars.clone());
    if rows.is_empty() {
        bail!("server-bar download returned no usable bars");
    }

    let dataset_dir = replay_cache_write_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract,
        write.request_start.date_naive(),
        write.target.as_ref(),
    )?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let relative_path = versioned_cache_relative_path(&server_bars_relative_path(
        write.request_start,
        write.request_end,
        write.bar_type,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut data = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut data, row)?;
        data.push(b'\n');
    }
    write_bytes_atomically(&data_path, &data)?;

    let first_timestamp = rows
        .first()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let last_timestamp = rows
        .last()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let row_count = rows.len() as u64;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::ServerBars,
        format: ReplayCacheFileFormat::Jsonl,
        schema_version: Some(SERVER_BARS_SCHEMA_VERSION),
        compression: None,
        market_shape: ReplayCacheMarketShape {
            bar_type: Some(write.bar_type),
            chart_mode: None,
            session_template: write.session_template.clone(),
        },
        row_count,
        first_timestamp,
        last_timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: fnv1a64_hex(&data),
        }),
        warnings: write.warnings.clone(),
        errors: Vec::new(),
    };

    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.provider,
            env: write.env,
            instrument: write.instrument.clone(),
            contract: write.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                replay_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                    write.bar_type,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };

    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument;
    manifest.contract = write.contract;
    manifest.source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.contract_metadata =
        merge_contract_metadata(manifest.contract_metadata.take(), write.contract_metadata);
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = write.warnings;
    manifest.errors.clear();
    if let Some(display_name) = write
        .display_name
        .clone()
        .filter(|name| !name.trim().is_empty())
    {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags.clone() {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes;
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.market_shape.bar_type == Some(write.bar_type)
                && file.format == ReplayCacheFileFormat::Jsonl
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::ServerBars
            && file.market_shape.bar_type == Some(write.bar_type)
            && file.format == ReplayCacheFileFormat::Jsonl)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::ServerBars);
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.request_start);
    manifest.available_bar_shapes.clear();
    for file in &manifest.files {
        if let Some(bar_type) = file.market_shape.bar_type
            && !manifest.available_bar_shapes.contains(&bar_type)
        {
            manifest.available_bar_shapes.push(bar_type);
        }
    }
    manifest.available_chart_modes = if write.bar_type.supports_candle_mode() {
        vec![CandleMode::Standard, CandleMode::HeikinAshi]
    } else {
        vec![CandleMode::Standard]
    };
    manifest.badges = manifest.derived_badges();

    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);

    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path,
        data_path,
        row_count,
    })
}

pub fn write_server_bars_parquet_cache(
    write: ReplayCacheServerBarsWrite,
) -> Result<ReplayCacheWriteOutcome> {
    if write.source_kind != ReplayCacheSourceKind::ServerBars {
        bail!("Parquet server-bar cache writer only accepts server-bars source data");
    }

    let rows = normalize_server_bar_rows(write.bars.clone());
    if rows.is_empty() {
        bail!("server-bar download returned no usable bars");
    }
    let dataset_dir = replay_cache_write_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract,
        write.request_start.date_naive(),
        write.target.as_ref(),
    )?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let relative_path = versioned_cache_relative_path(&server_bars_parquet_relative_path(
        write.request_start,
        write.request_end,
        write.bar_type,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_server_bars_parquet_file(&data_path, &rows)?;
    let data_hash = fnv1a64_file_hex(&data_path)?;
    let first_timestamp = rows.first().expect("rows are non-empty").timestamp;
    let last_timestamp = rows.last().expect("rows are non-empty").timestamp;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::ServerBars,
        format: ReplayCacheFileFormat::Parquet,
        schema_version: Some(SERVER_BARS_SCHEMA_VERSION),
        compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
        market_shape: ReplayCacheMarketShape {
            bar_type: Some(write.bar_type),
            chart_mode: None,
            session_template: write.session_template.clone(),
        },
        row_count: rows.len() as u64,
        first_timestamp,
        last_timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: data_hash,
        }),
        warnings: write.warnings.clone(),
        errors: Vec::new(),
    };
    let outcome = upsert_server_bars_manifest(
        &write,
        &dataset_dir,
        data_file,
        first_timestamp,
        last_timestamp,
    )?;
    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path: outcome.0,
        data_path,
        row_count: rows.len() as u64,
    })
}

fn upsert_server_bars_manifest(
    write: &ReplayCacheServerBarsWrite,
    dataset_dir: &Path,
    data_file: ReplayCacheDataFile,
    first_timestamp: DateTime<Utc>,
    last_timestamp: DateTime<Utc>,
) -> Result<(PathBuf, ReplayCacheManifest)> {
    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.provider,
            env: write.env,
            instrument: write.instrument.clone(),
            contract: write.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                replay_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                    write.bar_type,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };
    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument.clone();
    manifest.contract = write.contract.clone();
    manifest.source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.download_request = write.download_request.clone();
    manifest.tick_specs = write.tick_specs.clone();
    manifest.contract_metadata = merge_contract_metadata(
        manifest.contract_metadata.take(),
        write.contract_metadata.clone(),
    );
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = write.warnings.clone();
    manifest.errors.clear();
    if let Some(display_name) = write
        .display_name
        .clone()
        .filter(|name| !name.trim().is_empty())
    {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags.clone() {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes.clone();
    let shape = data_file.market_shape.bar_type;
    let format = data_file.format.clone();
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.market_shape.bar_type == shape
                && file.format == format
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::ServerBars
            && file.market_shape.bar_type == shape
            && file.format == format)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::ServerBars);
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.request_start);
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect();
    manifest
        .available_bar_shapes
        .sort_by_key(|bar_type| bar_type.value());
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = if write.bar_type.supports_candle_mode() {
        vec![CandleMode::Standard, CandleMode::HeikinAshi]
    } else {
        vec![CandleMode::Standard]
    };
    manifest.badges = manifest.derived_badges();
    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);
    Ok((manifest_path, manifest))
}

pub fn write_raw_ticks_parquet_cache(
    write: ReplayCacheRawTicksWrite,
) -> Result<ReplayCacheWriteOutcome> {
    let normalized = normalize_raw_tick_rows(write.ticks);
    if normalized.rows.is_empty() {
        bail!("raw tick download returned no usable ticks");
    }
    let missing_tick_ids = normalized
        .rows
        .iter()
        .filter(|row| row.tick_id.is_none())
        .count();
    if missing_tick_ids > 0 {
        bail!(
            "raw tick provider response omitted stable tick IDs for {missing_tick_ids} of {} usable row(s); schema v2 cache was not written because replay cannot distinguish duplicate provider trades safely",
            normalized.rows.len()
        );
    }
    validate_raw_tick_id_write_invariant(&normalized.rows)?;

    let mut warnings = write.warnings;
    if normalized.duplicate_tick_ids > 0 {
        warnings.push(format!(
            "Dropped {} duplicate raw tick id(s) while normalizing cache rows.",
            normalized.duplicate_tick_ids
        ));
    }
    if normalized.dropped_rows > 0 {
        warnings.push(format!(
            "Dropped {} malformed raw tick row(s) while normalizing cache rows.",
            normalized.dropped_rows
        ));
    }

    let dataset_dir = replay_cache_write_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract,
        write.request_start.date_naive(),
        write.target.as_ref(),
    )?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let relative_path = versioned_cache_relative_path(&raw_ticks_relative_path(
        write.request_start,
        write.request_end,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_raw_ticks_parquet_file(&data_path, &normalized.rows)?;

    let data_hash = fnv1a64_file_hex(&data_path)?;
    let first_timestamp = normalized
        .rows
        .first()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let last_timestamp = normalized
        .rows
        .last()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let row_count = normalized.rows.len() as u64;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::RawTicks,
        format: ReplayCacheFileFormat::Parquet,
        schema_version: Some(RAW_TICKS_SCHEMA_VERSION),
        compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
        market_shape: ReplayCacheMarketShape {
            bar_type: None,
            chart_mode: None,
            session_template: write.session_template.clone(),
        },
        row_count,
        first_timestamp,
        last_timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: data_hash,
        }),
        warnings: warnings.clone(),
        errors: Vec::new(),
    };

    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.provider,
            env: write.env,
            instrument: write.instrument.clone(),
            contract: write.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                raw_ticks_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::RawTicks,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };

    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument;
    manifest.contract = write.contract;
    manifest.source_kind = ReplayCacheSourceKind::RawTicks;
    manifest.completed_raw_tick_coverage = None;
    manifest.completed_raw_tick_windows.clear();
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.contract_metadata =
        merge_contract_metadata(manifest.contract_metadata.take(), write.contract_metadata);
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = warnings;
    manifest.errors.clear();
    if let Some(display_name) = write.display_name.filter(|name| !name.trim().is_empty()) {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes;
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::RawTicks
                && file.format == ReplayCacheFileFormat::Parquet
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::RawTicks
            && file.format == ReplayCacheFileFormat::Parquet)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::RawTicks);
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.request_start);
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect::<Vec<_>>();
    manifest.available_bar_shapes.sort_by_key(|bar_type| {
        (
            match bar_type.kind() {
                BarKind::Minute => 0,
                BarKind::Second => 1,
                BarKind::Tick => 2,
                BarKind::Volume => 3,
                BarKind::Range => 4,
            },
            bar_type.value(),
        )
    });
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.chart_mode)
        .collect::<Vec<_>>();
    manifest.available_chart_modes.dedup();
    manifest.badges = manifest.derived_badges();

    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);

    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path,
        data_path,
        row_count,
    })
}

const RAW_TICK_CHECKPOINT_FILE_NAME: &str = "download-checkpoint.json";

pub fn prepare_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
    initial_windows: &[DownloadWindow],
) -> Result<ReplayCacheRawTickCheckpointState> {
    validate_exact_window_coverage(write.identity.request, initial_windows)?;
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = dataset_dir
        .join("raw-ticks")
        .join(RAW_TICK_CHECKPOINT_FILE_NAME);
    let checkpoint = if checkpoint_path.exists() {
        let checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
        validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
        checkpoint
    } else {
        let now = Utc::now();
        let checkpoint = ReplayCacheRawTickCheckpoint {
            checkpoint_version: RAW_TICK_CHECKPOINT_VERSION,
            identity_key: write.identity.stable_key(),
            identity: write.identity.clone(),
            created_at: now,
            updated_at: now,
            chunks: initial_windows
                .iter()
                .copied()
                .map(|window| ReplayCacheRawTickCheckpointChunk {
                    window,
                    status: ReplayCacheRawTickChunkStatus::Pending,
                    attempts: Vec::new(),
                })
                .collect(),
        };
        write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
        checkpoint
    };
    Ok(ReplayCacheRawTickCheckpointState {
        dataset_dir,
        checkpoint_path,
        checkpoint,
    })
}

pub fn record_raw_tick_chunk_attempt(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    sanitized_evidence: Value,
) -> Result<ReplayCacheRawTickCheckpointState> {
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = dataset_dir
        .join("raw-ticks")
        .join(RAW_TICK_CHECKPOINT_FILE_NAME);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint_identity(write, &checkpoint)?;
    let chunk = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    chunk.attempts.push(sanitized_evidence);
    checkpoint.updated_at = Utc::now();
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok(ReplayCacheRawTickCheckpointState {
        dataset_dir,
        checkpoint_path,
        checkpoint,
    })
}

pub fn split_raw_tick_checkpoint_chunk(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    session_boundaries: &[DateTime<Utc>],
    minimum: chrono::Duration,
    sanitized_evidence: Value,
) -> Result<(DownloadWindow, DownloadWindow)> {
    let (left, right) = split_download_window(window, session_boundaries, minimum)
        .context("raw-tick chunk cannot be split without violating the minimum interval")?;
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = dataset_dir
        .join("raw-ticks")
        .join(RAW_TICK_CHECKPOINT_FILE_NAME);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint_identity(write, &checkpoint)?;
    let parent = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    if !matches!(parent.status, ReplayCacheRawTickChunkStatus::Pending) {
        bail!("only a pending raw-tick checkpoint chunk can be split");
    }
    parent.attempts.push(sanitized_evidence);
    parent.status = ReplayCacheRawTickChunkStatus::Split { left, right };
    for child in [left, right] {
        checkpoint.chunks.push(ReplayCacheRawTickCheckpointChunk {
            window: child,
            status: ReplayCacheRawTickChunkStatus::Pending,
            attempts: Vec::new(),
        });
    }
    checkpoint.updated_at = Utc::now();
    validate_checkpoint_leaf_coverage(&checkpoint)?;
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok((left, right))
}

pub fn write_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    ticks: Vec<ReplayCacheRawTickRow>,
    mut telemetry: HistoricalDownloadTelemetry,
) -> Result<ReplayCacheRawTickChunkWriteOutcome> {
    if telemetry.request != window {
        bail!("raw-tick chunk telemetry request does not match checkpoint window");
    }
    if !telemetry.completed_by_eoh() {
        bail!("raw-tick chunk cannot commit without explicit provider end-of-history");
    }
    let normalized = normalize_raw_tick_rows(ticks);
    let missing_tick_ids = normalized
        .rows
        .iter()
        .filter(|row| row.tick_id.is_none())
        .count();
    if missing_tick_ids > 0 {
        bail!(
            "raw tick provider response omitted stable tick IDs for {missing_tick_ids} of {} usable row(s); chunk was not committed",
            normalized.rows.len()
        );
    }
    validate_raw_tick_id_write_invariant(&normalized.rows)?;
    let start_ns = window
        .start
        .timestamp_nanos_opt()
        .context("raw-tick chunk start is outside nanosecond range")?;
    let end_ns = window
        .end
        .timestamp_nanos_opt()
        .context("raw-tick chunk end is outside nanosecond range")?;
    if normalized
        .rows
        .iter()
        .any(|row| row.ts_ns < start_ns || row.ts_ns >= end_ns)
    {
        bail!("raw-tick chunk contains rows outside its exact half-open request window");
    }
    telemetry.normalized_rows = normalized.rows.len() as u64;
    telemetry.duplicate_rows = normalized.duplicate_tick_ids as u64;
    telemetry.dropped_rows = telemetry
        .dropped_rows
        .saturating_add(normalized.dropped_rows as u64);
    telemetry.first_timestamp = normalized.rows.first().map(|row| row.timestamp);
    telemetry.last_timestamp = normalized.rows.last().map(|row| row.timestamp);

    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = dataset_dir
        .join("raw-ticks")
        .join(RAW_TICK_CHECKPOINT_FILE_NAME);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
    let existing = checkpoint
        .chunks
        .iter()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    if let ReplayCacheRawTickChunkStatus::Completed { file, .. } = &existing.status {
        let data_path = file
            .as_ref()
            .map(|file| resolve_cache_data_path(&dataset_dir, &file.relative_path))
            .transpose()?;
        return Ok(ReplayCacheRawTickChunkWriteOutcome {
            dataset_dir,
            checkpoint_path,
            data_path,
            row_count: file.as_ref().map(|file| file.row_count).unwrap_or_default(),
        });
    }
    if !matches!(existing.status, ReplayCacheRawTickChunkStatus::Pending) {
        bail!("raw-tick checkpoint chunk is not a pending leaf");
    }

    let (data_file, data_path) = if normalized.rows.is_empty() {
        (None, None)
    } else {
        let relative_path = versioned_cache_relative_path(&raw_tick_chunk_relative_path(window));
        let data_path = dataset_dir.join(&relative_path);
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        write_raw_ticks_parquet_file(&data_path, &normalized.rows)?;
        let data_hash = fnv1a64_file_hex(&data_path)?;
        let first_timestamp = normalized.rows.first().expect("rows are non-empty").timestamp;
        let last_timestamp = normalized.rows.last().expect("rows are non-empty").timestamp;
        let mut warnings = Vec::new();
        if normalized.duplicate_tick_ids > 0 {
            warnings.push(format!(
                "Dropped {} duplicate raw tick id(s) from this chunk.",
                normalized.duplicate_tick_ids
            ));
        }
        if normalized.dropped_rows > 0 {
            warnings.push(format!(
                "Dropped {} malformed raw tick row(s) from this chunk.",
                normalized.dropped_rows
            ));
        }
        (
            Some(ReplayCacheDataFile {
                relative_path,
                source_kind: ReplayCacheSourceKind::RawTicks,
                format: ReplayCacheFileFormat::Parquet,
                schema_version: Some(RAW_TICKS_SCHEMA_VERSION),
                compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
                market_shape: ReplayCacheMarketShape {
                    bar_type: None,
                    chart_mode: None,
                    session_template: write.session_template.clone(),
                },
                row_count: normalized.rows.len() as u64,
                first_timestamp,
                last_timestamp,
                request_start: Some(window.start),
                request_end: Some(window.end),
                data_hash: Some(ReplayCacheDataHash {
                    algorithm: "fnv1a64".to_string(),
                    value: data_hash,
                }),
                warnings,
                errors: Vec::new(),
            }),
            Some(data_path),
        )
    };
    let chunk = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .expect("chunk checked above");
    chunk.status = ReplayCacheRawTickChunkStatus::Completed {
        file: data_file,
        telemetry,
    };
    checkpoint.updated_at = Utc::now();
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok(ReplayCacheRawTickChunkWriteOutcome {
        dataset_dir,
        checkpoint_path,
        data_path,
        row_count: normalized.rows.len() as u64,
    })
}

pub fn finalize_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
) -> Result<ReplayCacheChunkedWriteOutcome> {
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = dataset_dir
        .join("raw-ticks")
        .join(RAW_TICK_CHECKPOINT_FILE_NAME);
    let checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
    validate_checkpoint_leaf_coverage(&checkpoint)?;
    if !checkpoint.pending_windows().is_empty() {
        bail!("raw-tick checkpoint is incomplete; requested coverage was not published");
    }
    let mut files = checkpoint
        .chunks
        .iter()
        .filter_map(|chunk| match &chunk.status {
            ReplayCacheRawTickChunkStatus::Completed {
                file: Some(file), ..
            } => Some(file.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|file| file.request_start);
    if files.is_empty() {
        bail!("raw-tick request completed with zero usable ticks; no replay dataset was published");
    }
    validate_raw_tick_chunk_file_sequence(&dataset_dir, &files)?;

    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        let first = files.first().expect("files are non-empty");
        let last = files.last().expect("files are non-empty");
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.identity.provider,
            env: write.identity.env,
            instrument: write.identity.instrument.clone(),
            contract: write.identity.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                raw_ticks_cache_display_name(
                    &write.identity.contract.symbol,
                    write.identity.request.start,
                    write.identity.request.end,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first.first_timestamp,
                end: last.last_timestamp,
                trading_date: Some(write.identity.request.start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::RawTicks,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };
    if manifest.provider != write.identity.provider
        || manifest.env != write.identity.env
        || !manifest
            .instrument
            .symbol
            .eq_ignore_ascii_case(&write.identity.instrument.symbol)
        || !manifest
            .contract
            .symbol
            .eq_ignore_ascii_case(&write.identity.contract.symbol)
        || (manifest.contract.id.is_some()
            && write.identity.contract.id.is_some()
            && manifest.contract.id != write.identity.contract.id)
    {
        bail!("raw-tick manifest identity does not match checkpoint identity");
    }
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::RawTicks
                && file.format == ReplayCacheFileFormat::Parquet
                && !files
                    .iter()
                    .any(|replacement| replacement.relative_path == file.relative_path)
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::RawTicks
            && file.format == ReplayCacheFileFormat::Parquet)
    });
    manifest.files.extend(files.clone());
    manifest.files.sort_by(|left, right| {
        left.request_start
            .cmp(&right.request_start)
            .then_with(|| left.relative_path.cmp(&right.relative_path))
    });
    manifest.provider = write.identity.provider;
    manifest.env = write.identity.env;
    manifest.instrument = write.identity.instrument.clone();
    manifest.contract = write.identity.contract.clone();
    manifest.download_request = write.download_request.clone();
    manifest.tick_specs = write.tick_specs.clone();
    manifest.contract_metadata = merge_contract_metadata(
        manifest.contract_metadata.take(),
        write.contract_metadata.clone(),
    );
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.identity.request.start);
    manifest.completed_raw_tick_coverage = Some(ReplayCacheCoverage {
        start: write.identity.request.start,
        end: write.identity.request.end,
        trading_date: Some(write.identity.request.start.date_naive()),
    });
    manifest.completed_raw_tick_windows = checkpoint
        .chunks
        .iter()
        .filter(|chunk| {
            matches!(chunk.status, ReplayCacheRawTickChunkStatus::Completed { .. })
        })
        .map(|chunk| chunk.window)
        .collect();
    manifest
        .completed_raw_tick_windows
        .sort_by_key(|window| window.start);
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::RawTicks);
    manifest.warnings = write.warnings.clone();
    manifest.errors.clear();
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    if let Some(display_name) = write
        .display_name
        .clone()
        .filter(|name| !name.trim().is_empty())
    {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags.clone() {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes.clone();
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect();
    manifest.available_bar_shapes.sort_by_key(|bar_type| bar_type.value());
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.chart_mode)
        .collect();
    manifest.available_chart_modes.dedup();
    manifest.badges = manifest.derived_badges();
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);
    let data_paths = files
        .iter()
        .map(|file| resolve_cache_data_path(&dataset_dir, &file.relative_path))
        .collect::<Result<Vec<_>>>()?;
    Ok(ReplayCacheChunkedWriteOutcome {
        dataset_dir,
        manifest_path,
        data_paths,
        row_count: files.iter().map(|file| file.row_count).sum(),
    })
}

fn raw_tick_chunk_dataset_dir(write: &ReplayCacheRawTickChunkPlanWrite) -> Result<PathBuf> {
    replay_cache_write_dataset_dir(
        &write.cache_root,
        write.identity.provider,
        write.identity.env,
        &write.identity.instrument.symbol,
        &write.identity.contract,
        write.identity.request.start.date_naive(),
        write.target.as_ref(),
    )
}

fn raw_tick_chunk_relative_path(window: DownloadWindow) -> PathBuf {
    PathBuf::from("raw-ticks").join("chunks").join(format!(
        "{}_to_{}_ticks.parquet",
        window.start.format("%Y%m%dT%H%M%S%.3fZ"),
        window.end.format("%Y%m%dT%H%M%S%.3fZ")
    ))
}

fn read_raw_tick_checkpoint(path: &Path) -> Result<ReplayCacheRawTickCheckpoint> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn write_raw_tick_checkpoint(path: &Path, checkpoint: &ReplayCacheRawTickCheckpoint) -> Result<()> {
    write_bytes_atomically(
        path,
        &serde_json::to_vec_pretty(checkpoint).context("serialize raw-tick checkpoint")?,
    )
}

fn validate_raw_tick_checkpoint_identity(
    write: &ReplayCacheRawTickChunkPlanWrite,
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    if checkpoint.checkpoint_version != RAW_TICK_CHECKPOINT_VERSION {
        bail!(
            "unsupported raw-tick checkpoint version {}",
            checkpoint.checkpoint_version
        );
    }
    if checkpoint.identity_key != write.identity.stable_key()
        || checkpoint.identity != write.identity
    {
        bail!("raw-tick checkpoint request identity mismatch; use a new dataset or remove the incompatible checkpoint");
    }
    Ok(())
}

fn validate_raw_tick_checkpoint(
    write: &ReplayCacheRawTickChunkPlanWrite,
    dataset_dir: &Path,
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    validate_raw_tick_checkpoint_identity(write, checkpoint)?;
    validate_checkpoint_leaf_coverage(checkpoint)?;
    for chunk in &checkpoint.chunks {
        let ReplayCacheRawTickChunkStatus::Completed {
            file: Some(file),
            telemetry,
        } = &chunk.status
        else {
            continue;
        };
        if file.request_start != Some(chunk.window.start)
            || file.request_end != Some(chunk.window.end)
            || telemetry.request != chunk.window
            || !telemetry.completed_by_eoh()
        {
            bail!("raw-tick checkpoint completed-chunk evidence mismatch");
        }
        let path = resolve_cache_data_path(dataset_dir, &file.relative_path)?;
        validate_cache_file_hash(&path, file)?;
    }
    Ok(())
}

fn validate_cache_file_hash(path: &Path, file: &ReplayCacheDataFile) -> Result<()> {
    let expected = file
        .data_hash
        .as_ref()
        .context("checkpoint data file has no hash")?;
    if expected.algorithm != "fnv1a64" {
        bail!("unsupported checkpoint hash algorithm {}", expected.algorithm);
    }
    let actual = fnv1a64_file_hex(path)?;
    if actual != expected.value {
        bail!(
            "checkpoint data hash mismatch for {}: expected {} actual {}",
            path.display(),
            expected.value,
            actual
        );
    }
    Ok(())
}

fn validate_checkpoint_leaf_coverage(checkpoint: &ReplayCacheRawTickCheckpoint) -> Result<()> {
    let mut leaves = checkpoint
        .chunks
        .iter()
        .filter(|chunk| !matches!(chunk.status, ReplayCacheRawTickChunkStatus::Split { .. }))
        .map(|chunk| chunk.window)
        .collect::<Vec<_>>();
    leaves.sort_by_key(|window| window.start);
    validate_exact_window_coverage(checkpoint.identity.request, &leaves)
}

fn validate_exact_window_coverage(parent: DownloadWindow, windows: &[DownloadWindow]) -> Result<()> {
    let first = windows.first().context("raw-tick chunk plan has no windows")?;
    if first.start != parent.start {
        bail!("raw-tick chunk plan does not begin at requested start");
    }
    let mut expected_start = parent.start;
    for window in windows {
        if window.start != expected_start || window.start >= window.end || window.end > parent.end {
            bail!("raw-tick chunk plan has a gap, overlap, or out-of-range window");
        }
        expected_start = window.end;
    }
    if expected_start != parent.end {
        bail!("raw-tick chunk plan does not end at requested end");
    }
    Ok(())
}

fn validate_raw_tick_chunk_file_sequence(
    dataset_dir: &Path,
    files: &[ReplayCacheDataFile],
) -> Result<()> {
    let mut previous_ts = None;
    let mut previous_id = None;
    for file in files {
        let request = DownloadWindow::new(
            file.request_start.context("raw-tick chunk file has no request start")?,
            file.request_end.context("raw-tick chunk file has no request end")?,
        )?;
        let path = resolve_cache_data_path(dataset_dir, &file.relative_path)?;
        validate_cache_file_hash(&path, file)?;
        let mut rows = 0_u64;
        stream_raw_ticks_parquet_file(&path, None, |row| {
            if row.timestamp < request.start || row.timestamp >= request.end {
                bail!("raw-tick chunk row is outside its manifest request window");
            }
            if previous_ts.is_some_and(|timestamp| row.ts_ns < timestamp) {
                bail!("raw-tick chunk files are not globally timestamp ordered");
            }
            let tick_id = row.tick_id.context("schema-v2 chunk row has no tick id")?;
            if previous_id.is_some_and(|id| tick_id <= id) {
                bail!("raw-tick chunk files are not globally tick-id ordered");
            }
            previous_ts = Some(row.ts_ns);
            previous_id = Some(tick_id);
            rows = rows.saturating_add(1);
            Ok(())
        })?;
        if rows != file.row_count {
            bail!(
                "raw-tick chunk row-count mismatch for {}: manifest={} actual={rows}",
                path.display(),
                file.row_count
            );
        }
    }
    Ok(())
}

pub fn replay_cache_dataset_dir(
    root: &Path,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument: &str,
    contract: &str,
    start_date: NaiveDate,
) -> PathBuf {
    root.join(provider.label().to_ascii_lowercase())
        .join(match env {
            TradingEnvironment::Sim => "sim",
            TradingEnvironment::Live => "live",
        })
        .join(safe_cache_segment(instrument))
        .join(safe_cache_segment(contract))
        .join(start_date.to_string())
}

fn replay_cache_write_dataset_dir(
    root: &Path,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument: &str,
    contract: &ReplayCacheContract,
    start_date: NaiveDate,
    target: Option<&ReplayDownloadCacheTarget>,
) -> Result<PathBuf> {
    let Some(target) = target else {
        fs::create_dir_all(root)
            .with_context(|| format!("create replay cache root {}", root.display()))?;
        let canonical_root = fs::canonicalize(root)
            .with_context(|| format!("resolve replay cache root {}", root.display()))?;
        let dataset_dir = replay_cache_dataset_dir(
            &canonical_root,
            provider,
            env,
            instrument,
            &contract.symbol,
            start_date,
        );
        create_cache_dataset_without_symlinks(&canonical_root, &dataset_dir)?;
        let canonical_dataset = fs::canonicalize(&dataset_dir)
            .with_context(|| format!("resolve replay cache dataset {}", dataset_dir.display()))?;
        if !canonical_dataset.starts_with(&canonical_root) {
            bail!(
                "replay cache dataset {} escapes cache root {}",
                canonical_dataset.display(),
                canonical_root.display()
            );
        }
        return Ok(canonical_dataset);
    };

    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("resolve replay cache root {}", root.display()))?;
    let canonical_dataset = fs::canonicalize(&target.dataset_dir).with_context(|| {
        format!(
            "resolve replay cache target directory {}",
            target.dataset_dir.display()
        )
    })?;
    if !canonical_dataset.starts_with(&canonical_root) {
        bail!(
            "replay cache target {} is outside cache root {}",
            canonical_dataset.display(),
            canonical_root.display()
        );
    }

    let canonical_manifest = fs::canonicalize(&target.manifest_path).with_context(|| {
        format!(
            "resolve replay cache target manifest {}",
            target.manifest_path.display()
        )
    })?;
    let expected_manifest = canonical_dataset.join(MANIFEST_FILE_NAME);
    if canonical_manifest != expected_manifest {
        bail!(
            "replay cache target manifest {} does not identify dataset {}",
            canonical_manifest.display(),
            canonical_dataset.display()
        );
    }

    let manifest = ReplayCacheManifest::from_path(&canonical_manifest)
        .with_context(|| format!("load replay cache target {}", canonical_manifest.display()))?;
    if manifest.provider != provider
        || manifest.env != env
        || !manifest.instrument.symbol.eq_ignore_ascii_case(instrument)
        || !manifest
            .contract
            .symbol
            .eq_ignore_ascii_case(&contract.symbol)
        || (manifest.contract.id.is_some()
            && contract.id.is_some()
            && manifest.contract.id != contract.id)
    {
        bail!(
            "replay cache target identity does not match the requested provider, environment, instrument, and contract"
        );
    }

    Ok(canonical_dataset)
}

fn create_cache_dataset_without_symlinks(root: &Path, dataset_dir: &Path) -> Result<()> {
    let relative = dataset_dir.strip_prefix(root).with_context(|| {
        format!(
            "replay cache dataset {} is not under root {}",
            dataset_dir.display(),
            root.display()
        )
    })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(segment) = component else {
            bail!("replay cache dataset contains an unsafe path component");
        };
        current.push(segment);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!(
                    "replay cache dataset path contains symlink {}",
                    current.display()
                );
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => bail!(
                "replay cache dataset component is not a directory: {}",
                current.display()
            ),
            Err(err) if err.kind() == ErrorKind::NotFound => match fs::create_dir(&current) {
                Ok(()) => {}
                Err(create_err) if create_err.kind() == ErrorKind::AlreadyExists => {
                    let metadata = fs::symlink_metadata(&current).with_context(|| {
                        format!("inspect replay cache dataset {}", current.display())
                    })?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        bail!(
                            "replay cache dataset component is unsafe: {}",
                            current.display()
                        );
                    }
                }
                Err(create_err) => {
                    return Err(create_err).with_context(|| {
                        format!("create replay cache dataset {}", current.display())
                    });
                }
            },
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("inspect replay cache dataset {}", current.display())
                });
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn server_bars_relative_path(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> PathBuf {
    PathBuf::from("server-bars").join(format!(
        "{}_to_{}_{}.jsonl",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type_file_label(bar_type)
    ))
}

pub fn server_bars_parquet_relative_path(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> PathBuf {
    PathBuf::from("server-bars").join(format!(
        "{}_to_{}_{}.parquet",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type_file_label(bar_type)
    ))
}

pub fn raw_ticks_relative_path(start: DateTime<Utc>, end: DateTime<Utc>) -> PathBuf {
    PathBuf::from("raw-ticks").join(format!(
        "{}_to_{}_ticks.parquet",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    ))
}

fn versioned_cache_relative_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("cache");
    let extension = base.extension().and_then(|value| value.to_str());
    let nonce = Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let sequence = CACHE_FILE_VERSION.fetch_add(1, Ordering::Relaxed);
    let file_name = match extension {
        Some(extension) => format!(
            "{stem}_v{nonce}-{}-{sequence}.{extension}",
            std::process::id()
        ),
        None => format!("{stem}_v{nonce}-{}-{sequence}", std::process::id()),
    };
    parent.join(file_name)
}

fn remove_superseded_cache_files(dataset_dir: &Path, relative_paths: &[PathBuf]) {
    for relative_path in relative_paths {
        let Ok(data_path) = resolve_cache_data_path(dataset_dir, relative_path) else {
            continue;
        };
        let _ = fs::remove_file(data_path);
    }
}

pub fn normalize_server_bar_rows(bars: Vec<Bar>) -> Vec<ReplayCacheServerBarRow> {
    let mut rows: Vec<_> = bars
        .into_iter()
        .filter(|bar| {
            bar.ts_ns > 0
                && bar.open.is_finite()
                && bar.high.is_finite()
                && bar.low.is_finite()
                && bar.close.is_finite()
                && bar.volume.is_none_or(f64::is_finite)
        })
        .map(|bar| ReplayCacheServerBarRow::from_bar(&bar))
        .collect();
    rows.sort_by_key(|row| row.ts_ns);
    rows.dedup_by_key(|row| row.ts_ns);
    rows
}

pub fn normalize_raw_tick_rows(
    ticks: Vec<ReplayCacheRawTickRow>,
) -> ReplayCacheRawTicksNormalizeOutcome {
    let mut dropped_rows = 0usize;
    let mut rows = Vec::with_capacity(ticks.len());
    for tick in ticks {
        if validate_raw_tick_row(&tick).is_ok() {
            rows.push(tick);
        } else {
            dropped_rows = dropped_rows.saturating_add(1);
        }
    }

    rows.sort_by(|left, right| {
        left.ts_ns.cmp(&right.ts_ns).then_with(|| {
            left.tick_id
                .unwrap_or(i64::MAX)
                .cmp(&right.tick_id.unwrap_or(i64::MAX))
        })
    });

    let mut seen_tick_ids = BTreeSet::new();
    let before_dedup = rows.len();
    rows.retain(|row| {
        row.tick_id
            .is_none_or(|tick_id| seen_tick_ids.insert(tick_id))
    });
    let duplicate_tick_ids = before_dedup.saturating_sub(rows.len());

    ReplayCacheRawTicksNormalizeOutcome {
        rows,
        duplicate_tick_ids,
        dropped_rows,
    }
}

fn validate_raw_tick_id_write_invariant(rows: &[ReplayCacheRawTickRow]) -> Result<()> {
    let mut previous = None;
    for (index, row) in rows.iter().enumerate() {
        let tick_id = row.tick_id.with_context(|| {
            format!(
                "raw tick row {index} has no tick id; schema v2 cache was not written because replay cannot prove global uniqueness"
            )
        })?;
        if previous.is_some_and(|last| tick_id <= last) {
            bail!(
                "raw tick ids must be strictly increasing in timestamp order; row {index} has id {tick_id} after {:?}",
                previous
            );
        }
        previous = Some(tick_id);
    }
    Ok(())
}

pub fn read_server_bars_jsonl_file(path: &Path) -> Result<Vec<Bar>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut bars = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.with_context(|| format!("read line {line_number} in {}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: ReplayCacheServerBarRow = serde_json::from_str(trimmed)
            .with_context(|| format!("parse JSONL bar line {line_number} in {}", path.display()))?;
        bars.push(
            server_bar_row_to_bar(row)
                .with_context(|| format!("validate JSONL bar line {line_number}"))?,
        );
    }

    if bars.is_empty() {
        bail!("server-bar cache file {} contained no bars", path.display());
    }
    normalize_read_server_bars(bars)
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    write_file_atomically(path, |file| {
        file.write_all(bytes)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    })
}

fn write_file_atomically(path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache-file");
    let nonce = Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let mut write = Some(write);

    for attempt in 0..16_u8 {
        let temp_path = parent.join(format!(
            ".{file_name}.tmp-{}-{nonce}-{attempt}",
            std::process::id()
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("create {}", temp_path.display()));
            }
        };

        let result = (|| {
            write.take().expect("atomic writer runs once")(&mut file)?;
            file.sync_all()
                .with_context(|| format!("sync {}", temp_path.display()))?;
            drop(file);
            fs::rename(&temp_path, path).with_context(|| {
                format!("replace {} with {}", path.display(), temp_path.display())
            })?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        return result;
    }

    bail!(
        "could not allocate temporary file beside {}",
        path.display()
    )
}

fn server_bars_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, true),
    ]))
}

pub fn write_server_bars_parquet_file(path: &Path, rows: &[ReplayCacheServerBarRow]) -> Result<()> {
    write_server_bars_parquet_file_with_limits(
        path,
        rows,
        PARQUET_ROW_GROUP_ROWS,
        PARQUET_WRITE_BATCH_ROWS,
    )
}

fn write_server_bars_parquet_file_with_limits(
    path: &Path,
    rows: &[ReplayCacheServerBarRow],
    row_group_rows: usize,
    batch_rows: usize,
) -> Result<()> {
    if rows.is_empty() {
        bail!("server-bar parquet writer requires at least one row");
    }
    if row_group_rows == 0 || batch_rows == 0 {
        bail!("parquet row-group and write-batch limits must be positive");
    }
    let schema = server_bars_parquet_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_size(row_group_rows)
        .set_write_batch_size(batch_rows)
        .build();
    write_file_atomically(path, |file| {
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        for chunk in rows.chunks(batch_rows) {
            writer.write(&server_bars_record_batch(schema.clone(), chunk)?)?;
        }
        writer.close()?;
        Ok(())
    })
}

fn server_bars_record_batch(
    schema: Arc<Schema>,
    rows: &[ReplayCacheServerBarRow],
) -> Result<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.timestamp.to_rfc3339())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.ts_ns).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.open).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.high).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.low).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.close).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.volume).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build server-bar parquet record batch")
}

pub fn read_server_bars_parquet_file(path: &Path) -> Result<Vec<Bar>> {
    let (bars, _) = read_server_bars_parquet_file_range(path, None)?;
    if bars.is_empty() {
        bail!(
            "server-bar parquet file {} contained no bars",
            path.display()
        );
    }
    Ok(bars)
}

pub fn read_server_bars_parquet_file_range(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<Bar>, ReplayCacheParquetReadStats)> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?;
    let (selected_row_groups, mut stats) = parquet_row_groups_for_range(&builder, range)?;
    if selected_row_groups.is_empty() {
        return Ok((Vec::new(), stats));
    }
    let reader = builder
        .with_row_groups(selected_row_groups)
        .with_batch_size(PARQUET_READ_BATCH_ROWS)
        .build()
        .with_context(|| format!("build parquet reader {}", path.display()))?;
    let mut bars = Vec::new();
    let mut last_ts_ns = None;
    for batch in reader {
        let batch = batch.with_context(|| format!("read parquet batch {}", path.display()))?;
        stats.record_batches = stats.record_batches.saturating_add(1);
        stats.max_record_batch_rows = stats.max_record_batch_rows.max(batch.num_rows());
        stats.decoded_rows = stats.decoded_rows.saturating_add(batch.num_rows() as u64);
        let timestamps = parquet_column::<StringArray>(&batch, 0, "timestamp")?;
        let ts_ns = parquet_column::<Int64Array>(&batch, 1, "ts_ns")?;
        let opens = parquet_column::<Float64Array>(&batch, 2, "open")?;
        let highs = parquet_column::<Float64Array>(&batch, 3, "high")?;
        let lows = parquet_column::<Float64Array>(&batch, 4, "low")?;
        let closes = parquet_column::<Float64Array>(&batch, 5, "close")?;
        let volumes = parquet_column::<Float64Array>(&batch, 6, "volume")?;
        for index in 0..batch.num_rows() {
            let timestamp = DateTime::parse_from_rfc3339(timestamps.value(index))
                .with_context(|| format!("parse server bar timestamp row {index}"))?
                .with_timezone(&Utc);
            let bar = server_bar_row_to_bar(ReplayCacheServerBarRow {
                timestamp,
                ts_ns: ts_ns.value(index),
                open: opens.value(index),
                high: highs.value(index),
                low: lows.value(index),
                close: closes.value(index),
                volume: optional_f64(volumes, index),
            })?;
            if last_ts_ns.is_some_and(|last| bar.ts_ns <= last) {
                bail!(
                    "server-bar parquet file {} is not strictly ordered at timestamp {}",
                    path.display(),
                    bar.ts_ns
                );
            }
            last_ts_ns = Some(bar.ts_ns);
            if timestamp_range_contains(range, bar.ts_ns)? {
                record_emitted_timestamp(&mut stats, bar.ts_ns);
                bars.push(bar);
            }
        }
    }
    Ok((bars, stats))
}

pub fn write_raw_ticks_parquet_file(path: &Path, rows: &[ReplayCacheRawTickRow]) -> Result<()> {
    write_raw_ticks_parquet_file_with_limits(
        path,
        rows,
        PARQUET_ROW_GROUP_ROWS,
        PARQUET_WRITE_BATCH_ROWS,
    )
}

pub(crate) fn write_raw_ticks_parquet_file_with_limits(
    path: &Path,
    rows: &[ReplayCacheRawTickRow],
    row_group_rows: usize,
    batch_rows: usize,
) -> Result<()> {
    if rows.is_empty() {
        bail!("raw tick parquet writer requires at least one row");
    }
    if row_group_rows == 0 || batch_rows == 0 {
        bail!("parquet row-group and write-batch limits must be positive");
    }

    let schema = raw_ticks_parquet_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_size(row_group_rows)
        .set_write_batch_size(batch_rows)
        .build();
    write_file_atomically(path, |file| {
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        for chunk in rows.chunks(batch_rows) {
            writer.write(&raw_ticks_record_batch(schema.clone(), chunk)?)?;
        }
        writer.close()?;
        Ok(())
    })
}

fn raw_ticks_record_batch(
    schema: Arc<Schema>,
    rows: &[ReplayCacheRawTickRow],
) -> Result<RecordBatch> {
    let timestamp_values = rows
        .iter()
        .map(|row| row.timestamp.to_rfc3339())
        .collect::<Vec<_>>();
    let packet_sources = rows
        .iter()
        .map(|row| row.packet_source.as_deref())
        .collect::<Vec<_>>();
    let tick_ids: ArrayRef = if schema.field(2).is_nullable() {
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
        ))
    } else {
        let values = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.tick_id.with_context(|| {
                    format!("raw tick row {index} has no tick id; cache schema requires one")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Arc::new(Int64Array::from(values))
    };
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(timestamp_values)),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.ts_ns).collect::<Vec<_>>(),
        )),
        tick_ids,
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.size).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bid_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bid_size).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.ask_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.ask_size).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.chart_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.trade_date).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(packet_sources)),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.packet_base_ts_ms)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.packet_base_price_ticks)
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build raw-tick parquet record batch")
}

#[allow(dead_code)]
pub fn read_raw_ticks_parquet_file(path: &Path) -> Result<Vec<ReplayCacheRawTickRow>> {
    let mut rows = Vec::new();
    let stats = stream_raw_ticks_parquet_file(path, None, |row| {
        rows.push(row);
        Ok(())
    })?;
    if stats.emitted_rows == 0 {
        bail!(
            "raw tick parquet file {} contained no usable ticks",
            path.display()
        );
    }
    Ok(rows)
}

pub fn read_raw_ticks_parquet_file_range(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<ReplayCacheRawTickRow>, ReplayCacheParquetReadStats)> {
    let mut rows = Vec::new();
    let stats = stream_raw_ticks_parquet_file(path, range, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok((rows, stats))
}

pub fn stream_raw_ticks_parquet_file<F>(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
    on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let file = Arc::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        return stream_raw_ticks_parquet_reader(
            PositionIndependentFile::new(file)?,
            path,
            range,
            None,
            on_tick,
        );
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (path, range, on_tick);
        bail!(
            "position-independent replay cache reads are unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

fn stream_raw_ticks_parquet_reader<T, F>(
    source: T,
    display_path: &Path,
    range: Option<&ReplayCacheTimeRange>,
    declared_schema_version: Option<u32>,
    mut on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    T: ChunkReader + 'static,
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    let builder = ParquetRecordBatchReaderBuilder::try_new(source)
        .with_context(|| format!("open parquet reader {}", display_path.display()))?;
    let read_policy = raw_tick_read_policy(&builder, display_path, declared_schema_version)?;
    if read_policy == RawTickReadPolicy::StrictV2ProviderIds {
        validate_raw_tick_id_row_groups(&builder, display_path)?;
    }
    let (selected_row_groups, mut stats) = if read_policy == RawTickReadPolicy::LegacyV1SequenceIds
    {
        let total_row_groups = builder.metadata().num_row_groups();
        (
            (0..total_row_groups).collect::<Vec<_>>(),
            ReplayCacheParquetReadStats {
                total_row_groups,
                selected_row_groups: total_row_groups,
                ..ReplayCacheParquetReadStats::default()
            },
        )
    } else {
        parquet_row_groups_for_range(&builder, range)?
    };
    if selected_row_groups.is_empty() {
        return Ok(stats);
    }
    let reader = builder
        .with_row_groups(selected_row_groups)
        .with_batch_size(PARQUET_READ_BATCH_ROWS)
        .build()
        .with_context(|| format!("build parquet reader {}", display_path.display()))?;
    let mut last_ts_ns = None;
    let mut last_tick_id = None;
    let mut legacy_sequence_id = 0_i64;

    for batch in reader {
        let batch =
            batch.with_context(|| format!("read parquet batch {}", display_path.display()))?;
        stats.record_batches = stats.record_batches.saturating_add(1);
        stats.max_record_batch_rows = stats.max_record_batch_rows.max(batch.num_rows());
        stats.decoded_rows = stats.decoded_rows.saturating_add(batch.num_rows() as u64);
        let timestamps = parquet_column::<StringArray>(&batch, 0, "timestamp")?;
        let ts_ns = parquet_column::<Int64Array>(&batch, 1, "ts_ns")?;
        let tick_ids = parquet_column::<Int64Array>(&batch, 2, "tick_id")?;
        let prices = parquet_column::<Float64Array>(&batch, 3, "price")?;
        let sizes = parquet_column::<Float64Array>(&batch, 4, "size")?;
        let bid_prices = parquet_column::<Float64Array>(&batch, 5, "bid_price")?;
        let bid_sizes = parquet_column::<Float64Array>(&batch, 6, "bid_size")?;
        let ask_prices = parquet_column::<Float64Array>(&batch, 7, "ask_price")?;
        let ask_sizes = parquet_column::<Float64Array>(&batch, 8, "ask_size")?;
        let chart_ids = parquet_column::<Int64Array>(&batch, 9, "chart_id")?;
        let trade_dates = parquet_column::<Int32Array>(&batch, 10, "trade_date")?;
        let packet_sources = parquet_column::<StringArray>(&batch, 11, "packet_source")?;
        let packet_base_ts_ms = parquet_column::<Int64Array>(&batch, 12, "packet_base_ts_ms")?;
        let packet_base_price_ticks =
            parquet_column::<Int64Array>(&batch, 13, "packet_base_price_ticks")?;

        for row_index in 0..batch.num_rows() {
            let timestamp = DateTime::parse_from_rfc3339(timestamps.value(row_index))
                .with_context(|| format!("parse raw tick timestamp row {row_index}"))?
                .with_timezone(&Utc);
            let mut row = ReplayCacheRawTickRow {
                timestamp,
                ts_ns: ts_ns.value(row_index),
                tick_id: optional_i64(tick_ids, row_index),
                price: prices.value(row_index),
                size: sizes.value(row_index),
                bid_price: optional_f64(bid_prices, row_index),
                bid_size: optional_f64(bid_sizes, row_index),
                ask_price: optional_f64(ask_prices, row_index),
                ask_size: optional_f64(ask_sizes, row_index),
                chart_id: optional_i64(chart_ids, row_index),
                trade_date: optional_i32(trade_dates, row_index),
                packet_source: optional_string(packet_sources, row_index),
                packet_base_ts_ms: optional_i64(packet_base_ts_ms, row_index),
                packet_base_price_ticks: optional_i64(packet_base_price_ticks, row_index),
            };
            validate_raw_tick_row(&row).with_context(|| {
                format!(
                    "validate raw tick row {row_index} in {}",
                    display_path.display()
                )
            })?;
            if last_ts_ns.is_some_and(|last| row.ts_ns < last) {
                bail!(
                    "raw tick parquet file {} is not ordered by timestamp at row {}",
                    display_path.display(),
                    row_index
                );
            }
            last_ts_ns = Some(row.ts_ns);
            match read_policy {
                RawTickReadPolicy::LegacyV1SequenceIds => {
                    legacy_sequence_id = legacy_sequence_id.checked_add(1).with_context(|| {
                        format!(
                            "raw tick schema v1 file {} exceeds deterministic sequence-id capacity",
                            display_path.display()
                        )
                    })?;
                    row.tick_id = Some(legacy_sequence_id);
                }
                RawTickReadPolicy::StrictV2ProviderIds => {
                    let tick_id = row.tick_id.with_context(|| {
                        format!(
                            "raw tick schema v2 file {} has no tick id at row {}",
                            display_path.display(),
                            row_index
                        )
                    })?;
                    if last_tick_id == Some(tick_id) {
                        bail!(
                            "raw tick parquet file {} contains duplicate tick id {} at timestamp {}",
                            display_path.display(),
                            tick_id,
                            row.ts_ns
                        );
                    }
                    if last_tick_id.is_some_and(|last| tick_id < last) {
                        bail!(
                            "raw tick parquet file {} cannot prove duplicate-id safety because tick ids are not strictly increasing: {} follows {:?}",
                            display_path.display(),
                            tick_id,
                            last_tick_id
                        );
                    }
                    last_tick_id = Some(tick_id);
                }
            }
            if timestamp_range_contains(range, row.ts_ns)? {
                record_emitted_timestamp(&mut stats, row.ts_ns);
                on_tick(row)?;
            }
        }
    }
    Ok(stats)
}

fn raw_tick_read_policy<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    path: &Path,
    declared_schema_version: Option<u32>,
) -> Result<RawTickReadPolicy> {
    let legacy_schema = parquet_column_matches_expected_i64(builder, 2, "tick_id", true);
    let strict_schema = parquet_column_matches_expected_i64(builder, 2, "tick_id", false);
    match declared_schema_version {
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION) if legacy_schema => {
            Ok(RawTickReadPolicy::LegacyV1SequenceIds)
        }
        Some(RAW_TICKS_SCHEMA_VERSION) if strict_schema => {
            Ok(RawTickReadPolicy::StrictV2ProviderIds)
        }
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION) => bail!(
            "raw tick schema v1 file {} must contain a nullable INT64 tick_id column",
            path.display()
        ),
        Some(RAW_TICKS_SCHEMA_VERSION) => bail!(
            "raw tick schema v2 file {} must contain a required INT64 tick_id column",
            path.display()
        ),
        Some(version) => bail!(
            "raw tick parquet file {} declares unsupported schema version {}",
            path.display(),
            version
        ),
        None if strict_schema => Ok(RawTickReadPolicy::StrictV2ProviderIds),
        None if legacy_schema => bail!(
            "raw tick schema v1 file {} requires a manifest-declared schema version and verified data hash; re-download the dataset if its manifest is unavailable",
            path.display()
        ),
        None => bail!(
            "raw tick parquet file {} has an unsupported tick_id schema",
            path.display()
        ),
    }
}

fn parquet_row_groups_for_range<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<usize>, ReplayCacheParquetReadStats)> {
    let total_row_groups = builder.metadata().num_row_groups();
    let mut stats = ReplayCacheParquetReadStats {
        total_row_groups,
        ..ReplayCacheParquetReadStats::default()
    };
    let Some(range) = range.copied() else {
        let selected = (0..total_row_groups).collect::<Vec<_>>();
        stats.selected_row_groups = selected.len();
        return Ok((selected, stats));
    };
    let (start_ns, end_ns) = range.bounds_ns()?;
    let trusted_ts_ns_schema = parquet_column_matches_expected_i64(builder, 1, "ts_ns", false);
    let mut selected = Vec::new();
    for (index, row_group) in builder.metadata().row_groups().iter().enumerate() {
        let overlaps = trusted_ts_ns_schema
            .then(|| row_group.columns().get(1))
            .flatten()
            .and_then(|column| exact_non_null_i64_bounds(column.statistics()))
            .map(|(min, max)| max >= start_ns && min < end_ns)
            .unwrap_or(true);
        if overlaps {
            selected.push(index);
        }
    }
    stats.selected_row_groups = selected.len();
    stats.pruned_row_groups = total_row_groups.saturating_sub(selected.len());
    Ok((selected, stats))
}

fn parquet_column_matches_expected_i64<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    index: usize,
    expected_name: &str,
    nullable: bool,
) -> bool {
    let Some(arrow_field) = builder.schema().fields().get(index) else {
        return false;
    };
    let Some(parquet_column) = builder.parquet_schema().columns().get(index) else {
        return false;
    };
    arrow_field.name() == expected_name
        && arrow_field.data_type() == &DataType::Int64
        && arrow_field.is_nullable() == nullable
        && parquet_column.name() == expected_name
        && parquet_column.path().parts() == [expected_name]
        && parquet_column.physical_type() == PhysicalType::INT64
        && parquet_column.logical_type().is_none()
        && parquet_column.converted_type() == ConvertedType::NONE
        && parquet_column.max_rep_level() == 0
        && parquet_column.max_def_level() == i16::from(nullable)
}

fn exact_non_null_i64_bounds(statistics: Option<&Statistics>) -> Option<(i64, i64)> {
    let Statistics::Int64(values) = statistics? else {
        return None;
    };
    if !values.min_is_exact() || !values.max_is_exact() || values.null_count_opt() != Some(0) {
        return None;
    }
    let (min, max) = (*values.min_opt()?, *values.max_opt()?);
    (min <= max).then_some((min, max))
}

/// Raw tick cache schema v2 relies on provider tick IDs being globally unique and
/// strictly increasing in timestamp order. Exact, non-null row-group statistics
/// prove that ID ranges are disjoint before any range pruning occurs. Within each
/// decoded group the stream enforces the same monotonic invariant in constant space.
fn validate_raw_tick_id_row_groups<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    path: &Path,
) -> Result<()> {
    if !parquet_column_matches_expected_i64(builder, 2, "tick_id", false) {
        bail!(
            "raw tick parquet file {} cannot prove duplicate-id safety: expected required INT64 tick_id schema column",
            path.display()
        );
    }

    let mut previous_max = None;
    for (index, row_group) in builder.metadata().row_groups().iter().enumerate() {
        let (min, max) = row_group
            .columns()
            .get(2)
            .and_then(|column| exact_non_null_i64_bounds(column.statistics()))
            .with_context(|| {
                format!(
                    "raw tick parquet file {} cannot prove duplicate-id safety: row group {} lacks exact non-null tick_id statistics",
                    path.display(),
                    index
                )
            })?;
        if previous_max.is_some_and(|previous| min <= previous) {
            bail!(
                "raw tick parquet file {} has overlapping or non-monotonic tick-id ranges at row group {}: min {} follows {:?}",
                path.display(),
                index,
                min,
                previous_max
            );
        }
        previous_max = Some(max);
    }
    Ok(())
}

fn timestamp_range_contains(range: Option<&ReplayCacheTimeRange>, ts_ns: i64) -> Result<bool> {
    range
        .copied()
        .map_or(Ok(true), |range| range.contains_ns(ts_ns))
}

fn record_emitted_timestamp(stats: &mut ReplayCacheParquetReadStats, ts_ns: i64) {
    stats.emitted_rows = stats.emitted_rows.saturating_add(1);
    stats.first_timestamp_ns.get_or_insert(ts_ns);
    stats.last_timestamp_ns = Some(ts_ns);
}

#[allow(dead_code)]
pub fn load_server_bars_jsonl_cache_file(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
) -> Result<ReplayCacheLoadedServerBars> {
    let resolved =
        dataset.resolve_server_bars_jsonl_file(bar_type, candle_mode, requested_coverage)?;
    let bars = read_server_bars_jsonl_file(&resolved.data_path)?;
    validate_loaded_server_bars_metadata(&resolved, &bars, None)?;

    Ok(ReplayCacheLoadedServerBars {
        manifest_path: resolved.manifest_path,
        dataset_dir: resolved.dataset_dir,
        data_path: resolved.data_path,
        manifest: resolved.manifest,
        file: resolved.file,
        bars,
    })
}

pub fn load_server_bars_cache_file(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
) -> Result<ReplayCacheLoadedServerBars> {
    load_server_bars_cache_file_range(dataset, bar_type, candle_mode, requested_coverage, None)
}

pub fn load_server_bars_cache_file_range(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<ReplayCacheLoadedServerBars> {
    let resolved = dataset.resolve_server_bars_file(bar_type, candle_mode, requested_coverage)?;
    let bars = match resolved.file.format {
        ReplayCacheFileFormat::Parquet => {
            read_server_bars_parquet_file_range(&resolved.data_path, timestamp_range)?.0
        }
        ReplayCacheFileFormat::Jsonl => read_server_bars_jsonl_file(&resolved.data_path)?
            .into_iter()
            .filter(|bar| timestamp_range_contains(timestamp_range, bar.ts_ns).unwrap_or(false))
            .collect(),
        format => bail!("unsupported server-bar cache format: {format:?}"),
    };
    validate_loaded_server_bars_metadata(&resolved, &bars, timestamp_range)?;
    Ok(ReplayCacheLoadedServerBars {
        manifest_path: resolved.manifest_path,
        dataset_dir: resolved.dataset_dir,
        data_path: resolved.data_path,
        manifest: resolved.manifest,
        file: resolved.file,
        bars,
    })
}

pub fn stream_resolved_raw_ticks_parquet<F>(
    resolved: &ReplayCacheResolvedRawTicksFile,
    timestamp_range: Option<&ReplayCacheTimeRange>,
    on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let source = PositionIndependentFile::new(resolved.data_file.clone())?;
        let stats = stream_raw_ticks_parquet_reader(
            source,
            &resolved.data_path,
            timestamp_range,
            resolved.file.schema_version,
            on_tick,
        )?;
        validate_streamed_raw_ticks_metadata(resolved, &stats, timestamp_range)?;
        return Ok(stats);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (resolved, timestamp_range, on_tick);
        bail!(
            "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

pub fn stream_resolved_raw_ticks<F>(
    resolved: &ReplayCacheResolvedRawTicks,
    timestamp_range: Option<&ReplayCacheTimeRange>,
    mut on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    let mut aggregate = ReplayCacheParquetReadStats::default();
    let mut previous_ts = None;
    let mut previous_id = None;
    let multi_file = resolved.files.len() > 1;
    for file in &resolved.files {
        if let Some(range) = timestamp_range
            && let (Some(start), Some(end)) = (file.file.request_start, file.file.request_end)
            && (end <= range.start || start >= range.end)
        {
            continue;
        }
        if multi_file && file.file.schema_version != Some(RAW_TICKS_SCHEMA_VERSION) {
            bail!("multi-file raw-tick replay requires schema-v2 chunk files");
        }
        let stats = stream_resolved_raw_ticks_parquet(file, timestamp_range, |row| {
            if previous_ts.is_some_and(|timestamp| row.ts_ns < timestamp) {
                bail!("multi-file raw-tick replay is not globally timestamp ordered");
            }
            let tick_id = row.tick_id.context("raw-tick replay row has no sequence id")?;
            if multi_file && previous_id.is_some_and(|id| tick_id <= id) {
                bail!("multi-file raw-tick replay is not globally tick-id ordered");
            }
            previous_ts = Some(row.ts_ns);
            previous_id = Some(tick_id);
            on_tick(row)
        })?;
        aggregate.total_row_groups = aggregate
            .total_row_groups
            .saturating_add(stats.total_row_groups);
        aggregate.selected_row_groups = aggregate
            .selected_row_groups
            .saturating_add(stats.selected_row_groups);
        aggregate.pruned_row_groups = aggregate
            .pruned_row_groups
            .saturating_add(stats.pruned_row_groups);
        aggregate.record_batches = aggregate.record_batches.saturating_add(stats.record_batches);
        aggregate.max_record_batch_rows = aggregate
            .max_record_batch_rows
            .max(stats.max_record_batch_rows);
        aggregate.decoded_rows = aggregate.decoded_rows.saturating_add(stats.decoded_rows);
        aggregate.emitted_rows = aggregate.emitted_rows.saturating_add(stats.emitted_rows);
        aggregate.first_timestamp_ns = aggregate
            .first_timestamp_ns
            .or(stats.first_timestamp_ns);
        aggregate.last_timestamp_ns = stats.last_timestamp_ns.or(aggregate.last_timestamp_ns);
    }
    Ok(aggregate)
}

fn validate_streamed_raw_ticks_metadata(
    resolved: &ReplayCacheResolvedRawTicksFile,
    stats: &ReplayCacheParquetReadStats,
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<()> {
    if let Some(range) = timestamp_range {
        if let Some(first) = stats.first_timestamp_ns
            && !range.contains_ns(first)?
        {
            bail!(
                "raw tick cache file {} returned its first timestamp outside the requested range",
                resolved.data_path.display()
            );
        }
        if let Some(last) = stats.last_timestamp_ns
            && !range.contains_ns(last)?
        {
            bail!(
                "raw tick cache file {} returned its last timestamp outside the requested range",
                resolved.data_path.display()
            );
        }
        return Ok(());
    }
    if stats.emitted_rows == 0 {
        bail!(
            "raw tick cache file {} contained no usable ticks",
            resolved.data_path.display()
        );
    }
    if stats.emitted_rows != resolved.file.row_count {
        bail!(
            "raw tick cache row count mismatch for {}: manifest={} actual={}",
            resolved.data_path.display(),
            resolved.file.row_count,
            stats.emitted_rows
        );
    }
    let first_timestamp = DateTime::<Utc>::from_timestamp_nanos(
        stats
            .first_timestamp_ns
            .context("raw tick stream did not report a first timestamp")?,
    );
    let last_timestamp = DateTime::<Utc>::from_timestamp_nanos(
        stats
            .last_timestamp_ns
            .context("raw tick stream did not report a last timestamp")?,
    );
    if first_timestamp != resolved.file.first_timestamp
        || last_timestamp != resolved.file.last_timestamp
    {
        bail!(
            "raw tick cache timestamp range mismatch for {}: manifest={}..{} actual={}..{}",
            resolved.data_path.display(),
            resolved.file.first_timestamp,
            resolved.file.last_timestamp,
            first_timestamp,
            last_timestamp
        );
    }
    Ok(())
}

fn server_bar_row_to_bar(row: ReplayCacheServerBarRow) -> Result<Bar> {
    if row.ts_ns <= 0 {
        bail!("bar timestamp nanoseconds must be positive");
    }
    let timestamp_ns = row
        .timestamp
        .timestamp_nanos_opt()
        .context("bar timestamp is outside supported nanosecond range")?;
    if timestamp_ns != row.ts_ns {
        bail!(
            "bar timestamp {} disagrees with ts_ns {}",
            row.timestamp,
            row.ts_ns
        );
    }
    let bar = Bar {
        ts_ns: row.ts_ns,
        open: row.open,
        high: row.high,
        low: row.low,
        close: row.close,
        volume: row.volume,
    };
    validate_server_bar(&bar)?;
    Ok(bar)
}

fn normalize_read_server_bars(mut bars: Vec<Bar>) -> Result<Vec<Bar>> {
    bars.sort_by_key(|bar| bar.ts_ns);
    bars.dedup_by_key(|bar| bar.ts_ns);
    for bar in &bars {
        validate_server_bar(bar)?;
    }
    Ok(bars)
}

fn validate_server_bar(bar: &Bar) -> Result<()> {
    if bar.ts_ns <= 0 {
        bail!("bar timestamp nanoseconds must be positive");
    }
    if !(bar.open.is_finite()
        && bar.high.is_finite()
        && bar.low.is_finite()
        && bar.close.is_finite()
        && bar.volume.is_none_or(f64::is_finite))
    {
        bail!("bar contains a non-finite price or volume");
    }
    if bar.high < bar.low {
        bail!("bar high is below low");
    }
    const EPSILON: f64 = 1e-9;
    if bar.open > bar.high + EPSILON
        || bar.close > bar.high + EPSILON
        || bar.open < bar.low - EPSILON
        || bar.close < bar.low - EPSILON
    {
        bail!("bar OHLC values fall outside high/low range");
    }
    Ok(())
}

fn validate_raw_tick_row(row: &ReplayCacheRawTickRow) -> Result<()> {
    if row.ts_ns <= 0 {
        bail!("raw tick timestamp nanoseconds must be positive");
    }
    let timestamp_ns = row
        .timestamp
        .timestamp_nanos_opt()
        .context("raw tick timestamp is outside supported nanosecond range")?;
    if timestamp_ns != row.ts_ns {
        bail!(
            "raw tick timestamp {} disagrees with ts_ns {}",
            row.timestamp,
            row.ts_ns
        );
    }
    if !(row.price.is_finite() && row.price > 0.0 && row.size.is_finite() && row.size > 0.0) {
        bail!("raw tick price and size must be finite positive values");
    }
    validate_optional_positive(row.bid_price, "bid price")?;
    validate_optional_nonnegative(row.bid_size, "bid size")?;
    validate_optional_positive(row.ask_price, "ask price")?;
    validate_optional_nonnegative(row.ask_size, "ask size")?;
    Ok(())
}

fn validate_optional_positive(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        bail!("raw tick {label} must be finite and positive");
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        bail!("raw tick {label} must be finite and non-negative");
    }
    Ok(())
}

fn raw_ticks_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("tick_id", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("size", DataType::Float64, false),
        Field::new("bid_price", DataType::Float64, true),
        Field::new("bid_size", DataType::Float64, true),
        Field::new("ask_price", DataType::Float64, true),
        Field::new("ask_size", DataType::Float64, true),
        Field::new("chart_id", DataType::Int64, true),
        Field::new("trade_date", DataType::Int32, true),
        Field::new("packet_source", DataType::Utf8, true),
        Field::new("packet_base_ts_ms", DataType::Int64, true),
        Field::new("packet_base_price_ticks", DataType::Int64, true),
    ]))
}

#[allow(dead_code)]
fn parquet_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    index: usize,
    name: &str,
) -> Result<&'a T> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("parquet column {name} had an unexpected type"))
}

#[allow(dead_code)]
fn optional_i64(array: &Int64Array, index: usize) -> Option<i64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_i32(array: &Int32Array, index: usize) -> Option<i32> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_f64(array: &Float64Array, index: usize) -> Option<f64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_string(array: &StringArray, index: usize) -> Option<String> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index).to_string())
    }
}

fn validate_loaded_server_bars_metadata(
    resolved: &ReplayCacheResolvedServerBarsFile,
    bars: &[Bar],
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<()> {
    if timestamp_range.is_none() && bars.is_empty() {
        bail!(
            "server-bar cache file {} contained no bars",
            resolved.data_path.display()
        );
    }
    if let Some(range) = timestamp_range {
        for bar in bars {
            if !range.contains_ns(bar.ts_ns)? {
                bail!(
                    "server-bar cache file {} returned timestamp {} outside requested range",
                    resolved.data_path.display(),
                    bar.ts_ns
                );
            }
        }
        return Ok(());
    }
    if bars.len() as u64 != resolved.file.row_count {
        bail!(
            "server-bar cache row count mismatch for {}: manifest={} actual={}",
            resolved.data_path.display(),
            resolved.file.row_count,
            bars.len()
        );
    }
    let first_timestamp = DateTime::<Utc>::from_timestamp_nanos(bars[0].ts_ns);
    let last_timestamp =
        DateTime::<Utc>::from_timestamp_nanos(bars.last().expect("bars are non-empty").ts_ns);
    if first_timestamp != resolved.file.first_timestamp
        || last_timestamp != resolved.file.last_timestamp
    {
        bail!(
            "server-bar cache timestamp range mismatch for {}: manifest={}..{} actual={}..{}",
            resolved.data_path.display(),
            resolved.file.first_timestamp,
            resolved.file.last_timestamp,
            first_timestamp,
            last_timestamp
        );
    }
    Ok(())
}

fn manifest_coverage_from_files(
    files: &[ReplayCacheDataFile],
    fallback_start: DateTime<Utc>,
) -> ReplayCacheCoverage {
    let start = files
        .iter()
        .map(|file| file.first_timestamp)
        .min()
        .unwrap_or(fallback_start);
    let end = files
        .iter()
        .map(|file| file.last_timestamp)
        .max()
        .unwrap_or(start);
    ReplayCacheCoverage {
        start,
        end,
        trading_date: Some(start.date_naive()),
    }
}

fn normalize_cache_tags(tags: Vec<String>) -> Vec<String> {
    let mut tags = tags
        .into_iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

fn replay_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> String {
    format!(
        "{} {} to {} {} server bars",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type.label()
    )
}

fn raw_ticks_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> String {
    format!(
        "{} raw ticks {} to {}",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    )
}

fn bar_type_file_label(bar_type: BarType) -> String {
    let kind = match bar_type.kind() {
        BarKind::Minute => "minute",
        BarKind::Second => "second",
        BarKind::Tick => "tick",
        BarKind::Volume => "volume",
        BarKind::Range => "range",
    };
    format!("{}{}", bar_type.value(), kind)
}

fn safe_cache_segment(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn fnv1a64_file_hex(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    fnv1a64_reader_hex(file, path)
}

fn fnv1a64_reader_hex<R: Read>(reader: R, path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(64 * 1024, reader);
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn raw_tick_manifest_entry_is_replayable(file: &ReplayCacheDataFile) -> bool {
    matches!(
        file.schema_version,
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION | RAW_TICKS_SCHEMA_VERSION)
    )
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn validate_raw_tick_manifest_hash(
    source: &PositionIndependentFile,
    file: &ReplayCacheDataFile,
    path: &Path,
) -> Result<()> {
    let expected = file
        .data_hash
        .as_ref()
        .with_context(|| {
            format!(
                "raw tick schema v{} cache {} has no writer data hash; re-download the dataset before replay",
                file.schema_version.unwrap_or_default(),
                path.display()
            )
        })?;
    if expected.algorithm != "fnv1a64" {
        bail!(
            "raw tick schema v{} cache {} uses unsupported hash algorithm {}; re-download the dataset before replay",
            file.schema_version.unwrap_or_default(),
            path.display(),
            expected.algorithm
        );
    }
    let actual = fnv1a64_reader_hex(
        source
            .get_read(0)
            .with_context(|| format!("open {} for position-independent hashing", path.display()))?,
        path,
    )?;
    if actual != expected.value {
        bail!(
            "raw tick schema v{} cache hash mismatch for {}: manifest={} actual={}; re-download the dataset before replay",
            file.schema_version.unwrap_or_default(),
            path.display(),
            expected.value,
            actual
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheDataset {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub manifest: ReplayCacheManifest,
}

impl ReplayCacheDataset {
    pub fn can_serve(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        self.manifest
            .supports_replay(bar_type, candle_mode, requested_coverage)
    }

    pub fn server_bars_jsonl_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty() {
            return None;
        }
        if requested_coverage.is_some_and(|requested| !self.manifest.coverage.contains(requested)) {
            return None;
        }
        self.manifest.files.iter().find(|file| {
            file.errors.is_empty()
                && file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.format == ReplayCacheFileFormat::Jsonl
                && file.market_shape.chart_mode != Some(CandleMode::HeikinAshi)
                && file
                    .schema_version
                    .is_none_or(|version| version == SERVER_BARS_SCHEMA_VERSION)
                && file.market_shape.supports(bar_type, candle_mode)
        })
    }

    pub fn server_bars_parquet_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty()
            || requested_coverage
                .is_some_and(|requested| !self.manifest.coverage.contains(requested))
        {
            return None;
        }
        self.manifest.files.iter().find(|file| {
            file.errors.is_empty()
                && file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.format == ReplayCacheFileFormat::Parquet
                && file.market_shape.chart_mode != Some(CandleMode::HeikinAshi)
                && file
                    .schema_version
                    .is_none_or(|version| version == SERVER_BARS_SCHEMA_VERSION)
                && file.market_shape.supports(bar_type, candle_mode)
        })
    }

    pub fn server_bars_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        self.server_bars_parquet_file_for(bar_type, candle_mode, requested_coverage)
            .or_else(|| self.server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage))
    }

    #[allow(dead_code)]
    pub fn resolve_server_bars_jsonl_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = self
            .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
            .with_context(|| {
                format!(
                    "no cached JSONL server bars for {} in {}",
                    bar_type.mode_label(candle_mode),
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }

    pub fn raw_ticks_parquet_file_for(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        self.raw_ticks_parquet_files_for(requested_coverage)
            .into_iter()
            .next()
    }

    pub fn raw_ticks_parquet_files_for(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Vec<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty() {
            return Vec::new();
        }
        let effective_coverage = self
            .manifest
            .completed_raw_tick_coverage
            .as_ref()
            .unwrap_or(&self.manifest.coverage);
        if requested_coverage.is_some_and(|requested| !effective_coverage.contains(requested)) {
            return Vec::new();
        }
        let mut files = self
            .manifest
            .files
            .iter()
            .filter(|file| {
                file.errors.is_empty()
                    && file.source_kind == ReplayCacheSourceKind::RawTicks
                    && file.format == ReplayCacheFileFormat::Parquet
                    && raw_tick_manifest_entry_is_replayable(file)
            })
            .collect::<Vec<_>>();
        files.sort_by(|left, right| {
            left.request_start
                .cmp(&right.request_start)
                .then_with(|| left.relative_path.cmp(&right.relative_path))
        });
        if self.manifest.completed_raw_tick_windows.is_empty() {
            return (files.len() == 1
                && files[0].request_start.is_none()
                && files[0].request_end.is_none())
            .then_some(files)
            .unwrap_or_default();
        }
        let Some(completed) = self.manifest.completed_raw_tick_coverage.as_ref() else {
            return Vec::new();
        };
        let Ok(parent) = DownloadWindow::new(completed.start, completed.end) else {
            return Vec::new();
        };
        if validate_exact_window_coverage(parent, &self.manifest.completed_raw_tick_windows).is_err()
        {
            return Vec::new();
        }
        let mut seen_windows = BTreeSet::new();
        if files.iter().any(|file| {
            let Some(start) = file.request_start else {
                return true;
            };
            let Some(end) = file.request_end else {
                return true;
            };
            let window = DownloadWindow { start, end };
            !self.manifest.completed_raw_tick_windows.contains(&window)
                || !seen_windows.insert((start, end))
        }) {
            return Vec::new();
        }
        files
    }

    pub fn resolve_raw_ticks_parquet_file(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedRawTicksFile> {
        let files = self.raw_ticks_parquet_files_for(requested_coverage);
        if files.len() != 1 {
            bail!(
                "raw-tick dataset {} resolves to {} Parquet files; use the multi-file resolver",
                self.manifest_path.display(),
                files.len()
            );
        }
        let file = files
            .into_iter()
            .next()
            .with_context(|| {
                format!(
                    "no cached Parquet raw ticks in {}",
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let data_file = Arc::new(
                File::open(&data_path)
                    .with_context(|| format!("open replay cache lease {}", data_path.display()))?,
            );
            let source = PositionIndependentFile::new(data_file.clone())?;
            validate_raw_tick_manifest_hash(&source, &file, &data_path)?;
            return Ok(ReplayCacheResolvedRawTicksFile {
                manifest_path: self.manifest_path.clone(),
                dataset_dir: self.dataset_dir.clone(),
                data_path,
                data_file,
                manifest: self.manifest.clone(),
                file,
            });
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = data_path;
            bail!(
                "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
            )
        }
    }

    pub fn resolve_raw_ticks_parquet_files(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedRawTicks> {
        let entries = self.raw_ticks_parquet_files_for(requested_coverage);
        if entries.is_empty() {
            bail!(
                "no complete cached Parquet raw-tick coverage in {}",
                self.manifest_path.display()
            );
        }
        let mut files = Vec::with_capacity(entries.len());
        for entry in entries {
            let file = entry.clone();
            let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                let data_file = Arc::new(File::open(&data_path).with_context(|| {
                    format!("open replay cache lease {}", data_path.display())
                })?);
                let source = PositionIndependentFile::new(data_file.clone())?;
                validate_raw_tick_manifest_hash(&source, &file, &data_path)?;
                files.push(ReplayCacheResolvedRawTicksFile {
                    manifest_path: self.manifest_path.clone(),
                    dataset_dir: self.dataset_dir.clone(),
                    data_path,
                    data_file,
                    manifest: self.manifest.clone(),
                    file,
                });
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                let _ = data_path;
                bail!(
                    "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
                );
            }
        }
        Ok(ReplayCacheResolvedRawTicks {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            manifest: self.manifest.clone(),
            files,
        })
    }

    #[allow(dead_code)]
    pub fn resolve_server_bars_parquet_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        self.resolve_server_bars_file_with_format(
            bar_type,
            candle_mode,
            requested_coverage,
            ReplayCacheFileFormat::Parquet,
        )
    }

    pub fn resolve_server_bars_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = self
            .server_bars_file_for(bar_type, candle_mode, requested_coverage)
            .with_context(|| {
                format!(
                    "no cached server bars for {} in {}",
                    bar_type.mode_label(candle_mode),
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }

    #[allow(dead_code)]
    fn resolve_server_bars_file_with_format(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
        format: ReplayCacheFileFormat,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = match format {
            ReplayCacheFileFormat::Parquet => {
                self.server_bars_parquet_file_for(bar_type, candle_mode, requested_coverage)
            }
            ReplayCacheFileFormat::Jsonl => {
                self.server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
            }
            _ => None,
        }
        .with_context(|| format!("no cached server bars in {}", self.manifest_path.display()))?
        .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayCacheLibrary {
    pub root: PathBuf,
    pub datasets: Vec<ReplayCacheDataset>,
    pub warnings: Vec<String>,
}

impl ReplayCacheLibrary {
    pub fn scan(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let mut library = Self {
            root: root.clone(),
            datasets: Vec::new(),
            warnings: Vec::new(),
        };
        scan_manifest_paths(
            &root,
            &mut |path| match ReplayCacheManifest::from_path(path) {
                Ok(manifest) => library.datasets.push(ReplayCacheDataset {
                    dataset_dir: path.parent().unwrap_or(root.as_path()).to_path_buf(),
                    manifest_path: path.to_path_buf(),
                    manifest,
                }),
                Err(err) => library
                    .warnings
                    .push(format!("{}: {err:#}", path.display())),
            },
        );
        library.datasets.sort_by(|left, right| {
            right
                .manifest
                .coverage
                .end
                .cmp(&left.manifest.coverage.end)
                .then_with(|| left.manifest.display_name.cmp(&right.manifest.display_name))
        });
        library
    }

    #[allow(dead_code)]
    pub fn first_serving(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets
            .iter()
            .find(|dataset| dataset.can_serve(bar_type, candle_mode, requested_coverage))
    }

    pub fn first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets.iter().find(|dataset| {
            dataset.can_serve(bar_type, candle_mode, requested_coverage)
                && dataset
                    .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
                    .is_some()
        })
    }

    pub fn first_server_bars(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets.iter().find(|dataset| {
            dataset.can_serve(bar_type, candle_mode, requested_coverage)
                && dataset
                    .server_bars_file_for(bar_type, candle_mode, requested_coverage)
                    .is_some()
        })
    }

    #[allow(dead_code)]
    pub fn load_first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedServerBars>> {
        let Some(dataset) = self.first_server_bars_jsonl(bar_type, candle_mode, requested_coverage)
        else {
            return Ok(None);
        };
        load_server_bars_jsonl_cache_file(dataset, bar_type, candle_mode, requested_coverage)
            .map(Some)
    }

    pub fn load_first_server_bars(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedServerBars>> {
        let Some(dataset) = self.first_server_bars(bar_type, candle_mode, requested_coverage)
        else {
            return Ok(None);
        };
        load_server_bars_cache_file(dataset, bar_type, candle_mode, requested_coverage).map(Some)
    }

    pub fn raw_ticks_parquet_datasets(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Vec<&ReplayCacheDataset> {
        self.datasets
            .iter()
            .filter(|dataset| {
                dataset
                    .raw_ticks_parquet_file_for(requested_coverage)
                    .is_some()
            })
            .collect()
    }

    pub fn load_unique_raw_ticks_parquet(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedRawTicks>> {
        self.load_unique_raw_ticks_parquet_range(requested_coverage, None)
    }

    pub fn resolve_unique_raw_ticks_parquet(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheResolvedRawTicksFile>> {
        let datasets = self.raw_ticks_parquet_datasets(requested_coverage);
        if datasets.len() > 1 {
            let names = datasets
                .iter()
                .map(|dataset| dataset.manifest.display_name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "multiple cached raw-tick datasets match replay: {names}; select one dataset before starting replay"
            );
        }
        let Some(dataset) = datasets.first().copied() else {
            return Ok(None);
        };
        dataset
            .resolve_raw_ticks_parquet_file(requested_coverage)
            .map(Some)
    }

    pub fn resolve_unique_raw_ticks_parquet_files(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheResolvedRawTicks>> {
        let datasets = self.raw_ticks_parquet_datasets(requested_coverage);
        if datasets.len() > 1 {
            let names = datasets
                .iter()
                .map(|dataset| dataset.manifest.display_name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "multiple cached raw-tick datasets match replay: {names}; select one dataset before starting replay"
            );
        }
        let Some(dataset) = datasets.first().copied() else {
            return Ok(None);
        };
        dataset
            .resolve_raw_ticks_parquet_files(requested_coverage)
            .map(Some)
    }

    pub fn load_unique_raw_ticks_parquet_range(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
        timestamp_range: Option<&ReplayCacheTimeRange>,
    ) -> Result<Option<ReplayCacheLoadedRawTicks>> {
        let Some(resolved) = self.resolve_unique_raw_ticks_parquet(requested_coverage)? else {
            return Ok(None);
        };
        let mut ticks = Vec::new();
        stream_resolved_raw_ticks_parquet(&resolved, timestamp_range, |row| {
            ticks.push(row);
            Ok(())
        })?;
        Ok(Some(ReplayCacheLoadedRawTicks {
            manifest_path: resolved.manifest_path,
            dataset_dir: resolved.dataset_dir,
            data_path: resolved.data_path,
            manifest: resolved.manifest,
            file: resolved.file,
            ticks,
        }))
    }

    pub fn load_raw_ticks_parquet_dataset(
        &self,
        dataset: &ReplayCacheDataset,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheLoadedRawTicks> {
        self.load_raw_ticks_parquet_dataset_range(dataset, requested_coverage, None)
    }

    pub fn load_raw_ticks_parquet_dataset_range(
        &self,
        dataset: &ReplayCacheDataset,
        requested_coverage: Option<&ReplayCacheCoverage>,
        timestamp_range: Option<&ReplayCacheTimeRange>,
    ) -> Result<ReplayCacheLoadedRawTicks> {
        let resolved = dataset.resolve_raw_ticks_parquet_file(requested_coverage)?;
        let mut ticks = Vec::new();
        stream_resolved_raw_ticks_parquet(&resolved, timestamp_range, |row| {
            ticks.push(row);
            Ok(())
        })?;
        Ok(ReplayCacheLoadedRawTicks {
            manifest_path: resolved.manifest_path,
            dataset_dir: resolved.dataset_dir,
            data_path: resolved.data_path,
            manifest: resolved.manifest,
            file: resolved.file,
            ticks,
        })
    }
}

fn scan_manifest_paths(root: &Path, on_manifest: &mut impl FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|value| value.to_str()) == Some(MANIFEST_FILE_NAME) {
            on_manifest(&path);
            continue;
        }
        if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            scan_manifest_paths(&path, on_manifest);
        }
    }
}

fn resolve_cache_data_path(dataset_dir: &Path, relative_path: &Path) -> Result<PathBuf> {
    if relative_path.is_absolute() {
        bail!(
            "cache data path {} must be relative to its manifest",
            relative_path.display()
        );
    }
    for component in relative_path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!(
                "cache data path {} is not a safe manifest-relative path",
                relative_path.display()
            ),
        }
    }
    let data_path = dataset_dir.join(relative_path);
    if !data_path.is_file() {
        bail!("cache data file {} was not found", data_path.display());
    }
    Ok(data_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dt(raw: &str) -> DateTime<Utc> {
        raw.parse().expect("valid timestamp")
    }

    fn temp_cache_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("trader-cache-{name}-{nonce}"))
    }

    fn sample_manifest() -> ReplayCacheManifest {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: Some("Micro E-mini S&P 500".to_string()),
                exchange: Some("CME".to_string()),
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: NaiveDate::from_ymd_opt(2026, 9, 18),
            },
            display_name: "MESU6 2026-07-23 1m HA".to_string(),
            coverage: ReplayCacheCoverage {
                start: dt("2026-07-23T13:30:00Z"),
                end: dt("2026-07-23T20:00:00Z"),
                trading_date: NaiveDate::from_ymd_opt(2026, 7, 23),
            },
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({
                "md": "getChart",
                "chartDescription": BarType::minute(1).chart_description(),
            }),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            files: vec![ReplayCacheDataFile {
                relative_path: PathBuf::from("server-bars/1m-heikin.parquet"),
                source_kind: ReplayCacheSourceKind::ServerBars,
                format: ReplayCacheFileFormat::Parquet,
                schema_version: Some(1),
                compression: Some("zstd".to_string()),
                market_shape: ReplayCacheMarketShape {
                    bar_type: Some(BarType::minute(1)),
                    chart_mode: Some(CandleMode::HeikinAshi),
                    session_template: Some("Globex".to_string()),
                },
                row_count: 390,
                first_timestamp: dt("2026-07-23T13:30:00Z"),
                last_timestamp: dt("2026-07-23T20:00:00Z"),
                data_hash: Some(ReplayCacheDataHash {
                    algorithm: "sha256".to_string(),
                    value: "abc123".to_string(),
                }),
                warnings: Vec::new(),
                errors: Vec::new(),
            }],
            app: Some(ReplayCacheAppMetadata {
                app_version: Some("0.1.0".to_string()),
                git_commit: Some("test".to_string()),
                generated_at: Some(dt("2026-07-24T00:00:00Z")),
            }),
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: vec!["regression".to_string()],
            notes: Some("sample".to_string()),
        }
    }

    fn sample_contract_metadata() -> ReplayCacheContractMetadata {
        let snapshot = |endpoint: &str, payload: Value| ReplayCacheMetadataSnapshot {
            endpoint: endpoint.to_string(),
            fetched_at: dt("2026-07-24T00:00:00Z"),
            source_timestamp: None,
            payload,
        };
        ReplayCacheContractMetadata {
            context: ReplayCacheMetadataContext {
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_id: Some(42),
                user_name: Some("tester".to_string()),
                accounts: vec![
                    ReplayCacheMetadataAccount {
                        id: 7,
                        name: "DEMO7".to_string(),
                    },
                    ReplayCacheMetadataAccount {
                        id: 9,
                        name: "DEMO9".to_string(),
                    },
                ],
                accounts_endpoint: Some("account/list".to_string()),
                accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
            },
            contract: snapshot("contract/item", json!({"id": 123})),
            maturity: Some(snapshot(
                "contractMaturity/item",
                json!({"id": 62531, "productId": 1878809}),
            )),
            maturity_chain: Some(snapshot("contractMaturity/deps", json!([{"id": 62531}]))),
            product: Some(snapshot("product/item", json!({"id": 1878809}))),
            product_sessions: Some(snapshot("productSession/deps", json!([{"id": 1}]))),
            product_margins: Some(snapshot("productMargin/deps", json!([{"id": 2}]))),
            contract_margins: Some(snapshot("contractMargin/deps", json!([{"id": 3}]))),
            fee_params: Some(snapshot(
                "contract/getproductfeeparams",
                json!({"commission": 0.39}),
            )),
            suggested_coverage: Some(ReplayCacheSuggestedCoverage {
                start_date: NaiveDate::from_ymd_opt(2026, 6, 19).expect("date"),
                end_date: NaiveDate::from_ymd_opt(2026, 9, 18).expect("date"),
                basis: "adjacent maturities".to_string(),
                estimated: true,
            }),
        }
    }

    fn bar(raw: &str, open: f64) -> Bar {
        Bar {
            ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            open,
            high: open + 1.0,
            low: open - 1.0,
            close: open + 0.5,
            volume: Some(10.0),
        }
    }

    fn bar_row_json(raw: &str, open: f64) -> String {
        json!({
            "timestamp": raw,
            "ts_ns": dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            "open": open,
            "high": open + 1.0,
            "low": open - 1.0,
            "close": open + 0.5,
            "volume": 10.0
        })
        .to_string()
    }

    fn raw_tick(raw: &str, tick_id: Option<i64>, price: f64) -> ReplayCacheRawTickRow {
        ReplayCacheRawTickRow {
            timestamp: dt(raw),
            ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            tick_id,
            price,
            size: 2.0,
            bid_price: Some(price - 0.25),
            bid_size: Some(10.0),
            ask_price: Some(price),
            ask_size: Some(12.0),
            chart_id: Some(77),
            trade_date: Some(20260723),
            packet_source: Some("db".to_string()),
            packet_base_ts_ms: Some(1_785_000_000_000),
            packet_base_price_ticks: Some(29_700),
        }
    }

    fn sequential_raw_ticks(count: usize) -> Vec<ReplayCacheRawTickRow> {
        let base = dt("2026-07-23T00:00:00Z")
            .timestamp_nanos_opt()
            .expect("timestamp ns");
        (0..count)
            .map(|index| {
                let ts_ns = base + index as i64;
                ReplayCacheRawTickRow {
                    timestamp: DateTime::<Utc>::from_timestamp_nanos(ts_ns),
                    ts_ns,
                    tick_id: Some(index as i64 + 1),
                    price: 100.0 + (index % 8) as f64 * 0.25,
                    size: 1.0,
                    bid_price: None,
                    bid_size: None,
                    ask_price: None,
                    ask_size: None,
                    chart_id: Some(77),
                    trade_date: Some(20260723),
                    packet_source: Some("db".to_string()),
                    packet_base_ts_ms: None,
                    packet_base_price_ticks: None,
                }
            })
            .collect()
    }

    fn raw_tick_cache_write(
        root: PathBuf,
        ticks: Vec<ReplayCacheRawTickRow>,
    ) -> ReplayCacheRawTicksWrite {
        ReplayCacheRawTicksWrite {
            cache_root: root,
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: Some("Micro E-mini S&P 500".to_string()),
                exchange: Some("CME".to_string()),
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: NaiveDate::from_ymd_opt(2026, 9, 18),
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            download_request: json!({"md": "getChart", "source": "raw-ticks"}),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            ticks,
            warnings: Vec::new(),
            display_name: Some("MESU6 legacy raw ticks".to_string()),
            tags: None,
            notes: None,
        }
    }

    fn write_raw_ticks_with_schema(
        path: &Path,
        rows: &[ReplayCacheRawTickRow],
        schema: Arc<Schema>,
        statistics: EnabledStatistics,
        row_group_rows: usize,
    ) {
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_statistics_enabled(statistics)
            .set_max_row_group_size(row_group_rows)
            .set_write_batch_size(1)
            .build();
        let file = File::create(path).expect("create parquet fixture");
        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), Some(props)).expect("create parquet writer");
        for row in rows {
            writer
                .write(&raw_ticks_record_batch(schema.clone(), std::slice::from_ref(row)).unwrap())
                .expect("write parquet row");
        }
        writer.close().expect("close parquet fixture");
    }

    fn write_legacy_v1_raw_tick_dataset(
        root: &Path,
        rows: &[ReplayCacheRawTickRow],
    ) -> ReplayCacheDataset {
        let dataset_dir = root.join("legacy-v1");
        let relative_path = PathBuf::from("raw-ticks/legacy-v1.parquet");
        let data_path = dataset_dir.join(&relative_path);
        fs::create_dir_all(data_path.parent().expect("raw tick parent"))
            .expect("create legacy cache directory");
        let mut fields = raw_ticks_parquet_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields[2] = Field::new("tick_id", DataType::Int64, true);
        write_raw_ticks_with_schema(
            &data_path,
            rows,
            Arc::new(Schema::new(fields)),
            EnabledStatistics::Chunk,
            2,
        );

        let mut manifest = sample_manifest();
        manifest.display_name = "MESU6 legacy v1 raw ticks".to_string();
        manifest.source_kind = ReplayCacheSourceKind::RawTicks;
        manifest.coverage = ReplayCacheCoverage {
            start: rows.first().expect("legacy rows").timestamp,
            end: rows.last().expect("legacy rows").timestamp,
            trading_date: Some(dt("2026-07-23T00:00:00Z").date_naive()),
        };
        manifest.files = vec![ReplayCacheDataFile {
            relative_path,
            source_kind: ReplayCacheSourceKind::RawTicks,
            format: ReplayCacheFileFormat::Parquet,
            schema_version: Some(RAW_TICKS_LEGACY_SCHEMA_VERSION),
            compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
            market_shape: ReplayCacheMarketShape {
                bar_type: None,
                chart_mode: None,
                session_template: Some("Globex".to_string()),
            },
            row_count: rows.len() as u64,
            first_timestamp: rows.first().expect("legacy rows").timestamp,
            last_timestamp: rows.last().expect("legacy rows").timestamp,
            data_hash: Some(ReplayCacheDataHash {
                algorithm: "fnv1a64".to_string(),
                value: fnv1a64_file_hex(&data_path).expect("hash legacy parquet"),
            }),
            warnings: vec!["Legacy schema v1 nullable tick IDs.".to_string()],
            errors: Vec::new(),
        }];
        manifest.badges = manifest.derived_badges();
        let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize legacy manifest"),
        )
        .expect("write legacy manifest");
        ReplayCacheDataset {
            manifest_path,
            dataset_dir,
            manifest,
        }
    }

    #[test]
    fn manifest_parsing_derives_library_badges_and_shapes() {
        let manifest = sample_manifest();
        let raw = serde_json::to_string(&manifest).expect("serialize manifest");
        let mut parsed: ReplayCacheManifest = serde_json::from_str(&raw).expect("parse manifest");
        parsed.normalize_derived_fields();

        assert_eq!(parsed.provider, BrokerKind::Tradovate);
        assert_eq!(parsed.row_count_total(), 390);
        assert_eq!(parsed.available_bar_shapes, vec![BarType::minute(1)]);
        assert_eq!(parsed.available_chart_modes, vec![CandleMode::HeikinAshi]);
        assert!(parsed.badges.contains(&"server-bars".to_string()));
        assert!(parsed.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
        assert!(!parsed.supports_replay(BarType::minute(5), CandleMode::HeikinAshi, None));
    }

    #[test]
    fn manifest_round_trips_contract_metadata_sources_and_suggested_coverage() {
        let mut manifest = sample_manifest();
        manifest.contract_metadata = Some(ReplayCacheContractMetadata {
            context: ReplayCacheMetadataContext {
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_id: Some(42),
                user_name: Some("tester".to_string()),
                accounts: vec![ReplayCacheMetadataAccount {
                    id: 7,
                    name: "DEMO7".to_string(),
                }],
                accounts_endpoint: Some("account/list".to_string()),
                accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
            },
            contract: ReplayCacheMetadataSnapshot {
                endpoint: "contract/item".to_string(),
                fetched_at: dt("2026-07-24T00:00:00Z"),
                source_timestamp: Some(dt("2026-07-23T23:59:00Z")),
                payload: json!({
                    "id": 123,
                    "contractMaturityId": 62531,
                    "providerTickSize": 0.25
                }),
            },
            maturity: Some(ReplayCacheMetadataSnapshot {
                endpoint: "contractMaturity/item".to_string(),
                fetched_at: dt("2026-07-24T00:00:01Z"),
                source_timestamp: None,
                payload: json!({
                    "id": 62531,
                    "productId": 1878809,
                    "expirationDate": "2026-09-18T13:30:00Z"
                }),
            }),
            maturity_chain: None,
            product: None,
            product_sessions: None,
            product_margins: None,
            contract_margins: None,
            fee_params: Some(ReplayCacheMetadataSnapshot {
                endpoint: "contract/getproductfeeparams".to_string(),
                fetched_at: dt("2026-07-24T00:00:02Z"),
                source_timestamp: None,
                payload: json!({"commission": 0.39, "dayMargin": 1380.98}),
            }),
            suggested_coverage: Some(ReplayCacheSuggestedCoverage {
                start_date: NaiveDate::from_ymd_opt(2026, 6, 19).expect("date"),
                end_date: NaiveDate::from_ymd_opt(2026, 9, 18).expect("date"),
                basis: "adjacent maturities".to_string(),
                estimated: true,
            }),
        });

        let raw = serde_json::to_string(&manifest).expect("serialize manifest");
        let parsed: ReplayCacheManifest = serde_json::from_str(&raw).expect("parse manifest");
        let metadata = parsed.contract_metadata.expect("contract metadata");

        assert_eq!(metadata.context.accounts[0].name, "DEMO7");
        assert_eq!(metadata.contract.endpoint, "contract/item");
        assert_eq!(
            metadata.fee_params.expect("fee params").payload["commission"],
            0.39
        );
        assert_eq!(
            metadata
                .suggested_coverage
                .expect("coverage")
                .start_date
                .to_string(),
            "2026-06-19"
        );
    }

    #[test]
    fn metadata_merge_reuses_missing_snapshots_only_for_the_same_complete_identity() {
        let existing = sample_contract_metadata();
        let mut incoming = existing.clone();
        incoming.context.accounts.reverse();
        incoming.product_sessions = None;
        incoming.product_margins = None;
        incoming.contract_margins = None;
        incoming.fee_params = None;
        incoming.suggested_coverage = None;

        let merged = merge_contract_metadata(Some(existing), Some(incoming))
            .expect("same identity metadata");

        assert!(merged.product_sessions.is_some());
        assert!(merged.product_margins.is_some());
        assert!(merged.contract_margins.is_some());
        assert!(merged.fee_params.is_some());
        assert!(merged.suggested_coverage.is_some());
    }

    #[test]
    fn metadata_merge_does_not_cross_provider_or_account_contract_product_identity() {
        let existing = sample_contract_metadata();
        let partial = || {
            let mut metadata = sample_contract_metadata();
            metadata.product_sessions = None;
            metadata.product_margins = None;
            metadata.contract_margins = None;
            metadata.fee_params = None;
            metadata.suggested_coverage = None;
            metadata
        };
        let assert_not_inherited = |incoming: ReplayCacheContractMetadata| {
            let merged = merge_contract_metadata(Some(existing.clone()), Some(incoming))
                .expect("incoming metadata");
            assert!(merged.product_sessions.is_none());
            assert!(merged.product_margins.is_none());
            assert!(merged.contract_margins.is_none());
            assert!(merged.fee_params.is_none());
            assert!(merged.suggested_coverage.is_none());
        };

        let mut changed = partial();
        changed.context.provider = BrokerKind::Ironbeam;
        assert_not_inherited(changed);

        let mut changed = partial();
        changed.context.env = TradingEnvironment::Live;
        assert_not_inherited(changed);

        let mut changed = partial();
        changed.context.user_id = Some(99);
        assert_not_inherited(changed);

        let mut changed = partial();
        changed.context.accounts[0].id = 99;
        assert_not_inherited(changed);

        let mut changed = partial();
        changed.contract.payload["id"] = json!(456);
        assert_not_inherited(changed);

        let mut changed = partial();
        changed.product.as_mut().expect("product").payload["id"] = json!(999);
        changed.maturity.as_mut().expect("maturity").payload["productId"] = json!(999);
        assert_not_inherited(changed);
    }

    #[test]
    fn metadata_merge_fails_closed_when_identity_is_unavailable() {
        let existing = sample_contract_metadata();
        let mut incoming = sample_contract_metadata();
        incoming.context.accounts.clear();
        incoming.fee_params = None;

        let merged = merge_contract_metadata(Some(existing.clone()), Some(incoming))
            .expect("incoming metadata");
        assert!(merged.fee_params.is_none());
        assert!(merge_contract_metadata(Some(existing), None).is_none());
    }

    #[test]
    fn manifest_errors_make_dataset_unservable() {
        let mut manifest = sample_manifest();
        manifest.errors.push("partial download".to_string());

        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));

        let mut manifest = sample_manifest();
        manifest.files[0]
            .errors
            .push("missing data page".to_string());

        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
    }

    #[test]
    fn cache_library_discovers_manifests_without_reading_data_files() {
        let root = temp_cache_dir("scan");
        let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
        fs::create_dir_all(dataset_dir.join("server-bars")).expect("create dirs");
        fs::write(
            dataset_dir.join(MANIFEST_FILE_NAME),
            serde_json::to_vec_pretty(&sample_manifest()).expect("serialize manifest"),
        )
        .expect("write manifest");
        fs::write(
            dataset_dir.join("server-bars/1m-heikin.parquet"),
            b"not parquet",
        )
        .expect("write ignored data file");

        let library = ReplayCacheLibrary::scan(&root);

        assert_eq!(library.datasets.len(), 1);
        assert!(library.warnings.is_empty());
        assert_eq!(
            library.datasets[0].manifest.display_name,
            "MESU6 2026-07-23 1m HA"
        );
        assert!(
            library
                .first_serving(BarType::minute(1), CandleMode::HeikinAshi, None)
                .is_some()
        );
    }

    #[test]
    fn server_bar_cache_path_shape_is_deterministic_and_sanitized() {
        let root = PathBuf::from("/tmp/cache-root");
        let dir = replay_cache_dataset_dir(
            &root,
            BrokerKind::Tradovate,
            TradingEnvironment::Sim,
            "ME S",
            "MES/U6",
            NaiveDate::from_ymd_opt(2026, 7, 23).expect("date"),
        );
        let relative = server_bars_relative_path(
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-25T00:00:00Z"),
            BarType::volume(6500),
        );

        assert_eq!(
            dir,
            PathBuf::from("/tmp/cache-root/tradovate/sim/ME_S/MESU6/2026-07-23")
        );
        assert_eq!(
            relative,
            PathBuf::from("server-bars/2026-07-23_to_2026-07-25_6500volume.jsonl")
        );
    }

    #[test]
    fn normalize_server_bars_filters_sorts_and_deduplicates_rows() {
        let rows = normalize_server_bar_rows(vec![
            bar("2026-07-23T00:01:00Z", 2.0),
            Bar {
                open: f64::NAN,
                ..bar("2026-07-23T00:02:00Z", 3.0)
            },
            bar("2026-07-23T00:00:00Z", 1.0),
            bar("2026-07-23T00:01:00Z", 4.0),
        ]);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].open, 1.0);
        assert_eq!(rows[1].open, 2.0);
    }

    #[test]
    fn normalize_raw_ticks_sorts_deduplicates_ids_and_drops_bad_rows() {
        let mut bad = raw_tick("2026-07-23T00:02:00Z", Some(3), 100.0);
        bad.size = f64::NAN;
        let normalized = normalize_raw_tick_rows(vec![
            raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
            bad,
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
            raw_tick("2026-07-23T00:01:01Z", Some(2), 102.0),
        ]);

        assert_eq!(normalized.rows.len(), 2);
        assert_eq!(normalized.duplicate_tick_ids, 1);
        assert_eq!(normalized.dropped_rows, 1);
        assert_eq!(normalized.rows[0].tick_id, Some(1));
        assert_eq!(normalized.rows[1].tick_id, Some(2));
    }

    #[test]
    fn raw_tick_loader_fails_closed_when_multiple_datasets_match() {
        let root = temp_cache_dir("ambiguous-raw-ticks");
        let mut first_manifest = sample_manifest();
        first_manifest.display_name = "MESU6 raw ticks".to_string();
        first_manifest.source_kind = ReplayCacheSourceKind::RawTicks;
        first_manifest.files[0].source_kind = ReplayCacheSourceKind::RawTicks;
        first_manifest.files[0].format = ReplayCacheFileFormat::Parquet;
        first_manifest.files[0].schema_version = Some(RAW_TICKS_SCHEMA_VERSION);
        first_manifest.files[0].data_hash = Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: "0000000000000000".to_string(),
        });
        first_manifest.files[0].market_shape = ReplayCacheMarketShape {
            bar_type: None,
            chart_mode: None,
            session_template: Some("Globex".to_string()),
        };

        let mut second_manifest = first_manifest.clone();
        second_manifest.display_name = "ESU6 raw ticks".to_string();
        second_manifest.instrument.symbol = "ES".to_string();
        second_manifest.contract.symbol = "ESU6".to_string();

        let library = ReplayCacheLibrary {
            root: root.clone(),
            datasets: vec![
                ReplayCacheDataset {
                    manifest_path: root.join("first/manifest.json"),
                    dataset_dir: root.join("first"),
                    manifest: first_manifest,
                },
                ReplayCacheDataset {
                    manifest_path: root.join("second/manifest.json"),
                    dataset_dir: root.join("second"),
                    manifest: second_manifest,
                },
            ],
            warnings: Vec::new(),
        };

        let err = library
            .load_unique_raw_ticks_parquet(None)
            .expect_err("ambiguous raw tick datasets must not be selected implicitly");
        assert!(
            err.to_string()
                .contains("multiple cached raw-tick datasets")
        );
        assert!(err.to_string().contains("MESU6 raw ticks"));
        assert!(err.to_string().contains("ESU6 raw ticks"));
    }

    #[test]
    fn write_server_bars_jsonl_cache_writes_manifest_and_data_file() {
        let root = temp_cache_dir("write");
        let outcome = write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
            cache_root: root.clone(),
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({
                "md": "getChart",
                "chartDescription": BarType::minute(1).chart_description()
            }),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            bars: vec![bar("2026-07-23T00:00:00Z", 1.0)],
            warnings: vec!["synthetic fixture".to_string()],
            display_name: None,
            tags: None,
            notes: Some("test write".to_string()),
        })
        .expect("write cache");

        assert_eq!(outcome.row_count, 1);
        assert!(outcome.data_path.exists());
        assert!(outcome.manifest_path.exists());

        let data = fs::read_to_string(&outcome.data_path).expect("read data");
        assert_eq!(data.lines().count(), 1);
        assert!(data.contains("\"timestamp\":\"2026-07-23T00:00:00Z\""));

        let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
        assert_eq!(manifest.provider, BrokerKind::Tradovate);
        assert_eq!(manifest.source_kind, ReplayCacheSourceKind::ServerBars);
        assert_eq!(manifest.files[0].format, ReplayCacheFileFormat::Jsonl);
        assert_eq!(manifest.files[0].row_count, 1);
        assert_eq!(
            manifest.files[0]
                .data_hash
                .as_ref()
                .map(|hash| hash.algorithm.as_str()),
            Some("fnv1a64")
        );
        assert_eq!(
            manifest.available_chart_modes,
            vec![CandleMode::Standard, CandleMode::HeikinAshi]
        );
        assert!(manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));

        let library = ReplayCacheLibrary::scan(root.clone());
        let loaded = library
            .load_first_server_bars_jsonl(BarType::minute(1), CandleMode::HeikinAshi, None)
            .expect("load first matching cached server bars")
            .expect("matching cached server bars");
        assert_eq!(loaded.bars.len(), 1);
        assert_eq!(loaded.bars[0].close, 1.5);
        assert_eq!(
            loaded.file.relative_path.parent(),
            Some(Path::new("server-bars"))
        );
        assert!(
            loaded
                .file
                .relative_path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| {
                    name.starts_with("2026-07-23_to_2026-07-24_1minute_v")
                        && name.ends_with(".jsonl")
                })
        );
    }

    #[test]
    fn concurrent_manifest_writers_preserve_distinct_shapes_with_advisory_locking() {
        fn write_for(root: PathBuf, bar_type: BarType, price: f64) -> ReplayCacheWriteOutcome {
            write_server_bars_parquet_cache(ReplayCacheServerBarsWrite {
                cache_root: root,
                target: None,
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                instrument: ReplayCacheInstrument {
                    symbol: "MES".to_string(),
                    name: None,
                    exchange: None,
                },
                contract: ReplayCacheContract {
                    symbol: "MESU6".to_string(),
                    id: Some(123),
                    expiration: None,
                },
                request_start: dt("2026-07-23T00:00:00Z"),
                request_end: dt("2026-07-24T00:00:00Z"),
                source_kind: ReplayCacheSourceKind::ServerBars,
                download_request: json!({"md": "getChart"}),
                bar_type,
                tick_specs: ReplayCacheTickSpecs {
                    tick_size: 0.25,
                    value_per_point: 5.0,
                },
                contract_metadata: None,
                session_template: Some("Globex".to_string()),
                bars: vec![bar("2026-07-23T00:00:00Z", price)],
                warnings: Vec::new(),
                display_name: None,
                tags: None,
                notes: None,
            })
            .expect("concurrent cache write")
        }

        let root = temp_cache_dir("concurrent-manifest-writers");
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut tasks = Vec::new();
        for (bar_type, price) in [(BarType::minute(1), 1.0), (BarType::volume(6500), 2.0)] {
            let root = root.clone();
            let barrier = barrier.clone();
            tasks.push(std::thread::spawn(move || {
                barrier.wait();
                write_for(root, bar_type, price)
            }));
        }
        barrier.wait();
        let outcomes = tasks
            .into_iter()
            .map(|task| task.join().expect("writer thread"))
            .collect::<Vec<_>>();

        let manifest = ReplayCacheManifest::from_path(&outcomes[0].manifest_path)
            .expect("concurrent manifest");
        assert_eq!(manifest.files.len(), 2);
        assert!(manifest.available_bar_shapes.contains(&BarType::minute(1)));
        assert!(
            manifest
                .available_bar_shapes
                .contains(&BarType::volume(6500))
        );
        assert!(
            outcomes[0]
                .dataset_dir
                .join(MANIFEST_LOCK_FILE_NAME)
                .is_file()
        );
        for entry in fs::read_dir(&outcomes[0].dataset_dir).expect("dataset directory") {
            let name = entry
                .expect("dataset entry")
                .file_name()
                .to_string_lossy()
                .into_owned();
            assert!(!name.contains(".tmp-"), "orphan temporary file: {name}");
        }
        for entry in fs::read_dir(outcomes[0].dataset_dir.join("server-bars"))
            .expect("server bars directory")
        {
            let name = entry
                .expect("server bar entry")
                .file_name()
                .to_string_lossy()
                .into_owned();
            assert!(!name.contains(".tmp-"), "orphan temporary file: {name}");
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn manifest_lock_contention_times_out_without_removing_the_stable_lock_file() {
        let root = temp_cache_dir("manifest-lock-timeout");
        fs::create_dir_all(&root).expect("lock directory");
        let lock_path = root.join(MANIFEST_LOCK_FILE_NAME);
        let lock = ReplayManifestLock::acquire(&root).expect("first lock");
        let started = Instant::now();
        let err = ReplayManifestLock::acquire_with_timeout(&root, Duration::from_millis(60))
            .err()
            .expect("contending lock should time out");

        assert!(err.to_string().contains("timed out waiting"));
        assert!(started.elapsed() >= Duration::from_millis(40));
        assert!(lock_path.exists());
        drop(lock);
        assert!(lock_path.exists());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn manifest_lock_waiter_acquires_after_raii_release() {
        let root = temp_cache_dir("manifest-lock-release");
        let lock = ReplayManifestLock::acquire(&root).expect("first lock");
        let ready = Arc::new(std::sync::Barrier::new(2));
        let ready_waiter = ready.clone();
        let waiter_root = root.clone();
        let waiter = std::thread::spawn(move || {
            ready_waiter.wait();
            ReplayManifestLock::acquire_with_timeout(&waiter_root, Duration::from_millis(500))
                .expect("waiter acquires released lock")
        });

        ready.wait();
        std::thread::sleep(Duration::from_millis(40));
        assert!(!waiter.is_finished());
        drop(lock);
        drop(waiter.join().expect("waiter thread"));
        assert!(root.join(MANIFEST_LOCK_FILE_NAME).is_file());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn manifest_lock_rejects_a_symlink_lock_file() {
        use std::os::unix::fs::symlink;

        let root = temp_cache_dir("manifest-lock-symlink");
        let outside = temp_cache_dir("manifest-lock-symlink-outside");
        fs::create_dir_all(&root).expect("lock directory");
        fs::create_dir_all(&outside).expect("outside directory");
        let outside_file = outside.join("outside.lock");
        fs::write(&outside_file, b"unchanged").expect("outside lock file");
        symlink(&outside_file, root.join(MANIFEST_LOCK_FILE_NAME)).expect("lock symlink");

        let err = ReplayManifestLock::acquire(&root)
            .err()
            .expect("symlink lock must be rejected");

        assert!(err.to_string().contains("open replay cache manifest lock"));
        assert_eq!(fs::read(&outside_file).expect("outside file"), b"unchanged");
    }

    #[cfg(unix)]
    #[test]
    fn generated_cache_dataset_rejects_symlink_path_escape_before_creation() {
        use std::os::unix::fs::symlink;

        let root = temp_cache_dir("generated-path-escape");
        let outside = temp_cache_dir("generated-path-outside");
        fs::create_dir_all(&root).expect("cache root");
        fs::create_dir_all(&outside).expect("outside root");
        symlink(&outside, root.join("tradovate")).expect("escape symlink");
        let canonical_root = fs::canonicalize(&root).expect("canonical root");
        let dataset_dir = replay_cache_dataset_dir(
            &canonical_root,
            BrokerKind::Tradovate,
            TradingEnvironment::Sim,
            "MES",
            "MESU6",
            NaiveDate::from_ymd_opt(2026, 7, 23).expect("date"),
        );

        let err = create_cache_dataset_without_symlinks(&canonical_root, &dataset_dir)
            .expect_err("symlink cache component must fail closed");

        assert!(err.to_string().contains("contains symlink"));
        assert!(!outside.join("sim").exists());
    }

    #[test]
    fn raw_tick_parquet_file_round_trips_rows() {
        let root = temp_cache_dir("raw-parquet-round-trip");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = vec![
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
            raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
        ];

        write_raw_ticks_parquet_file(&path, &rows).expect("write parquet");
        let loaded = read_raw_ticks_parquet_file(&path).expect("read parquet");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].tick_id, Some(1));
        assert_eq!(loaded[0].price, 100.0);
        assert_eq!(loaded[0].bid_price, Some(99.75));
        assert_eq!(loaded[1].packet_source.as_deref(), Some("db"));
    }

    #[test]
    fn raw_tick_stream_prunes_row_groups_and_uses_half_open_ranges() {
        let root = temp_cache_dir("raw-parquet-range");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = vec![
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
            raw_tick("2026-07-23T00:00:01Z", Some(2), 100.25),
            raw_tick("2026-07-23T00:00:02Z", Some(3), 100.5),
            raw_tick("2026-07-23T00:00:03Z", Some(4), 100.75),
            raw_tick("2026-07-23T00:00:04Z", Some(5), 101.0),
            raw_tick("2026-07-23T00:00:05Z", Some(6), 101.25),
        ];
        write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1)
            .expect("write multi-row-group parquet");
        let range =
            ReplayCacheTimeRange::new(dt("2026-07-23T00:00:02Z"), dt("2026-07-23T00:00:04Z"))
                .expect("range");

        let (selected, stats) =
            read_raw_ticks_parquet_file_range(&path, Some(&range)).expect("stream range");

        assert_eq!(
            selected.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
            vec![Some(3), Some(4)]
        );
        assert_eq!(stats.total_row_groups, 3);
        assert_eq!(stats.selected_row_groups, 1);
        assert_eq!(stats.pruned_row_groups, 2);
        assert_eq!(stats.decoded_rows, 2);
        assert_eq!(stats.emitted_rows, 2);
        assert_eq!(read_raw_ticks_parquet_file(&path).expect("full read"), rows);
    }

    #[test]
    fn parquet_pruning_bounds_fail_open_for_missing_inexact_and_reversed_statistics() {
        use parquet::file::statistics::ValueStatistics;

        assert_eq!(exact_non_null_i64_bounds(None), None);
        let missing = Statistics::int64(None, None, None, Some(0), false);
        assert_eq!(exact_non_null_i64_bounds(Some(&missing)), None);
        let inexact = Statistics::Int64(
            ValueStatistics::new(Some(10_i64), Some(20_i64), None, Some(0), false)
                .with_min_is_exact(false),
        );
        assert_eq!(exact_non_null_i64_bounds(Some(&inexact)), None);
        let reversed = Statistics::int64(Some(20), Some(10), None, Some(0), false);
        assert_eq!(exact_non_null_i64_bounds(Some(&reversed)), None);
        let nulls = Statistics::int64(Some(10), Some(20), None, Some(1), false);
        assert_eq!(exact_non_null_i64_bounds(Some(&nulls)), None);
        let trusted = Statistics::int64(Some(10), Some(20), None, Some(0), false);
        assert_eq!(exact_non_null_i64_bounds(Some(&trusted)), Some((10, 20)));
    }

    #[test]
    fn parquet_pruning_decodes_all_groups_when_ts_ns_schema_or_statistics_are_uncertain() {
        let root = temp_cache_dir("parquet-pruning-fail-open");
        fs::create_dir_all(&root).expect("create temp dir");
        let rows = sequential_raw_ticks(4);
        let range =
            ReplayCacheTimeRange::new(dt("2026-07-24T00:00:00Z"), dt("2026-07-24T00:01:00Z"))
                .expect("range");

        let no_stats_path = root.join("no-stats.parquet");
        write_raw_ticks_with_schema(
            &no_stats_path,
            &rows,
            raw_ticks_parquet_schema(),
            EnabledStatistics::None,
            2,
        );
        let builder = ParquetRecordBatchReaderBuilder::try_new(
            File::open(&no_stats_path).expect("open no-stats fixture"),
        )
        .expect("read no-stats metadata");
        let (selected, stats) =
            parquet_row_groups_for_range(&builder, Some(&range)).expect("select row groups");
        assert_eq!(selected, vec![0, 1]);
        assert_eq!(stats.pruned_row_groups, 0);

        let mut fields = raw_ticks_parquet_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields[1] = Field::new("not_ts_ns", DataType::Int64, false);
        let wrong_schema = Arc::new(Schema::new(fields));
        let wrong_column_path = root.join("wrong-column.parquet");
        write_raw_ticks_with_schema(
            &wrong_column_path,
            &rows,
            wrong_schema,
            EnabledStatistics::Chunk,
            2,
        );
        let builder = ParquetRecordBatchReaderBuilder::try_new(
            File::open(&wrong_column_path).expect("open wrong-column fixture"),
        )
        .expect("read wrong-column metadata");
        let (selected, stats) =
            parquet_row_groups_for_range(&builder, Some(&range)).expect("select row groups");
        assert_eq!(selected, vec![0, 1]);
        assert_eq!(stats.pruned_row_groups, 0);
    }

    #[test]
    fn raw_tick_stream_rejects_duplicate_ids_across_row_groups() {
        let root = temp_cache_dir("raw-parquet-cross-group-duplicate");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let mut rows = sequential_raw_ticks(4);
        rows[2].tick_id = rows[1].tick_id;
        write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1)
            .expect("write duplicate-id fixture");

        let err = stream_raw_ticks_parquet_file(&path, None, |_| Ok(()))
            .expect_err("cross-group duplicate id must fail closed");
        assert!(
            err.to_string()
                .contains("overlapping or non-monotonic tick-id ranges")
        );
    }

    #[test]
    fn raw_tick_schema_v1_requires_manifest_provenance_and_v2_rejects_unprovable_ids() {
        let root = temp_cache_dir("raw-parquet-legacy-id-schema");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = sequential_raw_ticks(2);
        let mut fields = raw_ticks_parquet_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields[2] = Field::new("tick_id", DataType::Int64, true);
        write_raw_ticks_with_schema(
            &path,
            &rows,
            Arc::new(Schema::new(fields)),
            EnabledStatistics::Chunk,
            2,
        );
        let err = stream_raw_ticks_parquet_file(&path, None, |_| Ok(()))
            .expect_err("nullable legacy tick IDs require manifest provenance");
        assert!(
            err.to_string()
                .contains("requires a manifest-declared schema version")
        );

        let mut missing = rows.clone();
        missing[0].tick_id = None;
        assert!(
            validate_raw_tick_id_write_invariant(&missing)
                .expect_err("missing tick id")
                .to_string()
                .contains("schema v2 cache was not written")
        );
        let mut non_monotonic = rows;
        non_monotonic[1].tick_id = Some(0);
        assert!(
            validate_raw_tick_id_write_invariant(&non_monotonic)
                .expect_err("non-monotonic tick id")
                .to_string()
                .contains("strictly increasing")
        );
    }

    #[test]
    fn schema_v1_manifest_hash_enables_sequence_ids_and_rejects_missing_or_modified_data() {
        let root = temp_cache_dir("raw-parquet-v1-compatibility");
        let rows = vec![
            raw_tick("2026-07-23T00:00:00Z", Some(900), 100.0),
            raw_tick("2026-07-23T00:00:01Z", None, 100.25),
            raw_tick("2026-07-23T00:00:02Z", Some(100), 100.5),
        ];
        let dataset = write_legacy_v1_raw_tick_dataset(&root, &rows);

        let resolved = dataset
            .resolve_raw_ticks_parquet_file(None)
            .expect("resolve hash-verified schema v1 cache");
        let mut replayed = Vec::new();
        stream_resolved_raw_ticks_parquet(&resolved, None, |row| {
            replayed.push(row);
            Ok(())
        })
        .expect("stream schema v1 cache");
        assert_eq!(
            replayed.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            replayed.iter().map(|row| row.price).collect::<Vec<_>>(),
            rows.iter().map(|row| row.price).collect::<Vec<_>>()
        );

        let mut missing_hash = dataset.clone();
        missing_hash.manifest.files[0].data_hash = None;
        let err = missing_hash
            .resolve_raw_ticks_parquet_file(None)
            .expect_err("schema v1 without writer hash must fail");
        assert!(err.to_string().contains("has no writer data hash"));
        assert!(err.to_string().contains("re-download"));

        let data_path = dataset
            .dataset_dir
            .join(&dataset.manifest.files[0].relative_path);
        OpenOptions::new()
            .append(true)
            .open(&data_path)
            .expect("open legacy file for corruption")
            .write_all(b"modified")
            .expect("modify legacy file");
        let err = dataset
            .resolve_raw_ticks_parquet_file(None)
            .expect_err("modified schema v1 cache must fail hash verification");
        assert!(err.to_string().contains("hash mismatch"));
        assert!(err.to_string().contains("re-download"));
    }

    #[test]
    fn schema_v2_download_with_missing_provider_id_fails_before_cache_commit() {
        let root = temp_cache_dir("raw-parquet-v2-missing-provider-id");
        let err = write_raw_ticks_parquet_cache(raw_tick_cache_write(
            root.clone(),
            vec![raw_tick("2026-07-23T00:00:00Z", None, 100.0)],
        ))
        .expect_err("schema v2 must reject a missing provider tick id");

        assert!(
            err.to_string()
                .contains("provider response omitted stable tick IDs")
        );
        assert!(err.to_string().contains("schema v2 cache was not written"));
        assert!(!root.join(MANIFEST_FILE_NAME).exists());
        assert!(ReplayCacheLibrary::scan(root).datasets.is_empty());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn concurrent_derivations_share_an_unlinked_lease_without_cursor_races() {
        let root = temp_cache_dir("raw-parquet-concurrent-lease");
        let original_rows = sequential_raw_ticks(PARQUET_ROW_GROUP_ROWS + 1);
        write_raw_ticks_parquet_cache(raw_tick_cache_write(root.clone(), original_rows.clone()))
            .expect("write original raw tick cache");
        let resolved = ReplayCacheLibrary::scan(&root)
            .resolve_unique_raw_ticks_parquet(None)
            .expect("resolve original cache")
            .expect("raw tick dataset");
        let old_path = resolved.data_path.clone();

        let mut refreshed_rows = original_rows;
        for row in &mut refreshed_rows {
            row.price += 10.0;
        }
        let refreshed = write_raw_ticks_parquet_cache(raw_tick_cache_write(root, refreshed_rows))
            .expect("refresh raw tick cache");
        assert_ne!(refreshed.data_path, old_path);
        assert!(!old_path.exists(), "refresh should unlink the old pathname");

        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut readers = Vec::new();
        for _ in 0..2 {
            let resolved = resolved.clone();
            let barrier = barrier.clone();
            readers.push(std::thread::spawn(move || {
                barrier.wait();
                let mut count = 0_u64;
                let mut first = None;
                let mut last = None;
                stream_resolved_raw_ticks_parquet(&resolved, None, |row| {
                    count += 1;
                    first.get_or_insert((row.tick_id, row.price));
                    last = Some((row.tick_id, row.price));
                    Ok(())
                })?;
                Ok::<_, anyhow::Error>((count, first, last))
            }));
        }
        barrier.wait();

        for reader in readers {
            let (count, first, last) = reader
                .join()
                .expect("reader thread")
                .expect("concurrent position-independent derivation");
            assert_eq!(count, (PARQUET_ROW_GROUP_ROWS + 1) as u64);
            assert_eq!(first, Some((Some(1), 100.0)));
            assert_eq!(
                last,
                Some((
                    Some((PARQUET_ROW_GROUP_ROWS + 1) as i64),
                    100.0 + (PARQUET_ROW_GROUP_ROWS % 8) as f64 * 0.25
                ))
            );
        }
    }

    #[test]
    fn raw_tick_stream_propagates_callback_failure_and_supports_empty_ranges() {
        let root = temp_cache_dir("raw-parquet-callback");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = sequential_raw_ticks(4);
        write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1).expect("write fixture");

        let mut callbacks = 0;
        let err = stream_raw_ticks_parquet_file(&path, None, |_| {
            callbacks += 1;
            if callbacks == 2 {
                bail!("synthetic callback failure");
            }
            Ok(())
        })
        .expect_err("callback failure must stop streaming");
        assert!(err.to_string().contains("synthetic callback failure"));
        assert_eq!(callbacks, 2);

        let range =
            ReplayCacheTimeRange::new(dt("2026-07-24T00:00:00Z"), dt("2026-07-24T00:01:00Z"))
                .expect("range");
        let (selected, stats) =
            read_raw_ticks_parquet_file_range(&path, Some(&range)).expect("empty range");
        assert!(selected.is_empty());
        assert_eq!(stats.selected_row_groups, 0);
        assert_eq!(stats.emitted_rows, 0);
    }

    #[test]
    fn raw_tick_parquet_honors_production_row_group_and_batch_bounds() {
        let root = temp_cache_dir("raw-parquet-production-bounds");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = sequential_raw_ticks(PARQUET_ROW_GROUP_ROWS + 1);
        write_raw_ticks_parquet_file(&path, &rows).expect("write production-sized parquet");

        let mut emitted = 0_u64;
        let stats = stream_raw_ticks_parquet_file(&path, None, |_| {
            emitted += 1;
            Ok(())
        })
        .expect("stream production-sized parquet");
        assert_eq!(stats.total_row_groups, 2);
        assert_eq!(stats.record_batches, 9);
        assert!(stats.max_record_batch_rows <= PARQUET_READ_BATCH_ROWS);
        assert_eq!(stats.max_record_batch_rows, PARQUET_READ_BATCH_ROWS);
        assert_eq!(emitted, (PARQUET_ROW_GROUP_ROWS + 1) as u64);
    }

    #[test]
    fn server_bar_range_reader_prunes_row_groups_and_rejects_malformed_rows() {
        let root = temp_cache_dir("server-parquet-range");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("bars.parquet");
        let rows = [
            bar("2026-07-23T00:00:00Z", 100.0),
            bar("2026-07-23T00:01:00Z", 101.0),
            bar("2026-07-23T00:02:00Z", 102.0),
            bar("2026-07-23T00:03:00Z", 103.0),
            bar("2026-07-23T00:04:00Z", 104.0),
            bar("2026-07-23T00:05:00Z", 105.0),
        ]
        .iter()
        .map(ReplayCacheServerBarRow::from_bar)
        .collect::<Vec<_>>();
        write_server_bars_parquet_file_with_limits(&path, &rows, 2, 1)
            .expect("write multi-row-group parquet");
        let range =
            ReplayCacheTimeRange::new(dt("2026-07-23T00:02:00Z"), dt("2026-07-23T00:04:00Z"))
                .expect("range");

        let (selected, stats) =
            read_server_bars_parquet_file_range(&path, Some(&range)).expect("read range");
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].open, 102.0);
        assert_eq!(selected[1].open, 103.0);
        assert_eq!(stats.pruned_row_groups, 2);

        let malformed_path = root.join("malformed-ticks.parquet");
        let mut malformed = raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0);
        malformed.ts_ns += 1;
        write_raw_ticks_parquet_file_with_limits(&malformed_path, &[malformed], 1, 1)
            .expect("write malformed fixture");
        let err = stream_raw_ticks_parquet_file(&malformed_path, None, |_| Ok(()))
            .expect_err("malformed tick must fail");
        assert!(err.to_string().contains("validate raw tick row"));
    }

    #[test]
    fn atomic_writer_preserves_existing_file_when_replacement_fails() {
        let root = temp_cache_dir("atomic-replacement");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("manifest.json");
        fs::write(&path, b"working-cache").expect("write original");

        let err = write_file_atomically(&path, |file| {
            file.write_all(b"partial-replacement")?;
            bail!("synthetic write failure")
        })
        .expect_err("replacement must fail");

        assert!(err.to_string().contains("synthetic write failure"));
        assert_eq!(fs::read(&path).expect("read original"), b"working-cache");
        assert!(
            fs::read_dir(&root)
                .expect("read temp dir")
                .flatten()
                .all(|entry| !entry.file_name().to_string_lossy().contains(".tmp-"))
        );
    }

    #[test]
    fn streaming_file_hash_matches_in_memory_hash_across_chunks() {
        let root = temp_cache_dir("streaming-hash");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("large.bin");
        let bytes = (0..(128 * 1024 + 17))
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        fs::write(&path, &bytes).expect("write hash fixture");

        assert_eq!(
            fnv1a64_file_hex(&path).expect("streaming hash"),
            fnv1a64_hex(&bytes)
        );
    }

    #[test]
    fn server_bar_parquet_file_round_trips_and_is_preferred_over_jsonl() {
        let root = temp_cache_dir("server-parquet");
        let write = ReplayCacheServerBarsWrite {
            cache_root: root.clone(),
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({"md": "getChart"}),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            bars: vec![bar("2026-07-23T00:00:00Z", 100.0)],
            warnings: Vec::new(),
            display_name: Some("MES baseline".to_string()),
            tags: Some(vec![
                " hma ".to_string(),
                "baseline".to_string(),
                "hma".to_string(),
            ]),
            notes: None,
        };
        let parquet = write_server_bars_parquet_cache(write.clone()).expect("write parquet");
        let jsonl =
            write_server_bars_jsonl_cache(write.clone()).expect("write jsonl compatibility file");
        assert_eq!(parquet.row_count, 1);
        assert_eq!(jsonl.row_count, 1);
        assert_eq!(
            parquet.data_path.extension().and_then(|ext| ext.to_str()),
            Some("parquet")
        );

        let library = ReplayCacheLibrary::scan(root.clone());
        let loaded = library
            .load_first_server_bars(BarType::minute(1), CandleMode::Standard, None)
            .expect("load parquet-preferred server bars")
            .expect("server bars exist");
        assert_eq!(loaded.file.format, ReplayCacheFileFormat::Parquet);
        assert_eq!(loaded.bars[0].close, 100.5);

        let manifest = ReplayCacheManifest::from_path(&loaded.manifest_path).expect("manifest");
        assert!(
            manifest
                .files
                .iter()
                .any(|file| file.format == ReplayCacheFileFormat::Jsonl)
        );
        assert!(
            manifest
                .files
                .iter()
                .any(|file| file.format == ReplayCacheFileFormat::Parquet)
        );
        assert_eq!(manifest.row_count_total(), 2);
        assert_eq!(manifest.preferred_row_count_total(), 1);
        assert_eq!(manifest.display_name, "MES baseline");
        assert_eq!(manifest.tags, vec!["baseline", "hma"]);

        let old_parquet_path = parquet.data_path.clone();
        let mut refresh_write = write;
        refresh_write.bars = vec![
            bar("2026-07-23T00:00:00Z", 101.0),
            bar("2026-07-23T00:01:00Z", 102.0),
        ];
        let refreshed =
            write_server_bars_parquet_cache(refresh_write.clone()).expect("refresh parquet cache");
        assert_ne!(refreshed.data_path, old_parquet_path);
        assert!(!old_parquet_path.exists());
        assert!(refreshed.data_path.exists());
        let refreshed_manifest =
            ReplayCacheManifest::from_path(&refreshed.manifest_path).expect("refreshed manifest");
        assert_eq!(
            refreshed_manifest
                .files
                .iter()
                .filter(|file| file.format == ReplayCacheFileFormat::Parquet)
                .count(),
            1
        );
        assert_eq!(refreshed_manifest.preferred_row_count_total(), 2);

        let original_dataset_dir = refreshed.dataset_dir.clone();
        let mut backward_extension = refresh_write;
        backward_extension.target = Some(ReplayDownloadCacheTarget {
            dataset_dir: original_dataset_dir.clone(),
            manifest_path: refreshed.manifest_path.clone(),
        });
        backward_extension.request_start = dt("2026-07-22T00:00:00Z");
        backward_extension.request_end = dt("2026-07-25T00:00:00Z");
        backward_extension.bars = vec![
            bar("2026-07-22T00:00:00Z", 99.0),
            bar("2026-07-24T23:59:00Z", 103.0),
        ];
        let extended = write_server_bars_parquet_cache(backward_extension)
            .expect("extend selected dataset backward");
        assert_eq!(extended.dataset_dir, original_dataset_dir);
        assert!(!root.join("tradovate/sim/MES/MESU6/2026-07-22").exists());
        let extended_manifest =
            ReplayCacheManifest::from_path(&extended.manifest_path).expect("extended manifest");
        assert_eq!(extended_manifest.coverage.start, dt("2026-07-22T00:00:00Z"));
        assert_eq!(extended_manifest.coverage.end, dt("2026-07-24T23:59:00Z"));

        let identity_err = replay_cache_write_dataset_dir(
            &root,
            BrokerKind::Tradovate,
            TradingEnvironment::Sim,
            "ES",
            &ReplayCacheContract {
                symbol: "ESU6".to_string(),
                id: Some(999),
                expiration: None,
            },
            NaiveDate::from_ymd_opt(2026, 7, 22).expect("date"),
            Some(&ReplayDownloadCacheTarget {
                dataset_dir: extended.dataset_dir,
                manifest_path: extended.manifest_path,
            }),
        )
        .expect_err("extension target identity mismatch must fail closed");
        assert!(identity_err.to_string().contains("identity does not match"));
    }

    #[test]
    fn cache_writer_rejects_extension_target_outside_configured_root() {
        let root = temp_cache_dir("target-root");
        let outside_root = temp_cache_dir("target-outside");
        fs::create_dir_all(&root).expect("create cache root");
        let outside_dataset = outside_root.join("dataset");
        fs::create_dir_all(&outside_dataset).expect("create outside dataset");
        let outside_manifest = outside_dataset.join(MANIFEST_FILE_NAME);
        fs::write(&outside_manifest, b"{}").expect("write outside manifest");

        let err = write_server_bars_parquet_cache(ReplayCacheServerBarsWrite {
            cache_root: root,
            target: Some(ReplayDownloadCacheTarget {
                dataset_dir: outside_dataset,
                manifest_path: outside_manifest,
            }),
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-22T00:00:00Z"),
            request_end: dt("2026-07-23T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({}),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: None,
            bars: vec![bar("2026-07-22T00:00:00Z", 100.0)],
            warnings: Vec::new(),
            display_name: None,
            tags: None,
            notes: None,
        })
        .expect_err("target outside cache root must fail closed");

        assert!(err.to_string().contains("outside cache root"));
    }

    #[test]
    fn manifest_marks_combined_server_bar_and_raw_tick_files_as_mixed() {
        let root = temp_cache_dir("mixed-sources");
        let server_write = ReplayCacheServerBarsWrite {
            cache_root: root.clone(),
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({"md": "getChart"}),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            bars: vec![bar("2026-07-23T00:00:00Z", 100.0)],
            warnings: Vec::new(),
            display_name: None,
            tags: None,
            notes: None,
        };
        let server = write_server_bars_parquet_cache(server_write).expect("write server bars");
        write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
            cache_root: root,
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            download_request: json!({"md": "getChart"}),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            ticks: vec![raw_tick("2026-07-23T00:00:01Z", Some(1), 100.25)],
            warnings: Vec::new(),
            display_name: None,
            tags: None,
            notes: None,
        })
        .expect("write raw ticks");

        let manifest = ReplayCacheManifest::from_path(&server.manifest_path).expect("manifest");
        assert_eq!(manifest.source_kind, ReplayCacheSourceKind::Mixed);
        assert_eq!(
            manifest.downloadable_source_kinds(),
            vec![
                ReplayCacheSourceKind::ServerBars,
                ReplayCacheSourceKind::RawTicks
            ]
        );
        assert!(manifest.badges.contains(&"server-bars".to_string()));
        assert!(manifest.badges.contains(&"raw-ticks".to_string()));
    }

    #[test]
    fn write_raw_ticks_parquet_cache_writes_manifest_without_direct_replay_support() {
        let root = temp_cache_dir("raw-parquet-cache");
        let outcome = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
            cache_root: root.clone(),
            target: None,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            download_request: json!({
                "md": "getChart",
                "chartDescription": {
                    "underlyingType": "Tick",
                    "elementSize": 1,
                    "elementSizeUnit": "UnderlyingUnits"
                }
            }),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            ticks: vec![
                raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
                raw_tick("2026-07-23T00:00:01Z", Some(1), 100.25),
            ],
            warnings: Vec::new(),
            display_name: None,
            tags: None,
            notes: Some("raw tick test".to_string()),
        })
        .expect("write raw tick cache");

        assert_eq!(outcome.row_count, 1);
        assert!(outcome.data_path.exists());
        assert_eq!(
            outcome
                .data_path
                .extension()
                .and_then(|value| value.to_str()),
            Some("parquet")
        );

        let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
        assert_eq!(manifest.source_kind, ReplayCacheSourceKind::RawTicks);
        assert_eq!(manifest.files[0].format, ReplayCacheFileFormat::Parquet);
        assert_eq!(
            manifest.files[0].schema_version,
            Some(RAW_TICKS_SCHEMA_VERSION)
        );
        assert_eq!(
            manifest.files[0].compression.as_deref(),
            Some(PARQUET_COMPRESSION_LABEL)
        );
        assert!(manifest.badges.contains(&"raw-ticks".to_string()));
        assert!(
            manifest
                .warnings
                .iter()
                .any(|warning| warning.contains("duplicate raw tick id"))
        );
        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::Standard, None));

        let loaded_ticks = read_raw_ticks_parquet_file(&outcome.data_path).expect("read ticks");
        assert_eq!(loaded_ticks.len(), 1);
        assert_eq!(loaded_ticks[0].tick_id, Some(1));
    }

    #[test]
    fn read_server_bars_jsonl_file_parses_sorts_and_deduplicates_rows() {
        let root = temp_cache_dir("read-jsonl");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("bars.jsonl");
        fs::write(
            &path,
            [
                bar_row_json("2026-07-23T00:01:00Z", 2.0),
                bar_row_json("2026-07-23T00:00:00Z", 1.0),
                bar_row_json("2026-07-23T00:01:00Z", 2.0),
            ]
            .join("\n"),
        )
        .expect("write jsonl");

        let bars = read_server_bars_jsonl_file(&path).expect("read bars");

        assert_eq!(bars.len(), 2);
        assert!(bars[0].ts_ns < bars[1].ts_ns);
        assert_eq!(bars[0].open, 1.0);
        assert_eq!(bars[1].open, 2.0);
    }

    #[test]
    fn read_server_bars_jsonl_file_rejects_invalid_rows() {
        let root = temp_cache_dir("read-bad-jsonl");
        fs::create_dir_all(&root).expect("create temp dir");
        let bad_ohlc = root.join("bad-ohlc.jsonl");
        fs::write(
            &bad_ohlc,
            json!({
                "timestamp": "2026-07-23T00:00:00Z",
                "ts_ns": dt("2026-07-23T00:00:00Z").timestamp_nanos_opt().expect("timestamp ns"),
                "open": 10.0,
                "high": 9.0,
                "low": 8.0,
                "close": 8.5
            })
            .to_string(),
        )
        .expect("write bad ohlc");
        let err = read_server_bars_jsonl_file(&bad_ohlc).expect_err("bad ohlc should fail");
        assert!(err.to_string().contains("validate JSONL bar line 1"));

        let bad_timestamp = root.join("bad-timestamp.jsonl");
        fs::write(
            &bad_timestamp,
            json!({
                "timestamp": "2026-07-23T00:00:00Z",
                "ts_ns": dt("2026-07-23T00:01:00Z").timestamp_nanos_opt().expect("timestamp ns"),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5
            })
            .to_string(),
        )
        .expect("write bad timestamp");
        let err = read_server_bars_jsonl_file(&bad_timestamp)
            .expect_err("timestamp mismatch should fail");
        assert!(err.to_string().contains("validate JSONL bar line 1"));
    }

    #[test]
    fn server_bars_jsonl_resolver_rejects_path_escape() {
        let root = temp_cache_dir("resolve-escape");
        let mut manifest = sample_manifest();
        manifest.files[0].relative_path = PathBuf::from("../escape.jsonl");
        manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
        manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
        manifest.files[0].market_shape.chart_mode = None;
        let dataset = ReplayCacheDataset {
            manifest_path: root.join(MANIFEST_FILE_NAME),
            dataset_dir: root,
            manifest,
        };

        let err = dataset
            .resolve_server_bars_jsonl_file(BarType::minute(1), CandleMode::Standard, None)
            .expect_err("escaping relative path should fail");

        assert!(
            err.to_string()
                .contains("not a safe manifest-relative path")
        );
    }

    #[test]
    fn load_server_bars_jsonl_cache_file_validates_manifest_metadata() {
        let root = temp_cache_dir("metadata-mismatch");
        let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
        let data_dir = dataset_dir.join("server-bars");
        fs::create_dir_all(&data_dir).expect("create data dir");
        let relative_path = PathBuf::from("server-bars/bars.jsonl");
        fs::write(
            dataset_dir.join(&relative_path),
            bar_row_json("2026-07-23T00:00:00Z", 1.0),
        )
        .expect("write jsonl");

        let mut manifest = sample_manifest();
        manifest.files[0].relative_path = relative_path;
        manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
        manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
        manifest.files[0].market_shape.chart_mode = None;
        manifest.files[0].row_count = 2;
        manifest.files[0].first_timestamp = dt("2026-07-23T00:00:00Z");
        manifest.files[0].last_timestamp = dt("2026-07-23T00:00:00Z");
        let dataset = ReplayCacheDataset {
            manifest_path: dataset_dir.join(MANIFEST_FILE_NAME),
            dataset_dir,
            manifest,
        };

        let err = load_server_bars_jsonl_cache_file(
            &dataset,
            BarType::minute(1),
            CandleMode::HeikinAshi,
            None,
        )
        .expect_err("row count mismatch should fail");

        assert!(err.to_string().contains("row count mismatch"));
    }
}
