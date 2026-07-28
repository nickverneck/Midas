use super::*;

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

    pub(crate) fn bounds_ns(self) -> Result<(i64, i64)> {
        Ok((
            self.start
                .timestamp_nanos_opt()
                .context("replay cache range start is outside supported nanosecond range")?,
            self.end
                .timestamp_nanos_opt()
                .context("replay cache range end is outside supported nanosecond range")?,
        ))
    }

    pub(super) fn contains_ns(self, ts_ns: i64) -> Result<bool> {
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
    #[serde(default)]
    pub request_protocol_version: u32,
}

impl ReplayCacheRawTickCheckpointIdentity {
    pub fn stable_key(&self) -> String {
        format!(
            "{}|{:?}|{}|{}|{}|{}|{}|{}|{}",
            self.provider.label(),
            self.env,
            self.instrument.symbol.trim().to_ascii_uppercase(),
            self.contract.symbol.trim().to_ascii_uppercase(),
            self.contract.id.unwrap_or_default(),
            self.request.start.timestamp_millis(),
            self.request.end.timestamp_millis(),
            self.chunk_seconds,
            self.request_protocol_version,
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
                matches!(
                    chunk.status,
                    ReplayCacheRawTickChunkStatus::Completed { .. }
                )
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
