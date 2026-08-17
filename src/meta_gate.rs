//! Causal event rows and lifecycle semantics for a crossover meta-gate.
//!
//! This module deliberately stops at the event boundary.  The existing GA/RL
//! environments operate on every bar; a meta-gate makes a decision only when
//! a configured crossover occurs.  Keeping extraction and action semantics in
//! one library prevents the CLI, web UI, and eventual Trader runtime from
//! disagreeing about what "skip" or "invert" means.

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use polars::prelude::{DataFrame, DataType, NamedFrom, ParquetReader, SerReader, Series, TimeUnit};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

pub const META_EVENT_SCHEMA: &str = "meta-event-v1";
pub const META_DATASET_INTEGRITY_SCHEMA: &str = "meta-dataset-integrity-v2";

static META_SOURCE_SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum AverageKind {
    Ema,
    Hma,
}

impl Default for AverageKind {
    fn default() -> Self {
        Self::Ema
    }
}

impl fmt::Display for AverageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ema => "ema",
            Self::Hma => "hma",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum GateAction {
    Normal,
    Skip,
    Invert,
}

impl fmt::Display for GateAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Normal => "normal",
            Self::Skip => "skip",
            Self::Invert => "invert",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaGateConfig {
    pub trigger_kind: AverageKind,
    pub trigger_fast: usize,
    pub trigger_slow: usize,
    pub context_kind: AverageKind,
    pub context_fast: usize,
    pub context_slow: usize,
    pub atr_period: usize,
    pub slope_lookback: usize,
    pub horizon_bars: usize,
    /// Currency (or account PnL units) earned per one price unit.
    pub contract_multiplier: f64,
    /// Total round-trip cost in the same units as the reported PnL.
    pub cost_pnl: f64,
}

impl Default for MetaGateConfig {
    fn default() -> Self {
        Self {
            trigger_kind: AverageKind::Ema,
            trigger_fast: 10,
            trigger_slow: 30,
            context_kind: AverageKind::Ema,
            context_fast: 210,
            context_slow: 240,
            atr_period: 14,
            slope_lookback: 5,
            horizon_bars: 30,
            contract_multiplier: 1.0,
            cost_pnl: 0.0,
        }
    }
}

impl MetaGateConfig {
    pub fn validate(&self) -> Result<()> {
        if self.trigger_fast == 0 || self.trigger_slow == 0 {
            bail!("trigger periods must be greater than zero");
        }
        if self.trigger_fast >= self.trigger_slow {
            bail!("trigger fast period must be less than trigger slow period");
        }
        if self.trigger_kind == AverageKind::Hma && self.trigger_fast < 2 {
            bail!("HMA trigger periods must be at least 2");
        }
        if self.trigger_kind == AverageKind::Hma && self.trigger_slow < 2 {
            bail!("HMA trigger periods must be at least 2");
        }
        if self.context_fast == 0 || self.context_slow == 0 {
            bail!("context periods must be greater than zero");
        }
        if self.context_fast >= self.context_slow {
            bail!("context fast period must be less than context slow period");
        }
        if self.context_kind == AverageKind::Hma && self.context_fast < 2 {
            bail!("HMA context periods must be at least 2");
        }
        if self.context_kind == AverageKind::Hma && self.context_slow < 2 {
            bail!("HMA context periods must be at least 2");
        }
        if self.atr_period == 0 {
            bail!("ATR period must be greater than zero");
        }
        if self.slope_lookback == 0 {
            bail!("slope lookback must be greater than zero");
        }
        if self.horizon_bars == 0 {
            bail!("horizon bars must be greater than zero");
        }
        if !self.contract_multiplier.is_finite() || self.contract_multiplier <= 0.0 {
            bail!("contract multiplier must be finite and greater than zero");
        }
        if !self.cost_pnl.is_finite() || self.cost_pnl < 0.0 {
            bail!("cost PnL must be finite and non-negative");
        }
        Ok(())
    }

    pub fn feature_schema(&self) -> String {
        format!(
            "trigger-{}{}-{}|context-{}{}-{}|atr-{}|slope-{}|horizon-{}",
            self.trigger_kind,
            self.trigger_fast,
            self.trigger_slow,
            self.context_kind,
            self.context_fast,
            self.context_slow,
            self.atr_period,
            self.slope_lookback,
            self.horizon_bars
        )
    }
}

#[derive(Debug, Clone)]
pub struct MetaBars {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamp_ns: Vec<i64>,
}

impl MetaBars {
    pub fn validate(&self) -> Result<()> {
        let len = self.close.len();
        if len == 0 {
            bail!("bar input is empty");
        }
        for (name, values) in [
            ("open", self.open.len()),
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("volume", self.volume.len()),
            ("timestamp_ns", self.timestamp_ns.len()),
        ] {
            if values != len {
                bail!("{name} length {values} does not match close length {len}");
            }
        }
        for (name, values) in [
            ("open", self.open.as_slice()),
            ("high", self.high.as_slice()),
            ("low", self.low.as_slice()),
            ("close", self.close.as_slice()),
            ("volume", self.volume.as_slice()),
        ] {
            if values.iter().any(|v| !v.is_finite()) {
                bail!("{name} contains non-finite values");
            }
        }
        if self
            .timestamp_ns
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            bail!("timestamp_ns must be strictly increasing");
        }
        Ok(())
    }
}

/// A single causal row.  The future outcome columns are labels/diagnostics;
/// none of them are used to build the feature values above the event boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaEvent {
    pub event_id: usize,
    pub row_idx: usize,
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub raw_direction: i8,
    pub trigger_fast_value: f64,
    pub trigger_slow_value: f64,
    pub context_fast_value: f64,
    pub context_slow_value: f64,
    pub atr: f64,
    pub trigger_spread_atr: f64,
    pub context_spread_atr: f64,
    pub trigger_fast_slope_atr: f64,
    pub trigger_slow_slope_atr: f64,
    pub context_fast_slope_atr: f64,
    pub context_slow_slope_atr: f64,
    pub efficiency_ratio: f64,
    pub rvol_20: f64,
    pub return_1: f64,
    pub return_5: f64,
    pub entry_price: f64,
    pub normal_pnl: f64,
    pub invert_pnl: f64,
    pub skip_pnl: f64,
    pub oracle_action: GateAction,
    pub entry_row_idx: usize,
    pub horizon_row_idx: usize,
}

/// Hash the serialized event payload rather than the parquet container.  This
/// makes the integrity check independent of parquet row-group/compression
/// details while still binding every generated event field, including labels
/// and audit values.
pub fn canonical_event_payload_sha256(events: &[MetaEvent]) -> String {
    let payload = serde_json::to_vec(events)
        .expect("MetaEvent serialization to an in-memory JSON buffer cannot fail");
    sha256_bytes(&payload)
}

/// Hash the source and generated-payload identities together with the
/// preparation provenance.  A trainer can recompute this from the source and
/// event parquet and reject a dataset whose metadata was changed alongside a
/// feature or reward value.
#[allow(clippy::too_many_arguments)]
pub fn canonical_dataset_fingerprint_sha256(
    source_sha256: &str,
    event_payload_sha256: &str,
    feature_schema: &str,
    instrument: &str,
    contract: &str,
    config_json: &str,
    source_path: &str,
    source_root: &str,
    source_size_bytes: i64,
    source_row_count: i64,
    timestamp_source: &str,
) -> String {
    #[derive(Serialize)]
    struct DatasetIntegrityPayload<'a> {
        schema: &'static str,
        source_sha256: &'a str,
        event_payload_sha256: &'a str,
        feature_schema: &'a str,
        instrument: &'a str,
        contract: &'a str,
        config_json: &'a str,
        source_path: &'a str,
        source_root: &'a str,
        source_size_bytes: i64,
        source_row_count: i64,
        timestamp_source: &'a str,
    }

    let payload = DatasetIntegrityPayload {
        schema: META_DATASET_INTEGRITY_SCHEMA,
        source_sha256,
        event_payload_sha256,
        feature_schema,
        instrument,
        contract,
        config_json,
        source_path,
        source_root,
        source_size_bytes,
        source_row_count,
        timestamp_source,
    };
    let bytes = serde_json::to_vec(&payload)
        .expect("dataset integrity serialization to an in-memory JSON buffer cannot fail");
    sha256_bytes(&bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

/// An immutable view of the source file used to prepare a meta-gate dataset.
///
/// The source is copied once from a no-follow descriptor after its identity
/// is checked. Hashing and parquet decoding use only that private snapshot,
/// so replacing or mutating the pathname after validation cannot redirect
/// either operation to a different file.
#[derive(Debug, Clone)]
pub struct MetaSourceSnapshot {
    pub canonical_path: PathBuf,
    pub canonical_root: PathBuf,
    pub size_bytes: i64,
    pub sha256: String,
    pub bars: MetaBars,
    pub timestamp_source: String,
}

/// Private immutable source copy used while preparing or validating a
/// meta-gate dataset.  The caller-supplied source path is only used to create
/// this copy; all hashing and decoding happens through the private file
/// descriptor after the source/root identities have been checked.
struct PrivateMetaSourceSnapshot {
    directory: PathBuf,
}

impl PrivateMetaSourceSnapshot {
    fn create() -> Result<Self> {
        let parent = std::env::temp_dir();
        for _ in 0..16 {
            let directory = parent.join(format!(
                ".midas-meta-source-{}-{}",
                std::process::id(),
                META_SOURCE_SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&directory) {
                Ok(()) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(
                            &directory,
                            std::fs::Permissions::from_mode(0o700),
                        )
                        .with_context(|| {
                            format!(
                                "restrict private meta-gate source snapshot {}",
                                directory.display()
                            )
                        })?;
                    }
                    return Ok(Self { directory });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "create private meta-gate source snapshot {}",
                            directory.display()
                        )
                    });
                }
            }
        }
        bail!("could not allocate a private meta-gate source snapshot directory")
    }

    /// Open the validated path without following the final symlink, compare
    /// its identity with the entry that passed root validation, and copy from
    /// that descriptor into a private, no-replace file.  A concurrent rename
    /// or replacement therefore fails closed before any source bytes are
    /// decoded or hashed.
    fn copy_validated_source(
        &self,
        source_path: &Path,
        validated_source_metadata: &std::fs::Metadata,
        root_path: &Path,
        validated_root_metadata: &std::fs::Metadata,
    ) -> Result<File> {
        if validated_source_metadata.file_type().is_symlink()
            || !validated_source_metadata.is_file()
        {
            bail!(
                "validated meta-gate source {} is not a regular file",
                source_path.display()
            );
        }

        let current_root_metadata = std::fs::metadata(root_path).with_context(|| {
            format!(
                "recheck trusted meta-gate source root {}",
                root_path.display()
            )
        })?;
        if !same_file_identity(validated_root_metadata, &current_root_metadata) {
            bail!(
                "trusted meta-gate source root {} was replaced during validation",
                root_path.display()
            );
        }

        let mut source_file = open_read_only_no_follow(source_path)?;
        let opened_source_metadata = source_file.metadata().with_context(|| {
            format!("inspect opened meta-gate source {}", source_path.display())
        })?;
        if !opened_source_metadata.is_file() {
            bail!(
                "opened meta-gate source {} is not a regular file",
                source_path.display()
            );
        }
        if !same_file_identity(validated_source_metadata, &opened_source_metadata) {
            bail!(
                "validated meta-gate source {} was replaced before its immutable snapshot could be created",
                source_path.display()
            );
        }

        let current_root_metadata = std::fs::metadata(root_path).with_context(|| {
            format!(
                "recheck trusted meta-gate source root {} after opening source",
                root_path.display()
            )
        })?;
        if !same_file_identity(validated_root_metadata, &current_root_metadata) {
            bail!(
                "trusted meta-gate source root {} was replaced while opening the source",
                root_path.display()
            );
        }

        let snapshot_path = self.directory.join("source.parquet");
        let mut destination_options = OpenOptions::new();
        destination_options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            destination_options.mode(0o600);
        }
        let mut destination = destination_options.open(&snapshot_path).with_context(|| {
            format!(
                "create private meta-gate source snapshot {}",
                snapshot_path.display()
            )
        })?;
        std::io::copy(&mut source_file, &mut destination).with_context(|| {
            format!(
                "copy meta-gate source {} into private snapshot",
                source_path.display()
            )
        })?;
        destination.sync_all().with_context(|| {
            format!(
                "sync private meta-gate source snapshot {}",
                snapshot_path.display()
            )
        })?;

        let snapshot_metadata = destination.metadata().with_context(|| {
            format!(
                "inspect private meta-gate source snapshot {}",
                snapshot_path.display()
            )
        })?;
        if snapshot_metadata.len() != opened_source_metadata.len() {
            bail!(
                "meta-gate source {} changed while its immutable snapshot was created",
                source_path.display()
            );
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&snapshot_path, std::fs::Permissions::from_mode(0o400))
                .with_context(|| {
                    format!(
                        "make private meta-gate source snapshot {} read-only",
                        snapshot_path.display()
                    )
                })?;
        }

        destination.seek(SeekFrom::Start(0))?;
        Ok(destination)
    }
}

impl Drop for PrivateMetaSourceSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Open and decode a trusted source snapshot. `source_root` is canonicalized
/// and the source must be a regular file below it. Any symlink component is
/// rejected. When `expected_timestamp_source` is supplied, the source parser
/// must produce exactly that representation; callers therefore cannot change
/// parsing semantics by rewriting only parquet provenance columns.
pub fn open_meta_source_snapshot(
    source_path: &Path,
    source_root: &Path,
    allow_index_timestamps: bool,
    expected_timestamp_source: Option<&str>,
) -> Result<MetaSourceSnapshot> {
    let canonical_root = canonical_path_without_symlinks(source_root, "source root")?;
    let root_metadata = std::fs::metadata(&canonical_root).map_err(|error| {
        anyhow::anyhow!("stat source root {}: {error}", canonical_root.display())
    })?;
    if !root_metadata.is_dir() {
        bail!(
            "source root {} is not a directory",
            canonical_root.display()
        );
    }

    let canonical_path = canonical_path_without_symlinks(source_path, "source")?;
    let source_metadata = std::fs::metadata(&canonical_path)
        .map_err(|error| anyhow::anyhow!("stat source {}: {error}", canonical_path.display()))?;
    if !source_metadata.is_file() {
        bail!("source {} is not a regular file", canonical_path.display());
    }
    if !canonical_path.starts_with(&canonical_root) {
        bail!(
            "source {} is outside trusted source root {}; reprepare with --source-root covering the source",
            canonical_path.display(),
            canonical_root.display()
        );
    }

    let source_snapshot = PrivateMetaSourceSnapshot::create()?;
    let mut file = source_snapshot.copy_validated_source(
        &canonical_path,
        &source_metadata,
        &canonical_root,
        &root_metadata,
    )?;
    let size_bytes = i64::try_from(file.metadata()?.len()).map_err(|_| {
        anyhow::anyhow!("source file size does not fit in signed 64-bit provenance")
    })?;
    let sha256 = sha256_reader(&file)?;
    file.seek(SeekFrom::Start(0))?;
    let allow_index_timestamps =
        allow_index_timestamps || expected_timestamp_source == Some("row-index-opt-in");
    let (bars, timestamp_source) = load_meta_bars_from_reader(file, allow_index_timestamps)?;
    if let Some(expected) = expected_timestamp_source {
        if timestamp_source != expected {
            bail!(
                "source timestamp representation changed: expected {expected:?}, decoded {timestamp_source:?}; reprepare the dataset"
            );
        }
    }

    Ok(MetaSourceSnapshot {
        canonical_path,
        canonical_root,
        size_bytes,
        sha256,
        bars,
        timestamp_source: timestamp_source.to_string(),
    })
}

fn canonical_path_without_symlinks(path: &Path, kind: &str) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut component = absolute.as_path();
    loop {
        let metadata = std::fs::symlink_metadata(component).map_err(|error| {
            anyhow::anyhow!(
                "stat {kind} path component {}: {error}",
                component.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            bail!(
                "{kind} path component {} is a symlink; reprepare from a real path",
                component.display()
            );
        }
        let Some(parent) = component.parent() else {
            break;
        };
        component = parent;
    }
    std::fs::canonicalize(&absolute)
        .map_err(|error| anyhow::anyhow!("canonicalize {kind} {}: {error}", path.display()))
}

#[cfg(unix)]
fn open_read_only_no_follow(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    // O_NOFOLLOW is 00400000 on Linux and 0100 on the BSD/Darwin family.
    #[cfg(target_os = "linux")]
    const O_NOFOLLOW_FLAG: i32 = 0o400000;
    #[cfg(not(target_os = "linux"))]
    const O_NOFOLLOW_FLAG: i32 = 0o100;
    options.custom_flags(O_NOFOLLOW_FLAG);
    options.open(path).map_err(|error| {
        anyhow::anyhow!(
            "open source snapshot {} without following symlinks: {error}",
            path.display()
        )
    })
}

#[cfg(not(unix))]
fn open_read_only_no_follow(path: &Path) -> Result<File> {
    File::open(path).map_err(|error| {
        anyhow::anyhow!(
            "open source snapshot {} without following symlinks: {error}",
            path.display()
        )
    })
}

#[cfg(unix)]
fn same_file_identity(first: &std::fs::Metadata, second: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    first.dev() == second.dev() && first.ino() == second.ino()
}

#[cfg(not(unix))]
fn same_file_identity(first: &std::fs::Metadata, second: &std::fs::Metadata) -> bool {
    first.len() == second.len() && first.modified().ok() == second.modified().ok()
}

fn sha256_reader(mut reader: &File) -> Result<String> {
    reader.seek(SeekFrom::Start(0))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn load_meta_bars_from_reader(
    file: File,
    allow_index_timestamps: bool,
) -> Result<(MetaBars, &'static str)> {
    let frame = ParquetReader::new(file).finish()?;
    let close = required_frame_f64(&frame, "close")?;
    let open = required_frame_f64(&frame, "open")?;
    let high = required_frame_f64(&frame, "high")?;
    let low = required_frame_f64(&frame, "low")?;
    let volume = optional_frame_f64(&frame, "volume")?.unwrap_or_else(|| vec![0.0; close.len()]);
    let (timestamp_ns, timestamp_source) =
        frame_timestamp_column(&frame, close.len(), allow_index_timestamps)?;
    let bars = MetaBars {
        open,
        high,
        low,
        close,
        volume,
        timestamp_ns,
    };
    bars.validate()?;
    Ok((bars, timestamp_source))
}

fn required_frame_f64(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    optional_frame_f64(frame, name)?
        .ok_or_else(|| anyhow::anyhow!("source is missing required column {name}"))
}

fn optional_frame_f64(frame: &DataFrame, name: &str) -> Result<Option<Vec<f64>>> {
    let Some(column) = frame.column(name).ok() else {
        return Ok(None);
    };
    let cast = column.as_materialized_series().cast(&DataType::Float64)?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value.ok_or_else(|| anyhow::anyhow!("source column {name} has null at row {row}"))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn frame_timestamp_column(
    frame: &DataFrame,
    len: usize,
    allow_index_timestamps: bool,
) -> Result<(Vec<i64>, &'static str)> {
    for name in ["ts_ns", "timestamp", "date"] {
        if let Some(column) = frame.column(name).ok() {
            return Ok((
                frame_timestamp_values(name, column.as_materialized_series(), name == "ts_ns")?,
                name,
            ));
        }
    }
    if allow_index_timestamps {
        return Ok((
            (0..len)
                .map(|row| {
                    i64::try_from(row)
                        .map_err(|_| anyhow::anyhow!("row index {row} does not fit in Int64"))
                })
                .collect::<Result<Vec<_>>>()?,
            "row-index-opt-in",
        ));
    }
    bail!(
        "source is missing a timestamp column (expected ts_ns, date, or timestamp); reprepare with --allow-index-timestamps only for synthetic data"
    )
}

fn frame_timestamp_values(
    name: &str,
    series: &Series,
    allow_numeric_nanoseconds: bool,
) -> Result<Vec<i64>> {
    match series.dtype() {
        DataType::Datetime(unit, _) => {
            let multiplier = match unit {
                TimeUnit::Nanoseconds => 1,
                TimeUnit::Microseconds => 1_000,
                TimeUnit::Milliseconds => 1_000_000,
            };
            let cast = series.cast(&DataType::Int64)?;
            checked_timestamp_values(name, cast.i64()?, multiplier)
        }
        DataType::Date => {
            let cast = series.cast(&DataType::Int32)?;
            cast.i32()?
                .into_iter()
                .enumerate()
                .map(|(row, value)| {
                    let days = value.ok_or_else(|| {
                        anyhow::anyhow!("source column {name} has null at row {row}")
                    })?;
                    i64::from(days)
                        .checked_mul(86_400_000_000_000)
                        .ok_or_else(|| {
                            anyhow::anyhow!("source date overflows nanoseconds at row {row}")
                        })
                })
                .collect()
        }
        dtype if allow_numeric_nanoseconds && is_integer_dtype(dtype) => {
            let cast = series.cast(&DataType::Int64)?;
            checked_timestamp_values(name, cast.i64()?, 1)
        }
        dtype if is_numeric_dtype(dtype) => {
            bail!(
                "source column {name} has ambiguous numeric dtype {dtype}; only ts_ns may be integer nanoseconds"
            )
        }
        dtype => bail!(
            "source column {name} has unsupported dtype {dtype}; expected Datetime, Date, or integer ts_ns"
        ),
    }
}

fn checked_timestamp_values(
    name: &str,
    values: &polars::prelude::ChunkedArray<polars::prelude::Int64Type>,
    multiplier: i64,
) -> Result<Vec<i64>> {
    values
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value
                .ok_or_else(|| anyhow::anyhow!("source column {name} has null at row {row}"))?
                .checked_mul(multiplier)
                .ok_or_else(|| anyhow::anyhow!("source timestamp overflows at row {row}"))
        })
        .collect()
}

fn is_integer_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn is_numeric_dtype(dtype: &DataType) -> bool {
    is_integer_dtype(dtype) || matches!(dtype, DataType::Float32 | DataType::Float64)
}

/// Convert a gate action into the target side for a raw crossover.
///
/// `skip` intentionally preserves the existing side.  It does not flatten,
/// and normal/invert never accumulate a duplicate position: the caller can
/// compare the returned target side with its current side before routing.
pub fn target_side(current_side: i8, raw_direction: i8, action: GateAction) -> i8 {
    match action {
        GateAction::Normal => raw_direction.clamp(-1, 1),
        GateAction::Invert => (-raw_direction).clamp(-1, 1),
        GateAction::Skip => current_side.clamp(-1, 1),
    }
}

/// Extract crossover events from OHLCV bars using only values available at the
/// event row.  Future prices are used only for the explicitly named diagnostic
/// outcomes, never for feature values.
pub fn extract_events(bars: &MetaBars, config: &MetaGateConfig) -> Result<Vec<MetaEvent>> {
    bars.validate()?;
    config.validate()?;

    let trigger_fast = average(config.trigger_kind, &bars.close, config.trigger_fast);
    let trigger_slow = average(config.trigger_kind, &bars.close, config.trigger_slow);
    let context_fast = average(config.context_kind, &bars.close, config.context_fast);
    let context_slow = average(config.context_kind, &bars.close, config.context_slow);
    let atr = atr_wilder(&bars.high, &bars.low, &bars.close, config.atr_period);
    let mut events = Vec::new();

    for i in 1..bars.close.len() {
        let previous_spread = trigger_fast[i - 1] - trigger_slow[i - 1];
        let spread = trigger_fast[i] - trigger_slow[i];
        if !previous_spread.is_finite()
            || !spread.is_finite()
            || !atr[i].is_finite()
            || atr[i] <= f64::EPSILON
        {
            continue;
        }

        let raw_direction = if previous_spread <= 0.0 && spread > 0.0 {
            1
        } else if previous_spread >= 0.0 && spread < 0.0 {
            -1
        } else {
            continue;
        };

        let average_warmup = average_warmup_index(config.trigger_kind, config.trigger_fast)
            .max(average_warmup_index(
                config.trigger_kind,
                config.trigger_slow,
            ))
            .max(average_warmup_index(
                config.context_kind,
                config.context_fast,
            ))
            .max(average_warmup_index(
                config.context_kind,
                config.context_slow,
            ));
        let required = average_warmup
            .saturating_add(config.slope_lookback.max(1))
            .max(config.atr_period.saturating_sub(1))
            .max(config.slope_lookback)
            .max(1);
        if i < required {
            continue;
        }

        // Every average used by the event row, its prior crossover value, and
        // its slope lookback must be genuinely warmed.  Do not turn missing
        // HMA/EMA values into zero: that would manufacture a feature and can
        // create a false crossover at the warmup boundary.
        let slope_idx = i - config.slope_lookback;
        let averages = [&trigger_fast, &trigger_slow, &context_fast, &context_slow];
        if averages.iter().any(|series| {
            !series[i].is_finite() || !series[i - 1].is_finite() || !series[slope_idx].is_finite()
        }) {
            continue;
        }

        // The event is observed after close[i].  A real decision enters at the
        // next bar's open and the fixed-horizon diagnostic exits at a later
        // open.  Drop tail events without a complete horizon rather than
        // silently shortening their labels.
        let entry_idx = i + 1;
        let future_idx = entry_idx + config.horizon_bars;
        if future_idx >= bars.close.len() {
            continue;
        }
        let entry_price = bars.open[entry_idx];
        let directional_move = raw_direction as f64 * (bars.open[future_idx] - entry_price);
        let normal_pnl = directional_move * config.contract_multiplier - config.cost_pnl;
        let invert_pnl = -directional_move * config.contract_multiplier - config.cost_pnl;
        let volume_mean = mean_finite(&bars.volume[i.saturating_sub(19)..=i]);
        let rvol_20 = if volume_mean > f64::EPSILON {
            bars.volume[i] / volume_mean
        } else {
            0.0
        };
        let efficiency_ratio = efficiency_ratio(&bars.close, i, config.atr_period);
        let oracle_action = if normal_pnl > 0.0 && normal_pnl >= invert_pnl {
            GateAction::Normal
        } else if invert_pnl > 0.0 && invert_pnl > normal_pnl {
            GateAction::Invert
        } else {
            GateAction::Skip
        };

        events.push(MetaEvent {
            event_id: events.len(),
            row_idx: i,
            timestamp_ns: bars.timestamp_ns[i],
            open: bars.open[i],
            high: bars.high[i],
            low: bars.low[i],
            close: bars.close[i],
            volume: bars.volume[i],
            raw_direction,
            trigger_fast_value: trigger_fast[i],
            trigger_slow_value: trigger_slow[i],
            context_fast_value: context_fast[i],
            context_slow_value: context_slow[i],
            atr: atr[i],
            trigger_spread_atr: spread / atr[i],
            context_spread_atr: (context_fast[i] - context_slow[i]) / atr[i],
            trigger_fast_slope_atr: slope_atr(&trigger_fast, i, config.slope_lookback, atr[i]),
            trigger_slow_slope_atr: slope_atr(&trigger_slow, i, config.slope_lookback, atr[i]),
            context_fast_slope_atr: slope_atr(&context_fast, i, config.slope_lookback, atr[i]),
            context_slow_slope_atr: slope_atr(&context_slow, i, config.slope_lookback, atr[i]),
            efficiency_ratio,
            rvol_20,
            return_1: normalized_return(&bars.close, i, 1, atr[i]),
            return_5: normalized_return(&bars.close, i, 5, atr[i]),
            entry_price,
            normal_pnl,
            invert_pnl,
            skip_pnl: 0.0,
            oracle_action,
            entry_row_idx: entry_idx,
            horizon_row_idx: future_idx,
        });
    }

    Ok(events)
}

/// Decode the complete canonical event payload from a prepared parquet frame.
/// Training and evaluation use this rather than reading only the feature or
/// reward columns, so regeneration can detect a payload whose metadata and
/// self-reported hashes were rewritten together.
pub fn meta_events_from_frame(frame: &DataFrame) -> Result<Vec<MetaEvent>> {
    let event_id = frame_usize(frame, "event_id")?;
    let row_idx = frame_usize(frame, "row_idx")?;
    let timestamp_ns = frame_i64(frame, "timestamp_ns")?;
    let open = frame_f64(frame, "open")?;
    let high = frame_f64(frame, "high")?;
    let low = frame_f64(frame, "low")?;
    let close = frame_f64(frame, "close")?;
    let volume = frame_f64(frame, "volume")?;
    let raw_direction = frame_i64(frame, "raw_direction")?;
    let trigger_fast_value = frame_f64(frame, "trigger_fast_value")?;
    let trigger_slow_value = frame_f64(frame, "trigger_slow_value")?;
    let context_fast_value = frame_f64(frame, "context_fast_value")?;
    let context_slow_value = frame_f64(frame, "context_slow_value")?;
    let atr = frame_f64(frame, "atr")?;
    let trigger_spread_atr = frame_f64(frame, "trigger_spread_atr")?;
    let context_spread_atr = frame_f64(frame, "context_spread_atr")?;
    let trigger_fast_slope_atr = frame_f64(frame, "trigger_fast_slope_atr")?;
    let trigger_slow_slope_atr = frame_f64(frame, "trigger_slow_slope_atr")?;
    let context_fast_slope_atr = frame_f64(frame, "context_fast_slope_atr")?;
    let context_slow_slope_atr = frame_f64(frame, "context_slow_slope_atr")?;
    let efficiency_ratio = frame_f64(frame, "efficiency_ratio")?;
    let rvol_20 = frame_f64(frame, "rvol_20")?;
    let return_1 = frame_f64(frame, "return_1")?;
    let return_5 = frame_f64(frame, "return_5")?;
    let entry_price = frame_f64(frame, "entry_price")?;
    let normal_pnl = frame_f64(frame, "normal_pnl")?;
    let invert_pnl = frame_f64(frame, "invert_pnl")?;
    let skip_pnl = frame_f64(frame, "skip_pnl")?;
    let oracle_action = frame_gate_actions(frame, "oracle_action")?;
    let entry_row_idx = frame_usize(frame, "entry_row_idx")?;
    let horizon_row_idx = frame_usize(frame, "horizon_row_idx")?;

    let columns = [
        event_id.len(),
        row_idx.len(),
        timestamp_ns.len(),
        open.len(),
        high.len(),
        low.len(),
        close.len(),
        volume.len(),
        raw_direction.len(),
        trigger_fast_value.len(),
        trigger_slow_value.len(),
        context_fast_value.len(),
        context_slow_value.len(),
        atr.len(),
        trigger_spread_atr.len(),
        context_spread_atr.len(),
        trigger_fast_slope_atr.len(),
        trigger_slow_slope_atr.len(),
        context_fast_slope_atr.len(),
        context_slow_slope_atr.len(),
        efficiency_ratio.len(),
        rvol_20.len(),
        return_1.len(),
        return_5.len(),
        entry_price.len(),
        normal_pnl.len(),
        invert_pnl.len(),
        skip_pnl.len(),
        oracle_action.len(),
        entry_row_idx.len(),
        horizon_row_idx.len(),
    ];
    let row_count = frame.height();
    if columns.iter().any(|length| *length != row_count) {
        bail!("event payload columns do not have a common row count");
    }

    let mut events = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let raw_direction = i8::try_from(raw_direction[row]).map_err(|_| {
            anyhow::anyhow!("raw_direction is outside the i8 range at event row {row}")
        })?;
        if !matches!(raw_direction, -1 | 1) {
            bail!("raw_direction must be -1 or 1 at event row {row}");
        }
        events.push(MetaEvent {
            event_id: event_id[row],
            row_idx: row_idx[row],
            timestamp_ns: timestamp_ns[row],
            open: open[row],
            high: high[row],
            low: low[row],
            close: close[row],
            volume: volume[row],
            raw_direction,
            trigger_fast_value: trigger_fast_value[row],
            trigger_slow_value: trigger_slow_value[row],
            context_fast_value: context_fast_value[row],
            context_slow_value: context_slow_value[row],
            atr: atr[row],
            trigger_spread_atr: trigger_spread_atr[row],
            context_spread_atr: context_spread_atr[row],
            trigger_fast_slope_atr: trigger_fast_slope_atr[row],
            trigger_slow_slope_atr: trigger_slow_slope_atr[row],
            context_fast_slope_atr: context_fast_slope_atr[row],
            context_slow_slope_atr: context_slow_slope_atr[row],
            efficiency_ratio: efficiency_ratio[row],
            rvol_20: rvol_20[row],
            return_1: return_1[row],
            return_5: return_5[row],
            entry_price: entry_price[row],
            normal_pnl: normal_pnl[row],
            invert_pnl: invert_pnl[row],
            skip_pnl: skip_pnl[row],
            oracle_action: oracle_action[row],
            entry_row_idx: entry_row_idx[row],
            horizon_row_idx: horizon_row_idx[row],
        });
    }
    Ok(events)
}

/// Serialize only the canonical event payload columns. This is shared by
/// integrity tests and keeps their fixtures aligned with the production
/// decoder.
pub fn meta_events_to_frame(events: &[MetaEvent]) -> Result<DataFrame> {
    let i64_values = |f: fn(&MetaEvent) -> i64| events.iter().map(f).collect::<Vec<_>>();
    let f64_values = |f: fn(&MetaEvent) -> f64| events.iter().map(f).collect::<Vec<_>>();
    let actions = events
        .iter()
        .map(|event| event.oracle_action.to_string())
        .collect::<Vec<_>>();
    DataFrame::new(vec![
        Series::new("event_id".into(), i64_values(|event| event.event_id as i64)).into(),
        Series::new("row_idx".into(), i64_values(|event| event.row_idx as i64)).into(),
        Series::new(
            "timestamp_ns".into(),
            i64_values(|event| event.timestamp_ns),
        )
        .into(),
        Series::new("open".into(), f64_values(|event| event.open)).into(),
        Series::new("high".into(), f64_values(|event| event.high)).into(),
        Series::new("low".into(), f64_values(|event| event.low)).into(),
        Series::new("close".into(), f64_values(|event| event.close)).into(),
        Series::new("volume".into(), f64_values(|event| event.volume)).into(),
        Series::new(
            "raw_direction".into(),
            events
                .iter()
                .map(|event| event.raw_direction as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_fast_value".into(),
            f64_values(|event| event.trigger_fast_value),
        )
        .into(),
        Series::new(
            "trigger_slow_value".into(),
            f64_values(|event| event.trigger_slow_value),
        )
        .into(),
        Series::new(
            "context_fast_value".into(),
            f64_values(|event| event.context_fast_value),
        )
        .into(),
        Series::new(
            "context_slow_value".into(),
            f64_values(|event| event.context_slow_value),
        )
        .into(),
        Series::new("atr".into(), f64_values(|event| event.atr)).into(),
        Series::new(
            "trigger_spread_atr".into(),
            f64_values(|event| event.trigger_spread_atr),
        )
        .into(),
        Series::new(
            "context_spread_atr".into(),
            f64_values(|event| event.context_spread_atr),
        )
        .into(),
        Series::new(
            "trigger_fast_slope_atr".into(),
            f64_values(|event| event.trigger_fast_slope_atr),
        )
        .into(),
        Series::new(
            "trigger_slow_slope_atr".into(),
            f64_values(|event| event.trigger_slow_slope_atr),
        )
        .into(),
        Series::new(
            "context_fast_slope_atr".into(),
            f64_values(|event| event.context_fast_slope_atr),
        )
        .into(),
        Series::new(
            "context_slow_slope_atr".into(),
            f64_values(|event| event.context_slow_slope_atr),
        )
        .into(),
        Series::new(
            "efficiency_ratio".into(),
            f64_values(|event| event.efficiency_ratio),
        )
        .into(),
        Series::new("rvol_20".into(), f64_values(|event| event.rvol_20)).into(),
        Series::new("return_1".into(), f64_values(|event| event.return_1)).into(),
        Series::new("return_5".into(), f64_values(|event| event.return_5)).into(),
        Series::new("entry_price".into(), f64_values(|event| event.entry_price)).into(),
        Series::new("normal_pnl".into(), f64_values(|event| event.normal_pnl)).into(),
        Series::new("invert_pnl".into(), f64_values(|event| event.invert_pnl)).into(),
        Series::new("skip_pnl".into(), f64_values(|event| event.skip_pnl)).into(),
        Series::new("oracle_action".into(), actions).into(),
        Series::new(
            "entry_row_idx".into(),
            i64_values(|event| event.entry_row_idx as i64),
        )
        .into(),
        Series::new(
            "horizon_row_idx".into(),
            i64_values(|event| event.horizon_row_idx as i64),
        )
        .into(),
    ])
    .map_err(Into::into)
}

fn frame_f64(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let column = frame
        .column(name)
        .map_err(|_| anyhow::anyhow!("event parquet is missing required column {name}"))?;
    let cast = column.as_materialized_series().cast(&DataType::Float64)?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {row}"))?;
            if !value.is_finite() {
                bail!("column {name} has non-finite value at row {row}");
            }
            Ok(value)
        })
        .collect()
}

fn frame_i64(frame: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = frame
        .column(name)
        .map_err(|_| anyhow::anyhow!("event parquet is missing required column {name}"))?;
    let cast = column.as_materialized_series().cast(&DataType::Int64)?;
    cast.i64()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {row}"))
        })
        .collect()
}

fn frame_usize(frame: &DataFrame, name: &str) -> Result<Vec<usize>> {
    frame_i64(frame, name)?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            usize::try_from(value)
                .map_err(|_| anyhow::anyhow!("column {name} has negative value at row {row}"))
        })
        .collect()
}

fn frame_gate_actions(frame: &DataFrame, name: &str) -> Result<Vec<GateAction>> {
    let column = frame
        .column(name)
        .map_err(|_| anyhow::anyhow!("event parquet is missing required column {name}"))?;
    let cast = column.as_materialized_series().cast(&DataType::String)?;
    cast.str()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {row}"))?;
            match value {
                "normal" => Ok(GateAction::Normal),
                "skip" => Ok(GateAction::Skip),
                "invert" => Ok(GateAction::Invert),
                _ => bail!("column {name} has invalid gate action at row {row}: {value:?}"),
            }
        })
        .collect()
}

fn average(kind: AverageKind, prices: &[f64], period: usize) -> Vec<f64> {
    match kind {
        AverageKind::Ema => crate::features::ema(prices, period),
        AverageKind::Hma => crate::features::hma(prices, period),
    }
}

/// Return the first index at which the selected average implementation can
/// produce a finite value.  HMA is a WMA over a WMA-derived series, so its
/// warmup is longer than the nominal period.
fn average_warmup_index(kind: AverageKind, period: usize) -> usize {
    match kind {
        AverageKind::Ema => period.saturating_sub(1),
        AverageKind::Hma if period >= 2 => {
            let sqrt_period = (period as f64).sqrt().round() as usize;
            period.saturating_add(sqrt_period).saturating_sub(2)
        }
        AverageKind::Hma => usize::MAX,
    }
}

fn mean_finite(values: &[f64]) -> f64 {
    let (sum, count) = values.iter().fold((0.0, 0usize), |(sum, count), value| {
        if value.is_finite() {
            (sum + *value, count + 1)
        } else {
            (sum, count)
        }
    });
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn slope_atr(values: &[f64], idx: usize, lookback: usize, atr: f64) -> f64 {
    (values[idx] - values[idx - lookback]) / lookback as f64 / atr
}

fn normalized_return(close: &[f64], idx: usize, lookback: usize, atr: f64) -> f64 {
    (close[idx] - close[idx - lookback]) / atr
}

fn efficiency_ratio(close: &[f64], idx: usize, lookback: usize) -> f64 {
    if lookback == 0 || idx < lookback {
        return 0.0;
    }
    let change = (close[idx] - close[idx - lookback]).abs();
    let volatility: f64 = (idx - lookback + 1..=idx)
        .map(|j| (close[j] - close[j - 1]).abs())
        .sum();
    if volatility > f64::EPSILON {
        (change / volatility).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 || close.len() < period || high.len() != close.len() || low.len() != close.len()
    {
        return out;
    }
    let mut tr = vec![0.0; close.len()];
    tr[0] = high[0] - low[0];
    for i in 1..close.len() {
        tr[i] = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
    }
    let mut value = tr[..period].iter().sum::<f64>() / period as f64;
    out[period - 1] = value;
    for i in period..close.len() {
        value = (value * (period as f64 - 1.0) + tr[i]) / period as f64;
        out[i] = value;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(close: &[f64]) -> MetaBars {
        MetaBars {
            open: close.to_vec(),
            high: close.iter().map(|v| v + 0.5).collect(),
            low: close.iter().map(|v| v - 0.5).collect(),
            close: close.to_vec(),
            volume: vec![1.0; close.len()],
            timestamp_ns: (0..close.len()).map(|v| v as i64).collect(),
        }
    }

    #[test]
    fn skip_preserves_side_and_invert_reverses_raw_direction() {
        assert_eq!(target_side(1, -1, GateAction::Skip), 1);
        assert_eq!(target_side(0, 1, GateAction::Normal), 1);
        assert_eq!(target_side(1, -1, GateAction::Normal), -1);
        assert_eq!(target_side(0, -1, GateAction::Invert), 1);
    }

    #[test]
    fn event_features_are_causal() {
        let config = MetaGateConfig {
            trigger_fast: 2,
            trigger_slow: 4,
            context_fast: 5,
            context_slow: 7,
            atr_period: 3,
            slope_lookback: 2,
            horizon_bars: 2,
            ..MetaGateConfig::default()
        };
        let prefix = vec![
            100.0, 99.0, 98.0, 98.5, 100.0, 102.0, 101.0, 99.0, 98.0, 99.0, 101.0, 103.0, 102.0,
            100.0, 99.0, 100.0,
        ];
        let mut extended = prefix.clone();
        extended.extend([140.0, 141.0, 142.0, 143.0]);
        let first = extract_events(&bars(&prefix), &config).unwrap();
        let second = extract_events(&bars(&extended), &config)
            .unwrap()
            .into_iter()
            .filter(|event| event.entry_row_idx + config.horizon_bars < prefix.len())
            .collect::<Vec<_>>();
        assert!(!first.is_empty());
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.row_idx, b.row_idx);
            assert_eq!(a.raw_direction, b.raw_direction);
            assert_eq!(a.trigger_spread_atr, b.trigger_spread_atr);
            assert_eq!(a.context_spread_atr, b.context_spread_atr);
        }
    }

    #[test]
    fn hma_warmup_does_not_emit_values_before_nested_wma_is_ready() {
        let prices = (0..64).map(|value| value as f64).collect::<Vec<_>>();
        let period = 10;
        let values = average(AverageKind::Hma, &prices, period);
        let warmup = average_warmup_index(AverageKind::Hma, period);

        assert!(values[..warmup].iter().all(|value| !value.is_finite()));
        assert!(values[warmup].is_finite());
    }

    #[test]
    fn canonical_meta_dataset_hash_is_stable_and_binds_event_payload() {
        let config = MetaGateConfig {
            trigger_fast: 2,
            trigger_slow: 4,
            context_fast: 5,
            context_slow: 7,
            atr_period: 3,
            slope_lookback: 2,
            horizon_bars: 2,
            ..MetaGateConfig::default()
        };
        let events = extract_events(
            &bars(&[
                100.0, 99.0, 98.0, 98.5, 100.0, 102.0, 101.0, 99.0, 98.0, 99.0, 101.0, 103.0,
                102.0, 100.0, 99.0, 100.0,
            ]),
            &config,
        )
        .unwrap();
        assert!(!events.is_empty());

        let original = canonical_event_payload_sha256(&events);
        assert_eq!(original, canonical_event_payload_sha256(&events));

        let mut modified = events.clone();
        modified[0].normal_pnl += 0.01;
        assert_ne!(original, canonical_event_payload_sha256(&modified));

        let dataset = canonical_dataset_fingerprint_sha256(
            &"source-hash"[..],
            &original,
            &config.feature_schema(),
            "GC",
            "GCZ6",
            &serde_json::to_string(&config).unwrap(),
            "/tmp/source.parquet",
            "/tmp",
            123,
            100,
            "ts_ns",
        );
        assert_ne!(
            dataset,
            canonical_dataset_fingerprint_sha256(
                "different-source-hash",
                &original,
                &config.feature_schema(),
                "GC",
                "GCZ6",
                &serde_json::to_string(&config).unwrap(),
                "/tmp/source.parquet",
                "/tmp",
                123,
                100,
                "ts_ns",
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn source_snapshot_rejects_regular_file_replacement_after_validation() {
        let directory = std::env::temp_dir().join(format!(
            "midas-meta-source-race-{}-{}",
            std::process::id(),
            META_SOURCE_SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let source = directory.join("source.parquet");
        let replacement = directory.join("replacement.parquet");
        std::fs::write(&source, b"validated source bytes").unwrap();

        // These are the identities captured by the validation phase.
        let validated_source_metadata = std::fs::metadata(&source).unwrap();
        let validated_root_metadata = std::fs::metadata(&directory).unwrap();

        // Replace the regular file with another regular file.  O_NOFOLLOW
        // alone cannot distinguish this case; the inode comparison must.
        std::fs::write(&replacement, b"replacement source bytes").unwrap();
        std::fs::rename(&replacement, &source).unwrap();

        let snapshot = PrivateMetaSourceSnapshot::create().unwrap();
        let error = snapshot
            .copy_validated_source(
                &source,
                &validated_source_metadata,
                &directory,
                &validated_root_metadata,
            )
            .unwrap_err();
        assert!(error.to_string().contains("was replaced"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
