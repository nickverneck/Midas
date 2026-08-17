use anyhow::{Context, Result, bail};
use clap::Parser;
use midas_env::meta_gate::{
    META_EVENT_SCHEMA, MetaBars, MetaGateConfig, canonical_dataset_fingerprint_sha256,
    canonical_event_payload_sha256, extract_events, open_meta_source_snapshot,
};
use polars::prelude::{
    CsvWriter, DataFrame, DataType, NamedFrom, ParquetReader, ParquetWriter, SerReader, SerWriter,
    Series, TimeUnit,
};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const NANOS_PER_DAY: i64 = 86_400_000_000_000;

#[derive(Debug, Parser)]
#[command(about = "Create causal crossover-event rows for a normal/skip/invert meta-gate")]
struct Args {
    /// Input server-bar parquet file.
    #[arg(long)]
    input: PathBuf,
    /// Output parquet path (CSV is used when the extension is .csv).
    #[arg(long)]
    output: PathBuf,
    /// Instrument label stored in every event row.
    #[arg(long, default_value = "UNKNOWN")]
    instrument: String,
    /// Contract label stored in every event row.
    #[arg(long, default_value = "UNKNOWN")]
    contract: String,
    /// Permit a missing timestamp column by using the input row index.
    /// This is for synthetic/diagnostic data only and is labeled in output.
    #[arg(long)]
    allow_index_timestamps: bool,
    /// Trusted directory containing the source file. Defaults to the source's
    /// canonical parent directory.
    #[arg(long, value_name = "DIR")]
    source_root: Option<PathBuf>,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Debug, clap::Args)]
struct ConfigArgs {
    #[arg(long, value_enum, default_value_t = midas_env::meta_gate::AverageKind::Ema)]
    trigger_kind: midas_env::meta_gate::AverageKind,
    #[arg(long, default_value_t = 10)]
    trigger_fast: usize,
    #[arg(long, default_value_t = 30)]
    trigger_slow: usize,
    #[arg(long, value_enum, default_value_t = midas_env::meta_gate::AverageKind::Ema)]
    context_kind: midas_env::meta_gate::AverageKind,
    #[arg(long, default_value_t = 210)]
    context_fast: usize,
    #[arg(long, default_value_t = 240)]
    context_slow: usize,
    #[arg(long, default_value_t = 14)]
    atr_period: usize,
    #[arg(long, default_value_t = 5)]
    slope_lookback: usize,
    #[arg(long, default_value_t = 30)]
    horizon_bars: usize,
    /// Price units converted into account PnL by the contract multiplier.
    #[arg(long, default_value_t = 1.0)]
    contract_multiplier: f64,
    /// Total round-trip cost in account PnL units.
    #[arg(long = "cost-pnl", default_value_t = 0.0)]
    cost_pnl: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    refuse_input_output_alias(&args.input, &args.output)?;
    let config = MetaGateConfig {
        trigger_kind: args.config.trigger_kind,
        trigger_fast: args.config.trigger_fast,
        trigger_slow: args.config.trigger_slow,
        context_kind: args.config.context_kind,
        context_fast: args.config.context_fast,
        context_slow: args.config.context_slow,
        atr_period: args.config.atr_period,
        slope_lookback: args.config.slope_lookback,
        horizon_bars: args.config.horizon_bars,
        contract_multiplier: args.config.contract_multiplier,
        cost_pnl: args.config.cost_pnl,
    };
    config.validate()?;
    if args.instrument.trim().is_empty() {
        bail!("--instrument must be non-empty for event provenance");
    }
    if args.contract.trim().is_empty() {
        bail!("--contract must be non-empty for event provenance");
    }
    let source_root_hint = args
        .source_root
        .clone()
        .or_else(|| args.input.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let snapshot = open_meta_source_snapshot(
        &args.input,
        &source_root_hint,
        args.allow_index_timestamps,
        None,
    )
    .with_context(|| {
        format!(
            "open trusted source snapshot for {}; reprepare with a real source path and trusted --source-root",
            args.input.display()
        )
    })?;
    let source_size_bytes = snapshot.size_bytes;
    let source_sha256 = snapshot.sha256.clone();
    let bars = snapshot.bars;
    let timestamp_source = snapshot.timestamp_source.as_str();
    let source_row_count = i64::try_from(bars.close.len())
        .context("input row count does not fit in signed 64-bit provenance column")?;
    let events = extract_events(&bars, &config)?;
    if events.is_empty() {
        bail!(
            "no crossover events found after warmup; input has {} bars and requires at least {}",
            bars.close.len(),
            config
                .context_slow
                .max(config.trigger_slow)
                .max(config.atr_period)
                .max(config.slope_lookback)
        );
    }
    let event_payload_sha256 = canonical_event_payload_sha256(&events);
    let dataset_fingerprint_sha256 = canonical_dataset_fingerprint_sha256(
        &source_sha256,
        &event_payload_sha256,
        &config.feature_schema(),
        &args.instrument,
        &args.contract,
        &serde_json::to_string(&config).context("serialize meta-gate config")?,
        &snapshot.canonical_path.display().to_string(),
        &snapshot.canonical_root.display().to_string(),
        source_size_bytes,
        source_row_count,
        &timestamp_source,
    );
    write_events(
        &args.output,
        &args.instrument,
        &args.contract,
        &config,
        &events,
        &snapshot.canonical_path,
        &snapshot.canonical_root,
        source_size_bytes,
        source_row_count,
        &timestamp_source,
        &source_sha256,
        &event_payload_sha256,
        &dataset_fingerprint_sha256,
    )?;
    println!(
        "wrote {} events to {} schema={} trigger={} {}/{} context={} {}/{} horizon={} timestamp_source={}",
        events.len(),
        args.output.display(),
        META_EVENT_SCHEMA,
        config.trigger_kind,
        config.trigger_fast,
        config.trigger_slow,
        config.context_kind,
        config.context_fast,
        config.context_slow,
        config.horizon_bars,
        &timestamp_source,
    );
    Ok(())
}

fn refuse_input_output_alias(input: &Path, output: &Path) -> Result<()> {
    let input_metadata =
        std::fs::metadata(input).with_context(|| format!("stat input {}", input.display()))?;
    let input = std::fs::canonicalize(input)
        .with_context(|| format!("canonicalize input {}", input.display()))?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create output directory {}", parent.display()))?;
    let output_name = output
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output path has no file name"))?;

    // Inspect the directory entry rather than following the path. This also
    // rejects dangling symlinks, which `exists()` would miss.
    let output_link_metadata = std::fs::symlink_metadata(output);
    if let Ok(metadata) = &output_link_metadata {
        if metadata.file_type().is_symlink() {
            bail!(
                "output {} is a symlink; refusing to follow or replace it",
                output.display()
            );
        }
        let output = std::fs::canonicalize(output)
            .with_context(|| format!("canonicalize output {}", output.display()))?;
        if input == output || same_file_identity(&input_metadata, metadata) {
            bail!(
                "output must be different from input; refusing to overwrite source data or an alias"
            );
        }
        bail!(
            "output {} already exists; refusing to clobber an existing file",
            output.display()
        );
    }
    if output_link_metadata
        .as_ref()
        .is_err_and(|error| error.kind() != std::io::ErrorKind::NotFound)
    {
        return Err(output_link_metadata.unwrap_err())
            .with_context(|| format!("inspect output path {}", output.display()));
    }

    let canonical_parent = std::fs::canonicalize(parent)
        .with_context(|| format!("canonicalize output directory {}", parent.display()))?;
    let output = canonical_parent.join(output_name);
    if input == output {
        bail!("output must be different from input; refusing to overwrite source data");
    }
    Ok(())
}

fn same_file_identity(input: &std::fs::Metadata, output: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        return input.dev() == output.dev() && input.ino() == output.ino();
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        return input.volume_serial_number() == output.volume_serial_number()
            && input.file_index() == output.file_index();
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (input, output);
        false
    }
}

fn load_bars(path: &Path, allow_index_timestamps: bool) -> Result<(MetaBars, &'static str)> {
    let file = File::open(path).with_context(|| format!("open input {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;
    let close = required_f64(&df, "close")?;
    let open = required_f64(&df, "open")?;
    let high = required_f64(&df, "high")?;
    let low = required_f64(&df, "low")?;
    let volume = optional_f64(&df, "volume")?.unwrap_or_else(|| vec![0.0; close.len()]);
    let (timestamp_ns, timestamp_source) =
        timestamp_column(&df, close.len(), allow_index_timestamps)?;
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

fn required_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    optional_f64(df, name)?
        .ok_or_else(|| anyhow::anyhow!("input is missing required column {name}"))
}

fn optional_f64(df: &DataFrame, name: &str) -> Result<Option<Vec<f64>>> {
    let Some(column) = df.column(name).ok() else {
        return Ok(None);
    };
    let cast = column.as_materialized_series().cast(&DataType::Float64)?;
    let values = cast
        .f64()?
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {idx}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(values))
}

fn timestamp_column(
    df: &DataFrame,
    len: usize,
    allow_index_timestamps: bool,
) -> Result<(Vec<i64>, &'static str)> {
    for name in ["ts_ns", "timestamp", "date"] {
        if let Some(column) = df.column(name).ok() {
            let series = column.as_materialized_series();
            let values = timestamp_values(name, series, name == "ts_ns")?;
            return Ok((values, name));
        }
    }
    if allow_index_timestamps {
        let values = (0..len)
            .map(|idx| {
                i64::try_from(idx)
                    .map_err(|_| anyhow::anyhow!("row index {idx} does not fit in Int64"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((values, "row-index-opt-in"))
    } else {
        bail!(
            "input is missing a timestamp column (expected ts_ns, date, or timestamp); pass --allow-index-timestamps only for synthetic/diagnostic data"
        )
    }
}

fn timestamp_values(
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
            checked_scaled_values(name, cast.i64()?, multiplier)
        }
        DataType::Date => {
            let cast = series.cast(&DataType::Int32)?;
            cast.i32()?
                .into_iter()
                .enumerate()
                .map(|(idx, value)| {
                    let days = value
                        .ok_or_else(|| anyhow::anyhow!("column {name} has null at row {idx}"))?;
                    i64::from(days)
                        .checked_mul(NANOS_PER_DAY)
                        .ok_or_else(|| anyhow::anyhow!("date overflow at row {idx}"))
                })
                .collect()
        }
        dtype if allow_numeric_nanoseconds && is_integer_dtype(dtype) => {
            let cast = series.cast(&DataType::Int64)?;
            checked_scaled_values(name, cast.i64()?, 1)
        }
        dtype if is_numeric_dtype(dtype) => {
            bail!(
                "column {name} has ambiguous numeric dtype {dtype}; only ts_ns may use integer nanoseconds, while timestamp/date must be Datetime or Date"
            )
        }
        dtype => {
            bail!(
                "column {name} has unsupported dtype {dtype}; expected Datetime, Date, or integer ts_ns"
            )
        }
    }
}

fn checked_scaled_values(
    name: &str,
    values: &polars::prelude::ChunkedArray<polars::prelude::Int64Type>,
    multiplier: i64,
) -> Result<Vec<i64>> {
    values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {idx}"))?;
            value
                .checked_mul(multiplier)
                .ok_or_else(|| anyhow::anyhow!("timestamp overflow at row {idx}"))
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

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("open source {} for integrity hashing", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read source {} for integrity hashing", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_events(
    output: &Path,
    instrument: &str,
    contract: &str,
    config: &MetaGateConfig,
    events: &[midas_env::meta_gate::MetaEvent],
    source_path: &Path,
    source_root: &Path,
    source_size_bytes: i64,
    source_row_count: i64,
    timestamp_source: &str,
    source_sha256: &str,
    event_payload_sha256: &str,
    dataset_fingerprint_sha256: &str,
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let strings = |value: &str| vec![value.to_string(); events.len()];
    let config_json = serde_json::to_string(config).context("serialize meta-gate config")?;
    let mut df = DataFrame::new(vec![
        Series::new("schema_version".into(), strings(META_EVENT_SCHEMA)).into(),
        Series::new("feature_schema".into(), strings(&config.feature_schema())).into(),
        Series::new("config_json".into(), strings(&config_json)).into(),
        Series::new("instrument".into(), strings(instrument)).into(),
        Series::new("contract".into(), strings(contract)).into(),
        Series::new(
            "source_path".into(),
            strings(&source_path.display().to_string()),
        )
        .into(),
        Series::new(
            "source_root".into(),
            strings(&source_root.display().to_string()),
        )
        .into(),
        Series::new(
            "source_size_bytes".into(),
            vec![source_size_bytes; events.len()],
        )
        .into(),
        Series::new(
            "source_row_count".into(),
            vec![source_row_count; events.len()],
        )
        .into(),
        Series::new("timestamp_source".into(), strings(timestamp_source)).into(),
        Series::new("source_sha256".into(), strings(source_sha256)).into(),
        Series::new("event_payload_sha256".into(), strings(event_payload_sha256)).into(),
        Series::new(
            "dataset_fingerprint_sha256".into(),
            strings(dataset_fingerprint_sha256),
        )
        .into(),
        Series::new(
            "event_id".into(),
            events.iter().map(|e| e.event_id as i64).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "row_idx".into(),
            events.iter().map(|e| e.row_idx as i64).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "timestamp_ns".into(),
            events.iter().map(|e| e.timestamp_ns).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "open".into(),
            events.iter().map(|e| e.open).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "high".into(),
            events.iter().map(|e| e.high).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "low".into(),
            events.iter().map(|e| e.low).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "close".into(),
            events.iter().map(|e| e.close).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "volume".into(),
            events.iter().map(|e| e.volume).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "raw_direction".into(),
            events
                .iter()
                .map(|e| e.raw_direction as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_fast_value".into(),
            events
                .iter()
                .map(|e| e.trigger_fast_value)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_slow_value".into(),
            events
                .iter()
                .map(|e| e.trigger_slow_value)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "context_fast_value".into(),
            events
                .iter()
                .map(|e| e.context_fast_value)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "context_slow_value".into(),
            events
                .iter()
                .map(|e| e.context_slow_value)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "atr".into(),
            events.iter().map(|e| e.atr).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_spread_atr".into(),
            events
                .iter()
                .map(|e| e.trigger_spread_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "context_spread_atr".into(),
            events
                .iter()
                .map(|e| e.context_spread_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_fast_slope_atr".into(),
            events
                .iter()
                .map(|e| e.trigger_fast_slope_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "trigger_slow_slope_atr".into(),
            events
                .iter()
                .map(|e| e.trigger_slow_slope_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "context_fast_slope_atr".into(),
            events
                .iter()
                .map(|e| e.context_fast_slope_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "context_slow_slope_atr".into(),
            events
                .iter()
                .map(|e| e.context_slow_slope_atr)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "efficiency_ratio".into(),
            events
                .iter()
                .map(|e| e.efficiency_ratio)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "rvol_20".into(),
            events.iter().map(|e| e.rvol_20).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "return_1".into(),
            events.iter().map(|e| e.return_1).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "entry_price".into(),
            events.iter().map(|e| e.entry_price).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "return_5".into(),
            events.iter().map(|e| e.return_5).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "normal_pnl".into(),
            events.iter().map(|e| e.normal_pnl).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "invert_pnl".into(),
            events.iter().map(|e| e.invert_pnl).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "skip_pnl".into(),
            events.iter().map(|e| e.skip_pnl).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "oracle_action".into(),
            events
                .iter()
                .map(|e| e.oracle_action.to_string())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "entry_row_idx".into(),
            events
                .iter()
                .map(|e| e.entry_row_idx as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "horizon_row_idx".into(),
            events
                .iter()
                .map(|e| e.horizon_row_idx as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
    ])?;
    let (temporary_path, temporary_file) = create_unique_output_file(output)?;
    let mut temporary_output = TemporaryOutput {
        path: temporary_path,
        committed: false,
    };
    if output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
    {
        CsvWriter::new(temporary_file)
            .include_header(true)
            .finish(&mut df)?;
    } else {
        ParquetWriter::new(temporary_file).finish(&mut df)?;
    }
    publish_noreplace(&temporary_output.path, output)
        .with_context(|| format!("atomically commit meta dataset {}", output.display()))?;
    temporary_output.committed = true;
    Ok(())
}

static TEMPORARY_OUTPUT_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TemporaryOutput {
    path: PathBuf,
    committed: bool,
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn create_unique_output_file(output: &Path) -> Result<(PathBuf, File)> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    let name = output
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output path has no file name"))?
        .to_string_lossy();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for _ in 0..128 {
        let counter = TEMPORARY_OUTPUT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{name}.tmp-{}-{timestamp}-{counter}",
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create temporary meta dataset {}", candidate.display())
                });
            }
        }
    }
    bail!(
        "could not allocate a unique temporary meta dataset beside {}",
        output.display()
    )
}

fn publish_noreplace(temporary: &Path, destination: &Path) -> Result<()> {
    // `rename` replaces an existing destination on Unix. A same-directory
    // hard link is the portable no-replace primitive: publication is atomic,
    // and any existing directory entry (including a symlink) makes it fail.
    match std::fs::hard_link(temporary, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // Keep this helper safe when used without TemporaryOutput as
            // well. The normal writer path also owns the temp file and will
            // harmlessly observe it already gone in Drop.
            let _ = std::fs::remove_file(temporary);
            bail!(
                "output {} appeared while preparing; refusing to clobber it",
                destination.display()
            );
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "publish temporary dataset {} as {}",
                    temporary.display(),
                    destination.display()
                )
            });
        }
    }
    std::fs::remove_file(temporary)
        .with_context(|| format!("remove temporary dataset {}", temporary.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "midas-meta-dataset-{name}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn timestamp_priority_prefers_ts_ns() {
        let frame = DataFrame::new(vec![
            Series::new("ts_ns".into(), &[1_000_i64, 2_000_i64]).into(),
            Series::new("timestamp".into(), &[3_000_i64, 4_000_i64]).into(),
            Series::new("date".into(), &[5_i32, 6_i32]).into(),
        ])
        .unwrap();

        let (values, source) = timestamp_column(&frame, 2, false).unwrap();
        assert_eq!(values, vec![1_000, 2_000]);
        assert_eq!(source, "ts_ns");
    }

    #[test]
    fn datetime_units_and_date_are_converted_to_nanoseconds() {
        let datetime = Series::new("timestamp".into(), &[1_i64, 2_i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();
        let frame = DataFrame::new(vec![datetime.into()]).unwrap();
        let (values, source) = timestamp_column(&frame, 2, false).unwrap();
        assert_eq!(values, vec![1_000_000, 2_000_000]);
        assert_eq!(source, "timestamp");

        let dates = Series::new("date".into(), &[0_i32, 1_i32])
            .cast(&DataType::Date)
            .unwrap();
        let frame = DataFrame::new(vec![dates.into()]).unwrap();
        let (values, source) = timestamp_column(&frame, 2, false).unwrap();
        assert_eq!(values, vec![0, NANOS_PER_DAY]);
        assert_eq!(source, "date");
    }

    #[test]
    fn ambiguous_numeric_timestamp_and_date_are_rejected() {
        let timestamp = DataFrame::new(vec![
            Series::new("timestamp".into(), &[1_i64, 2_i64]).into(),
        ])
        .unwrap();
        let error = timestamp_column(&timestamp, 2, false).unwrap_err();
        assert!(error.to_string().contains("ambiguous numeric"));

        let date =
            DataFrame::new(vec![Series::new("date".into(), &[1_i64, 2_i64]).into()]).unwrap();
        let error = timestamp_column(&date, 2, false).unwrap_err();
        assert!(error.to_string().contains("ambiguous numeric"));
    }

    #[test]
    fn no_replace_publication_preserves_existing_destination_and_cleans_temp() {
        let directory = test_directory("no-replace");
        let temporary = directory.join("dataset.tmp");
        let destination = directory.join("dataset.parquet");
        std::fs::write(&temporary, b"new").unwrap();
        std::fs::write(&destination, b"keep-this").unwrap();

        let error = {
            let _temporary_output = TemporaryOutput {
                path: temporary.clone(),
                committed: false,
            };
            publish_noreplace(&temporary, &destination).unwrap_err()
        };

        assert!(error.to_string().contains("refusing to clobber"));
        assert_eq!(std::fs::read(&destination).unwrap(), b"keep-this");
        assert!(!temporary.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn no_replace_publication_rejects_existing_hard_link_and_symlink() {
        let directory = test_directory("no-replace-special-files");
        let source = directory.join("source");
        let hard_link_destination = directory.join("hard-link-destination");
        let symlink_target = directory.join("symlink-target");
        let symlink_destination = directory.join("symlink-destination");
        std::fs::write(&source, b"keep-hard-link").unwrap();
        std::fs::hard_link(&source, &hard_link_destination).unwrap();
        std::fs::write(&symlink_target, b"keep-symlink").unwrap();
        std::os::unix::fs::symlink(&symlink_target, &symlink_destination).unwrap();

        for (destination, replacement) in [
            (&hard_link_destination, b"new-hard-link".as_slice()),
            (&symlink_destination, b"new-symlink".as_slice()),
        ] {
            let temporary = directory.join(format!(
                "{}.tmp",
                destination.file_name().unwrap().to_string_lossy()
            ));
            std::fs::write(&temporary, replacement).unwrap();
            let error = {
                let _temporary_output = TemporaryOutput {
                    path: temporary.clone(),
                    committed: false,
                };
                publish_noreplace(&temporary, destination).unwrap_err()
            };
            assert!(error.to_string().contains("refusing to clobber"));
            assert!(!temporary.exists());
        }

        assert_eq!(std::fs::read(&source).unwrap(), b"keep-hard-link");
        assert_eq!(std::fs::read(&symlink_target).unwrap(), b"keep-symlink");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_existing_output_before_reading_input() {
        let directory = test_directory("existing-output");
        let input = directory.join("input.parquet");
        let output = directory.join("output.parquet");
        std::fs::write(&input, b"source").unwrap();
        std::fs::write(&output, b"keep-this").unwrap();

        let error = refuse_input_output_alias(&input, &output).unwrap_err();

        assert!(error.to_string().contains("already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), b"keep-this");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_hard_link_input_output_alias() {
        let directory = test_directory("hard-link");
        let input = directory.join("input.parquet");
        let output = directory.join("output.parquet");
        std::fs::write(&input, b"source").unwrap();
        std::fs::hard_link(&input, &output).unwrap();

        let error = refuse_input_output_alias(&input, &output).unwrap_err();

        assert!(error.to_string().contains("alias"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_output_alias() {
        let directory = test_directory("symlink");
        let input = directory.join("input.parquet");
        let output = directory.join("output.parquet");
        std::fs::write(&input, b"source").unwrap();
        std::os::unix::fs::symlink(&input, &output).unwrap();

        let error = refuse_input_output_alias(&input, &output).unwrap_err();

        assert!(error.to_string().contains("symlink"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
