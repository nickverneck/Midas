//! CLI for the event-level supervised-learning pipeline.
//!
//! The first command is `prepare`: it normalizes a canonical OHLCV parquet or
//! CSV, extracts causal EMA crossover states, and writes one labeled row per
//! event.  The training and evaluation subcommands are implemented in the
//! same CLI so a prepared artifact has a stable entry point for the web UI.

use anyhow::{Context, Result, bail};
use chrono::{
    DateTime, Duration, LocalResult, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc,
};
use chrono_tz::Tz;
use clap::{Args, Parser, Subcommand};
use midas_env::ml::{self, TrainerKind};
use midas_env::supervised::{
    IndicatorKind, LabelMode, SUPERVISED_DATASET_SCHEMA, SUPERVISED_LABEL_SCHEMA,
    SupervisedBarFeatureRow, SupervisedBars, SupervisedConfig, SupervisedEvent,
    TRAINING_BAR_DATASET_SCHEMA, prepare_bar_features, prepare_events,
    standard_contract_multiplier, standard_tick_size,
};
use polars::prelude::{
    CsvReader, DataFrame, DataType, NamedFrom, ParquetReader, ParquetWriter, SerReader, Series,
    TimeUnit,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "supervised/supervised_train.rs"]
mod supervised_train;

#[derive(Debug, Parser)]
#[command(
    name = "supervised",
    about = "Prepare and train event-level supervised policies"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Prepare one labeled row for each causal crossover event.
    Prepare(PrepareArgs),
    /// Train a causal classifier from a prepared supervised-event parquet.
    Train(TrainArgs),
    /// Evaluation-only; evaluate a trained policy on a prepared parquet.
    /// Legacy supervised-policy-v1 artifacts are allowed here, but they can
    /// never be resumed by `train`.
    Evaluate(EvaluateArgs),
    /// Report backend/device availability without touching market data.
    Probe(ProbeArgs),
}

#[derive(Debug, Clone, Args)]
struct PrepareArgs {
    /// YAML, JSON, or TOML config. Explicit CLI flags override it.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Canonical OHLCV parquet/CSV, raw Databento trades, or NinjaTrader .Last.txt input.
    #[arg(long)]
    input: PathBuf,
    /// Output supervised-event parquet.
    #[arg(long)]
    output: PathBuf,
    /// Optional dense bar-level parquet shared by supervised, GA, and RL.
    /// The file retains every source bar, causal features, and nullable event
    /// labels/audit columns. The sparse event parquet remains at --output.
    #[arg(long, value_name = "PATH")]
    bar_output: Option<PathBuf>,
    #[arg(long, default_value = "UNKNOWN")]
    instrument: String,
    #[arg(long, default_value = "UNKNOWN")]
    contract: String,
    #[arg(long)]
    trigger_kind: Option<String>,
    #[arg(long)]
    trigger_fast: Option<usize>,
    #[arg(long)]
    trigger_slow: Option<usize>,
    #[arg(long)]
    context_kind: Option<String>,
    #[arg(long)]
    context_fast: Option<usize>,
    #[arg(long)]
    context_slow: Option<usize>,
    #[arg(long)]
    atr_period: Option<usize>,
    #[arg(long)]
    slope_lookback: Option<usize>,
    /// JSON or YAML array of FeatureSpec objects.
    #[arg(long)]
    features: Option<String>,
    /// JSON or YAML DerivedFeatureConfig object.
    #[arg(long)]
    derived_features: Option<String>,
    #[arg(long)]
    session_timezone: Option<String>,
    #[arg(long)]
    session_start_hour: Option<u32>,
    #[arg(long)]
    session_end_hour: Option<u32>,
    /// minute, second, tick, volume, or range. The file must already contain
    /// bars of this kind; preparation never silently aggregates a different type.
    #[arg(long)]
    bar_kind: Option<String>,
    #[arg(long)]
    bar_value: Option<f64>,
    #[arg(long)]
    contract_multiplier: Option<f64>,
    /// Total cost of a round trip; entries/exits charge half.
    #[arg(long, alias = "cost-pnl")]
    round_trip_cost: Option<f64>,
    /// Hindsight target actions: normal-skip-invert (default) or
    /// normal-invert. The latter is also accepted as normal-reverse and
    /// excludes skip from label selection while retaining its audit value.
    #[arg(long)]
    label_mode: Option<String>,
    /// Unit for a numeric timestamp/date column: ns, us, ms, or s.
    #[arg(long)]
    timestamp_unit: Option<String>,
    /// Use a legacy index as seconds only when the source has no timestamp.
    #[arg(long)]
    allow_index_timestamps: bool,
    /// Optional scale for raw Databento fixed-point prices (for example 1e9).
    /// When omitted, the loader uses a conservative magnitude-based auto mode.
    #[arg(long)]
    databento_price_scale: Option<f64>,
    /// Minimum price increment used by raw-trade range aggregation. When
    /// omitted, common GC/ES/NQ contracts are inferred from instrument/contract.
    #[arg(long)]
    tick_size: Option<f64>,
}

#[derive(Debug, Clone, Args)]
struct TrainArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    outdir: PathBuf,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 0.0001)]
    l2: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "cpu-linear")]
    backend: String,
    #[arg(long)]
    resume_policy: Option<PathBuf>,
    /// Trusted root used when regenerating the original source behind an
    /// event parquet. Defaults to the current working directory. A root
    /// outside that directory requires the explicit `--allow-external-source`
    /// opt-in below.
    #[arg(long, value_name = "DIR")]
    source_root: Option<PathBuf>,
    /// Deliberate opt-in for a trusted source root outside the current working
    /// directory. The source must still be contained by `--source-root` and
    /// must not contain symlink components.
    #[arg(long, requires = "source_root")]
    allow_external_source: bool,
    /// Evaluate train/validation metrics at this many epochs. Holdout is
    /// intentionally excluded from intermediate checkpoints. Zero disables
    /// intermediate checkpoint reporting.
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    #[arg(long, default_value_t = 0.60)]
    train_fraction: f64,
    #[arg(long, default_value_t = 0.20)]
    validation_fraction: f64,
}

#[derive(Debug, Clone, Args)]
struct EvaluateArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    policy: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    /// Optional JSON metrics output path.
    #[arg(long)]
    metrics: Option<PathBuf>,
    /// Abstain and hold the current position when the model's maximum class
    /// probability is below this threshold. Zero disables confidence gating.
    #[arg(long, default_value_t = 0.0)]
    confidence_threshold: f64,
    /// Action to use when confidence is below the threshold. `hold` preserves
    /// the original abstention behavior; `normal` or `invert` provide a
    /// directional default for an exception-gate experiment.
    #[arg(long, default_value = "hold", value_parser = ["hold", "normal", "invert"])]
    confidence_fallback: String,
    /// Session split to evaluate. Uses the policy's recorded chronological
    /// split boundaries; `all` evaluates every row in the input artifact.
    #[arg(long, default_value = "all", value_parser = ["all", "train", "validation", "holdout"])]
    split: String,
    /// Trusted root used when regenerating the original source behind an
    /// event parquet. Defaults to the current working directory.
    #[arg(long, value_name = "DIR")]
    source_root: Option<PathBuf>,
    /// Deliberate opt-in for a trusted source root outside the current working
    /// directory. The source must still be contained by `--source-root` and
    /// must not contain symlink components.
    #[arg(long, requires = "source_root")]
    allow_external_source: bool,
}

#[derive(Debug, Clone, Args)]
struct ProbeArgs {
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "cpu-linear")]
    backend: String,
}

#[derive(Debug, Clone, Serialize)]
struct PrepareSummary {
    schema_version: &'static str,
    label_schema: &'static str,
    label_mode: String,
    output: String,
    bar_output: Option<String>,
    input: String,
    source_hash_sha256: String,
    dataset_fingerprint_sha256: String,
    source_rows: usize,
    event_rows: usize,
    session_count: usize,
    feature_count: usize,
    feature_schema: String,
    timestamp_source: String,
    timestamp_unit: String,
    timestamp_timezone: String,
    index_timestamp_fallback: bool,
    raw_price_scale: Option<f64>,
    tick_size: Option<f64>,
    bar_rows: Option<usize>,
    bar_kind: String,
    bar_value: f64,
    pnl_currency: &'static str,
    contract_multiplier: f64,
    round_trip_cost: f64,
    contract_multiplier_source: &'static str,
    leakage_check: &'static str,
}

#[derive(Debug, Clone)]
struct LoadedSource {
    bars: SupervisedBars,
    timestamp_source: String,
    timestamp_unit: String,
    timestamp_timezone: String,
    index_timestamp_fallback: bool,
    raw_price_scale: Option<f64>,
    tick_size: Option<f64>,
    volume_present: bool,
    source_row_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare(args) => prepare(args),
        Command::Train(args) => train(args),
        Command::Evaluate(args) => evaluate(args),
        Command::Probe(args) => probe(args),
    }
}

fn prepare(args: PrepareArgs) -> Result<()> {
    refuse_input_output_alias(&args.input, &args.output)?;
    if let Some(bar_output) = &args.bar_output {
        refuse_input_output_alias(&args.input, bar_output)?;
        if bar_output == &args.output {
            bail!("--bar-output must be different from --output");
        }
    }
    let (mut config, config_has_explicit_multiplier) = if let Some(path) = &args.config {
        read_config(path)?
    } else {
        (SupervisedConfig::default(), false)
    };
    apply_overrides(&mut config, &args)?;
    let contract_multiplier_source = if args.contract_multiplier.is_some() {
        "cli"
    } else if config_has_explicit_multiplier {
        "config"
    } else {
        config.contract_multiplier = standard_contract_multiplier(&args.instrument, &args.contract)
            .ok_or_else(|| anyhow::anyhow!(
                "no contract_multiplier supplied and instrument/contract {:?}/{:?} is not recognized; pass --contract-multiplier explicitly instead of producing raw price-unit PnL",
                args.instrument,
                args.contract,
            ))?;
        "instrument-default"
    };
    config.validate()?;

    let tick_size = args
        .tick_size
        .or_else(|| standard_tick_size(&args.instrument, &args.contract));
    if let Some(tick_size) = tick_size {
        if !tick_size.is_finite() || tick_size <= 0.0 {
            bail!("tick_size must be finite and positive");
        }
    }
    let loaded = load_source(
        &args.input,
        args.timestamp_unit.as_deref(),
        args.allow_index_timestamps,
        &config.session_timezone,
        &config.bar_kind,
        config.bar_value,
        args.databento_price_scale,
        tick_size,
    )?;
    if config
        .features
        .iter()
        .any(|feature| matches!(feature.indicator, IndicatorKind::Rvol))
        && !loaded.volume_present
    {
        bail!(
            "the selected RVOL feature requires a volume column; refusing to replace missing volume with zeros"
        );
    }
    let events = prepare_events(&loaded.bars, &config)?;
    let events = retain_complete_sessions(events, &loaded.bars, &config)?;
    if events.is_empty() {
        bail!(
            "source produced no crossover events in complete {}:00–{}:00 sessions; a trailing partial session is excluded",
            config.session_start_hour,
            config.session_end_hour
        );
    }
    write_dataset(
        &args.output,
        &args.input,
        &args.instrument,
        &args.contract,
        &loaded,
        &config,
        &events,
    )?;
    let bar_rows = if let Some(bar_output) = &args.bar_output {
        write_training_bar_dataset(
            bar_output,
            &args.input,
            &args.instrument,
            &args.contract,
            &loaded,
            &config,
            &events,
        )?;
        Some(loaded.bars.close.len())
    } else {
        None
    };
    let summary = PrepareSummary {
        schema_version: SUPERVISED_DATASET_SCHEMA,
        label_schema: SUPERVISED_LABEL_SCHEMA,
        label_mode: config.label_mode.to_string(),
        output: args.output.display().to_string(),
        bar_output: args
            .bar_output
            .as_ref()
            .map(|path| path.display().to_string()),
        input: args.input.display().to_string(),
        source_hash_sha256: sha256_file(&args.input)?,
        dataset_fingerprint_sha256: dataset_fingerprint(&args.input, &loaded, &config, &events)?,
        source_rows: loaded.source_row_count,
        event_rows: events.len(),
        session_count: events
            .iter()
            .map(|event| event.session_id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        feature_count: config.feature_names().len(),
        feature_schema: config.feature_schema(),
        timestamp_source: loaded.timestamp_source,
        timestamp_unit: loaded.timestamp_unit,
        timestamp_timezone: loaded.timestamp_timezone,
        index_timestamp_fallback: loaded.index_timestamp_fallback,
        raw_price_scale: loaded.raw_price_scale,
        tick_size: loaded.tick_size,
        bar_rows,
        bar_kind: config.bar_kind,
        bar_value: config.bar_value,
        pnl_currency: "USD",
        contract_multiplier: config.contract_multiplier,
        round_trip_cost: config.round_trip_cost,
        contract_multiplier_source,
        leakage_check: "passed",
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
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

    // `exists()` follows symlinks and is false for dangling symlinks. Inspect
    // the directory entry first so an output symlink can never be replaced by
    // the atomic commit below.
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

    // A missing output has no inode to compare, but canonicalizing its parent
    // still catches path aliases such as `input/../output` and keeps the
    // temporary file in the directory where the final name will live.
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

fn read_config(path: &Path) -> Result<(SupervisedConfig, bool)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read supervised config {}", path.display()))?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("yaml")
        .to_ascii_lowercase();
    let value = match extension.as_str() {
        "json" => serde_json::from_str::<serde_json::Value>(&text)
            .context("parse supervised JSON config")?,
        "toml" => toml::from_str::<toml::Value>(&text)
            .context("parse supervised TOML config")?
            .try_into()
            .context("convert TOML supervised config")?,
        _ => serde_yaml::from_str::<serde_yaml::Value>(&text)
            .context("parse supervised YAML config")
            .and_then(|value| serde_json::to_value(value).context("normalize YAML config"))?,
    };
    let value = value
        .get("supervised")
        .or_else(|| value.get("config"))
        .cloned()
        .unwrap_or(value);
    let has_explicit_multiplier = value.get("contract_multiplier").is_some();
    let config = serde_json::from_value(value).context("decode supervised config")?;
    Ok((config, has_explicit_multiplier))
}

fn apply_overrides(config: &mut SupervisedConfig, args: &PrepareArgs) -> Result<()> {
    if let Some(value) = &args.trigger_kind {
        config.trigger_kind = value.parse()?;
    }
    if let Some(value) = args.trigger_fast {
        config.trigger_fast = value;
    }
    if let Some(value) = args.trigger_slow {
        config.trigger_slow = value;
    }
    if let Some(value) = &args.context_kind {
        config.context_kind = value.parse()?;
    }
    if let Some(value) = args.context_fast {
        config.context_fast = value;
    }
    if let Some(value) = args.context_slow {
        config.context_slow = value;
    }
    if let Some(value) = args.atr_period {
        config.atr_period = value;
    }
    if let Some(value) = args.slope_lookback {
        config.slope_lookback = value;
    }
    if let Some(value) = &args.features {
        config.features = parse_value(value).context("parse --features FeatureSpec array")?;
    }
    if let Some(value) = &args.derived_features {
        config.derived_features =
            parse_value(value).context("parse --derived-features configuration")?;
    }
    if let Some(value) = &args.session_timezone {
        config.session_timezone = value.clone();
    }
    if let Some(value) = args.session_start_hour {
        config.session_start_hour = value;
    }
    if let Some(value) = args.session_end_hour {
        config.session_end_hour = value;
    }
    if let Some(value) = &args.bar_kind {
        config.bar_kind = value.clone();
    }
    if let Some(value) = args.bar_value {
        config.bar_value = value;
    }
    if let Some(value) = args.contract_multiplier {
        config.contract_multiplier = value;
    }
    if let Some(value) = args.round_trip_cost {
        config.round_trip_cost = value;
    }
    if let Some(value) = &args.label_mode {
        config.label_mode = value.parse::<LabelMode>()?;
    }
    Ok(())
}

fn parse_value<T: DeserializeOwned>(text: &str) -> Result<T> {
    if let Ok(value) = serde_json::from_str(text) {
        return Ok(value);
    }
    serde_yaml::from_str(text).context("value is neither valid JSON nor YAML")
}

fn load_source(
    path: &Path,
    timestamp_unit: Option<&str>,
    allow_index_timestamps: bool,
    timezone_name: &str,
    bar_kind: &str,
    bar_value: f64,
    databento_price_scale: Option<f64>,
    tick_size: Option<f64>,
) -> Result<LoadedSource> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("txt"))
        && looks_like_ninja_last(path)?
    {
        return load_ninja_last_text(path, timezone_name, bar_kind, bar_value, tick_size);
    }
    let file = File::open(path).with_context(|| format!("open source {}", path.display()))?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let df = if extension == "csv" || extension == "txt" {
        CsvReader::new(file).finish().context("read source CSV")?
    } else {
        ParquetReader::new(file)
            .finish()
            .context("read source parquet")?
    };
    if find_column(&df, &["open", "o"])?.is_none() && find_column(&df, &["price"])?.is_some() {
        return aggregate_databento_trades(
            &df,
            timestamp_unit,
            allow_index_timestamps,
            timezone_name,
            bar_kind,
            bar_value,
            databento_price_scale,
            tick_size,
        );
    }
    load_dataframe(
        &df,
        timestamp_unit,
        allow_index_timestamps,
        timezone_name,
        tick_size,
    )
}

fn aggregate_databento_trades(
    df: &DataFrame,
    timestamp_unit: Option<&str>,
    allow_index_timestamps: bool,
    timezone_name: &str,
    bar_kind: &str,
    bar_value: f64,
    databento_price_scale: Option<f64>,
    tick_size: Option<f64>,
) -> Result<LoadedSource> {
    let kind = bar_kind.to_ascii_lowercase();
    let seconds = match kind.as_str() {
        "minute" | "minutes" | "1m" => 60.0 * bar_value,
        "second" | "seconds" | "1s" => bar_value,
        "range" => 0.0,
        "tick" | "ticks" => 0.0,
        "volume" => 0.0,
        _ => bail!("unsupported raw Databento bar kind `{bar_kind}`"),
    };
    if !matches!(kind.as_str(), "range" | "tick" | "ticks" | "volume")
        && (!seconds.is_finite() || seconds <= 0.0)
    {
        bail!("raw trade bar value must be positive");
    }
    let (prices, raw_price_scale) = databento_price_column(df, databento_price_scale)?;
    let sizes = optional_numeric_column(df, &["size"])?;
    let size_present = sizes.is_some();
    let sizes = sizes.unwrap_or_else(|| vec![0.0; prices.len()]);
    let (timestamps, timestamp_source, timestamp_unit, index_timestamp_fallback) =
        timestamp_column(
            df,
            prices.len(),
            timestamp_unit,
            allow_index_timestamps,
            timezone_name,
        )?;
    if timestamps.windows(2).any(|window| window[1] < window[0]) {
        bail!("raw Databento timestamps must be non-decreasing");
    }
    if matches!(kind.as_str(), "range") {
        let tick_size = tick_size.ok_or_else(|| {
            anyhow::anyhow!(
                "raw range aggregation requires --tick-size or a recognized GC/ES/NQ instrument/contract"
            )
        })?;
        let bars = aggregate_range_trades(&timestamps, &prices, &sizes, tick_size, bar_value)?;
        if bars.len() < 2 {
            bail!("raw Databento source produced fewer than two range bars");
        }
        return Ok(LoadedSource {
            bars: SupervisedBars {
                timestamp_ns: bars.iter().map(|row| row.0).collect(),
                open: bars.iter().map(|row| row.1).collect(),
                high: bars.iter().map(|row| row.2).collect(),
                low: bars.iter().map(|row| row.3).collect(),
                close: bars.iter().map(|row| row.4).collect(),
                volume: bars.iter().map(|row| row.5).collect(),
            },
            timestamp_source: format!("databento_{timestamp_source}_range_ticks"),
            timestamp_unit,
            timestamp_timezone: timezone_name.to_string(),
            index_timestamp_fallback,
            raw_price_scale: Some(raw_price_scale),
            tick_size: Some(tick_size),
            volume_present: size_present,
            source_row_count: df.height(),
        });
    }
    if matches!(kind.as_str(), "tick" | "ticks") {
        let ticks_per_bar = bar_value.round();
        if !bar_value.is_finite() || bar_value <= 0.0 || (ticks_per_bar - bar_value).abs() > 1e-9 {
            bail!("raw tick bars require a positive integer --bar-value");
        }
        let ticks_per_bar = ticks_per_bar as usize;
        let mut rows: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
        for (index, ((timestamp, price), size)) in timestamps
            .iter()
            .zip(prices.iter())
            .zip(sizes.iter())
            .enumerate()
        {
            let bar_index = index / ticks_per_bar;
            if let Some(last) = rows.get_mut(bar_index) {
                last.2 = last.2.max(*price);
                last.3 = last.3.min(*price);
                last.4 = *price;
                last.5 += *size;
            } else {
                rows.push((*timestamp, *price, *price, *price, *price, *size));
            }
        }
        make_timestamps_strictly_increasing(&mut rows);
        if rows.len() < 2 {
            bail!("raw Databento source produced fewer than two tick bars");
        }
        return Ok(LoadedSource {
            bars: SupervisedBars {
                timestamp_ns: rows.iter().map(|row| row.0).collect(),
                open: rows.iter().map(|row| row.1).collect(),
                high: rows.iter().map(|row| row.2).collect(),
                low: rows.iter().map(|row| row.3).collect(),
                close: rows.iter().map(|row| row.4).collect(),
                volume: rows.iter().map(|row| row.5).collect(),
            },
            timestamp_source: format!("databento_{timestamp_source}_tick_count"),
            timestamp_unit,
            timestamp_timezone: timezone_name.to_string(),
            index_timestamp_fallback,
            raw_price_scale: Some(raw_price_scale),
            tick_size,
            volume_present: size_present,
            source_row_count: df.height(),
        });
    }
    if kind == "volume" {
        if !bar_value.is_finite() || bar_value <= 0.0 {
            bail!("raw volume bars require a positive --bar-value");
        }
        if !size_present {
            bail!("raw volume bars require a size column");
        }
        let mut rows: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
        let mut current: Option<(i64, f64, f64, f64, f64, f64)> = None;
        let mut current_volume = 0.0;
        for ((timestamp, price), size) in timestamps.iter().zip(prices.iter()).zip(sizes.iter()) {
            let mut remaining = size.max(0.0);
            while remaining > 0.0 {
                if current.is_none() {
                    current = Some((*timestamp, *price, *price, *price, *price, 0.0));
                    current_volume = 0.0;
                }
                let capacity = (bar_value - current_volume).max(0.0);
                let consumed = remaining.min(capacity.max(f64::EPSILON));
                if let Some(bar) = current.as_mut() {
                    bar.2 = bar.2.max(*price);
                    bar.3 = bar.3.min(*price);
                    bar.4 = *price;
                    bar.5 += consumed;
                    bar.0 = *timestamp;
                }
                current_volume += consumed;
                remaining -= consumed;
                if current_volume >= bar_value - 1e-9 {
                    rows.push(current.take().expect("volume bar exists"));
                    current_volume = 0.0;
                }
            }
        }
        if let Some(bar) = current {
            rows.push(bar);
        }
        make_timestamps_strictly_increasing(&mut rows);
        if rows.len() < 2 {
            bail!("raw Databento source produced fewer than two volume bars");
        }
        return Ok(LoadedSource {
            bars: SupervisedBars {
                timestamp_ns: rows.iter().map(|row| row.0).collect(),
                open: rows.iter().map(|row| row.1).collect(),
                high: rows.iter().map(|row| row.2).collect(),
                low: rows.iter().map(|row| row.3).collect(),
                close: rows.iter().map(|row| row.4).collect(),
                volume: rows.iter().map(|row| row.5).collect(),
            },
            timestamp_source: format!("databento_{timestamp_source}_volume"),
            timestamp_unit,
            timestamp_timezone: timezone_name.to_string(),
            index_timestamp_fallback,
            raw_price_scale: Some(raw_price_scale),
            tick_size,
            volume_present: true,
            source_row_count: df.height(),
        });
    }
    let interval_ns = (seconds * 1_000_000_000.0).round();
    if interval_ns < 1.0 || interval_ns > i64::MAX as f64 {
        bail!("raw trade bar interval is outside supported range");
    }
    let interval_ns = interval_ns as i64;
    let mut rows: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
    for ((timestamp, price), size) in timestamps.iter().zip(prices.iter()).zip(sizes.iter()) {
        let bucket = timestamp.div_euclid(interval_ns) * interval_ns;
        if let Some(last) = rows.last_mut() {
            if bucket < last.0 {
                bail!("raw Databento buckets are not ordered");
            }
            if bucket == last.0 {
                last.2 = last.2.max(*price);
                last.3 = last.3.min(*price);
                last.4 = *price;
                last.5 += *size;
                continue;
            }
        }
        rows.push((bucket, *price, *price, *price, *price, *size));
    }
    if rows.len() < 2 {
        bail!("raw Databento source produced fewer than two bars");
    }
    Ok(LoadedSource {
        bars: SupervisedBars {
            timestamp_ns: rows.iter().map(|row| row.0).collect(),
            open: rows.iter().map(|row| row.1).collect(),
            high: rows.iter().map(|row| row.2).collect(),
            low: rows.iter().map(|row| row.3).collect(),
            close: rows.iter().map(|row| row.4).collect(),
            volume: rows.iter().map(|row| row.5).collect(),
        },
        timestamp_source: format!("databento_{timestamp_source}_{kind}_bucket"),
        timestamp_unit,
        timestamp_timezone: timezone_name.to_string(),
        index_timestamp_fallback,
        raw_price_scale: Some(raw_price_scale),
        tick_size,
        volume_present: size_present,
        source_row_count: df.height(),
    })
}

/// Derive range bars from an ordered raw trade stream.  This mirrors Trader's
/// boundary state machine: a source trade can complete multiple bars, each
/// synthetic boundary close is emitted in order, and the final unfinished bar
/// is retained.  `bar_value` is a tick count; volume from the boundary trade
/// is assigned once to the first bar it completes and subsequent synthetic
/// bars receive zero volume.
fn aggregate_range_trades(
    timestamps: &[i64],
    prices: &[f64],
    sizes: &[f64],
    tick_size: f64,
    bar_value: f64,
) -> Result<Vec<(i64, f64, f64, f64, f64, f64)>> {
    if !tick_size.is_finite() || tick_size <= 0.0 {
        bail!("range tick_size must be finite and positive");
    }
    if !bar_value.is_finite() || bar_value <= 0.0 {
        bail!("range bar_value must be finite and positive");
    }
    if timestamps.len() != prices.len() || prices.len() != sizes.len() {
        bail!("raw range columns have inconsistent lengths");
    }
    let range_size = tick_size * bar_value;
    if !range_size.is_finite() || range_size <= 0.0 {
        bail!("range size is outside the supported numeric range");
    }

    // (timestamp, open, high, low, close, volume)
    let mut output = Vec::new();
    let mut current: Option<(i64, f64, f64, f64, f64, f64)> = None;
    const EPSILON: f64 = 1e-9;
    for ((timestamp, price), size) in timestamps.iter().zip(prices).zip(sizes) {
        if !price.is_finite() || *timestamp == i64::MIN {
            bail!("raw range trade contains an invalid timestamp or price");
        }
        let size = if size.is_finite() && *size >= 0.0 {
            *size
        } else {
            0.0
        };
        let mut current_bar = current
            .take()
            .unwrap_or((*timestamp, *price, *price, *price, *price, 0.0));
        current_bar.5 += size;

        loop {
            let tentative_high = current_bar.2.max(*price);
            let tentative_low = current_bar.3.min(*price);
            let breaks_up =
                *price > current_bar.2 && (*price - tentative_low) >= range_size - EPSILON;
            let breaks_down =
                *price < current_bar.3 && (tentative_high - *price) >= range_size - EPSILON;
            if breaks_up {
                let close = tentative_low + range_size;
                current_bar.2 = close;
                current_bar.4 = close;
                current_bar.0 = *timestamp;
                output.push(current_bar);
                current_bar = (*timestamp, close, close, close, close, 0.0);
                if *price <= close + EPSILON {
                    break;
                }
                continue;
            }
            if breaks_down {
                let close = tentative_high - range_size;
                current_bar.3 = close;
                current_bar.4 = close;
                current_bar.0 = *timestamp;
                output.push(current_bar);
                current_bar = (*timestamp, close, close, close, close, 0.0);
                if *price >= close - EPSILON {
                    break;
                }
                continue;
            }
            current_bar.2 = tentative_high;
            current_bar.3 = tentative_low;
            current_bar.4 = *price;
            current_bar.0 = *timestamp;
            break;
        }
        current = Some(current_bar);
    }
    if let Some(current) = current {
        output.push(current);
    }
    make_timestamps_strictly_increasing(&mut output);
    Ok(output)
}

fn make_timestamps_strictly_increasing(rows: &mut [(i64, f64, f64, f64, f64, f64)]) {
    let mut previous = None;
    for row in rows {
        if let Some(last) = previous {
            if row.0 <= last {
                row.0 = last.saturating_add(1);
            }
        }
        previous = Some(row.0);
    }
}

fn looks_like_ninja_last(path: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read text source {}", path.display()))?;
    Ok(text.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || !line.contains(';') {
            return false;
        }
        let first = line.split(';').next().unwrap_or_default().trim();
        ninja_timestamp_prefix(first)
    }))
}

fn ninja_timestamp_prefix(value: &str) -> bool {
    value.len() >= 8
        && value
            .as_bytes()
            .get(..8)
            .is_some_and(|bytes| bytes.iter().all(u8::is_ascii_digit))
}

/// Read NinjaTrader Market Replay `.Last.txt` lines and form sparse one-minute
/// bars.  These files do not carry trusted trade size in the format we have
/// locally, so volume is intentionally zero/missing and volume features are
/// rejected by `prepare` rather than fabricated.
fn load_ninja_last_text(
    path: &Path,
    timezone_name: &str,
    bar_kind: &str,
    bar_value: f64,
    tick_size: Option<f64>,
) -> Result<LoadedSource> {
    if !matches!(
        bar_kind.to_ascii_lowercase().as_str(),
        "minute" | "minutes" | "1m"
    ) || (bar_value - 1.0).abs() > f64::EPSILON
    {
        bail!(
            "NinjaTrader .Last.txt currently supports only one-minute bars; convert it with Trader for seconds, tick, volume, or range bars"
        );
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read NinjaTrader source {}", path.display()))?;
    let timezone = timezone_name
        .parse::<Tz>()
        .with_context(|| format!("parse timezone {timezone_name}"))?;
    let minute_ns = 60_000_000_000_i64;
    let mut rows: Vec<(i64, f64, f64, f64, f64)> = Vec::new();
    let mut previous_timestamp_ns = None;
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = line.split(';').map(str::trim).collect::<Vec<_>>();
        if !ninja_timestamp_prefix(fields.first().copied().unwrap_or_default()) {
            // Ninja exports may include a header such as `Date;Last;...`.
            continue;
        }
        if fields.len() < 2 {
            bail!("NinjaTrader line {} has no price field", line_number + 1);
        }
        let timestamp = parse_ninja_last_timestamp(fields[0], timezone, previous_timestamp_ns)
            .with_context(|| format!("parse NinjaTrader timestamp at line {}", line_number + 1))?;
        if previous_timestamp_ns.is_some_and(|previous| timestamp < previous) {
            bail!(
                "NinjaTrader timestamps are not increasing at line {}",
                line_number + 1
            );
        }
        previous_timestamp_ns = Some(timestamp);
        let price = fields[1]
            .parse::<f64>()
            .with_context(|| format!("parse NinjaTrader price at line {}", line_number + 1))?;
        if !price.is_finite() {
            bail!(
                "NinjaTrader price at line {} is non-finite",
                line_number + 1
            );
        }
        let bucket = timestamp.div_euclid(minute_ns) * minute_ns;
        if let Some(last) = rows.last_mut() {
            if bucket < last.0 {
                bail!(
                    "NinjaTrader timestamps are not increasing at line {}",
                    line_number + 1
                );
            }
            if bucket == last.0 {
                last.2 = last.2.max(price);
                last.3 = last.3.min(price);
                last.4 = price;
                continue;
            }
        }
        rows.push((bucket, price, price, price, price));
    }
    if rows.len() < 2 {
        bail!("NinjaTrader source produced fewer than two one-minute bars");
    }
    Ok(LoadedSource {
        bars: SupervisedBars {
            timestamp_ns: rows.iter().map(|row| row.0).collect(),
            open: rows.iter().map(|row| row.1).collect(),
            high: rows.iter().map(|row| row.2).collect(),
            low: rows.iter().map(|row| row.3).collect(),
            close: rows.iter().map(|row| row.4).collect(),
            volume: vec![0.0; rows.len()],
        },
        timestamp_source: "ninjatrader_last_text_minute_bucket".to_string(),
        timestamp_unit: "ns".to_string(),
        timestamp_timezone: timezone_name.to_string(),
        index_timestamp_fallback: false,
        raw_price_scale: None,
        tick_size,
        volume_present: false,
        source_row_count: text.lines().filter(|line| !line.trim().is_empty()).count(),
    })
}

fn retain_complete_sessions(
    events: Vec<SupervisedEvent>,
    bars: &SupervisedBars,
    config: &SupervisedConfig,
) -> Result<Vec<SupervisedEvent>> {
    let timezone = config
        .session_timezone
        .parse::<Tz>()
        .with_context(|| format!("parse timezone {}", config.session_timezone))?;
    let mut complete = std::collections::BTreeSet::new();
    for event in events.iter().filter(|event| event.terminal_event) {
        let timestamp_ns = bars
            .timestamp_ns
            .get(event.interval_end_row_idx)
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "terminal event {} refers to missing interval row {}",
                    event.event_id,
                    event.interval_end_row_idx
                )
            })?;
        let utc = DateTime::<Utc>::from_timestamp(
            timestamp_ns.div_euclid(1_000_000_000),
            timestamp_ns.rem_euclid(1_000_000_000) as u32,
        )
        .ok_or_else(|| anyhow::anyhow!("session close timestamp is outside chrono range"))?;
        let local = utc.with_timezone(&timezone);
        let end = NaiveTime::from_hms_opt(config.session_end_hour, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("invalid session end hour"))?;
        // A one-minute bar stamped 16:59 represents the final 16:59–17:00
        // interval. Five minutes allows the same convention for sparse bars
        // without accepting a clearly truncated session.
        let threshold = end - Duration::minutes(5);
        if local.time() >= threshold {
            complete.insert(event.session_id.clone());
        }
    }
    Ok(events
        .into_iter()
        .filter(|event| complete.contains(&event.session_id))
        .collect())
}

/// Write the dense companion parquet.  It is intentionally self-describing:
/// GA/RL can load the ordinary OHLCV columns and the ordered feature columns,
/// while supervised training can use the sparse event parquet emitted by the
/// same invocation.  Label and counterfactual columns are nullable outside an
/// event row and are never part of `feature_schema`.
fn write_training_bar_dataset(
    output: &Path,
    input: &Path,
    instrument: &str,
    contract: &str,
    source: &LoadedSource,
    config: &SupervisedConfig,
    events: &[SupervisedEvent],
) -> Result<()> {
    let rows = prepare_bar_features(&source.bars, config)?;
    if rows.len() != source.bars.close.len() {
        bail!("dense feature row count does not match source bars");
    }
    let source_hash = sha256_file(input)?;
    let fingerprint = bar_dataset_fingerprint(input, source, config, &rows)?;
    let config_json = serde_json::to_string(config)?;
    let provenance_json = serde_json::json!({
        "source_path": input.display().to_string(),
        "timestamp_source": source.timestamp_source,
        "timestamp_unit": source.timestamp_unit,
        "timestamp_timezone": source.timestamp_timezone,
        "index_timestamp_fallback": source.index_timestamp_fallback,
        "raw_price_scale": source.raw_price_scale,
        "tick_size": source.tick_size,
        "bar_kind": config.bar_kind,
        "bar_value": config.bar_value,
        "source_row_count": source.source_row_count,
        "bar_row_count": rows.len(),
        "source_hash_sha256": source_hash,
        "dataset_fingerprint_sha256": fingerprint,
        "range_volume_policy": if config.bar_kind.eq_ignore_ascii_case("range") {
            "close_tick_once"
        } else {
            "native_or_source_volume"
        },
    });
    let event_by_row = events
        .iter()
        .map(|event| (event.row_idx, event))
        .collect::<BTreeMap<_, _>>();
    let repeated = |value: &str| vec![value.to_string(); rows.len()];
    let source_size = input.metadata()?.len() as i64;
    let (session_open, minutes_to_close) = rows
        .iter()
        .map(|row| session_context(row.timestamp_ns, config))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip::<bool, f64, Vec<bool>, Vec<f64>>();

    let mut columns = vec![
        Series::new(
            "schema_version".into(),
            repeated(TRAINING_BAR_DATASET_SCHEMA),
        )
        .into(),
        Series::new("dataset_role".into(), repeated("dense_bar_features")).into(),
        Series::new("label_schema".into(), repeated(SUPERVISED_LABEL_SCHEMA)).into(),
        Series::new("feature_schema".into(), repeated(&config.feature_schema())).into(),
        Series::new("config_json".into(), repeated(&config_json)).into(),
        Series::new("source_path".into(), repeated(&input.display().to_string())).into(),
        Series::new("source_hash_sha256".into(), repeated(&source_hash)).into(),
        Series::new("dataset_fingerprint_sha256".into(), repeated(&fingerprint)).into(),
        Series::new(
            "source_row_count".into(),
            vec![source.source_row_count as i64; rows.len()],
        )
        .into(),
        Series::new("source_size_bytes".into(), vec![source_size; rows.len()]).into(),
        Series::new("bar_row_count".into(), vec![rows.len() as i64; rows.len()]).into(),
        Series::new("instrument".into(), repeated(instrument)).into(),
        Series::new("contract".into(), repeated(contract)).into(),
        Series::new("symbol".into(), repeated(instrument)).into(),
        Series::new("bar_kind".into(), repeated(&config.bar_kind)).into(),
        Series::new("bar_value".into(), vec![config.bar_value; rows.len()]).into(),
        Series::new(
            "tick_size".into(),
            vec![source.tick_size.unwrap_or(f64::NAN); rows.len()],
        )
        .into(),
        Series::new("pnl_currency".into(), repeated("USD")).into(),
        Series::new(
            "contract_multiplier".into(),
            vec![config.contract_multiplier; rows.len()],
        )
        .into(),
        Series::new(
            "round_trip_cost".into(),
            vec![config.round_trip_cost; rows.len()],
        )
        .into(),
        Series::new(
            "timestamp_source".into(),
            repeated(&source.timestamp_source),
        )
        .into(),
        Series::new("timestamp_unit".into(), repeated(&source.timestamp_unit)).into(),
        Series::new(
            "timestamp_timezone".into(),
            repeated(&source.timestamp_timezone),
        )
        .into(),
        Series::new(
            "provenance_json".into(),
            repeated(&provenance_json.to_string()),
        )
        .into(),
        Series::new(
            "index_timestamp_fallback".into(),
            vec![source.index_timestamp_fallback; rows.len()],
        )
        .into(),
        Series::new(
            "raw_price_scale".into(),
            vec![source.raw_price_scale.unwrap_or(1.0); rows.len()],
        )
        .into(),
        Series::new("leakage_check".into(), repeated("passed")).into(),
        Series::new(
            "row_idx".into(),
            rows.iter()
                .map(|row| row.row_idx as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "timestamp_ns".into(),
            rows.iter().map(|row| row.timestamp_ns).collect::<Vec<_>>(),
        )
        .into(),
        Series::new("open".into(), source.bars.open.clone()).into(),
        Series::new("high".into(), source.bars.high.clone()).into(),
        Series::new("low".into(), source.bars.low.clone()).into(),
        Series::new("close".into(), source.bars.close.clone()).into(),
        Series::new("volume".into(), source.bars.volume.clone()).into(),
        Series::new(
            "session_id".into(),
            rows.iter()
                .map(|row| row.session_id.clone().unwrap_or_default())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new("session_open".into(), session_open).into(),
        Series::new("minutes_to_close".into(), minutes_to_close).into(),
        Series::new(
            "feature_ready".into(),
            rows.iter().map(|row| row.ready).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "cross_direction".into(),
            rows.iter()
                .map(|row| row.cross_direction.unwrap_or(0) as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "is_event".into(),
            rows.iter()
                .map(|row| event_by_row.contains_key(&row.row_idx))
                .collect::<Vec<_>>(),
        )
        .into(),
    ];

    for name in config.feature_names() {
        columns.push(
            Series::new(
                name.clone().into(),
                rows.iter()
                    .map(|row| row.features.get(&name).copied().unwrap_or(f64::NAN))
                    .collect::<Vec<_>>(),
            )
            .into(),
        );
    }

    let event_ids = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.event_id as i64)
        })
        .collect::<Vec<_>>();
    let session_event_indices = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.session_event_index as i64)
        })
        .collect::<Vec<_>>();
    let entry_rows = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.row_idx.saturating_add(1) as i64)
        })
        .collect::<Vec<_>>();
    let interval_end_rows = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.interval_end_row_idx as i64)
        })
        .collect::<Vec<_>>();
    let terminal = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.terminal_event)
        })
        .collect::<Vec<_>>();
    let decision_prices = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.decision_price)
        })
        .collect::<Vec<_>>();
    let interval_end_prices = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.interval_end_price)
        })
        .collect::<Vec<_>>();
    let action_normal = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.action_value_normal)
        })
        .collect::<Vec<_>>();
    let action_skip = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.action_value_skip)
        })
        .collect::<Vec<_>>();
    let action_invert = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.action_value_invert)
        })
        .collect::<Vec<_>>();
    let labels = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.label_action as i32)
        })
        .collect::<Vec<_>>();
    let label_names = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.label_name.clone())
        })
        .collect::<Vec<_>>();
    let oracle_before = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.oracle_position_before as i32)
        })
        .collect::<Vec<_>>();
    let oracle_after = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.oracle_position_after as i32)
        })
        .collect::<Vec<_>>();
    let oracle_values = rows
        .iter()
        .map(|row| {
            event_by_row
                .get(&row.row_idx)
                .map(|event| event.oracle_value)
        })
        .collect::<Vec<_>>();
    columns.extend([
        Series::new("event_id".into(), event_ids).into(),
        Series::new("session_event_index".into(), session_event_indices).into(),
        Series::new("entry_row_idx".into(), entry_rows).into(),
        Series::new("interval_end_row_idx".into(), interval_end_rows).into(),
        Series::new("terminal_event".into(), terminal).into(),
        Series::new("decision_price".into(), decision_prices).into(),
        Series::new("interval_end_price".into(), interval_end_prices).into(),
        Series::new("action_value_normal".into(), action_normal).into(),
        Series::new("action_value_skip".into(), action_skip).into(),
        Series::new("action_value_invert".into(), action_invert).into(),
        Series::new("label_action".into(), labels).into(),
        Series::new("label_name".into(), label_names).into(),
        Series::new("oracle_position_before".into(), oracle_before).into(),
        Series::new("oracle_position_after".into(), oracle_after).into(),
        Series::new("oracle_value".into(), oracle_values).into(),
    ]);

    let mut frame = DataFrame::new(columns)?;
    let (temporary_path, temporary_file) = create_unique_output_file(output)?;
    let mut temporary_output = TemporaryOutput {
        path: temporary_path,
        committed: false,
    };
    ParquetWriter::new(temporary_file).finish(&mut frame)?;
    publish_noreplace(&temporary_output.path, output).with_context(|| {
        format!(
            "atomically commit training bar dataset {}",
            output.display()
        )
    })?;
    temporary_output.committed = true;
    Ok(())
}

fn session_context(timestamp_ns: i64, config: &SupervisedConfig) -> Result<(bool, f64)> {
    let timezone = config
        .session_timezone
        .parse::<Tz>()
        .with_context(|| format!("parse timezone {}", config.session_timezone))?;
    let utc = DateTime::<Utc>::from_timestamp(
        timestamp_ns.div_euclid(1_000_000_000),
        timestamp_ns.rem_euclid(1_000_000_000) as u32,
    )
    .ok_or_else(|| anyhow::anyhow!("timestamp outside chrono range"))?;
    let local = utc.with_timezone(&timezone);
    let start = NaiveTime::from_hms_opt(config.session_start_hour, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("invalid session start hour"))?;
    let end = NaiveTime::from_hms_opt(config.session_end_hour, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("invalid session end hour"))?;
    let time = local.time();
    let open = time >= start || time < end;
    let seconds = time.num_seconds_from_midnight() as f64;
    let end_seconds = end.num_seconds_from_midnight() as f64;
    let minutes = if time < end {
        (end_seconds - seconds) / 60.0
    } else if time >= start {
        (86_400.0 - seconds + end_seconds) / 60.0
    } else {
        0.0
    };
    Ok((open, minutes.max(0.0)))
}

fn bar_dataset_fingerprint(
    input: &Path,
    source: &LoadedSource,
    config: &SupervisedConfig,
    rows: &[SupervisedBarFeatureRow],
) -> Result<String> {
    let source_hash = sha256_file(input)?;
    let mut digest = Sha256::new();
    digest.update(source_hash.as_bytes());
    digest.update([0]);
    digest.update(serde_json::to_vec(config)?);
    digest.update([0]);
    digest.update(source.timestamp_source.as_bytes());
    digest.update([0]);
    digest.update(source.timestamp_unit.as_bytes());
    digest.update([0]);
    digest.update(source.timestamp_timezone.as_bytes());
    digest.update(source.tick_size.unwrap_or(f64::NAN).to_le_bytes());
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        digest.update(serde_json::to_vec(row)?);
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn parse_ninja_last_timestamp(
    value: &str,
    timezone: Tz,
    previous_timestamp_ns: Option<i64>,
) -> Result<i64> {
    let parts = value.split_whitespace().collect::<Vec<_>>();
    if parts.len() < 2 {
        bail!("expected `YYYYMMDD HHMMSS [fraction]`");
    }
    let date = NaiveDate::parse_from_str(parts[0], "%Y%m%d")?;
    let time = NaiveTime::parse_from_str(parts[1], "%H%M%S")?;
    let nanos = if let Some(fraction) = parts.get(2) {
        let digits = fraction
            .trim_end_matches(|value: char| !value.is_ascii_digit())
            .chars()
            .take(9)
            .collect::<String>();
        if digits.is_empty() {
            0
        } else {
            let value = digits.parse::<u32>()?;
            value * 10_u32.pow(9_u32.saturating_sub(digits.len() as u32))
        }
    } else {
        0
    };
    let naive = NaiveDateTime::new(date, time)
        .checked_add_signed(chrono::Duration::nanoseconds(nanos as i64))
        .ok_or_else(|| anyhow::anyhow!("NinjaTrader timestamp overflow"))?;
    let local = match timezone.from_local_datetime(&naive) {
        LocalResult::Single(value) => value,
        LocalResult::Ambiguous(first, second) => {
            let first_ns = first.with_timezone(&Utc).timestamp_nanos_opt();
            let second_ns = second.with_timezone(&Utc).timestamp_nanos_opt();
            let candidates = [
                (
                    first_ns
                        .ok_or_else(|| anyhow::anyhow!("timestamp outside nanosecond range"))?,
                    first,
                ),
                (
                    second_ns
                        .ok_or_else(|| anyhow::anyhow!("timestamp outside nanosecond range"))?,
                    second,
                ),
            ];
            candidates
                .iter()
                .filter(|(timestamp, _)| {
                    previous_timestamp_ns.is_none_or(|previous| *timestamp >= previous)
                })
                .min_by_key(|(timestamp, _)| *timestamp)
                .map(|(_, value)| *value)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ambiguous NinjaTrader timestamp cannot be made monotonic; source needs an offset"
                    )
                })?
        }
        LocalResult::None => {
            bail!("NinjaTrader timestamp is invalid in timezone (DST gap)")
        }
    };
    local
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
        .ok_or_else(|| anyhow::anyhow!("NinjaTrader timestamp outside nanosecond range"))
}

fn load_dataframe(
    df: &DataFrame,
    timestamp_unit: Option<&str>,
    allow_index_timestamps: bool,
    timezone_name: &str,
    tick_size: Option<f64>,
) -> Result<LoadedSource> {
    let open = numeric_column(df, &["open", "o"])?;
    let high = numeric_column(df, &["high", "h"])?;
    let low = numeric_column(df, &["low", "l"])?;
    let close = numeric_column(df, &["close", "c"])?;
    let volume_column = optional_numeric_column(df, &["volume", "vol", "total_volume"])?;
    let volume_present = volume_column.is_some();
    let volume = volume_column.unwrap_or_else(|| vec![0.0; close.len()]);
    let (timestamp_ns, timestamp_source, timestamp_unit, index_timestamp_fallback) =
        timestamp_column(
            df,
            close.len(),
            timestamp_unit,
            allow_index_timestamps,
            timezone_name,
        )?;
    let bars = SupervisedBars {
        open,
        high,
        low,
        close,
        volume,
        timestamp_ns,
    };
    bars.validate()?;
    Ok(LoadedSource {
        bars,
        timestamp_source,
        timestamp_unit,
        timestamp_timezone: timezone_name.to_string(),
        index_timestamp_fallback,
        raw_price_scale: None,
        tick_size,
        volume_present,
        source_row_count: df.height(),
    })
}

fn find_column<'a>(
    df: &'a DataFrame,
    names: &[&str],
) -> Result<Option<&'a polars::prelude::Column>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            return Ok(Some(column));
        }
    }
    Ok(None)
}

fn numeric_column(df: &DataFrame, names: &[&str]) -> Result<Vec<f64>> {
    let column = find_column(df, names)?
        .ok_or_else(|| anyhow::anyhow!("source is missing one of columns {names:?}"))?;
    let name = column.name().to_string();
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Float64)
        .with_context(|| format!("cast column {name} to float64"))?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at {index}"))?;
            if !value.is_finite() {
                bail!("column {name} has non-finite value at {index}");
            }
            Ok(value)
        })
        .collect()
}

fn optional_numeric_column(df: &DataFrame, names: &[&str]) -> Result<Option<Vec<f64>>> {
    let Some(column) = find_column(df, names)? else {
        return Ok(None);
    };
    numeric_column(df, &[column.name().as_str()]).map(Some)
}

/// Databento's native trade schema stores prices as fixed-point integer
/// nanounits.  Parquet exports can also already contain ordinary decimal
/// prices, so keep an explicit override while making the default conservative:
/// only values whose magnitude is clearly fixed-point are divided by 1e9.
fn databento_price_column(df: &DataFrame, explicit_scale: Option<f64>) -> Result<(Vec<f64>, f64)> {
    let prices = numeric_column(df, &["price"])?;
    let scale = if let Some(scale) = explicit_scale {
        if !scale.is_finite() || scale <= 0.0 {
            bail!("--databento-price-scale must be finite and positive");
        }
        scale
    } else {
        let max_abs = prices
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if max_abs >= 100_000_000.0 {
            1_000_000_000.0
        } else {
            1.0
        }
    };
    let scaled = prices
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value / scale;
            if !value.is_finite() || value <= 0.0 {
                bail!("raw Databento price is invalid after scale at row {index}");
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((scaled, scale))
}

fn timestamp_column(
    df: &DataFrame,
    len: usize,
    timestamp_unit: Option<&str>,
    allow_index_timestamps: bool,
    timezone_name: &str,
) -> Result<(Vec<i64>, String, String, bool)> {
    let Some(column) = find_column(
        df,
        &[
            "ts_ns",
            "timestamp_ns",
            "ts_event",
            "timestamp",
            "date",
            "datetime",
            "time",
        ],
    )?
    else {
        if !allow_index_timestamps {
            bail!(
                "source has no timestamp column; pass --allow-index-timestamps only for legacy files"
            );
        }
        return Ok((
            (0..len)
                .map(|value| (value as i64).checked_mul(1_000_000_000))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow::anyhow!("index timestamp overflow"))?,
            "index".to_string(),
            "s".to_string(),
            true,
        ));
    };
    let name = column.name().to_string();
    let series = column.as_materialized_series();
    match series.dtype() {
        DataType::Datetime(unit, _) => {
            let cast = series.cast(&DataType::Int64)?;
            let raw = cast.i64()?.into_iter();
            let factor = time_unit_factor(*unit)?;
            let values = raw
                .enumerate()
                .map(|(index, value)| {
                    let value =
                        value.ok_or_else(|| anyhow::anyhow!("timestamp null at {index}"))?;
                    value
                        .checked_mul(factor)
                        .ok_or_else(|| anyhow::anyhow!("timestamp overflow at {index}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                values,
                name,
                format!("{unit:?}").to_ascii_lowercase(),
                false,
            ))
        }
        DataType::Date => {
            let cast = series.cast(&DataType::Int32)?;
            let raw = cast.i32()?.into_iter();
            let values = raw
                .enumerate()
                .map(|(index, value)| {
                    let value = value.ok_or_else(|| anyhow::anyhow!("date null at {index}"))?;
                    (value as i64)
                        .checked_mul(86_400_000_000_000)
                        .ok_or_else(|| anyhow::anyhow!("date overflow at {index}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((values, name, "date".to_string(), false))
        }
        DataType::String => {
            let timezone = timezone_name
                .parse::<Tz>()
                .with_context(|| format!("parse timezone {timezone_name}"))?;
            let values = series
                .str()?
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let value =
                        value.ok_or_else(|| anyhow::anyhow!("timestamp null at {index}"))?;
                    parse_timestamp_text(value, timezone)
                        .with_context(|| format!("parse timestamp at {index}: {value}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((values, name, "text".to_string(), false))
        }
        dtype if is_integer(dtype) || matches!(dtype, DataType::Float32 | DataType::Float64) => {
            let Some(unit) = timestamp_unit else {
                if matches!(name.as_str(), "ts_ns" | "ts_event") {
                    return integer_timestamp_series(series, &name, "ns");
                }
                bail!("numeric {name} timestamps are ambiguous; pass --timestamp-unit ns|us|ms|s");
            };
            integer_timestamp_series(series, &name, unit)
        }
        dtype => bail!("unsupported timestamp dtype for {name}: {dtype}"),
    }
}

fn integer_timestamp_series(
    series: &polars::prelude::Series,
    name: &str,
    unit: &str,
) -> Result<(Vec<i64>, String, String, bool)> {
    let factor = match unit {
        "ns" => 1_i64,
        "us" => 1_000_i64,
        "ms" => 1_000_000_i64,
        "s" => 1_000_000_000_i64,
        other => bail!("unsupported timestamp unit `{other}`; use ns, us, ms, or s"),
    };
    let values = if is_integer(series.dtype()) {
        let cast = series.cast(&DataType::Int64)?;
        cast.i64()?
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let value = value.ok_or_else(|| anyhow::anyhow!("timestamp null at {index}"))?;
                value
                    .checked_mul(factor)
                    .ok_or_else(|| anyhow::anyhow!("timestamp overflow at {index}"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let cast = series.cast(&DataType::Float64)?;
        let factor_f64 = factor as f64;
        cast.f64()?
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let value = value.ok_or_else(|| anyhow::anyhow!("timestamp null at {index}"))?;
                let scaled = value * factor_f64;
                if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
                    bail!("timestamp overflow at {index}");
                }
                Ok(scaled.round() as i64)
            })
            .collect::<Result<Vec<_>>>()?
    };
    Ok((values, name.to_string(), unit.to_string(), false))
}

fn parse_timestamp_text(value: &str, timezone: Tz) -> Result<i64> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return parsed
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow::anyhow!("timestamp outside nanosecond range"));
    }
    let formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ];
    for format in formats {
        if let Ok(naive) = NaiveDateTime::parse_from_str(value, format) {
            let local = timezone
                .from_local_datetime(&naive)
                .single()
                .ok_or_else(|| anyhow::anyhow!("timestamp is ambiguous or invalid in timezone"))?;
            return local
                .with_timezone(&Utc)
                .timestamp_nanos_opt()
                .ok_or_else(|| anyhow::anyhow!("timestamp outside nanosecond range"));
        }
        if let Ok(date) = NaiveDate::parse_from_str(value, format) {
            let naive = date
                .and_hms_opt(0, 0, 0)
                .ok_or_else(|| anyhow::anyhow!("invalid date"))?;
            let local = timezone
                .from_local_datetime(&naive)
                .single()
                .ok_or_else(|| anyhow::anyhow!("date is ambiguous or invalid in timezone"))?;
            return local
                .with_timezone(&Utc)
                .timestamp_nanos_opt()
                .ok_or_else(|| anyhow::anyhow!("date outside nanosecond range"));
        }
    }
    bail!("unsupported timestamp text")
}

fn time_unit_factor(unit: TimeUnit) -> Result<i64> {
    Ok(match unit {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
    })
}

fn is_integer(dtype: &DataType) -> bool {
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

fn write_dataset(
    output: &Path,
    input: &Path,
    instrument: &str,
    contract: &str,
    source: &LoadedSource,
    config: &SupervisedConfig,
    events: &[SupervisedEvent],
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let config_json = serde_json::to_string(config)?;
    let source_hash = sha256_file(input)?;
    let dataset_fingerprint = dataset_fingerprint(input, source, config, events)?;
    let provenance_json = serde_json::json!({
        "source_path": input.display().to_string(),
        "timestamp_source": source.timestamp_source,
        "timestamp_unit": source.timestamp_unit,
        "timestamp_timezone": source.timestamp_timezone,
        "index_timestamp_fallback": source.index_timestamp_fallback,
        "raw_price_scale": source.raw_price_scale,
        "tick_size": source.tick_size,
        "bar_kind": config.bar_kind,
        "bar_value": config.bar_value,
        "source_row_count": source.source_row_count,
        "source_hash_sha256": source_hash,
        "dataset_fingerprint_sha256": dataset_fingerprint,
    });
    let names = config.feature_names();
    let repeated = |value: &str| vec![value.to_string(); events.len()];
    let mut columns = vec![
        Series::new("schema_version".into(), repeated(SUPERVISED_DATASET_SCHEMA)).into(),
        Series::new("label_schema".into(), repeated(SUPERVISED_LABEL_SCHEMA)).into(),
        Series::new("feature_schema".into(), repeated(&config.feature_schema())).into(),
        Series::new("config_json".into(), repeated(&config_json)).into(),
        Series::new("source_path".into(), repeated(&input.display().to_string())).into(),
        Series::new("source_hash_sha256".into(), repeated(&source_hash)).into(),
        Series::new(
            "dataset_fingerprint_sha256".into(),
            repeated(&dataset_fingerprint),
        )
        .into(),
        Series::new(
            "source_row_count".into(),
            vec![source.source_row_count as i64; events.len()],
        )
        .into(),
        Series::new(
            "source_size_bytes".into(),
            vec![input.metadata()?.len() as i64; events.len()],
        )
        .into(),
        Series::new("instrument".into(), repeated(instrument)).into(),
        Series::new("contract".into(), repeated(contract)).into(),
        Series::new("bar_kind".into(), repeated(&config.bar_kind)).into(),
        Series::new("bar_value".into(), vec![config.bar_value; events.len()]).into(),
        Series::new("pnl_currency".into(), repeated("USD")).into(),
        Series::new(
            "contract_multiplier".into(),
            vec![config.contract_multiplier; events.len()],
        )
        .into(),
        Series::new(
            "round_trip_cost".into(),
            vec![config.round_trip_cost; events.len()],
        )
        .into(),
        Series::new(
            "timestamp_source".into(),
            repeated(&source.timestamp_source),
        )
        .into(),
        Series::new("timestamp_unit".into(), repeated(&source.timestamp_unit)).into(),
        Series::new(
            "timestamp_timezone".into(),
            repeated(&source.timestamp_timezone),
        )
        .into(),
        Series::new(
            "provenance_json".into(),
            repeated(&provenance_json.to_string()),
        )
        .into(),
        Series::new(
            "index_timestamp_fallback".into(),
            vec![source.index_timestamp_fallback; events.len()],
        )
        .into(),
        Series::new(
            "raw_price_scale".into(),
            vec![source.raw_price_scale.unwrap_or(1.0); events.len()],
        )
        .into(),
        Series::new("leakage_check".into(), repeated("passed")).into(),
        Series::new(
            "event_id".into(),
            events
                .iter()
                .map(|event| event.event_id as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "session_id".into(),
            events
                .iter()
                .map(|event| event.session_id.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "session_event_index".into(),
            events
                .iter()
                .map(|event| event.session_event_index as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "row_idx".into(),
            events
                .iter()
                .map(|event| event.row_idx as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "timestamp_ns".into(),
            events
                .iter()
                .map(|event| event.timestamp_ns)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "entry_row_idx".into(),
            events
                .iter()
                .map(|event| event.row_idx.saturating_add(1) as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "raw_direction".into(),
            events
                .iter()
                .map(|event| event.raw_direction as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "decision_price".into(),
            events
                .iter()
                .map(|event| event.decision_price)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "interval_end_price".into(),
            events
                .iter()
                .map(|event| event.interval_end_price)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "interval_end_row_idx".into(),
            events
                .iter()
                .map(|event| event.interval_end_row_idx as i64)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "terminal_event".into(),
            events
                .iter()
                .map(|event| event.terminal_event)
                .collect::<Vec<_>>(),
        )
        .into(),
    ];

    for name in [
        "event_open",
        "event_high",
        "event_low",
        "event_close",
        "event_volume",
    ] {
        let values = events
            .iter()
            .map(|event| match name {
                "event_open" => source.bars.open[event.row_idx],
                "event_high" => source.bars.high[event.row_idx],
                "event_low" => source.bars.low[event.row_idx],
                "event_close" => source.bars.close[event.row_idx],
                _ => source.bars.volume[event.row_idx],
            })
            .collect::<Vec<_>>();
        columns.push(Series::new(name.into(), values).into());
    }

    for name in &names {
        if name == "raw_direction" {
            // The configured feature is also the stable structural column
            // emitted above; keep one physical column in the parquet schema.
            continue;
        }
        let values = events
            .iter()
            .map(|event| {
                event
                    .features
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("event missing feature {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        columns.push(Series::new(name.clone().into(), values).into());
    }

    columns.extend([
        Series::new(
            "action_value_normal".into(),
            events
                .iter()
                .map(|event| event.action_value_normal)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "action_value_skip".into(),
            events
                .iter()
                .map(|event| event.action_value_skip)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "action_value_invert".into(),
            events
                .iter()
                .map(|event| event.action_value_invert)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "label_action".into(),
            events
                .iter()
                .map(|event| event.label_action as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "label_name".into(),
            events
                .iter()
                .map(|event| event.label_name.clone())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "oracle_position_before".into(),
            events
                .iter()
                .map(|event| event.oracle_position_before as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "oracle_position_after".into(),
            events
                .iter()
                .map(|event| event.oracle_position_after as i32)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "oracle_value".into(),
            events
                .iter()
                .map(|event| event.oracle_value)
                .collect::<Vec<_>>(),
        )
        .into(),
    ]);
    let mut df = DataFrame::new(columns)?;
    let (temporary_path, temporary_file) = create_unique_output_file(output)?;
    let mut temporary_output = TemporaryOutput {
        path: temporary_path,
        committed: false,
    };
    // The destination is checked again immediately before commit. This keeps
    // a pre-existing result intact even if another process creates the name
    // while the Parquet writer is running.
    ParquetWriter::new(temporary_file).finish(&mut df)?;
    publish_noreplace(&temporary_output.path, output)
        .with_context(|| format!("atomically commit supervised dataset {}", output.display()))?;
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
                    format!(
                        "create temporary supervised dataset {}",
                        candidate.display()
                    )
                });
            }
        }
    }
    bail!(
        "could not allocate a unique temporary supervised dataset beside {}",
        output.display()
    )
}

fn publish_noreplace(temporary: &Path, destination: &Path) -> Result<()> {
    // `rename` is not safe here: on Unix it replaces an existing destination,
    // and the behavior is not a portable no-replace publication primitive.
    // A same-directory hard link is created atomically and fails with
    // AlreadyExists if any directory entry (including a symlink) won the race.
    match std::fs::hard_link(temporary, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // The caller normally owns this path through TemporaryOutput, but
            // clean it here too so this primitive is safe to use on its own.
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

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn dataset_fingerprint(
    input: &Path,
    source: &LoadedSource,
    config: &SupervisedConfig,
    events: &[SupervisedEvent],
) -> Result<String> {
    let source_hash = sha256_file(input)?;
    let config_json = serde_json::to_string(config)?;
    let mut digest = Sha256::new();
    digest.update(source_hash.as_bytes());
    digest.update([0]);
    digest.update(config_json.as_bytes());
    digest.update([0]);
    digest.update(source.timestamp_source.as_bytes());
    digest.update([0]);
    digest.update(source.timestamp_unit.as_bytes());
    digest.update([0]);
    digest.update(source.timestamp_timezone.as_bytes());
    digest.update([0]);
    digest.update([u8::from(source.index_timestamp_fallback)]);
    digest.update([0]);
    digest.update(source.raw_price_scale.unwrap_or(1.0).to_le_bytes());
    digest.update((source.source_row_count as u64).to_le_bytes());
    digest.update((events.len() as u64).to_le_bytes());
    for event in events {
        digest.update(serde_json::to_vec(event)?);
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn train(args: TrainArgs) -> Result<()> {
    supervised_train::run_train(args)
}

fn evaluate(args: EvaluateArgs) -> Result<()> {
    supervised_train::run_evaluate(args)
}

fn probe(args: ProbeArgs) -> Result<()> {
    let stack = ml::resolve_training_stack(TrainerKind::Supervised, &args.backend, &args.device)?;
    let summary = serde_json::json!({
        "schema_version": "supervised-device-probe-v1",
        "requested_backend": stack.backend,
        "requested_device": stack.requested_runtime,
        "effective_device": stack.effective_runtime,
        "cargo_feature": stack.cargo_feature,
        "implementation_status": stack.implementation_status,
        "supported": stack.is_implemented(),
        "notes": stack.notes,
        "lookahead_guard": "feature columns are written separately from labels and future price columns"
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::Series;
    use std::collections::BTreeMap;

    fn test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "midas-supervised-{name}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn parses_naive_ninjatrader_timestamp_in_configured_timezone() {
        let tz = "America/New_York".parse::<Tz>().unwrap();
        let value = parse_timestamp_text("2026-08-13 18:00:00", tz).unwrap();
        let expected = DateTime::parse_from_rfc3339("2026-08-13T22:00:00Z")
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        assert_eq!(value, expected);
    }

    #[test]
    fn rejects_ambiguous_numeric_timestamp_without_unit() {
        let df = DataFrame::new(vec![
            Series::new("timestamp".into(), &[1_i64, 2_i64]).into(),
        ])
        .unwrap();
        let error = timestamp_column(&df, 2, None, false, "America/New_York").unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn preserves_integer_ts_event_nanoseconds_without_float_rounding() {
        let expected = 1_775_433_600_123_456_789_i64;
        let df = DataFrame::new(vec![Series::new("ts_event".into(), &[expected]).into()]).unwrap();
        let (values, source, unit, used_index) =
            timestamp_column(&df, 1, None, false, "America/New_York").unwrap();
        assert_eq!(values, vec![expected]);
        assert_eq!(source, "ts_event");
        assert_eq!(unit, "ns");
        assert!(!used_index);
    }

    #[test]
    fn auto_scales_fixed_point_databento_prices() {
        let df = DataFrame::new(vec![
            Series::new("price".into(), &[5_198_700_000_000_i64, 5_198_800_000_000]).into(),
        ])
        .unwrap();
        let (prices, scale) = databento_price_column(&df, None).unwrap();
        assert_eq!(scale, 1_000_000_000.0);
        assert!((prices[0] - 5_198.7).abs() < 1e-9);
    }

    #[test]
    fn range_aggregation_emits_replay_style_boundaries() {
        let rows = aggregate_range_trades(
            &[1, 2, 3, 4],
            &[100.0, 100.5, 101.0, 102.5],
            &[1.0, 1.0, 1.0, 1.0],
            0.5,
            2.0,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].1, 100.0);
        assert_eq!(rows[0].4, 101.0);
        assert_eq!(rows[0].5, 3.0);
        assert_eq!(rows[1].4, 102.0);
        assert_eq!(rows[1].5, 1.0);
        assert_eq!(rows[2].4, 102.5);
        assert!(rows.windows(2).all(|window| window[1].0 > window[0].0));
    }

    #[test]
    fn disambiguates_ninjatrader_fall_back_using_source_order() {
        let timezone = "America/New_York".parse::<Tz>().unwrap();
        let first = parse_ninja_last_timestamp("20261101 015959", timezone, None).unwrap();
        let repeated =
            parse_ninja_last_timestamp("20261101 010000", timezone, Some(first)).unwrap();
        assert!(repeated > first);
    }

    #[test]
    fn rejects_existing_output_without_clobbering_it() {
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

    #[test]
    fn temporary_output_is_same_directory_and_cleans_up_on_error() {
        let directory = test_directory("temporary-output");
        let output = directory.join("dataset.parquet");
        let (temporary_path, temporary_file) = create_unique_output_file(&output).unwrap();
        drop(temporary_file);
        assert_eq!(temporary_path.parent(), output.parent());
        assert!(temporary_path.exists());
        {
            let _temporary_output = TemporaryOutput {
                path: temporary_path.clone(),
                committed: false,
            };
        }
        assert!(!temporary_path.exists());
        assert!(!output.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn parquet_commit_is_atomic_and_preserves_existing_destination() {
        let directory = test_directory("parquet-commit");
        let input = directory.join("input.csv");
        let existing = directory.join("existing.parquet");
        let committed = directory.join("committed.parquet");
        std::fs::write(&input, b"source").unwrap();
        std::fs::write(&existing, b"keep-this").unwrap();

        let config = SupervisedConfig::default();
        let feature_names = config
            .feature_names()
            .into_iter()
            .filter(|name| name != "raw_direction")
            .map(|name| (name, 0.0))
            .collect::<BTreeMap<_, _>>();
        let source = LoadedSource {
            bars: SupervisedBars {
                open: vec![1.0, 1.0],
                high: vec![1.0, 1.0],
                low: vec![1.0, 1.0],
                close: vec![1.0, 1.0],
                volume: vec![1.0, 1.0],
                timestamp_ns: vec![1, 2],
            },
            timestamp_source: "test".to_string(),
            timestamp_unit: "ns".to_string(),
            timestamp_timezone: "UTC".to_string(),
            index_timestamp_fallback: false,
            raw_price_scale: None,
            tick_size: None,
            volume_present: true,
            source_row_count: 2,
        };
        let event = SupervisedEvent {
            event_id: 0,
            session_id: "session".to_string(),
            session_event_index: 0,
            row_idx: 0,
            timestamp_ns: 1,
            raw_direction: 1,
            decision_price: 1.0,
            interval_end_price: 1.0,
            interval_end_row_idx: 1,
            terminal_event: true,
            features: feature_names,
            action_value_normal: 0.0,
            action_value_skip: 0.0,
            action_value_invert: 0.0,
            label_action: 0,
            label_name: "normal".to_string(),
            oracle_position_before: 0,
            oracle_position_after: 0,
            oracle_value: 0.0,
        };

        let error = write_dataset(
            &existing,
            &input,
            "TEST",
            "TEST",
            &source,
            &config,
            std::slice::from_ref(&event),
        )
        .unwrap_err();
        assert!(error.to_string().contains("atomically commit"));
        assert!(format!("{error:#}").contains("refusing to clobber"));
        assert_eq!(std::fs::read(&existing).unwrap(), b"keep-this");
        let leftover_temporary_files = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-"));
        assert!(!leftover_temporary_files);

        write_dataset(
            &committed,
            &input,
            "TEST",
            "TEST",
            &source,
            &config,
            std::slice::from_ref(&event),
        )
        .unwrap();
        assert!(committed.exists());
        let leftover_temporary_files = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-"));
        assert!(!leftover_temporary_files);
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
}
