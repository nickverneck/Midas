use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use clap::Parser;
use midas_env::meta_gate::{
    META_EVENT_SCHEMA, MetaGateConfig, canonical_dataset_fingerprint_sha256,
    canonical_event_payload_sha256, extract_events, meta_events_from_frame, meta_events_to_frame,
    open_meta_source_snapshot,
};
use polars::prelude::{
    DataFrame, DataType, NamedFrom, ParquetReader, ParquetWriter, SerReader, SerWriter, Series,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const NANOS_PER_SECOND: i64 = 1_000_000_000;
const CAUSAL_FEATURES: [&str; 11] = [
    "raw_direction",
    "trigger_spread_atr",
    "context_spread_atr",
    "trigger_fast_slope_atr",
    "trigger_slow_slope_atr",
    "context_fast_slope_atr",
    "context_slow_slope_atr",
    "efficiency_ratio",
    "rvol_20",
    "return_1",
    "return_5",
];

#[derive(Debug, Clone, Serialize)]
struct SourceFingerprint {
    source_path: String,
    source_root: String,
    source_size_bytes: i64,
    source_row_count: i64,
    timestamp_source: String,
    source_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetProvenance {
    schema_version: String,
    feature_schema: String,
    instrument: String,
    contract: String,
    config_json: String,
    source_path: String,
    source_root: String,
    source_size_bytes: i64,
    source_row_count: i64,
    timestamp_source: String,
    source_sha256: String,
    event_payload_sha256: String,
    dataset_fingerprint_sha256: String,
    source_fingerprint: SourceFingerprint,
}

/// Evaluate fixed event policies against a prepared meta-gate parquet file.
///
/// The `*_pnl` columns are fixed-horizon diagnostic outcomes written by
/// `prepare_meta_dataset`; they are not a sequential position-aware replay.
#[derive(Debug, Parser)]
#[command(
    about = "Evaluate fixed normal/skip/invert policies on a meta-gate event parquet",
    long_about = "Evaluate fixed normal/skip/invert policies on a meta-gate event parquet.\n\nThe normal_pnl, skip_pnl, and invert_pnl columns are treated as fixed-horizon diagnostic outcomes. This command does not replay positions, overlapping trades, protections, or lifecycle transitions. The oracle-label policy is hindsight-only: it selects the largest of the three stored outcomes for each event.\n\nPer-day grouping interprets timestamp_ns as Unix nanoseconds and reports UTC calendar days."
)]
struct Args {
    /// Event parquet produced by prepare_meta_dataset.
    #[arg(long, value_name = "PATH")]
    input: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    AlwaysNormal,
    AlwaysSkip,
    AlwaysInvert,
    OracleLabel,
}

impl Policy {
    const ALL: [Self; 4] = [
        Self::AlwaysNormal,
        Self::AlwaysSkip,
        Self::AlwaysInvert,
        Self::OracleLabel,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::AlwaysNormal => "always-normal",
            Self::AlwaysSkip => "always-skip",
            Self::AlwaysInvert => "always-invert",
            Self::OracleLabel => "oracle-label",
        }
    }
}

impl fmt::Display for Policy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedAction {
    Normal,
    Skip,
    Invert,
}

#[derive(Debug, Clone, Copy)]
struct EventOutcome {
    timestamp_ns: i64,
    normal_pnl: f64,
    skip_pnl: f64,
    invert_pnl: f64,
}

#[derive(Debug, Clone, Copy)]
struct SelectedOutcome {
    action: SelectedAction,
    pnl: f64,
}

#[derive(Debug, Clone)]
struct Summary {
    events: usize,
    normal_events: usize,
    skip_events: usize,
    invert_events: usize,
    sum_pnl: f64,
    mean_pnl: f64,
    win_rate: f64,
    max_drawdown: f64,
}

#[derive(Debug, Clone)]
struct DaySummary {
    day: String,
    summary: Summary,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (events, provenance) = load_events(&args.input)?;
    if events.is_empty() {
        bail!("event parquet {} contains no rows", args.input.display());
    }

    println!(
        "Fixed-horizon diagnostic outcomes: normal_pnl, skip_pnl, and invert_pnl are read directly from the event parquet; no sequential position lifecycle or overlap handling is performed."
    );
    println!(
        "Oracle-label is hindsight-only and selects the largest stored outcome per event (ties prefer skip, then normal, then invert)."
    );
    println!(
        "Input: {}\nEvents: {}\nPer-day timestamps: timestamp_ns interpreted as Unix nanoseconds in UTC.",
        args.input.display(),
        events.len()
    );
    println!(
        "Dataset provenance: instrument={} contract={} feature_schema={} timestamp_source={} source_rows={}",
        provenance.instrument,
        provenance.contract,
        provenance.feature_schema,
        provenance.timestamp_source,
        provenance.source_row_count
    );

    print_overall(&events);
    print_per_day(&events)?;
    Ok(())
}

fn load_events(path: &PathBuf) -> Result<(Vec<EventOutcome>, DatasetProvenance)> {
    let file = File::open(path).with_context(|| format!("open input {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("read event parquet {}", path.display()))?;
    if df.height() == 0 {
        bail!("event parquet {} contains no rows", path.display());
    }
    let provenance = validate_event_provenance(&df)?;
    validate_dataset_integrity(&df, &provenance)?;
    let timestamp_ns = required_i64(&df, "timestamp_ns")?;
    let row_idx = required_i64(&df, "row_idx")?;
    let entry_row_idx = required_i64(&df, "entry_row_idx")?;
    let horizon_row_idx = required_i64(&df, "horizon_row_idx")?;
    let event_id = required_i64(&df, "event_id")?;
    let raw_direction = required_i64(&df, "raw_direction")?;
    for name in ["open", "high", "low", "close", "volume", "entry_price"] {
        required_f64(&df, name)?;
    }
    for name in CAUSAL_FEATURES {
        if name != "raw_direction" {
            required_f64(&df, name)?;
        }
    }
    let oracle_actions = required_strings(&df, "oracle_action")?;
    let normal_pnl = required_f64(&df, "normal_pnl")?;
    let skip_pnl = required_f64(&df, "skip_pnl")?;
    let invert_pnl = required_f64(&df, "invert_pnl")?;

    let len = df.height();
    for (name, values) in [
        ("timestamp_ns", timestamp_ns.len()),
        ("event_id", event_id.len()),
        ("row_idx", row_idx.len()),
        ("entry_row_idx", entry_row_idx.len()),
        ("horizon_row_idx", horizon_row_idx.len()),
        ("raw_direction", raw_direction.len()),
        ("oracle_action", oracle_actions.len()),
        ("normal_pnl", normal_pnl.len()),
        ("skip_pnl", skip_pnl.len()),
        ("invert_pnl", invert_pnl.len()),
    ] {
        if values != len {
            bail!("column {name} has {values} values but event parquet has {len} rows");
        }
    }

    if timestamps_strictly_not_increasing(&timestamp_ns) {
        bail!("timestamp_ns must be strictly increasing for chronological evaluation");
    }
    if timestamps_strictly_not_increasing(&row_idx) {
        bail!("row_idx must be strictly increasing for chronological evaluation");
    }
    for row in 0..len {
        if entry_row_idx[row] <= row_idx[row] {
            bail!(
                "entry_row_idx must be after row_idx at event row {row}: {} <= {}",
                entry_row_idx[row],
                row_idx[row]
            );
        }
        if horizon_row_idx[row] < entry_row_idx[row] {
            bail!(
                "horizon_row_idx must be at or after entry_row_idx at event row {row}: {} < {}",
                horizon_row_idx[row],
                entry_row_idx[row]
            );
        }
        if !matches!(raw_direction[row], -1 | 1) {
            bail!(
                "raw_direction must be -1 or 1 at event row {row}, got {}",
                raw_direction[row]
            );
        }
        if !matches!(oracle_actions[row].as_str(), "normal" | "skip" | "invert") {
            bail!(
                "oracle_action has invalid value {:?} at event row {row}",
                oracle_actions[row]
            );
        }
    }

    let events = timestamp_ns
        .into_iter()
        .zip(normal_pnl)
        .zip(skip_pnl)
        .zip(invert_pnl)
        .enumerate()
        .map(
            |(_, (((timestamp_ns, normal_pnl), skip_pnl), invert_pnl))| EventOutcome {
                timestamp_ns,
                normal_pnl,
                skip_pnl,
                invert_pnl,
            },
        )
        .collect::<Vec<_>>();

    Ok((events, provenance))
}

fn validate_event_provenance(frame: &DataFrame) -> Result<DatasetProvenance> {
    let schema_version = required_consistent_string(frame, "schema_version")?;
    if schema_version != META_EVENT_SCHEMA {
        bail!(
            "unsupported event parquet schema_version {schema_version:?}; expected {META_EVENT_SCHEMA:?}"
        );
    }
    let feature_schema = required_consistent_string(frame, "feature_schema")?;
    let instrument = required_consistent_string(frame, "instrument")?;
    let contract = required_consistent_string(frame, "contract")?;
    let config_json = required_consistent_string(frame, "config_json")?;
    let source_path = required_consistent_string(frame, "source_path")?;
    let source_root = required_consistent_string(frame, "source_root")?;
    let source_size_bytes = required_consistent_i64(frame, "source_size_bytes")?;
    let source_row_count = required_consistent_i64(frame, "source_row_count")?;
    let timestamp_source = required_consistent_string(frame, "timestamp_source")?;
    let source_sha256 = required_consistent_string(frame, "source_sha256")?;
    let event_payload_sha256 = required_consistent_string(frame, "event_payload_sha256")?;
    let dataset_fingerprint_sha256 =
        required_consistent_string(frame, "dataset_fingerprint_sha256")?;

    if source_size_bytes < 0 {
        bail!("source_size_bytes must be non-negative, got {source_size_bytes}");
    }
    if source_row_count <= 0 {
        bail!("source_row_count must be positive, got {source_row_count}");
    }
    for (name, value) in [
        ("source_sha256", &source_sha256),
        ("event_payload_sha256", &event_payload_sha256),
        ("dataset_fingerprint_sha256", &dataset_fingerprint_sha256),
    ] {
        if value.len() != 64
            || value.bytes().any(|byte| !byte.is_ascii_hexdigit())
            || *value != value.to_ascii_lowercase()
        {
            bail!("provenance column {name} must contain a lowercase SHA-256 digest");
        }
    }
    let config: MetaGateConfig = serde_json::from_str(&config_json)
        .context("config_json is not valid MetaGateConfig JSON")?;
    config
        .validate()
        .context("config_json contains an invalid meta-gate configuration")?;
    let expected_feature_schema = config.feature_schema();
    if feature_schema != expected_feature_schema {
        bail!(
            "feature_schema {feature_schema:?} does not match config_json-derived schema {expected_feature_schema:?}"
        );
    }

    Ok(DatasetProvenance {
        schema_version,
        feature_schema,
        instrument,
        contract,
        config_json,
        source_path: source_path.clone(),
        source_root: source_root.clone(),
        source_size_bytes,
        source_row_count,
        timestamp_source: timestamp_source.clone(),
        source_sha256: source_sha256.clone(),
        event_payload_sha256,
        dataset_fingerprint_sha256,
        source_fingerprint: SourceFingerprint {
            source_path,
            source_root,
            source_size_bytes,
            source_row_count,
            timestamp_source,
            source_sha256,
        },
    })
}

fn validate_dataset_integrity(frame: &DataFrame, provenance: &DatasetProvenance) -> Result<()> {
    let stored_events = meta_events_from_frame(frame)
        .context("decode event payload for integrity validation; reprepare the dataset")?;
    let config: MetaGateConfig = serde_json::from_str(&provenance.config_json)
        .context("decode trusted meta-gate config; reprepare the dataset")?;
    let snapshot = open_meta_source_snapshot(
        PathBuf::from(&provenance.source_path).as_path(),
        PathBuf::from(&provenance.source_root).as_path(),
        false,
        Some(&provenance.timestamp_source),
    )
    .context("open trusted source snapshot; restore the source or reprepare the dataset")?;

    if snapshot.canonical_path.display().to_string() != provenance.source_path
        || snapshot.canonical_root.display().to_string() != provenance.source_root
    {
        bail!("source path/root are not canonical; reprepare the dataset");
    }
    if snapshot.size_bytes != provenance.source_size_bytes {
        bail!(
            "meta-gate source size changed: expected {}, found {}; reprepare the dataset",
            provenance.source_size_bytes,
            snapshot.size_bytes
        );
    }
    if i64::try_from(snapshot.bars.close.len())? != provenance.source_row_count {
        bail!(
            "meta-gate source row count changed: expected {}, found {}; reprepare the dataset",
            provenance.source_row_count,
            snapshot.bars.close.len()
        );
    }
    if snapshot.sha256 != provenance.source_sha256 {
        bail!(
            "meta-gate source integrity check failed: expected {}, computed {}; reprepare the dataset",
            provenance.source_sha256,
            snapshot.sha256
        );
    }

    let regenerated_events = extract_events(&snapshot.bars, &config)
        .context("regenerate events from trusted source; reprepare the dataset")?;
    if stored_events != regenerated_events {
        bail!(
            "meta-gate event payload does not match events regenerated from the trusted source; reprepare the dataset"
        );
    }
    let computed_event_payload_sha256 = canonical_event_payload_sha256(&regenerated_events);
    if computed_event_payload_sha256 != provenance.event_payload_sha256 {
        bail!(
            "meta-gate event payload integrity check failed: expected {}, computed {}; reprepare the dataset",
            provenance.event_payload_sha256,
            computed_event_payload_sha256
        );
    }

    let computed_dataset_fingerprint_sha256 = canonical_dataset_fingerprint_sha256(
        &snapshot.sha256,
        &computed_event_payload_sha256,
        &provenance.feature_schema,
        &provenance.instrument,
        &provenance.contract,
        &provenance.config_json,
        &provenance.source_path,
        &provenance.source_root,
        provenance.source_size_bytes,
        provenance.source_row_count,
        &provenance.timestamp_source,
    );
    if computed_dataset_fingerprint_sha256 != provenance.dataset_fingerprint_sha256 {
        bail!(
            "meta-gate dataset fingerprint integrity check failed: expected {}, computed {}; reprepare the dataset",
            provenance.dataset_fingerprint_sha256,
            computed_dataset_fingerprint_sha256
        );
    }
    Ok(())
}

fn required_consistent_string(frame: &DataFrame, name: &str) -> Result<String> {
    let values = required_strings(frame, name)?;
    let Some(first) = values.first() else {
        bail!("input parquet contains no rows while validating {name}");
    };
    if first.trim().is_empty() {
        bail!("provenance column {name} must be non-empty at row 0");
    }
    if let Some((row, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| value.trim().is_empty())
    {
        bail!("provenance column {name} must be non-empty at row {row}");
    }
    if values.iter().any(|value| value != first) {
        bail!("provenance column {name} is inconsistent across event rows");
    }
    Ok(first.clone())
}

fn required_consistent_i64(frame: &DataFrame, name: &str) -> Result<i64> {
    let values = required_i64(frame, name)?;
    let Some(first) = values.first() else {
        bail!("input parquet contains no rows while validating {name}");
    };
    if values.iter().any(|value| value != first) {
        bail!("provenance column {name} is inconsistent across event rows");
    }
    Ok(*first)
}

fn required_strings(frame: &DataFrame, name: &str) -> Result<Vec<String>> {
    let column = frame
        .column(name)
        .with_context(|| format!("event parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::String)
        .with_context(|| format!("cast {name} to String"))?;
    cast.str()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow::anyhow!("column {name} has null at row {index}"))
        })
        .collect()
}

fn timestamps_strictly_not_increasing(values: &[i64]) -> bool {
    values.windows(2).any(|window| window[1] <= window[0])
}

fn required_i64(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = df
        .column(name)
        .with_context(|| format!("event parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Int64)
        .with_context(|| format!("cast {name} to Int64"))?;
    cast.i64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {index}"))
        })
        .collect()
}

fn required_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let column = df
        .column(name)
        .with_context(|| format!("event parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Float64)
        .with_context(|| format!("cast {name} to Float64"))?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {index}"))?;
            if !value.is_finite() {
                bail!("column {name} has non-finite value at row {index}");
            }
            Ok(value)
        })
        .collect()
}

fn select_outcome(policy: Policy, event: EventOutcome) -> SelectedOutcome {
    match policy {
        Policy::AlwaysNormal => SelectedOutcome {
            action: SelectedAction::Normal,
            pnl: event.normal_pnl,
        },
        Policy::AlwaysSkip => SelectedOutcome {
            action: SelectedAction::Skip,
            pnl: event.skip_pnl,
        },
        Policy::AlwaysInvert => SelectedOutcome {
            action: SelectedAction::Invert,
            pnl: event.invert_pnl,
        },
        Policy::OracleLabel => oracle_outcome(event.normal_pnl, event.skip_pnl, event.invert_pnl),
    }
}

fn oracle_outcome(normal_pnl: f64, skip_pnl: f64, invert_pnl: f64) -> SelectedOutcome {
    // Start with skip so a tie at zero (or any other equal outcome) does not
    // manufacture an unnecessary trade. Strict comparisons make the tie
    // order deterministic: skip, then normal, then invert.
    let mut best = SelectedOutcome {
        action: SelectedAction::Skip,
        pnl: skip_pnl,
    };
    if normal_pnl > best.pnl {
        best = SelectedOutcome {
            action: SelectedAction::Normal,
            pnl: normal_pnl,
        };
    }
    if invert_pnl > best.pnl {
        best = SelectedOutcome {
            action: SelectedAction::Invert,
            pnl: invert_pnl,
        };
    }
    best
}

fn summarize(policy: Policy, events: &[EventOutcome]) -> Summary {
    let mut pnls = Vec::with_capacity(events.len());
    let mut normal_events = 0;
    let mut skip_events = 0;
    let mut invert_events = 0;
    for &event in events {
        let selected = select_outcome(policy, event);
        match selected.action {
            SelectedAction::Normal => normal_events += 1,
            SelectedAction::Skip => skip_events += 1,
            SelectedAction::Invert => invert_events += 1,
        }
        pnls.push(selected.pnl);
    }
    summarize_pnls(pnls, normal_events, skip_events, invert_events)
}

fn summarize_pnls(
    pnls: Vec<f64>,
    normal_events: usize,
    skip_events: usize,
    invert_events: usize,
) -> Summary {
    let events = pnls.len();
    let sum_pnl = pnls.iter().sum::<f64>();
    let mean_pnl = if events == 0 {
        0.0
    } else {
        sum_pnl / events as f64
    };
    let wins = pnls.iter().filter(|&&pnl| pnl > 0.0).count();
    let win_rate = if events == 0 {
        0.0
    } else {
        wins as f64 / events as f64
    };
    Summary {
        events,
        normal_events,
        skip_events,
        invert_events,
        sum_pnl,
        mean_pnl,
        win_rate,
        max_drawdown: max_drawdown(&pnls),
    }
}

fn max_drawdown(pnls: &[f64]) -> f64 {
    let mut cumulative: f64 = 0.0;
    let mut peak: f64 = 0.0;
    let mut drawdown: f64 = 0.0;
    for &pnl in pnls {
        cumulative += pnl;
        peak = peak.max(cumulative);
        drawdown = drawdown.max(peak - cumulative);
    }
    drawdown
}

fn day_key(timestamp_ns: i64) -> Result<String> {
    let seconds = timestamp_ns.div_euclid(NANOS_PER_SECOND);
    let nanos = timestamp_ns.rem_euclid(NANOS_PER_SECOND) as u32;
    let timestamp = DateTime::<Utc>::from_timestamp(seconds, nanos)
        .ok_or_else(|| anyhow::anyhow!("timestamp_ns {timestamp_ns} is outside chrono range"))?;
    Ok(timestamp.date_naive().to_string())
}

fn daily_summaries(policy: Policy, events: &[EventOutcome]) -> Result<Vec<DaySummary>> {
    let mut grouped: BTreeMap<String, Vec<EventOutcome>> = BTreeMap::new();
    for &event in events {
        grouped
            .entry(day_key(event.timestamp_ns)?)
            .or_default()
            .push(event);
    }

    grouped
        .into_iter()
        .map(|(day, events)| {
            let summary = summarize(policy, &events);
            Ok(DaySummary { day, summary })
        })
        .collect()
}

fn print_overall(events: &[EventOutcome]) {
    println!("\nOverall policy metrics");
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>14} {:>14} {:>10} {:>14}",
        "policy",
        "events",
        "normal",
        "skip",
        "invert",
        "sum_pnl",
        "mean_pnl",
        "win_rate",
        "max_drawdown"
    );
    for policy in Policy::ALL {
        print_summary(policy.label(), &summarize(policy, events));
    }
}

fn print_summary(label: &str, summary: &Summary) {
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>14.4} {:>14.4} {:>9.2}% {:>14.4}",
        label,
        summary.events,
        summary.normal_events,
        summary.skip_events,
        summary.invert_events,
        summary.sum_pnl,
        summary.mean_pnl,
        summary.win_rate * 100.0,
        summary.max_drawdown
    );
}

fn print_per_day(events: &[EventOutcome]) -> Result<()> {
    println!("\nPer-day policy metrics (UTC)");
    println!(
        "{:<15} {:<12} {:>8} {:>14} {:>14} {:>10} {:>14}",
        "policy", "day", "events", "sum_pnl", "mean_pnl", "win_rate", "max_drawdown"
    );
    for policy in Policy::ALL {
        for daily in daily_summaries(policy, events)? {
            println!(
                "{:<15} {:<12} {:>8} {:>14.4} {:>14.4} {:>9.2}% {:>14.4}",
                policy.label(),
                daily.day,
                daily.summary.events,
                daily.summary.sum_pnl,
                daily.summary.mean_pnl,
                daily.summary.win_rate * 100.0,
                daily.summary.max_drawdown
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(normal_pnl: f64, skip_pnl: f64, invert_pnl: f64) -> EventOutcome {
        EventOutcome {
            timestamp_ns: 0,
            normal_pnl,
            skip_pnl,
            invert_pnl,
        }
    }

    #[test]
    fn cumulative_drawdown_starts_from_zero_and_tracks_peak_to_trough() {
        assert_eq!(max_drawdown(&[10.0, -4.0, -8.0, 6.0]), 12.0);
        assert_eq!(max_drawdown(&[-5.0, 2.0]), 5.0);
        assert_eq!(max_drawdown(&[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn fixed_policies_select_their_named_outcome() {
        let event = event(2.0, 0.0, -3.0);
        assert_eq!(
            select_outcome(Policy::AlwaysNormal, event).action,
            SelectedAction::Normal
        );
        assert_eq!(
            select_outcome(Policy::AlwaysSkip, event).action,
            SelectedAction::Skip
        );
        assert_eq!(
            select_outcome(Policy::AlwaysInvert, event).action,
            SelectedAction::Invert
        );
    }

    #[test]
    fn oracle_selects_best_outcome_and_prefers_skip_on_ties() {
        assert_eq!(oracle_outcome(2.0, 0.0, 1.0).action, SelectedAction::Normal);
        assert_eq!(
            oracle_outcome(-2.0, 0.0, 3.0).action,
            SelectedAction::Invert
        );
        assert_eq!(oracle_outcome(-1.0, 0.0, -2.0).action, SelectedAction::Skip);
        assert_eq!(oracle_outcome(1.0, 1.0, 1.0).action, SelectedAction::Skip);
    }

    #[test]
    fn day_key_handles_negative_epoch_nanos() {
        assert_eq!(day_key(-1).unwrap(), "1969-12-31");
        assert_eq!(day_key(0).unwrap(), "1970-01-01");
    }

    #[test]
    fn evaluation_rejects_tampered_payload_even_when_hashes_are_rewritten() {
        let (directory, mut frame) = integrity_fixture();
        let provenance = validate_event_provenance(&frame).unwrap();
        let mut tampered_events = meta_events_from_frame(&frame).unwrap();
        tampered_events[0].normal_pnl += 123.0;
        frame
            .with_column(Series::new(
                "normal_pnl".into(),
                tampered_events
                    .iter()
                    .map(|event| event.normal_pnl)
                    .collect::<Vec<_>>(),
            ))
            .unwrap();
        let tampered_event_hash = canonical_event_payload_sha256(&tampered_events);
        let tampered_fingerprint = canonical_dataset_fingerprint_sha256(
            &provenance.source_sha256,
            &tampered_event_hash,
            &provenance.feature_schema,
            &provenance.instrument,
            &provenance.contract,
            &provenance.config_json,
            &provenance.source_path,
            &provenance.source_root,
            provenance.source_size_bytes,
            provenance.source_row_count,
            &provenance.timestamp_source,
        );
        frame
            .with_column(Series::new(
                "event_payload_sha256".into(),
                vec![tampered_event_hash; frame.height()],
            ))
            .unwrap();
        frame
            .with_column(Series::new(
                "dataset_fingerprint_sha256".into(),
                vec![tampered_fingerprint; frame.height()],
            ))
            .unwrap();

        let provenance = validate_event_provenance(&frame).unwrap();
        let error = validate_dataset_integrity(&frame, &provenance).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("regenerated from the trusted source")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    fn integrity_fixture() -> (PathBuf, DataFrame) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "midas-evaluate-meta-gate-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let source_path = directory.join("bars.parquet");
        let close = vec![
            100.0, 99.0, 98.0, 98.5, 100.0, 102.0, 101.0, 99.0, 98.0, 99.0, 101.0, 103.0, 102.0,
            100.0, 99.0, 100.0,
        ];
        let mut source = DataFrame::new(vec![
            Series::new("open".into(), close.clone()).into(),
            Series::new(
                "high".into(),
                close.iter().map(|value| value + 0.5).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "low".into(),
                close.iter().map(|value| value - 0.5).collect::<Vec<_>>(),
            )
            .into(),
            Series::new("close".into(), close).into(),
            Series::new("volume".into(), vec![1.0; 16]).into(),
            Series::new("ts_ns".into(), (0_i64..16).collect::<Vec<_>>()).into(),
        ])
        .unwrap();
        ParquetWriter::new(File::create(&source_path).unwrap())
            .finish(&mut source)
            .unwrap();

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
        let snapshot = open_meta_source_snapshot(&source_path, &directory, false, None).unwrap();
        let events = extract_events(&snapshot.bars, &config).unwrap();
        let event_hash = canonical_event_payload_sha256(&events);
        let config_json = serde_json::to_string(&config).unwrap();
        let fingerprint = canonical_dataset_fingerprint_sha256(
            &snapshot.sha256,
            &event_hash,
            &config.feature_schema(),
            "GC",
            "GCZ6",
            &config_json,
            &snapshot.canonical_path.display().to_string(),
            &snapshot.canonical_root.display().to_string(),
            snapshot.size_bytes,
            snapshot.bars.close.len() as i64,
            &snapshot.timestamp_source,
        );
        let mut frame = meta_events_to_frame(&events).unwrap();
        let row_count = frame.height();
        let strings = |value: &str| vec![value.to_string(); row_count];
        for (name, value) in [
            ("schema_version", META_EVENT_SCHEMA.to_string()),
            ("feature_schema", config.feature_schema()),
            ("config_json", config_json),
            ("instrument", "GC".to_string()),
            ("contract", "GCZ6".to_string()),
            ("source_path", snapshot.canonical_path.display().to_string()),
            ("source_root", snapshot.canonical_root.display().to_string()),
            ("timestamp_source", snapshot.timestamp_source.clone()),
            ("source_sha256", snapshot.sha256.clone()),
            ("event_payload_sha256", event_hash),
            ("dataset_fingerprint_sha256", fingerprint),
        ] {
            frame
                .with_column(Series::new(name.into(), strings(&value)))
                .unwrap();
        }
        frame
            .with_column(Series::new(
                "source_size_bytes".into(),
                vec![snapshot.size_bytes; row_count],
            ))
            .unwrap();
        frame
            .with_column(Series::new(
                "source_row_count".into(),
                vec![snapshot.bars.close.len() as i64; row_count],
            ))
            .unwrap();
        (directory, frame)
    }
}
