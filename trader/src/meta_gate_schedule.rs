//! Read-only inspection of a trained crossover meta-gate policy.
//!
//! This command intentionally lives outside the engine, broker, and TUI
//! paths.  It reads the parent Midas event parquet and policy artifact,
//! reconstructs the small linear policy, and prints only a bounded schedule.
//! It does not load `AppConfig`, start an engine, open an account, or route an
//! order.

#![cfg(feature = "replay")]

use crate::cli::{MetaGateInitialSide, MetaGateScheduleArgs, MetaGateScheduleFormat};
use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

const MAX_OUTPUT_ROWS: usize = 100_000;
const PARQUET_BATCH_ROWS: usize = 8_192;
const META_EVENT_SCHEMA: &str = "meta-event-v1";
const META_GATE_POLICY_SCHEMA: &str = "meta-gate-policy-v1";
const META_GATE_ARTIFACT_MANIFEST_SCHEMA: &str = "meta-gate-artifacts-v1";
const META_GATE_ARTIFACT_STATUS_COMPLETE: &str = "complete";
const META_GATE_ARTIFACT_MANIFEST_FILE: &str = ".meta-gate-manifest.json";
const META_GATE_POLICY_FILE: &str = "policy.json";
const META_GATE_METRICS_FILE: &str = "metrics.json";
const FEATURE_NAMES: [&str; 11] = [
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

#[derive(Debug, Deserialize)]
struct PolicyArtifact {
    schema_version: String,
    actions: Vec<String>,
    feature_names: Vec<String>,
    scaler: ScalerArtifact,
    weights: Vec<Vec<f64>>,
    bias_index: usize,
}

#[derive(Debug, Deserialize)]
struct ScalerArtifact {
    feature_names: Vec<String>,
    means: Vec<f64>,
    stds: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ArtifactManifest {
    schema_version: String,
    status: String,
    generation: String,
    policy_file: String,
    metrics_file: String,
    policy_sha256: String,
    metrics_sha256: String,
}

#[derive(Debug, Clone, Copy)]
enum Action {
    Normal,
    Skip,
    Invert,
}

impl Action {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "normal" => Ok(Self::Normal),
            "skip" => Ok(Self::Skip),
            "invert" => Ok(Self::Invert),
            other => bail!(
                "policy contains unsupported action {other:?}; expected normal, skip, or invert"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Skip => "skip",
            Self::Invert => "invert",
        }
    }
}

struct LinearPolicy {
    actions: Vec<Action>,
    means: Vec<f64>,
    stds: Vec<f64>,
    weights: Vec<Vec<f64>>,
    bias_index: usize,
}

impl LinearPolicy {
    fn load(path: &Path) -> Result<Self> {
        let policy_bytes = read_verified_policy(path)?;
        let artifact: PolicyArtifact = serde_json::from_slice(&policy_bytes)
            .with_context(|| format!("parse policy JSON {}", path.display()))?;

        if artifact.schema_version != META_GATE_POLICY_SCHEMA {
            bail!(
                "unsupported policy schema in {}: expected {META_GATE_POLICY_SCHEMA}, found {:?}",
                path.display(),
                artifact.schema_version
            );
        }
        validate_feature_names("policy", &artifact.feature_names)?;
        validate_feature_names("policy scaler", &artifact.scaler.feature_names)?;
        if artifact.scaler.means.len() != FEATURE_NAMES.len()
            || artifact.scaler.stds.len() != FEATURE_NAMES.len()
        {
            bail!(
                "policy scaler in {} must contain exactly {} means and standard deviations",
                path.display(),
                FEATURE_NAMES.len()
            );
        }
        for (index, (mean, std)) in artifact
            .scaler
            .means
            .iter()
            .zip(&artifact.scaler.stds)
            .enumerate()
        {
            if !mean.is_finite() || !std.is_finite() || *std <= 0.0 {
                bail!(
                    "policy scaler feature {} has non-finite mean or non-positive standard deviation",
                    FEATURE_NAMES[index]
                );
            }
        }
        if artifact.actions.is_empty() {
            bail!("policy {} contains no actions", path.display());
        }
        let mut actions: Vec<Action> = Vec::with_capacity(artifact.actions.len());
        for action in &artifact.actions {
            let parsed = Action::parse(action.to_ascii_lowercase().as_str())?;
            if actions
                .iter()
                .any(|existing| existing.label() == parsed.label())
            {
                bail!(
                    "policy {} contains duplicate action {action:?}",
                    path.display()
                );
            }
            actions.push(parsed);
        }
        if artifact.bias_index != FEATURE_NAMES.len() {
            bail!(
                "policy bias_index in {} must be {}, found {}",
                path.display(),
                FEATURE_NAMES.len(),
                artifact.bias_index
            );
        }
        if artifact.weights.len() != actions.len() {
            bail!(
                "policy weights in {} contain {} rows for {} actions",
                path.display(),
                artifact.weights.len(),
                actions.len()
            );
        }
        for (index, weights) in artifact.weights.iter().enumerate() {
            if weights.len() != FEATURE_NAMES.len() + 1 {
                bail!(
                    "policy weights row {index} in {} must contain {} values including bias",
                    path.display(),
                    FEATURE_NAMES.len() + 1
                );
            }
            if weights.iter().any(|weight| !weight.is_finite()) {
                bail!(
                    "policy weights row {index} in {} contains a non-finite value",
                    path.display()
                );
            }
        }

        Ok(Self {
            actions,
            means: artifact.scaler.means,
            stds: artifact.scaler.stds,
            weights: artifact.weights,
            bias_index: artifact.bias_index,
        })
    }

    fn choose(&self, raw_features: &[f64; FEATURE_NAMES.len()]) -> Action {
        let mut normalized = [0.0; FEATURE_NAMES.len() + 1];
        for index in 0..FEATURE_NAMES.len() {
            normalized[index] = (raw_features[index] - self.means[index]) / self.stds[index];
        }
        normalized[self.bias_index] = 1.0;

        // The parent trainer selects the maximum softmax probability.  Since
        // softmax is monotonic in its logit, selecting the maximum linear
        // score is equivalent and avoids unnecessary exponentiation.  The
        // strict comparison preserves the parent's first-action tie break.
        let mut best_index = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (index, weights) in self.weights.iter().enumerate() {
            let score = weights
                .iter()
                .zip(normalized)
                .map(|(weight, feature)| weight * feature)
                .sum::<f64>();
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        self.actions[best_index]
    }
}

#[derive(Debug, Serialize)]
struct ScheduleRow {
    event_id: i64,
    timestamp_ns: i64,
    raw_direction: i8,
    selected_action: &'static str,
    current_side: i8,
    target_side: i8,
}

pub(crate) fn print_schedule(args: &MetaGateScheduleArgs) -> Result<()> {
    if args.limit == 0 {
        bail!("--limit must be at least 1; output is always bounded");
    }
    if args.limit > MAX_OUTPUT_ROWS {
        bail!(
            "--limit {} exceeds the maximum of {} rows",
            args.limit,
            MAX_OUTPUT_ROWS
        );
    }

    let policy = LinearPolicy::load(&args.policy)?;
    let events_file = open_readonly_file(&args.events, "event parquet")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(events_file)
        .with_context(|| format!("open event parquet reader {}", args.events.display()))?;
    validate_event_schema(builder.schema().as_ref(), &args.events)?;
    let reader = builder
        .with_batch_size(PARQUET_BATCH_ROWS)
        .build()
        .with_context(|| format!("build event parquet reader {}", args.events.display()))?;

    let mut current_side = initial_side(args.initial_side);
    let mut previous_timestamp = None;
    let mut previous_event_id = None;
    let mut total_events = 0usize;
    let mut rows = Vec::with_capacity(args.limit.min(1024));

    for batch_result in reader {
        let batch = batch_result
            .with_context(|| format!("read event parquet batch from {}", args.events.display()))?;
        let columns = EventColumns::from_batch(&batch, &args.events)?;
        for row in 0..batch.num_rows() {
            let schema_version = string_at(&columns.schema_version, row, "schema_version")?;
            if schema_version != META_EVENT_SCHEMA {
                bail!(
                    "event parquet {} contains schema_version {:?} at row {}; expected {META_EVENT_SCHEMA}",
                    args.events.display(),
                    schema_version,
                    total_events
                );
            }
            let event_id = integer_at(&columns.event_id, row, "event_id")?;
            let timestamp_ns = integer_at(&columns.timestamp_ns, row, "timestamp_ns")?;
            let raw_value = integer_at(&columns.raw_direction, row, "raw_direction")?;
            let raw_direction = i8::try_from(raw_value).with_context(|| {
                format!(
                    "event parquet {} raw_direction at row {} is outside the i8 range",
                    args.events.display(),
                    total_events
                )
            })?;
            if !matches!(raw_direction, -1 | 1) {
                bail!(
                    "event parquet {} raw_direction at row {} must be -1 or 1, found {}",
                    args.events.display(),
                    total_events,
                    raw_direction
                );
            }
            if previous_timestamp.is_some_and(|previous| timestamp_ns <= previous) {
                bail!(
                    "event parquet {} timestamps must be strictly increasing; row {} has {} after {}",
                    args.events.display(),
                    total_events,
                    timestamp_ns,
                    previous_timestamp.unwrap_or_default()
                );
            }
            if previous_event_id.is_some_and(|previous| event_id <= previous) {
                bail!(
                    "event parquet {} event_id values must be strictly increasing; row {} has {}",
                    args.events.display(),
                    total_events,
                    event_id
                );
            }

            let mut raw_features = [0.0; FEATURE_NAMES.len()];
            for (index, column) in columns.features.iter().enumerate() {
                raw_features[index] = float_at(column, row, FEATURE_NAMES[index])?;
            }
            let action = policy.choose(&raw_features);
            let target_side = target_side(current_side, raw_direction, action);
            if rows.len() < args.limit {
                rows.push(ScheduleRow {
                    event_id,
                    timestamp_ns,
                    raw_direction,
                    selected_action: action.label(),
                    current_side,
                    target_side,
                });
            }
            current_side = target_side;
            previous_timestamp = Some(timestamp_ns);
            previous_event_id = Some(event_id);
            total_events = total_events.saturating_add(1);
        }
    }

    if total_events == 0 {
        bail!(
            "event parquet {} contains no event rows",
            args.events.display()
        );
    }
    match args.format {
        MetaGateScheduleFormat::Table => print_table(&rows),
        MetaGateScheduleFormat::Jsonl => print_jsonl(&rows)?,
    }
    eprintln!(
        "read {total_events} meta-event-v1 rows; printed {} (limit {}) from policy meta-gate-policy-v1; read-only schedule",
        rows.len(),
        args.limit
    );
    Ok(())
}

fn open_readonly_file(path: &Path, label: &str) -> Result<File> {
    let metadata = fs::symlink_metadata(path).with_context(|| {
        format!(
            "{} path does not exist or is unreadable: {}",
            label,
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        bail!("{label} path is a symlink: {}", path.display());
    }
    if !metadata.file_type().is_file() {
        bail!("{label} path is not a regular file: {}", path.display());
    }
    File::open(path).with_context(|| format!("open {label} {} for reading", path.display()))
}

fn read_verified_policy(policy_path: &Path) -> Result<Vec<u8>> {
    if policy_path.file_name().and_then(|name| name.to_str()) != Some(META_GATE_POLICY_FILE) {
        bail!(
            "meta-gate schedule policy must be named {META_GATE_POLICY_FILE} so its completion manifest is unambiguous"
        );
    }
    let outdir = policy_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest_path = outdir.join(META_GATE_ARTIFACT_MANIFEST_FILE);
    let metrics_path = outdir.join(META_GATE_METRICS_FILE);
    let manifest_bytes = read_regular_file(&manifest_path, "artifact manifest")?;
    let manifest: ArtifactManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse artifact manifest {}", manifest_path.display()))?;
    if manifest.schema_version != META_GATE_ARTIFACT_MANIFEST_SCHEMA
        || manifest.status != META_GATE_ARTIFACT_STATUS_COMPLETE
        || manifest.generation.is_empty()
        || manifest.policy_file != META_GATE_POLICY_FILE
        || manifest.metrics_file != META_GATE_METRICS_FILE
    {
        bail!(
            "artifact manifest {} is not a complete meta-gate artifact pair",
            manifest_path.display()
        );
    }

    let policy_bytes = read_regular_file(policy_path, "policy JSON")?;
    let metrics_bytes = read_regular_file(&metrics_path, "metrics JSON")?;
    if sha256_bytes(&policy_bytes) != manifest.policy_sha256
        || sha256_bytes(&metrics_bytes) != manifest.metrics_sha256
    {
        bail!(
            "meta-gate policy/metrics pair does not match completion manifest {}",
            manifest_path.display()
        );
    }
    Ok(policy_bytes)
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let mut file = open_readonly_file(path, label)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read {label} {}", path.display()))?;
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_feature_names(label: &str, names: &[String]) -> Result<()> {
    if names.len() != FEATURE_NAMES.len()
        || names
            .iter()
            .zip(FEATURE_NAMES)
            .any(|(actual, expected)| actual != expected)
    {
        bail!(
            "{label} feature_names do not match the required causal 11-feature contract: expected {:?}, found {:?}",
            FEATURE_NAMES,
            names
        );
    }
    Ok(())
}

struct EventColumns {
    schema_version: ArrayRef,
    event_id: ArrayRef,
    timestamp_ns: ArrayRef,
    raw_direction: ArrayRef,
    features: [ArrayRef; FEATURE_NAMES.len()],
}

impl EventColumns {
    fn from_batch(batch: &arrow_array::RecordBatch, path: &Path) -> Result<Self> {
        let schema = batch.schema();
        let column = |name: &str| -> Result<ArrayRef> {
            let index = schema.index_of(name).with_context(|| {
                format!("event parquet {} is missing column {name}", path.display())
            })?;
            Ok(batch.column(index).clone())
        };
        let features = FEATURE_NAMES
            .map(column)
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let features: [ArrayRef; FEATURE_NAMES.len()] = features.try_into().map_err(|_| {
            anyhow::anyhow!("internal error while collecting meta-gate feature columns")
        })?;
        Ok(Self {
            schema_version: column("schema_version")?,
            event_id: column("event_id")?,
            timestamp_ns: column("timestamp_ns")?,
            raw_direction: column("raw_direction")?,
            features,
        })
    }
}

fn validate_event_schema(schema: &arrow_schema::Schema, path: &Path) -> Result<()> {
    for name in [
        "schema_version",
        "event_id",
        "timestamp_ns",
        "raw_direction",
    ] {
        if schema.index_of(name).is_err() {
            bail!(
                "event parquet {} is missing required column {name}",
                path.display()
            );
        }
    }
    for name in FEATURE_NAMES {
        if schema.index_of(name).is_err() {
            bail!(
                "event parquet {} is missing required causal feature column {name}",
                path.display()
            );
        }
    }
    Ok(())
}

fn string_at<'a>(column: &'a ArrayRef, row: usize, name: &str) -> Result<&'a str> {
    if column.is_null(row) {
        bail!("event column {name} has null at row {row}");
    }
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row));
    }
    bail!("event column {name} has a non-string Arrow type")
}

fn float_at(column: &ArrayRef, row: usize, name: &str) -> Result<f64> {
    if column.is_null(row) {
        bail!("event column {name} has null at row {row}");
    }
    let value = if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        values.value(row)
    } else if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<Int8Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<Int16Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<UInt8Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<UInt16Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<UInt32Array>() {
        values.value(row) as f64
    } else if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
        values.value(row) as f64
    } else {
        bail!("event column {name} has an unsupported numeric Arrow type")
    };
    if !value.is_finite() {
        bail!("event column {name} has non-finite value at row {row}");
    }
    Ok(value)
}

fn integer_at(column: &ArrayRef, row: usize, name: &str) -> Result<i64> {
    if column.is_null(row) {
        bail!("event column {name} has null at row {row}");
    }
    if let Some(values) = column.as_any().downcast_ref::<Int8Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<Int16Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt8Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt16Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
        return i64::try_from(values.value(row))
            .with_context(|| format!("event column {name} overflows i64 at row {row}"));
    }
    bail!("event column {name} has an unsupported integer Arrow type")
}

fn initial_side(side: MetaGateInitialSide) -> i8 {
    match side {
        MetaGateInitialSide::Flat => 0,
        MetaGateInitialSide::Long => 1,
        MetaGateInitialSide::Short => -1,
    }
}

fn target_side(current_side: i8, raw_direction: i8, action: Action) -> i8 {
    match action {
        Action::Normal => raw_direction,
        Action::Invert => -raw_direction,
        Action::Skip => current_side,
    }
}

fn print_table(rows: &[ScheduleRow]) {
    println!(
        "{:<10} {:<20} {:>4} {:<16} {:>12} {:>11} {:>11}",
        "event_id", "timestamp_ns", "raw", "action", "current_side", "target_side", "target"
    );
    for row in rows {
        println!(
            "{:<10} {:<20} {:>4} {:<16} {:>12} {:>11} {:>11}",
            row.event_id,
            row.timestamp_ns,
            row.raw_direction,
            row.selected_action,
            row.current_side,
            row.target_side,
            side_label(row.target_side),
        );
    }
}

fn print_jsonl(rows: &[ScheduleRow]) -> Result<()> {
    for row in rows {
        println!("{}", serde_json::to_string(row)?);
    }
    Ok(())
}

fn side_label(side: i8) -> &'static str {
    match side {
        -1 => "short",
        0 => "flat",
        1 => "long",
        _ => "invalid",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_output_dir() -> std::path::PathBuf {
        for _ in 0..100 {
            let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "midas-meta-gate-schedule-test-{}-{sequence}",
                std::process::id()
            ));
            if fs::create_dir(&path).is_ok() {
                return path;
            }
        }
        panic!("could not create a unique schedule test directory");
    }

    fn write_complete_pair(outdir: &Path) {
        let policy = b"verified policy bytes";
        let metrics = b"verified metrics bytes";
        fs::write(outdir.join(META_GATE_POLICY_FILE), policy).unwrap();
        fs::write(outdir.join(META_GATE_METRICS_FILE), metrics).unwrap();
        let manifest = serde_json::json!({
            "schema_version": META_GATE_ARTIFACT_MANIFEST_SCHEMA,
            "status": META_GATE_ARTIFACT_STATUS_COMPLETE,
            "generation": "test-generation",
            "policy_file": META_GATE_POLICY_FILE,
            "metrics_file": META_GATE_METRICS_FILE,
            "policy_sha256": sha256_bytes(policy),
            "metrics_sha256": sha256_bytes(metrics),
        });
        fs::write(
            outdir.join(META_GATE_ARTIFACT_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn schedule_policy_requires_completion_manifest() {
        let outdir = test_output_dir();
        let policy_path = outdir.join(META_GATE_POLICY_FILE);
        fs::write(&policy_path, b"policy").unwrap();

        let error = read_verified_policy(&policy_path).unwrap_err();
        assert!(error.to_string().contains("artifact manifest"));

        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn schedule_policy_rejects_tampered_pair() {
        let outdir = test_output_dir();
        let policy_path = outdir.join(META_GATE_POLICY_FILE);
        write_complete_pair(&outdir);
        fs::write(&policy_path, b"tampered policy bytes").unwrap();

        let error = read_verified_policy(&policy_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match completion manifest")
        );

        fs::remove_dir_all(outdir).unwrap();
    }
}
