use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use midas_env::meta_gate::{
    GateAction, META_EVENT_SCHEMA, MetaEvent, MetaGateConfig, canonical_dataset_fingerprint_sha256,
    canonical_event_payload_sha256, extract_events, meta_events_from_frame, meta_events_to_frame,
    open_meta_source_snapshot,
};
use polars::prelude::{
    DataFrame, DataType, NamedFrom, ParquetReader, ParquetWriter, SerReader, SerWriter, Series,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

const FIXED_HORIZON_WARNING: &str = "fixed-horizon diagnostic training only; this prototype is not sequential Trader replay. Future PnL columns are labels and are never features. Results do not model position lifecycle, protections, overlapping-event handling, or execution fills.";
const ARTIFACT_MANIFEST_SCHEMA: &str = "meta-gate-artifacts-v1";
const ARTIFACT_MANIFEST_FILE: &str = ".meta-gate-manifest.json";
const TRANSACTION_FILE: &str = ".meta-gate-transaction.json";
const POLICY_FILE: &str = "policy.json";
const METRICS_FILE: &str = "metrics.json";
const ARTIFACT_STATUS_COMPLETE: &str = "complete";
const EXPECTED_ARTIFACT_FILES: [&str; 3] = [POLICY_FILE, METRICS_FILE, ARTIFACT_MANIFEST_FILE];

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Algorithm {
    Ga,
    Rl,
}

impl Algorithm {
    fn label(self) -> &'static str {
        match self {
            Self::Ga => "ga",
            Self::Rl => "rl",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

impl Device {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "train_meta_gate",
    about = "Train a deterministic event-level normal/skip/invert meta-gate prototype",
    long_about = "Train a small event-level contextual-bandit prototype over prepare_meta_dataset parquet output.\n\nThis is fixed-horizon diagnostic training, not sequential Trader replay: normal_pnl, skip_pnl, and invert_pnl are labels, while only causal feature columns are used as inputs."
)]
struct Args {
    /// Event parquet produced by prepare_meta_dataset.
    #[arg(long, value_name = "PATH")]
    input: PathBuf,

    /// Directory receiving policy.json and metrics.json.
    #[arg(long, value_name = "PATH")]
    outdir: PathBuf,

    /// Training algorithm.
    #[arg(long, value_enum, default_value = "ga")]
    algorithm: Algorithm,

    /// Compute device. CUDA is intentionally rejected by this prototype.
    #[arg(long, value_enum, default_value = "cpu")]
    device: Device,

    /// Root seed for deterministic initialization and mutation.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of GA generations. Ignored by --algorithm rl.
    #[arg(long, default_value_t = 25)]
    generations: usize,

    /// Number of RL policy-gradient epochs. Ignored by --algorithm ga.
    #[arg(long, default_value_t = 25)]
    epochs: usize,

    /// GA population size. Accepted for a common CLI shape; RL uses one policy.
    #[arg(long, default_value_t = 32)]
    population: usize,

    /// Chronological fraction used to fit the scaler and train the policy.
    #[arg(long, default_value_t = 0.6)]
    train_fraction: f64,

    /// Chronological validation fraction after the training prefix.
    #[arg(long, default_value_t = 0.2)]
    val_fraction: f64,

    /// Number of source-bar rows to embargo after each split boundary.
    #[arg(long, default_value_t = 30)]
    purge_bars: usize,

    /// Selected learned actions. Normal and skip are required; invert is optional.
    #[arg(long, default_value = "normal,skip")]
    actions: String,

    /// Replace existing policy.json and metrics.json artifacts.
    ///
    /// By default, an existing artifact is never overwritten. Temporary files
    /// are published with a no-replace operation so a rerun cannot silently
    /// destroy a previous result.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    Normal,
    Skip,
    Invert,
}

impl Action {
    const ALL: [Self; 3] = [Self::Normal, Self::Skip, Self::Invert];

    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Skip => "skip",
            Self::Invert => "invert",
        }
    }

    fn reward(self, rewards: [f64; 3]) -> f64 {
        match self {
            Self::Normal => rewards[0],
            Self::Skip => rewards[1],
            Self::Invert => rewards[2],
        }
    }
}

#[derive(Debug, Clone)]
struct EventRow {
    row_idx: usize,
    horizon_row_idx: usize,
    raw_features: Vec<f64>,
    rewards: [f64; 3],
}

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

#[derive(Debug, Clone)]
struct Sample {
    features: Vec<f64>,
    rewards: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
struct ScalerArtifact {
    feature_names: Vec<String>,
    means: Vec<f64>,
    stds: Vec<f64>,
    fit_row_start: usize,
    fit_row_end_exclusive: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SplitRanges {
    purge_bars: usize,
    train_start: usize,
    train_end_exclusive: usize,
    validation_start: usize,
    validation_end_exclusive: usize,
    holdout_start: usize,
    holdout_end_exclusive: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArtifactManifest {
    schema_version: String,
    status: String,
    generation: String,
    policy_file: String,
    metrics_file: String,
    policy_sha256: String,
    metrics_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransactionArtifact {
    destination: String,
    backup: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransactionRecord {
    schema_version: String,
    generation: String,
    artifacts: Vec<TransactionArtifact>,
}

impl SplitRanges {
    fn train(&self) -> Range<usize> {
        self.train_start..self.train_end_exclusive
    }

    fn validation(&self) -> Range<usize> {
        self.validation_start..self.validation_end_exclusive
    }

    fn holdout(&self) -> Range<usize> {
        self.holdout_start..self.holdout_end_exclusive
    }
}

#[derive(Debug, Clone, Serialize)]
struct SplitCounts {
    total: usize,
    retained: usize,
    purged: usize,
    train: usize,
    validation: usize,
    holdout: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    event_count: usize,
    action_counts: BTreeMap<String, usize>,
    action_fractions: BTreeMap<String, f64>,
    sum_pnl: f64,
    mean_pnl: f64,
    win_rate: f64,
    max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TrainingMetadata {
    purge_bars: usize,
    iterations: usize,
    population: usize,
    learning_rate: Option<f64>,
    mutation_rate: Option<f64>,
    mutation_step: Option<f64>,
    reward_scale: Option<f64>,
    objective: String,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyArtifact {
    schema_version: &'static str,
    warning: &'static str,
    input: String,
    algorithm: String,
    device: String,
    seed: u64,
    actions: Vec<String>,
    feature_names: Vec<String>,
    provenance: DatasetProvenance,
    scaler: ScalerArtifact,
    split_counts: SplitCounts,
    split_ranges: SplitRanges,
    weights: Vec<Vec<f64>>,
    bias_index: usize,
    training: TrainingMetadata,
}

#[derive(Debug, Clone, Serialize)]
struct MetricsArtifact {
    schema_version: &'static str,
    warning: &'static str,
    input: String,
    algorithm: String,
    device: String,
    seed: u64,
    actions: Vec<String>,
    feature_names: Vec<String>,
    provenance: DatasetProvenance,
    scaler: ScalerArtifact,
    split_counts: SplitCounts,
    split_ranges: SplitRanges,
    fixed_policies: BTreeMap<String, BTreeMap<String, Summary>>,
    learned_policy: BTreeMap<String, Summary>,
    training: TrainingMetadata,
}

#[derive(Debug, Clone)]
struct LinearPolicy {
    actions: Vec<Action>,
    weights: Vec<Vec<f64>>,
}

impl LinearPolicy {
    fn zero(actions: &[Action], feature_count: usize) -> Self {
        Self {
            actions: actions.to_vec(),
            weights: vec![vec![0.0; feature_count + 1]; actions.len()],
        }
    }

    fn random(actions: &[Action], feature_count: usize, rng: &mut StdRng, scale: f64) -> Self {
        let mut policy = Self::zero(actions, feature_count);
        for row in &mut policy.weights {
            for weight in row {
                *weight = rng.gen_range(-scale..scale);
            }
        }
        policy
    }

    fn probabilities(&self, features_with_bias: &[f64]) -> Vec<f64> {
        let logits = self
            .weights
            .iter()
            .map(|row| {
                row.iter()
                    .zip(features_with_bias)
                    .map(|(weight, feature)| weight * feature)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits
            .iter()
            .map(|logit| (logit - max_logit).exp())
            .collect::<Vec<_>>();
        let normalizer = probabilities.iter().sum::<f64>();
        if normalizer <= f64::EPSILON || !normalizer.is_finite() {
            let uniform = 1.0 / probabilities.len() as f64;
            probabilities.fill(uniform);
        } else {
            for probability in &mut probabilities {
                *probability /= normalizer;
            }
        }
        probabilities
    }

    fn choose_index(&self, features_with_bias: &[f64]) -> usize {
        let probabilities = self.probabilities(features_with_bias);
        probabilities
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn choose_action(&self, features_with_bias: &[f64]) -> Action {
        self.actions[self.choose_index(features_with_bias)]
    }

    fn mutate(&mut self, rng: &mut StdRng, mutation_rate: f64, mutation_step: f64) {
        let mut changed = false;
        for row in &mut self.weights {
            for weight in row {
                if rng.gen_bool(mutation_rate) {
                    *weight =
                        (*weight + rng.gen_range(-mutation_step..mutation_step)).clamp(-20.0, 20.0);
                    changed = true;
                }
            }
        }
        if !changed {
            let row = rng.gen_range(0..self.weights.len());
            let column = rng.gen_range(0..self.weights[row].len());
            self.weights[row][column] = (self.weights[row][column]
                + rng.gen_range(-mutation_step..mutation_step))
            .clamp(-20.0, 20.0);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if matches!(args.device, Device::Cuda) {
        bail!(
            "--device cuda is not yet supported by train_meta_gate: this prototype has a deterministic CPU implementation only; use --device cpu"
        );
    }
    if args.population < 2 && matches!(args.algorithm, Algorithm::Ga) {
        bail!("--population must be at least 2 for --algorithm ga");
    }
    let iterations = match args.algorithm {
        Algorithm::Ga => args.generations,
        Algorithm::Rl => args.epochs,
    };
    if iterations == 0 {
        bail!("the selected algorithm requires a positive generations/epochs value");
    }
    let actions = parse_actions(&args.actions)?;
    let (events, provenance) = load_events(&args.input)?;
    let splits = split_ranges(
        &events,
        args.train_fraction,
        args.val_fraction,
        args.purge_bars,
    )?;
    let scaler = fit_scaler(&events, splits.train());
    let samples = normalize_events(&events, &scaler);

    let (policy, mut training) = match args.algorithm {
        Algorithm::Ga => train_ga(
            &samples,
            splits.train(),
            &actions,
            args.generations,
            args.population,
            args.seed,
        ),
        Algorithm::Rl => train_rl(
            &samples,
            splits.train(),
            &actions,
            args.epochs,
            args.population,
            args.seed,
        ),
    };
    training.purge_bars = args.purge_bars;

    let split_counts = split_counts(&splits);
    let fixed_policies = fixed_metrics(&samples, &splits);
    let learned_policy = learned_metrics(&samples, &splits, &policy);
    let action_labels = actions
        .iter()
        .map(|action| action.label().to_string())
        .collect::<Vec<_>>();

    std::fs::create_dir_all(&args.outdir)
        .with_context(|| format!("create output directory {}", args.outdir.display()))?;

    let policy_artifact = PolicyArtifact {
        schema_version: "meta-gate-policy-v1",
        warning: FIXED_HORIZON_WARNING,
        input: args.input.display().to_string(),
        algorithm: args.algorithm.label().to_string(),
        device: args.device.label().to_string(),
        seed: args.seed,
        actions: action_labels.clone(),
        feature_names: causal_feature_names(),
        provenance: provenance.clone(),
        scaler: scaler.clone(),
        split_counts: split_counts.clone(),
        split_ranges: splits.clone(),
        weights: policy.weights.clone(),
        bias_index: CAUSAL_FEATURES.len(),
        training: training.clone(),
    };
    let metrics_artifact = MetricsArtifact {
        schema_version: "meta-gate-metrics-v1",
        warning: FIXED_HORIZON_WARNING,
        input: args.input.display().to_string(),
        algorithm: args.algorithm.label().to_string(),
        device: args.device.label().to_string(),
        seed: args.seed,
        actions: action_labels,
        feature_names: causal_feature_names(),
        provenance,
        scaler,
        split_counts,
        split_ranges: splits,
        fixed_policies,
        learned_policy,
        training,
    };

    write_json_artifacts(
        &args.outdir,
        &policy_artifact,
        &metrics_artifact,
        args.overwrite,
    )?;

    print_report(&metrics_artifact);
    println!(
        "wrote {} and {}",
        args.outdir.join("policy.json").display(),
        args.outdir.join("metrics.json").display()
    );
    Ok(())
}

fn causal_feature_names() -> Vec<String> {
    CAUSAL_FEATURES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn parse_actions(raw: &str) -> Result<Vec<Action>> {
    let mut seen = [false; 3];
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let action = match token.to_ascii_lowercase().as_str() {
            "normal" => Action::Normal,
            "skip" => Action::Skip,
            "invert" => Action::Invert,
            other => bail!(
                "invalid action {other:?}; --actions must be normal,skip or normal,skip,invert"
            ),
        };
        let index = match action {
            Action::Normal => 0,
            Action::Skip => 1,
            Action::Invert => 2,
        };
        if seen[index] {
            bail!("duplicate action {token:?} in --actions");
        }
        seen[index] = true;
    }
    if !seen[0] || !seen[1] {
        bail!("--actions must include both normal and skip; invert is optional");
    }
    Ok(Action::ALL
        .into_iter()
        .filter(|action| match action {
            Action::Normal => seen[0],
            Action::Skip => seen[1],
            Action::Invert => seen[2],
        })
        .collect())
}

fn load_events(path: &Path) -> Result<(Vec<EventRow>, DatasetProvenance)> {
    let file = File::open(path).with_context(|| format!("open input {}", path.display()))?;
    let frame = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("read input parquet {}", path.display()))?;
    if frame.height() == 0 {
        bail!("input parquet {} contains no event rows", path.display());
    }

    let provenance = validate_event_provenance(&frame)?;
    validate_dataset_integrity(&frame, &provenance)?;
    let timestamps = required_i64(&frame, "timestamp_ns")?;
    let row_indices = required_usize(&frame, "row_idx")?;
    let entry_row_indices = required_usize(&frame, "entry_row_idx")?;
    let horizon_row_indices = required_usize(&frame, "horizon_row_idx")?;
    let columns = CAUSAL_FEATURES
        .iter()
        .map(|name| required_f64(&frame, name))
        .collect::<Result<Vec<_>>>()?;
    // These are outcomes, not observations. They are deliberately loaded only
    // into the reward tuple and never passed to fit_scaler or the policy.
    let normal_pnl = required_f64(&frame, "normal_pnl")?;
    let skip_pnl = required_f64(&frame, "skip_pnl")?;
    let invert_pnl = required_f64(&frame, "invert_pnl")?;

    let row_count = frame.height();
    if timestamps.len() != row_count
        || columns.iter().any(|column| column.len() != row_count)
        || normal_pnl.len() != row_count
        || skip_pnl.len() != row_count
        || invert_pnl.len() != row_count
        || row_indices.len() != row_count
        || entry_row_indices.len() != row_count
        || horizon_row_indices.len() != row_count
    {
        bail!("input parquet columns do not have a common row count");
    }
    if timestamps.windows(2).any(|window| window[1] <= window[0]) {
        bail!("timestamp_ns must be strictly increasing for chronological splitting");
    }
    if row_indices.windows(2).any(|window| window[1] <= window[0]) {
        bail!("row_idx must be strictly increasing for chronological splitting");
    }

    let mut events = Vec::with_capacity(row_count);
    for row in 0..row_count {
        if entry_row_indices[row] <= row_indices[row] {
            bail!(
                "entry_row_idx must be after row_idx at event row {row}: {} <= {}",
                entry_row_indices[row],
                row_indices[row]
            );
        }
        if horizon_row_indices[row] < entry_row_indices[row] {
            bail!(
                "horizon_row_idx must be at or after entry_row_idx at event row {row}: {} < {}",
                horizon_row_indices[row],
                entry_row_indices[row]
            );
        }
        let raw_features = columns.iter().map(|column| column[row]).collect::<Vec<_>>();
        events.push(EventRow {
            row_idx: row_indices[row],
            horizon_row_idx: horizon_row_indices[row],
            raw_features,
            rewards: [normal_pnl[row], skip_pnl[row], invert_pnl[row]],
        });
    }
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
        if !is_sha256(value) {
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

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
}

fn validate_dataset_integrity(frame: &DataFrame, provenance: &DatasetProvenance) -> Result<()> {
    let stored_events = meta_events_from_frame(frame)
        .context("decode event payload for integrity validation; reprepare the dataset")?;
    let config: MetaGateConfig = serde_json::from_str(&provenance.config_json)
        .context("decode trusted meta-gate config; reprepare the dataset")?;
    let snapshot = open_meta_source_snapshot(
        Path::new(&provenance.source_path),
        Path::new(&provenance.source_root),
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

fn event_payload_from_frame(frame: &DataFrame) -> Result<Vec<MetaEvent>> {
    let event_id = required_usize(frame, "event_id")?;
    let row_idx = required_usize(frame, "row_idx")?;
    let timestamp_ns = required_i64(frame, "timestamp_ns")?;
    let open = required_f64(frame, "open")?;
    let high = required_f64(frame, "high")?;
    let low = required_f64(frame, "low")?;
    let close = required_f64(frame, "close")?;
    let volume = required_f64(frame, "volume")?;
    let raw_direction = required_i64(frame, "raw_direction")?;
    let trigger_fast_value = required_f64(frame, "trigger_fast_value")?;
    let trigger_slow_value = required_f64(frame, "trigger_slow_value")?;
    let context_fast_value = required_f64(frame, "context_fast_value")?;
    let context_slow_value = required_f64(frame, "context_slow_value")?;
    let atr = required_f64(frame, "atr")?;
    let trigger_spread_atr = required_f64(frame, "trigger_spread_atr")?;
    let context_spread_atr = required_f64(frame, "context_spread_atr")?;
    let trigger_fast_slope_atr = required_f64(frame, "trigger_fast_slope_atr")?;
    let trigger_slow_slope_atr = required_f64(frame, "trigger_slow_slope_atr")?;
    let context_fast_slope_atr = required_f64(frame, "context_fast_slope_atr")?;
    let context_slow_slope_atr = required_f64(frame, "context_slow_slope_atr")?;
    let efficiency_ratio = required_f64(frame, "efficiency_ratio")?;
    let rvol_20 = required_f64(frame, "rvol_20")?;
    let return_1 = required_f64(frame, "return_1")?;
    let return_5 = required_f64(frame, "return_5")?;
    let entry_price = required_f64(frame, "entry_price")?;
    let normal_pnl = required_f64(frame, "normal_pnl")?;
    let invert_pnl = required_f64(frame, "invert_pnl")?;
    let skip_pnl = required_f64(frame, "skip_pnl")?;
    let oracle_action = required_gate_actions(frame, "oracle_action")?;
    let entry_row_idx = required_usize(frame, "entry_row_idx")?;
    let horizon_row_idx = required_usize(frame, "horizon_row_idx")?;

    let row_count = frame.height();
    let lengths = [
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
    if lengths.iter().any(|length| *length != row_count) {
        bail!("event payload columns do not have a common row count");
    }

    let mut events = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let raw_direction = i8::try_from(raw_direction[row])
            .with_context(|| format!("raw_direction is outside the i8 range at event row {row}"))?;
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

fn required_gate_actions(frame: &DataFrame, name: &str) -> Result<Vec<GateAction>> {
    required_strings(frame, name)?
        .into_iter()
        .enumerate()
        .map(|(row, value)| match value.as_str() {
            "normal" => Ok(GateAction::Normal),
            "skip" => Ok(GateAction::Skip),
            "invert" => Ok(GateAction::Invert),
            _ => bail!("column {name} has invalid gate action at row {row}: {value:?}"),
        })
        .collect()
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
        .with_context(|| format!("input parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::String)
        .with_context(|| format!("cast {name} to String"))?;
    cast.str()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow::anyhow!("column {name} has null at row {row}"))
        })
        .collect()
}

fn required_f64(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Float64)
        .with_context(|| format!("cast {name} to Float64"))?;
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

fn required_i64(frame: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input parquet is missing required column {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Int64)
        .with_context(|| format!("cast {name} to Int64"))?;
    cast.i64()?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at row {row}"))
        })
        .collect()
}

fn required_usize(frame: &DataFrame, name: &str) -> Result<Vec<usize>> {
    required_i64(frame, name)?
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            usize::try_from(value).map_err(|_| {
                anyhow::anyhow!(
                    "column {name} has negative or oversized value at row {row}: {value}"
                )
            })
        })
        .collect()
}

fn split_ranges(
    events: &[EventRow],
    train_fraction: f64,
    val_fraction: f64,
    purge_bars: usize,
) -> Result<SplitRanges> {
    let row_count = events.len();
    if !train_fraction.is_finite()
        || !val_fraction.is_finite()
        || train_fraction <= 0.0
        || val_fraction <= 0.0
        || train_fraction + val_fraction >= 1.0
    {
        bail!("train_fraction and val_fraction must be finite, positive, and sum to less than 1");
    }
    let train_end_exclusive = (row_count as f64 * train_fraction).floor() as usize;
    let validation_end_exclusive =
        (row_count as f64 * (train_fraction + val_fraction)).floor() as usize;
    if train_end_exclusive == 0
        || validation_end_exclusive <= train_end_exclusive
        || validation_end_exclusive >= row_count
    {
        bail!(
            "fractions produce an empty chronological split for {row_count} rows; increase the input or adjust train_fraction/val_fraction"
        );
    }

    // First retain a right-hand row at least purge_bars after each nominal
    // boundary.  Then trim the left split to ensure every retained left label
    // is fully outside the embargoed interval.  Both coordinates are source
    // bar rows, not event-vector offsets.
    let validation_start = first_embargoed_row(events, train_end_exclusive, purge_bars)?;
    let holdout_start = first_embargoed_row(events, validation_end_exclusive, purge_bars)?;
    let train_end_exclusive = purged_left_end(
        events,
        0,
        train_end_exclusive,
        events[validation_start].row_idx,
        purge_bars,
    );
    let validation_end_exclusive = purged_left_end(
        events,
        validation_start,
        validation_end_exclusive,
        events[holdout_start].row_idx,
        purge_bars,
    );

    if train_end_exclusive == 0 {
        bail!(
            "purge/embargo of {purge_bars} bars removes every training event; reduce --purge-bars or adjust the split fractions"
        );
    }
    if validation_start >= validation_end_exclusive {
        bail!(
            "purge/embargo of {purge_bars} bars leaves no validation events (start={} end={}); reduce --purge-bars or adjust the split fractions",
            validation_start,
            validation_end_exclusive
        );
    }
    if holdout_start >= row_count {
        bail!(
            "purge/embargo of {purge_bars} bars leaves no holdout events; reduce --purge-bars or adjust the split fractions"
        );
    }
    validate_boundary(events, train_end_exclusive, validation_start, purge_bars)?;
    validate_boundary(events, validation_end_exclusive, holdout_start, purge_bars)?;

    Ok(SplitRanges {
        purge_bars,
        train_start: 0,
        train_end_exclusive,
        validation_start,
        validation_end_exclusive,
        holdout_start,
        holdout_end_exclusive: row_count,
    })
}

fn first_embargoed_row(
    events: &[EventRow],
    nominal_start: usize,
    purge_bars: usize,
) -> Result<usize> {
    let boundary_row = events[nominal_start].row_idx;
    let minimum_row = boundary_row.checked_add(purge_bars).ok_or_else(|| {
        anyhow::anyhow!(
            "row_idx overflow while applying purge/embargo of {purge_bars} bars at event {nominal_start}"
        )
    })?;
    events[nominal_start..]
        .iter()
        .position(|event| event.row_idx >= minimum_row)
        .map(|offset| nominal_start + offset)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "purge/embargo of {purge_bars} bars removes every event after nominal boundary {nominal_start}"
            )
        })
}

fn purged_left_end(
    events: &[EventRow],
    left_start: usize,
    nominal_left_end: usize,
    first_right_row: usize,
    purge_bars: usize,
) -> usize {
    events[left_start..nominal_left_end]
        .iter()
        .position(|event| crosses_boundary(event, first_right_row, purge_bars))
        .map_or(nominal_left_end, |offset| left_start + offset)
}

fn crosses_boundary(event: &EventRow, first_right_row: usize, purge_bars: usize) -> bool {
    event
        .horizon_row_idx
        .checked_add(purge_bars)
        .map(|last_embargoed_row| last_embargoed_row >= first_right_row)
        .unwrap_or(true)
}

fn validate_boundary(
    events: &[EventRow],
    left_end: usize,
    right_start: usize,
    purge_bars: usize,
) -> Result<()> {
    let first_right_row = events[right_start].row_idx;
    if events[..left_end]
        .iter()
        .any(|event| crosses_boundary(event, first_right_row, purge_bars))
    {
        bail!(
            "internal purge/embargo error: a retained event before split row {right_start} overlaps the first retained right event at source row {first_right_row}"
        );
    }
    Ok(())
}

fn split_counts(splits: &SplitRanges) -> SplitCounts {
    let train = splits.train_end_exclusive - splits.train_start;
    let validation = splits.validation_end_exclusive - splits.validation_start;
    let holdout = splits.holdout_end_exclusive - splits.holdout_start;
    let retained = train + validation + holdout;
    let total = splits.holdout_end_exclusive;
    SplitCounts {
        total,
        retained,
        purged: total - retained,
        train,
        validation,
        holdout,
    }
}

fn fit_scaler(events: &[EventRow], train_range: Range<usize>) -> ScalerArtifact {
    let feature_count = CAUSAL_FEATURES.len();
    let fit_count = train_range.end - train_range.start;
    let mut means = vec![0.0; feature_count];
    for event in &events[train_range.clone()] {
        for (mean, value) in means.iter_mut().zip(&event.raw_features) {
            *mean += *value;
        }
    }
    for mean in &mut means {
        *mean /= fit_count as f64;
    }

    let mut stds = vec![0.0; feature_count];
    for event in &events[train_range.clone()] {
        for ((std, mean), value) in stds.iter_mut().zip(&means).zip(&event.raw_features) {
            let delta = value - mean;
            *std += delta * delta;
        }
    }
    for std in &mut stds {
        *std = (*std / fit_count as f64).sqrt();
        if !std.is_finite() || *std <= 1.0e-12 {
            *std = 1.0;
        }
    }

    ScalerArtifact {
        feature_names: causal_feature_names(),
        means,
        stds,
        fit_row_start: train_range.start,
        fit_row_end_exclusive: train_range.end,
    }
}

fn normalize_events(events: &[EventRow], scaler: &ScalerArtifact) -> Vec<Sample> {
    events
        .iter()
        .map(|event| {
            let mut features = event
                .raw_features
                .iter()
                .zip(&scaler.means)
                .zip(&scaler.stds)
                .map(|((value, mean), std)| (value - mean) / std)
                .collect::<Vec<_>>();
            features.push(1.0);
            Sample {
                features,
                rewards: event.rewards,
            }
        })
        .collect()
}

fn train_ga(
    samples: &[Sample],
    train_range: Range<usize>,
    actions: &[Action],
    generations: usize,
    population_size: usize,
    seed: u64,
) -> (LinearPolicy, TrainingMetadata) {
    const MUTATION_RATE: f64 = 0.20;
    const MUTATION_STEP: f64 = 0.25;
    let feature_count = samples[0].features.len() - 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population = Vec::with_capacity(population_size);
    population.push(LinearPolicy::zero(actions, feature_count));
    for _ in 1..population_size {
        population.push(LinearPolicy::random(actions, feature_count, &mut rng, 0.05));
    }

    let mut best_policy = population[0].clone();
    let mut best_score = f64::NEG_INFINITY;
    for _generation in 0..generations {
        let mut scored = population
            .into_iter()
            .map(|policy| {
                let score = score_policy(&policy, samples, train_range.clone());
                (score, policy)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        if scored[0].0 > best_score {
            best_score = scored[0].0;
            best_policy = scored[0].1.clone();
        }

        let elite_count = (population_size / 5).max(1).min(population_size);
        let mut next_population = scored
            .iter()
            .take(elite_count)
            .map(|(_, policy)| policy.clone())
            .collect::<Vec<_>>();
        while next_population.len() < population_size {
            let parent_index = rng.gen_range(0..elite_count);
            let mut child = scored[parent_index].1.clone();
            child.mutate(&mut rng, MUTATION_RATE, MUTATION_STEP);
            next_population.push(child);
        }
        population = next_population;
    }

    (
        best_policy,
        TrainingMetadata {
            purge_bars: 0,
            iterations: generations,
            population: population_size,
            learning_rate: None,
            mutation_rate: Some(MUTATION_RATE),
            mutation_step: Some(MUTATION_STEP),
            reward_scale: None,
            objective: "maximize sum of selected fixed-horizon diagnostic rewards on the chronological training prefix".to_string(),
        },
    )
}

fn train_rl(
    samples: &[Sample],
    train_range: Range<usize>,
    actions: &[Action],
    epochs: usize,
    population: usize,
    seed: u64,
) -> (LinearPolicy, TrainingMetadata) {
    const LEARNING_RATE: f64 = 0.05;
    const L2: f64 = 1.0e-4;
    const GRADIENT_CLIP: f64 = 1.0;
    let feature_count = samples[0].features.len() - 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut policy = LinearPolicy::random(actions, feature_count, &mut rng, 0.01);
    let reward_scale = training_reward_scale(samples, train_range.clone(), actions);
    let train_count = train_range.end - train_range.start;

    for _epoch in 0..epochs {
        let mut gradients = vec![vec![0.0; feature_count + 1]; actions.len()];
        for sample in &samples[train_range.clone()] {
            let probabilities = policy.probabilities(&sample.features);
            let expected_reward = actions
                .iter()
                .enumerate()
                .map(|(index, action)| {
                    probabilities[index] * action.reward(sample.rewards) / reward_scale
                })
                .sum::<f64>();
            for (action_index, action) in actions.iter().enumerate() {
                let scaled_reward = action.reward(sample.rewards) / reward_scale;
                let logit_gradient =
                    probabilities[action_index] * (scaled_reward - expected_reward);
                for (gradient, feature) in gradients[action_index].iter_mut().zip(&sample.features)
                {
                    *gradient += logit_gradient * feature;
                }
            }
        }
        for (row, gradient_row) in policy.weights.iter_mut().zip(gradients) {
            for (weight, gradient) in row.iter_mut().zip(gradient_row) {
                let update = (gradient / train_count as f64 - L2 * *weight)
                    .clamp(-GRADIENT_CLIP, GRADIENT_CLIP);
                *weight = (*weight + LEARNING_RATE * update).clamp(-20.0, 20.0);
            }
        }
    }

    (
        policy,
        TrainingMetadata {
            purge_bars: 0,
            iterations: epochs,
            population,
            learning_rate: Some(LEARNING_RATE),
            mutation_rate: None,
            mutation_step: None,
            reward_scale: Some(reward_scale),
            objective: "maximize expected selected-action fixed-horizon diagnostic reward with a full-batch policy-gradient update on the chronological training prefix".to_string(),
        },
    )
}

fn training_reward_scale(samples: &[Sample], train_range: Range<usize>, actions: &[Action]) -> f64 {
    let denominator = ((train_range.end - train_range.start) * actions.len()) as f64;
    let mean_abs = samples[train_range]
        .iter()
        .flat_map(|sample| {
            actions
                .iter()
                .map(|action| action.reward(sample.rewards).abs())
        })
        .sum::<f64>()
        / denominator;
    mean_abs.max(1.0)
}

fn score_policy(policy: &LinearPolicy, samples: &[Sample], range: Range<usize>) -> f64 {
    samples[range]
        .iter()
        .map(|sample| {
            let action = policy.choose_action(&sample.features);
            action.reward(sample.rewards)
        })
        .sum()
}

fn fixed_metrics(
    samples: &[Sample],
    splits: &SplitRanges,
) -> BTreeMap<String, BTreeMap<String, Summary>> {
    let mut result = BTreeMap::new();
    for action in Action::ALL {
        let mut per_split = BTreeMap::new();
        per_split.insert(
            "train".to_string(),
            evaluate_fixed(samples, splits.train(), action),
        );
        per_split.insert(
            "validation".to_string(),
            evaluate_fixed(samples, splits.validation(), action),
        );
        per_split.insert(
            "holdout".to_string(),
            evaluate_fixed(samples, splits.holdout(), action),
        );
        result.insert(action.label().to_string(), per_split);
    }
    result
}

fn learned_metrics(
    samples: &[Sample],
    splits: &SplitRanges,
    policy: &LinearPolicy,
) -> BTreeMap<String, Summary> {
    let mut result = BTreeMap::new();
    result.insert(
        "train".to_string(),
        evaluate_learned(samples, splits.train(), policy),
    );
    result.insert(
        "validation".to_string(),
        evaluate_learned(samples, splits.validation(), policy),
    );
    result.insert(
        "holdout".to_string(),
        evaluate_learned(samples, splits.holdout(), policy),
    );
    result
}

fn evaluate_fixed(samples: &[Sample], range: Range<usize>, action: Action) -> Summary {
    evaluate_range(samples, range, |_| action)
}

fn evaluate_learned(samples: &[Sample], range: Range<usize>, policy: &LinearPolicy) -> Summary {
    evaluate_range(samples, range, |sample| {
        policy.choose_action(&sample.features)
    })
}

fn evaluate_range<F>(samples: &[Sample], range: Range<usize>, mut choose: F) -> Summary
where
    F: FnMut(&Sample) -> Action,
{
    let mut action_counts = Action::ALL
        .into_iter()
        .map(|action| (action.label().to_string(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut pnls = Vec::with_capacity(range.end - range.start);
    for sample in &samples[range] {
        let action = choose(sample);
        *action_counts.entry(action.label().to_string()).or_default() += 1;
        pnls.push(action.reward(sample.rewards));
    }
    let event_count = pnls.len();
    let sum_pnl = pnls.iter().sum::<f64>();
    let mean_pnl = if event_count == 0 {
        0.0
    } else {
        sum_pnl / event_count as f64
    };
    let win_rate = if event_count == 0 {
        0.0
    } else {
        pnls.iter().filter(|pnl| **pnl > 0.0).count() as f64 / event_count as f64
    };
    let max_drawdown = max_drawdown(&pnls);
    let action_fractions = action_counts
        .iter()
        .map(|(action, count)| {
            (
                action.clone(),
                if event_count == 0 {
                    0.0
                } else {
                    *count as f64 / event_count as f64
                },
            )
        })
        .collect();
    Summary {
        event_count,
        action_counts,
        action_fractions,
        sum_pnl,
        mean_pnl,
        win_rate,
        max_drawdown,
    }
}

fn max_drawdown(pnls: &[f64]) -> f64 {
    let mut cumulative: f64 = 0.0;
    let mut peak: f64 = 0.0;
    let mut drawdown: f64 = 0.0;
    for pnl in pnls {
        cumulative += pnl;
        peak = peak.max(cumulative);
        drawdown = drawdown.max(peak - cumulative);
    }
    drawdown
}

fn write_json_artifacts<P: Serialize, M: Serialize>(
    outdir: &Path,
    policy: &P,
    metrics: &M,
    overwrite: bool,
) -> Result<()> {
    write_json_artifacts_internal(outdir, policy, metrics, overwrite, None)
}

fn write_json_artifacts_internal<P: Serialize, M: Serialize>(
    outdir: &Path,
    policy: &P,
    metrics: &M,
    overwrite: bool,
    #[cfg(test)] fail_before_publication: Option<usize>,
    #[cfg(not(test))] _fail_before_publication: Option<usize>,
) -> Result<()> {
    recover_incomplete_transaction(outdir)?;
    cleanup_stale_temporary_files(outdir);

    let policy_path = outdir.join(POLICY_FILE);
    let metrics_path = outdir.join(METRICS_FILE);
    let manifest_path = outdir.join(ARTIFACT_MANIFEST_FILE);
    let transaction_path = outdir.join(TRANSACTION_FILE);
    let destinations = [
        policy_path.clone(),
        metrics_path.clone(),
        manifest_path.clone(),
    ];

    if !overwrite {
        for destination in &destinations {
            if path_exists(destination) {
                bail!(
                    "refusing to overwrite existing meta-gate artifact {}; pass --overwrite to replace it explicitly",
                    destination.display()
                );
            }
        }
    }

    let generation = next_generation();
    let mut temporary_paths = Vec::with_capacity(destinations.len());
    let staged = (|| -> Result<()> {
        let policy_temp = write_json_temp(&policy_path, policy)?;
        temporary_paths.push(policy_temp.clone());
        let metrics_temp = write_json_temp(&metrics_path, metrics)?;
        temporary_paths.push(metrics_temp.clone());
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA.to_string(),
            status: ARTIFACT_STATUS_COMPLETE.to_string(),
            generation: generation.clone(),
            policy_file: POLICY_FILE.to_string(),
            metrics_file: METRICS_FILE.to_string(),
            policy_sha256: sha256_file(&policy_temp)?,
            metrics_sha256: sha256_file(&metrics_temp)?,
        };
        let manifest_temp = write_json_temp(&manifest_path, &manifest)?;
        temporary_paths.push(manifest_temp);
        Ok(())
    })();
    if let Err(error) = staged {
        cleanup_paths(&temporary_paths);
        return Err(error);
    }

    let transaction = TransactionRecord {
        schema_version: ARTIFACT_MANIFEST_SCHEMA.to_string(),
        generation: generation.clone(),
        artifacts: destinations
            .iter()
            .map(|destination| TransactionArtifact {
                destination: destination
                    .file_name()
                    .expect("artifact destination has a file name")
                    .to_string_lossy()
                    .into_owned(),
                backup: path_exists(destination).then(|| {
                    backup_path(outdir, destination, &generation)
                        .file_name()
                        .expect("backup destination has a file name")
                        .to_string_lossy()
                        .into_owned()
                }),
            })
            .collect(),
    };
    let transaction_temp = match write_json_temp(&transaction_path, &transaction) {
        Ok(path) => path,
        Err(error) => {
            cleanup_paths(&temporary_paths);
            return Err(error);
        }
    };
    if let Err(error) = publish_json_temp(&transaction_temp, &transaction_path, false) {
        cleanup_paths(&temporary_paths);
        cleanup_paths(std::slice::from_ref(&transaction_temp));
        return Err(error);
    }
    cleanup_paths(std::slice::from_ref(&transaction_temp));
    sync_directory(outdir)?;

    let mut backed_up = Vec::new();
    let backup_result = (|| -> Result<()> {
        for artifact in &transaction.artifacts {
            let destination = outdir.join(&artifact.destination);
            if let Some(backup_name) = &artifact.backup {
                let backup = outdir.join(backup_name);
                fs::rename(&destination, &backup).with_context(|| {
                    format!(
                        "move existing meta-gate artifact {} to transaction backup {}",
                        destination.display(),
                        backup.display()
                    )
                })?;
                backed_up.push((destination, backup));
            }
        }
        Ok(())
    })();
    if let Err(error) = backup_result {
        cleanup_paths(&temporary_paths);
        rollback_transaction(outdir, &transaction, &[])
            .context("rollback after meta-gate backup failure")?;
        cleanup_paths(std::slice::from_ref(&transaction_path));
        return Err(error);
    }
    sync_directory(outdir)?;

    let mut published = Vec::new();
    let publication = (|| -> Result<()> {
        for (index, (temporary, destination)) in
            temporary_paths.iter().zip(&destinations).enumerate()
        {
            #[cfg(not(test))]
            let _ = index;
            #[cfg(test)]
            if fail_before_publication == Some(index) {
                bail!("injected meta-gate publication failure before artifact {index}");
            }
            publish_json_temp(temporary, destination, false)?;
            published.push(destination.clone());
        }
        validate_artifact_pair(outdir, Some(&generation))?;
        Ok(())
    })();

    cleanup_paths(&temporary_paths);
    if let Err(error) = publication {
        rollback_transaction(outdir, &transaction, &published)
            .context("rollback after meta-gate publication failure")?;
        cleanup_paths(std::slice::from_ref(&transaction_path));
        return Err(error);
    }

    cleanup_paths(
        &backed_up
            .iter()
            .map(|(_, backup)| backup.clone())
            .collect::<Vec<_>>(),
    );
    cleanup_paths(std::slice::from_ref(&transaction_path));
    sync_directory(outdir)?;
    Ok(())
}

fn write_json_temp<T: Serialize>(destination: &Path, value: &T) -> Result<PathBuf> {
    let parent = destination.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot create a same-directory temporary file for {}",
            destination.display()
        )
    })?;
    let file_name = destination
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output path {} has no file name", destination.display()))?;

    for _ in 0..100 {
        let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary_name = format!(
            ".{}.tmp-{}-{}",
            file_name.to_string_lossy(),
            std::process::id(),
            sequence
        );
        let temporary = parent.join(temporary_name);
        let file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create temporary JSON file {}", temporary.display())
                });
            }
        };

        let result = (|| -> Result<()> {
            let mut file = file;
            serde_json::to_writer_pretty(&mut file, value)
                .with_context(|| format!("write JSON {}", temporary.display()))?;
            file.flush()
                .with_context(|| format!("flush JSON {}", temporary.display()))?;
            file.sync_all()
                .with_context(|| format!("sync JSON {}", temporary.display()))?;
            Ok(())
        })();

        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        return Ok(temporary);
    }

    bail!(
        "could not allocate a unique same-directory temporary file for {}",
        destination.display()
    )
}

fn publish_json_temp(temporary: &Path, destination: &Path, overwrite: bool) -> Result<()> {
    if overwrite {
        fs::rename(temporary, destination).with_context(|| {
            format!(
                "replace JSON artifact {} from {}",
                destination.display(),
                temporary.display()
            )
        })?;
    } else {
        // hard_link is an atomic no-replace publication on the same
        // filesystem. Removing the temporary link after publication leaves
        // the destination containing exactly the fully written bytes.
        fs::hard_link(temporary, destination).with_context(|| {
            format!(
                "publish JSON artifact {} without replacing an existing file",
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn next_generation() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn backup_path(outdir: &Path, destination: &Path, generation: &str) -> PathBuf {
    outdir.join(format!(
        ".meta-gate-backup-{generation}-{}",
        destination
            .file_name()
            .expect("backup destination has a file name")
            .to_string_lossy()
    ))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {} for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_artifact_pair(outdir: &Path, expected_generation: Option<&str>) -> Result<()> {
    let manifest_path = outdir.join(ARTIFACT_MANIFEST_FILE);
    let manifest: ArtifactManifest = serde_json::from_reader(
        File::open(&manifest_path)
            .with_context(|| format!("open artifact manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse artifact manifest {}", manifest_path.display()))?;
    if manifest.schema_version != ARTIFACT_MANIFEST_SCHEMA
        || manifest.status != ARTIFACT_STATUS_COMPLETE
        || manifest.policy_file != POLICY_FILE
        || manifest.metrics_file != METRICS_FILE
    {
        bail!(
            "invalid meta-gate artifact manifest {}",
            manifest_path.display()
        );
    }
    if let Some(expected_generation) = expected_generation {
        if manifest.generation != expected_generation {
            bail!(
                "meta-gate artifact manifest generation mismatch: expected {}, found {}",
                expected_generation,
                manifest.generation
            );
        }
    }
    let policy_path = outdir.join(POLICY_FILE);
    let metrics_path = outdir.join(METRICS_FILE);
    if sha256_file(&policy_path)? != manifest.policy_sha256
        || sha256_file(&metrics_path)? != manifest.metrics_sha256
    {
        bail!("meta-gate policy/metrics pair does not match its completion manifest");
    }
    Ok(())
}

fn recover_incomplete_transaction(outdir: &Path) -> Result<()> {
    let transaction_path = outdir.join(TRANSACTION_FILE);
    if !path_exists(&transaction_path) {
        return Ok(());
    }
    reject_symlink(&transaction_path, "transaction journal")?;
    let transaction: TransactionRecord =
        serde_json::from_reader(File::open(&transaction_path).with_context(|| {
            format!("open incomplete transaction {}", transaction_path.display())
        })?)
        .with_context(|| {
            format!(
                "parse incomplete transaction {}",
                transaction_path.display()
            )
        })?;
    validate_transaction_record(outdir, &transaction)?;

    if validate_artifact_pair(outdir, Some(&transaction.generation)).is_ok() {
        cleanup_transaction_backups(outdir, &transaction)?;
        cleanup_paths(std::slice::from_ref(&transaction_path));
        return Ok(());
    }

    rollback_transaction(outdir, &transaction, &[])?;
    cleanup_paths(std::slice::from_ref(&transaction_path));
    Ok(())
}

fn validate_transaction_record(outdir: &Path, transaction: &TransactionRecord) -> Result<()> {
    if transaction.schema_version != ARTIFACT_MANIFEST_SCHEMA {
        bail!(
            "unsupported meta-gate transaction schema; refusing recovery without touching artifacts"
        );
    }
    if transaction.generation.is_empty() || !is_single_path_component(&transaction.generation) {
        bail!(
            "invalid meta-gate transaction generation; refusing recovery without touching artifacts"
        );
    }
    if transaction.artifacts.len() != EXPECTED_ARTIFACT_FILES.len() {
        bail!(
            "meta-gate transaction must contain exactly {} artifacts; refusing recovery without touching artifacts",
            EXPECTED_ARTIFACT_FILES.len()
        );
    }

    let mut backups = BTreeMap::new();
    for (index, (artifact, expected_destination)) in transaction
        .artifacts
        .iter()
        .zip(EXPECTED_ARTIFACT_FILES)
        .enumerate()
    {
        if artifact.destination != expected_destination
            || !is_single_path_component(&artifact.destination)
        {
            bail!(
                "meta-gate transaction artifact {index} has unsafe destination; expected {expected_destination:?}; refusing recovery without touching artifacts"
            );
        }
        let destination = outdir.join(expected_destination);
        ensure_safe_journal_entry(&destination, "destination")?;

        if let Some(backup_name) = artifact.backup.as_deref() {
            let expected_backup_name = backup_path(outdir, &destination, &transaction.generation)
                .file_name()
                .expect("backup destination has a file name")
                .to_string_lossy()
                .into_owned();
            if backup_name != expected_backup_name
                || !is_single_path_component(backup_name)
                || backups.insert(backup_name.to_string(), index).is_some()
            {
                bail!(
                    "meta-gate transaction artifact {index} has an unexpected or duplicate backup name; expected {expected_backup_name:?}; refusing recovery without touching artifacts"
                );
            }
            ensure_safe_journal_entry(&outdir.join(backup_name), "backup")?;
        }
    }
    Ok(())
}

fn is_single_path_component(value: &str) -> bool {
    if value.is_empty() || value.contains('/') || value.contains('\\') {
        return false;
    }
    let path = Path::new(value);
    let mut components = path.components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(component)), None) if component == path.as_os_str()
    )
}

fn ensure_safe_journal_entry(path: &Path, kind: &str) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.file_type().is_symlink() {
        bail!(
            "meta-gate transaction {kind} {} is a symlink; refusing recovery without touching artifacts",
            path.display()
        );
    }
    if !metadata.file_type().is_file() {
        bail!(
            "meta-gate transaction {kind} {} is not a regular file; refusing recovery without touching artifacts",
            path.display()
        );
    }
    Ok(())
}

fn reject_symlink(path: &Path, kind: &str) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() {
            bail!(
                "meta-gate {kind} {} is a symlink; refusing recovery",
                path.display()
            );
        }
    }
    Ok(())
}

fn rollback_transaction(
    outdir: &Path,
    transaction: &TransactionRecord,
    published: &[PathBuf],
) -> Result<()> {
    validate_transaction_record(outdir, transaction)?;
    cleanup_paths(published);
    for artifact in transaction.artifacts.iter().rev() {
        let destination = outdir.join(&artifact.destination);
        if let Some(backup_name) = &artifact.backup {
            let backup = outdir.join(backup_name);
            if path_exists(&backup) {
                remove_path(&destination);
                fs::rename(&backup, &destination).with_context(|| {
                    format!(
                        "restore meta-gate transaction backup {} to {}",
                        backup.display(),
                        destination.display()
                    )
                })?;
            }
        } else {
            remove_path(&destination);
        }
    }
    Ok(())
}

fn cleanup_transaction_backups(outdir: &Path, transaction: &TransactionRecord) -> Result<()> {
    validate_transaction_record(outdir, transaction)?;
    for artifact in &transaction.artifacts {
        if let Some(backup_name) = &artifact.backup {
            remove_path(&outdir.join(backup_name));
        }
    }
    Ok(())
}

fn cleanup_stale_temporary_files(outdir: &Path) {
    let prefixes = [
        ".policy.json.tmp-",
        ".metrics.json.tmp-",
        ".meta-gate-manifest.json.tmp-",
        ".meta-gate-transaction.json.tmp-",
        "..meta-gate-manifest.json.tmp-",
        "..meta-gate-transaction.json.tmp-",
    ];
    let Ok(entries) = fs::read_dir(outdir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            remove_path(&entry.path());
        }
    }
}

fn remove_path(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if metadata.file_type().is_dir() {
        let _ = fs::remove_dir(path);
    } else {
        let _ = fs::remove_file(path);
    }
}

fn sync_directory(path: &Path) -> Result<()> {
    let directory = File::open(path)
        .with_context(|| format!("open output directory {} for syncing", path.display()))?;
    directory
        .sync_all()
        .with_context(|| format!("sync output directory {}", path.display()))?;
    Ok(())
}

fn path_exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

fn cleanup_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn print_report(metrics: &MetricsArtifact) {
    println!("{FIXED_HORIZON_WARNING}");
    println!(
        "algorithm={} seed={} purge_bars={} features={} input/retained/purged={}/{}/{} train/validation/holdout={}/{}/{}",
        metrics.algorithm,
        metrics.seed,
        metrics.split_ranges.purge_bars,
        metrics.feature_names.len(),
        metrics.split_counts.total,
        metrics.split_counts.retained,
        metrics.split_counts.purged,
        metrics.split_counts.train,
        metrics.split_counts.validation,
        metrics.split_counts.holdout
    );
    for (policy_name, per_split) in &metrics.fixed_policies {
        for (split, summary) in per_split {
            print_summary(&format!("fixed/{policy_name}/{split}"), summary);
        }
    }
    for (split, summary) in &metrics.learned_policy {
        print_summary(&format!("learned/{split}"), summary);
    }
}

fn print_summary(name: &str, summary: &Summary) {
    println!(
        "{name}: events={} sum_pnl={:.6} mean_pnl={:.6} win_rate={:.4} max_drawdown={:.6} action_counts={:?}",
        summary.event_count,
        summary.sum_pnl,
        summary.mean_pnl,
        summary.win_rate,
        summary.max_drawdown,
        summary.action_counts
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{NamedFrom, Series};

    #[test]
    fn causal_feature_list_excludes_future_outcomes() {
        assert_eq!(CAUSAL_FEATURES.len(), 11);
        for future_column in ["normal_pnl", "skip_pnl", "invert_pnl", "oracle_action"] {
            assert!(!CAUSAL_FEATURES.contains(&future_column));
        }
    }

    #[test]
    fn split_boundaries_are_chronological_and_cover_all_rows() {
        let events = synthetic_events(10, 0);
        let splits = split_ranges(&events, 10.0 / 10.0 * 0.6, 0.2, 0).unwrap();
        assert_eq!(splits.train(), 0..6);
        assert_eq!(splits.validation(), 6..8);
        assert_eq!(splits.holdout(), 8..10);
        assert_eq!(split_counts(&splits).total, 10);
    }

    #[test]
    fn purge_removes_overlapping_boundary_events_and_embargoes_right_side() {
        let mut events = synthetic_events(15, 5);
        // Nominal boundaries are event offsets 9 and 12.  The first
        // retained validation row is source row 100, so the event at source
        // row 80 with horizon 100 must be purged from training.  The first
        // retained holdout row is source row 130, so the last nominal
        // validation event with horizon 130 must also be purged.
        events[8].horizon_row_idx = 100;
        events[11].horizon_row_idx = 130;

        let splits = split_ranges(&events, 0.6, 0.2, 5).unwrap();

        assert_eq!(splits.train(), 0..8);
        assert_eq!(splits.validation(), 10..11);
        assert_eq!(splits.holdout(), 13..15);
        assert_eq!(splits.purge_bars, 5);
        assert_eq!(split_counts(&splits).purged, 4);

        let first_validation_row = events[splits.validation_start].row_idx;
        let first_holdout_row = events[splits.holdout_start].row_idx;
        assert!(
            events[splits.train()]
                .iter()
                .all(|event| { event.horizon_row_idx + splits.purge_bars < first_validation_row })
        );
        assert!(
            events[splits.validation()]
                .iter()
                .all(|event| { event.horizon_row_idx + splits.purge_bars < first_holdout_row })
        );
    }

    fn synthetic_events(count: usize, horizon_offset: usize) -> Vec<EventRow> {
        (0..count)
            .map(|index| EventRow {
                row_idx: index * 10,
                horizon_row_idx: index * 10 + horizon_offset,
                raw_features: vec![0.0; CAUSAL_FEATURES.len()],
                rewards: [0.0; 3],
            })
            .collect()
    }

    #[test]
    fn policy_output_uses_only_selected_actions_and_is_deterministic() {
        let actions = vec![Action::Normal, Action::Skip];
        let mut policy = LinearPolicy::zero(&actions, 2);
        policy.weights[0][0] = 2.0;
        policy.weights[1][0] = -2.0;
        let features = vec![1.0, 0.0, 1.0];
        assert_eq!(policy.choose_action(&features), Action::Normal);
        assert_eq!(policy.choose_action(&features), Action::Normal);
        assert_eq!(policy.probabilities(&features).len(), 2);
    }

    #[test]
    fn provenance_rejects_mixed_identity_rows() {
        let config = MetaGateConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let frame = DataFrame::new(vec![
            Series::new("schema_version".into(), vec![META_EVENT_SCHEMA; 2]).into(),
            Series::new("feature_schema".into(), vec![config.feature_schema(); 2]).into(),
            Series::new("config_json".into(), vec![config_json; 2]).into(),
            Series::new("instrument".into(), vec!["GC", "ES"]).into(),
            Series::new("contract".into(), vec!["GCZ6"; 2]).into(),
            Series::new("source_path".into(), vec!["/tmp/bars.parquet"; 2]).into(),
            Series::new("source_size_bytes".into(), vec![100_i64; 2]).into(),
            Series::new("source_row_count".into(), vec![1000_i64; 2]).into(),
            Series::new("timestamp_source".into(), vec!["ts_ns"; 2]).into(),
        ])
        .unwrap();

        let error = validate_event_provenance(&frame).unwrap_err();
        assert!(error.to_string().contains("instrument is inconsistent"));
    }

    #[test]
    fn regenerated_source_rejects_tampered_payload_even_when_hashes_are_rewritten() {
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
        let directory = test_output_dir();
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
                vec![snapshot.size_bytes; frame.height()],
            ))
            .unwrap();
        frame
            .with_column(Series::new(
                "source_row_count".into(),
                vec![snapshot.bars.close.len() as i64; frame.height()],
            ))
            .unwrap();
        (directory, frame)
    }

    #[test]
    fn artifacts_publish_without_replacing_or_leaving_temporary_files() {
        let outdir = test_output_dir();
        let policy_path = outdir.join("policy.json");
        let metrics_path = outdir.join("metrics.json");

        write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path).unwrap()).unwrap(),
            serde_json::json!({"version": 1})
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&metrics_path).unwrap()).unwrap(),
            serde_json::json!({"events": 2})
        );
        validate_artifact_pair(&outdir, None).unwrap();
        assert!(!has_json_temp_files(&outdir));

        let error = write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 3}),
            &serde_json::json!({"events": 4}),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("refusing to overwrite"));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path).unwrap()).unwrap(),
            serde_json::json!({"version": 1})
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&metrics_path).unwrap()).unwrap(),
            serde_json::json!({"events": 2})
        );
        validate_artifact_pair(&outdir, None).unwrap();
        assert!(!has_json_temp_files(&outdir));

        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn explicit_overwrite_replaces_both_artifacts() {
        let outdir = test_output_dir();
        write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        let first_manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(outdir.join(ARTIFACT_MANIFEST_FILE)).unwrap())
                .unwrap();
        write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 2}),
            &serde_json::json!({"events": 3}),
            true,
        )
        .unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &fs::read(outdir.join("policy.json")).unwrap()
            )
            .unwrap(),
            serde_json::json!({"version": 2})
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &fs::read(outdir.join("metrics.json")).unwrap()
            )
            .unwrap(),
            serde_json::json!({"events": 3})
        );
        let second_manifest: ArtifactManifest =
            serde_json::from_slice(&fs::read(outdir.join(ARTIFACT_MANIFEST_FILE)).unwrap())
                .unwrap();
        assert_ne!(first_manifest.generation, second_manifest.generation);
        validate_artifact_pair(&outdir, Some(&second_manifest.generation)).unwrap();
        assert!(!has_json_temp_files(&outdir));

        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn failed_overwrite_restores_the_previous_complete_pair() {
        let outdir = test_output_dir();
        write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        let old_policy = fs::read(outdir.join(POLICY_FILE)).unwrap();
        let old_metrics = fs::read(outdir.join(METRICS_FILE)).unwrap();
        let old_manifest = fs::read(outdir.join(ARTIFACT_MANIFEST_FILE)).unwrap();

        let error = write_json_artifacts_internal(
            &outdir,
            &serde_json::json!({"version": 2}),
            &serde_json::json!({"events": 3}),
            true,
            Some(1),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected meta-gate publication failure")
        );
        assert_eq!(fs::read(outdir.join(POLICY_FILE)).unwrap(), old_policy);
        assert_eq!(fs::read(outdir.join(METRICS_FILE)).unwrap(), old_metrics);
        assert_eq!(
            fs::read(outdir.join(ARTIFACT_MANIFEST_FILE)).unwrap(),
            old_manifest
        );
        validate_artifact_pair(&outdir, None).unwrap();
        assert!(!path_exists(&outdir.join(TRANSACTION_FILE)));
        assert!(!has_json_temp_files(&outdir));
        assert!(!has_backup_files(&outdir));

        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn interrupted_transaction_is_invalid_until_recovered() {
        let outdir = test_output_dir();
        write_json_artifacts(
            &outdir,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        let generation = "interrupted-test".to_string();
        let destinations = [POLICY_FILE, METRICS_FILE, ARTIFACT_MANIFEST_FILE];
        let mut artifacts = Vec::new();
        for destination_name in destinations {
            let destination = outdir.join(destination_name);
            let backup = backup_path(&outdir, &destination, &generation);
            fs::rename(&destination, &backup).unwrap();
            artifacts.push(TransactionArtifact {
                destination: destination_name.to_string(),
                backup: Some(backup.file_name().unwrap().to_string_lossy().into_owned()),
            });
        }
        let transaction = TransactionRecord {
            schema_version: ARTIFACT_MANIFEST_SCHEMA.to_string(),
            generation,
            artifacts,
        };
        fs::write(
            outdir.join(TRANSACTION_FILE),
            serde_json::to_vec_pretty(&transaction).unwrap(),
        )
        .unwrap();
        fs::write(outdir.join(POLICY_FILE), br#"{"partial":true}"#).unwrap();

        recover_incomplete_transaction(&outdir).unwrap();
        validate_artifact_pair(&outdir, None).unwrap();
        assert!(!path_exists(&outdir.join(TRANSACTION_FILE)));
        assert!(!has_backup_files(&outdir));
        assert!(!has_json_temp_files(&outdir));

        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn traversal_transaction_journal_is_rejected_before_recovery_touches_outside_path() {
        let outdir = test_output_dir();
        let outside = test_output_dir();
        let outside_path = outside.join("protected.json");
        fs::write(&outside_path, b"must remain unchanged").unwrap();
        let traversal_backup = format!(
            "../{}/{}",
            outside.file_name().unwrap().to_string_lossy(),
            outside_path.file_name().unwrap().to_string_lossy()
        );
        let transaction = TransactionRecord {
            schema_version: ARTIFACT_MANIFEST_SCHEMA.to_string(),
            generation: "traversal-test".to_string(),
            artifacts: vec![
                TransactionArtifact {
                    destination: POLICY_FILE.to_string(),
                    backup: Some(traversal_backup),
                },
                TransactionArtifact {
                    destination: METRICS_FILE.to_string(),
                    backup: None,
                },
                TransactionArtifact {
                    destination: ARTIFACT_MANIFEST_FILE.to_string(),
                    backup: None,
                },
            ],
        };
        fs::write(
            outdir.join(TRANSACTION_FILE),
            serde_json::to_vec_pretty(&transaction).unwrap(),
        )
        .unwrap();

        let error = recover_incomplete_transaction(&outdir).unwrap_err();
        assert!(error.to_string().contains("refusing recovery"));
        assert_eq!(fs::read(&outside_path).unwrap(), b"must remain unchanged");
        assert!(path_exists(&outdir.join(TRANSACTION_FILE)));

        fs::remove_dir_all(outdir).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn transaction_recovery_rejects_a_confined_but_wrong_backup_name() {
        let outdir = test_output_dir();
        let generation = "exact-backup-test".to_string();
        let transaction = TransactionRecord {
            schema_version: ARTIFACT_MANIFEST_SCHEMA.to_string(),
            generation: generation.clone(),
            artifacts: vec![
                TransactionArtifact {
                    destination: POLICY_FILE.to_string(),
                    backup: Some(format!(".meta-gate-backup-{generation}-{METRICS_FILE}")),
                },
                TransactionArtifact {
                    destination: METRICS_FILE.to_string(),
                    backup: None,
                },
                TransactionArtifact {
                    destination: ARTIFACT_MANIFEST_FILE.to_string(),
                    backup: None,
                },
            ],
        };
        fs::write(
            outdir.join(TRANSACTION_FILE),
            serde_json::to_vec_pretty(&transaction).unwrap(),
        )
        .unwrap();

        let error = recover_incomplete_transaction(&outdir).unwrap_err();
        assert!(error.to_string().contains("unexpected or duplicate backup"));
        assert!(path_exists(&outdir.join(TRANSACTION_FILE)));
        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn sync_directory_propagates_open_errors() {
        let outdir = test_output_dir();
        let missing = outdir.join("missing");
        let error = sync_directory(&missing).unwrap_err();
        assert!(error.to_string().contains("open output directory"));
        fs::remove_dir_all(outdir).unwrap();
    }

    #[test]
    fn missing_or_tampered_manifest_is_rejected() {
        let missing = test_output_dir();
        write_json_artifacts(
            &missing,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        fs::remove_file(missing.join(ARTIFACT_MANIFEST_FILE)).unwrap();
        assert!(validate_artifact_pair(&missing, None).is_err());
        fs::remove_dir_all(missing).unwrap();

        let tampered = test_output_dir();
        write_json_artifacts(
            &tampered,
            &serde_json::json!({"version": 1}),
            &serde_json::json!({"events": 2}),
            false,
        )
        .unwrap();
        let manifest_path = tampered.join(ARTIFACT_MANIFEST_FILE);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["status"] = serde_json::Value::String("staged".to_string());
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let error = validate_artifact_pair(&tampered, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid meta-gate artifact manifest")
        );
        fs::remove_dir_all(tampered).unwrap();
    }

    fn test_output_dir() -> PathBuf {
        for _ in 0..100 {
            let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "midas-train-meta-gate-test-{}-{}",
                std::process::id(),
                sequence
            ));
            if fs::create_dir(&path).is_ok() {
                return path;
            }
        }
        panic!("could not create a unique temporary test directory");
    }

    fn has_json_temp_files(outdir: &Path) -> bool {
        fs::read_dir(outdir).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        })
    }

    fn has_backup_files(outdir: &Path) -> bool {
        fs::read_dir(outdir).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".meta-gate-backup-")
        })
    }
}
