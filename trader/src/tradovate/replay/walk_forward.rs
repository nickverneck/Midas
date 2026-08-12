//! Persisted, bounded walk-forward/out-of-sample plans for replay sweeps.
//!
//! A walk-forward plan is intentionally a planning boundary rather than a
//! second replay runner.  It takes an immutable [`ReplaySweepSpec`], derives
//! chronological train/validation/test views, and writes one normal sweep
//! specification per phase.  Each generated specification can therefore be
//! executed by the existing `run-replay-sweep` command without changing the
//! sweep result schema or sharing replay state between phases.

use super::sweep::ReplaySweepSpec;
use crate::replay_cache::{ReplayDatasetSessionPreset, ReplayDatasetView};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const REPLAY_WALK_FORWARD_PLAN_SCHEMA_VERSION: u32 = 1;

/// Hard bounds keep a malformed or accidentally broad plan from generating a
/// large number of independent sweep trees.  A single fold produces at most
/// three normal sweep specifications.
const MAX_WALK_FORWARD_FOLDS: usize = 64;
const MAX_WALK_FORWARD_PHASES: usize = MAX_WALK_FORWARD_FOLDS * 3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayWalkForwardPhase {
    Train,
    Validation,
    Test,
}

impl ReplayWalkForwardPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

/// Chronological evaluation window for one generated sweep specification.
///
/// `evaluation_start`/`evaluation_end` are half-open (`[start, end)`) just
/// like replay dataset views.  Indicators can use the recorded warmup range,
/// but the generated view remains flat until `evaluation_start` according to
/// the existing replay warmup policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayWalkForwardWindow {
    pub(crate) phase: ReplayWalkForwardPhase,
    pub(crate) evaluation_start: DateTime<Utc>,
    pub(crate) evaluation_end: DateTime<Utc>,
    pub(crate) warmup_start: DateTime<Utc>,
    pub(crate) sweep_id: String,
    pub(crate) view_id: String,
    pub(crate) spec_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayWalkForwardFold {
    pub(crate) fold_index: usize,
    pub(crate) train: ReplayWalkForwardWindow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) validation: Option<ReplayWalkForwardWindow>,
    pub(crate) test: ReplayWalkForwardWindow,
}

/// Persisted policy used to derive the fold boundaries.
///
/// Fractions are relative to the original dataset view's evaluation span.
/// A fold is a rolling window: the next fold starts at `step_fraction` of the
/// original span after the prior fold.  `purge_seconds` creates a gap at each
/// train/validation and validation/test boundary, preventing adjacent samples
/// from leaking across the model-selection boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayWalkForwardConfig {
    pub(crate) train_fraction: f64,
    pub(crate) validation_fraction: f64,
    pub(crate) test_fraction: f64,
    pub(crate) folds: usize,
    pub(crate) step_fraction: Option<f64>,
    pub(crate) purge_seconds: u64,
    /// When omitted, each generated view uses the base view's warmup policy.
    /// A larger explicit value is rejected because the base view does not
    /// guarantee source coverage before its existing warmup start.
    pub(crate) warmup_seconds: Option<u64>,
}

impl Default for ReplayWalkForwardConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.70,
            validation_fraction: 0.15,
            test_fraction: 0.15,
            folds: 1,
            step_fraction: None,
            purge_seconds: 0,
            warmup_seconds: None,
        }
    }
}

impl ReplayWalkForwardConfig {
    fn validate(&self) -> Result<()> {
        validate_fraction(self.train_fraction, "train_fraction", false)?;
        validate_fraction(self.validation_fraction, "validation_fraction", true)?;
        validate_fraction(self.test_fraction, "test_fraction", false)?;
        if self.folds == 0 || self.folds > MAX_WALK_FORWARD_FOLDS {
            bail!("walk-forward folds must be between 1 and {MAX_WALK_FORWARD_FOLDS}");
        }
        if let Some(step) = self.step_fraction {
            validate_fraction(step, "step_fraction", false)?;
        }
        let phase_total = self.train_fraction + self.validation_fraction + self.test_fraction;
        if !phase_total.is_finite() || phase_total > 1.0 + f64::EPSILON {
            bail!("train_fraction + validation_fraction + test_fraction must be at most 1.0");
        }
        // The absolute duration check is performed once the base view span is
        // known.  Keep this check here for obvious overflow and typo errors.
        if self.purge_seconds > i64::MAX as u64 {
            bail!("purge_seconds is too large");
        }
        Ok(())
    }
}

fn validate_fraction(value: f64, name: &str, allow_zero: bool) -> Result<()> {
    if !value.is_finite() || value < 0.0 || (!allow_zero && value <= 0.0) || value > 1.0 {
        let requirement = if allow_zero {
            "between 0 and 1"
        } else {
            "greater than 0 and at most 1"
        };
        bail!("{name} must be finite and {requirement}");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardPlan {
    pub(crate) schema_version: u32,
    pub(crate) source_spec_path: PathBuf,
    pub(crate) source_sweep_id: String,
    pub(crate) base_spec: ReplaySweepSpec,
    pub(crate) config: ReplayWalkForwardConfig,
    pub(crate) output_dir: PathBuf,
    pub(crate) generated_at_utc: DateTime<Utc>,
    pub(crate) folds: Vec<ReplayWalkForwardFold>,
    #[serde(default)]
    pub(crate) warnings: Vec<String>,
}

impl ReplayWalkForwardPlan {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != REPLAY_WALK_FORWARD_PLAN_SCHEMA_VERSION {
            bail!(
                "unsupported replay walk-forward plan version {}; expected {}",
                self.schema_version,
                REPLAY_WALK_FORWARD_PLAN_SCHEMA_VERSION
            );
        }
        self.config.validate()?;
        self.base_spec.validate()?;
        if self.folds.is_empty() {
            bail!("walk-forward plan must contain at least one fold");
        }
        if self.source_sweep_id != self.base_spec.sweep_id {
            bail!("walk-forward source sweep id does not match the embedded base spec");
        }
        if self.output_dir.as_os_str().is_empty() {
            bail!("walk-forward output directory cannot be empty");
        }
        if self
            .config
            .warmup_seconds
            .unwrap_or(self.base_spec.base_dataset_view.warmup.duration_seconds)
            > self.base_spec.base_dataset_view.warmup.duration_seconds
        {
            bail!("walk-forward warmup exceeds the base dataset view warmup");
        }
        if self.folds.len() > self.config.folds || self.folds.len() > MAX_WALK_FORWARD_FOLDS {
            bail!("walk-forward plan contains too many folds");
        }
        let mut phases = 0;
        let base_start = self.base_spec.base_dataset_view.evaluation_start;
        let base_end = self.base_spec.base_dataset_view.evaluation_end;
        for (expected_index, fold) in self.folds.iter().enumerate() {
            if fold.fold_index != expected_index {
                bail!("walk-forward fold indexes must be contiguous starting at zero");
            }
            phases += 2 + usize::from(fold.validation.is_some());
            validate_window(
                &fold.train,
                base_start,
                base_end,
                ReplayWalkForwardPhase::Train,
            )?;
            if let Some(validation) = fold.validation.as_ref() {
                validate_window(
                    validation,
                    base_start,
                    base_end,
                    ReplayWalkForwardPhase::Validation,
                )?;
            }
            validate_window(
                &fold.test,
                base_start,
                base_end,
                ReplayWalkForwardPhase::Test,
            )?;
            if let Some(validation) = fold.validation.as_ref() {
                if fold.train.evaluation_end > validation.evaluation_start {
                    bail!("walk-forward train and validation windows overlap");
                }
                if validation.evaluation_end > fold.test.evaluation_start {
                    bail!("walk-forward validation and test windows overlap");
                }
            } else if fold.train.evaluation_end > fold.test.evaluation_start {
                bail!("walk-forward train and test windows overlap");
            }
        }
        if phases > MAX_WALK_FORWARD_PHASES {
            bail!("walk-forward plan contains too many generated phases");
        }
        Ok(())
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        write_json_atomic(path, self)
    }

    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("read replay walk-forward plan {}", path.display()))?;
        let plan = serde_json::from_slice::<Self>(&bytes)
            .with_context(|| format!("parse replay walk-forward plan {}", path.display()))?;
        plan.validate()
            .with_context(|| format!("validate replay walk-forward plan {}", path.display()))?;
        Ok(plan)
    }
}

fn validate_window(
    window: &ReplayWalkForwardWindow,
    base_start: DateTime<Utc>,
    base_end: DateTime<Utc>,
    expected_phase: ReplayWalkForwardPhase,
) -> Result<()> {
    if window.phase != expected_phase {
        bail!("walk-forward window phase metadata does not match its fold field");
    }
    if window.evaluation_start >= window.evaluation_end {
        bail!(
            "walk-forward {} window must have a positive evaluation range",
            window.phase.label()
        );
    }
    if window.warmup_start > window.evaluation_start {
        bail!("walk-forward warmup starts after evaluation");
    }
    if window.evaluation_start < base_start || window.evaluation_end > base_end {
        bail!("walk-forward window lies outside the base evaluation range");
    }
    if window.sweep_id.trim().is_empty()
        || window.view_id.trim().is_empty()
        || window.spec_path.as_os_str().is_empty()
    {
        bail!("walk-forward window metadata cannot be empty");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayWalkForwardOptions {
    pub(crate) train_fraction: f64,
    pub(crate) validation_fraction: f64,
    pub(crate) test_fraction: f64,
    pub(crate) folds: usize,
    pub(crate) step_fraction: Option<f64>,
    pub(crate) purge_seconds: u64,
    pub(crate) warmup_seconds: Option<u64>,
    pub(crate) output_dir: Option<PathBuf>,
}

impl Default for ReplayWalkForwardOptions {
    fn default() -> Self {
        let defaults = ReplayWalkForwardConfig::default();
        Self {
            train_fraction: defaults.train_fraction,
            validation_fraction: defaults.validation_fraction,
            test_fraction: defaults.test_fraction,
            folds: defaults.folds,
            step_fraction: defaults.step_fraction,
            purge_seconds: defaults.purge_seconds,
            warmup_seconds: defaults.warmup_seconds,
            output_dir: None,
        }
    }
}

impl From<ReplayWalkForwardOptions> for ReplayWalkForwardConfig {
    fn from(value: ReplayWalkForwardOptions) -> Self {
        Self {
            train_fraction: value.train_fraction,
            validation_fraction: value.validation_fraction,
            test_fraction: value.test_fraction,
            folds: value.folds,
            step_fraction: value.step_fraction,
            purge_seconds: value.purge_seconds,
            warmup_seconds: value.warmup_seconds,
        }
    }
}

/// Build and persist a walk-forward plan plus one normal replay-sweep spec per
/// fold phase.  The generated specs can be run independently with
/// `run-replay-sweep`; no market data is opened here.
pub(crate) fn plan_replay_walk_forward(
    source_spec_path: &Path,
    options: ReplayWalkForwardOptions,
    plan_path: &Path,
) -> Result<ReplayWalkForwardPlan> {
    let base_spec = ReplaySweepSpec::load(source_spec_path)?;
    let config = ReplayWalkForwardConfig::from(options.clone());
    config.validate()?;
    let output_dir = options
        .output_dir
        .unwrap_or_else(|| base_spec.output_dir.join("walk-forward"));
    if output_dir.as_os_str().is_empty() {
        bail!("walk-forward output directory cannot be empty");
    }

    let base_start = base_spec.base_dataset_view.evaluation_start;
    let base_end = base_spec.base_dataset_view.evaluation_end;
    let total_seconds = (base_end - base_start).num_seconds();
    if total_seconds <= 0 {
        bail!("base sweep dataset view must span at least one second");
    }
    let train_seconds = fraction_seconds(total_seconds, config.train_fraction, "train")?;
    let validation_seconds = if config.validation_fraction > 0.0 {
        fraction_seconds(total_seconds, config.validation_fraction, "validation")?
    } else {
        0
    };
    let test_seconds = fraction_seconds(total_seconds, config.test_fraction, "test")?;
    let purge_seconds = i64::try_from(config.purge_seconds)
        .context("purge_seconds is too large for the timestamp range")?;
    let boundary_count = usize::from(validation_seconds > 0) + 1;
    let required_seconds = train_seconds
        .checked_add(validation_seconds)
        .and_then(|value| value.checked_add(test_seconds))
        .and_then(|value| value.checked_add(purge_seconds.checked_mul(boundary_count as i64)?))
        .context("walk-forward split duration overflow")?;
    if required_seconds > total_seconds {
        bail!(
            "walk-forward windows require {required_seconds}s but the base evaluation range has {total_seconds}s; reduce purge or phase fractions"
        );
    }
    let step_fraction = config.step_fraction.unwrap_or(config.test_fraction);
    let step_seconds = fraction_seconds(total_seconds, step_fraction, "step")?;

    let warmup_seconds = config
        .warmup_seconds
        .unwrap_or(base_spec.base_dataset_view.warmup.duration_seconds);
    if warmup_seconds > base_spec.base_dataset_view.warmup.duration_seconds {
        bail!(
            "warmup_seconds ({warmup_seconds}) cannot exceed the base dataset view warmup ({})",
            base_spec.base_dataset_view.warmup.duration_seconds
        );
    }
    let warmup = base_spec.base_dataset_view.warmup;
    let mut folds = Vec::new();
    let mut warnings = Vec::new();
    for fold_index in 0..config.folds {
        let offset_seconds = step_seconds
            .checked_mul(fold_index as i64)
            .context("walk-forward fold offset overflow")?;
        let fold_start = base_start
            .checked_add_signed(Duration::seconds(offset_seconds))
            .context("walk-forward fold starts outside timestamp range")?;
        let train_end = fold_start
            .checked_add_signed(Duration::seconds(train_seconds))
            .context("walk-forward train window overflow")?;
        let (validation, test_start) = if validation_seconds > 0 {
            let validation_start = train_end
                .checked_add_signed(Duration::seconds(purge_seconds))
                .context("walk-forward validation start overflow")?;
            let validation_end = validation_start
                .checked_add_signed(Duration::seconds(validation_seconds))
                .context("walk-forward validation end overflow")?;
            let test_start = validation_end
                .checked_add_signed(Duration::seconds(purge_seconds))
                .context("walk-forward test start overflow")?;
            (Some((validation_start, validation_end)), test_start)
        } else {
            let test_start = train_end
                .checked_add_signed(Duration::seconds(purge_seconds))
                .context("walk-forward test start overflow")?;
            (None, test_start)
        };
        let test_end = test_start
            .checked_add_signed(Duration::seconds(test_seconds))
            .context("walk-forward test window overflow")?;
        if test_end > base_end {
            break;
        }
        let train = build_window(
            &base_spec,
            &output_dir,
            fold_index,
            ReplayWalkForwardPhase::Train,
            fold_start,
            train_end,
            warmup_seconds,
            warmup,
        )?;
        let validation = validation
            .map(|(start, end)| {
                build_window(
                    &base_spec,
                    &output_dir,
                    fold_index,
                    ReplayWalkForwardPhase::Validation,
                    start,
                    end,
                    warmup_seconds,
                    warmup,
                )
            })
            .transpose()?;
        let test = build_window(
            &base_spec,
            &output_dir,
            fold_index,
            ReplayWalkForwardPhase::Test,
            test_start,
            test_end,
            warmup_seconds,
            warmup,
        )?;
        folds.push(ReplayWalkForwardFold {
            fold_index,
            train,
            validation,
            test,
        });
    }
    if folds.is_empty() {
        bail!("walk-forward configuration does not fit one complete fold in the base range");
    }
    if folds.len() < config.folds {
        warnings.push(format!(
            "requested {} folds but only {} complete fold(s) fit in the base evaluation range",
            config.folds,
            folds.len()
        ));
    }

    let plan = ReplayWalkForwardPlan {
        schema_version: REPLAY_WALK_FORWARD_PLAN_SCHEMA_VERSION,
        source_spec_path: source_spec_path.to_path_buf(),
        source_sweep_id: base_spec.sweep_id.clone(),
        base_spec,
        config,
        output_dir,
        generated_at_utc: Utc::now(),
        folds,
        warnings,
    };
    plan.save(plan_path)?;
    Ok(plan)
}

fn fraction_seconds(total_seconds: i64, fraction: f64, label: &str) -> Result<i64> {
    let value = (total_seconds as f64 * fraction).floor();
    if !value.is_finite() || value < 1.0 || value > i64::MAX as f64 {
        bail!("{label} fraction produces a window shorter than one second");
    }
    Ok(value as i64)
}

fn build_window(
    base_spec: &ReplaySweepSpec,
    output_dir: &Path,
    fold_index: usize,
    phase: ReplayWalkForwardPhase,
    evaluation_start: DateTime<Utc>,
    evaluation_end: DateTime<Utc>,
    warmup_seconds: u64,
    mut warmup: crate::replay_cache::ReplayDatasetWarmupPolicy,
) -> Result<ReplayWalkForwardWindow> {
    if evaluation_start >= evaluation_end {
        bail!("walk-forward {} window is empty", phase.label());
    }
    let key = format!(
        "{}:{}:{}:{}:{}:{}",
        base_spec.sweep_id,
        fold_index,
        phase.label(),
        evaluation_start.timestamp_nanos_opt().unwrap_or_default(),
        evaluation_end.timestamp_nanos_opt().unwrap_or_default(),
        warmup_seconds,
    );
    let hash = stable_hash(&key);
    let sweep_id = format!(
        "{}-wf{:016x}-{:03}-{}",
        short_id(&base_spec.sweep_id),
        hash,
        fold_index + 1,
        phase.label()
    );
    let view_id = format!("wf-{:016x}-{:03}-{}", hash, fold_index + 1, phase.label());
    let phase_dir = output_dir
        .join(format!("fold-{:03}", fold_index + 1))
        .join(phase.label());
    let spec_path = phase_dir.join("sweep.json");
    let warmup_start = evaluation_start
        .checked_sub_signed(Duration::seconds(
            i64::try_from(warmup_seconds).context("warmup_seconds is too large")?,
        ))
        .context("walk-forward warmup starts outside timestamp range")?;

    warmup.duration_seconds = warmup_seconds;
    let mut view: ReplayDatasetView = base_spec.base_dataset_view.clone();
    view.id = view_id.clone();
    view.evaluation_start = evaluation_start;
    view.evaluation_end = evaluation_end;
    view.input_timezone = "UTC".to_string();
    view.session_preset = ReplayDatasetSessionPreset::CustomUtc;
    view.warmup = warmup;
    view.validate_model()?;

    let mut phase_spec = base_spec.clone();
    phase_spec.sweep_id = sweep_id.clone();
    phase_spec.name = format!("{} ({})", base_spec.name, phase.label());
    phase_spec.notes = format!(
        "Walk-forward fold {} {} phase derived from sweep {}. Evaluation range [{} , {}); warmup starts at {}.",
        fold_index + 1,
        phase.label(),
        base_spec.sweep_id,
        evaluation_start,
        evaluation_end,
        warmup_start
    );
    phase_spec.base_dataset_view = view;
    phase_spec.output_dir = phase_dir;
    phase_spec.save(&spec_path)?;

    Ok(ReplayWalkForwardWindow {
        phase,
        evaluation_start,
        evaluation_end,
        warmup_start,
        sweep_id,
        view_id,
        spec_path,
    })
}

fn short_id(value: &str) -> String {
    let mut result = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .take(32)
        .collect::<String>();
    if result.is_empty() {
        result.push_str("sweep");
    }
    result
}

fn stable_hash(value: &str) -> u64 {
    // Use an explicit hash instead of `DefaultHasher`: generated IDs are
    // persisted in plans and should remain reproducible across processes and
    // Rust implementation changes.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }
    hash
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create walk-forward output {}", parent.display()))?;
    let bytes = serde_json::to_vec_pretty(value).context("serialize replay walk-forward JSON")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("plan.json");
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    fs::write(&temporary, bytes).with_context(|| {
        format!(
            "write temporary walk-forward output {}",
            temporary.display()
        )
    })?;
    fs::rename(&temporary, path)
        .with_context(|| format!("replace walk-forward output {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{
        BarType, BrokerKind, CandleMode, ReplayEngineMode, ReplayFillModel, ReplayLatencyConfig,
    };
    use crate::config::TradingEnvironment;
    use crate::replay_cache::{
        ReplayDatasetSessionPreset, ReplayDatasetSourceRef, ReplayDatasetWarmupPolicy,
    };
    use crate::strategy::ExecutionStrategyConfig;

    fn sample_spec() -> ReplaySweepSpec {
        let start = DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .expect("start")
            .with_timezone(&Utc);
        let end = start + Duration::hours(10);
        ReplaySweepSpec {
            schema_version: super::super::sweep::REPLAY_SWEEP_SPEC_SCHEMA_VERSION,
            sweep_id: "mes-ema".to_string(),
            name: "MES EMA".to_string(),
            notes: String::new(),
            base_dataset_view: ReplayDatasetView {
                view_version: crate::replay_cache::REPLAY_DATASET_VIEW_VERSION,
                id: "base-view".to_string(),
                source: ReplayDatasetSourceRef {
                    manifest_id: "tradovate/sim/MES/MESU6/2026-01-01/manifest.json".to_string(),
                    provider: BrokerKind::Tradovate,
                    env: TradingEnvironment::Sim,
                    instrument: "MES".to_string(),
                    contract: "MESU6".to_string(),
                },
                evaluation_start: start,
                evaluation_end: end,
                input_timezone: "UTC".to_string(),
                session_preset: ReplayDatasetSessionPreset::FullSource,
                daily_session: None,
                warmup: ReplayDatasetWarmupPolicy {
                    duration_seconds: 60,
                    ..ReplayDatasetWarmupPolicy::default()
                },
            },
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            base_strategy: ExecutionStrategyConfig::default(),
            parameters: Vec::new(),
            constraints: Vec::new(),
            engine_mode: ReplayEngineMode::Deterministic,
            evaluator_mode: crate::broker::ReplayEvaluatorMode::Legacy,
            fill_model: ReplayFillModel::RawBarOpen,
            latency: ReplayLatencyConfig::default(),
            bar_protection_policy: crate::broker::ReplayBarProtectionPolicy::Conservative,
            primary_fee_schedule: crate::tradovate::ReplayFeeSchedule::default(),
            fee_scenarios: Vec::new(),
            initial_capital: 50_000.0,
            margin: None,
            max_runs: 1,
            parallelism: 1,
            execution_mode: Default::default(),
            guardrails: Default::default(),
            replay_markov_orientation_gate: Default::default(),
            output_dir: PathBuf::from("runs/mes-ema"),
            output_formats: vec![super::super::sweep::ReplaySweepOutputFormat::JsonSummary],
        }
    }

    #[test]
    fn planner_persists_one_fold_and_three_phase_specs() {
        let root = std::env::temp_dir().join(format!("trader-walk-forward-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let source = root.join("source.json");
        let output = root.join("walk-forward.json");
        let mut spec = sample_spec();
        spec.output_dir = root.join("runs");
        spec.save(&source).expect("save source");
        let plan = plan_replay_walk_forward(
            &source,
            ReplayWalkForwardOptions {
                output_dir: Some(root.join("phases")),
                ..ReplayWalkForwardOptions::default()
            },
            &output,
        )
        .expect("plan");
        assert_eq!(plan.folds.len(), 1);
        assert!(plan.folds[0].validation.is_some());
        assert!(plan.folds[0].train.spec_path.is_file());
        assert!(plan.folds[0].test.spec_path.is_file());
        let loaded = ReplayWalkForwardPlan::load(&output).expect("load plan");
        assert_eq!(loaded, plan);
        let validation = plan.folds[0].validation.as_ref().expect("validation");
        assert!(plan.folds[0].train.evaluation_end <= validation.evaluation_start);
        assert!(validation.evaluation_end <= plan.folds[0].test.evaluation_start);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn planner_applies_purge_and_bounds_fold_count() {
        let root =
            std::env::temp_dir().join(format!("trader-walk-forward-purge-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let source = root.join("source.json");
        let output = root.join("walk-forward.json");
        let mut spec = sample_spec();
        spec.output_dir = root.join("runs");
        spec.save(&source).expect("save source");
        let mut options = ReplayWalkForwardOptions::default();
        options.train_fraction = 0.4;
        options.validation_fraction = 0.1;
        options.test_fraction = 0.2;
        options.folds = 3;
        options.step_fraction = Some(0.2);
        options.purge_seconds = 60;
        options.output_dir = Some(root.join("phases"));
        let plan = plan_replay_walk_forward(&source, options, &output).expect("plan");
        assert_eq!(plan.folds.len(), 2);
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("only 2"))
        );
        for fold in &plan.folds {
            let validation = fold.validation.as_ref().expect("validation");
            assert_eq!(
                validation.evaluation_start - fold.train.evaluation_end,
                Duration::minutes(1)
            );
            assert_eq!(
                fold.test.evaluation_start - validation.evaluation_end,
                Duration::minutes(1)
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn planner_rejects_invalid_fraction_and_warmup() {
        let root = std::env::temp_dir().join(format!(
            "trader-walk-forward-invalid-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let source = root.join("source.json");
        let output = root.join("walk-forward.json");
        let spec = sample_spec();
        spec.save(&source).expect("save source");
        let mut options = ReplayWalkForwardOptions::default();
        options.train_fraction = 0.9;
        options.test_fraction = 0.9;
        assert!(plan_replay_walk_forward(&source, options, &output).is_err());
        let mut options = ReplayWalkForwardOptions::default();
        options.warmup_seconds = Some(61);
        assert!(plan_replay_walk_forward(&source, options, &output).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
