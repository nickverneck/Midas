//! Replay parameter-sweep specifications and deterministic child expansion.
//!
//! A sweep is a persisted description of one dataset, one accounting model,
//! and a Cartesian grid of strategy overrides. The headless runner can
//! consume [`ReplaySweepPlan`] without duplicating validation or
//! parameter-resolution rules; this module also owns the pre-launch resource
//! estimate and guardrail policy.

use super::{ReplayFeeSchedule, ReplayMarginConfig};
use crate::broker::{
    BarKind, BarType, CandleMode, ReplayBarProtectionPolicy, ReplayEngineMode, ReplayFillModel,
    ReplayLatencyConfig, ReplayLatencyModel,
};
use crate::replay_cache::ReplayDatasetView;
use crate::replay_cache::{
    ReplayCacheCoverage, ReplayCacheDataFile, ReplayCacheTimeRange, ReplayDatasetViewStore,
};
use crate::strategies::ema_cross::EmaCrossConfig;
use crate::strategies::hma_angle::HmaAngleConfig;
use crate::strategies::hma_cross::HmaCrossConfig;
use crate::strategy::ExecutionStrategyConfig;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub(crate) const REPLAY_SWEEP_SPEC_SCHEMA_VERSION: u32 = 1;
pub(crate) const REPLAY_SWEEP_PLAN_SCHEMA_VERSION: u32 = 1;

/// Selects how sweep candidates are scheduled.
///
/// `isolated_services` is the historical runner: every candidate owns an
/// isolated replay service and therefore has independent execution state.
/// `batch_cpu` keeps that same isolation/determinism contract while sharing
/// immutable dataset preparation and scheduling candidates through the CPU
/// worker pool.  It deliberately does not split a single candidate into
/// chronological windows; fills, protection, and account state remain ordered
/// within each candidate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplaySweepExecutionMode {
    IsolatedServices,
    BatchCpu,
}

impl Default for ReplaySweepExecutionMode {
    fn default() -> Self {
        Self::IsolatedServices
    }
}

impl ReplaySweepExecutionMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::IsolatedServices => "isolated_services",
            Self::BatchCpu => "batch_cpu",
        }
    }

    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "isolated_services" | "isolated" | "legacy" | "service" | "services" => {
                Ok(Self::IsolatedServices)
            }
            "batch_cpu" | "batch-cpu" | "batch" | "cpu" => Ok(Self::BatchCpu),
            _ => bail!(
                "unknown replay sweep execution mode `{raw}`; choose isolated_services or batch_cpu"
            ),
        }
    }
}

/// A deliberately conservative cap for the materialized validation plan.
/// Resource guardrails are applied before a runner materializes this many
/// children, but this remains a final safety net for validation and plans
/// loaded by older callers.
const MAX_MATERIALIZED_SWEEP_RUNS: usize = 1_000_000;

const DEFAULT_MAX_SWEEP_COMBINATIONS: usize = 10_000;
const DEFAULT_MAX_SWEEP_PARALLEL_JOBS: usize = 8;
const DEFAULT_MAX_CACHE_ROWS_PER_WORKER: u64 = 20_000_000;
const DEFAULT_LARGE_SWEEP_THRESHOLD: usize = 100;
const DEFAULT_MAX_ESTIMATED_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_ESTIMATED_OUTPUT_BYTES: u64 = 64 * 1024 * 1024 * 1024;

const SERVER_BAR_BYTES_PER_ROW: u64 = 128;
const RAW_TICK_BYTES_PER_ROW: u64 = 192;
const UNKNOWN_INPUT_BYTES_PER_ROW: u64 = 192;
const WORKER_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;
const OUTPUT_BASE_BYTES_PER_RUN: u64 = 256 * 1024;
const OUTPUT_BYTES_PER_INPUT_ROW: u64 = 96;
const SERVER_BAR_ROWS_PER_SECOND: u64 = 50_000;
const RAW_TICK_ROWS_PER_SECOND: u64 = 100_000;
const UNKNOWN_ROWS_PER_SECOND: u64 = 25_000;

/// Launch limits that protect the host from an accidentally enormous sweep.
///
/// The memory and output limits are optional so a deployment can disable one
/// of those checks while retaining the combination/parallelism/cache limits.
/// They are deliberately part of the persisted spec: a rerun therefore has
/// the same safety policy as the original launch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplaySweepGuardrails {
    pub(crate) max_combinations: usize,
    pub(crate) max_parallel_jobs: usize,
    pub(crate) max_cache_read_rows_per_worker: u64,
    pub(crate) large_sweep_threshold: usize,
    pub(crate) max_estimated_memory_bytes: Option<u64>,
    pub(crate) max_estimated_output_bytes: Option<u64>,
}

impl Default for ReplaySweepGuardrails {
    fn default() -> Self {
        Self {
            max_combinations: DEFAULT_MAX_SWEEP_COMBINATIONS,
            max_parallel_jobs: DEFAULT_MAX_SWEEP_PARALLEL_JOBS,
            max_cache_read_rows_per_worker: DEFAULT_MAX_CACHE_ROWS_PER_WORKER,
            large_sweep_threshold: DEFAULT_LARGE_SWEEP_THRESHOLD,
            max_estimated_memory_bytes: Some(DEFAULT_MAX_ESTIMATED_MEMORY_BYTES),
            max_estimated_output_bytes: Some(DEFAULT_MAX_ESTIMATED_OUTPUT_BYTES),
        }
    }
}

impl ReplaySweepGuardrails {
    fn validate(&self) -> Result<()> {
        if self.max_combinations == 0 {
            bail!("guardrails.max_combinations must be greater than zero");
        }
        if self.max_parallel_jobs == 0 {
            bail!("guardrails.max_parallel_jobs must be greater than zero");
        }
        if self.max_cache_read_rows_per_worker == 0 {
            bail!("guardrails.max_cache_read_rows_per_worker must be greater than zero");
        }
        if self.large_sweep_threshold == 0 {
            bail!("guardrails.large_sweep_threshold must be greater than zero");
        }
        if self.max_estimated_memory_bytes == Some(0) {
            bail!("guardrails.max_estimated_memory_bytes must be greater than zero when set");
        }
        if self.max_estimated_output_bytes == Some(0) {
            bail!("guardrails.max_estimated_output_bytes must be greater than zero when set");
        }
        Ok(())
    }
}

/// Conservative, metadata-driven launch estimate. `estimated_input_rows` is
/// `None` for non-time-based bars when no matching cache manifest is present.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepResourceEstimate {
    pub(crate) combinations: usize,
    pub(crate) parallel_jobs: usize,
    pub(crate) estimated_input_rows: Option<u64>,
    pub(crate) estimated_input_rows_per_worker: Option<u64>,
    pub(crate) input_source: String,
    pub(crate) estimated_memory_bytes: Option<u64>,
    pub(crate) estimated_output_bytes: Option<u64>,
    pub(crate) estimated_runtime_seconds: Option<u64>,
    #[serde(default)]
    pub(crate) notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepGuardrailReport {
    pub(crate) estimate: ReplaySweepResourceEstimate,
    pub(crate) violations: Vec<String>,
    pub(crate) warnings: Vec<String>,
    pub(crate) requires_confirmation: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplaySweepOutputFormat {
    JsonSummary,
    CsvRows,
    ParquetRows,
}

impl ReplaySweepOutputFormat {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::JsonSummary => "json_summary",
            Self::CsvRows => "csv_rows",
            Self::ParquetRows => "parquet_rows",
        }
    }
}

fn default_output_formats() -> Vec<ReplaySweepOutputFormat> {
    vec![ReplaySweepOutputFormat::JsonSummary]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepParameter {
    /// A dot-separated scalar path in [`ExecutionStrategyConfig`].
    ///
    /// Keeping this as a named path rather than an opaque serialized object
    /// makes every child override auditable and gives validation a finite
    /// allow-list of fields that the runner can safely resolve.
    pub(crate) path: String,
    pub(crate) values: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ReplaySweepConstraint {
    Less { left: String, right: String },
    LessOrEqual { left: String, right: String },
    Greater { left: String, right: String },
    GreaterOrEqual { left: String, right: String },
    Equal { left: String, right: String },
    NotEqual { left: String, right: String },
}

impl ReplaySweepConstraint {
    fn paths(&self) -> (&str, &str) {
        match self {
            Self::Less { left, right }
            | Self::LessOrEqual { left, right }
            | Self::Greater { left, right }
            | Self::GreaterOrEqual { left, right }
            | Self::Equal { left, right }
            | Self::NotEqual { left, right } => (left, right),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Less { .. } => "<",
            Self::LessOrEqual { .. } => "<=",
            Self::Greater { .. } => ">",
            Self::GreaterOrEqual { .. } => ">=",
            Self::Equal { .. } => "==",
            Self::NotEqual { .. } => "!=",
        }
    }

    fn evaluate(&self, strategy: &ExecutionStrategyConfig) -> Result<bool> {
        let encoded = serde_json::to_value(strategy).context("serialize child strategy")?;
        let (left_path, right_path) = self.paths();
        let left = value_at_path(&encoded, left_path)
            .with_context(|| format!("constraint path {left_path}"))?;
        let right = value_at_path(&encoded, right_path)
            .with_context(|| format!("constraint path {right_path}"))?;

        match self {
            Self::Equal { .. } => Ok(left == right),
            Self::NotEqual { .. } => Ok(left != right),
            Self::Less { .. }
            | Self::LessOrEqual { .. }
            | Self::Greater { .. }
            | Self::GreaterOrEqual { .. } => {
                let left = numeric_value(left, left_path)?;
                let right = numeric_value(right, right_path)?;
                Ok(match self {
                    Self::Less { .. } => left < right,
                    Self::LessOrEqual { .. } => left <= right,
                    Self::Greater { .. } => left > right,
                    Self::GreaterOrEqual { .. } => left >= right,
                    Self::Equal { .. } | Self::NotEqual { .. } => unreachable!(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplaySweepSpec {
    pub(crate) schema_version: u32,
    pub(crate) sweep_id: String,
    pub(crate) name: String,
    pub(crate) notes: String,
    pub(crate) base_dataset_view: ReplayDatasetView,
    #[serde(default)]
    pub(crate) bar_type: BarType,
    #[serde(default)]
    pub(crate) candle_mode: CandleMode,
    pub(crate) base_strategy: ExecutionStrategyConfig,
    pub(crate) parameters: Vec<ReplaySweepParameter>,
    pub(crate) constraints: Vec<ReplaySweepConstraint>,
    pub(crate) engine_mode: ReplayEngineMode,
    /// Indicator implementation used by each child. Batch CPU mode defaults
    /// to streaming at launch time, while this persisted value keeps legacy
    /// compatibility explicit for isolated runs and parity baselines.
    #[serde(default)]
    pub(crate) evaluator_mode: crate::broker::ReplayEvaluatorMode,
    pub(crate) fill_model: ReplayFillModel,
    pub(crate) latency: ReplayLatencyConfig,
    pub(crate) bar_protection_policy: ReplayBarProtectionPolicy,
    pub(crate) primary_fee_schedule: ReplayFeeSchedule,
    pub(crate) fee_scenarios: Vec<ReplayFeeSchedule>,
    pub(crate) initial_capital: f64,
    pub(crate) margin: Option<ReplayMarginConfig>,
    pub(crate) max_runs: usize,
    pub(crate) parallelism: usize,
    /// Candidate scheduling mode. This does not change the ordered execution
    /// semantics of an individual candidate.
    #[serde(default)]
    pub(crate) execution_mode: ReplaySweepExecutionMode,
    #[serde(default)]
    pub(crate) guardrails: ReplaySweepGuardrails,
    pub(crate) output_dir: std::path::PathBuf,
    pub(crate) output_formats: Vec<ReplaySweepOutputFormat>,
}

impl Default for ReplaySweepSpec {
    fn default() -> Self {
        Self {
            schema_version: REPLAY_SWEEP_SPEC_SCHEMA_VERSION,
            sweep_id: String::new(),
            name: String::new(),
            notes: String::new(),
            base_dataset_view: ReplayDatasetView {
                view_version: 0,
                id: String::new(),
                source: crate::replay_cache::ReplayDatasetSourceRef {
                    manifest_id: String::new(),
                    provider: crate::broker::BrokerKind::Tradovate,
                    env: crate::config::TradingEnvironment::Sim,
                    instrument: String::new(),
                    contract: String::new(),
                },
                evaluation_start: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                evaluation_end: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                input_timezone: "UTC".to_string(),
                session_preset: crate::replay_cache::ReplayDatasetSessionPreset::FullSource,
                warmup: crate::replay_cache::ReplayDatasetWarmupPolicy::default(),
            },
            bar_type: BarType::default(),
            candle_mode: CandleMode::default(),
            base_strategy: ExecutionStrategyConfig::default(),
            parameters: Vec::new(),
            constraints: Vec::new(),
            engine_mode: ReplayEngineMode::Deterministic,
            evaluator_mode: crate::broker::ReplayEvaluatorMode::default(),
            fill_model: ReplayFillModel::RawBarOpen,
            latency: ReplayLatencyConfig::default(),
            bar_protection_policy: ReplayBarProtectionPolicy::default(),
            primary_fee_schedule: ReplayFeeSchedule::default(),
            fee_scenarios: Vec::new(),
            initial_capital: 100_000.0,
            margin: None,
            max_runs: 1,
            parallelism: 1,
            execution_mode: ReplaySweepExecutionMode::default(),
            guardrails: ReplaySweepGuardrails::default(),
            output_dir: std::path::PathBuf::from("runs"),
            output_formats: default_output_formats(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepChildSpec {
    pub(crate) run_id: String,
    pub(crate) run_index: usize,
    pub(crate) parent_sweep_id: String,
    pub(crate) resolved_strategy: ExecutionStrategyConfig,
    /// All values selected by the grid, including values equal to the base.
    pub(crate) parameter_values: BTreeMap<String, Value>,
    /// Only values that differ from the base strategy configuration.
    pub(crate) overrides: BTreeMap<String, Value>,
    pub(crate) base_dataset_view: ReplayDatasetView,
    pub(crate) bar_type: BarType,
    pub(crate) candle_mode: CandleMode,
    pub(crate) engine_mode: ReplayEngineMode,
    #[serde(default)]
    pub(crate) evaluator_mode: crate::broker::ReplayEvaluatorMode,
    pub(crate) fill_model: ReplayFillModel,
    pub(crate) latency: ReplayLatencyConfig,
    pub(crate) bar_protection_policy: ReplayBarProtectionPolicy,
    pub(crate) primary_fee_schedule: ReplayFeeSchedule,
    pub(crate) fee_scenarios: Vec<ReplayFeeSchedule>,
    pub(crate) initial_capital: f64,
    pub(crate) margin: Option<ReplayMarginConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPlan {
    pub(crate) schema_version: u32,
    pub(crate) spec: ReplaySweepSpec,
    pub(crate) children: Vec<ReplaySweepChildSpec>,
}

impl ReplaySweepSpec {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != REPLAY_SWEEP_SPEC_SCHEMA_VERSION {
            bail!(
                "unsupported replay sweep spec version {}; expected {}",
                self.schema_version,
                REPLAY_SWEEP_SPEC_SCHEMA_VERSION
            );
        }
        validate_sweep_id(&self.sweep_id)?;
        if self.name.trim().is_empty() {
            bail!("replay sweep name cannot be empty");
        }
        self.base_dataset_view
            .validate_model()
            .context("validate base dataset view")?;
        if self.base_dataset_view.source.provider != crate::broker::BrokerKind::Tradovate {
            bail!("replay sweeps currently support Tradovate dataset views only");
        }
        validate_strategy_config(&self.base_strategy)?;
        validate_replay_models(self.engine_mode, self.fill_model, &self.latency)?;
        self.primary_fee_schedule.validate()?;
        validate_fee_scenarios(&self.primary_fee_schedule, &self.fee_scenarios)?;
        if !self.initial_capital.is_finite() || self.initial_capital <= 0.0 {
            bail!("initial_capital must be finite and greater than zero");
        }
        if let Some(margin) = &self.margin {
            margin.validate()?;
        }
        if self.max_runs == 0 {
            bail!("max_runs must be greater than zero");
        }
        if self.parallelism == 0 {
            bail!("parallelism must be greater than zero");
        }
        self.guardrails.validate()?;
        if self.output_dir.as_os_str().is_empty() {
            bail!("output_dir cannot be empty");
        }
        validate_output_formats(&self.output_formats)?;

        let mut parameter_paths = BTreeSet::new();
        for parameter in &self.parameters {
            validate_parameter_path(&parameter.path)?;
            if !parameter_paths.insert(parameter.path.clone()) {
                bail!("duplicate sweep parameter path: {}", parameter.path);
            }
            if parameter.values.is_empty() {
                bail!("sweep parameter {} has no values", parameter.path);
            }
            let mut unique_values = Vec::new();
            for value in &parameter.values {
                if !is_scalar_value(value) {
                    bail!(
                        "sweep parameter {} values must be scalar JSON values",
                        parameter.path
                    );
                }
                if unique_values.iter().any(|existing| *existing == value) {
                    bail!(
                        "sweep parameter {} contains duplicate value {}",
                        parameter.path,
                        value
                    );
                }
                unique_values.push(value);
            }
        }
        for constraint in &self.constraints {
            let (left, right) = constraint.paths();
            validate_parameter_path(left)
                .with_context(|| format!("validate constraint left path {left}"))?;
            validate_parameter_path(right)
                .with_context(|| format!("validate constraint right path {right}"))?;
        }

        let combinations = self.combination_count()?;
        if combinations > self.max_runs {
            bail!(
                "sweep grid produces {combinations} runs, exceeding max_runs {}",
                self.max_runs
            );
        }
        if combinations > MAX_MATERIALIZED_SWEEP_RUNS {
            bail!(
                "sweep grid produces {combinations} runs; maximum materialized validation plan is {MAX_MATERIALIZED_SWEEP_RUNS}"
            );
        }
        Ok(())
    }

    /// Estimate the work a sweep will perform without opening the data file.
    /// When a cache root is supplied, manifest row counts are preferred. If a
    /// manifest is not available yet, time-based bars fall back to a duration
    /// estimate and tick/volume/range bars remain explicitly unknown.
    pub(crate) fn resource_estimate(
        &self,
        cache_root: Option<&Path>,
    ) -> Result<ReplaySweepResourceEstimate> {
        let combinations = self.combination_count()?;
        let parallel_jobs = self.parallelism.max(1).min(combinations.max(1));
        let load_range = self.base_dataset_view.load_range()?;
        let mut notes = Vec::new();

        let mut estimated_input_rows = None;
        let mut input_source = String::new();
        if let Some(cache_root) = cache_root {
            match estimate_cached_input_rows(
                cache_root,
                &self.base_dataset_view,
                self.bar_type,
                self.candle_mode,
                load_range,
            ) {
                Ok(Some((rows, source))) => {
                    estimated_input_rows = Some(rows);
                    input_source = source;
                }
                Ok(None) => {
                    notes.push(
                        "no matching cache file metadata was found; using a bar-duration estimate when possible"
                            .to_string(),
                    );
                }
                Err(error) => {
                    notes.push(format!(
                        "cached input row estimate unavailable ({error}); using a bar-duration estimate when possible"
                    ));
                }
            }
        }

        if estimated_input_rows.is_none() {
            if let Some(rows) = estimate_time_based_rows(self.bar_type, load_range)? {
                estimated_input_rows = Some(rows);
                input_source = if cache_root.is_some() {
                    "time-based bar duration fallback".to_string()
                } else {
                    "time-based bar duration estimate".to_string()
                };
            } else {
                input_source = if cache_root.is_some() {
                    "unknown until the replay cache manifest is resolved".to_string()
                } else {
                    "unknown without a replay cache manifest".to_string()
                };
                notes.push(
                    "input rows cannot be estimated for this bar type without matching cache metadata"
                        .to_string(),
                );
            }
        }

        let estimated_input_rows_per_worker = estimated_input_rows;
        let input_bytes_per_row = if input_source.contains("raw tick") {
            RAW_TICK_BYTES_PER_ROW
        } else if input_source.contains("bar") {
            SERVER_BAR_BYTES_PER_ROW
        } else {
            UNKNOWN_INPUT_BYTES_PER_ROW
        };
        let estimated_memory_bytes = estimated_input_rows.map(|rows| {
            rows.saturating_mul(input_bytes_per_row)
                .saturating_add(WORKER_OVERHEAD_BYTES)
                .saturating_mul(parallel_jobs as u64)
        });
        let estimated_output_bytes = estimated_input_rows.map(|rows| {
            let per_run = OUTPUT_BASE_BYTES_PER_RUN
                .saturating_add(rows.saturating_mul(OUTPUT_BYTES_PER_INPUT_ROW));
            per_run.saturating_mul(combinations as u64)
        });
        let estimated_runtime_seconds = estimated_input_rows.map(|rows| {
            let throughput = if input_source.contains("raw tick") {
                RAW_TICK_ROWS_PER_SECOND
            } else if input_source.contains("bar") {
                SERVER_BAR_ROWS_PER_SECOND
            } else {
                UNKNOWN_ROWS_PER_SECOND
            };
            let work = (rows as u128).saturating_mul(combinations as u128);
            let capacity = (throughput as u128).saturating_mul(parallel_jobs as u128);
            let seconds = (work.saturating_add(capacity.saturating_sub(1))) / capacity.max(1);
            seconds.min(u64::MAX as u128) as u64
        });

        Ok(ReplaySweepResourceEstimate {
            combinations,
            parallel_jobs,
            estimated_input_rows,
            estimated_input_rows_per_worker,
            input_source,
            estimated_memory_bytes,
            estimated_output_bytes,
            estimated_runtime_seconds,
            notes,
        })
    }

    pub(crate) fn guardrail_report(
        &self,
        cache_root: Option<&Path>,
    ) -> Result<ReplaySweepGuardrailReport> {
        self.guardrails.validate()?;
        let estimate = self.resource_estimate(cache_root)?;
        let mut violations = Vec::new();
        let mut warnings = estimate.notes.clone();

        if estimate.combinations > self.guardrails.max_combinations {
            violations.push(format!(
                "estimated {} combinations exceeds guardrails.max_combinations {}",
                estimate.combinations, self.guardrails.max_combinations
            ));
        }
        if self.parallelism > self.guardrails.max_parallel_jobs {
            violations.push(format!(
                "parallelism {} exceeds guardrails.max_parallel_jobs {}",
                self.parallelism, self.guardrails.max_parallel_jobs
            ));
        }
        if let Some(rows) = estimate.estimated_input_rows_per_worker {
            if rows > self.guardrails.max_cache_read_rows_per_worker {
                violations.push(format!(
                    "estimated {} cache rows per worker exceeds guardrails.max_cache_read_rows_per_worker {}",
                    rows, self.guardrails.max_cache_read_rows_per_worker
                ));
            }
        }
        if let (Some(limit), Some(memory)) = (
            self.guardrails.max_estimated_memory_bytes,
            estimate.estimated_memory_bytes,
        ) {
            if memory > limit {
                violations.push(format!(
                    "estimated memory {} exceeds guardrails.max_estimated_memory_bytes {}",
                    format_bytes(memory),
                    format_bytes(limit)
                ));
            }
        }
        if let (Some(limit), Some(output)) = (
            self.guardrails.max_estimated_output_bytes,
            estimate.estimated_output_bytes,
        ) {
            if output > limit {
                violations.push(format!(
                    "estimated output {} exceeds guardrails.max_estimated_output_bytes {}",
                    format_bytes(output),
                    format_bytes(limit)
                ));
            }
        }

        let requires_confirmation = estimate.combinations >= self.guardrails.large_sweep_threshold;
        if requires_confirmation {
            warnings.push(format!(
                "{} combinations meets the large-sweep confirmation threshold {}",
                estimate.combinations, self.guardrails.large_sweep_threshold
            ));
        }
        if estimate.estimated_input_rows.is_none() {
            warnings.push(
                "runtime, memory, and output estimates remain incomplete until a matching cache manifest is available"
                    .to_string(),
            );
        }

        Ok(ReplaySweepGuardrailReport {
            estimate,
            violations,
            warnings,
            requires_confirmation,
        })
    }

    pub(crate) fn combination_count(&self) -> Result<usize> {
        self.parameters
            .iter()
            .try_fold(1_usize, |count, parameter| {
                count
                    .checked_mul(parameter.values.len())
                    .context("sweep combination count overflow")
            })
    }

    pub(crate) fn expand(&self) -> Result<Vec<ReplaySweepChildSpec>> {
        self.validate()?;
        let combinations = cartesian_values(&self.parameters);
        let base_encoded = serde_json::to_value(&self.base_strategy)
            .context("serialize base strategy for sweep expansion")?;
        let mut children = Vec::with_capacity(combinations.len());

        for (run_index, values) in combinations.into_iter().enumerate() {
            let parameter_values = self
                .parameters
                .iter()
                .zip(values)
                .map(|(parameter, value)| (parameter.path.clone(), value))
                .collect::<BTreeMap<_, _>>();
            let resolved_strategy = resolve_strategy(&self.base_strategy, &parameter_values)
                .with_context(|| format!("resolve sweep child {}", run_index + 1))?;
            for constraint in &self.constraints {
                if !constraint.evaluate(&resolved_strategy)? {
                    let (left, right) = constraint.paths();
                    bail!(
                        "sweep child {} violates constraint: {left} {} {right}",
                        run_index + 1,
                        constraint.label()
                    );
                }
            }

            let resolved_encoded = serde_json::to_value(&resolved_strategy)
                .context("serialize resolved child strategy")?;
            let mut overrides = BTreeMap::new();
            for (path, value) in &parameter_values {
                let base_value = value_at_path(&base_encoded, path)
                    .with_context(|| format!("base strategy parameter path {path}"))?;
                if base_value != value {
                    overrides.insert(path.clone(), value.clone());
                }
                // The resolver already deserialized this field, but checking
                // the path here catches future allow-list/config drift before
                // a runner is allowed to start.
                let _ = value_at_path(&resolved_encoded, path)
                    .with_context(|| format!("resolved strategy parameter path {path}"))?;
            }

            children.push(ReplaySweepChildSpec {
                run_id: format!("{}-{:06}", self.sweep_id, run_index + 1),
                run_index,
                parent_sweep_id: self.sweep_id.clone(),
                resolved_strategy,
                parameter_values,
                overrides,
                base_dataset_view: self.base_dataset_view.clone(),
                bar_type: self.bar_type,
                candle_mode: self.candle_mode,
                engine_mode: self.engine_mode,
                evaluator_mode: self.evaluator_mode,
                fill_model: self.fill_model,
                latency: self.latency.clone(),
                bar_protection_policy: self.bar_protection_policy,
                primary_fee_schedule: self.primary_fee_schedule.clone(),
                fee_scenarios: self.fee_scenarios.clone(),
                initial_capital: self.initial_capital,
                margin: self.margin.clone(),
            });
        }
        Ok(children)
    }

    pub(crate) fn plan(&self) -> Result<ReplaySweepPlan> {
        Ok(ReplaySweepPlan {
            schema_version: REPLAY_SWEEP_PLAN_SCHEMA_VERSION,
            spec: self.clone(),
            children: self.expand()?,
        })
    }

    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("read replay sweep spec {}", path.display()))?;
        let spec = serde_json::from_slice::<Self>(&bytes)
            .with_context(|| format!("parse replay sweep spec {}", path.display()))?;
        spec.validate()
            .with_context(|| format!("validate replay sweep spec {}", path.display()))?;
        Ok(spec)
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        write_json(path, self, "replay sweep spec")
    }
}

fn estimate_cached_input_rows(
    cache_root: &Path,
    view: &ReplayDatasetView,
    bar_type: BarType,
    candle_mode: CandleMode,
    load_range: ReplayCacheTimeRange,
) -> Result<Option<(u64, String)>> {
    let store = ReplayDatasetViewStore::new(cache_root);
    let view_path = store.path_for(&view.id)?;
    let resolved = store.resolve_model(view, view_path)?;
    let requested = ReplayCacheCoverage {
        start: load_range.start,
        end: load_range.end,
        trading_date: None,
    };
    if let Some(file) =
        resolved
            .dataset
            .server_bars_file_for(bar_type, candle_mode, Some(&requested))
    {
        return Ok(Some((
            estimate_file_rows(file, load_range),
            "cached server bars".to_string(),
        )));
    }

    let files = resolved
        .dataset
        .raw_ticks_parquet_files_for(Some(&requested));
    if !files.is_empty() {
        let rows = files
            .iter()
            .map(|file| estimate_file_rows(file, load_range))
            .fold(0_u64, u64::saturating_add);
        return Ok(Some((rows, "cached raw ticks".to_string())));
    }
    Ok(None)
}

fn estimate_file_rows(file: &ReplayCacheDataFile, requested: ReplayCacheTimeRange) -> u64 {
    if file.row_count == 0 || file.first_timestamp > file.last_timestamp {
        return 0;
    }
    let first = file
        .first_timestamp
        .timestamp_nanos_opt()
        .unwrap_or(i64::MIN) as i128;
    let last = file
        .last_timestamp
        .timestamp_nanos_opt()
        .unwrap_or(i64::MAX) as i128;
    let requested_start = requested.start.timestamp_nanos_opt().unwrap_or(i64::MIN) as i128;
    let requested_end = requested.end.timestamp_nanos_opt().unwrap_or(i64::MAX) as i128;
    let overlap_start = first.max(requested_start);
    let overlap_end = last.saturating_add(1).min(requested_end);
    if overlap_end <= overlap_start {
        return 0;
    }
    let source_span = last.saturating_sub(first).saturating_add(1).max(1);
    let overlap_span = overlap_end.saturating_sub(overlap_start);
    let numerator = (file.row_count as u128).saturating_mul(overlap_span as u128);
    let estimate = numerator
        .saturating_add(source_span as u128 - 1)
        .checked_div(source_span as u128)
        .unwrap_or(0);
    estimate.min(file.row_count as u128) as u64
}

fn estimate_time_based_rows(bar_type: BarType, range: ReplayCacheTimeRange) -> Result<Option<u64>> {
    let interval_nanos = match bar_type.kind() {
        BarKind::Minute => (bar_type.value() as u64)
            .saturating_mul(60)
            .saturating_mul(1_000_000_000),
        BarKind::Second => (bar_type.value() as u64).saturating_mul(1_000_000_000),
        BarKind::Tick | BarKind::Volume | BarKind::Range => return Ok(None),
    };
    if interval_nanos == 0 {
        return Ok(None);
    }
    let duration = range
        .end
        .signed_duration_since(range.start)
        .num_nanoseconds()
        .context("replay sweep range is too large to estimate")?;
    if duration <= 0 {
        return Ok(Some(0));
    }
    let duration = duration as u128;
    let interval = interval_nanos as u128;
    let rows = duration
        .saturating_add(interval.saturating_sub(1))
        .checked_div(interval)
        .unwrap_or(0);
    Ok(Some(rows.min(u64::MAX as u128) as u64))
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

impl ReplaySweepPlan {
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        if self.schema_version != REPLAY_SWEEP_PLAN_SCHEMA_VERSION {
            bail!(
                "unsupported replay sweep plan version {}",
                self.schema_version
            );
        }
        self.spec.validate()?;
        let expected_children = self.spec.combination_count()?;
        if self.children.len() != expected_children {
            bail!(
                "replay sweep plan contains {} children, expected {}",
                self.children.len(),
                expected_children
            );
        }
        write_json(path, self, "replay sweep plan")
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T, label: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {} parent {}", label, parent.display()))?;
        }
    }
    let bytes = serde_json::to_vec_pretty(value).with_context(|| format!("serialize {label}"))?;
    fs::write(path, bytes).with_context(|| format!("write {label} {}", path.display()))
}

fn validate_sweep_id(id: &str) -> Result<()> {
    if id.trim().is_empty() {
        bail!("sweep_id cannot be empty");
    }
    if id == "." || id == ".." || id.chars().any(char::is_control) {
        bail!("sweep_id contains an unsafe path component: {id:?}");
    }
    if id.contains('/') || id.contains('\\') || id.chars().any(char::is_whitespace) {
        bail!("sweep_id must not contain whitespace or path separators");
    }
    Ok(())
}

fn validate_replay_models(
    engine_mode: ReplayEngineMode,
    fill_model: ReplayFillModel,
    latency: &ReplayLatencyConfig,
) -> Result<()> {
    if engine_mode == ReplayEngineMode::Deterministic
        && fill_model == ReplayFillModel::LegacyReferencePrice
    {
        bail!("deterministic sweep cannot use legacy_reference_price fill model");
    }
    if fill_model == ReplayFillModel::Dom && engine_mode != ReplayEngineMode::Deterministic {
        bail!("Level 2 DOM sweep requires deterministic replay_engine_mode");
    }
    if engine_mode == ReplayEngineMode::Deterministic {
        match latency.model {
            ReplayLatencyModel::IgnoredLegacy => {
                bail!("deterministic sweep cannot use ignored_legacy latency");
            }
            ReplayLatencyModel::Fixed => {}
            ReplayLatencyModel::ObservedMean
            | ReplayLatencyModel::ObservedP95
            | ReplayLatencyModel::ObservedP99
            | ReplayLatencyModel::SeededObserved
                if latency.observed_samples_ms.is_empty() =>
            {
                bail!(
                    "{} sweep latency requires observed_samples_ms",
                    latency.model.label()
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_fee_scenarios(
    primary: &ReplayFeeSchedule,
    scenarios: &[ReplayFeeSchedule],
) -> Result<()> {
    let mut names = BTreeSet::new();
    names.insert(primary.name.clone());
    for scenario in scenarios {
        scenario.validate()?;
        if !names.insert(scenario.name.clone()) {
            bail!("duplicate fee scenario name: {}", scenario.name);
        }
    }
    Ok(())
}

fn validate_output_formats(formats: &[ReplaySweepOutputFormat]) -> Result<()> {
    if formats.is_empty() {
        bail!("output_formats must contain at least one format");
    }
    let mut seen = BTreeSet::new();
    for format in formats {
        if !seen.insert(format.label()) {
            bail!("duplicate output format: {}", format.label());
        }
    }
    Ok(())
}

fn validate_strategy_config(config: &ExecutionStrategyConfig) -> Result<()> {
    if config.order_qty <= 0 {
        bail!("base strategy order_qty must be greater than zero");
    }
    if !config.blockout_minutes_before_close.is_finite()
        || config.blockout_minutes_before_close < 0.0
    {
        bail!("base strategy blockout_minutes_before_close must be finite and non-negative");
    }
    validate_hma_angle(&config.native_hma)?;
    validate_ema(&config.native_ema)?;
    validate_hma_cross(&config.native_hma_cross)?;
    Ok(())
}

fn validate_hma_angle(config: &HmaAngleConfig) -> Result<()> {
    if config.hma_length == 0 || config.angle_lookback == 0 || config.bars_required_to_trade == 0 {
        bail!("HMA angle lengths and bars_required_to_trade must be greater than zero");
    }
    if !config.min_angle.is_finite() {
        bail!("HMA angle min_angle must be finite");
    }
    validate_protection_values(
        "HMA angle",
        config.take_profit_ticks,
        config.stop_loss_ticks,
        config.trail_trigger_ticks,
        config.trail_offset_ticks,
    )
}

fn validate_ema(config: &EmaCrossConfig) -> Result<()> {
    if config.fast_length == 0 || config.slow_length == 0 {
        bail!("EMA fast_length and slow_length must be greater than zero");
    }
    if config.fast_length >= config.slow_length {
        bail!(
            "EMA fast_length ({}) must be less than slow_length ({})",
            config.fast_length,
            config.slow_length
        );
    }
    validate_protection_values(
        "EMA",
        config.take_profit_ticks,
        config.stop_loss_ticks,
        config.trail_trigger_ticks,
        config.trail_offset_ticks,
    )
}

fn validate_hma_cross(config: &HmaCrossConfig) -> Result<()> {
    if config.fast_length == 0 || config.slow_length == 0 {
        bail!("HMA crossover fast_length and slow_length must be greater than zero");
    }
    if config.fast_length >= config.slow_length {
        bail!(
            "HMA crossover fast_length ({}) must be less than slow_length ({})",
            config.fast_length,
            config.slow_length
        );
    }
    validate_protection_values(
        "HMA crossover",
        config.take_profit_ticks,
        config.stop_loss_ticks,
        config.trail_trigger_ticks,
        config.trail_offset_ticks,
    )
}

fn validate_protection_values(
    label: &str,
    take_profit_ticks: f64,
    stop_loss_ticks: f64,
    trail_trigger_ticks: f64,
    trail_offset_ticks: f64,
) -> Result<()> {
    for (field, value) in [
        ("take_profit_ticks", take_profit_ticks),
        ("stop_loss_ticks", stop_loss_ticks),
        ("trail_trigger_ticks", trail_trigger_ticks),
        ("trail_offset_ticks", trail_offset_ticks),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{label} {field} must be finite and non-negative");
        }
    }
    Ok(())
}

const SWEEPABLE_PARAMETER_PATHS: &[&str] = &[
    "kind",
    "native_strategy",
    "native_signal_timing",
    "native_signal_delay_bars",
    "native_execution_path",
    "native_reversal_mode",
    "blockout_enabled",
    "blockout_minutes_before_close",
    "native_hma.hma_length",
    "native_hma.min_angle",
    "native_hma.angle_lookback",
    "native_hma.bars_required_to_trade",
    "native_hma.longs_only",
    "native_hma.inverted",
    "native_hma.take_profit_ticks",
    "native_hma.stop_loss_ticks",
    "native_hma.use_trailing_stop",
    "native_hma.trail_trigger_ticks",
    "native_hma.trail_offset_ticks",
    "native_ema.fast_length",
    "native_ema.slow_length",
    "native_ema.inverted",
    "native_ema.take_profit_ticks",
    "native_ema.stop_loss_ticks",
    "native_ema.use_trailing_stop",
    "native_ema.trail_trigger_ticks",
    "native_ema.trail_offset_ticks",
    "native_hma_cross.fast_length",
    "native_hma_cross.slow_length",
    "native_hma_cross.inverted",
    "native_hma_cross.take_profit_ticks",
    "native_hma_cross.stop_loss_ticks",
    "native_hma_cross.use_trailing_stop",
    "native_hma_cross.trail_trigger_ticks",
    "native_hma_cross.trail_offset_ticks",
    "order_qty",
];

fn validate_parameter_path(path: &str) -> Result<()> {
    if !SWEEPABLE_PARAMETER_PATHS.contains(&path) {
        bail!(
            "unsupported sweep parameter path {path:?}; supported paths are strategy scalar fields"
        );
    }
    Ok(())
}

fn is_scalar_value(value: &Value) -> bool {
    !value.is_array() && !value.is_object()
}

fn numeric_value<'a>(value: &'a Value, path: &str) -> Result<f64> {
    value
        .as_f64()
        .with_context(|| format!("constraint path {path} must resolve to a number"))
}

fn value_at_path<'a>(root: &'a Value, path: &str) -> Result<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        let object = current
            .as_object()
            .with_context(|| format!("path {path} does not resolve through an object"))?;
        current = object
            .get(segment)
            .with_context(|| format!("path {path} has no field {segment}"))?;
    }
    Ok(current)
}

fn set_value_at_path(root: &mut Value, path: &str, value: Value) -> Result<()> {
    let segments = path.split('.').collect::<Vec<_>>();
    let (last, parents) = segments
        .split_last()
        .context("sweep parameter path cannot be empty")?;
    let mut current = root;
    for segment in parents {
        let object = current
            .as_object_mut()
            .with_context(|| format!("path {path} does not resolve through an object"))?;
        current = object
            .get_mut(*segment)
            .with_context(|| format!("path {path} has no field {segment}"))?;
    }
    let object = current
        .as_object_mut()
        .with_context(|| format!("path {path} parent is not an object"))?;
    if !object.contains_key(*last) {
        bail!("path {path} has no field {last}");
    }
    object.insert((*last).to_string(), value);
    Ok(())
}

fn resolve_strategy(
    base: &ExecutionStrategyConfig,
    values: &BTreeMap<String, Value>,
) -> Result<ExecutionStrategyConfig> {
    let mut encoded = serde_json::to_value(base).context("serialize base strategy")?;
    for (path, value) in values {
        set_value_at_path(&mut encoded, path, value.clone())?;
    }
    let resolved = serde_json::from_value(encoded).context("deserialize resolved strategy")?;
    validate_strategy_config(&resolved)?;
    Ok(resolved)
}

fn cartesian_values(parameters: &[ReplaySweepParameter]) -> Vec<Vec<Value>> {
    let mut combinations = vec![Vec::new()];
    for parameter in parameters {
        let mut next = Vec::with_capacity(combinations.len() * parameter.values.len());
        for prefix in combinations {
            for value in &parameter.values {
                let mut combination = prefix.clone();
                combination.push(value.clone());
                next.push(combination);
            }
        }
        combinations = next;
    }
    combinations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{BrokerKind, ReplayBarProtectionPolicy};
    use crate::config::TradingEnvironment;
    use crate::replay_cache::{
        ReplayDatasetSessionPreset, ReplayDatasetSourceRef, ReplayDatasetWarmupPolicy,
    };
    use chrono::{Duration, TimeZone, Utc};

    fn sample_view() -> ReplayDatasetView {
        ReplayDatasetView {
            view_version: crate::replay_cache::REPLAY_DATASET_VIEW_VERSION,
            id: "mes-sample".to_string(),
            source: ReplayDatasetSourceRef {
                manifest_id: "mes/manifest.json".to_string(),
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
            },
            evaluation_start: Utc.with_ymd_and_hms(2026, 7, 1, 13, 30, 0).unwrap(),
            evaluation_end: Utc.with_ymd_and_hms(2026, 7, 1, 14, 30, 0).unwrap(),
            input_timezone: "UTC".to_string(),
            session_preset: ReplayDatasetSessionPreset::FullSource,
            warmup: ReplayDatasetWarmupPolicy {
                duration_seconds: 600,
                ..ReplayDatasetWarmupPolicy::default()
            },
        }
    }

    fn sample_spec() -> ReplaySweepSpec {
        let mut base_strategy = ExecutionStrategyConfig::default();
        base_strategy.native_strategy = crate::strategy::NativeStrategyKind::EmaCross;
        ReplaySweepSpec {
            schema_version: REPLAY_SWEEP_SPEC_SCHEMA_VERSION,
            sweep_id: "ema-mes".to_string(),
            name: "MES EMA grid".to_string(),
            notes: "regression fixture".to_string(),
            base_dataset_view: sample_view(),
            base_strategy,
            parameters: vec![
                ReplaySweepParameter {
                    path: "native_ema.fast_length".to_string(),
                    values: vec![Value::from(5), Value::from(10)],
                },
                ReplaySweepParameter {
                    path: "native_ema.slow_length".to_string(),
                    values: vec![Value::from(20), Value::from(30)],
                },
            ],
            constraints: vec![ReplaySweepConstraint::Less {
                left: "native_ema.fast_length".to_string(),
                right: "native_ema.slow_length".to_string(),
            }],
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            engine_mode: ReplayEngineMode::Deterministic,
            evaluator_mode: crate::broker::ReplayEvaluatorMode::Legacy,
            fill_model: ReplayFillModel::RawBarOpen,
            latency: ReplayLatencyConfig::default(),
            bar_protection_policy: ReplayBarProtectionPolicy::Conservative,
            primary_fee_schedule: ReplayFeeSchedule::default(),
            fee_scenarios: vec![ReplayFeeSchedule {
                name: "observed".to_string(),
                commission_per_contract: 1.25,
                ..ReplayFeeSchedule::default()
            }],
            initial_capital: 50_000.0,
            margin: Some(ReplayMarginConfig {
                margin_per_contract: 1_500.0,
                ..ReplayMarginConfig::default()
            }),
            max_runs: 4,
            parallelism: 2,
            execution_mode: ReplaySweepExecutionMode::BatchCpu,
            guardrails: ReplaySweepGuardrails::default(),
            output_dir: "runs/ema-mes".into(),
            output_formats: vec![
                ReplaySweepOutputFormat::JsonSummary,
                ReplaySweepOutputFormat::CsvRows,
            ],
        }
    }

    #[test]
    fn expands_cartesian_grid_with_exact_values_and_differing_overrides() {
        let plan = sample_spec().plan().expect("valid plan");
        assert_eq!(plan.spec.execution_mode, ReplaySweepExecutionMode::BatchCpu);
        assert_eq!(plan.children.len(), 4);
        assert_eq!(plan.children[0].run_id, "ema-mes-000001");
        assert_eq!(
            plan.children[0].parameter_values["native_ema.fast_length"],
            Value::from(5)
        );
        assert_eq!(
            plan.children[0].parameter_values["native_ema.slow_length"],
            Value::from(20)
        );
        assert_eq!(
            plan.children[0].overrides["native_ema.fast_length"],
            Value::from(5)
        );
        assert_eq!(plan.children[0].resolved_strategy.native_ema.fast_length, 5);
        assert_eq!(
            plan.children[0].resolved_strategy.native_ema.slow_length,
            20
        );
        assert_eq!(plan.children[0].primary_fee_schedule.name, "fee_neutral");
        assert_eq!(plan.children[0].fee_scenarios[0].name, "observed");
        assert_eq!(
            plan.children[0]
                .margin
                .as_ref()
                .unwrap()
                .margin_per_contract,
            1_500.0
        );
    }

    #[test]
    fn execution_mode_parses_stable_cli_aliases() {
        assert_eq!(
            ReplaySweepExecutionMode::parse("batch-cpu").expect("batch mode"),
            ReplaySweepExecutionMode::BatchCpu
        );
        assert_eq!(
            ReplaySweepExecutionMode::parse("legacy").expect("legacy alias"),
            ReplaySweepExecutionMode::IsolatedServices
        );
        assert!(ReplaySweepExecutionMode::parse("gpu").is_err());
        let encoded =
            serde_json::to_string(&ReplaySweepExecutionMode::BatchCpu).expect("serialize mode");
        assert_eq!(encoded, "\"batch_cpu\"");
    }

    #[test]
    fn base_values_are_not_reported_as_differing_overrides() {
        let mut spec = sample_spec();
        spec.parameters[0].values = vec![Value::from(21)];
        spec.parameters[1].values = vec![Value::from(55)];
        spec.max_runs = 1;
        let child = &spec.plan().expect("valid plan").children[0];
        assert_eq!(child.parameter_values.len(), 2);
        assert!(child.overrides.is_empty());
    }

    #[test]
    fn invalid_fast_slow_combo_is_rejected_before_run() {
        let mut spec = sample_spec();
        spec.parameters[0].values = vec![Value::from(40)];
        spec.parameters[1].values = vec![Value::from(30)];
        spec.max_runs = 1;
        let error = spec.plan().expect_err("invalid constraint must fail");
        assert!(format!("{error:#}").contains("fast_length"));
    }

    #[test]
    fn invalid_spec_limits_and_unknown_paths_are_rejected() {
        let mut spec = sample_spec();
        spec.max_runs = 3;
        let error = spec.validate().expect_err("max runs must reject grid");
        assert!(error.to_string().contains("exceeding max_runs"));

        let mut spec = sample_spec();
        spec.parameters[0].path = "native_ema.not_a_field".to_string();
        assert!(spec.validate().is_err());
    }

    #[test]
    fn spec_and_plan_round_trip_through_json_files() {
        let spec = sample_spec();
        let suffix = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let path = std::env::temp_dir().join(format!("trader-replay-sweep-{suffix}.json"));
        let plan_path = std::env::temp_dir().join(format!("trader-replay-plan-{suffix}.json"));
        spec.save(&path).expect("save spec");
        let loaded = ReplaySweepSpec::load(&path).expect("load spec");
        assert_eq!(loaded, spec);
        let plan = loaded.plan().expect("plan");
        plan.save(&plan_path).expect("save plan");
        let encoded = std::fs::read_to_string(&plan_path).expect("read plan");
        assert!(encoded.contains("ema-mes-000001"));
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(plan_path);
    }

    #[test]
    fn dataset_view_model_is_checked_but_source_files_are_not_required() {
        let view = sample_view();
        view.validate_model().expect("model-only validation");
        assert_eq!(
            view.load_range().unwrap().start,
            view.evaluation_start - Duration::seconds(600)
        );
    }

    #[test]
    fn resource_estimate_includes_warmup_and_parallel_memory() {
        let spec = sample_spec();
        let estimate = spec.resource_estimate(None).expect("estimate");
        assert_eq!(estimate.combinations, 4);
        assert_eq!(estimate.parallel_jobs, 2);
        assert_eq!(estimate.estimated_input_rows, Some(70));
        assert_eq!(estimate.estimated_input_rows_per_worker, Some(70));
        assert!(estimate.estimated_memory_bytes.unwrap_or_default() > 0);
        assert!(estimate.estimated_output_bytes.unwrap_or_default() > 0);
        assert!(estimate.estimated_runtime_seconds.unwrap_or_default() >= 1);
        assert!(estimate.input_source.contains("time-based"));
    }

    #[test]
    fn guardrail_report_separates_hard_limits_from_large_confirmation() {
        let mut spec = sample_spec();
        spec.guardrails.max_combinations = 3;
        spec.guardrails.large_sweep_threshold = 4;
        let report = spec.guardrail_report(None).expect("guardrail report");
        assert_eq!(report.estimate.combinations, 4);
        assert!(
            report
                .violations
                .iter()
                .any(|message| message.contains("max_combinations"))
        );
        assert!(report.requires_confirmation);
        assert!(
            report
                .warnings
                .iter()
                .any(|message| message.contains("confirmation threshold"))
        );
    }

    #[test]
    fn tick_estimate_is_explicitly_unknown_without_cache_metadata() {
        let mut spec = sample_spec();
        spec.bar_type = BarType::tick(100);
        let estimate = spec.resource_estimate(None).expect("estimate");
        assert_eq!(estimate.estimated_input_rows, None);
        assert!(estimate.estimated_memory_bytes.is_none());
        assert!(
            estimate
                .notes
                .iter()
                .any(|message| message.contains("cannot be estimated"))
        );
    }
}
