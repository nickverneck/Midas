//! Offline aggregation and out-of-sample evaluation for walk-forward sweeps.
//!
//! A walk-forward plan creates one ordinary sweep for each fold/phase.  This
//! module joins those persisted phase artifacts back together without opening
//! the replay cache or rerunning a child.  Candidate identity is the exact
//! pair of `parameter_values` and `overrides`; generated run ids are only
//! diagnostics and are never used as a join key.

use super::results::{ReplayFeeScenario, ReplayResultDocument, ReplayResultStatus};
use super::sweep::{ReplaySweepChildSpec, ReplaySweepPlan, ReplaySweepSpec};
use super::sweep_analytics::ReplaySweepRankingMetric;
use super::sweep_runner::ReplaySweepSummaryDocument;
use super::walk_forward::{ReplayWalkForwardPhase, ReplayWalkForwardPlan, ReplayWalkForwardWindow};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const REPLAY_WALK_FORWARD_EVALUATION_SCHEMA_VERSION: u32 = 1;

/// Which phase(s) are allowed to select the candidate.  Test metrics are
/// deliberately absent from this policy and can therefore never influence
/// selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayWalkForwardSelectionPolicy {
    /// Select the highest configured metric on the training phase.
    Train,
    /// Require a valid training result, then select by validation metric.  If
    /// validation is unavailable, selection falls back to training with a
    /// warning.  Test is never consulted.
    TrainThenValidation,
}

impl Default for ReplayWalkForwardSelectionPolicy {
    fn default() -> Self {
        Self::Train
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub(crate) struct ReplayWalkForwardEvaluationOptions {
    pub(crate) metric: ReplaySweepRankingMetric,
    /// `None` and `"active"` use the result's active fee scenario.  A named
    /// scenario is read from the saved accounting overlay only.
    pub(crate) fee_scenario: Option<String>,
    pub(crate) selection_policy: ReplayWalkForwardSelectionPolicy,
}

impl Default for ReplayWalkForwardEvaluationOptions {
    fn default() -> Self {
        Self {
            metric: ReplaySweepRankingMetric::Robustness,
            fee_scenario: None,
            selection_policy: ReplayWalkForwardSelectionPolicy::Train,
        }
    }
}

impl ReplayWalkForwardEvaluationOptions {
    fn validate(&self) -> Result<()> {
        if self
            .fee_scenario
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            bail!("walk-forward fee_scenario cannot be empty when supplied");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayWalkForwardPhaseStatus {
    Completed,
    Partial,
    Failed,
    Missing,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardPhaseReport {
    pub(crate) fold_index: usize,
    pub(crate) phase: ReplayWalkForwardPhase,
    pub(crate) sweep_id: String,
    pub(crate) status: ReplayWalkForwardPhaseStatus,
    pub(crate) spec_path: PathBuf,
    pub(crate) sweep_plan_path: Option<PathBuf>,
    pub(crate) sweep_summary_path: Option<PathBuf>,
    pub(crate) expected_run_count: usize,
    pub(crate) completed_count: usize,
    pub(crate) failed_count: usize,
    pub(crate) candidate_count: usize,
    pub(crate) warnings: Vec<String>,
}

/// Metrics from one persisted candidate/phase.  `metric_value` is populated
/// for every phase for transparency, but only train/validation values are
/// eligible for selection.  Validation and test rows are the OOS outputs of
/// the selected candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardCandidatePhase {
    pub(crate) phase: ReplayWalkForwardPhase,
    pub(crate) status: ReplayWalkForwardPhaseStatus,
    pub(crate) run_id: Option<String>,
    pub(crate) result_path: Option<PathBuf>,
    pub(crate) metric_value: Option<f64>,
    pub(crate) gross_pnl: Option<f64>,
    pub(crate) net_pnl: Option<f64>,
    pub(crate) fees: Option<f64>,
    pub(crate) ending_equity: Option<f64>,
    pub(crate) return_on_initial_capital_pct: Option<f64>,
    pub(crate) required_starting_capital: Option<f64>,
    pub(crate) return_on_required_account_size_pct: Option<f64>,
    pub(crate) max_drawdown: Option<f64>,
    pub(crate) max_drawdown_pct: Option<f64>,
    pub(crate) profit_factor: Option<f64>,
    pub(crate) closed_trade_count: Option<usize>,
    pub(crate) win_rate_pct: Option<f64>,
    pub(crate) error: Option<String>,
}

impl ReplayWalkForwardCandidatePhase {
    fn missing(phase: ReplayWalkForwardPhase) -> Self {
        Self {
            phase,
            status: ReplayWalkForwardPhaseStatus::Missing,
            run_id: None,
            result_path: None,
            metric_value: None,
            gross_pnl: None,
            net_pnl: None,
            fees: None,
            ending_equity: None,
            return_on_initial_capital_pct: None,
            required_starting_capital: None,
            return_on_required_account_size_pct: None,
            max_drawdown: None,
            max_drawdown_pct: None,
            profit_factor: None,
            closed_trade_count: None,
            win_rate_pct: None,
            error: None,
        }
    }
}

/// One candidate joined across train, validation, and test by its parameter
/// values and overrides.  The selected row's validation/test metrics are the
/// only out-of-sample result used by callers for reporting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardCandidateReport {
    pub(crate) fold_index: usize,
    pub(crate) parameter_values: BTreeMap<String, Value>,
    pub(crate) overrides: BTreeMap<String, Value>,
    pub(crate) selected: bool,
    pub(crate) selection_phase: Option<ReplayWalkForwardPhase>,
    pub(crate) selection_metric_value: Option<f64>,
    pub(crate) train: Option<ReplayWalkForwardCandidatePhase>,
    pub(crate) validation: Option<ReplayWalkForwardCandidatePhase>,
    pub(crate) test: Option<ReplayWalkForwardCandidatePhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardFoldReport {
    pub(crate) fold_index: usize,
    pub(crate) phases: Vec<ReplayWalkForwardPhaseReport>,
    pub(crate) candidates: Vec<ReplayWalkForwardCandidateReport>,
    pub(crate) selected_parameter_values: Option<BTreeMap<String, Value>>,
    pub(crate) selected_overrides: Option<BTreeMap<String, Value>>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayWalkForwardEvaluationDocument {
    pub(crate) schema_version: u32,
    pub(crate) plan_path: PathBuf,
    pub(crate) generated_at_utc: DateTime<Utc>,
    pub(crate) options: ReplayWalkForwardEvaluationOptions,
    pub(crate) folds: Vec<ReplayWalkForwardFoldReport>,
    pub(crate) warnings: Vec<String>,
}

/// Evaluate a persisted walk-forward plan using only saved phase plans,
/// summaries, and child `result.json` files.  `output_path` and `csv_path`
/// are optional persisted report locations.
pub(crate) fn evaluate_replay_walk_forward(
    plan_path: &Path,
    options: ReplayWalkForwardEvaluationOptions,
    output_path: Option<&Path>,
    csv_path: Option<&Path>,
) -> Result<ReplayWalkForwardEvaluationDocument> {
    options.validate()?;
    let plan = ReplayWalkForwardPlan::load(plan_path)?;
    let mut warnings = plan.warnings.clone();
    let mut folds = Vec::with_capacity(plan.folds.len());
    for fold in &plan.folds {
        let (report, mut fold_warnings) = evaluate_fold(plan_path, &plan, fold, &options);
        warnings.append(&mut fold_warnings);
        folds.push(report);
    }
    let document = ReplayWalkForwardEvaluationDocument {
        schema_version: REPLAY_WALK_FORWARD_EVALUATION_SCHEMA_VERSION,
        plan_path: plan_path.to_path_buf(),
        generated_at_utc: Utc::now(),
        options,
        folds,
        warnings,
    };
    if let Some(path) = output_path {
        write_json_atomic(path, &document)?;
    }
    if let Some(path) = csv_path {
        write_bytes_atomic(path, &evaluation_csv(&document))?;
    }
    Ok(document)
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct CandidateKey {
    parameter_values: String,
    overrides: String,
}

impl CandidateKey {
    fn new(
        parameter_values: &BTreeMap<String, Value>,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<Self> {
        Ok(Self {
            parameter_values: serde_json::to_string(parameter_values)
                .context("serialize walk-forward parameter values")?,
            overrides: serde_json::to_string(overrides)
                .context("serialize walk-forward overrides")?,
        })
    }
}

#[derive(Debug, Clone)]
struct PhaseData {
    report: ReplayWalkForwardPhaseReport,
    candidates: BTreeMap<CandidateKey, ReplayWalkForwardCandidatePhase>,
    candidate_maps: BTreeMap<CandidateKey, (BTreeMap<String, Value>, BTreeMap<String, Value>)>,
}

fn evaluate_fold(
    plan_path: &Path,
    plan: &ReplayWalkForwardPlan,
    fold: &super::walk_forward::ReplayWalkForwardFold,
    options: &ReplayWalkForwardEvaluationOptions,
) -> (ReplayWalkForwardFoldReport, Vec<String>) {
    let mut warnings = Vec::new();
    let mut phase_data = Vec::new();
    phase_data.push(load_phase(
        plan_path,
        &plan.base_spec,
        fold.fold_index,
        &fold.train,
        options,
    ));
    if let Some(validation) = fold.validation.as_ref() {
        phase_data.push(load_phase(
            plan_path,
            &plan.base_spec,
            fold.fold_index,
            validation,
            options,
        ));
    }
    phase_data.push(load_phase(
        plan_path,
        &plan.base_spec,
        fold.fold_index,
        &fold.test,
        options,
    ));

    let phases = phase_data
        .iter()
        .map(|phase| phase.report.clone())
        .collect::<Vec<_>>();
    let mut keys = BTreeSet::new();
    for phase in &phase_data {
        keys.extend(phase.candidates.keys().cloned());
        warnings.extend(phase.report.warnings.iter().cloned());
    }

    let mut candidates = Vec::with_capacity(keys.len());
    for key in keys {
        let mut maps = None;
        let mut train = None;
        let mut validation = None;
        let mut test = None;
        for phase in &phase_data {
            if let Some(candidate) = phase.candidates.get(&key) {
                maps = phase.candidate_maps.get(&key).cloned().or(maps);
                match candidate.phase {
                    ReplayWalkForwardPhase::Train => train = Some(candidate.clone()),
                    ReplayWalkForwardPhase::Validation => validation = Some(candidate.clone()),
                    ReplayWalkForwardPhase::Test => test = Some(candidate.clone()),
                }
            }
        }
        let Some((parameter_values, overrides)) = maps else {
            warnings.push(format!(
                "fold {} candidate key had no parameter metadata and was omitted",
                fold.fold_index
            ));
            continue;
        };
        candidates.push(ReplayWalkForwardCandidateReport {
            fold_index: fold.fold_index,
            parameter_values,
            overrides,
            selected: false,
            selection_phase: None,
            selection_metric_value: None,
            train,
            validation,
            test,
        });
    }

    let selected = select_candidate(
        &mut candidates,
        options,
        fold.validation.is_some(),
        &mut warnings,
    );
    let (selected_parameter_values, selected_overrides) = selected
        .map(|index| {
            candidates[index].selected = true;
            candidates[index].selection_phase = match options.selection_policy {
                ReplayWalkForwardSelectionPolicy::Train => Some(ReplayWalkForwardPhase::Train),
                ReplayWalkForwardSelectionPolicy::TrainThenValidation => {
                    if candidates[index]
                        .validation
                        .as_ref()
                        .and_then(|metrics| metrics.metric_value)
                        .is_some_and(f64::is_finite)
                    {
                        Some(ReplayWalkForwardPhase::Validation)
                    } else {
                        Some(ReplayWalkForwardPhase::Train)
                    }
                }
            };
            candidates[index].selection_metric_value = candidates[index]
                .selection_phase
                .and_then(|phase| phase_metrics(&candidates[index], phase))
                .and_then(|metrics| metrics.metric_value);
            (
                Some(candidates[index].parameter_values.clone()),
                Some(candidates[index].overrides.clone()),
            )
        })
        .unwrap_or((None, None));

    let report = ReplayWalkForwardFoldReport {
        fold_index: fold.fold_index,
        phases,
        candidates,
        selected_parameter_values,
        selected_overrides,
        warnings: warnings.clone(),
    };
    (report, warnings)
}

fn select_candidate(
    candidates: &mut [ReplayWalkForwardCandidateReport],
    options: &ReplayWalkForwardEvaluationOptions,
    has_validation_phase: bool,
    warnings: &mut Vec<String>,
) -> Option<usize> {
    let train_best = || {
        candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let value = candidate
                    .train
                    .as_ref()
                    .and_then(|phase| phase.metric_value)
                    .filter(|value| value.is_finite())?;
                Some((index, value))
            })
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right).then_with(|| {
                    parameter_sort_key(&candidates[*right_index])
                        .cmp(&parameter_sort_key(&candidates[*left_index]))
                })
            })
            .map(|(index, _)| index)
    };

    match options.selection_policy {
        ReplayWalkForwardSelectionPolicy::Train => train_best(),
        ReplayWalkForwardSelectionPolicy::TrainThenValidation => {
            if !has_validation_phase {
                warnings.push(
                    "train-then-validation selection requested but this fold has no validation phase; used train"
                        .to_string(),
                );
                return train_best();
            }
            let validation_best = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    let train = candidate
                        .train
                        .as_ref()
                        .and_then(|phase| phase.metric_value)
                        .filter(|value| value.is_finite())?;
                    let validation = candidate
                        .validation
                        .as_ref()
                        .and_then(|phase| phase.metric_value)
                        .filter(|value| value.is_finite())?;
                    Some((index, train, validation))
                })
                .max_by(
                    |(left_index, left_train, left_validation),
                     (right_index, right_train, right_validation)| {
                        left_validation
                            .total_cmp(right_validation)
                            .then_with(|| left_train.total_cmp(right_train))
                            .then_with(|| {
                                parameter_sort_key(&candidates[*right_index])
                                    .cmp(&parameter_sort_key(&candidates[*left_index]))
                            })
                    },
                )
                .map(|(index, _, _)| index);
            if validation_best.is_none() {
                warnings.push(
                    "no candidate had both completed train and validation metrics; used train"
                        .to_string(),
                );
                train_best()
            } else {
                validation_best
            }
        }
    }
}

fn phase_metrics<'a>(
    candidate: &'a ReplayWalkForwardCandidateReport,
    phase: ReplayWalkForwardPhase,
) -> Option<&'a ReplayWalkForwardCandidatePhase> {
    match phase {
        ReplayWalkForwardPhase::Train => candidate.train.as_ref(),
        ReplayWalkForwardPhase::Validation => candidate.validation.as_ref(),
        ReplayWalkForwardPhase::Test => candidate.test.as_ref(),
    }
}

fn parameter_sort_key(candidate: &ReplayWalkForwardCandidateReport) -> String {
    format!(
        "{}|{}",
        serde_json::to_string(&candidate.parameter_values).unwrap_or_default(),
        serde_json::to_string(&candidate.overrides).unwrap_or_default()
    )
}

fn load_phase(
    plan_path: &Path,
    base_spec: &ReplaySweepSpec,
    fold_index: usize,
    window: &ReplayWalkForwardWindow,
    options: &ReplayWalkForwardEvaluationOptions,
) -> PhaseData {
    let anchor = plan_path.parent().unwrap_or_else(|| Path::new("."));
    let spec_path = resolve_path(anchor, &window.spec_path);
    let mut report = ReplayWalkForwardPhaseReport {
        fold_index,
        phase: window.phase,
        sweep_id: window.sweep_id.clone(),
        status: ReplayWalkForwardPhaseStatus::Missing,
        spec_path: spec_path.clone(),
        sweep_plan_path: None,
        sweep_summary_path: None,
        expected_run_count: 0,
        completed_count: 0,
        failed_count: 0,
        candidate_count: 0,
        warnings: Vec::new(),
    };
    if !spec_path.is_file() {
        report.warnings.push(format!(
            "missing {} phase sweep spec {}",
            phase_label(window.phase),
            spec_path.display()
        ));
        return empty_phase(&report);
    }
    let phase_spec = match ReplaySweepSpec::load(&spec_path) {
        Ok(spec) => spec,
        Err(error) => {
            report.status = ReplayWalkForwardPhaseStatus::Invalid;
            report.warnings.push(format!(
                "could not load {} phase sweep spec {}: {error}",
                phase_label(window.phase),
                spec_path.display()
            ));
            return empty_phase(&report);
        }
    };
    let mismatches = phase_spec_mismatches(base_spec, &phase_spec, window);
    if !mismatches.is_empty() {
        report.status = ReplayWalkForwardPhaseStatus::Invalid;
        report.warnings.extend(mismatches);
        return empty_phase(&report);
    }

    let phase_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let sweep_plan_path = phase_dir.join("sweep-plan.json");
    let sweep_summary_path = phase_dir.join("sweep-summary.json");
    report.sweep_plan_path = Some(sweep_plan_path.clone());
    report.sweep_summary_path = Some(sweep_summary_path.clone());
    if !sweep_plan_path.is_file() || !sweep_summary_path.is_file() {
        report.warnings.push(format!(
            "missing {} phase artifact(s) under {}{}{}",
            phase_label(window.phase),
            phase_dir.display(),
            if !sweep_plan_path.is_file() {
                " sweep-plan.json"
            } else {
                ""
            },
            if !sweep_summary_path.is_file() {
                " sweep-summary.json"
            } else {
                ""
            },
        ));
        return empty_phase(&report);
    }
    let phase_plan = match load_sweep_plan(&sweep_plan_path) {
        Ok(plan) => plan,
        Err(error) => {
            report.status = ReplayWalkForwardPhaseStatus::Invalid;
            report.warnings.push(format!(
                "could not load {} phase sweep plan: {error}",
                phase_label(window.phase)
            ));
            return empty_phase(&report);
        }
    };
    if phase_plan.spec != phase_spec {
        report.status = ReplayWalkForwardPhaseStatus::Invalid;
        report
            .warnings
            .push("phase sweep-plan.json spec does not match sweep.json".to_string());
        return empty_phase(&report);
    }
    report.expected_run_count = phase_plan.children.len();
    let expected = expected_children(&phase_plan.children, &mut report.warnings);

    let summary = match load_sweep_summary(&sweep_summary_path) {
        Ok(summary) => summary,
        Err(error) => {
            report.status = ReplayWalkForwardPhaseStatus::Invalid;
            report.warnings.push(format!(
                "could not load {} phase sweep summary: {error}",
                phase_label(window.phase)
            ));
            return empty_phase(&report);
        }
    };
    if summary.sweep_id != phase_spec.sweep_id {
        report.status = ReplayWalkForwardPhaseStatus::Invalid;
        report.warnings.push(format!(
            "phase summary sweep id {} does not match phase spec {}",
            summary.sweep_id, phase_spec.sweep_id
        ));
        return empty_phase(&report);
    }
    if summary.run_count != phase_plan.children.len() {
        report.warnings.push(format!(
            "phase summary reports {} runs but phase plan contains {}",
            summary.run_count,
            phase_plan.children.len()
        ));
    }

    let mut candidates = BTreeMap::new();
    let mut candidate_maps = BTreeMap::new();
    for run in &summary.runs {
        if run.status != "completed" {
            report.failed_count += 1;
            report.warnings.push(format!(
                "phase child {} is {}{}",
                run.run_id,
                run.status,
                run.error
                    .as_deref()
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            ));
            continue;
        }
        report.completed_count += 1;
        let Some(raw_result_path) = run.result_path.as_deref() else {
            report.failed_count += 1;
            report.warnings.push(format!(
                "completed phase child {} has no result path",
                run.run_id
            ));
            continue;
        };
        let result_path = resolve_path(
            sweep_summary_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
            raw_result_path,
        );
        let result = match load_result(&result_path) {
            Ok(result) => result,
            Err(error) => {
                report.failed_count += 1;
                report.warnings.push(format!(
                    "could not load phase child result {}: {error}",
                    result_path.display()
                ));
                continue;
            }
        };
        if !matches!(result.status, ReplayResultStatus::Completed) {
            report.failed_count += 1;
            report.warnings.push(format!(
                "phase child {} result is not completed{}",
                run.run_id,
                result
                    .error
                    .as_deref()
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            ));
            continue;
        }
        let Some(sweep) = result.sweep.as_ref() else {
            report.failed_count += 1;
            report.warnings.push(format!(
                "phase child result {} has no persisted sweep metadata; cannot join by parameters",
                result_path.display()
            ));
            continue;
        };
        let (parameter_values, overrides) = match extract_parameter_maps(sweep) {
            Ok(maps) => maps,
            Err(error) => {
                report.failed_count += 1;
                report.warnings.push(format!(
                    "phase child result {} has invalid parameter metadata: {error}",
                    result_path.display()
                ));
                continue;
            }
        };
        let key = match CandidateKey::new(&parameter_values, &overrides) {
            Ok(key) => key,
            Err(error) => {
                report.failed_count += 1;
                report.warnings.push(error.to_string());
                continue;
            }
        };
        let Some(child) = expected.get(&key) else {
            report.failed_count += 1;
            report.warnings.push(format!(
                "phase child result {} parameters do not exist in sweep-plan.json",
                result_path.display()
            ));
            continue;
        };
        let mut mismatches = sweep_mismatches(sweep, child);
        mismatches.extend(result_metadata_mismatches(&result, &phase_spec, child));
        if !mismatches.is_empty() {
            report.failed_count += 1;
            report.warnings.extend(
                mismatches.into_iter().map(|message| {
                    format!("phase child result {}: {message}", result_path.display())
                }),
            );
            continue;
        }
        let metrics = match metrics_from_result(
            window.phase,
            &result,
            result_path.clone(),
            run.run_id.clone(),
            options,
        ) {
            Ok(metrics) => metrics,
            Err(error) => {
                report.failed_count += 1;
                report.warnings.push(format!(
                    "phase child result {} has no requested fee scenario: {error}",
                    result_path.display()
                ));
                continue;
            }
        };
        if candidates.insert(key.clone(), metrics).is_some() {
            report.failed_count += 1;
            report.warnings.push(format!(
                "duplicate parameter_values/overrides candidate in phase {}",
                phase_label(window.phase)
            ));
            continue;
        }
        candidate_maps.insert(key, (parameter_values, overrides));
    }
    report.candidate_count = candidates.len();
    report.status = if report.completed_count == 0 && report.failed_count > 0 {
        ReplayWalkForwardPhaseStatus::Failed
    } else if report.failed_count > 0 {
        ReplayWalkForwardPhaseStatus::Partial
    } else {
        ReplayWalkForwardPhaseStatus::Completed
    };
    PhaseData {
        report,
        candidates,
        candidate_maps,
    }
}

fn load_sweep_plan(path: &Path) -> Result<ReplaySweepPlan> {
    let bytes = fs::read(path).with_context(|| format!("read sweep plan {}", path.display()))?;
    let plan = serde_json::from_slice::<ReplaySweepPlan>(&bytes)
        .with_context(|| format!("parse sweep plan {}", path.display()))?;
    if plan.schema_version != super::sweep::REPLAY_SWEEP_PLAN_SCHEMA_VERSION {
        bail!(
            "unsupported sweep plan schema {}; expected {}",
            plan.schema_version,
            super::sweep::REPLAY_SWEEP_PLAN_SCHEMA_VERSION
        );
    }
    plan.spec.validate()?;
    Ok(plan)
}

fn empty_phase(report: &ReplayWalkForwardPhaseReport) -> PhaseData {
    PhaseData {
        report: report.clone(),
        candidates: BTreeMap::new(),
        candidate_maps: BTreeMap::new(),
    }
}

fn load_sweep_summary(path: &Path) -> Result<ReplaySweepSummaryDocument> {
    let bytes = fs::read(path).with_context(|| format!("read sweep summary {}", path.display()))?;
    let summary = serde_json::from_slice::<ReplaySweepSummaryDocument>(&bytes)
        .with_context(|| format!("parse sweep summary {}", path.display()))?;
    if summary.schema_version != super::sweep_runner::REPLAY_SWEEP_SUMMARY_SCHEMA_VERSION {
        bail!(
            "unsupported sweep summary schema {}; expected {}",
            summary.schema_version,
            super::sweep_runner::REPLAY_SWEEP_SUMMARY_SCHEMA_VERSION
        );
    }
    Ok(summary)
}

fn load_result(path: &Path) -> Result<ReplayResultDocument> {
    let bytes = fs::read(path).with_context(|| format!("read phase result {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse phase result {}", path.display()))
}

fn expected_children<'a>(
    children: &'a [ReplaySweepChildSpec],
    warnings: &mut Vec<String>,
) -> BTreeMap<CandidateKey, &'a ReplaySweepChildSpec> {
    let mut expected = BTreeMap::new();
    for child in children {
        let Ok(key) = CandidateKey::new(&child.parameter_values, &child.overrides) else {
            warnings.push(format!(
                "could not serialize phase child {} parameter metadata",
                child.run_id
            ));
            continue;
        };
        if expected.insert(key, child).is_some() {
            warnings.push(format!(
                "duplicate parameter_values/overrides in phase plan at child {}",
                child.run_id
            ));
        }
    }
    expected
}

fn extract_parameter_maps(
    sweep: &Value,
) -> Result<(BTreeMap<String, Value>, BTreeMap<String, Value>)> {
    let parameter_values = sweep
        .get("parameter_values")
        .cloned()
        .context("missing parameter_values")
        .and_then(|value| {
            serde_json::from_value(value).context("parse parameter_values as an object")
        })?;
    let overrides = sweep
        .get("overrides")
        .cloned()
        .context("missing overrides")
        .and_then(|value| serde_json::from_value(value).context("parse overrides as an object"))?;
    Ok((parameter_values, overrides))
}

fn phase_spec_mismatches(
    base: &ReplaySweepSpec,
    phase: &ReplaySweepSpec,
    window: &ReplayWalkForwardWindow,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    if phase.sweep_id != window.sweep_id {
        mismatches.push(format!(
            "phase sweep id {} does not match plan window {}",
            phase.sweep_id, window.sweep_id
        ));
    }
    if phase.base_dataset_view.id != window.view_id {
        mismatches.push(format!(
            "phase dataset view id {} does not match plan window {}",
            phase.base_dataset_view.id, window.view_id
        ));
    }
    if phase.base_dataset_view.evaluation_start != window.evaluation_start
        || phase.base_dataset_view.evaluation_end != window.evaluation_end
    {
        mismatches.push("phase evaluation range does not match walk-forward window".to_string());
    }
    if phase.base_dataset_view.source != base.base_dataset_view.source {
        mismatches
            .push("phase dataset source/contract does not match the source sweep view".to_string());
    }
    if phase.base_strategy != base.base_strategy {
        mismatches.push("phase base strategy does not match source sweep".to_string());
    }
    if phase.parameters != base.parameters {
        mismatches.push("phase parameter grid does not match source sweep".to_string());
    }
    if phase.constraints != base.constraints {
        mismatches.push("phase constraints do not match source sweep".to_string());
    }
    if phase.bar_type != base.bar_type {
        mismatches.push("phase bar type does not match source sweep".to_string());
    }
    if phase.candle_mode != base.candle_mode {
        mismatches.push("phase candle mode does not match source sweep".to_string());
    }
    if phase.engine_mode != base.engine_mode {
        mismatches.push("phase engine mode does not match source sweep".to_string());
    }
    if phase.fill_model != base.fill_model {
        mismatches.push("phase fill model does not match source sweep".to_string());
    }
    if phase.latency != base.latency {
        mismatches.push("phase latency configuration does not match source sweep".to_string());
    }
    if phase.bar_protection_policy != base.bar_protection_policy {
        mismatches.push("phase bar protection policy does not match source sweep".to_string());
    }
    if phase.primary_fee_schedule != base.primary_fee_schedule
        || phase.fee_scenarios != base.fee_scenarios
    {
        mismatches.push("phase fee schedules do not match source sweep".to_string());
    }
    if phase.initial_capital.to_bits() != base.initial_capital.to_bits() {
        mismatches.push("phase initial capital does not match source sweep".to_string());
    }
    if phase.margin != base.margin {
        mismatches.push("phase margin configuration does not match source sweep".to_string());
    }
    mismatches
}

fn sweep_mismatches(sweep: &Value, child: &ReplaySweepChildSpec) -> Vec<String> {
    let mut mismatches = Vec::new();
    check_value(
        sweep,
        "parent_sweep_id",
        &child.parent_sweep_id,
        &mut mismatches,
    );
    check_value(
        sweep,
        "parameter_values",
        &child.parameter_values,
        &mut mismatches,
    );
    check_value(sweep, "overrides", &child.overrides, &mut mismatches);
    check_value(
        sweep,
        "resolved_strategy",
        &child.resolved_strategy,
        &mut mismatches,
    );
    check_value(
        sweep,
        "dataset_view",
        &child.base_dataset_view,
        &mut mismatches,
    );
    check_value(sweep, "bar_type", &child.bar_type, &mut mismatches);
    check_value(sweep, "candle_mode", &child.candle_mode, &mut mismatches);
    check_value(sweep, "engine_mode", &child.engine_mode, &mut mismatches);
    check_value(
        sweep,
        "evaluator_mode",
        &child.evaluator_mode,
        &mut mismatches,
    );
    check_value(sweep, "fill_model", &child.fill_model, &mut mismatches);
    check_value(sweep, "latency", &child.latency, &mut mismatches);
    check_value(
        sweep,
        "bar_protection_policy",
        &child.bar_protection_policy,
        &mut mismatches,
    );
    check_value(
        sweep,
        "primary_fee_schedule",
        &child.primary_fee_schedule,
        &mut mismatches,
    );
    check_value(
        sweep,
        "fee_scenarios",
        &child.fee_scenarios,
        &mut mismatches,
    );
    check_value(
        sweep,
        "initial_capital",
        &child.initial_capital,
        &mut mismatches,
    );
    check_value(sweep, "margin", &child.margin, &mut mismatches);
    mismatches
}

fn check_value<T: Serialize>(
    sweep: &Value,
    field: &str,
    expected: &T,
    mismatches: &mut Vec<String>,
) {
    let Ok(expected) = serde_json::to_value(expected) else {
        mismatches.push(format!("could not serialize expected {field}"));
        return;
    };
    if sweep.get(field) != Some(&expected) {
        mismatches.push(format!(
            "sweep metadata field {field} does not match phase plan"
        ));
    }
}

fn result_metadata_mismatches(
    result: &ReplayResultDocument,
    phase: &ReplaySweepSpec,
    child: &ReplaySweepChildSpec,
) -> Vec<String> {
    let metadata = &result.metadata;
    let view = &phase.base_dataset_view;
    let mut mismatches = Vec::new();
    if !metadata
        .contract_name
        .eq_ignore_ascii_case(&view.source.contract)
    {
        mismatches.push(format!(
            "result contract {} does not match {}",
            metadata.contract_name, view.source.contract
        ));
    }
    if metadata.dataset_view.as_deref() != Some(view.id.as_str()) {
        mismatches.push("result dataset view does not match phase view".to_string());
    }
    if metadata.evaluation_start_utc != Some(view.evaluation_start)
        || metadata.evaluation_end_utc != Some(view.evaluation_end)
    {
        mismatches.push("result evaluation range does not match phase view".to_string());
    }
    if metadata.bar_type != child.bar_type {
        mismatches.push("result bar type does not match phase plan".to_string());
    }
    if metadata.candle_mode != child.candle_mode {
        mismatches.push("result candle mode does not match phase plan".to_string());
    }
    if metadata.fill_model != child.fill_model {
        mismatches.push("result fill model does not match phase plan".to_string());
    }
    if metadata.latency_model != child.latency.model
        || metadata.fixed_latency_ms != child.latency.fixed_latency_ms
    {
        mismatches.push("result latency does not match phase plan".to_string());
    }
    if metadata.initial_capital.to_bits() != child.initial_capital.to_bits() {
        mismatches.push("result initial capital does not match phase plan".to_string());
    }
    if metadata.strategy != child.resolved_strategy {
        mismatches.push("result resolved strategy does not match phase plan".to_string());
    }
    mismatches
}

#[derive(Debug, Clone)]
struct ScenarioMetrics {
    gross_pnl: f64,
    net_pnl: f64,
    fees: f64,
    ending_equity: f64,
    return_on_initial_capital_pct: Option<f64>,
    required_starting_capital: Option<f64>,
    max_drawdown: f64,
    max_drawdown_pct: Option<f64>,
    profit_factor: Option<f64>,
}

fn metrics_from_result(
    phase: ReplayWalkForwardPhase,
    result: &ReplayResultDocument,
    result_path: PathBuf,
    run_id: String,
    options: &ReplayWalkForwardEvaluationOptions,
) -> Result<ReplayWalkForwardCandidatePhase> {
    let scenario = select_scenario(result, options.fee_scenario.as_deref()).with_context(|| {
        options
            .fee_scenario
            .as_deref()
            .unwrap_or("active")
            .to_string()
    })?;
    let required_return = scenario
        .required_starting_capital
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|capital| scenario.net_pnl / capital * 100.0);
    let average_trade = (result.summary.closed_trade_count > 0)
        .then_some(scenario.net_pnl / result.summary.closed_trade_count as f64);
    let max_drawdown_pct = scenario.max_drawdown_pct.or_else(|| {
        (result.summary.initial_capital > 0.0)
            .then_some(scenario.max_drawdown / result.summary.initial_capital * 100.0)
    });
    let metric_value = metric_value(
        options.metric,
        &scenario,
        result.summary.win_rate_pct,
        result.summary.closed_trade_count,
        average_trade,
        required_return,
        max_drawdown_pct,
        result.summary.average_giveback,
        result.summary.largest_giveback,
        result.summary.average_mfe_capture_ratio,
    );
    Ok(ReplayWalkForwardCandidatePhase {
        phase,
        status: ReplayWalkForwardPhaseStatus::Completed,
        run_id: Some(run_id),
        result_path: Some(result_path),
        metric_value,
        gross_pnl: Some(scenario.gross_pnl),
        net_pnl: Some(scenario.net_pnl),
        fees: Some(scenario.fees),
        ending_equity: Some(scenario.ending_equity),
        return_on_initial_capital_pct: scenario.return_on_initial_capital_pct,
        required_starting_capital: scenario.required_starting_capital,
        return_on_required_account_size_pct: required_return,
        max_drawdown: Some(scenario.max_drawdown),
        max_drawdown_pct,
        profit_factor: scenario.profit_factor,
        closed_trade_count: Some(result.summary.closed_trade_count),
        win_rate_pct: result.summary.win_rate_pct,
        error: None,
    })
}

fn select_scenario(
    result: &ReplayResultDocument,
    requested: Option<&str>,
) -> Option<ScenarioMetrics> {
    let requested = requested.map(str::trim).filter(|value| !value.is_empty());
    if requested.is_none() || requested.is_some_and(|value| value.eq_ignore_ascii_case("active")) {
        let summary = &result.summary;
        return Some(ScenarioMetrics {
            gross_pnl: summary.gross_pnl,
            net_pnl: summary.net_pnl,
            fees: summary.fees,
            ending_equity: summary.ending_equity,
            return_on_initial_capital_pct: summary.return_on_initial_capital_pct,
            required_starting_capital: summary.required_starting_capital,
            max_drawdown: summary.max_drawdown,
            max_drawdown_pct: summary.max_drawdown_pct,
            profit_factor: summary.profit_factor,
        });
    }
    let requested = requested?;
    result
        .fee_scenarios
        .iter()
        .find(|scenario| scenario.schedule.name.eq_ignore_ascii_case(requested))
        .map(|scenario| scenario_metrics_from_fee_scenario(scenario, result))
}

fn scenario_metrics_from_fee_scenario(
    scenario: &ReplayFeeScenario,
    result: &ReplayResultDocument,
) -> ScenarioMetrics {
    let required_starting_capital = result
        .margin_analysis
        .as_ref()
        .filter(|analysis| {
            analysis
                .fee_scenario
                .eq_ignore_ascii_case(&scenario.schedule.name)
        })
        .map(|analysis| analysis.required_starting_capital);
    ScenarioMetrics {
        gross_pnl: scenario.gross_pnl,
        net_pnl: scenario.net_pnl,
        fees: scenario.fees,
        ending_equity: scenario.ending_equity,
        return_on_initial_capital_pct: scenario.return_on_initial_capital_pct,
        required_starting_capital,
        max_drawdown: scenario.max_drawdown,
        max_drawdown_pct: scenario.max_drawdown_pct,
        profit_factor: scenario.profit_factor,
    }
}

#[allow(clippy::too_many_arguments)]
fn metric_value(
    metric: ReplaySweepRankingMetric,
    scenario: &ScenarioMetrics,
    win_rate_pct: Option<f64>,
    closed_trade_count: usize,
    average_trade: Option<f64>,
    return_on_required_account_size_pct: Option<f64>,
    max_drawdown_pct: Option<f64>,
    average_giveback: Option<f64>,
    largest_giveback: Option<f64>,
    average_mfe_capture_ratio: Option<f64>,
) -> Option<f64> {
    let quality = scenario
        .return_on_initial_capital_pct
        .filter(|value| value.is_finite())
        .unwrap_or(scenario.net_pnl)
        - max_drawdown_pct
            .filter(|value| value.is_finite())
            .unwrap_or_default()
            .abs()
            * 0.75;
    let value = match metric {
        ReplaySweepRankingMetric::Robustness => quality,
        ReplaySweepRankingMetric::NetPnl => scenario.net_pnl,
        ReplaySweepRankingMetric::GrossPnl => scenario.gross_pnl,
        ReplaySweepRankingMetric::Fees => -scenario.fees,
        ReplaySweepRankingMetric::MaxDrawdown => -scenario.max_drawdown,
        ReplaySweepRankingMetric::RequiredAccountSize => -scenario.required_starting_capital?,
        ReplaySweepRankingMetric::ReturnOnRequiredAccountSize => {
            return_on_required_account_size_pct?
        }
        ReplaySweepRankingMetric::ProfitFactor => scenario.profit_factor?,
        ReplaySweepRankingMetric::WinRate => win_rate_pct?,
        ReplaySweepRankingMetric::TradeCount => closed_trade_count as f64,
        ReplaySweepRankingMetric::AverageTrade => average_trade?,
        ReplaySweepRankingMetric::AverageGiveback => -average_giveback?,
        ReplaySweepRankingMetric::LargestGiveback => -largest_giveback?,
        ReplaySweepRankingMetric::MfeCapture => average_mfe_capture_ratio?,
    };
    value.is_finite().then_some(value)
}

fn resolve_path(anchor: &Path, raw: &Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        let anchored = anchor.join(raw);
        // Current sweep artifacts are resolved relative to the phase/spec
        // directory. Keep a compatibility fallback for older summaries that
        // persisted a path relative to the process working directory.
        if anchored.exists() || !raw.exists() {
            anchored
        } else {
            raw.to_path_buf()
        }
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize walk-forward evaluation")?;
    write_bytes_atomic(path, &bytes)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create evaluation output {}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("report");
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    fs::write(&temporary, bytes)
        .with_context(|| format!("write temporary evaluation output {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("replace evaluation output {}", path.display()))
}

fn evaluation_csv(document: &ReplayWalkForwardEvaluationDocument) -> Vec<u8> {
    let mut output = String::from(
        "fold_index,parameter_values,overrides,selected,selection_phase,selection_metric_value,train_status,train_metric,train_net_pnl,train_gross_pnl,train_fees,train_max_drawdown,validation_status,validation_metric,validation_net_pnl,validation_gross_pnl,validation_fees,validation_max_drawdown,test_status,test_metric,test_net_pnl,test_gross_pnl,test_fees,test_max_drawdown\n",
    );
    for fold in &document.folds {
        for candidate in &fold.candidates {
            let fields = [
                candidate.fold_index.to_string(),
                serde_json::to_string(&candidate.parameter_values).unwrap_or_default(),
                serde_json::to_string(&candidate.overrides).unwrap_or_default(),
                candidate.selected.to_string(),
                candidate
                    .selection_phase
                    .map(phase_label)
                    .unwrap_or_default()
                    .to_string(),
                optional_f64(candidate.selection_metric_value),
                phase_status(candidate.train.as_ref()),
                optional_phase_f64(candidate.train.as_ref(), |phase| phase.metric_value),
                optional_phase_f64(candidate.train.as_ref(), |phase| phase.net_pnl),
                optional_phase_f64(candidate.train.as_ref(), |phase| phase.gross_pnl),
                optional_phase_f64(candidate.train.as_ref(), |phase| phase.fees),
                optional_phase_f64(candidate.train.as_ref(), |phase| phase.max_drawdown),
                phase_status(candidate.validation.as_ref()),
                optional_phase_f64(candidate.validation.as_ref(), |phase| phase.metric_value),
                optional_phase_f64(candidate.validation.as_ref(), |phase| phase.net_pnl),
                optional_phase_f64(candidate.validation.as_ref(), |phase| phase.gross_pnl),
                optional_phase_f64(candidate.validation.as_ref(), |phase| phase.fees),
                optional_phase_f64(candidate.validation.as_ref(), |phase| phase.max_drawdown),
                phase_status(candidate.test.as_ref()),
                optional_phase_f64(candidate.test.as_ref(), |phase| phase.metric_value),
                optional_phase_f64(candidate.test.as_ref(), |phase| phase.net_pnl),
                optional_phase_f64(candidate.test.as_ref(), |phase| phase.gross_pnl),
                optional_phase_f64(candidate.test.as_ref(), |phase| phase.fees),
                optional_phase_f64(candidate.test.as_ref(), |phase| phase.max_drawdown),
            ];
            output.push_str(
                &fields
                    .iter()
                    .map(|value| csv_field(value))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            output.push('\n');
        }
    }
    output.into_bytes()
}

fn phase_label(phase: ReplayWalkForwardPhase) -> &'static str {
    match phase {
        ReplayWalkForwardPhase::Train => "train",
        ReplayWalkForwardPhase::Validation => "validation",
        ReplayWalkForwardPhase::Test => "test",
    }
}

fn phase_status(phase: Option<&ReplayWalkForwardCandidatePhase>) -> String {
    phase
        .map(|phase| match phase.status {
            ReplayWalkForwardPhaseStatus::Completed => "completed",
            ReplayWalkForwardPhaseStatus::Partial => "partial",
            ReplayWalkForwardPhaseStatus::Failed => "failed",
            ReplayWalkForwardPhaseStatus::Missing => "missing",
            ReplayWalkForwardPhaseStatus::Invalid => "invalid",
        })
        .unwrap_or("missing")
        .to_string()
}

fn optional_phase_f64(
    phase: Option<&ReplayWalkForwardCandidatePhase>,
    value: impl Fn(&ReplayWalkForwardCandidatePhase) -> Option<f64>,
) -> String {
    optional_f64(phase.and_then(value))
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{BarType, BrokerKind, CandleMode};
    use crate::config::TradingEnvironment;
    use crate::replay_cache::{
        ReplayDatasetSessionPreset, ReplayDatasetSourceRef, ReplayDatasetView,
        ReplayDatasetWarmupPolicy,
    };
    use crate::strategy::ExecutionStrategyConfig;
    use crate::tradovate::replay::sweep::ReplaySweepOutputFormat;
    use crate::tradovate::replay::walk_forward::{
        ReplayWalkForwardOptions, plan_replay_walk_forward,
    };

    fn fixture_spec() -> ReplaySweepSpec {
        let start = DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .expect("fixture start")
            .with_timezone(&Utc);
        let mut spec = ReplaySweepSpec::default();
        spec.sweep_id = "fixture-sweep".to_string();
        spec.name = "Fixture sweep".to_string();
        spec.base_strategy = ExecutionStrategyConfig::default();
        spec.base_dataset_view = ReplayDatasetView {
            view_version: crate::replay_cache::REPLAY_DATASET_VIEW_VERSION,
            id: "fixture-view".to_string(),
            source: ReplayDatasetSourceRef {
                manifest_id: "fixture/manifest.json".to_string(),
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
            },
            evaluation_start: start,
            evaluation_end: start + chrono::Duration::hours(10),
            input_timezone: "UTC".to_string(),
            session_preset: ReplayDatasetSessionPreset::FullSource,
            warmup: ReplayDatasetWarmupPolicy {
                duration_seconds: 60,
                ..ReplayDatasetWarmupPolicy::default()
            },
        };
        spec.bar_type = BarType::minute(1);
        spec.candle_mode = CandleMode::Standard;
        spec.output_formats = vec![ReplaySweepOutputFormat::JsonSummary];
        spec.max_runs = 1;
        spec.parallelism = 1;
        spec.initial_capital = 50_000.0;
        spec
    }

    fn candidate(
        index: usize,
        train: Option<f64>,
        validation: Option<f64>,
        test: Option<f64>,
    ) -> ReplayWalkForwardCandidateReport {
        let parameter_values = BTreeMap::from([(String::from("x"), Value::from(index as i64))]);
        let overrides = BTreeMap::new();
        let phase = |phase: ReplayWalkForwardPhase, value: Option<f64>| {
            value.map(|metric_value| ReplayWalkForwardCandidatePhase {
                phase,
                status: ReplayWalkForwardPhaseStatus::Completed,
                run_id: None,
                result_path: None,
                metric_value: Some(metric_value),
                gross_pnl: Some(metric_value),
                net_pnl: Some(metric_value),
                fees: Some(0.0),
                ending_equity: Some(100.0 + metric_value),
                return_on_initial_capital_pct: None,
                required_starting_capital: None,
                return_on_required_account_size_pct: None,
                max_drawdown: Some(0.0),
                max_drawdown_pct: Some(0.0),
                profit_factor: None,
                closed_trade_count: Some(1),
                win_rate_pct: Some(100.0),
                error: None,
            })
        };
        ReplayWalkForwardCandidateReport {
            fold_index: 0,
            parameter_values,
            overrides,
            selected: false,
            selection_phase: None,
            selection_metric_value: None,
            train: phase(ReplayWalkForwardPhase::Train, train),
            validation: phase(ReplayWalkForwardPhase::Validation, validation),
            test: phase(ReplayWalkForwardPhase::Test, test),
        }
    }

    #[test]
    fn candidate_identity_includes_parameter_values_and_overrides() {
        let values = BTreeMap::from([(String::from("x"), Value::from(1))]);
        let empty = BTreeMap::new();
        let overrides = BTreeMap::from([(String::from("x"), Value::from(1))]);
        assert_ne!(
            CandidateKey::new(&values, &empty).expect("key"),
            CandidateKey::new(&values, &overrides).expect("key")
        );
    }

    #[test]
    fn train_then_validation_never_uses_test_for_selection() {
        let mut candidates = vec![
            candidate(1, Some(10.0), Some(5.0), Some(1_000.0)),
            candidate(2, Some(9.0), Some(7.0), Some(-1_000.0)),
        ];
        let options = ReplayWalkForwardEvaluationOptions {
            metric: ReplaySweepRankingMetric::NetPnl,
            fee_scenario: None,
            selection_policy: ReplayWalkForwardSelectionPolicy::TrainThenValidation,
        };
        let mut warnings = Vec::new();
        let selected = select_candidate(&mut candidates, &options, true, &mut warnings)
            .expect("selected candidate");
        assert_eq!(selected, 1);
        assert!(warnings.is_empty());
    }

    #[test]
    fn csv_preserves_train_validation_test_labels() {
        let document = ReplayWalkForwardEvaluationDocument {
            schema_version: REPLAY_WALK_FORWARD_EVALUATION_SCHEMA_VERSION,
            plan_path: PathBuf::from("plan.json"),
            generated_at_utc: Utc::now(),
            options: ReplayWalkForwardEvaluationOptions::default(),
            folds: vec![ReplayWalkForwardFoldReport {
                fold_index: 0,
                phases: Vec::new(),
                candidates: vec![candidate(1, Some(1.0), Some(2.0), Some(3.0))],
                selected_parameter_values: None,
                selected_overrides: None,
                warnings: Vec::new(),
            }],
            warnings: Vec::new(),
        };
        let csv = String::from_utf8(evaluation_csv(&document)).expect("csv");
        assert!(csv.contains("train_status"));
        assert!(csv.contains("validation_status"));
        assert!(csv.contains("test_status"));
        assert_eq!(csv.lines().count(), 2);
    }

    #[test]
    fn persisted_plan_reports_missing_phase_artifacts_and_writes_reports() {
        let root = std::env::temp_dir().join(format!(
            "trader-walk-forward-eval-fixture-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&root).expect("fixture root");
        let source_path = root.join("source-sweep.json");
        let plan_path = root.join("walk-forward.json");
        fixture_spec().save(&source_path).expect("source spec");
        let plan = plan_replay_walk_forward(
            &source_path,
            ReplayWalkForwardOptions {
                output_dir: Some(root.join("phases")),
                ..ReplayWalkForwardOptions::default()
            },
            &plan_path,
        )
        .expect("walk-forward plan");
        let json_path = root.join("evaluation.json");
        let csv_path = root.join("evaluation.csv");
        let document = evaluate_replay_walk_forward(
            &plan_path,
            ReplayWalkForwardEvaluationOptions::default(),
            Some(&json_path),
            Some(&csv_path),
        )
        .expect("evaluate fixture");
        assert_eq!(document.folds.len(), plan.folds.len());
        assert!(
            document.folds[0]
                .phases
                .iter()
                .all(|phase| phase.status == ReplayWalkForwardPhaseStatus::Missing)
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("missing"))
        );
        assert!(json_path.is_file());
        assert!(csv_path.is_file());
        let persisted: ReplayWalkForwardEvaluationDocument =
            serde_json::from_slice(&fs::read(&json_path).expect("evaluation JSON"))
                .expect("parse evaluation JSON");
        assert_eq!(
            persisted.schema_version,
            REPLAY_WALK_FORWARD_EVALUATION_SCHEMA_VERSION
        );
    }
}
