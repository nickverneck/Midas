//! Read-only ranking and robustness analytics for completed replay sweeps.
//!
//! Ranking deliberately consumes saved child results and the parent summary;
//! it never opens a market-data connection or replays a child.  This keeps
//! changing a ranking metric cheap and makes fee-scenario comparisons safe.

use super::results::{ReplayFeeScenario, ReplayResultDocument};
use super::sweep::ReplaySweepPlan;
use super::sweep_runner::ReplaySweepSummaryDocument;
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const REPLAY_SWEEP_RANKING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplaySweepRankingMetric {
    Robustness,
    NetPnl,
    GrossPnl,
    Fees,
    MaxDrawdown,
    RequiredAccountSize,
    ReturnOnRequiredAccountSize,
    ProfitFactor,
    WinRate,
    TradeCount,
    AverageTrade,
    AverageGiveback,
    LargestGiveback,
    MfeCapture,
}

impl ReplaySweepRankingMetric {
    pub(crate) fn all() -> &'static [Self; 14] {
        &ALL_RANKING_METRICS
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Robustness => "robustness",
            Self::NetPnl => "net_pnl",
            Self::GrossPnl => "gross_pnl",
            Self::Fees => "fees",
            Self::MaxDrawdown => "max_drawdown",
            Self::RequiredAccountSize => "required_account_size",
            Self::ReturnOnRequiredAccountSize => "return_on_required_account_size",
            Self::ProfitFactor => "profit_factor",
            Self::WinRate => "win_rate",
            Self::TradeCount => "trade_count",
            Self::AverageTrade => "average_trade",
            Self::AverageGiveback => "average_giveback",
            Self::LargestGiveback => "largest_giveback",
            Self::MfeCapture => "mfe_capture",
        }
    }

    pub(crate) fn parse(raw: &str) -> Result<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "robustness" | "robust" => Ok(Self::Robustness),
            "net_pnl" | "net" => Ok(Self::NetPnl),
            "gross_pnl" | "gross" => Ok(Self::GrossPnl),
            "fees" | "fee" => Ok(Self::Fees),
            "max_drawdown" | "drawdown" => Ok(Self::MaxDrawdown),
            "required_account_size" | "required_capital" | "account_size" => {
                Ok(Self::RequiredAccountSize)
            }
            "return_on_required_account_size" | "return_required" | "required_return" => {
                Ok(Self::ReturnOnRequiredAccountSize)
            }
            "profit_factor" | "pf" => Ok(Self::ProfitFactor),
            "win_rate" | "win_rate_pct" => Ok(Self::WinRate),
            "trade_count" | "trades" => Ok(Self::TradeCount),
            "average_trade" | "avg_trade" => Ok(Self::AverageTrade),
            "average_giveback" | "avg_giveback" => Ok(Self::AverageGiveback),
            "largest_giveback" | "max_giveback" => Ok(Self::LargestGiveback),
            "mfe_capture" | "mfe_capture_ratio" => Ok(Self::MfeCapture),
            _ => bail!(
                "unknown sweep ranking metric `{raw}`; choose one of: {}",
                ALL_RANKING_METRICS
                    .iter()
                    .map(|metric| metric.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// One saved ranking artifact discovered for the offline Analytics view.
#[derive(Debug, Clone)]
pub(crate) struct ReplaySweepRankingEntry {
    pub(crate) path: PathBuf,
    pub(crate) document: ReplaySweepRankingDocument,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReplaySweepRankingLibrarySnapshot {
    pub(crate) entries: Vec<ReplaySweepRankingEntry>,
    pub(crate) warnings: Vec<String>,
}

/// Scan for the conventional `sweep-ranking.json` artifacts without opening
/// the replay cache. The depth cap and directory exclusions keep refreshing
/// the TUI cheap even when the repository contains build outputs.
pub(crate) fn load_replay_sweep_ranking_entries(root: &Path) -> ReplaySweepRankingLibrarySnapshot {
    let mut snapshot = ReplaySweepRankingLibrarySnapshot::default();
    let mut paths = Vec::new();
    collect_ranking_paths(root, 0, &mut paths);
    paths.sort();
    paths.dedup();

    for path in paths {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                snapshot.warnings.push(format!(
                    "Could not read sweep ranking {}: {error}",
                    path.display()
                ));
                continue;
            }
        };
        match serde_json::from_slice::<ReplaySweepRankingDocument>(&bytes) {
            Ok(document) => snapshot
                .entries
                .push(ReplaySweepRankingEntry { path, document }),
            Err(error) => snapshot.warnings.push(format!(
                "Could not parse sweep ranking {}: {error}",
                path.display()
            )),
        }
    }
    snapshot
}

fn collect_ranking_paths(root: &Path, depth: usize, paths: &mut Vec<PathBuf>) {
    if depth > 5 || !root.exists() {
        return;
    }
    if root.is_file() {
        if root
            .file_name()
            .is_some_and(|name| name == "sweep-ranking.json")
        {
            paths.push(root.to_path_buf());
        }
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path.file_name().is_some_and(|name| {
                matches!(name.to_str(), Some(".git" | "target" | "node_modules"))
            })
        {
            continue;
        }
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|name| name == "sweep-ranking.json")
        {
            paths.push(path);
        } else if path.is_dir() {
            collect_ranking_paths(&path, depth + 1, paths);
        }
    }
}

const ALL_RANKING_METRICS: [ReplaySweepRankingMetric; 14] = [
    ReplaySweepRankingMetric::Robustness,
    ReplaySweepRankingMetric::NetPnl,
    ReplaySweepRankingMetric::GrossPnl,
    ReplaySweepRankingMetric::Fees,
    ReplaySweepRankingMetric::MaxDrawdown,
    ReplaySweepRankingMetric::RequiredAccountSize,
    ReplaySweepRankingMetric::ReturnOnRequiredAccountSize,
    ReplaySweepRankingMetric::ProfitFactor,
    ReplaySweepRankingMetric::WinRate,
    ReplaySweepRankingMetric::TradeCount,
    ReplaySweepRankingMetric::AverageTrade,
    ReplaySweepRankingMetric::AverageGiveback,
    ReplaySweepRankingMetric::LargestGiveback,
    ReplaySweepRankingMetric::MfeCapture,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepRankingOptions {
    pub(crate) metric: ReplaySweepRankingMetric,
    /// `None` or `active` ranks each result's active (normally primary)
    /// scenario. A named scenario ranks the fee overlay without replaying.
    pub(crate) fee_scenario: Option<String>,
    pub(crate) limit: usize,
    pub(crate) min_closed_trades: usize,
    pub(crate) max_drawdown_pct: Option<f64>,
}

impl Default for ReplaySweepRankingOptions {
    fn default() -> Self {
        Self {
            metric: ReplaySweepRankingMetric::Robustness,
            fee_scenario: None,
            limit: 20,
            min_closed_trades: 0,
            max_drawdown_pct: None,
        }
    }
}

impl ReplaySweepRankingOptions {
    fn validate(&self) -> Result<()> {
        if self.limit == 0 {
            bail!("ranking limit must be greater than zero");
        }
        if let Some(value) = self.max_drawdown_pct {
            if !value.is_finite() || value < 0.0 {
                bail!("max_drawdown_pct must be finite and non-negative");
            }
        }
        if self
            .fee_scenario
            .as_deref()
            .is_some_and(|scenario| scenario.trim().is_empty())
        {
            bail!("fee_scenario cannot be empty when supplied");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepRankingRow {
    pub(crate) rank: usize,
    pub(crate) run_id: String,
    pub(crate) run_index: usize,
    pub(crate) fee_scenario: String,
    pub(crate) parameter_values: BTreeMap<String, Value>,
    pub(crate) overrides: BTreeMap<String, Value>,
    pub(crate) metric_value: Option<f64>,
    pub(crate) robustness_score: Option<f64>,
    pub(crate) neighborhood_count: usize,
    pub(crate) neighborhood_median_quality: Option<f64>,
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
    pub(crate) average_trade: Option<f64>,
    pub(crate) average_giveback: Option<f64>,
    pub(crate) median_giveback: Option<f64>,
    pub(crate) largest_giveback: Option<f64>,
    pub(crate) average_mfe_capture_ratio: Option<f64>,
    pub(crate) long_quantity: Option<f64>,
    pub(crate) short_quantity: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepRankingDocument {
    pub(crate) schema_version: u32,
    pub(crate) sweep_id: String,
    pub(crate) name: String,
    pub(crate) generated_at_utc: DateTime<Utc>,
    pub(crate) source_summary: PathBuf,
    pub(crate) metric: ReplaySweepRankingMetric,
    pub(crate) fee_scenario: String,
    pub(crate) options: ReplaySweepRankingOptions,
    pub(crate) total_completed_candidates: usize,
    pub(crate) filtered_candidates: usize,
    pub(crate) rows: Vec<ReplaySweepRankingRow>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct RankingCandidate {
    row: ReplaySweepRankingRow,
    quality: f64,
}

#[derive(Debug, Clone)]
struct ScenarioMetrics {
    name: String,
    gross_pnl: f64,
    net_pnl: f64,
    fees: f64,
    ending_equity: f64,
    return_on_initial_capital_pct: Option<f64>,
    max_drawdown: f64,
    max_drawdown_pct: Option<f64>,
    profit_factor: Option<f64>,
    required_starting_capital: Option<f64>,
}

pub(crate) fn rank_replay_sweep(
    summary_path: &Path,
    plan_path: Option<&Path>,
    options: ReplaySweepRankingOptions,
    output_path: Option<&Path>,
    csv_path: Option<&Path>,
) -> Result<ReplaySweepRankingDocument> {
    options.validate()?;
    let summary = load_summary(summary_path)?;
    let plan = load_plan(summary_path, plan_path)?;
    let fallback_parameters = plan
        .as_ref()
        .map(|plan| {
            plan.children
                .iter()
                .map(|child| {
                    (
                        child.run_index,
                        (child.parameter_values.clone(), child.overrides.clone()),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    let mut warnings = summary.warnings.clone();
    let mut candidates = Vec::new();
    for run in &summary.runs {
        if run.status != "completed" || run.result_path.is_none() {
            continue;
        }
        let result_path = resolve_result_path(summary_path, run.result_path.as_deref().unwrap());
        let result = match load_result(&result_path) {
            Ok(result) => result,
            Err(error) => {
                warnings.push(format!(
                    "could not load completed sweep result {}: {error}",
                    result_path.display()
                ));
                continue;
            }
        };
        let Some(scenario) = select_scenario(&result, options.fee_scenario.as_deref()) else {
            let requested = options
                .fee_scenario
                .as_deref()
                .unwrap_or("active")
                .to_string();
            warnings.push(format!(
                "sweep child {} does not contain fee scenario {}; omitted",
                run.run_id, requested
            ));
            continue;
        };
        let (parameter_values, overrides) = result
            .sweep
            .as_ref()
            .map(extract_sweep_parameters)
            .unwrap_or_else(|| {
                fallback_parameters
                    .get(&run.run_index)
                    .cloned()
                    .unwrap_or_default()
            });
        candidates.push(build_candidate(
            run,
            &result,
            scenario,
            parameter_values,
            overrides,
        ));
    }

    let total_completed_candidates = candidates.len();
    compute_neighborhoods(&mut candidates, plan.as_ref());

    let mut filtered = candidates
        .into_iter()
        .filter(|candidate| {
            let closed_trades = candidate.row.closed_trade_count.unwrap_or(0);
            if closed_trades < options.min_closed_trades {
                return false;
            }
            if let Some(max_drawdown_pct) = options.max_drawdown_pct {
                let Some(actual) = candidate.row.max_drawdown_pct else {
                    return false;
                };
                if actual > max_drawdown_pct {
                    return false;
                }
            }
            true
        })
        .collect::<Vec<_>>();
    for candidate in &mut filtered {
        candidate.row.metric_value = ranking_value(&candidate.row, options.metric);
    }
    filtered.retain(|candidate| candidate.row.metric_value.is_some_and(f64::is_finite));
    let filtered_candidates = filtered.len();
    filtered.sort_by(|left, right| {
        right
            .row
            .metric_value
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&left.row.metric_value.unwrap_or(f64::NEG_INFINITY))
            .then_with(|| left.row.run_index.cmp(&right.row.run_index))
    });
    filtered.truncate(options.limit);
    let rows = filtered
        .into_iter()
        .enumerate()
        .map(|(index, mut candidate)| {
            candidate.row.rank = index + 1;
            candidate.row
        })
        .collect::<Vec<_>>();
    let fee_scenario = options
        .fee_scenario
        .clone()
        .unwrap_or_else(|| "active".to_string());
    let document = ReplaySweepRankingDocument {
        schema_version: REPLAY_SWEEP_RANKING_SCHEMA_VERSION,
        sweep_id: summary.sweep_id,
        name: summary.name,
        generated_at_utc: Utc::now(),
        source_summary: summary_path.to_path_buf(),
        metric: options.metric,
        fee_scenario,
        options,
        total_completed_candidates,
        filtered_candidates,
        rows,
        warnings,
    };
    if let Some(path) = output_path {
        write_json(path, &document)?;
    }
    if let Some(path) = csv_path {
        write_bytes(path, &ranking_csv(&document))?;
    }
    Ok(document)
}

fn load_summary(path: &Path) -> Result<ReplaySweepSummaryDocument> {
    let bytes = fs::read(path).with_context(|| format!("read sweep summary {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse sweep summary {}", path.display()))
}

fn load_plan(summary_path: &Path, requested: Option<&Path>) -> Result<Option<ReplaySweepPlan>> {
    let derived = summary_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("sweep-plan.json");
    let Some(path) = requested.or_else(|| derived.is_file().then_some(derived.as_path())) else {
        return Ok(None);
    };
    if !path.is_file() {
        bail!("sweep plan {} does not exist", path.display());
    }
    let bytes = fs::read(path).with_context(|| format!("read sweep plan {}", path.display()))?;
    let plan = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse sweep plan {}", path.display()))?;
    Ok(Some(plan))
}

fn load_result(path: &Path) -> Result<ReplayResultDocument> {
    let bytes = fs::read(path).with_context(|| format!("read sweep child {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse sweep child {}", path.display()))
}

fn resolve_result_path(summary_path: &Path, raw: &Path) -> PathBuf {
    if raw.is_absolute() || raw.is_file() {
        return raw.to_path_buf();
    }
    let parent = summary_path.parent().unwrap_or_else(|| Path::new("."));
    let relative = parent.join(raw);
    if relative.is_file() {
        relative
    } else {
        raw.to_path_buf()
    }
}

fn select_scenario<'a>(
    result: &'a ReplayResultDocument,
    requested: Option<&str>,
) -> Option<ScenarioMetrics> {
    let requested = requested.map(str::trim).filter(|value| !value.is_empty());
    if requested.is_none() || requested.is_some_and(|value| value.eq_ignore_ascii_case("active")) {
        let summary = &result.summary;
        return Some(ScenarioMetrics {
            name: result.active_fee_scenario.clone(),
            gross_pnl: summary.gross_pnl,
            net_pnl: summary.net_pnl,
            fees: summary.fees,
            ending_equity: summary.ending_equity,
            return_on_initial_capital_pct: summary.return_on_initial_capital_pct,
            max_drawdown: summary.max_drawdown,
            max_drawdown_pct: summary.max_drawdown_pct,
            profit_factor: summary.profit_factor,
            required_starting_capital: summary.required_starting_capital,
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
        name: scenario.schedule.name.clone(),
        gross_pnl: scenario.gross_pnl,
        net_pnl: scenario.net_pnl,
        fees: scenario.fees,
        ending_equity: scenario.ending_equity,
        return_on_initial_capital_pct: scenario.return_on_initial_capital_pct,
        max_drawdown: scenario.max_drawdown,
        max_drawdown_pct: scenario.max_drawdown_pct,
        profit_factor: scenario.profit_factor,
        required_starting_capital,
    }
}

fn extract_sweep_parameters(value: &Value) -> (BTreeMap<String, Value>, BTreeMap<String, Value>) {
    let parameters = value
        .get("parameter_values")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    let overrides = value
        .get("overrides")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    (parameters, overrides)
}

fn build_candidate(
    run: &super::sweep_runner::ReplaySweepRunSummary,
    result: &ReplayResultDocument,
    scenario: ScenarioMetrics,
    parameter_values: BTreeMap<String, Value>,
    overrides: BTreeMap<String, Value>,
) -> RankingCandidate {
    let summary = &result.summary;
    let required_return = scenario
        .required_starting_capital
        .filter(|capital| capital.is_finite() && *capital > 0.0)
        .map(|capital| scenario.net_pnl / capital * 100.0);
    let average_trade = (summary.closed_trade_count > 0)
        .then_some(scenario.net_pnl / summary.closed_trade_count as f64);
    let (long_quantity, short_quantity) = result
        .trade_excursions
        .as_deref()
        .map(|excursions| {
            excursions.iter().fold((0.0, 0.0), |(long, short), trade| {
                if trade.side.eq_ignore_ascii_case("long") {
                    (long + trade.quantity.abs(), short)
                } else if trade.side.eq_ignore_ascii_case("short") {
                    (long, short + trade.quantity.abs())
                } else {
                    (long, short)
                }
            })
        })
        .unwrap_or((0.0, 0.0));
    let max_drawdown_pct = scenario.max_drawdown_pct.or_else(|| {
        (summary.initial_capital > 0.0)
            .then_some(scenario.max_drawdown / summary.initial_capital * 100.0)
    });
    let quality = quality_score(
        scenario.net_pnl,
        required_return.or(scenario.return_on_initial_capital_pct),
        max_drawdown_pct,
    );
    RankingCandidate {
        row: ReplaySweepRankingRow {
            rank: 0,
            run_id: run.run_id.clone(),
            run_index: run.run_index,
            fee_scenario: scenario.name,
            parameter_values,
            overrides,
            metric_value: None,
            robustness_score: None,
            neighborhood_count: 0,
            neighborhood_median_quality: None,
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
            closed_trade_count: Some(summary.closed_trade_count),
            win_rate_pct: summary.win_rate_pct,
            average_trade,
            average_giveback: summary.average_giveback,
            median_giveback: summary.median_giveback,
            largest_giveback: summary.largest_giveback,
            average_mfe_capture_ratio: summary.average_mfe_capture_ratio,
            long_quantity: Some(long_quantity),
            short_quantity: Some(short_quantity),
        },
        quality,
    }
}

fn quality_score(net_pnl: f64, return_pct: Option<f64>, drawdown_pct: Option<f64>) -> f64 {
    let return_pct = return_pct
        .filter(|value| value.is_finite())
        .unwrap_or(net_pnl);
    let drawdown_pct = drawdown_pct
        .filter(|value| value.is_finite())
        .unwrap_or_default()
        .abs();
    return_pct - drawdown_pct * 0.75
}

fn compute_neighborhoods(candidates: &mut [RankingCandidate], plan: Option<&ReplaySweepPlan>) {
    let positions = parameter_positions(plan, candidates);
    let qualities = candidates
        .iter()
        .map(|candidate| candidate.quality)
        .collect::<Vec<_>>();
    for index in 0..candidates.len() {
        let neighbors = (0..candidates.len())
            .filter(|other| {
                *other != index
                    && are_neighbors(
                        &candidates[index].row.parameter_values,
                        &candidates[*other].row.parameter_values,
                        &positions,
                    )
            })
            .map(|other| qualities[other])
            .collect::<Vec<_>>();
        if neighbors.is_empty() {
            candidates[index].row.robustness_score = Some(candidates[index].quality * 0.75);
        } else {
            let median = median(neighbors.clone());
            candidates[index].row.neighborhood_count = neighbors.len();
            candidates[index].row.neighborhood_median_quality = Some(median);
            candidates[index].row.robustness_score =
                Some(candidates[index].quality * 0.7 + median * 0.3);
        }
    }
}

fn parameter_positions(
    plan: Option<&ReplaySweepPlan>,
    candidates: &[RankingCandidate],
) -> BTreeMap<String, BTreeMap<String, usize>> {
    let mut values = BTreeMap::<String, Vec<Value>>::new();
    if let Some(plan) = plan {
        for parameter in &plan.spec.parameters {
            values.insert(parameter.path.clone(), parameter.values.clone());
        }
    }
    for candidate in candidates {
        for (path, value) in &candidate.row.parameter_values {
            let entry = values.entry(path.clone()).or_default();
            if !entry.iter().any(|existing| existing == value) {
                entry.push(value.clone());
            }
        }
    }
    values
        .into_iter()
        .map(|(path, values)| {
            (
                path,
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| (value_key(&value), index))
                    .collect(),
            )
        })
        .collect()
}

fn are_neighbors(
    left: &BTreeMap<String, Value>,
    right: &BTreeMap<String, Value>,
    positions: &BTreeMap<String, BTreeMap<String, usize>>,
) -> bool {
    if left.is_empty() || left.len() != right.len() {
        return false;
    }
    let mut differs = false;
    for (path, left_value) in left {
        let Some(right_value) = right.get(path) else {
            return false;
        };
        if left_value == right_value {
            continue;
        }
        differs = true;
        let distance = positions
            .get(path)
            .and_then(|path_positions| {
                Some(
                    path_positions
                        .get(&value_key(left_value))?
                        .abs_diff(*path_positions.get(&value_key(right_value))?),
                )
            })
            .unwrap_or(usize::MAX);
        if distance > 1 {
            return false;
        }
    }
    differs
}

fn value_key(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn ranking_value(row: &ReplaySweepRankingRow, metric: ReplaySweepRankingMetric) -> Option<f64> {
    match metric {
        ReplaySweepRankingMetric::Robustness => row.robustness_score,
        ReplaySweepRankingMetric::NetPnl => row.net_pnl,
        ReplaySweepRankingMetric::GrossPnl => row.gross_pnl,
        ReplaySweepRankingMetric::Fees => row.fees.map(|value| -value),
        ReplaySweepRankingMetric::MaxDrawdown => row.max_drawdown.map(|value| -value),
        ReplaySweepRankingMetric::RequiredAccountSize => {
            row.required_starting_capital.map(|value| -value)
        }
        ReplaySweepRankingMetric::ReturnOnRequiredAccountSize => {
            row.return_on_required_account_size_pct
        }
        ReplaySweepRankingMetric::ProfitFactor => row.profit_factor,
        ReplaySweepRankingMetric::WinRate => row.win_rate_pct,
        ReplaySweepRankingMetric::TradeCount => row.closed_trade_count.map(|value| value as f64),
        ReplaySweepRankingMetric::AverageTrade => row.average_trade,
        ReplaySweepRankingMetric::AverageGiveback => row.average_giveback.map(|value| -value),
        ReplaySweepRankingMetric::LargestGiveback => row.largest_giveback.map(|value| -value),
        ReplaySweepRankingMetric::MfeCapture => row.average_mfe_capture_ratio,
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize sweep ranking JSON")?;
    write_bytes(path, &bytes)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create ranking output {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("write ranking output {}", path.display()))
}

fn ranking_csv(document: &ReplaySweepRankingDocument) -> Vec<u8> {
    let mut output = String::from(
        "rank,run_id,run_index,fee_scenario,metric_value,robustness_score,neighborhood_count,neighborhood_median_quality,gross_pnl,net_pnl,fees,ending_equity,return_on_initial_capital_pct,required_starting_capital,return_on_required_account_size_pct,max_drawdown,max_drawdown_pct,profit_factor,closed_trade_count,win_rate_pct,average_trade,average_giveback,median_giveback,largest_giveback,average_mfe_capture_ratio,long_quantity,short_quantity,parameter_values,overrides\n",
    );
    for row in &document.rows {
        let fields = [
            row.rank.to_string(),
            row.run_id.clone(),
            row.run_index.to_string(),
            row.fee_scenario.clone(),
            optional_f64(row.metric_value),
            optional_f64(row.robustness_score),
            row.neighborhood_count.to_string(),
            optional_f64(row.neighborhood_median_quality),
            optional_f64(row.gross_pnl),
            optional_f64(row.net_pnl),
            optional_f64(row.fees),
            optional_f64(row.ending_equity),
            optional_f64(row.return_on_initial_capital_pct),
            optional_f64(row.required_starting_capital),
            optional_f64(row.return_on_required_account_size_pct),
            optional_f64(row.max_drawdown),
            optional_f64(row.max_drawdown_pct),
            optional_f64(row.profit_factor),
            optional_usize(row.closed_trade_count),
            optional_f64(row.win_rate_pct),
            optional_f64(row.average_trade),
            optional_f64(row.average_giveback),
            optional_f64(row.median_giveback),
            optional_f64(row.largest_giveback),
            optional_f64(row.average_mfe_capture_ratio),
            optional_f64(row.long_quantity),
            optional_f64(row.short_quantity),
            serde_json::to_string(&row.parameter_values).unwrap_or_default(),
            serde_json::to_string(&row.overrides).unwrap_or_default(),
        ];
        output.push_str(
            &fields
                .iter()
                .map(|field| csv_field(field))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push('\n');
    }
    output.into_bytes()
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_usize(value: Option<usize>) -> String {
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
    use crate::broker::{BarType, CandleMode, ReplayBarProtectionPolicy, ReplayFillModel};
    use crate::config::TradingEnvironment;
    use crate::replay_cache::{
        ReplayDatasetSessionPreset, ReplayDatasetSourceRef, ReplayDatasetView,
        ReplayDatasetWarmupPolicy,
    };
    use crate::strategy::ExecutionStrategyConfig;
    use crate::tradovate::replay::fees::ReplayFeeSchedule;
    use crate::tradovate::replay::results::{
        ReplayResultArtifacts, ReplayResultMetadata, ReplayResultSummary,
    };
    use crate::tradovate::replay::sweep_runner::{
        ReplaySweepRunSummary, ReplaySweepSummaryDocument,
    };
    use chrono::TimeZone;

    fn result_document(
        run_id: &str,
        _run_index: usize,
        parameters: BTreeMap<String, Value>,
        net_pnl: f64,
        max_drawdown: f64,
    ) -> ReplayResultDocument {
        let mut strategy = ExecutionStrategyConfig::default();
        strategy.native_strategy = crate::strategy::NativeStrategyKind::EmaCross;
        let view = ReplayDatasetView {
            view_version: crate::replay_cache::REPLAY_DATASET_VIEW_VERSION,
            id: "ranking-view".to_string(),
            source: ReplayDatasetSourceRef {
                manifest_id: "mes/manifest.json".to_string(),
                provider: crate::broker::BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
            },
            evaluation_start: Utc.with_ymd_and_hms(2026, 7, 1, 13, 30, 0).unwrap(),
            evaluation_end: Utc.with_ymd_and_hms(2026, 7, 1, 14, 30, 0).unwrap(),
            input_timezone: "UTC".to_string(),
            session_preset: ReplayDatasetSessionPreset::FullSource,
            warmup: ReplayDatasetWarmupPolicy::default(),
        };
        let now = Utc.with_ymd_and_hms(2026, 7, 1, 13, 30, 0).unwrap();
        ReplayResultDocument {
            schema_version: crate::tradovate::replay::results::REPLAY_RESULT_SCHEMA_VERSION,
            result_type: "replay_result".to_string(),
            run_id: run_id.to_string(),
            status: super::super::results::ReplayResultStatus::Completed,
            error: None,
            created_at_utc: now,
            started_at_utc: now,
            completed_at_utc: now,
            metadata: ReplayResultMetadata {
                app_version: "test".to_string(),
                package_version: "test".to_string(),
                git_commit: None,
                broker: "tradovate".to_string(),
                environment: TradingEnvironment::Sim,
                run_mode: "sweep_child".to_string(),
                strategy,
                signal_source: "server_bars".to_string(),
                fill_model: ReplayFillModel::RawBarOpen,
                fill_price_sources: Vec::new(),
                execution_precisions: Vec::new(),
                latency_model: crate::broker::ReplayLatencyModel::Fixed,
                fixed_latency_ms: 0,
                latency_seed: None,
                observed_latency_sample_count: 0,
                bar_protection_policy: ReplayBarProtectionPolicy::Conservative,
                fee_model: "primary".to_string(),
                initial_capital: 10_000.0,
                account_currency: "USD".to_string(),
                margin_model: "fixed".to_string(),
                account_id: 1,
                account_name: "test".to_string(),
                contract_id: 1,
                contract_name: "MESU6".to_string(),
                tick_size: Some(0.25),
                value_per_point: Some(5.0),
                bar_type: BarType::minute(1),
                candle_mode: CandleMode::Standard,
                timezone: "UTC".to_string(),
                replay_file_path: String::new(),
                replay_dom_file_path: None,
                replay_cache_dir: String::new(),
                post_exit_continuation_horizon_bars: 0,
                dataset_view: Some(view.id.clone()),
                evaluation_start_utc: Some(view.evaluation_start),
                evaluation_end_utc: Some(view.evaluation_end),
                warmup_start_utc: None,
                market_first_timestamp_ns: None,
                market_last_timestamp_ns: None,
                warmup_rows: 0,
                evaluation_rows_total: 60,
                evaluation_rows_processed: 60,
                signal_diagnostics_enabled: false,
                signal_diagnostic_count: 0,
            },
            summary: ReplayResultSummary {
                initial_capital: 10_000.0,
                ending_equity: 10_000.0 + net_pnl,
                gross_pnl: net_pnl + 4.0,
                net_pnl,
                fees: 4.0,
                return_on_initial_capital_pct: Some(net_pnl / 100.0),
                fill_count: 4,
                trade_count: 2,
                closed_trade_count: 2,
                win_count: 1,
                loss_count: 1,
                win_rate_pct: Some(50.0),
                profit_factor: Some(1.5),
                max_drawdown,
                max_drawdown_pct: Some(max_drawdown / 100.0),
                max_open_position: 1.0,
                precision: Vec::new(),
                exit_reason_counts: BTreeMap::new(),
                evaluation_rows_processed: 60,
                required_starting_capital: Some(2_000.0),
                peak_margin_requirement: Some(1_500.0),
                minimum_equity_buffer_over_margin: Some(8_000.0),
                initial_capital_sufficient: Some(true),
                average_mfe_pnl: Some(10.0),
                average_mae_pnl: Some(3.0),
                average_giveback: Some(2.0),
                median_giveback: Some(2.0),
                largest_giveback: Some(3.0),
                average_mfe_capture_ratio: Some(0.8),
                average_post_exit_favorable_pnl: None,
                largest_post_exit_favorable_pnl: None,
            },
            artifacts: ReplayResultArtifacts {
                result_json: "result.json".to_string(),
                trades_csv: "trades.csv".to_string(),
                fills_csv: "fills.csv".to_string(),
                equity_csv: "equity.csv".to_string(),
                fee_scenarios_csv: None,
                margin_csv: None,
                trade_excursions_csv: None,
                signals_csv: None,
            },
            active_fee_scenario: "primary".to_string(),
            fee_scenarios: Vec::new(),
            margin_analysis: None,
            trade_excursions: None,
            sweep: Some(serde_json::json!({
                "parameter_values": parameters,
                "overrides": {}
            })),
            ledger: Default::default(),
        }
    }

    #[test]
    fn metric_parser_accepts_aliases_and_rejects_unknown_values() {
        assert_eq!(
            ReplaySweepRankingMetric::parse("return-required").expect("metric"),
            ReplaySweepRankingMetric::ReturnOnRequiredAccountSize
        );
        assert!(ReplaySweepRankingMetric::parse("nope").is_err());
    }

    #[test]
    fn ranking_artifact_loader_discovers_nested_json() {
        let root = std::env::temp_dir().join(format!(
            "trader-sweep-ranking-library-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let ranking_path = root.join("runs/example/sweep-ranking.json");
        fs::create_dir_all(ranking_path.parent().expect("ranking parent")).expect("create root");
        let document = ReplaySweepRankingDocument {
            schema_version: REPLAY_SWEEP_RANKING_SCHEMA_VERSION,
            sweep_id: "example".to_string(),
            name: "Example sweep".to_string(),
            generated_at_utc: Utc::now(),
            source_summary: PathBuf::from("sweep-summary.json"),
            metric: ReplaySweepRankingMetric::Robustness,
            fee_scenario: "active".to_string(),
            options: ReplaySweepRankingOptions::default(),
            total_completed_candidates: 1,
            filtered_candidates: 1,
            rows: Vec::new(),
            warnings: Vec::new(),
        };
        fs::write(
            &ranking_path,
            serde_json::to_vec(&document).expect("ranking json"),
        )
        .expect("write ranking");

        let snapshot = load_replay_sweep_ranking_entries(&root);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].document.sweep_id, "example");
        assert!(snapshot.warnings.is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn quality_uses_return_and_drawdown_instead_of_pnl_alone() {
        let safer = quality_score(100.0, Some(5.0), Some(1.0));
        let riskier = quality_score(200.0, Some(5.0), Some(20.0));
        assert!(safer > riskier);
    }

    #[test]
    fn neighboring_grid_values_are_detected() {
        let left = BTreeMap::from([(String::from("fast"), Value::from(5))]);
        let right = BTreeMap::from([(String::from("fast"), Value::from(10))]);
        let positions = BTreeMap::from([(
            String::from("fast"),
            BTreeMap::from([
                (value_key(&Value::from(5)), 0),
                (value_key(&Value::from(10)), 1),
            ]),
        )]);
        assert!(are_neighbors(&left, &right, &positions));
    }

    #[test]
    fn named_fee_scenario_is_selected_without_replay() {
        let mut result = result_document("fee-scenario", 0, BTreeMap::new(), 100.0, 2.0);
        result.fee_scenarios = vec![ReplayFeeScenario {
            schedule: ReplayFeeSchedule {
                name: "high_fees".to_string(),
                commission_per_contract: 2.0,
                ..ReplayFeeSchedule::default()
            },
            fees: 16.0,
            gross_pnl: 104.0,
            net_pnl: 88.0,
            ending_equity: 10_088.0,
            return_on_initial_capital_pct: Some(0.88),
            max_drawdown: 4.0,
            max_drawdown_pct: Some(0.04),
            profit_factor: Some(1.25),
        }];

        let selected = select_scenario(&result, Some("HIGH_FEES")).expect("scenario");
        assert_eq!(selected.name, "high_fees");
        assert_eq!(selected.net_pnl, 88.0);
        assert_eq!(selected.fees, 16.0);
    }

    #[test]
    fn ranking_reads_saved_summary_and_orders_by_robustness() {
        let root = std::env::temp_dir().join(format!(
            "trader-sweep-ranking-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(root.join("runs/a")).expect("create a");
        fs::create_dir_all(root.join("runs/b")).expect("create b");
        let a = result_document(
            "sweep-000001",
            0,
            BTreeMap::from([(String::from("fast"), Value::from(5))]),
            100.0,
            1_000.0,
        );
        let b = result_document(
            "sweep-000002",
            1,
            BTreeMap::from([(String::from("fast"), Value::from(10))]),
            90.0,
            2.0,
        );
        fs::write(
            root.join("runs/a/result.json"),
            serde_json::to_vec(&a).expect("a json"),
        )
        .expect("write a");
        fs::write(
            root.join("runs/b/result.json"),
            serde_json::to_vec(&b).expect("b json"),
        )
        .expect("write b");
        let summary = ReplaySweepSummaryDocument {
            schema_version: 1,
            sweep_id: "sweep".to_string(),
            name: "Sweep".to_string(),
            created_at_utc: Utc::now(),
            run_count: 2,
            completed_count: 2,
            failed_count: 0,
            skipped_count: 0,
            warnings: Vec::new(),
            runs: vec![
                ReplaySweepRunSummary {
                    run_id: "sweep-000001".to_string(),
                    run_index: 0,
                    status: "completed".to_string(),
                    skipped: false,
                    result_path: Some(root.join("runs/a/result.json")),
                    error: None,
                    gross_pnl: Some(104.0),
                    net_pnl: Some(100.0),
                    fees: Some(4.0),
                    max_drawdown: Some(10.0),
                    trade_count: Some(2),
                    fill_count: Some(4),
                },
                ReplaySweepRunSummary {
                    run_id: "sweep-000002".to_string(),
                    run_index: 1,
                    status: "completed".to_string(),
                    skipped: false,
                    result_path: Some(root.join("runs/b/result.json")),
                    error: None,
                    gross_pnl: Some(94.0),
                    net_pnl: Some(90.0),
                    fees: Some(4.0),
                    max_drawdown: Some(2.0),
                    trade_count: Some(2),
                    fill_count: Some(4),
                },
            ],
            fee_scenarios: Vec::new(),
            resource_estimate: None,
        };
        let summary_path = root.join("sweep-summary.json");
        fs::write(
            &summary_path,
            serde_json::to_vec(&summary).expect("summary json"),
        )
        .expect("write summary");
        let document = rank_replay_sweep(
            &summary_path,
            None,
            ReplaySweepRankingOptions::default(),
            None,
            None,
        )
        .expect("rank");
        assert_eq!(document.rows.len(), 2);
        assert_eq!(document.rows[0].run_id, "sweep-000002");
        assert_eq!(document.rows[0].neighborhood_count, 1);

        let filtered = rank_replay_sweep(
            &summary_path,
            None,
            ReplaySweepRankingOptions {
                metric: ReplaySweepRankingMetric::NetPnl,
                max_drawdown_pct: Some(1.0),
                ..ReplaySweepRankingOptions::default()
            },
            None,
            None,
        )
        .expect("filter");
        assert_eq!(filtered.filtered_candidates, 1);
        assert_eq!(filtered.rows[0].run_id, "sweep-000002");
        let _ = fs::remove_dir_all(root);
    }
}
