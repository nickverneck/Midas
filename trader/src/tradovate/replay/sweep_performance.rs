//! Empirical performance reporting for headless replay sweeps.
//!
//! This module deliberately sits beside the sweep runner instead of inside
//! the CLI.  A performance run loads the same persisted sweep specification,
//! invokes [`super::sweep_runner::run_replay_sweep`], and records what the
//! process actually observed.  It does not open a second market-data path or
//! change the source specification.  By default an isolated output directory
//! is selected below the configured sweep directory; callers that want to
//! reuse/resume an existing output tree must opt into that path explicitly.

use super::sweep::{ReplaySweepResourceEstimate, ReplaySweepSpec};
use super::sweep_runner::{ReplaySweepRunSummary, run_replay_sweep};
use crate::config::AppConfig;
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(crate) const REPLAY_SWEEP_PERFORMANCE_SCHEMA_VERSION: u32 = 1;

/// Controls one empirical sweep-performance run.
///
/// `output_dir` is intentionally optional.  When omitted, the runner writes
/// to a fresh, isolated child directory next to the configured sweep output;
/// the persisted specification and its existing artifacts are never replaced.
/// Supplying `output_dir` explicitly is useful for a deliberate resume or for
/// comparing a report with an existing output tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplaySweepPerformanceOptions {
    /// Run at most this many deterministic prefix combinations. `None` runs
    /// the complete validated specification. The actual planned count is
    /// recorded because independent Cartesian axes can make the prefix less
    /// than the requested bound.
    pub(crate) sample_runs: Option<usize>,
    /// Explicit child-output directory. `None` selects an isolated directory.
    pub(crate) output_dir: Option<PathBuf>,
    /// Do not reuse completed child artifacts. This is the default for a
    /// meaningful wall-clock probe.
    pub(crate) no_resume: bool,
    pub(crate) allow_large: bool,
    pub(crate) override_guardrails: bool,
}

impl Default for ReplaySweepPerformanceOptions {
    fn default() -> Self {
        Self {
            sample_runs: None,
            output_dir: None,
            no_resume: true,
            allow_large: false,
            override_guardrails: false,
        }
    }
}

/// A compact process-level measurement. Fields are optional because `/proc`
/// is not available on every supported platform or inside every container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepProcessMetrics {
    pub(crate) peak_rss_bytes: Option<u64>,
    pub(crate) user_cpu_seconds: Option<f64>,
    pub(crate) system_cpu_seconds: Option<f64>,
}

impl Default for ReplaySweepProcessMetrics {
    fn default() -> Self {
        Self {
            peak_rss_bytes: None,
            user_cpu_seconds: None,
            system_cpu_seconds: None,
        }
    }
}

/// Artifact measurements for one child result. Missing artifacts remain
/// `None`, so a failed or interrupted child is distinguishable from a child
/// that produced an empty CSV.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPerformanceChildRow {
    pub(crate) run_id: String,
    pub(crate) run_index: usize,
    pub(crate) status: String,
    pub(crate) skipped: bool,
    pub(crate) result_path: Option<PathBuf>,
    pub(crate) result_bytes: Option<u64>,
    pub(crate) artifact_bytes: Option<u64>,
    pub(crate) result_json_rows: Option<u64>,
    pub(crate) trade_rows: Option<u64>,
    pub(crate) fill_rows: Option<u64>,
    pub(crate) equity_rows: Option<u64>,
}

/// Aggregate observations made after the headless runner returned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPerformanceObserved {
    pub(crate) wall_clock_ms: u64,
    pub(crate) completed_count: usize,
    pub(crate) failed_count: usize,
    pub(crate) skipped_count: usize,
    /// Total bytes currently present in the selected output tree.
    pub(crate) output_bytes: u64,
    /// Increase in output-tree size during this invocation. This is useful
    /// when an explicitly selected output directory already contains runs.
    pub(crate) output_bytes_written: u64,
    pub(crate) child_rows: u64,
    pub(crate) result_json_rows: u64,
    pub(crate) trade_rows: u64,
    pub(crate) fill_rows: u64,
    pub(crate) equity_rows: u64,
    pub(crate) process: ReplaySweepProcessMetrics,
}

/// Ratios and throughput derived from observed and executed estimates. A
/// ratio is omitted when the estimate is unknown or zero rather than emitting
/// an invalid/infinite number.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPerformanceComparison {
    pub(crate) runtime_ratio_observed_to_estimated: Option<f64>,
    pub(crate) output_ratio_observed_to_estimated: Option<f64>,
    pub(crate) output_written_ratio_to_estimated: Option<f64>,
    pub(crate) completed_ratio_to_planned: Option<f64>,
    pub(crate) result_rows_per_second: Option<f64>,
}

impl Default for ReplaySweepPerformanceComparison {
    fn default() -> Self {
        Self {
            runtime_ratio_observed_to_estimated: None,
            output_ratio_observed_to_estimated: None,
            output_written_ratio_to_estimated: None,
            completed_ratio_to_planned: None,
            result_rows_per_second: None,
        }
    }
}

/// JSON report for one full or bounded empirical run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPerformanceReport {
    pub(crate) schema_version: u32,
    pub(crate) sweep_id: String,
    pub(crate) name: String,
    pub(crate) source_spec: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) started_at_utc: DateTime<Utc>,
    pub(crate) finished_at_utc: DateTime<Utc>,
    pub(crate) requested_sample_runs: Option<usize>,
    pub(crate) planned_run_count: usize,
    /// Estimate from the caller's complete persisted specification.
    pub(crate) configured_estimate: ReplaySweepResourceEstimate,
    /// Estimate from the exact bounded specification that was executed.
    pub(crate) executed_estimate: ReplaySweepResourceEstimate,
    pub(crate) observed: ReplaySweepPerformanceObserved,
    pub(crate) comparison: ReplaySweepPerformanceComparison,
    pub(crate) children: Vec<ReplaySweepPerformanceChildRow>,
    pub(crate) warnings: Vec<String>,
}

/// A typed row used by [`replay_sweep_performance_csv`]. Aggregate fields are
/// repeated on each child row so the CSV remains a single rectangular table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepPerformanceCsvRow {
    pub(crate) row_kind: String,
    pub(crate) sweep_id: String,
    pub(crate) run_id: String,
    pub(crate) run_index: Option<usize>,
    pub(crate) status: String,
    pub(crate) skipped: Option<bool>,
    pub(crate) result_path: Option<PathBuf>,
    pub(crate) result_bytes: Option<u64>,
    pub(crate) artifact_bytes: Option<u64>,
    pub(crate) result_json_rows: Option<u64>,
    pub(crate) trade_rows: Option<u64>,
    pub(crate) fill_rows: Option<u64>,
    pub(crate) equity_rows: Option<u64>,
    pub(crate) wall_clock_ms: u64,
    pub(crate) completed_count: usize,
    pub(crate) failed_count: usize,
    pub(crate) skipped_count: usize,
    pub(crate) output_bytes: u64,
    pub(crate) output_bytes_written: u64,
    pub(crate) child_rows: u64,
    pub(crate) result_json_rows_total: u64,
    pub(crate) trade_rows_total: u64,
    pub(crate) fill_rows_total: u64,
    pub(crate) equity_rows_total: u64,
    pub(crate) estimated_runtime_seconds: Option<u64>,
    pub(crate) estimated_output_bytes: Option<u64>,
    pub(crate) runtime_ratio_observed_to_estimated: Option<f64>,
    pub(crate) output_ratio_observed_to_estimated: Option<f64>,
    pub(crate) peak_rss_bytes: Option<u64>,
    pub(crate) user_cpu_seconds: Option<f64>,
    pub(crate) system_cpu_seconds: Option<f64>,
}

/// Run a persisted sweep through the existing headless execution boundary and
/// return an empirical report. No broker/network operation is added here: the
/// child runner consumes the selected replay cache view exactly as a normal
/// `run-replay-sweep` invocation does.
pub(crate) async fn run_replay_sweep_performance(
    config: &AppConfig,
    spec_path: &Path,
    options: ReplaySweepPerformanceOptions,
) -> Result<ReplaySweepPerformanceReport> {
    let source_spec = ReplaySweepSpec::load(spec_path)?;
    let configured_guardrails = source_spec
        .guardrail_report(Some(&config.replay_cache_dir))
        .context("estimate configured replay sweep")?;
    let configured_estimate = configured_guardrails.estimate.clone();

    let (mut execution_spec, sample_warning) =
        bounded_execution_spec(&source_spec, options.sample_runs)?;
    let mut warnings = Vec::new();
    if let Some(warning) = sample_warning {
        warnings.push(warning);
    }

    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| isolated_output_dir(&source_spec.output_dir));
    execution_spec.output_dir = output_dir.clone();
    let executed_guardrails = execution_spec
        .guardrail_report(Some(&config.replay_cache_dir))
        .context("estimate executed replay sweep")?;
    let executed_estimate = executed_guardrails.estimate.clone();
    warnings.extend(executed_guardrails.warnings.clone());
    if options.sample_runs.is_some()
        && configured_estimate.combinations != executed_estimate.combinations
    {
        warnings.push(format!(
            "performance probe executed {} of {} configured combinations",
            executed_estimate.combinations, configured_estimate.combinations
        ));
    }

    let temporary_spec_dir = unique_temporary_directory("trader-replay-sweep-performance-spec");
    fs::create_dir_all(&temporary_spec_dir).with_context(|| {
        format!(
            "create temporary performance spec directory {}",
            temporary_spec_dir.display()
        )
    })?;
    let temporary_spec_path = temporary_spec_dir.join("sweep.json");
    execution_spec.save(&temporary_spec_path)?;

    let output_before = directory_size(&output_dir).unwrap_or(0);
    let process_before = read_process_snapshot();
    let started_at_utc = Utc::now();
    let started = Instant::now();
    let summary = run_replay_sweep(
        config,
        &temporary_spec_path,
        options.no_resume,
        options.allow_large,
        options.override_guardrails,
    )
    .await;
    let elapsed = started.elapsed();
    let finished_at_utc = Utc::now();
    let process_after = read_process_snapshot();
    let _ = fs::remove_dir_all(&temporary_spec_dir);
    let summary = summary?;

    warnings.extend(summary.warnings.clone());
    let output_after = directory_size(&output_dir).unwrap_or(output_before);
    let children = summary.runs.iter().map(observe_child).collect::<Vec<_>>();
    let result_json_rows = children.iter().filter_map(|row| row.result_json_rows).sum();
    let trade_rows = children.iter().filter_map(|row| row.trade_rows).sum();
    let fill_rows = children.iter().filter_map(|row| row.fill_rows).sum();
    let equity_rows = children.iter().filter_map(|row| row.equity_rows).sum();
    let wall_clock_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
    let observed = ReplaySweepPerformanceObserved {
        wall_clock_ms,
        completed_count: summary.completed_count,
        failed_count: summary.failed_count,
        skipped_count: summary.skipped_count,
        output_bytes: output_after,
        output_bytes_written: output_after.saturating_sub(output_before),
        child_rows: children.len() as u64,
        result_json_rows,
        trade_rows,
        fill_rows,
        equity_rows,
        process: process_delta(process_before, process_after),
    };
    let comparison = compare_observed(&observed, &executed_estimate, children.len());

    Ok(ReplaySweepPerformanceReport {
        schema_version: REPLAY_SWEEP_PERFORMANCE_SCHEMA_VERSION,
        sweep_id: summary.sweep_id,
        name: summary.name,
        source_spec: spec_path.to_path_buf(),
        output_dir,
        started_at_utc,
        finished_at_utc,
        requested_sample_runs: options.sample_runs,
        planned_run_count: executed_estimate.combinations,
        configured_estimate,
        executed_estimate,
        observed,
        comparison,
        children,
        warnings,
    })
}

/// Serialize a report as pretty JSON without writing it. Keeping this
/// separate lets a caller place the report outside the sweep output tree.
pub(crate) fn replay_sweep_performance_json(
    report: &ReplaySweepPerformanceReport,
) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(report).context("serialize replay sweep performance JSON")
}

/// Build typed rectangular rows for a CSV consumer.
pub(crate) fn replay_sweep_performance_csv_rows(
    report: &ReplaySweepPerformanceReport,
) -> Vec<ReplaySweepPerformanceCsvRow> {
    let aggregate = |row_kind: &str,
                     child: Option<&ReplaySweepPerformanceChildRow>,
                     run_id: String,
                     run_index: Option<usize>,
                     status: String,
                     skipped: Option<bool>| ReplaySweepPerformanceCsvRow {
        row_kind: row_kind.to_string(),
        sweep_id: report.sweep_id.clone(),
        run_id,
        run_index,
        status,
        skipped,
        result_path: child.and_then(|row| row.result_path.clone()),
        result_bytes: child.and_then(|row| row.result_bytes),
        artifact_bytes: child.and_then(|row| row.artifact_bytes),
        result_json_rows: child.and_then(|row| row.result_json_rows),
        trade_rows: child.and_then(|row| row.trade_rows),
        fill_rows: child.and_then(|row| row.fill_rows),
        equity_rows: child.and_then(|row| row.equity_rows),
        wall_clock_ms: report.observed.wall_clock_ms,
        completed_count: report.observed.completed_count,
        failed_count: report.observed.failed_count,
        skipped_count: report.observed.skipped_count,
        output_bytes: report.observed.output_bytes,
        output_bytes_written: report.observed.output_bytes_written,
        child_rows: report.observed.child_rows,
        result_json_rows_total: report.observed.result_json_rows,
        trade_rows_total: report.observed.trade_rows,
        fill_rows_total: report.observed.fill_rows,
        equity_rows_total: report.observed.equity_rows,
        estimated_runtime_seconds: report.executed_estimate.estimated_runtime_seconds,
        estimated_output_bytes: report.executed_estimate.estimated_output_bytes,
        runtime_ratio_observed_to_estimated: report.comparison.runtime_ratio_observed_to_estimated,
        output_ratio_observed_to_estimated: report.comparison.output_ratio_observed_to_estimated,
        peak_rss_bytes: report.observed.process.peak_rss_bytes,
        user_cpu_seconds: report.observed.process.user_cpu_seconds,
        system_cpu_seconds: report.observed.process.system_cpu_seconds,
    };

    if report.children.is_empty() {
        return vec![aggregate(
            "summary",
            None,
            String::new(),
            None,
            "summary".to_string(),
            None,
        )];
    }
    report
        .children
        .iter()
        .map(|child| {
            aggregate(
                "child",
                Some(child),
                child.run_id.clone(),
                Some(child.run_index),
                child.status.clone(),
                Some(child.skipped),
            )
        })
        .collect()
}

/// Serialize typed performance rows as CSV without writing them.
pub(crate) fn replay_sweep_performance_csv(report: &ReplaySweepPerformanceReport) -> Vec<u8> {
    let mut output = String::from(
        "row_kind,sweep_id,run_id,run_index,status,skipped,result_path,result_bytes,artifact_bytes,result_json_rows,trade_rows,fill_rows,equity_rows,wall_clock_ms,completed_count,failed_count,skipped_count,output_bytes,output_bytes_written,child_rows,result_json_rows_total,trade_rows_total,fill_rows_total,equity_rows_total,estimated_runtime_seconds,estimated_output_bytes,runtime_ratio_observed_to_estimated,output_ratio_observed_to_estimated,peak_rss_bytes,user_cpu_seconds,system_cpu_seconds\n",
    );
    for row in replay_sweep_performance_csv_rows(report) {
        let fields = [
            row.row_kind,
            row.sweep_id,
            row.run_id,
            optional_usize(row.run_index),
            row.status,
            optional_bool(row.skipped),
            row.result_path
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            optional_u64(row.result_bytes),
            optional_u64(row.artifact_bytes),
            optional_u64(row.result_json_rows),
            optional_u64(row.trade_rows),
            optional_u64(row.fill_rows),
            optional_u64(row.equity_rows),
            row.wall_clock_ms.to_string(),
            row.completed_count.to_string(),
            row.failed_count.to_string(),
            row.skipped_count.to_string(),
            row.output_bytes.to_string(),
            row.output_bytes_written.to_string(),
            row.child_rows.to_string(),
            row.result_json_rows_total.to_string(),
            row.trade_rows_total.to_string(),
            row.fill_rows_total.to_string(),
            row.equity_rows_total.to_string(),
            optional_u64(row.estimated_runtime_seconds),
            optional_u64(row.estimated_output_bytes),
            optional_f64(row.runtime_ratio_observed_to_estimated),
            optional_f64(row.output_ratio_observed_to_estimated),
            optional_u64(row.peak_rss_bytes),
            optional_f64(row.user_cpu_seconds),
            optional_f64(row.system_cpu_seconds),
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

/// Write either or both report formats. No output is written when both paths
/// are `None`, which keeps report generation side-effect free for library
/// callers.
pub(crate) fn write_replay_sweep_performance_report(
    report: &ReplaySweepPerformanceReport,
    json_path: Option<&Path>,
    csv_path: Option<&Path>,
) -> Result<()> {
    if json_path.is_none() && csv_path.is_none() {
        return Ok(());
    }
    if let Some(path) = json_path {
        write_bytes_atomic(path, &replay_sweep_performance_json(report)?)?;
    }
    if let Some(path) = csv_path {
        write_bytes_atomic(path, &replay_sweep_performance_csv(report))?;
    }
    Ok(())
}

fn bounded_execution_spec(
    source: &ReplaySweepSpec,
    requested: Option<usize>,
) -> Result<(ReplaySweepSpec, Option<String>)> {
    let Some(requested) = requested else {
        return Ok((source.clone(), None));
    };
    if requested == 0 {
        bail!("performance sample_runs must be greater than zero when set");
    }
    let total = source.combination_count()?;
    if requested >= total {
        return Ok((source.clone(), None));
    }

    let mut bounded = source.clone();
    let mut product = 1_usize;
    for parameter in &mut bounded.parameters {
        let remaining = requested / product.max(1);
        let keep = parameter.values.len().min(remaining.max(1));
        parameter.values.truncate(keep);
        product = product.saturating_mul(keep);
    }
    // `keep >= 1` above means this is always nonzero. Setting max_runs to the
    // effective count makes the persisted temporary spec pass the existing
    // validation boundary without weakening the source guardrail policy.
    bounded.max_runs = product;
    Ok((
        bounded,
        Some(format!(
            "bounded performance probe requested {requested} combinations; executing deterministic prefix of at most {product}"
        )),
    ))
}

fn isolated_output_dir(configured: &Path) -> PathBuf {
    let suffix = unique_suffix();
    let parent = configured.parent().unwrap_or_else(|| Path::new("."));
    let name = configured
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("replay-sweep");
    parent.join(format!(".{name}-performance-{suffix}"))
}

fn unique_temporary_directory(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        unique_suffix()
    ))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn observe_child(summary: &ReplaySweepRunSummary) -> ReplaySweepPerformanceChildRow {
    let mut row = ReplaySweepPerformanceChildRow {
        run_id: summary.run_id.clone(),
        run_index: summary.run_index,
        status: summary.status.clone(),
        skipped: summary.skipped,
        result_path: summary.result_path.clone(),
        result_bytes: None,
        artifact_bytes: None,
        result_json_rows: None,
        trade_rows: None,
        fill_rows: None,
        equity_rows: None,
    };
    let Some(result_path) = summary.result_path.as_deref() else {
        return row;
    };
    row.result_bytes = fs::metadata(result_path)
        .ok()
        .map(|metadata| metadata.len());
    row.result_json_rows = fs::read(result_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .map(|_| 1);
    if let Some(directory) = result_path.parent() {
        row.artifact_bytes = directory_size(directory).ok();
        row.trade_rows = csv_data_rows(&directory.join("trades.csv"));
        row.fill_rows = csv_data_rows(&directory.join("fills.csv"));
        row.equity_rows = csv_data_rows(&directory.join("equity.csv"));
    }
    row
}

fn csv_data_rows(path: &Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    Some(
        text.lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .count() as u64,
    )
}

fn compare_observed(
    observed: &ReplaySweepPerformanceObserved,
    estimate: &ReplaySweepResourceEstimate,
    planned_runs: usize,
) -> ReplaySweepPerformanceComparison {
    let observed_seconds = observed.wall_clock_ms as f64 / 1_000.0;
    let result_rows = observed.result_json_rows + observed.trade_rows + observed.fill_rows;
    ReplaySweepPerformanceComparison {
        runtime_ratio_observed_to_estimated: ratio(
            observed_seconds,
            estimate
                .estimated_runtime_seconds
                .map(|seconds| seconds as f64),
        ),
        output_ratio_observed_to_estimated: ratio(
            observed.output_bytes as f64,
            estimate.estimated_output_bytes.map(|bytes| bytes as f64),
        ),
        output_written_ratio_to_estimated: ratio(
            observed.output_bytes_written as f64,
            estimate.estimated_output_bytes.map(|bytes| bytes as f64),
        ),
        completed_ratio_to_planned: (planned_runs > 0)
            .then(|| observed.completed_count as f64 / planned_runs as f64),
        result_rows_per_second: (observed_seconds > 0.0)
            .then(|| result_rows as f64 / observed_seconds),
    }
}

fn ratio(actual: f64, estimate: Option<f64>) -> Option<f64> {
    let estimate = estimate?;
    (estimate.is_finite() && estimate > 0.0 && actual.is_finite()).then_some(actual / estimate)
}

#[derive(Debug, Clone, Copy)]
struct ProcessSnapshot {
    peak_rss_bytes: Option<u64>,
    user_cpu_ticks: Option<u64>,
    system_cpu_ticks: Option<u64>,
    clock_ticks_per_second: Option<u64>,
}

#[cfg(target_os = "linux")]
fn read_process_snapshot() -> Option<ProcessSnapshot> {
    let peak_rss_bytes = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let value = line.strip_prefix("VmHWM:")?.split_whitespace().next()?;
                value.parse::<u64>().ok().map(|kilobytes| kilobytes * 1024)
            })
        });
    let (user_cpu_ticks, system_cpu_ticks) = fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            let close = stat.rfind(')')?;
            let fields = stat
                .get(close + 1..)?
                .split_whitespace()
                .collect::<Vec<_>>();
            // `fields[0]` is state (field 3); utime/ stime are fields 14/15.
            Some((fields.get(11)?.parse().ok()?, fields.get(12)?.parse().ok()?))
        })
        .map_or((None, None), |(user, system)| (Some(user), Some(system)));
    // SAFETY: sysconf is a read-only libc query and has no pointer arguments.
    let clock_ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let clock_ticks_per_second =
        (clock_ticks_per_second > 0).then_some(clock_ticks_per_second as u64);
    if peak_rss_bytes.is_none() && user_cpu_ticks.is_none() && system_cpu_ticks.is_none() {
        return None;
    }
    Some(ProcessSnapshot {
        peak_rss_bytes,
        user_cpu_ticks,
        system_cpu_ticks,
        clock_ticks_per_second,
    })
}

#[cfg(not(target_os = "linux"))]
fn read_process_snapshot() -> Option<ProcessSnapshot> {
    None
}

fn process_delta(
    before: Option<ProcessSnapshot>,
    after: Option<ProcessSnapshot>,
) -> ReplaySweepProcessMetrics {
    let peak_rss_bytes = match (
        before.and_then(|value| value.peak_rss_bytes),
        after.and_then(|value| value.peak_rss_bytes),
    ) {
        (Some(before), Some(after)) => Some(before.max(after)),
        (None, Some(after)) => Some(after),
        (Some(before), None) => Some(before),
        (None, None) => None,
    };
    let user_cpu_seconds = cpu_delta(before, after, true);
    let system_cpu_seconds = cpu_delta(before, after, false);
    ReplaySweepProcessMetrics {
        peak_rss_bytes,
        user_cpu_seconds,
        system_cpu_seconds,
    }
}

fn cpu_delta(
    before: Option<ProcessSnapshot>,
    after: Option<ProcessSnapshot>,
    user: bool,
) -> Option<f64> {
    let before = before?;
    let after = after?;
    let ticks = if user {
        after.user_cpu_ticks?.saturating_sub(before.user_cpu_ticks?)
    } else {
        after
            .system_cpu_ticks?
            .saturating_sub(before.system_cpu_ticks?)
    };
    let clock = after
        .clock_ticks_per_second
        .or(before.clock_ticks_per_second)?;
    (clock > 0).then_some(ticks as f64 / clock as f64)
}

fn directory_size(root: &Path) -> Result<u64> {
    if !root.exists() {
        return Ok(0);
    }
    if root.is_file() {
        return Ok(fs::metadata(root)?.len());
    }
    let mut total = 0_u64;
    for entry in
        fs::read_dir(root).with_context(|| format!("read output directory {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            total = total.saturating_add(directory_size(&path)?);
        } else if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("performance report path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create performance report directory {}", parent.display()))?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("report"),
        std::process::id(),
        unique_suffix()
    ));
    fs::write(&temporary, bytes)
        .with_context(|| format!("write temporary performance report {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("replace performance report {}", path.display()))
}

fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_bool(value: Option<bool>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
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
    use serde_json::json;

    fn estimate() -> ReplaySweepResourceEstimate {
        ReplaySweepResourceEstimate {
            combinations: 2,
            parallel_jobs: 1,
            estimated_input_rows: Some(100),
            estimated_input_rows_per_worker: Some(100),
            input_source: "fixture bars".to_string(),
            estimated_memory_bytes: Some(1024),
            estimated_output_bytes: Some(2_000),
            estimated_runtime_seconds: Some(2),
            notes: Vec::new(),
        }
    }

    fn report_with_child(root: &Path) -> ReplaySweepPerformanceReport {
        let child = ReplaySweepPerformanceChildRow {
            run_id: "sweep-000001".to_string(),
            run_index: 0,
            status: "completed".to_string(),
            skipped: false,
            result_path: Some(root.join("result.json")),
            result_bytes: Some(20),
            artifact_bytes: Some(120),
            result_json_rows: Some(1),
            trade_rows: Some(2),
            fill_rows: Some(3),
            equity_rows: Some(4),
        };
        ReplaySweepPerformanceReport {
            schema_version: REPLAY_SWEEP_PERFORMANCE_SCHEMA_VERSION,
            sweep_id: "fixture".to_string(),
            name: "Fixture".to_string(),
            source_spec: root.join("spec.json"),
            output_dir: root.to_path_buf(),
            started_at_utc: Utc::now(),
            finished_at_utc: Utc::now(),
            requested_sample_runs: Some(1),
            planned_run_count: 1,
            configured_estimate: estimate(),
            executed_estimate: ReplaySweepResourceEstimate {
                combinations: 1,
                ..estimate()
            },
            observed: ReplaySweepPerformanceObserved {
                wall_clock_ms: 1_000,
                completed_count: 1,
                failed_count: 0,
                skipped_count: 0,
                output_bytes: 120,
                output_bytes_written: 120,
                child_rows: 1,
                result_json_rows: 1,
                trade_rows: 2,
                fill_rows: 3,
                equity_rows: 4,
                process: ReplaySweepProcessMetrics::default(),
            },
            comparison: ReplaySweepPerformanceComparison {
                runtime_ratio_observed_to_estimated: Some(0.5),
                output_ratio_observed_to_estimated: Some(0.06),
                output_written_ratio_to_estimated: Some(0.06),
                completed_ratio_to_planned: Some(1.0),
                result_rows_per_second: Some(6.0),
            },
            children: vec![child],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn bounded_spec_keeps_source_unchanged_and_never_exceeds_sample() {
        let mut source = ReplaySweepSpec::default();
        source.parameters = vec![
            super::super::sweep::ReplaySweepParameter {
                path: "native_ema.fast_length".to_string(),
                values: vec![json!(1), json!(2), json!(3), json!(4)],
            },
            super::super::sweep::ReplaySweepParameter {
                path: "native_ema.slow_length".to_string(),
                values: vec![json!(10), json!(20), json!(30)],
            },
        ];
        let original = source.clone();
        let (bounded, warning) = bounded_execution_spec(&source, Some(5)).expect("bound");
        assert_eq!(source, original);
        assert!(bounded.combination_count().expect("count") <= 5);
        assert!(warning.is_some());
        assert_eq!(
            bounded.max_runs,
            bounded.combination_count().expect("count")
        );
    }

    #[test]
    fn child_observation_counts_fixture_rows_and_bytes() {
        let root =
            std::env::temp_dir().join(format!("sweep-performance-fixture-{}", unique_suffix()));
        fs::create_dir_all(&root).expect("fixture directory");
        fs::write(root.join("result.json"), br#"{"status":"completed"}"#).expect("result");
        fs::write(root.join("trades.csv"), b"header\na\nb\n").expect("trades");
        fs::write(root.join("fills.csv"), b"header\na\n").expect("fills");
        fs::write(root.join("equity.csv"), b"header\na\nb\nc\n").expect("equity");
        let summary = ReplaySweepRunSummary {
            run_id: "fixture-000001".to_string(),
            run_index: 0,
            status: "completed".to_string(),
            skipped: false,
            result_path: Some(root.join("result.json")),
            error: None,
            gross_pnl: None,
            net_pnl: None,
            fees: None,
            max_drawdown: None,
            trade_count: None,
            fill_count: None,
            execution_backend: None,
            fallback_reason: None,
        };
        let row = observe_child(&summary);
        assert_eq!(row.result_json_rows, Some(1));
        assert_eq!(row.trade_rows, Some(2));
        assert_eq!(row.fill_rows, Some(1));
        assert_eq!(row.equity_rows, Some(3));
        assert!(row.artifact_bytes.unwrap_or_default() >= row.result_bytes.unwrap_or_default());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn json_round_trip_and_csv_include_observed_and_estimated_fields() {
        let root =
            std::env::temp_dir().join(format!("sweep-performance-report-{}", unique_suffix()));
        let report = report_with_child(&root);
        let bytes = replay_sweep_performance_json(&report).expect("json");
        let parsed: ReplaySweepPerformanceReport = serde_json::from_slice(&bytes).expect("parse");
        assert_eq!(parsed, report);
        let csv = String::from_utf8(replay_sweep_performance_csv(&report)).expect("csv");
        assert!(csv.starts_with("row_kind,sweep_id,run_id"));
        assert!(csv.contains("estimated_runtime_seconds"));
        assert!(csv.contains("observed_to_estimated") || csv.contains("runtime_ratio"));
    }

    #[test]
    fn process_delta_is_optional_and_cpu_is_nonnegative() {
        let before = ProcessSnapshot {
            peak_rss_bytes: Some(10),
            user_cpu_ticks: Some(5),
            system_cpu_ticks: Some(2),
            clock_ticks_per_second: Some(100),
        };
        let after = ProcessSnapshot {
            peak_rss_bytes: Some(20),
            user_cpu_ticks: Some(15),
            system_cpu_ticks: Some(4),
            clock_ticks_per_second: Some(100),
        };
        let metrics = process_delta(Some(before), Some(after));
        assert_eq!(metrics.peak_rss_bytes, Some(20));
        assert_eq!(metrics.user_cpu_seconds, Some(0.1));
        assert_eq!(metrics.system_cpu_seconds, Some(0.02));
    }

    #[test]
    fn report_writer_is_noop_without_paths() {
        let root = std::env::temp_dir().join(format!("sweep-performance-noop-{}", unique_suffix()));
        let report = report_with_child(&root);
        write_replay_sweep_performance_report(&report, None, None).expect("no-op writer");
        assert!(!root.exists());
    }
}
