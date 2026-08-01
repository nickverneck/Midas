//! Headless execution of a validated replay sweep.
//!
//! Each child gets an isolated Tradovate replay service and broker gateway.
//! That keeps execution state, ledgers, positions, and strategy runtime state
//! separate while still reusing the exact engine path used by the TUI replay
//! session.  The runner only coordinates services and output; it does not
//! implement a second fill or strategy engine.

use super::sweep::{
    ReplaySweepChildSpec, ReplaySweepGuardrailReport, ReplaySweepOutputFormat,
    ReplaySweepResourceEstimate, ReplaySweepSpec,
};
use crate::broker::{BrokerKind, MarketSnapshot, ReplaySpeed, ServiceCommand, ServiceEvent};
use crate::config::AppConfig;
use crate::replay_cache::{ReplayDatasetView, ReplayDatasetViewStore};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use futures_util::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

pub(crate) const REPLAY_SWEEP_SUMMARY_SCHEMA_VERSION: u32 = 1;
const REPLAY_SWEEP_EVENT_TIMEOUT: Duration = Duration::from_secs(12 * 60 * 60);
const REPLAY_SWEEP_SERVICE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepRunSummary {
    pub(crate) run_id: String,
    pub(crate) run_index: usize,
    pub(crate) status: String,
    pub(crate) skipped: bool,
    pub(crate) result_path: Option<PathBuf>,
    pub(crate) error: Option<String>,
    pub(crate) gross_pnl: Option<f64>,
    pub(crate) net_pnl: Option<f64>,
    pub(crate) fees: Option<f64>,
    pub(crate) max_drawdown: Option<f64>,
    pub(crate) trade_count: Option<usize>,
    pub(crate) fill_count: Option<usize>,
}

impl ReplaySweepRunSummary {
    fn failed(child: &ReplaySweepChildSpec, error: impl Into<String>) -> Self {
        Self {
            run_id: child.run_id.clone(),
            run_index: child.run_index,
            status: "failed".to_string(),
            skipped: false,
            result_path: None,
            error: Some(error.into()),
            gross_pnl: None,
            net_pnl: None,
            fees: None,
            max_drawdown: None,
            trade_count: None,
            fill_count: None,
        }
    }
}

/// One accounting overlay for one completed child. The child result keeps the
/// authoritative copy; this compact parent row makes fee-scenario ranking
/// possible without opening every run directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepFeeScenarioSummary {
    pub(crate) run_id: String,
    pub(crate) run_index: usize,
    pub(crate) scenario_name: String,
    pub(crate) currency: String,
    pub(crate) total_per_contract: f64,
    pub(crate) fees: f64,
    pub(crate) gross_pnl: f64,
    pub(crate) net_pnl: f64,
    pub(crate) ending_equity: f64,
    pub(crate) return_on_initial_capital_pct: Option<f64>,
    pub(crate) max_drawdown: f64,
    pub(crate) max_drawdown_pct: Option<f64>,
    pub(crate) profit_factor: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplaySweepSummaryDocument {
    pub(crate) schema_version: u32,
    pub(crate) sweep_id: String,
    pub(crate) name: String,
    pub(crate) created_at_utc: DateTime<Utc>,
    pub(crate) run_count: usize,
    pub(crate) completed_count: usize,
    pub(crate) failed_count: usize,
    pub(crate) skipped_count: usize,
    pub(crate) warnings: Vec<String>,
    pub(crate) runs: Vec<ReplaySweepRunSummary>,
    #[serde(default)]
    pub(crate) fee_scenarios: Vec<ReplaySweepFeeScenarioSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) resource_estimate: Option<ReplaySweepResourceEstimate>,
}

pub(crate) async fn run_replay_sweep(
    config: &AppConfig,
    spec_path: &Path,
    no_resume: bool,
    allow_large: bool,
    override_guardrails: bool,
) -> Result<ReplaySweepSummaryDocument> {
    let spec = ReplaySweepSpec::load(spec_path)?;
    if config.broker != BrokerKind::Tradovate {
        bail!("headless replay sweeps currently require the Tradovate broker");
    }
    let guardrail_report = spec.guardrail_report(Some(&config.replay_cache_dir))?;
    enforce_guardrails(&guardrail_report, allow_large, override_guardrails)?;
    let plan = spec.plan()?;
    let output_root = spec.output_dir.clone();
    fs::create_dir_all(&output_root)
        .with_context(|| format!("create sweep output directory {}", output_root.display()))?;
    let runs_root = output_root.join("runs");
    fs::create_dir_all(&runs_root)
        .with_context(|| format!("create sweep runs directory {}", runs_root.display()))?;

    // Persist canonical copies before starting any child. A completed child
    // can therefore be resumed even if the original spec is moved later.
    spec.save(&output_root.join("sweep.json"))?;
    plan.save(&output_root.join("sweep-plan.json"))?;
    let view_path = ensure_dataset_view_file(&config.replay_cache_dir, &spec.base_dataset_view)?;

    let parallelism = spec.parallelism.max(1).min(plan.children.len().max(1));
    let base_config = config.clone();
    let summaries = stream::iter(plan.children.clone())
        .map(|child| {
            let base_config = base_config.clone();
            let spec = spec.clone();
            let view_path = view_path.clone();
            async move {
                match run_child(&base_config, &spec, child.clone(), &view_path, no_resume).await {
                    Ok(summary) => summary,
                    Err(error) => ReplaySweepRunSummary::failed(&child, error.to_string()),
                }
            }
        })
        .buffer_unordered(parallelism)
        .collect::<Vec<_>>()
        .await;

    let mut runs = summaries;
    runs.sort_by_key(|summary| summary.run_index);
    let completed_count = runs
        .iter()
        .filter(|summary| summary.status == "completed")
        .count();
    let failed_count = runs
        .iter()
        .filter(|summary| summary.status == "failed")
        .count();
    let skipped_count = runs.iter().filter(|summary| summary.skipped).count();
    let mut warnings = guardrail_report.warnings.clone();
    if override_guardrails && !guardrail_report.violations.is_empty() {
        warnings.push(
            "one or more configured replay sweep guardrails were explicitly overridden at launch"
                .to_string(),
        );
        warnings.extend(
            guardrail_report
                .violations
                .iter()
                .map(|violation| format!("overridden guardrail: {violation}")),
        );
    }
    if guardrail_report.requires_confirmation && (allow_large || override_guardrails) {
        warnings.push("large replay sweep confirmation accepted at launch".to_string());
    }
    if spec
        .output_formats
        .contains(&ReplaySweepOutputFormat::ParquetRows)
    {
        warnings.push(
            "Parquet sweep summary output is reserved for a later analytics slice; JSON/CSV rows were written."
                .to_string(),
        );
    }
    let fee_scenarios = collect_fee_scenario_summaries(&runs, &mut warnings);
    let document = ReplaySweepSummaryDocument {
        schema_version: REPLAY_SWEEP_SUMMARY_SCHEMA_VERSION,
        sweep_id: spec.sweep_id.clone(),
        name: spec.name.clone(),
        created_at_utc: Utc::now(),
        run_count: runs.len(),
        completed_count,
        failed_count,
        skipped_count,
        warnings,
        runs,
        fee_scenarios,
        resource_estimate: Some(guardrail_report.estimate.clone()),
    };
    write_summary_outputs(&output_root, &spec, &document)?;
    Ok(document)
}

fn enforce_guardrails(
    report: &ReplaySweepGuardrailReport,
    allow_large: bool,
    override_guardrails: bool,
) -> Result<()> {
    if !report.violations.is_empty() && !override_guardrails {
        let details = report
            .violations
            .iter()
            .map(|violation| format!("- {violation}"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!(
            "replay sweep launch blocked by resource guardrails:\n{details}\nUse --override-guardrails only after reviewing the estimate."
        );
    }
    if report.requires_confirmation && !allow_large && !override_guardrails {
        bail!(
            "replay sweep has {} combinations and requires explicit confirmation; rerun with --allow-large",
            report.estimate.combinations
        );
    }
    Ok(())
}

fn ensure_dataset_view_file(cache_root: &Path, view: &ReplayDatasetView) -> Result<PathBuf> {
    let store = ReplayDatasetViewStore::new(cache_root);
    let path = store.path_for(&view.id)?;
    if path.is_file() {
        let existing = store.load_path(&path)?;
        if existing != *view {
            bail!(
                "dataset view id {} already exists with different contents at {}",
                view.id,
                path.display()
            );
        }
        return Ok(path);
    }
    store.save(view)
}

async fn run_child(
    base_config: &AppConfig,
    spec: &ReplaySweepSpec,
    child: ReplaySweepChildSpec,
    view_path: &Path,
    no_resume: bool,
) -> Result<ReplaySweepRunSummary> {
    if child.resolved_strategy.kind != crate::strategy::StrategyKind::Native {
        bail!(
            "child {} uses {:?}; headless replay currently supports native strategies only",
            child.run_id,
            child.resolved_strategy.kind
        );
    }

    let runs_root = spec.output_dir.join("runs");
    let expected_result_path = runs_root.join(&child.run_id).join("result.json");
    if !no_resume && completed_result_matches(&expected_result_path, &child, spec)? {
        return Ok(summary_from_result(&child, expected_result_path, true)?);
    }

    let mut cfg = base_config.clone();
    cfg.broker = BrokerKind::Tradovate;
    cfg.env = spec.base_dataset_view.source.env;
    cfg.candle_mode = child.candle_mode;
    cfg.order_qty = child.resolved_strategy.order_qty;
    cfg.replay_result_dir = runs_root;
    cfg.replay_run_id = Some(child.run_id.clone());
    cfg.replay_headless = true;
    cfg.replay_initial_capital = child.initial_capital;
    cfg.replay_account_currency = child.primary_fee_schedule.currency.clone();
    cfg.replay_engine_mode = child.engine_mode;
    cfg.replay_fill_model = child.fill_model;
    cfg.replay_latency_model = child.latency.model;
    cfg.replay_fixed_latency_ms = child.latency.fixed_latency_ms;
    cfg.replay_observed_latency_ms = child.latency.observed_samples_ms.clone();
    cfg.replay_latency_seed = child.latency.seed;
    cfg.replay_bar_protection_policy = child.bar_protection_policy;
    if let Some(margin) = child.margin.as_ref() {
        cfg.replay_margin_model = margin.model.clone();
        cfg.replay_account_currency = margin.currency.clone();
        cfg.replay_margin_per_contract = margin.margin_per_contract;
        cfg.replay_safety_buffer = margin.safety_buffer;
        cfg.replay_safety_buffer_percent = margin.safety_buffer_percent;
    } else {
        cfg.replay_margin_per_contract = 0.0;
        cfg.replay_safety_buffer = 0.0;
        cfg.replay_safety_buffer_percent = 0.0;
    }
    cfg.validate()?;

    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (market_tx, _market_rx) = watch::channel(MarketSnapshot::default());
    let service_task = tokio::spawn(crate::tradovate::service_loop(
        command_rx, event_tx, market_tx,
    ));
    let result = drive_child_service(
        &command_tx,
        event_rx,
        cfg,
        child.clone(),
        view_path.to_path_buf(),
    )
    .await;
    drop(command_tx);
    shutdown_service_task(service_task).await;
    let mut summary = result?;
    if summary.status == "completed" {
        apply_fee_scenarios(&summary, &child)?;
        patch_result_with_sweep_metadata(&summary, &child, spec)?;
        summary = summary_from_result(&child, summary.result_path.clone().unwrap(), false)?;
    }
    Ok(summary)
}

async fn drive_child_service(
    command_tx: &mpsc::UnboundedSender<ServiceCommand>,
    mut event_rx: mpsc::UnboundedReceiver<ServiceEvent>,
    config: AppConfig,
    child: ReplaySweepChildSpec,
    view_path: PathBuf,
) -> Result<ReplaySweepRunSummary> {
    command_tx
        .send(ServiceCommand::EnterReplayMode {
            config,
            bar_type: child.bar_type,
            candle_mode: child.candle_mode,
            replay_dataset_manifest: None,
            replay_dataset_view: Some(view_path),
        })
        .map_err(|_| anyhow::anyhow!("headless replay service is unavailable"))?;

    // Queue setup immediately behind EnterReplayMode. The backend handles
    // commands in order, so the strategy is configured and armed before a
    // no-pacing worker can drain a small dataset.
    command_tx
        .send(ServiceCommand::SetExecutionStrategyConfig(
            child.resolved_strategy.clone(),
        ))
        .map_err(|_| anyhow::anyhow!("headless replay service closed before strategy setup"))?;
    command_tx
        .send(ServiceCommand::SetReplaySpeed {
            speed: ReplaySpeed::X25,
        })
        .map_err(|_| anyhow::anyhow!("headless replay service closed before speed setup"))?;
    command_tx
        .send(ServiceCommand::ArmExecutionStrategy)
        .map_err(|_| anyhow::anyhow!("headless replay service closed before arming"))?;

    let mut connected = false;
    while let Some(event) = timeout(REPLAY_SWEEP_EVENT_TIMEOUT, event_rx.recv())
        .await
        .context("waiting for headless replay event")?
    {
        match event {
            ServiceEvent::Connected { .. } => connected = true,
            ServiceEvent::ReplayResultSaved {
                run_id,
                result_path,
                status: _,
                ..
            } => {
                if run_id != child.run_id {
                    bail!(
                        "headless replay saved unexpected run id {run_id}; expected {}",
                        child.run_id
                    );
                }
                return summary_from_result(&child, result_path, false);
            }
            ServiceEvent::Error(message) => {
                if !connected {
                    bail!(
                        "headless replay setup failed for {}: {message}",
                        child.run_id
                    );
                }
                // Replay worker errors are followed by ReplayResultSaved with
                // a failed document. Keep listening so that failed children
                // remain resumable instead of becoming orphaned partial runs.
                eprintln!("sweep child {}: {message}", child.run_id);
            }
            _ => {}
        }
    }
    bail!(
        "headless replay service closed before saving {}",
        child.run_id
    )
}

async fn shutdown_service_task(mut task: JoinHandle<()>) {
    if timeout(REPLAY_SWEEP_SERVICE_SHUTDOWN_TIMEOUT, &mut task)
        .await
        .is_err()
    {
        task.abort();
        let _ = task.await;
    }
}

fn completed_result_matches(
    path: &Path,
    child: &ReplaySweepChildSpec,
    spec: &ReplaySweepSpec,
) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let bytes =
        fs::read(path).with_context(|| format!("read existing result {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse existing result {}", path.display()))?;
    if value.get("status").and_then(Value::as_str) != Some("completed")
        || value.get("run_id").and_then(Value::as_str) != Some(child.run_id.as_str())
    {
        return Ok(false);
    }
    let Some(sweep) = value.get("sweep") else {
        // A result written before the sweep metadata patch is not safe to
        // resume: it may have been interrupted between the two atomic writes.
        return Ok(false);
    };
    Ok(
        sweep.get("parent_sweep_id").and_then(Value::as_str) == Some(spec.sweep_id.as_str())
            && sweep.get("run_index").and_then(Value::as_u64) == Some(child.run_index as u64)
            && sweep.get("parameter_values") == Some(&json!(child.parameter_values))
            && sweep.get("overrides") == Some(&json!(child.overrides))
            && sweep.get("resolved_strategy") == Some(&json!(child.resolved_strategy))
            && sweep.get("dataset_view") == Some(&json!(child.base_dataset_view))
            && sweep.get("bar_type") == Some(&json!(child.bar_type))
            && sweep.get("candle_mode") == Some(&json!(child.candle_mode)),
    )
}

fn summary_from_result(
    child: &ReplaySweepChildSpec,
    result_path: PathBuf,
    skipped: bool,
) -> Result<ReplaySweepRunSummary> {
    let bytes = fs::read(&result_path)
        .with_context(|| format!("read replay child result {}", result_path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse replay child result {}", result_path.display()))?;
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("failed")
        .to_string();
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let summary = value.get("summary");
    Ok(ReplaySweepRunSummary {
        run_id: child.run_id.clone(),
        run_index: child.run_index,
        status,
        skipped,
        result_path: Some(result_path),
        error,
        gross_pnl: summary
            .and_then(|value| value.get("gross_pnl"))
            .and_then(Value::as_f64),
        net_pnl: summary
            .and_then(|value| value.get("net_pnl"))
            .and_then(Value::as_f64),
        fees: summary
            .and_then(|value| value.get("fees"))
            .and_then(Value::as_f64),
        max_drawdown: summary
            .and_then(|value| value.get("max_drawdown"))
            .and_then(Value::as_f64),
        trade_count: summary
            .and_then(|value| value.get("trade_count"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
        fill_count: summary
            .and_then(|value| value.get("fill_count"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok()),
    })
}

fn apply_fee_scenarios(
    summary: &ReplaySweepRunSummary,
    child: &ReplaySweepChildSpec,
) -> Result<()> {
    let Some(result_path) = summary.result_path.as_deref() else {
        bail!("completed child {} has no result path", child.run_id);
    };
    for schedule in &child.fee_scenarios {
        crate::tradovate::reprice_replay_result(result_path, schedule.clone())?;
    }
    // Always finish on the primary schedule. Otherwise a neutral primary plus
    // one or more alternate overlays would leave the last alternate as the
    // active summary even though the child is ranked under its primary.
    if !child.fee_scenarios.is_empty()
        || child.primary_fee_schedule.name != "fee_neutral"
        || child.primary_fee_schedule.total_per_contract().abs() > f64::EPSILON
    {
        crate::tradovate::reprice_replay_result(result_path, child.primary_fee_schedule.clone())?;
    }
    Ok(())
}

fn collect_fee_scenario_summaries(
    runs: &[ReplaySweepRunSummary],
    warnings: &mut Vec<String>,
) -> Vec<ReplaySweepFeeScenarioSummary> {
    let mut rows = Vec::new();
    for run in runs {
        if run.status != "completed" {
            continue;
        }
        let Some(path) = run.result_path.as_deref() else {
            warnings.push(format!(
                "completed sweep child {} has no result path; fee scenarios omitted",
                run.run_id
            ));
            continue;
        };
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                warnings.push(format!(
                    "could not read fee scenarios for {}: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                warnings.push(format!(
                    "could not parse fee scenarios for {}: {error}",
                    path.display()
                ));
                continue;
            }
        };
        let Some(scenarios) = value.get("fee_scenarios").and_then(Value::as_array) else {
            continue;
        };
        for scenario in scenarios {
            let Some(schedule) = scenario.get("schedule") else {
                warnings.push(format!(
                    "fee scenario in {} is missing its schedule",
                    path.display()
                ));
                continue;
            };
            let Some(scenario_name) = schedule.get("name").and_then(Value::as_str) else {
                warnings.push(format!(
                    "fee scenario in {} is missing its name",
                    path.display()
                ));
                continue;
            };
            let number = |name: &str| scenario.get(name).and_then(Value::as_f64).unwrap_or(0.0);
            let optional_number = |name: &str| scenario.get(name).and_then(Value::as_f64);
            rows.push(ReplaySweepFeeScenarioSummary {
                run_id: run.run_id.clone(),
                run_index: run.run_index,
                scenario_name: scenario_name.to_string(),
                currency: schedule
                    .get("currency")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                total_per_contract: schedule
                    .get("total_per_contract")
                    .and_then(Value::as_f64)
                    .unwrap_or_else(|| {
                        [
                            "commission_per_contract",
                            "exchange_per_contract",
                            "clearing_per_contract",
                            "regulatory_per_contract",
                            "misc_per_contract",
                        ]
                        .into_iter()
                        .filter_map(|name| schedule.get(name).and_then(Value::as_f64))
                        .sum()
                    }),
                fees: number("fees"),
                gross_pnl: number("gross_pnl"),
                net_pnl: number("net_pnl"),
                ending_equity: number("ending_equity"),
                return_on_initial_capital_pct: optional_number("return_on_initial_capital_pct"),
                max_drawdown: number("max_drawdown"),
                max_drawdown_pct: optional_number("max_drawdown_pct"),
                profit_factor: optional_number("profit_factor"),
            });
        }
    }
    rows.sort_by(|left, right| {
        left.run_index
            .cmp(&right.run_index)
            .then_with(|| left.scenario_name.cmp(&right.scenario_name))
    });
    rows
}

fn patch_result_with_sweep_metadata(
    summary: &ReplaySweepRunSummary,
    child: &ReplaySweepChildSpec,
    spec: &ReplaySweepSpec,
) -> Result<()> {
    let result_path = summary
        .result_path
        .as_deref()
        .context("completed child has no result path")?;
    let bytes = fs::read(result_path)
        .with_context(|| format!("read result for sweep metadata {}", result_path.display()))?;
    let mut document: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse result for sweep metadata {}", result_path.display()))?;
    if let Some(metadata) = document.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.insert(
            "run_mode".to_string(),
            Value::String("sweep_child".to_string()),
        );
    }
    document["sweep"] = json!({
        "parent_sweep_id": spec.sweep_id,
        "run_id": child.run_id,
        "run_index": child.run_index,
        "parameter_values": child.parameter_values,
        "overrides": child.overrides,
        "resolved_strategy": child.resolved_strategy,
        "dataset_view": child.base_dataset_view,
        "bar_type": child.bar_type,
        "candle_mode": child.candle_mode,
        "engine_mode": child.engine_mode,
        "fill_model": child.fill_model,
        "latency": child.latency,
        "bar_protection_policy": child.bar_protection_policy,
        "primary_fee_schedule": child.primary_fee_schedule,
        "fee_scenarios": child.fee_scenarios,
        "initial_capital": child.initial_capital,
        "margin": child.margin,
    });
    write_json_atomic(result_path, &document)
}

fn write_summary_outputs(
    output_root: &Path,
    spec: &ReplaySweepSpec,
    document: &ReplaySweepSummaryDocument,
) -> Result<()> {
    if let Some(estimate) = document.resource_estimate.as_ref() {
        write_json_atomic(&output_root.join("sweep-resource-estimate.json"), estimate)?;
    }
    let wants_json = spec
        .output_formats
        .contains(&ReplaySweepOutputFormat::JsonSummary)
        || !spec.output_formats.iter().any(|format| {
            matches!(
                format,
                ReplaySweepOutputFormat::JsonSummary | ReplaySweepOutputFormat::CsvRows
            )
        });
    if wants_json {
        write_json_atomic(&output_root.join("sweep-summary.json"), document)?;
    }
    if spec
        .output_formats
        .contains(&ReplaySweepOutputFormat::CsvRows)
    {
        write_bytes_atomic(
            &output_root.join("sweep-summary.csv"),
            &summary_csv(document),
        )?;
        if !document.fee_scenarios.is_empty() {
            write_bytes_atomic(
                &output_root.join("sweep-fee-scenarios.csv"),
                &fee_scenario_csv(document),
            )?;
        }
    }
    Ok(())
}

fn summary_csv(document: &ReplaySweepSummaryDocument) -> Vec<u8> {
    let mut output = String::from(
        "run_id,run_index,status,skipped,result_path,error,gross_pnl,net_pnl,fees,max_drawdown,trade_count,fill_count\n",
    );
    for run in &document.runs {
        let fields = [
            run.run_id.clone(),
            run.run_index.to_string(),
            run.status.clone(),
            run.skipped.to_string(),
            run.result_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            run.error.clone().unwrap_or_default(),
            optional_f64(run.gross_pnl),
            optional_f64(run.net_pnl),
            optional_f64(run.fees),
            optional_f64(run.max_drawdown),
            run.trade_count
                .map(|value| value.to_string())
                .unwrap_or_default(),
            run.fill_count
                .map(|value| value.to_string())
                .unwrap_or_default(),
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

fn fee_scenario_csv(document: &ReplaySweepSummaryDocument) -> Vec<u8> {
    let mut output = String::from(
        "run_id,run_index,scenario_name,currency,total_per_contract,fees,gross_pnl,net_pnl,ending_equity,return_on_initial_capital_pct,max_drawdown,max_drawdown_pct,profit_factor\n",
    );
    for scenario in &document.fee_scenarios {
        let fields = [
            scenario.run_id.clone(),
            scenario.run_index.to_string(),
            scenario.scenario_name.clone(),
            scenario.currency.clone(),
            scenario.total_per_contract.to_string(),
            scenario.fees.to_string(),
            scenario.gross_pnl.to_string(),
            scenario.net_pnl.to_string(),
            scenario.ending_equity.to_string(),
            optional_f64(scenario.return_on_initial_capital_pct),
            scenario.max_drawdown.to_string(),
            optional_f64(scenario.max_drawdown_pct),
            optional_f64(scenario.profit_factor),
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

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize sweep JSON")?;
    write_bytes_atomic(path, &bytes)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("sweep output path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create sweep output {}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::write(&temporary, bytes)
        .with_context(|| format!("write temporary sweep output {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replace sweep output {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{
        BarType, CandleMode, ReplayEngineMode, ReplayFillModel, ReplayLatencyConfig,
    };
    use crate::config::TradingEnvironment;
    use crate::replay_cache::{
        ReplayDatasetSessionPreset, ReplayDatasetSourceRef, ReplayDatasetWarmupPolicy,
    };
    use crate::strategy::ExecutionStrategyConfig;

    fn child() -> ReplaySweepChildSpec {
        let view = ReplayDatasetView {
            view_version: crate::replay_cache::REPLAY_DATASET_VIEW_VERSION,
            id: "runner-view".to_string(),
            source: ReplayDatasetSourceRef {
                manifest_id: "tradovate/sim/MES/MESU6/2026-07-01/manifest.json".to_string(),
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
            },
            evaluation_start: Utc::now(),
            evaluation_end: Utc::now() + chrono::Duration::minutes(1),
            input_timezone: "UTC".to_string(),
            session_preset: ReplayDatasetSessionPreset::FullSource,
            warmup: ReplayDatasetWarmupPolicy::default(),
        };
        ReplaySweepChildSpec {
            run_id: "runner-000001".to_string(),
            run_index: 0,
            parent_sweep_id: "runner".to_string(),
            resolved_strategy: ExecutionStrategyConfig::default(),
            parameter_values: Default::default(),
            overrides: Default::default(),
            base_dataset_view: view,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            engine_mode: ReplayEngineMode::Deterministic,
            fill_model: ReplayFillModel::RawBarOpen,
            latency: ReplayLatencyConfig::default(),
            bar_protection_policy: crate::broker::ReplayBarProtectionPolicy::Conservative,
            primary_fee_schedule: super::super::ReplayFeeSchedule::default(),
            fee_scenarios: Vec::new(),
            initial_capital: 10_000.0,
            margin: None,
        }
    }

    #[test]
    fn csv_summary_escapes_errors_and_preserves_numeric_columns() {
        let run = ReplaySweepRunSummary {
            run_id: "runner-000001".to_string(),
            run_index: 0,
            status: "failed".to_string(),
            skipped: false,
            result_path: None,
            error: Some("bad, \"input\"".to_string()),
            gross_pnl: Some(1.25),
            net_pnl: Some(1.0),
            fees: Some(0.25),
            max_drawdown: Some(-2.0),
            trade_count: Some(2),
            fill_count: Some(4),
        };
        let document = ReplaySweepSummaryDocument {
            schema_version: REPLAY_SWEEP_SUMMARY_SCHEMA_VERSION,
            sweep_id: "runner".to_string(),
            name: "Runner".to_string(),
            created_at_utc: Utc::now(),
            run_count: 1,
            completed_count: 0,
            failed_count: 1,
            skipped_count: 0,
            warnings: Vec::new(),
            runs: vec![run],
            fee_scenarios: Vec::new(),
            resource_estimate: None,
        };
        let csv = String::from_utf8(summary_csv(&document)).expect("csv");
        assert!(csv.contains("\"bad, \"\"input\"\"\""));
        assert!(csv.contains("1.25,1,0.25,-2,2,4"));
    }

    #[test]
    fn result_resume_requires_completed_status() {
        let path = std::env::temp_dir().join(format!(
            "trader-sweep-resume-{}.json",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::write(&path, br#"{"status":"failed"}"#).expect("write failed result");
        let child = child();
        let mut spec = ReplaySweepSpec::default();
        spec.sweep_id = child.parent_sweep_id.clone();
        spec.name = "Runner".to_string();
        spec.base_dataset_view = child.base_dataset_view.clone();
        spec.base_strategy = child.resolved_strategy.clone();
        spec.bar_type = child.bar_type;
        spec.candle_mode = child.candle_mode;
        spec.max_runs = 1;
        spec.output_dir = std::env::temp_dir().join("runner-sweep-output");
        assert!(!completed_result_matches(&path, &child, &spec).expect("inspect failed result"));
        fs::write(&path, br#"{"status":"completed"}"#).expect("write complete result");
        assert!(
            !completed_result_matches(&path, &child, &spec).expect("inspect incomplete result")
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn failed_summary_keeps_child_identity() {
        let child = child();
        let summary = ReplaySweepRunSummary::failed(&child, "boom");
        assert_eq!(summary.run_id, child.run_id);
        assert_eq!(summary.run_index, child.run_index);
        assert_eq!(summary.status, "failed");
    }

    #[test]
    fn fee_scenario_rows_are_flattened_from_child_results() {
        let path = std::env::temp_dir().join(format!(
            "trader-sweep-fees-{}.json",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::write(
            &path,
            br#"{
                "fee_scenarios": [{
                    "schedule": {
                        "name": "broker-standard",
                        "currency": "USD",
                        "commission_per_contract": 1.25,
                        "exchange_per_contract": 0.5
                    },
                    "fees": 3.5,
                    "gross_pnl": 10.0,
                    "net_pnl": 6.5,
                    "ending_equity": 10006.5,
                    "return_on_initial_capital_pct": 0.065,
                    "max_drawdown": -2.0,
                    "max_drawdown_pct": -0.02,
                    "profit_factor": 2.0
                }]
            }"#,
        )
        .expect("write fee result");
        let run = ReplaySweepRunSummary {
            run_id: "runner-000001".to_string(),
            run_index: 0,
            status: "completed".to_string(),
            skipped: false,
            result_path: Some(path.clone()),
            error: None,
            gross_pnl: Some(10.0),
            net_pnl: Some(6.5),
            fees: Some(3.5),
            max_drawdown: Some(-2.0),
            trade_count: Some(1),
            fill_count: Some(2),
        };
        let mut warnings = Vec::new();
        let rows = collect_fee_scenario_summaries(&[run], &mut warnings);
        assert!(warnings.is_empty());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].scenario_name, "broker-standard");
        assert!((rows[0].total_per_contract - 1.75).abs() < f64::EPSILON);
        assert!((rows[0].net_pnl - 6.5).abs() < f64::EPSILON);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn launch_guardrails_require_confirmation_and_support_explicit_override() {
        let report = ReplaySweepGuardrailReport {
            estimate: ReplaySweepResourceEstimate {
                combinations: 101,
                parallel_jobs: 8,
                estimated_input_rows: Some(20_000_001),
                estimated_input_rows_per_worker: Some(20_000_001),
                input_source: "cached raw ticks".to_string(),
                estimated_memory_bytes: Some(9 * 1024 * 1024 * 1024),
                estimated_output_bytes: Some(65 * 1024 * 1024 * 1024),
                estimated_runtime_seconds: Some(3600),
                notes: Vec::new(),
            },
            violations: vec!["cache rows exceeded".to_string()],
            warnings: vec!["large sweep".to_string()],
            requires_confirmation: true,
        };
        assert!(enforce_guardrails(&report, false, false).is_err());
        assert!(enforce_guardrails(&report, true, false).is_err());
        assert!(enforce_guardrails(&report, false, true).is_ok());
    }
}
