use anyhow::{Context, Result, bail};
use clap::Parser;
use midas_env::env::MarginMode;
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "train_ga/actions.rs"]
mod actions;
#[path = "train_ga/config.rs"]
mod config;
#[path = "train_ga/data.rs"]
mod data;
#[path = "train_ga/metrics.rs"]
mod metrics;
#[path = "train_ga/portable.rs"]
mod portable;
#[path = "train_ga/types.rs"]
mod types;

#[cfg(feature = "backend-candle")]
#[path = "train_ga/backends/candle.rs"]
mod candle_backend;

#[derive(Parser, Debug)]
#[command(about = "Replay a saved GA portable policy on a parquet file and dump behavior")]
struct Args {
    #[arg(long)]
    run_summary: Option<PathBuf>,
    #[arg(long)]
    policy: Option<PathBuf>,
    #[arg(long)]
    parquet: Option<PathBuf>,
    #[arg(long)]
    outdir: Option<PathBuf>,
    #[arg(long)]
    windowed: bool,
    #[arg(long)]
    window: Option<usize>,
    #[arg(long)]
    step: Option<usize>,
    #[arg(long)]
    eval_windows: Option<usize>,
    #[arg(long, default_value_t = true)]
    globex: bool,
    #[arg(long)]
    rth: bool,
    #[arg(long, default_value = "config/symbols.yaml")]
    symbol_config: PathBuf,
}

#[derive(Serialize)]
struct ReplayMetrics {
    fitness: f64,
    eval_pnl: f64,
    eval_pnl_realized: f64,
    eval_pnl_total: f64,
    eval_sortino: f64,
    eval_drawdown: f64,
    eval_ret_mean: f64,
    debug_non_hold: usize,
    debug_non_zero_pos: usize,
    debug_mean_abs_pnl: f64,
    debug_buy: usize,
    debug_sell: usize,
    debug_hold: usize,
    debug_revert: usize,
    debug_session_violations: usize,
    debug_margin_violations: usize,
    debug_position_violations: usize,
    debug_drawdown_penalty: f64,
    debug_invalid_revert_penalty: f64,
    debug_hold_duration_penalty: f64,
    debug_flat_hold_penalty: f64,
    debug_session_close_penalty: f64,
}

#[derive(Default, Serialize)]
struct BehaviorStats {
    rows: usize,
    windows: usize,
    target_short_steps: usize,
    target_flat_steps: usize,
    target_long_steps: usize,
    effective_buy_steps: usize,
    effective_sell_steps: usize,
    effective_hold_steps: usize,
    effective_revert_steps: usize,
    flat_steps: usize,
    long_steps: usize,
    short_steps: usize,
    entries: usize,
    exits: usize,
    flips: usize,
    auto_closes: usize,
    session_closed_violations: usize,
    margin_call_violations: usize,
    position_limit_violations: usize,
    winning_exits: usize,
    losing_exits: usize,
    total_reward: f64,
    total_realized_pnl_change: f64,
    total_pnl_change: f64,
    total_commission_paid: f64,
    total_slippage_paid: f64,
    avg_bars_per_trade: f64,
    max_bars_per_trade: usize,
    pct_flat: f64,
}

#[derive(Serialize)]
struct ReplaySummary {
    generated_at: String,
    run_summary: Option<String>,
    policy: String,
    parquet: String,
    outdir: String,
    full_file: bool,
    window: usize,
    step: usize,
    eval_windows: usize,
    metrics: ReplayMetrics,
    behavior: BehaviorStats,
}

#[derive(Default)]
struct TradeTracker {
    entry_step: Option<usize>,
}

#[derive(Clone, Copy)]
struct ReplayConfig {
    initial_balance: f64,
    max_position: i32,
    margin_mode: MarginMode,
    contract_multiplier: f64,
    margin_per_contract: f64,
    disable_margin: bool,
    w_pnl: f64,
    w_sortino: f64,
    w_mdd: f64,
    sortino_annualization: f64,
    drawdown_penalty: f64,
    drawdown_penalty_growth: f64,
    session_close_penalty: f64,
    auto_close_minutes_before_close: f64,
    max_hold_bars_positive: usize,
    max_hold_bars_drawdown: usize,
    hold_duration_penalty: f64,
    hold_duration_penalty_growth: f64,
    hold_duration_penalty_positive_scale: f64,
    hold_duration_penalty_negative_scale: f64,
    min_hold_bars: usize,
    early_exit_penalty: f64,
    early_flip_penalty: f64,
    invalid_revert_penalty: f64,
    invalid_revert_penalty_growth: f64,
    flat_hold_penalty: f64,
    flat_hold_penalty_growth: f64,
    max_flat_hold_bars: usize,
}

fn main() -> Result<()> {
    #[cfg(feature = "backend-candle")]
    {
        return run();
    }
    #[cfg(not(feature = "backend-candle"))]
    {
        bail!(
            "inspect_ga_policy requires the 'backend-candle' Cargo feature. Re-run with --features backend-candle."
        )
    }
}

#[cfg(feature = "backend-candle")]
fn run() -> Result<()> {
    let args = Args::parse();

    let run_summary_json = if let Some(path) = args.run_summary.as_ref() {
        Some(read_json(path)?)
    } else {
        None
    };

    let run_dir = run_summary_json
        .as_ref()
        .and_then(|json| json_string(json, "runDir"))
        .map(PathBuf::from);
    let params = run_summary_json.as_ref().and_then(|json| json.get("params"));

    let policy_path = if let Some(path) = args.policy.as_ref() {
        path.clone()
    } else if let Some(dir) = run_dir.as_ref() {
        dir.join("best_overall_policy.portable.json")
    } else {
        bail!("provide either --policy or --run-summary");
    };

    let parquet_path = if let Some(path) = args.parquet.as_ref() {
        path.clone()
    } else if let Some(params) = params {
        PathBuf::from(
            json_string(params, "test-parquet")
                .context("run summary params missing 'test-parquet'")?,
        )
    } else {
        bail!("provide either --parquet or --run-summary with params.test-parquet");
    };

    let outdir = if let Some(path) = args.outdir.as_ref() {
        path.clone()
    } else if let Some(dir) = run_dir.as_ref() {
        dir.join("inspection_test_full")
    } else {
        PathBuf::from("runs_ga/inspection_test_full")
    };
    std::fs::create_dir_all(&outdir)?;

    let loaded_policy = portable::load_policy_json(&policy_path)
        .with_context(|| format!("load portable policy {}", policy_path.display()))?;

    let default_session = args.globex && !args.rth;
    let dataset = data::load_dataset(&parquet_path, default_session)
        .with_context(|| format!("load dataset {}", parquet_path.display()))?;
    let (margin_cfg, session_cfg) = load_symbol_config(&args.symbol_config, &dataset.symbol)?;
    let use_globex = if let Some(session) = session_cfg {
        match session.as_str() {
            "rth" => false,
            "globex" => true,
            _ => !args.rth,
        }
    } else {
        !args.rth
    };
    let dataset = dataset.with_session(use_globex);

    if loaded_policy.input_dim != dataset.obs_dim {
        bail!(
            "portable policy input_dim {} does not match dataset obs_dim {}",
            loaded_policy.input_dim,
            dataset.obs_dim
        );
    }

    let replay_cfg = build_replay_config(params, &dataset.symbol, margin_cfg)?;
    let full_file = !args.windowed;
    let window = args
        .window
        .or_else(|| params.and_then(|value| json_usize(value, "window")))
        .unwrap_or(700);
    let step = args
        .step
        .or_else(|| params.and_then(|value| json_usize(value, "step")))
        .unwrap_or(128);
    let requested_eval_windows = args
        .eval_windows
        .or_else(|| params.and_then(|value| json_usize(value, "eval-windows")))
        .unwrap_or(usize::MAX);
    let windows = build_windows(&dataset, full_file, window, step)?;
    let eval_windows = if full_file {
        1
    } else {
        requested_eval_windows.min(windows.len())
    };

    let cfg = config::CandidateConfig {
        initial_balance: replay_cfg.initial_balance,
        max_position: replay_cfg.max_position,
        margin_mode: replay_cfg.margin_mode,
        contract_multiplier: replay_cfg.contract_multiplier,
        margin_per_contract: replay_cfg.margin_per_contract,
        disable_margin: replay_cfg.disable_margin,
        w_pnl: replay_cfg.w_pnl,
        w_sortino: replay_cfg.w_sortino,
        w_mdd: replay_cfg.w_mdd,
        sortino_annualization: replay_cfg.sortino_annualization,
        hidden: loaded_policy.hidden_dim,
        layers: loaded_policy.hidden_layers,
        eval_windows,
        device: config::ExecutionTarget::Cpu,
        ignore_session: false,
        drawdown_penalty: replay_cfg.drawdown_penalty,
        drawdown_penalty_growth: replay_cfg.drawdown_penalty_growth,
        session_close_penalty: replay_cfg.session_close_penalty,
        auto_close_minutes_before_close: replay_cfg.auto_close_minutes_before_close,
        max_hold_bars_positive: replay_cfg.max_hold_bars_positive,
        max_hold_bars_drawdown: replay_cfg.max_hold_bars_drawdown,
        hold_duration_penalty: replay_cfg.hold_duration_penalty,
        hold_duration_penalty_growth: replay_cfg.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: replay_cfg.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: replay_cfg.hold_duration_penalty_negative_scale,
        min_hold_bars: replay_cfg.min_hold_bars,
        early_exit_penalty: replay_cfg.early_exit_penalty,
        early_flip_penalty: replay_cfg.early_flip_penalty,
        invalid_revert_penalty: replay_cfg.invalid_revert_penalty,
        invalid_revert_penalty_growth: replay_cfg.invalid_revert_penalty_growth,
        flat_hold_penalty: replay_cfg.flat_hold_penalty,
        flat_hold_penalty_growth: replay_cfg.flat_hold_penalty_growth,
        max_flat_hold_bars: replay_cfg.max_flat_hold_bars,
    };

    let (metrics, history) = candle_backend::evaluate_candidate_with_history(
        &loaded_policy.genome,
        &dataset,
        &windows,
        &cfg,
    )?;

    let behavior_csv_path = outdir.join("behavior.csv");
    write_behavior_csv(&behavior_csv_path, &dataset, &history)?;

    let summary = ReplaySummary {
        generated_at: chrono::Utc::now().to_rfc3339(),
        run_summary: args
            .run_summary
            .as_ref()
            .map(|path| path.display().to_string()),
        policy: policy_path.display().to_string(),
        parquet: parquet_path.display().to_string(),
        outdir: outdir.display().to_string(),
        full_file,
        window,
        step,
        eval_windows,
        metrics: ReplayMetrics {
            fitness: metrics.fitness,
            eval_pnl: metrics.eval_pnl,
            eval_pnl_realized: metrics.eval_pnl_realized,
            eval_pnl_total: metrics.eval_pnl_total,
            eval_sortino: metrics.eval_sortino,
            eval_drawdown: metrics.eval_drawdown,
            eval_ret_mean: metrics.eval_ret_mean,
            debug_non_hold: metrics.debug_non_hold,
            debug_non_zero_pos: metrics.debug_non_zero_pos,
            debug_mean_abs_pnl: metrics.debug_mean_abs_pnl,
            debug_buy: metrics.debug_buy,
            debug_sell: metrics.debug_sell,
            debug_hold: metrics.debug_hold,
            debug_revert: metrics.debug_revert,
            debug_session_violations: metrics.debug_session_violations,
            debug_margin_violations: metrics.debug_margin_violations,
            debug_position_violations: metrics.debug_position_violations,
            debug_drawdown_penalty: metrics.debug_drawdown_penalty,
            debug_invalid_revert_penalty: metrics.debug_invalid_revert_penalty,
            debug_hold_duration_penalty: metrics.debug_hold_duration_penalty,
            debug_flat_hold_penalty: metrics.debug_flat_hold_penalty,
            debug_session_close_penalty: metrics.debug_session_close_penalty,
        },
        behavior: summarize_behavior(&history),
    };

    let summary_path = outdir.join("replay_summary.json");
    std::fs::write(&summary_path, format!("{}\n", serde_json::to_string_pretty(&summary)?))?;

    println!(
        "Replay complete: pnl {:.2}, sortino {:.2}, mdd {:.2}, entries {}, exits {}, pct_flat {:.2}%",
        summary.metrics.eval_pnl_total,
        summary.metrics.eval_sortino,
        summary.metrics.eval_drawdown,
        summary.behavior.entries,
        summary.behavior.exits,
        summary.behavior.pct_flat * 100.0
    );
    println!("Behavior CSV: {}", behavior_csv_path.display());
    println!("Replay summary: {}", summary_path.display());
    Ok(())
}

#[cfg(feature = "backend-candle")]
fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(feature = "backend-candle")]
fn json_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str()
}

#[cfg(feature = "backend-candle")]
fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

#[cfg(feature = "backend-candle")]
fn json_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64()
}

#[cfg(feature = "backend-candle")]
fn json_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)?
        .as_u64()
        .and_then(|parsed| usize::try_from(parsed).ok())
}

#[cfg(feature = "backend-candle")]
fn json_i32(value: &Value, key: &str) -> Option<i32> {
    value
        .get(key)?
        .as_i64()
        .and_then(|parsed| i32::try_from(parsed).ok())
}

#[cfg(feature = "backend-candle")]
fn build_replay_config(
    params: Option<&Value>,
    symbol: &str,
    margin_cfg: Option<f64>,
) -> Result<ReplayConfig> {
    let params = params.unwrap_or(&Value::Null);
    let margin_mode = match json_string(params, "margin-mode").unwrap_or("auto") {
            "per-contract" => MarginMode::PerContract,
            "price" => MarginMode::Price,
            _ => infer_margin_mode(symbol, margin_cfg),
        };
    let contract_multiplier = json_f64(params, "contract-multiplier").unwrap_or(1.0);
    let margin_per_contract = json_f64(params, "margin-per-contract")
        .or(margin_cfg)
        .unwrap_or_else(|| infer_margin(symbol));

    Ok(ReplayConfig {
        initial_balance: json_f64(params, "initial-balance").unwrap_or(10000.0),
        max_position: json_i32(params, "max-position").unwrap_or(1),
        margin_mode,
        contract_multiplier,
        margin_per_contract,
        disable_margin: json_bool(params, "disable-margin").unwrap_or(false),
        w_pnl: json_f64(params, "w-pnl").unwrap_or(1.0),
        w_sortino: json_f64(params, "w-sortino").unwrap_or(1.0),
        w_mdd: json_f64(params, "w-mdd").unwrap_or(0.5),
        sortino_annualization: json_f64(params, "sortino-annualization").unwrap_or(1.0),
        drawdown_penalty: json_f64(params, "drawdown-penalty").unwrap_or(0.0),
        drawdown_penalty_growth: json_f64(params, "drawdown-penalty-growth").unwrap_or(0.0),
        session_close_penalty: json_f64(params, "session-close-penalty").unwrap_or(0.0),
        auto_close_minutes_before_close: json_f64(params, "auto-close-minutes-before-close")
            .unwrap_or(5.0),
        max_hold_bars_positive: json_usize(params, "max-hold-bars-positive").unwrap_or(15),
        max_hold_bars_drawdown: json_usize(params, "max-hold-bars-drawdown").unwrap_or(15),
        hold_duration_penalty: json_f64(params, "hold-duration-penalty").unwrap_or(1.0),
        hold_duration_penalty_growth: json_f64(params, "hold-duration-penalty-growth")
            .unwrap_or(0.05),
        hold_duration_penalty_positive_scale: json_f64(
            params,
            "hold-duration-penalty-positive-scale",
        )
        .unwrap_or(0.5),
        hold_duration_penalty_negative_scale: json_f64(
            params,
            "hold-duration-penalty-negative-scale",
        )
        .unwrap_or(1.5),
        min_hold_bars: json_usize(params, "min-hold-bars").unwrap_or(0),
        early_exit_penalty: json_f64(params, "early-exit-penalty").unwrap_or(0.0),
        early_flip_penalty: json_f64(params, "early-flip-penalty").unwrap_or(0.0),
        invalid_revert_penalty: json_f64(params, "invalid-revert-penalty").unwrap_or(7.0),
        invalid_revert_penalty_growth: json_f64(params, "invalid-revert-penalty-growth")
            .unwrap_or(0.5),
        flat_hold_penalty: json_f64(params, "flat-hold-penalty").unwrap_or(2.2),
        flat_hold_penalty_growth: json_f64(params, "flat-hold-penalty-growth").unwrap_or(0.05),
        max_flat_hold_bars: json_usize(params, "max-flat-hold-bars").unwrap_or(100),
    })
}

#[cfg(feature = "backend-candle")]
fn build_windows(
    dataset: &data::DataSet,
    full_file: bool,
    window: usize,
    step: usize,
) -> Result<Vec<(usize, usize)>> {
    let raw_windows = if full_file {
        vec![(0, dataset.close.len())]
    } else {
        midas_env::sampler::windows(dataset.close.len(), window, step)
    };
    let min_start = midas_env::features::feature_warmup_bars().saturating_sub(1);
    let windows = midas_env::sampler::enforce_min_start(&raw_windows, min_start);
    if windows.is_empty() {
        bail!(
            "no replay windows available after applying feature warmup to dataset of length {}",
            dataset.close.len()
        );
    }
    Ok(windows)
}

#[cfg(feature = "backend-candle")]
fn load_symbol_config(path: &Path, symbol: &str) -> Result<(Option<f64>, Option<String>)> {
    if !path.exists() {
        return Ok((None, None));
    }
    let text = std::fs::read_to_string(path)?;
    let cfg: serde_yaml::Value = serde_yaml::from_str(&text)?;
    if let Some(entry) = cfg.get(symbol) {
        let margin = entry.get("margin_per_contract").and_then(|v| v.as_f64());
        let session = entry
            .get("session")
            .and_then(|v| v.as_str())
            .map(|value| value.to_ascii_lowercase());
        Ok((margin, session))
    } else {
        Ok((None, None))
    }
}

#[cfg(feature = "backend-candle")]
fn infer_margin(symbol: &str) -> f64 {
    if is_futures_symbol(symbol) {
        let sym = symbol.to_ascii_uppercase();
        if sym.contains("MES") {
            return 50.0;
        }
        return 500.0;
    }
    100.0
}

#[cfg(feature = "backend-candle")]
fn infer_margin_mode(symbol: &str, margin_cfg: Option<f64>) -> MarginMode {
    if margin_cfg.is_some() || is_futures_symbol(symbol) {
        MarginMode::PerContract
    } else {
        MarginMode::Price
    }
}

#[cfg(feature = "backend-candle")]
fn is_futures_symbol(symbol: &str) -> bool {
    let sym = symbol.to_ascii_uppercase();
    sym.contains("MES") || sym == "ES" || sym.contains("ES@") || sym.ends_with("ES")
}

#[cfg(feature = "backend-candle")]
fn summarize_behavior(rows: &[types::BehaviorRow]) -> BehaviorStats {
    let mut out = BehaviorStats {
        rows: rows.len(),
        windows: rows.iter().map(|row| row.window_idx).max().map(|v| v + 1).unwrap_or(0),
        ..BehaviorStats::default()
    };
    let mut tracker = TradeTracker::default();
    let mut total_trade_bars = 0usize;

    for row in rows {
        match row.action.as_str() {
            "short" => out.target_short_steps += 1,
            "flat" => out.target_flat_steps += 1,
            "long" => out.target_long_steps += 1,
            _ => {}
        }
        match row.effective_action.as_str() {
            "buy" => out.effective_buy_steps += 1,
            "sell" => out.effective_sell_steps += 1,
            "hold" => out.effective_hold_steps += 1,
            "revert" => out.effective_revert_steps += 1,
            _ => {}
        }
        match row.position_after.cmp(&0) {
            std::cmp::Ordering::Less => out.short_steps += 1,
            std::cmp::Ordering::Equal => out.flat_steps += 1,
            std::cmp::Ordering::Greater => out.long_steps += 1,
        }
        if row.position_before == 0 && row.position_after != 0 {
            out.entries += 1;
            tracker.entry_step = Some(row.step);
        }
        if row.position_before != 0 && row.position_after == 0 {
            out.exits += 1;
            if row.realized_pnl_change > 0.0 {
                out.winning_exits += 1;
            } else if row.realized_pnl_change < 0.0 {
                out.losing_exits += 1;
            }
            if let Some(entry_step) = tracker.entry_step.take() {
                let bars = row.step.saturating_sub(entry_step) + 1;
                total_trade_bars += bars;
                if bars > out.max_bars_per_trade {
                    out.max_bars_per_trade = bars;
                }
            }
        }
        if row.position_before.signum() != row.position_after.signum()
            && row.position_before != 0
            && row.position_after != 0
        {
            out.flips += 1;
        }

        out.auto_closes += usize::from(row.auto_close_executed);
        out.session_closed_violations += usize::from(row.session_closed_violation);
        out.margin_call_violations += usize::from(row.margin_call_violation);
        out.position_limit_violations += usize::from(row.position_limit_violation);
        out.total_reward += row.reward;
        out.total_realized_pnl_change += row.realized_pnl_change;
        out.total_pnl_change += row.pnl_change;
        out.total_commission_paid += row.commission_paid;
        out.total_slippage_paid += row.slippage_paid;
    }

    if out.exits > 0 {
        out.avg_bars_per_trade = total_trade_bars as f64 / out.exits as f64;
    }
    if out.rows > 0 {
        out.pct_flat = out.flat_steps as f64 / out.rows as f64;
    }
    out
}

#[cfg(feature = "backend-candle")]
fn write_behavior_csv(
    path: &Path,
    data: &data::DataSet,
    rows: &[types::BehaviorRow],
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "window_idx,step,data_idx,date,open,close,volume,action,effective_action,action_idx,position_before,position_after,equity_before,equity_after,cash,unrealized_pnl,realized_pnl,pnl_change,realized_pnl_change,reward,commission_paid,slippage_paid,drawdown_penalty,session_close_penalty,invalid_revert_penalty,hold_duration_penalty,flat_hold_penalty,auto_close_executed,session_open,margin_ok,minutes_to_close,session_closed_violation,margin_call_violation,position_limit_violation"
    )?;

    for row in rows {
        let idx = row.data_idx;
        let date = data
            .datetime_ns
            .as_ref()
            .and_then(|d| d.get(idx))
            .map(|ts| ts.to_string())
            .unwrap_or_default();
        let open = data
            .open
            .get(idx)
            .map(|value| value.to_string())
            .unwrap_or_default();
        let close = data
            .close
            .get(idx)
            .map(|value| value.to_string())
            .unwrap_or_default();
        let volume = data
            .volume
            .as_ref()
            .and_then(|values| values.get(idx))
            .map(|value| value.to_string())
            .unwrap_or_default();
        let columns = [
            row.window_idx.to_string(),
            row.step.to_string(),
            row.data_idx.to_string(),
            date,
            open,
            close,
            volume,
            row.action.clone(),
            row.effective_action.clone(),
            row.action_idx.to_string(),
            row.position_before.to_string(),
            row.position_after.to_string(),
            row.equity_before.to_string(),
            row.equity_after.to_string(),
            row.cash.to_string(),
            row.unrealized_pnl.to_string(),
            row.realized_pnl.to_string(),
            row.pnl_change.to_string(),
            row.realized_pnl_change.to_string(),
            row.reward.to_string(),
            row.commission_paid.to_string(),
            row.slippage_paid.to_string(),
            row.drawdown_penalty.to_string(),
            row.session_close_penalty.to_string(),
            row.invalid_revert_penalty.to_string(),
            row.hold_duration_penalty.to_string(),
            row.flat_hold_penalty.to_string(),
            row.auto_close_executed.to_string(),
            row.session_open.to_string(),
            row.margin_ok.to_string(),
            row.minutes_to_close.map(|value| value.to_string()).unwrap_or_default(),
            row.session_closed_violation.to_string(),
            row.margin_call_violation.to_string(),
            row.position_limit_violation.to_string(),
        ];
        writeln!(file, "{}", columns.join(","))?;
    }

    Ok(())
}
