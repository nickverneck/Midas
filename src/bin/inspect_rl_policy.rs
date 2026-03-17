use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap, ops::softmax};
use clap::Parser;
use midas_env::env::{Action, EnvConfig, MarginMode, StepContext, TradingEnv};
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "train_rl/args.rs"]
mod args;
#[path = "train_rl/candle.rs"]
mod candle_backend;
#[path = "train_rl/common.rs"]
mod common;
#[path = "train_rl/data.rs"]
mod data;
#[path = "train_rl/metrics.rs"]
mod metrics;

#[derive(Parser, Debug)]
#[command(about = "Replay a saved RL Candle checkpoint on a parquet file and dump behavior")]
struct Args {
    #[arg(long)]
    run_summary: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "fitness", value_parser = ["fitness", "eval-pnl", "final"])]
    checkpoint_metric: String,
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
    #[arg(long)]
    sample: bool,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = true)]
    globex: bool,
    #[arg(long)]
    rth: bool,
    #[arg(long, default_value = "config/symbols.yaml")]
    symbol_config: PathBuf,
}

#[derive(Default, Clone)]
struct ReplayConfig {
    initial_balance: f64,
    max_position: i32,
    margin_mode: MarginMode,
    contract_multiplier: f64,
    margin_per_contract: f64,
    disable_margin: bool,
    ignore_session: bool,
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
    hidden: usize,
    layers: usize,
}

#[derive(Default, Clone, Serialize)]
struct ReplayMetrics {
    fitness: f64,
    pnl: f64,
    sortino: f64,
    drawdown: f64,
    ret_mean: f64,
}

#[derive(Clone)]
struct BehaviorRow {
    window_idx: usize,
    step: usize,
    data_idx: usize,
    action_idx: i32,
    action: String,
    effective_action: String,
    position_before: i32,
    position_after: i32,
    equity_before: f64,
    equity_after: f64,
    cash: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
    pnl_change: f64,
    realized_pnl_change: f64,
    reward: f64,
    commission_paid: f64,
    slippage_paid: f64,
    drawdown_penalty: f64,
    session_close_penalty: f64,
    invalid_revert_penalty: f64,
    hold_duration_penalty: f64,
    flat_hold_penalty: f64,
    auto_close_executed: bool,
    session_open: bool,
    margin_ok: bool,
    minutes_to_close: Option<f64>,
    session_closed_violation: bool,
    margin_call_violation: bool,
    position_limit_violation: bool,
}

#[derive(Default, Serialize)]
struct BehaviorStats {
    rows: usize,
    windows: usize,
    requested_buy_steps: usize,
    requested_sell_steps: usize,
    requested_hold_steps: usize,
    requested_revert_steps: usize,
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
    checkpoint: String,
    checkpoint_metric: String,
    parquet: String,
    outdir: String,
    full_file: bool,
    window: usize,
    step: usize,
    eval_windows: usize,
    sample: bool,
    seed: Option<u64>,
    metrics: ReplayMetrics,
    behavior: BehaviorStats,
}

#[derive(Default)]
struct TradeTracker {
    entry_step: Option<usize>,
}

fn main() -> Result<()> {
    #[cfg(feature = "backend-candle")]
    {
        return run();
    }
    #[cfg(not(feature = "backend-candle"))]
    {
        bail!(
            "inspect_rl_policy requires the 'backend-candle' Cargo feature. Re-run with --features backend-candle."
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

    let algorithm = params
        .and_then(|value| json_string(value, "algorithm"))
        .unwrap_or("ppo");
    let checkpoint_path = resolve_checkpoint_path(
        args.checkpoint.as_ref(),
        run_dir.as_ref(),
        run_summary_json.as_ref(),
        algorithm,
        &args.checkpoint_metric,
    )?;

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
        PathBuf::from("runs_rl/inspection_test_full")
    };
    std::fs::create_dir_all(&outdir)?;

    let default_session = args.globex && !args.rth;
    let dataset = data::load_dataset(&parquet_path, default_session)
        .with_context(|| format!("load dataset {}", parquet_path.display()))?;
    let (margin_cfg, session_cfg) = common::load_symbol_config(&args.symbol_config, &dataset.symbol)?;
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

    let replay_cfg = build_replay_config(params, &dataset.symbol, margin_cfg)?;
    if replay_cfg.hidden == 0 || replay_cfg.layers == 0 {
        bail!("run summary params missing valid hidden/layers for RL replay");
    }

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

    let mut varmap = VarMap::new();
    let device = Device::Cpu;
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let policy = candle_backend::build_policy(
        var_builder.pp("policy"),
        dataset.obs_dim,
        replay_cfg.hidden,
        replay_cfg.layers,
        4,
    )?;
    candle_backend::load_checkpoint(&mut varmap, &checkpoint_path)?;

    let env_cfg = EnvConfig {
        max_position: replay_cfg.max_position,
        margin_mode: replay_cfg.margin_mode,
        contract_multiplier: replay_cfg.contract_multiplier,
        margin_per_contract: replay_cfg.margin_per_contract,
        enforce_margin: !replay_cfg.disable_margin,
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
        ..EnvConfig::default()
    };

    let seed = if args.sample {
        Some(
            args.seed
                .or_else(|| params.and_then(|value| json_u64(value, "seed")))
                .unwrap_or_else(|| rand::thread_rng().r#gen()),
        )
    } else {
        args.seed.or_else(|| params.and_then(|value| json_u64(value, "seed")))
    };
    let mut rng = seed.map(rand::rngs::StdRng::seed_from_u64);
    let (metrics, history) = replay_policy(
        &dataset,
        &windows,
        eval_windows,
        &policy,
        &env_cfg,
        &replay_cfg,
        args.sample,
        rng.as_mut(),
        &device,
    )?;

    let behavior_csv_path = outdir.join("behavior.csv");
    write_behavior_csv(&behavior_csv_path, &dataset, &history)?;

    let summary = ReplaySummary {
        generated_at: chrono::Utc::now().to_rfc3339(),
        run_summary: args
            .run_summary
            .as_ref()
            .map(|path| path.display().to_string()),
        checkpoint: checkpoint_path.display().to_string(),
        checkpoint_metric: args.checkpoint_metric.clone(),
        parquet: parquet_path.display().to_string(),
        outdir: outdir.display().to_string(),
        full_file,
        window,
        step,
        eval_windows,
        sample: args.sample,
        seed,
        metrics,
        behavior: summarize_behavior(&history),
    };

    let summary_path = outdir.join("replay_summary.json");
    std::fs::write(&summary_path, format!("{}\n", serde_json::to_string_pretty(&summary)?))?;

    println!(
        "Replay complete: pnl {:.2}, sortino {:.2}, mdd {:.2}, entries {}, exits {}, pct_flat {:.2}%",
        summary.metrics.pnl,
        summary.metrics.sortino,
        summary.metrics.drawdown,
        summary.behavior.entries,
        summary.behavior.exits,
        summary.behavior.pct_flat * 100.0
    );
    println!("Behavior CSV: {}", behavior_csv_path.display());
    println!("Replay summary: {}", summary_path.display());
    Ok(())
}

#[cfg(feature = "backend-candle")]
fn replay_policy(
    data: &data::DataSet,
    windows: &[(usize, usize)],
    eval_windows: usize,
    policy: &candle_nn::Sequential,
    env_cfg: &EnvConfig,
    cfg: &ReplayConfig,
    sample: bool,
    mut rng: Option<&mut rand::rngs::StdRng>,
    device: &Device,
) -> Result<(ReplayMetrics, Vec<BehaviorRow>)> {
    let mut history = Vec::new();
    let mut pnls = Vec::new();
    let mut drawdowns = Vec::new();
    let mut all_returns = Vec::new();

    for (window_idx, &(start, end)) in windows.iter().take(eval_windows).enumerate() {
        if end <= start + 1 {
            continue;
        }
        let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
        let mut position = 0i32;
        let mut equity = cfg.initial_balance;
        let mut prev_equity = cfg.initial_balance;
        let mut pnl_total = 0.0;
        let mut return_series = Vec::with_capacity(end - start - 1);
        let mut equity_curve = Vec::with_capacity(end - start - 1);

        for t in (start + 1)..end {
            let position_before = position;
            let equity_before = equity;
            let obs = data::build_observation(
                data,
                t,
                position,
                equity,
                env.state().unrealized_pnl,
                env.state().realized_pnl,
                cfg.initial_balance,
            );
            let obs_tensor = Tensor::from_vec(obs, (1, data.obs_dim), device)?;
            let logits = policy.forward(&obs_tensor)?;
            let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
            let action_idx = if sample {
                sample_from_probs(
                    &probs,
                    rng.as_deref_mut().context("sample replay requested without RNG seed")?,
                )
            } else {
                argmax_index(&probs)
            };
            let action = action_from_index(action_idx);
            let action_name = action_label(action).to_string();

            let session_open = if cfg.ignore_session {
                true
            } else {
                data.session_open
                    .as_ref()
                    .and_then(|m| m.get(t))
                    .copied()
                    .unwrap_or(true)
            };
            let minutes_to_close = data
                .minutes_to_close
                .as_ref()
                .and_then(|m| m.get(t))
                .copied();
            let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);

            let (reward, info) = env.step(
                action,
                data.close[t],
                StepContext {
                    session_open,
                    margin_ok,
                    minutes_to_close,
                },
            );
            position = env.state().position;
            equity = env.state().cash + env.state().unrealized_pnl;
            pnl_total += info.pnl_change;

            let denom = if prev_equity.abs() < 1e-8 {
                1e-8
            } else {
                prev_equity
            };
            return_series.push(info.pnl_change / denom);
            equity_curve.push(equity);
            prev_equity = equity;

            history.push(BehaviorRow {
                window_idx,
                step: t - start - 1,
                data_idx: t,
                action_idx: action_idx as i32,
                action: action_name,
                effective_action: action_label(info.effective_action).to_string(),
                position_before,
                position_after: position,
                equity_before,
                equity_after: equity,
                cash: env.state().cash,
                unrealized_pnl: env.state().unrealized_pnl,
                realized_pnl: env.state().realized_pnl,
                pnl_change: info.pnl_change,
                realized_pnl_change: info.realized_pnl_change,
                reward,
                commission_paid: info.commission_paid,
                slippage_paid: info.slippage_paid,
                drawdown_penalty: info.drawdown_penalty,
                session_close_penalty: info.session_close_penalty,
                invalid_revert_penalty: info.invalid_revert_penalty,
                hold_duration_penalty: info.hold_duration_penalty,
                flat_hold_penalty: info.flat_hold_penalty,
                auto_close_executed: info.auto_close_executed,
                session_open,
                margin_ok,
                minutes_to_close,
                session_closed_violation: info.session_closed_violation,
                margin_call_violation: info.margin_call_violation,
                position_limit_violation: info.position_limit_violation,
            });
        }

        pnls.push(pnl_total);
        all_returns.extend(return_series.iter().copied());
        drawdowns.push(metrics::max_drawdown(&equity_curve));
    }

    let pnl = average_f64(&pnls);
    let ret_mean = average_f64(&all_returns);
    let drawdown = average_f64(&drawdowns);
    let sortino = metrics::compute_sortino(&all_returns, cfg.sortino_annualization, 0.0, 50.0);
    let fitness = (cfg.w_pnl * pnl) + (cfg.w_sortino * sortino) - (cfg.w_mdd * drawdown);

    Ok((
        ReplayMetrics {
            fitness,
            pnl,
            sortino,
            drawdown,
            ret_mean,
        },
        history,
    ))
}

#[cfg(feature = "backend-candle")]
fn resolve_checkpoint_path(
    explicit: Option<&PathBuf>,
    run_dir: Option<&PathBuf>,
    run_summary: Option<&Value>,
    algorithm: &str,
    metric: &str,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.clone());
    }

    let run_dir = run_dir.context("provide either --checkpoint or --run-summary")?;
    if metric == "final" {
        let final_name = if algorithm == "grpo" {
            "grpo_final.safetensors"
        } else {
            "ppo_final.safetensors"
        };
        return Ok(run_dir.join(final_name));
    }

    let run_summary = run_summary.context("run summary JSON unavailable")?;
    let checkpoint_epoch = if metric == "eval-pnl" {
        run_summary
            .get("bestEvalPnlRow")
            .and_then(|row| json_usize(row, "epoch"))
    } else {
        run_summary
            .get("bestRow")
            .and_then(|row| json_usize(row, "epoch"))
    };
    if let Some(epoch) = checkpoint_epoch {
        let candidate = run_dir.join(format!("checkpoint_epoch{epoch}.safetensors"));
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let fallback = if algorithm == "grpo" {
        run_dir.join("grpo_final.safetensors")
    } else {
        run_dir.join("ppo_final.safetensors")
    };
    Ok(fallback)
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
        _ => common::infer_margin_mode(symbol, margin_cfg),
    };
    let contract_multiplier = json_f64(params, "contract-multiplier").unwrap_or(1.0);
    let margin_per_contract = json_f64(params, "margin-per-contract")
        .or(margin_cfg)
        .unwrap_or_else(|| common::infer_margin(symbol));

    Ok(ReplayConfig {
        initial_balance: json_f64(params, "initial-balance").unwrap_or(10000.0),
        max_position: json_i32(params, "max-position").unwrap_or(1),
        margin_mode,
        contract_multiplier,
        margin_per_contract,
        disable_margin: json_bool(params, "disable-margin").unwrap_or(false),
        ignore_session: json_bool(params, "ignore-session").unwrap_or(false),
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
        hidden: json_usize(params, "hidden").unwrap_or(0),
        layers: json_usize(params, "layers").unwrap_or(0),
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
fn summarize_behavior(rows: &[BehaviorRow]) -> BehaviorStats {
    let mut out = BehaviorStats {
        rows: rows.len(),
        windows: rows.iter().map(|row| row.window_idx).max().map(|v| v + 1).unwrap_or(0),
        ..BehaviorStats::default()
    };
    let mut tracker = TradeTracker::default();
    let mut total_trade_bars = 0usize;

    for row in rows {
        match row.action.as_str() {
            "buy" => out.requested_buy_steps += 1,
            "sell" => out.requested_sell_steps += 1,
            "hold" => out.requested_hold_steps += 1,
            "revert" => out.requested_revert_steps += 1,
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
fn write_behavior_csv(path: &Path, data: &data::DataSet, rows: &[BehaviorRow]) -> Result<()> {
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
            row.minutes_to_close
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.session_closed_violation.to_string(),
            row.margin_call_violation.to_string(),
            row.position_limit_violation.to_string(),
        ];
        writeln!(file, "{}", columns.join(","))?;
    }

    Ok(())
}

#[cfg(feature = "backend-candle")]
fn action_from_index(idx: usize) -> Action {
    match idx {
        0 => Action::Buy,
        1 => Action::Sell,
        2 => Action::Hold,
        _ => Action::Revert,
    }
}

#[cfg(feature = "backend-candle")]
fn action_label(action: Action) -> &'static str {
    match action {
        Action::Buy => "buy",
        Action::Sell => "sell",
        Action::Hold => "hold",
        Action::Revert => "revert",
    }
}

#[cfg(feature = "backend-candle")]
fn argmax_index(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best_val {
            best_val = *value;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(feature = "backend-candle")]
fn sample_from_probs(probs: &[f32], rng: &mut rand::rngs::StdRng) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::Distribution;

    let weights: Vec<f64> = probs
        .iter()
        .map(|prob| {
            if prob.is_finite() && *prob > 0.0 {
                *prob as f64
            } else {
                0.0
            }
        })
        .collect();

    if weights.iter().all(|weight| *weight <= 0.0) {
        return argmax_index(probs);
    }

    WeightedIndex::new(&weights)
        .map(|dist| dist.sample(rng))
        .unwrap_or_else(|_| argmax_index(probs))
}

#[cfg(feature = "backend-candle")]
fn average_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
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
fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key)?.as_u64()
}

#[cfg(feature = "backend-candle")]
fn json_i32(value: &Value, key: &str) -> Option<i32> {
    value
        .get(key)?
        .as_i64()
        .and_then(|parsed| i32::try_from(parsed).ok())
}
