#[path = "train_rl/args.rs"]
mod args;
#[cfg(feature = "backend-candle")]
#[path = "train_rl/candle.rs"]
mod candle;
#[cfg(any(feature = "torch", feature = "backend-candle"))]
#[path = "train_rl/common.rs"]
mod common;
#[cfg(any(feature = "torch", feature = "backend-candle"))]
#[path = "train_rl/data.rs"]
mod data;
#[cfg(feature = "torch")]
#[path = "train_rl/grpo.rs"]
mod grpo;
#[cfg(any(feature = "torch", feature = "backend-candle"))]
#[path = "train_rl/metrics.rs"]
mod metrics;
#[cfg(feature = "torch")]
#[path = "train_rl/model.rs"]
mod model;
#[cfg(feature = "torch")]
#[path = "train_rl/ppo.rs"]
mod ppo;
#[cfg(feature = "torch")]
#[path = "train_rl/util.rs"]
mod util;

#[cfg(feature = "torch")]
use anyhow::Context;
use std::path::{Path, PathBuf};

use chrono::Local;
use clap::Parser;
#[cfg(feature = "torch")]
use midas_env::env::{EnvConfig, MarginMode};
use midas_env::ml::{self, MlBackend, TrainerKind};
#[cfg(feature = "torch")]
use rand::rngs::StdRng;
#[cfg(feature = "torch")]
use rand::{Rng, SeedableRng, seq::SliceRandom};
#[cfg(feature = "torch")]
use std::io::Write;
#[cfg(feature = "torch")]
use std::time::Instant;
#[cfg(feature = "torch")]
use tch::nn;
#[cfg(feature = "torch")]
use tch::nn::OptimizerConfig;

use args::Args;
#[cfg(feature = "torch")]
use ppo::{LossStats, RolloutConfig, RolloutMetrics};

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    args.outdir = resolve_outdir(args.outdir, "runs_rl", args.load_checkpoint.is_some());
    let stack = ml::resolve_training_stack(TrainerKind::Rl, &args.backend, &args.device)?;
    ml::print_training_stack(&stack);
    match stack.backend {
        MlBackend::Libtorch => run_libtorch(args, stack),
        MlBackend::Candle => run_candle(args, stack),
        MlBackend::Burn | MlBackend::Mlx => ml::ensure_backend_is_implemented(&stack),
    }
}

#[cfg(feature = "torch")]
fn run_libtorch(args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    run(args, stack)
}

#[cfg(not(feature = "torch"))]
fn run_libtorch(_args: Args, _stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    anyhow::bail!(
        "backend 'libtorch' requires the 'torch' Cargo feature. Re-run with `cargo run --features torch --bin train_rl -- --backend libtorch ...`."
    )
}

#[cfg(feature = "backend-candle")]
fn run_candle(args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    candle::run(args, stack)
}

#[cfg(not(feature = "backend-candle"))]
fn run_candle(_args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    anyhow::bail!(
        "backend '{}' requires the 'backend-candle' Cargo feature. Re-run with `cargo run --features backend-candle --bin train_rl -- --backend candle ...`.",
        stack.backend
    )
}

#[cfg(feature = "torch")]
fn run(args: Args, mut stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    println!("info: run directory {}", args.outdir.display());

    if !(0.0..1.0).contains(&args.dropout) {
        anyhow::bail!("--dropout must be in [0, 1), got {}", args.dropout);
    }

    let device = util::resolve_device(stack.requested_runtime);
    stack.effective_runtime = util::runtime_from_device(&device);
    ml::write_run_metadata(
        &args.outdir.join("training_stack.json"),
        &stack,
        Some(&args.algorithm),
        Some(data::OBSERVATION_SCHEMA_NORMALIZED),
    )?;
    println!(
        "info: effective runtime resolved to {}",
        stack.effective_runtime
    );
    util::print_device(&device);

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().r#gen());
    let mut rng = StdRng::seed_from_u64(seed);

    let (train_path, val_path, test_path) = common::resolve_paths(&args)?;
    let train = data::load_dataset(&train_path, args.globex && !args.rth)?;
    let val = data::load_dataset(&val_path, args.globex && !args.rth)?;
    let test = data::load_dataset(&test_path, args.globex && !args.rth)?;

    let (margin_cfg, session_cfg) = common::load_symbol_config(&args.symbol_config, &train.symbol)?;
    let margin_mode = match args.margin_mode.as_str() {
        "per-contract" => MarginMode::PerContract,
        "price" => MarginMode::Price,
        _ => common::infer_margin_mode(&train.symbol, margin_cfg),
    };
    let contract_multiplier = if args.contract_multiplier > 0.0 {
        args.contract_multiplier
    } else {
        1.0
    };
    let margin_per_contract = args
        .margin_per_contract
        .or(margin_cfg)
        .unwrap_or_else(|| common::infer_margin(&train.symbol));
    let use_globex = if let Some(session) = session_cfg {
        match session.as_str() {
            "rth" => false,
            "globex" => true,
            _ => !args.rth,
        }
    } else {
        !args.rth
    };

    let train = train.with_session(use_globex);
    let val = val.with_session(use_globex);
    let test = test.with_session(use_globex);

    let full_file = if args.parquet.is_some() {
        args.full_file
    } else {
        args.full_file || !args.windowed
    };
    let raw_windows_train = if full_file {
        vec![(0, train.close.len())]
    } else {
        midas_env::sampler::windows(train.close.len(), args.window, args.step)
    };
    let raw_windows_val = if full_file {
        vec![(0, val.close.len())]
    } else {
        midas_env::sampler::windows(val.close.len(), args.window, args.step)
    };
    let raw_windows_test = if full_file {
        vec![(0, test.close.len())]
    } else {
        midas_env::sampler::windows(test.close.len(), args.window, args.step)
    };

    let feature_warmup = midas_env::features::feature_warmup_bars();
    let min_window_start = feature_warmup.saturating_sub(1);
    let adjust_windows = |label: &str, windows: Vec<(usize, usize)>| {
        let before = windows.len();
        let adjusted = midas_env::sampler::enforce_min_start(&windows, min_window_start);
        let dropped = before.saturating_sub(adjusted.len());
        if dropped > 0 {
            println!(
                "info: dropped {} {} window(s) before feature warmup ({} bars)",
                dropped, label, feature_warmup
            );
        }
        adjusted
    };

    let windows_train = adjust_windows("train", raw_windows_train);
    let windows_val = adjust_windows("val", raw_windows_val);
    let windows_test = adjust_windows("test", raw_windows_test);

    if windows_train.is_empty() {
        anyhow::bail!(
            "no training windows available after applying feature warmup ({} bars)",
            feature_warmup
        );
    }

    let env_cfg = EnvConfig {
        max_position: args.max_position,
        margin_mode,
        contract_multiplier,
        margin_per_contract,
        enforce_margin: !args.disable_margin,
        drawdown_penalty: args.drawdown_penalty,
        drawdown_penalty_growth: args.drawdown_penalty_growth,
        session_close_penalty: args.session_close_penalty,
        auto_close_minutes_before_close: args.auto_close_minutes_before_close,
        max_hold_bars_positive: args.max_hold_bars_positive,
        max_hold_bars_drawdown: args.max_hold_bars_drawdown,
        hold_duration_penalty: args.hold_duration_penalty,
        hold_duration_penalty_growth: args.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
        min_hold_bars: args.min_hold_bars,
        early_exit_penalty: args.early_exit_penalty,
        early_flip_penalty: args.early_flip_penalty,
        invalid_revert_penalty: args.invalid_revert_penalty,
        invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
        flat_hold_penalty: args.flat_hold_penalty,
        flat_hold_penalty_growth: args.flat_hold_penalty_growth,
        max_flat_hold_bars: args.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let obs_dim = train.obs_dim as i64;
    let action_dim = 4i64;

    let use_grpo = args.algorithm == "grpo";

    let mut vs = nn::VarStore::new(device);
    let policy = model::build_policy(
        &vs.root().sub("policy"),
        obs_dim,
        args.hidden as i64,
        args.layers,
        action_dim,
        args.dropout,
    );
    let value = if use_grpo {
        None
    } else {
        Some(model::build_value(
            &vs.root().sub("value"),
            obs_dim,
            args.hidden as i64,
            args.layers,
            args.dropout,
        ))
    };

    if args.dropout > 0.0 {
        println!(
            "info: libtorch dropout set to {:.3} on hidden layers during training; eval/test disable dropout",
            args.dropout
        );
    }

    if let Some(path) = &args.load_checkpoint {
        load_checkpoint(&mut vs, device, path)?;
        println!("info: loaded checkpoint {}", path.display());
    }

    let mut opt = nn::Adam::default()
        .build(&vs, args.lr)
        .context("build optimizer")?;

    let log_path = args.outdir.join("rl_log.csv");
    common::ensure_csv_header(&log_path, common::RL_LOG_HEADER_V2)?;

    let rollout_cfg = RolloutConfig {
        gamma: args.gamma,
        lam: args.lam,
        initial_balance: args.initial_balance,
        device,
        ignore_session: args.ignore_session,
    };
    let ppo_cfg = ppo::PpoConfig {
        clip: args.clip,
        vf_coef: args.vf_coef,
        ent_coef: args.ent_coef,
        ppo_epochs: args.ppo_epochs,
    };
    let grpo_cfg = grpo::GrpoConfig {
        group_size: args.group_size,
        clip: args.clip,
        ent_coef: args.ent_coef,
        grpo_epochs: args.grpo_epochs,
    };

    let mut train_windows = windows_train;

    let training_start = Instant::now();
    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        train_windows.shuffle(&mut rng);
        let train_count = if args.train_windows == 0 {
            train_windows.len()
        } else {
            args.train_windows.min(train_windows.len())
        };

        let mut train_metrics = Vec::with_capacity(train_count);
        let mut loss_stats_ppo = Vec::with_capacity(train_count);
        let mut loss_stats_grpo = Vec::with_capacity(train_count);

        if use_grpo {
            // Train on distinct shuffled windows each epoch; each GRPO group
            // replays the same market segment with fresh stochastic actions.
            for window in train_windows.iter().take(train_count) {
                let group = grpo::rollout_group(
                    &train,
                    &[*window],
                    &policy,
                    &env_cfg,
                    grpo_cfg.group_size,
                    device,
                    true,
                    &mut rng,
                    false,
                );
                let advantages = grpo::compute_grpo_advantages(&group);
                let losses =
                    grpo::grpo_update(&policy, &vs, &mut opt, &group, &advantages, &grpo_cfg);
                let metrics = grpo::summarize_group(&group, args.sortino_annualization);
                train_metrics.push(metrics);
                loss_stats_grpo.push(losses);
            }
        } else {
            // PPO training: single rollout with value function
            for window in train_windows.iter().take(train_count) {
                let batch = ppo::rollout(
                    &train,
                    *window,
                    &policy,
                    value.as_ref().unwrap(),
                    &env_cfg,
                    &rollout_cfg,
                    true,
                    &mut rng,
                    false,
                );
                let metrics = ppo::summarize_batch(&batch, args.sortino_annualization);
                let losses = ppo::ppo_update(
                    &policy,
                    value.as_ref().unwrap(),
                    &vs,
                    &mut opt,
                    &batch,
                    &ppo_cfg,
                );
                train_metrics.push(metrics);
                loss_stats_ppo.push(losses);
            }
        }

        let train_summary = average_metrics(&train_metrics);

        let eval_count = args.eval_windows.min(windows_val.len()).max(1);
        let mut eval_metrics = Vec::with_capacity(eval_count);
        for window in windows_val.iter().take(eval_count) {
            if use_grpo {
                let group = grpo::rollout_group(
                    &val,
                    &[*window],
                    &policy,
                    &env_cfg,
                    1,
                    device,
                    false,
                    &mut rng,
                    false,
                );
                eval_metrics.push(grpo::summarize_group(&group, args.sortino_annualization));
            } else {
                let batch = ppo::rollout(
                    &val,
                    *window,
                    &policy,
                    value.as_ref().unwrap(),
                    &env_cfg,
                    &rollout_cfg,
                    false,
                    &mut rng,
                    false,
                );
                eval_metrics.push(ppo::summarize_batch(&batch, args.sortino_annualization));
            }
        }
        let eval_summary = average_metrics(&eval_metrics);

        let probe_summary = if args.log_interval > 0 && epoch % args.log_interval == 0 {
            windows_val.first().map(|window| {
                if use_grpo {
                    let group = grpo::rollout_group(
                        &val,
                        &[*window],
                        &policy,
                        &env_cfg,
                        1,
                        device,
                        false,
                        &mut rng,
                        true,
                    );
                    grpo::summarize_group(&group, args.sortino_annualization)
                } else {
                    let batch = ppo::rollout(
                        &val,
                        *window,
                        &policy,
                        value.as_ref().unwrap(),
                        &env_cfg,
                        &rollout_cfg,
                        false,
                        &mut rng,
                        true,
                    );
                    ppo::summarize_batch(&batch, args.sortino_annualization)
                }
            })
        } else {
            None
        };

        let fitness_source = if args.fitness_use_eval {
            eval_summary
        } else {
            train_summary
        };
        let fitness = (args.w_pnl * fitness_source.pnl) + (args.w_sortino * fitness_source.sortino)
            - (args.w_mdd * fitness_source.drawdown);

        println!(
            "epoch {} | train ret {:.4} | train pnl {:.4} | eval pnl {:.4} | eval sortino {:.4} | eval mdd {:.4} | eval conf {:.3} | fitness {:.4} | time {}",
            epoch,
            train_summary.ret_mean,
            train_summary.pnl,
            eval_summary.pnl,
            eval_summary.sortino,
            eval_summary.drawdown,
            eval_summary.mean_max_prob,
            fitness,
            format_duration(epoch_start.elapsed())
        );

        if args.log_interval > 0 && epoch % args.log_interval == 0 {
            let mut file = std::fs::OpenOptions::new().append(true).open(&log_path)?;
            if use_grpo {
                let loss_summary = average_grpo_losses(&loss_stats_grpo);
                write_rl_log_row(
                    &mut file,
                    epoch,
                    "grpo",
                    &train_summary,
                    &eval_summary,
                    probe_summary.as_ref(),
                    fitness,
                    loss_summary.policy_loss,
                    None,
                    loss_summary.entropy,
                    loss_summary.total_loss,
                    loss_summary.policy_grad_norm,
                    None,
                    None,
                    Some(loss_summary.kl_div),
                    loss_summary.clip_frac,
                )?;
            } else {
                let loss_summary = average_losses(&loss_stats_ppo);
                write_rl_log_row(
                    &mut file,
                    epoch,
                    "ppo",
                    &train_summary,
                    &eval_summary,
                    probe_summary.as_ref(),
                    fitness,
                    loss_summary.policy_loss,
                    Some(loss_summary.value_loss),
                    loss_summary.entropy,
                    loss_summary.total_loss,
                    loss_summary.policy_grad_norm,
                    Some(loss_summary.value_grad_norm),
                    Some(loss_summary.approx_kl),
                    None,
                    loss_summary.clip_frac,
                )?;
            }
        }

        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            let ckpt_path = args.outdir.join(format!("checkpoint_epoch{}.pt", epoch));
            vs.save(&ckpt_path)?;
        }
    }

    let eval_count = args.eval_windows.min(windows_test.len()).max(1);
    let mut test_metrics = Vec::with_capacity(eval_count);
    for window in windows_test.iter().take(eval_count) {
        if use_grpo {
            let group = grpo::rollout_group(
                &test,
                &[*window],
                &policy,
                &env_cfg,
                1,
                device,
                false,
                &mut rng,
                false,
            );
            test_metrics.push(grpo::summarize_group(&group, args.sortino_annualization));
        } else {
            let batch = ppo::rollout(
                &test,
                *window,
                &policy,
                value.as_ref().unwrap(),
                &env_cfg,
                &rollout_cfg,
                false,
                &mut rng,
                false,
            );
            test_metrics.push(ppo::summarize_batch(&batch, args.sortino_annualization));
        }
    }
    let test_summary = average_metrics(&test_metrics);
    println!(
        "test | ret {:.4} | pnl {:.4} | sortino {:.4} | mdd {:.4}",
        test_summary.ret_mean, test_summary.pnl, test_summary.sortino, test_summary.drawdown
    );
    println!(
        "total training time: {}",
        format_duration(training_start.elapsed())
    );

    let final_path = if use_grpo {
        args.outdir.join("grpo_final.pt")
    } else {
        args.outdir.join("ppo_final.pt")
    };
    vs.save(&final_path)?;
    let algo_name = if use_grpo { "GRPO" } else { "PPO" };
    println!(
        "Saved final {} checkpoint to {}",
        algo_name,
        final_path.display()
    );

    Ok(())
}

fn resolve_outdir(outdir: PathBuf, default_base: &str, is_resume: bool) -> PathBuf {
    if is_resume || !is_default_outdir(&outdir, default_base) {
        return outdir;
    }
    let stamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    outdir.join(stamp)
}

fn is_default_outdir(outdir: &Path, default_base: &str) -> bool {
    outdir == Path::new(default_base) || outdir == Path::new(&format!("./{default_base}"))
}

#[cfg(feature = "torch")]
fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    let minutes = secs / 60;
    let seconds = secs % 60;
    if minutes > 0 {
        format!("{minutes}m{seconds:02}.{millis:03}s")
    } else {
        format!("{seconds}.{millis:03}s")
    }
}

#[cfg(feature = "torch")]
fn average_metrics(values: &[RolloutMetrics]) -> RolloutMetrics {
    if values.is_empty() {
        return RolloutMetrics {
            ret_mean: 0.0,
            pnl: 0.0,
            realized_pnl: 0.0,
            sortino: 0.0,
            drawdown: 0.0,
            commission: 0.0,
            slippage: 0.0,
            buy_frac: 0.0,
            sell_frac: 0.0,
            hold_frac: 0.0,
            revert_frac: 0.0,
            mean_max_prob: 0.0,
            entries: 0.0,
            exits: 0.0,
            flips: 0.0,
            avg_hold: 0.0,
        };
    }
    let mut ret_mean = 0.0;
    let mut pnl = 0.0;
    let mut realized_pnl = 0.0;
    let mut sortino = 0.0;
    let mut drawdown = 0.0;
    let mut commission = 0.0;
    let mut slippage = 0.0;
    let mut buy_frac = 0.0;
    let mut sell_frac = 0.0;
    let mut hold_frac = 0.0;
    let mut revert_frac = 0.0;
    let mut mean_max_prob = 0.0;
    let mut entries = 0.0;
    let mut exits = 0.0;
    let mut flips = 0.0;
    let mut avg_hold = 0.0;
    for v in values {
        ret_mean += v.ret_mean;
        pnl += v.pnl;
        realized_pnl += v.realized_pnl;
        sortino += v.sortino;
        drawdown += v.drawdown;
        commission += v.commission;
        slippage += v.slippage;
        buy_frac += v.buy_frac;
        sell_frac += v.sell_frac;
        hold_frac += v.hold_frac;
        revert_frac += v.revert_frac;
        mean_max_prob += v.mean_max_prob;
        entries += v.entries;
        exits += v.exits;
        flips += v.flips;
        avg_hold += v.avg_hold;
    }
    let denom = values.len() as f64;
    RolloutMetrics {
        ret_mean: ret_mean / denom,
        pnl: pnl / denom,
        realized_pnl: realized_pnl / denom,
        sortino: sortino / denom,
        drawdown: drawdown / denom,
        commission: commission / denom,
        slippage: slippage / denom,
        buy_frac: buy_frac / denom,
        sell_frac: sell_frac / denom,
        hold_frac: hold_frac / denom,
        revert_frac: revert_frac / denom,
        mean_max_prob: mean_max_prob / denom,
        entries: entries / denom,
        exits: exits / denom,
        flips: flips / denom,
        avg_hold: avg_hold / denom,
    }
}

#[cfg(feature = "torch")]
fn format_log_f64(value: f64, precision: usize) -> String {
    format!("{value:.precision$}")
}

#[cfg(feature = "torch")]
fn format_log_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|inner| format_log_f64(inner, precision))
        .unwrap_or_default()
}

#[cfg(feature = "torch")]
fn append_rollout_fields(fields: &mut Vec<String>, metrics: Option<&RolloutMetrics>) {
    let Some(metrics) = metrics else {
        for _ in 0..16 {
            fields.push(String::new());
        }
        return;
    };
    fields.push(format_log_f64(metrics.ret_mean, 4));
    fields.push(format_log_f64(metrics.pnl, 4));
    fields.push(format_log_f64(metrics.realized_pnl, 4));
    fields.push(format_log_f64(metrics.sortino, 4));
    fields.push(format_log_f64(metrics.drawdown, 4));
    fields.push(format_log_f64(metrics.commission, 4));
    fields.push(format_log_f64(metrics.slippage, 4));
    fields.push(format_log_f64(metrics.buy_frac, 6));
    fields.push(format_log_f64(metrics.sell_frac, 6));
    fields.push(format_log_f64(metrics.hold_frac, 6));
    fields.push(format_log_f64(metrics.revert_frac, 6));
    fields.push(format_log_f64(metrics.mean_max_prob, 6));
    fields.push(format_log_f64(metrics.entries, 4));
    fields.push(format_log_f64(metrics.exits, 4));
    fields.push(format_log_f64(metrics.flips, 4));
    fields.push(format_log_f64(metrics.avg_hold, 4));
}

#[cfg(feature = "torch")]
fn write_rl_log_row(
    file: &mut std::fs::File,
    epoch: usize,
    algorithm: &str,
    train: &RolloutMetrics,
    eval: &RolloutMetrics,
    probe: Option<&RolloutMetrics>,
    fitness: f64,
    policy_loss: f64,
    value_loss: Option<f64>,
    entropy: f64,
    total_loss: f64,
    policy_grad_norm: f64,
    value_grad_norm: Option<f64>,
    approx_kl: Option<f64>,
    kl_div: Option<f64>,
    clip_frac: f64,
) -> anyhow::Result<()> {
    let mut fields = Vec::with_capacity(60);
    fields.push(epoch.to_string());
    fields.push(algorithm.to_string());
    append_rollout_fields(&mut fields, Some(train));
    append_rollout_fields(&mut fields, Some(eval));
    append_rollout_fields(&mut fields, probe);
    fields.push(format_log_f64(fitness, 4));
    fields.push(format_log_f64(policy_loss, 6));
    fields.push(format_log_opt(value_loss, 6));
    fields.push(format_log_f64(entropy, 6));
    fields.push(format_log_f64(entropy.exp(), 6));
    fields.push(format_log_f64(total_loss, 6));
    fields.push(format_log_f64(policy_grad_norm, 6));
    fields.push(format_log_opt(value_grad_norm, 6));
    fields.push(format_log_opt(approx_kl, 6));
    fields.push(format_log_opt(kl_div, 6));
    fields.push(format_log_f64(clip_frac, 6));
    writeln!(file, "{}", fields.join(","))?;
    Ok(())
}

#[cfg(feature = "torch")]
fn average_losses(values: &[LossStats]) -> LossStats {
    if values.is_empty() {
        return LossStats {
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            policy_grad_norm: 0.0,
            value_grad_norm: 0.0,
            approx_kl: 0.0,
            clip_frac: 0.0,
        };
    }
    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    let mut policy_grad_norm = 0.0;
    let mut value_grad_norm = 0.0;
    let mut approx_kl = 0.0;
    let mut clip_frac = 0.0;
    for v in values {
        policy_loss += v.policy_loss;
        value_loss += v.value_loss;
        entropy += v.entropy;
        total_loss += v.total_loss;
        policy_grad_norm += v.policy_grad_norm;
        value_grad_norm += v.value_grad_norm;
        approx_kl += v.approx_kl;
        clip_frac += v.clip_frac;
    }
    let denom = values.len() as f64;
    LossStats {
        policy_loss: policy_loss / denom,
        value_loss: value_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        policy_grad_norm: policy_grad_norm / denom,
        value_grad_norm: value_grad_norm / denom,
        approx_kl: approx_kl / denom,
        clip_frac: clip_frac / denom,
    }
}

#[cfg(feature = "torch")]
fn average_grpo_losses(values: &[grpo::GrpoLossStats]) -> grpo::GrpoLossStats {
    if values.is_empty() {
        return grpo::GrpoLossStats {
            policy_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            kl_div: 0.0,
            policy_grad_norm: 0.0,
            clip_frac: 0.0,
        };
    }
    let mut policy_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    let mut kl_div = 0.0;
    let mut policy_grad_norm = 0.0;
    let mut clip_frac = 0.0;
    for v in values {
        policy_loss += v.policy_loss;
        entropy += v.entropy;
        total_loss += v.total_loss;
        kl_div += v.kl_div;
        policy_grad_norm += v.policy_grad_norm;
        clip_frac += v.clip_frac;
    }
    let denom = values.len() as f64;
    grpo::GrpoLossStats {
        policy_loss: policy_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        kl_div: kl_div / denom,
        policy_grad_norm: policy_grad_norm / denom,
        clip_frac: clip_frac / denom,
    }
}

#[cfg(feature = "torch")]
fn load_checkpoint(vs: &mut nn::VarStore, device: tch::Device, path: &Path) -> anyhow::Result<()> {
    match vs.load(path) {
        Ok(()) => return Ok(()),
        Err(err) => {
            let err_msg = err.to_string();
            let short_err = err_msg.lines().next().unwrap_or(&err_msg);
            eprintln!(
                "info: native RL checkpoint load failed for {} ({}), trying compatible import",
                path.display(),
                short_err
            );
            if let Err(ts_err) = load_torchscript_checkpoint(vs, device, path) {
                anyhow::bail!(
                    "load checkpoint {} failed: {}; TorchScript fallback failed: {}",
                    path.display(),
                    err_msg,
                    ts_err
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "torch")]
fn load_torchscript_checkpoint(
    vs: &mut nn::VarStore,
    device: tch::Device,
    path: &Path,
) -> anyhow::Result<()> {
    let module = tch::CModule::load_on_device(path, device)
        .with_context(|| format!("load torchscript checkpoint {}", path.display()))?;
    let params = module
        .named_parameters()
        .context("read torchscript parameters")?;
    let summary = apply_named_parameters(vs, params, path)?;
    if summary.policy_aliases > 0 || summary.skipped_value_tensors > 0 {
        println!(
            "info: imported {} tensor(s) from {} (policy aliases: {}, value tensors left initialized: {})",
            summary.imported_tensors,
            path.display(),
            summary.policy_aliases,
            summary.skipped_value_tensors,
        );
    }
    Ok(())
}

#[cfg(feature = "torch")]
struct CheckpointImportSummary {
    imported_tensors: usize,
    policy_aliases: usize,
    skipped_value_tensors: usize,
}

#[cfg(feature = "torch")]
fn apply_named_parameters(
    vs: &mut nn::VarStore,
    params: Vec<(String, tch::Tensor)>,
    path: &Path,
) -> anyhow::Result<CheckpointImportSummary> {
    let mut by_name = std::collections::HashMap::with_capacity(params.len());
    for (name, tensor) in params {
        let name = name.replace('|', ".");
        by_name.insert(name, tensor);
    }

    let has_value_tensors = by_name.keys().any(|name| name.starts_with("value."));
    let mut imported_tensors = 0usize;
    let mut policy_aliases = 0usize;
    let mut skipped_value_tensors = 0usize;

    let mut variables = vs.variables_.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        let mut aliased_policy = false;
        let src = if let Some(src) = by_name.get(name) {
            src
        } else if let Some(policy_name) = name.strip_prefix("policy.") {
            let Some(src) = by_name.get(policy_name) else {
                anyhow::bail!("checkpoint missing tensor {name} in {}", path.display());
            };
            aliased_policy = true;
            src
        } else if name.starts_with("value.") && !has_value_tensors {
            skipped_value_tensors += 1;
            continue;
        } else {
            anyhow::bail!("checkpoint missing tensor {name} in {}", path.display());
        };
        tch::no_grad(|| {
            var.set_data(&var.to_kind(src.kind()));
            var.f_copy_(src)
        })
        .with_context(|| format!("copy tensor {name} from {}", path.display()))?;
        imported_tensors += 1;
        if aliased_policy {
            policy_aliases += 1;
        }
    }

    Ok(CheckpointImportSummary {
        imported_tensors,
        policy_aliases,
        skipped_value_tensors,
    })
}
