use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::{
    AdamW, Module, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap, linear,
    ops::{log_softmax, softmax},
    sequential::seq,
};
use midas_env::env::{Action, EnvConfig, MarginMode, StepContext, TradingEnv};
use midas_env::ml::{self, ComputeRuntime};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::args::Args;
use crate::common;
use crate::data::{DataSet, build_observation};
use crate::metrics::{compute_sortino, max_drawdown};

#[derive(Debug, Clone, Copy)]
struct RolloutMetrics {
    ret_mean: f64,
    pnl: f64,
    sortino: f64,
    drawdown: f64,
}

#[derive(Debug, Clone, Copy)]
struct LossStats {
    policy_loss: f64,
    value_loss: f64,
    entropy: f64,
    total_loss: f64,
}

#[derive(Debug, Clone, Copy)]
struct GrpoLossStats {
    policy_loss: f64,
    entropy: f64,
    total_loss: f64,
    kl_div: f64,
}

struct RolloutBatch {
    obs: Tensor,
    actions: Tensor,
    logp: Tensor,
    adv: Vec<f32>,
    ret: Tensor,
    rewards: Vec<f64>,
    pnl: Vec<f64>,
    returns: Vec<f64>,
    equity: Vec<f64>,
}

struct GrpoRollout {
    obs: Tensor,
    actions: Tensor,
    logp: Tensor,
    reward: f64,
    pnl: f64,
    returns: Vec<f64>,
    equity: Vec<f64>,
}

struct GrpoGroup {
    rollouts: Vec<GrpoRollout>,
    mean_reward: f64,
    std_reward: f64,
}

struct RolloutConfig {
    gamma: f64,
    lam: f64,
    initial_balance: f64,
    ignore_session: bool,
}

struct PpoConfig {
    clip: f64,
    vf_coef: f64,
    ent_coef: f64,
    ppo_epochs: usize,
}

struct GrpoConfig {
    group_size: usize,
    clip: f64,
    ent_coef: f64,
    grpo_epochs: usize,
}

pub fn run(args: Args, mut stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    println!("info: run directory {}", args.outdir.display());

    let device = resolve_device(stack.requested_runtime)?;
    stack.effective_runtime = runtime_from_device(&device);
    ml::write_run_metadata(
        &args.outdir.join("training_stack.json"),
        &stack,
        Some(&args.algorithm),
    )?;
    println!(
        "info: effective runtime resolved to {}",
        stack.effective_runtime
    );
    print_device(&device);

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().r#gen());
    if let Err(err) = device.set_seed(seed) {
        println!("info: candle backend device seeding unavailable: {err}");
    }
    let mut rng = StdRng::seed_from_u64(seed);

    let (train_path, val_path, test_path) = common::resolve_paths(&args)?;
    let train = crate::data::load_dataset(&train_path, args.globex && !args.rth)?;
    let val = crate::data::load_dataset(&val_path, args.globex && !args.rth)?;
    let test = crate::data::load_dataset(&test_path, args.globex && !args.rth)?;

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

    let mut train_windows = adjust_windows("train", raw_windows_train);
    let windows_val = adjust_windows("val", raw_windows_val);
    let windows_test = adjust_windows("test", raw_windows_test);

    if train_windows.is_empty() {
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

    let obs_dim = train.obs_dim;
    let action_dim = 4usize;
    let use_grpo = args.algorithm == "grpo";

    let mut varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let policy = build_policy(
        var_builder.pp("policy"),
        obs_dim,
        args.hidden,
        args.layers,
        action_dim,
    )?;
    let value = if use_grpo {
        None
    } else {
        Some(build_value(
            var_builder.pp("value"),
            obs_dim,
            args.hidden,
            args.layers,
        )?)
    };

    if let Some(path) = &args.load_checkpoint {
        load_checkpoint(&mut varmap, path)?;
        println!("info: loaded checkpoint {}", path.display());
    }

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )
    .context("build candle optimizer")?;

    let log_path = args.outdir.join("rl_log.csv");
    if !log_path.exists() || std::fs::metadata(&log_path)?.len() == 0 {
        let header = if use_grpo {
            "epoch,train_ret_mean,train_pnl,train_sortino,train_drawdown,eval_ret_mean,eval_pnl,eval_sortino,eval_drawdown,fitness,policy_loss,entropy,total_loss,kl_div\n"
        } else {
            "epoch,train_ret_mean,train_pnl,train_sortino,train_drawdown,eval_ret_mean,eval_pnl,eval_sortino,eval_drawdown,fitness,policy_loss,value_loss,entropy,total_loss\n"
        };
        std::fs::write(&log_path, header)?;
    }

    let rollout_cfg = RolloutConfig {
        gamma: args.gamma,
        lam: args.lam,
        initial_balance: args.initial_balance,
        ignore_session: args.ignore_session,
    };
    let ppo_cfg = PpoConfig {
        clip: args.clip,
        vf_coef: args.vf_coef,
        ent_coef: args.ent_coef,
        ppo_epochs: args.ppo_epochs,
    };
    let grpo_cfg = GrpoConfig {
        group_size: args.group_size,
        clip: args.clip,
        ent_coef: args.ent_coef,
        grpo_epochs: args.grpo_epochs,
    };

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
                let group = rollout_group(
                    &train,
                    &[*window],
                    &policy,
                    &env_cfg,
                    &rollout_cfg,
                    grpo_cfg.group_size,
                    &device,
                    &mut rng,
                )?;
                let advantages = compute_grpo_advantages(&group);
                let losses = grpo_update(&policy, &mut opt, &group, &advantages, &grpo_cfg)?;
                let metrics = summarize_group(&group, args.sortino_annualization);
                train_metrics.push(metrics);
                loss_stats_grpo.push(losses);
            }
        } else {
            for window in train_windows.iter().take(train_count) {
                let batch = rollout(
                    &train,
                    *window,
                    &policy,
                    value.as_ref().expect("value network"),
                    &env_cfg,
                    &rollout_cfg,
                    &device,
                    &mut rng,
                )?;
                let metrics = summarize_batch(&batch, args.sortino_annualization);
                let losses = ppo_update(
                    &policy,
                    value.as_ref().expect("value network"),
                    &mut opt,
                    &batch,
                    &device,
                    &ppo_cfg,
                )?;
                train_metrics.push(metrics);
                loss_stats_ppo.push(losses);
            }
        }

        let train_summary = average_metrics(&train_metrics);

        let eval_count = args.eval_windows.min(windows_val.len()).max(1);
        let mut eval_metrics = Vec::with_capacity(eval_count);
        for window in windows_val.iter().take(eval_count) {
            if use_grpo {
                let group = rollout_group(
                    &val,
                    &[*window],
                    &policy,
                    &env_cfg,
                    &rollout_cfg,
                    1,
                    &device,
                    &mut rng,
                )?;
                eval_metrics.push(summarize_group(&group, args.sortino_annualization));
            } else {
                let batch = rollout(
                    &val,
                    *window,
                    &policy,
                    value.as_ref().expect("value network"),
                    &env_cfg,
                    &rollout_cfg,
                    &device,
                    &mut rng,
                )?;
                eval_metrics.push(summarize_batch(&batch, args.sortino_annualization));
            }
        }
        let eval_summary = average_metrics(&eval_metrics);

        let fitness_source = if args.fitness_use_eval {
            eval_summary
        } else {
            train_summary
        };
        let fitness = (args.w_pnl * fitness_source.pnl) + (args.w_sortino * fitness_source.sortino)
            - (args.w_mdd * fitness_source.drawdown);

        println!(
            "epoch {} | train ret {:.4} | train pnl {:.4} | eval pnl {:.4} | eval sortino {:.4} | eval mdd {:.4} | fitness {:.4} | time {}",
            epoch,
            train_summary.ret_mean,
            train_summary.pnl,
            eval_summary.pnl,
            eval_summary.sortino,
            eval_summary.drawdown,
            fitness,
            format_duration(epoch_start.elapsed())
        );

        if args.log_interval > 0 && epoch % args.log_interval == 0 {
            let mut file = std::fs::OpenOptions::new().append(true).open(&log_path)?;
            if use_grpo {
                let loss_summary = average_grpo_losses(&loss_stats_grpo);
                writeln!(
                    file,
                    "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}",
                    epoch,
                    train_summary.ret_mean,
                    train_summary.pnl,
                    train_summary.sortino,
                    train_summary.drawdown,
                    eval_summary.ret_mean,
                    eval_summary.pnl,
                    eval_summary.sortino,
                    eval_summary.drawdown,
                    fitness,
                    loss_summary.policy_loss,
                    loss_summary.entropy,
                    loss_summary.total_loss,
                    loss_summary.kl_div
                )?;
            } else {
                let loss_summary = average_losses(&loss_stats_ppo);
                writeln!(
                    file,
                    "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}",
                    epoch,
                    train_summary.ret_mean,
                    train_summary.pnl,
                    train_summary.sortino,
                    train_summary.drawdown,
                    eval_summary.ret_mean,
                    eval_summary.pnl,
                    eval_summary.sortino,
                    eval_summary.drawdown,
                    fitness,
                    loss_summary.policy_loss,
                    loss_summary.value_loss,
                    loss_summary.entropy,
                    loss_summary.total_loss
                )?;
            }
        }

        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            let ckpt_path = args
                .outdir
                .join(format!("checkpoint_epoch{}.safetensors", epoch));
            varmap
                .save(&ckpt_path)
                .with_context(|| format!("save checkpoint {}", ckpt_path.display()))?;
        }
    }

    let eval_count = args.eval_windows.min(windows_test.len()).max(1);
    let mut test_metrics = Vec::with_capacity(eval_count);
    for window in windows_test.iter().take(eval_count) {
        if use_grpo {
            let group = rollout_group(
                &test,
                &[*window],
                &policy,
                &env_cfg,
                &rollout_cfg,
                1,
                &device,
                &mut rng,
            )?;
            test_metrics.push(summarize_group(&group, args.sortino_annualization));
        } else {
            let batch = rollout(
                &test,
                *window,
                &policy,
                value.as_ref().expect("value network"),
                &env_cfg,
                &rollout_cfg,
                &device,
                &mut rng,
            )?;
            test_metrics.push(summarize_batch(&batch, args.sortino_annualization));
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
        args.outdir.join("grpo_final.safetensors")
    } else {
        args.outdir.join("ppo_final.safetensors")
    };
    varmap
        .save(&final_path)
        .with_context(|| format!("save final checkpoint {}", final_path.display()))?;
    let algo_name = if use_grpo { "GRPO" } else { "PPO" };
    println!(
        "Saved final {} checkpoint to {}",
        algo_name,
        final_path.display()
    );

    Ok(())
}

pub(crate) fn build_policy(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    action_dim: usize,
) -> Result<Sequential> {
    build_mlp(vb, input_dim, hidden, layers, action_dim)
}

fn build_value(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
) -> Result<Sequential> {
    build_mlp(vb, input_dim, hidden, layers, 1)
}

fn build_mlp(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    output_dim: usize,
) -> Result<Sequential> {
    let mut seq = seq();
    let mut in_dim = input_dim;
    for i in 0..layers {
        seq = seq
            .add(linear(in_dim, hidden, vb.pp(format!("layer_{i}")))?)
            .add_fn(|xs| xs.tanh());
        in_dim = hidden;
    }
    seq = seq.add(linear(in_dim, output_dim, vb.pp("out"))?);
    Ok(seq)
}

fn rollout(
    data: &DataSet,
    window: (usize, usize),
    policy: &Sequential,
    value: &Sequential,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    device: &Device,
    rng: &mut StdRng,
) -> Result<RolloutBatch> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let obs_dim = data.obs_dim;

    let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = cfg.initial_balance;
    let mut prev_equity = cfg.initial_balance;

    let mut obs_buf = Vec::with_capacity(steps * obs_dim);
    let mut act_buf = Vec::with_capacity(steps);
    let mut logp_buf = Vec::with_capacity(steps);
    let mut val_buf = Vec::with_capacity(steps);
    let mut rew_buf = Vec::with_capacity(steps);
    let mut pnl_buf = Vec::with_capacity(steps);
    let mut ret_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);

    for t in (start + 1)..end {
        let obs = build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            cfg.initial_balance,
        );
        obs_buf.extend_from_slice(&obs);
        let obs_tensor = Tensor::from_vec(obs, (1, obs_dim), device)?;

        let logits = policy.forward(&obs_tensor)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = sample_from_probs(&probs, rng) as u32;
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();
        let value_val = value
            .forward(&obs_tensor)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()?;

        let action = match action_idx {
            0 => Action::Buy,
            1 => Action::Sell,
            2 => Action::Hold,
            _ => Action::Revert,
        };

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

        act_buf.push(action_idx);
        logp_buf.push(logp_val);
        val_buf.push(value_val);
        rew_buf.push(reward);
        pnl_buf.push(info.pnl_change);
        let denom = if prev_equity.abs() < 1e-8 {
            1e-8
        } else {
            prev_equity
        };
        ret_series.push(info.pnl_change / denom);
        equity_curve.push(equity);
        prev_equity = equity;
    }

    let (adv_buf, ret_buf) = compute_gae(&rew_buf, &val_buf, cfg.gamma, cfg.lam);

    Ok(RolloutBatch {
        obs: Tensor::from_vec(obs_buf, (steps, obs_dim), device)?,
        actions: Tensor::from_vec(act_buf, steps, device)?,
        logp: Tensor::from_vec(logp_buf, steps, device)?,
        adv: adv_buf,
        ret: Tensor::from_vec(ret_buf, steps, device)?,
        rewards: rew_buf,
        pnl: pnl_buf,
        returns: ret_series,
        equity: equity_curve,
    })
}

fn summarize_batch(batch: &RolloutBatch, annualization: f64) -> RolloutMetrics {
    let ret_mean = if batch.returns.is_empty() {
        0.0
    } else {
        batch.returns.iter().sum::<f64>() / batch.returns.len() as f64
    };
    let pnl_total = batch.pnl.iter().sum::<f64>();
    let sortino = compute_sortino(&batch.returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&batch.equity);
    let _ = batch.rewards.len();
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        sortino,
        drawdown,
    }
}

fn ppo_update(
    policy: &Sequential,
    value: &Sequential,
    opt: &mut AdamW,
    batch: &RolloutBatch,
    device: &Device,
    cfg: &PpoConfig,
) -> Result<LossStats> {
    let normalized_adv = normalize_advantages(&batch.adv);
    let adv_tensor = Tensor::from_vec(normalized_adv, batch.adv.len(), device)?;

    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut total_loss_sum = 0.0;

    for _ in 0..cfg.ppo_epochs {
        let logits = policy.forward(&batch.obs)?;
        let log_probs = log_softmax(&logits, 1)?;
        let actions = batch.actions.unsqueeze(1)?;
        let act_logp = log_probs.gather(&actions, 1)?.squeeze(1)?;
        let ratio = (&act_logp - &batch.logp)?.exp()?;

        let surr1 = (&ratio * &adv_tensor)?;
        let clipped_ratio = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
        let surr2 = (&clipped_ratio * &adv_tensor)?;
        let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
        let policy_loss = surr.neg()?.mean_all()?;

        let value_pred = value.forward(&batch.obs)?.squeeze(1)?;
        let value_loss = (&value_pred - &batch.ret)?.sqr()?.mean_all()?;

        let probs = softmax(&logits, 1)?;
        let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

        let loss = (&policy_loss + (&value_loss * cfg.vf_coef)?)?;
        let loss = (&loss - (&entropy * cfg.ent_coef)?)?;

        opt.backward_step(&loss)?;

        policy_loss_sum += policy_loss.to_scalar::<f32>()? as f64;
        value_loss_sum += value_loss.to_scalar::<f32>()? as f64;
        entropy_sum += entropy.to_scalar::<f32>()? as f64;
        total_loss_sum += loss.to_scalar::<f32>()? as f64;
    }

    let denom = cfg.ppo_epochs.max(1) as f64;
    Ok(LossStats {
        policy_loss: policy_loss_sum / denom,
        value_loss: value_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
    })
}

fn rollout_single(
    data: &DataSet,
    window: (usize, usize),
    policy: &Sequential,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    device: &Device,
    rng: &mut StdRng,
) -> Result<GrpoRollout> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let obs_dim = data.obs_dim;

    let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = cfg.initial_balance;
    let mut prev_equity = cfg.initial_balance;

    let mut obs_buf = Vec::with_capacity(steps * obs_dim);
    let mut act_buf = Vec::with_capacity(steps);
    let mut logp_buf = Vec::with_capacity(steps);
    let mut total_reward = 0.0;
    let mut total_pnl = 0.0;
    let mut ret_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);

    for t in (start + 1)..end {
        let obs = build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            cfg.initial_balance,
        );
        obs_buf.extend_from_slice(&obs);
        let obs_tensor = Tensor::from_vec(obs, (1, obs_dim), device)?;

        let logits = policy.forward(&obs_tensor)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = sample_from_probs(&probs, rng) as u32;
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();

        let action = match action_idx {
            0 => Action::Buy,
            1 => Action::Sell,
            2 => Action::Hold,
            _ => Action::Revert,
        };
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

        act_buf.push(action_idx);
        logp_buf.push(logp_val);
        total_reward += reward;
        total_pnl += info.pnl_change;
        let denom = if prev_equity.abs() < 1e-8 {
            1e-8
        } else {
            prev_equity
        };
        ret_series.push(info.pnl_change / denom);
        equity_curve.push(equity);
        prev_equity = equity;
    }

    Ok(GrpoRollout {
        obs: Tensor::from_vec(obs_buf, (steps, obs_dim), device)?,
        actions: Tensor::from_vec(act_buf, steps, device)?,
        logp: Tensor::from_vec(logp_buf, steps, device)?,
        reward: total_reward,
        pnl: total_pnl,
        returns: ret_series,
        equity: equity_curve,
    })
}

fn rollout_group(
    data: &DataSet,
    windows: &[(usize, usize)],
    policy: &Sequential,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    group_size: usize,
    device: &Device,
    rng: &mut StdRng,
) -> Result<GrpoGroup> {
    let mut rollouts = Vec::with_capacity(group_size);
    for i in 0..group_size {
        let window = windows[i % windows.len()];
        rollouts.push(rollout_single(
            data, window, policy, env_cfg, cfg, device, rng,
        )?);
    }

    let rewards: Vec<f64> = rollouts.iter().map(|r| r.reward).collect();
    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards
        .iter()
        .map(|r| (r - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    let std_reward = variance.sqrt().max(1e-8);

    Ok(GrpoGroup {
        rollouts,
        mean_reward,
        std_reward,
    })
}

fn compute_grpo_advantages(group: &GrpoGroup) -> Vec<f64> {
    group
        .rollouts
        .iter()
        .map(|r| (r.reward - group.mean_reward) / group.std_reward)
        .collect()
}

fn summarize_group(group: &GrpoGroup, annualization: f64) -> RolloutMetrics {
    let all_returns: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|r| r.returns.iter().copied())
        .collect();
    let ret_mean = if all_returns.is_empty() {
        0.0
    } else {
        all_returns.iter().sum::<f64>() / all_returns.len() as f64
    };
    let pnl_total = group.rollouts.iter().map(|r| r.pnl).sum::<f64>();
    let all_equity: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|r| r.equity.iter().copied())
        .collect();
    let sortino = compute_sortino(&all_returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&all_equity);
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        sortino,
        drawdown,
    }
}

fn grpo_update(
    policy: &Sequential,
    opt: &mut AdamW,
    group: &GrpoGroup,
    advantages: &[f64],
    cfg: &GrpoConfig,
) -> Result<GrpoLossStats> {
    let normalized_adv = normalize_advantages_f64(advantages);
    let mut policy_loss_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut total_loss_sum = 0.0;
    let mut kl_sum = 0.0;

    for _ in 0..cfg.grpo_epochs {
        let mut epoch_policy_loss = 0.0;
        let mut epoch_entropy = 0.0;
        let mut epoch_kl = 0.0;

        for (i, rollout) in group.rollouts.iter().enumerate() {
            let logits = policy.forward(&rollout.obs)?;
            let log_probs = log_softmax(&logits, 1)?;
            let act_logp = log_probs
                .gather(&rollout.actions.unsqueeze(1)?, 1)?
                .squeeze(1)?;
            let ratio = (&act_logp - &rollout.logp)?.exp()?;

            let adv = normalized_adv.get(i).copied().unwrap_or(0.0);
            let surr1 = (&ratio * adv)?;
            let clipped_ratio = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
            let surr2 = (&clipped_ratio * adv)?;
            let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
            let policy_loss = surr.neg()?.mean_all()?;

            let probs = softmax(&logits, 1)?;
            let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

            let loss = (&policy_loss - (&entropy * cfg.ent_coef)?)?;
            opt.backward_step(&loss)?;

            epoch_policy_loss += policy_loss.to_scalar::<f32>()? as f64;
            epoch_entropy += entropy.to_scalar::<f32>()? as f64;
            epoch_kl += (&rollout.logp - &act_logp)?
                .mean_all()?
                .abs()?
                .to_scalar::<f32>()? as f64;
        }

        let denom = group.rollouts.len().max(1) as f64;
        policy_loss_sum += epoch_policy_loss / denom;
        entropy_sum += epoch_entropy / denom;
        kl_sum += epoch_kl / denom;
        total_loss_sum += (epoch_policy_loss - cfg.ent_coef * epoch_entropy) / denom;
    }

    let denom = cfg.grpo_epochs.max(1) as f64;
    Ok(GrpoLossStats {
        policy_loss: policy_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        kl_div: kl_sum / denom,
    })
}

fn compute_gae(rewards: &[f64], values: &[f32], gamma: f64, lam: f64) -> (Vec<f32>, Vec<f32>) {
    let mut adv = vec![0.0f32; rewards.len()];
    let mut ret = vec![0.0f32; rewards.len()];
    let mut gae = 0.0f64;
    let mut next_value = 0.0f64;

    for t in (0..rewards.len()).rev() {
        let v = values.get(t).copied().unwrap_or(0.0) as f64;
        let delta = rewards[t] + gamma * next_value - v;
        gae = delta + gamma * lam * gae;
        adv[t] = gae as f32;
        ret[t] = (gae + v) as f32;
        next_value = v;
    }

    (adv, ret)
}

fn normalize_advantages(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std = var.sqrt().max(1e-8);
    values
        .iter()
        .map(|v| ((*v as f64 - mean) / std) as f32)
        .collect()
}

fn normalize_advantages_f64(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std = var.sqrt().max(1e-8);
    values.iter().map(|v| (*v - mean) / std).collect()
}

fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
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
        return probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    WeightedIndex::new(&weights)
        .map(|dist| dist.sample(rng))
        .unwrap_or_else(|_| {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
}

fn average_metrics(values: &[RolloutMetrics]) -> RolloutMetrics {
    if values.is_empty() {
        return RolloutMetrics {
            ret_mean: 0.0,
            pnl: 0.0,
            sortino: 0.0,
            drawdown: 0.0,
        };
    }
    let mut ret_mean = 0.0;
    let mut pnl = 0.0;
    let mut sortino = 0.0;
    let mut drawdown = 0.0;
    for value in values {
        ret_mean += value.ret_mean;
        pnl += value.pnl;
        sortino += value.sortino;
        drawdown += value.drawdown;
    }
    let denom = values.len() as f64;
    RolloutMetrics {
        ret_mean: ret_mean / denom,
        pnl: pnl / denom,
        sortino: sortino / denom,
        drawdown: drawdown / denom,
    }
}

fn average_losses(values: &[LossStats]) -> LossStats {
    if values.is_empty() {
        return LossStats {
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
        };
    }
    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    for value in values {
        policy_loss += value.policy_loss;
        value_loss += value.value_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
    }
    let denom = values.len() as f64;
    LossStats {
        policy_loss: policy_loss / denom,
        value_loss: value_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
    }
}

fn average_grpo_losses(values: &[GrpoLossStats]) -> GrpoLossStats {
    if values.is_empty() {
        return GrpoLossStats {
            policy_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            kl_div: 0.0,
        };
    }
    let mut policy_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    let mut kl_div = 0.0;
    for value in values {
        policy_loss += value.policy_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
        kl_div += value.kl_div;
    }
    let denom = values.len() as f64;
    GrpoLossStats {
        policy_loss: policy_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        kl_div: kl_div / denom,
    }
}

fn resolve_device(requested: ComputeRuntime) -> Result<Device> {
    match requested {
        ComputeRuntime::Cpu => Ok(Device::Cpu),
        ComputeRuntime::Auto => auto_device(),
        ComputeRuntime::Cuda => explicit_cuda_device(),
        ComputeRuntime::Mps => bail!(
            "candle RL backend in this branch does not expose Apple GPU execution; use --device cpu for Candle or use Burn/libtorch for Apple GPU experiments"
        ),
    }
}

fn runtime_from_device(device: &Device) -> ComputeRuntime {
    match device {
        Device::Cuda(_) => ComputeRuntime::Cuda,
        Device::Metal(_) => ComputeRuntime::Mps,
        _ => ComputeRuntime::Cpu,
    }
}

fn print_device(device: &Device) {
    match device {
        Device::Cpu => println!("info: candle backend using cpu"),
        Device::Cuda(_) => println!("info: candle backend using cuda:0"),
        Device::Metal(_) => println!("info: candle backend using mps"),
    }
}

fn auto_device() -> Result<Device> {
    #[cfg(feature = "backend-candle-cuda")]
    {
        if Device::new_cuda(0).is_ok() {
            return Device::new_cuda(0).context("initialize candle cuda device");
        }
    }

    Ok(Device::Cpu)
}

fn explicit_cuda_device() -> Result<Device> {
    #[cfg(feature = "backend-candle-cuda")]
    {
        return Device::new_cuda(0).context("initialize candle cuda device");
    }

    #[cfg(not(feature = "backend-candle-cuda"))]
    {
        bail!(
            "candle cuda support is not compiled into this build; re-run with the 'backend-candle-cuda' Cargo feature"
        )
    }
}

pub(crate) fn load_checkpoint(varmap: &mut VarMap, path: &Path) -> anyhow::Result<()> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        anyhow::bail!(
            "candle RL checkpoints must be .safetensors files (received {})",
            path.display()
        );
    }
    varmap
        .load(path)
        .with_context(|| format!("load candle checkpoint {}", path.display()))
}

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
