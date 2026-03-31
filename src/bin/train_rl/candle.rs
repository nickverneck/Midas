use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::{
    AdamW, Dropout, Linear, Module, ModuleT, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear,
    ops::{log_softmax, softmax},
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
    realized_pnl: f64,
    sortino: f64,
    drawdown: f64,
    commission: f64,
    slippage: f64,
    buy_frac: f64,
    sell_frac: f64,
    hold_frac: f64,
    revert_frac: f64,
    mean_max_prob: f64,
    entries: f64,
    exits: f64,
    flips: f64,
    avg_hold: f64,
}

#[derive(Debug, Clone, Copy)]
struct LossStats {
    policy_loss: f64,
    value_loss: f64,
    entropy: f64,
    total_loss: f64,
    policy_grad_norm: f64,
    value_grad_norm: f64,
    approx_kl: f64,
    clip_frac: f64,
}

#[derive(Debug, Clone, Copy)]
struct GrpoLossStats {
    policy_loss: f64,
    entropy: f64,
    total_loss: f64,
    kl_div: f64,
    policy_grad_norm: f64,
    clip_frac: f64,
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
    realized_pnl: f64,
    commission: f64,
    slippage: f64,
    action_counts: [usize; 4],
    max_prob_sum: f64,
    entries: usize,
    exits: usize,
    flips: usize,
    total_trade_bars: usize,
}

struct GrpoRollout {
    obs: Tensor,
    actions: Tensor,
    logp: Tensor,
    reward: f64,
    pnl: f64,
    returns: Vec<f64>,
    equity: Vec<f64>,
    realized_pnl: f64,
    commission: f64,
    slippage: f64,
    action_counts: [usize; 4],
    max_prob_sum: f64,
    entries: usize,
    exits: usize,
    flips: usize,
    total_trade_bars: usize,
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

pub(crate) struct Mlp {
    hidden_layers: Vec<Linear>,
    out: Linear,
    dropout: Option<Dropout>,
}

impl Mlp {
    pub(crate) fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.hidden_layers.iter() {
            xs = layer.forward(&xs)?;
            xs = xs.tanh()?;
            if let Some(dropout) = &self.dropout {
                xs = dropout.forward_t(&xs, train)?;
            }
        }
        Ok(self.out.forward(&xs)?)
    }
}

pub fn run(args: Args, mut stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    println!("info: run directory {}", args.outdir.display());

    if !(0.0..1.0).contains(&args.dropout) {
        bail!("--dropout must be in [0, 1), got {}", args.dropout);
    }

    let device = resolve_device(stack.requested_runtime)?;
    stack.effective_runtime = runtime_from_device(&device);
    ml::write_run_metadata(
        &args.outdir.join("training_stack.json"),
        &stack,
        Some(&args.algorithm),
        Some(crate::data::OBSERVATION_SCHEMA_NORMALIZED),
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
        args.dropout,
    )?;
    let value = if use_grpo {
        None
    } else {
        Some(build_value(
            var_builder.pp("value"),
            obs_dim,
            args.hidden,
            args.layers,
            args.dropout,
        )?)
    };

    if args.dropout > 0.0 {
        println!(
            "info: candle dropout set to {:.3} on hidden layers during training; eval/test disable dropout",
            args.dropout
        );
    }

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
    common::ensure_csv_header(&log_path, common::RL_LOG_HEADER_V2)?;

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
                    true,
                    grpo_cfg.group_size,
                    &device,
                    &mut rng,
                    false,
                )?;
                let advantages = compute_grpo_advantages(&group);
                let losses =
                    grpo_update(&policy, &varmap, &mut opt, &group, &advantages, &grpo_cfg)?;
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
                    true,
                    &device,
                    &mut rng,
                    false,
                )?;
                let metrics = summarize_batch(&batch, args.sortino_annualization);
                let losses = ppo_update(
                    &policy,
                    value.as_ref().expect("value network"),
                    &varmap,
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
                    false,
                    1,
                    &device,
                    &mut rng,
                    false,
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
                    false,
                    &device,
                    &mut rng,
                    false,
                )?;
                eval_metrics.push(summarize_batch(&batch, args.sortino_annualization));
            }
        }
        let eval_summary = average_metrics(&eval_metrics);

        let probe_summary = if args.log_interval > 0 && epoch % args.log_interval == 0 {
            windows_val
                .first()
                .map(|window| {
                    if use_grpo {
                        rollout_group(
                            &val,
                            &[*window],
                            &policy,
                            &env_cfg,
                            &rollout_cfg,
                            false,
                            1,
                            &device,
                            &mut rng,
                            true,
                        )
                        .map(|group| summarize_group(&group, args.sortino_annualization))
                    } else {
                        rollout(
                            &val,
                            *window,
                            &policy,
                            value.as_ref().expect("value network"),
                            &env_cfg,
                            &rollout_cfg,
                            false,
                            &device,
                            &mut rng,
                            true,
                        )
                        .map(|batch| summarize_batch(&batch, args.sortino_annualization))
                    }
                })
                .transpose()?
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
                false,
                1,
                &device,
                &mut rng,
                false,
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
                false,
                &device,
                &mut rng,
                false,
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
    dropout: f64,
) -> Result<Mlp> {
    build_mlp(vb, input_dim, hidden, layers, action_dim, dropout)
}

fn build_value(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    dropout: f64,
) -> Result<Mlp> {
    build_mlp(vb, input_dim, hidden, layers, 1, dropout)
}

fn build_mlp(
    vb: VarBuilder,
    input_dim: usize,
    hidden: usize,
    layers: usize,
    output_dim: usize,
    dropout: f64,
) -> Result<Mlp> {
    let mut hidden_layers = Vec::with_capacity(layers);
    let mut in_dim = input_dim;
    for i in 0..layers {
        hidden_layers.push(linear(in_dim, hidden, vb.pp(format!("layer_{i}")))?);
        in_dim = hidden;
    }
    Ok(Mlp {
        hidden_layers,
        out: linear(in_dim, output_dim, vb.pp("out"))?,
        dropout: (dropout > 0.0).then(|| Dropout::new(dropout as f32)),
    })
}

fn rollout(
    data: &DataSet,
    window: (usize, usize),
    policy: &Mlp,
    value: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
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
    let mut realized_pnl = 0.0f64;
    let mut commission = 0.0f64;
    let mut slippage = 0.0f64;
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

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

        let logits = policy.forward(&obs_tensor, train)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = if greedy {
            argmax_index(&probs) as u32
        } else {
            sample_from_probs(&probs, rng) as u32
        };
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();
        let max_prob = probs
            .iter()
            .copied()
            .fold(0.0f32, |acc, value| acc.max(value)) as f64;
        let value_val = value
            .forward(&obs_tensor, train)?
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

        let position_before = position;
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
        let position_after = position;
        let step_idx = t.saturating_sub(start + 1);

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
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += max_prob;
        if let Some(count) = action_counts.get_mut(action_idx as usize) {
            *count += 1;
        }
        if position_before == 0 && position_after != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position_after == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before.signum() != position_after.signum()
            && position_before != 0
            && position_after != 0
        {
            flips += 1;
        }
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
        realized_pnl,
        commission,
        slippage,
        action_counts,
        max_prob_sum,
        entries,
        exits,
        flips,
        total_trade_bars,
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
    let steps = batch.returns.len().max(1) as f64;
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        realized_pnl: batch.realized_pnl,
        sortino,
        drawdown,
        commission: batch.commission,
        slippage: batch.slippage,
        buy_frac: batch.action_counts[0] as f64 / steps,
        sell_frac: batch.action_counts[1] as f64 / steps,
        hold_frac: batch.action_counts[2] as f64 / steps,
        revert_frac: batch.action_counts[3] as f64 / steps,
        mean_max_prob: batch.max_prob_sum / steps,
        entries: batch.entries as f64,
        exits: batch.exits as f64,
        flips: batch.flips as f64,
        avg_hold: if batch.exits > 0 {
            batch.total_trade_bars as f64 / batch.exits as f64
        } else {
            0.0
        },
    }
}

fn ppo_update(
    policy: &Mlp,
    value: &Mlp,
    varmap: &VarMap,
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
    let mut policy_grad_norm_sum = 0.0;
    let mut value_grad_norm_sum = 0.0;
    let mut approx_kl_sum = 0.0;
    let mut clip_frac_sum = 0.0;

    for _ in 0..cfg.ppo_epochs {
        let logits = policy.forward(&batch.obs, true)?;
        let log_probs = log_softmax(&logits, 1)?;
        let actions = batch.actions.unsqueeze(1)?;
        let act_logp = log_probs.gather(&actions, 1)?.squeeze(1)?;
        let log_ratio = (&act_logp - &batch.logp)?;
        let ratio = log_ratio.exp()?;

        let surr1 = (&ratio * &adv_tensor)?;
        let clipped_ratio = ratio.clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
        let surr2 = (&clipped_ratio * &adv_tensor)?;
        let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
        let policy_loss = surr.neg()?.mean_all()?;

        let value_pred = value.forward(&batch.obs, true)?.squeeze(1)?;
        let value_loss = (&value_pred - &batch.ret)?.sqr()?.mean_all()?;

        let probs = softmax(&logits, 1)?;
        let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

        let loss = (&policy_loss + (&value_loss * cfg.vf_coef)?)?;
        let loss = (&loss - (&entropy * cfg.ent_coef)?)?;

        let grads = loss.backward()?;
        policy_grad_norm_sum += named_grad_l2_norm(varmap, "policy", &grads)?;
        value_grad_norm_sum += named_grad_l2_norm(varmap, "value", &grads)?;
        opt.step(&grads)?;

        let ratio_values = ratio.to_vec1::<f32>()?;
        let log_ratio_values = log_ratio.to_vec1::<f32>()?;
        let denom = ratio_values.len().max(1) as f64;
        let approx_kl = ratio_values
            .iter()
            .zip(log_ratio_values.iter())
            .map(|(ratio_value, log_ratio_value)| {
                (*ratio_value as f64 - 1.0) - *log_ratio_value as f64
            })
            .sum::<f64>()
            / denom;
        let clip_frac = ratio_values
            .iter()
            .filter(|ratio_value| {
                let ratio_value = **ratio_value as f64;
                ratio_value < 1.0 - cfg.clip || ratio_value > 1.0 + cfg.clip
            })
            .count() as f64
            / denom;

        policy_loss_sum += policy_loss.to_scalar::<f32>()? as f64;
        value_loss_sum += value_loss.to_scalar::<f32>()? as f64;
        entropy_sum += entropy.to_scalar::<f32>()? as f64;
        total_loss_sum += loss.to_scalar::<f32>()? as f64;
        approx_kl_sum += approx_kl;
        clip_frac_sum += clip_frac;
    }

    let denom = cfg.ppo_epochs.max(1) as f64;
    Ok(LossStats {
        policy_loss: policy_loss_sum / denom,
        value_loss: value_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        value_grad_norm: value_grad_norm_sum / denom,
        approx_kl: approx_kl_sum / denom,
        clip_frac: clip_frac_sum / denom,
    })
}

fn rollout_single(
    data: &DataSet,
    window: (usize, usize),
    policy: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
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
    let mut realized_pnl = 0.0f64;
    let mut commission = 0.0f64;
    let mut slippage = 0.0f64;
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

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

        let logits = policy.forward(&obs_tensor, train)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = if greedy {
            argmax_index(&probs) as u32
        } else {
            sample_from_probs(&probs, rng) as u32
        };
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();
        let max_prob = probs
            .iter()
            .copied()
            .fold(0.0f32, |acc, value| acc.max(value)) as f64;

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

        let position_before = position;
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
        let position_after = position;
        let step_idx = t.saturating_sub(start + 1);

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
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += max_prob;
        if let Some(count) = action_counts.get_mut(action_idx as usize) {
            *count += 1;
        }
        if position_before == 0 && position_after != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position_after == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before.signum() != position_after.signum()
            && position_before != 0
            && position_after != 0
        {
            flips += 1;
        }
    }

    Ok(GrpoRollout {
        obs: Tensor::from_vec(obs_buf, (steps, obs_dim), device)?,
        actions: Tensor::from_vec(act_buf, steps, device)?,
        logp: Tensor::from_vec(logp_buf, steps, device)?,
        reward: total_reward,
        pnl: total_pnl,
        returns: ret_series,
        equity: equity_curve,
        realized_pnl,
        commission,
        slippage,
        action_counts,
        max_prob_sum,
        entries,
        exits,
        flips,
        total_trade_bars,
    })
}

fn rollout_group(
    data: &DataSet,
    windows: &[(usize, usize)],
    policy: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    group_size: usize,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
) -> Result<GrpoGroup> {
    let mut rollouts = Vec::with_capacity(group_size);
    for i in 0..group_size {
        let window = windows[i % windows.len()];
        rollouts.push(rollout_single(
            data, window, policy, env_cfg, cfg, train, device, rng, greedy,
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
    let realized_pnl = group.rollouts.iter().map(|r| r.realized_pnl).sum::<f64>();
    let commission = group.rollouts.iter().map(|r| r.commission).sum::<f64>();
    let slippage = group.rollouts.iter().map(|r| r.slippage).sum::<f64>();
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut steps = 0usize;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut total_trade_bars = 0usize;
    for rollout in &group.rollouts {
        for (idx, count) in rollout.action_counts.iter().enumerate() {
            action_counts[idx] += *count;
            steps += *count;
        }
        max_prob_sum += rollout.max_prob_sum;
        entries += rollout.entries;
        exits += rollout.exits;
        flips += rollout.flips;
        total_trade_bars += rollout.total_trade_bars;
    }
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
        realized_pnl,
        sortino,
        drawdown,
        commission,
        slippage,
        buy_frac: action_counts[0] as f64 / steps.max(1) as f64,
        sell_frac: action_counts[1] as f64 / steps.max(1) as f64,
        hold_frac: action_counts[2] as f64 / steps.max(1) as f64,
        revert_frac: action_counts[3] as f64 / steps.max(1) as f64,
        mean_max_prob: max_prob_sum / steps.max(1) as f64,
        entries: entries as f64,
        exits: exits as f64,
        flips: flips as f64,
        avg_hold: if exits > 0 {
            total_trade_bars as f64 / exits as f64
        } else {
            0.0
        },
    }
}

fn grpo_update(
    policy: &Mlp,
    varmap: &VarMap,
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
    let mut policy_grad_norm_sum = 0.0;
    let mut clip_frac_sum = 0.0;

    for _ in 0..cfg.grpo_epochs {
        let mut epoch_policy_loss = 0.0;
        let mut epoch_entropy = 0.0;
        let mut epoch_kl = 0.0;
        let mut epoch_clip_frac = 0.0;

        for (i, rollout) in group.rollouts.iter().enumerate() {
            let logits = policy.forward(&rollout.obs, true)?;
            let log_probs = log_softmax(&logits, 1)?;
            let act_logp = log_probs
                .gather(&rollout.actions.unsqueeze(1)?, 1)?
                .squeeze(1)?;
            let ratio = (&act_logp - &rollout.logp)?.exp()?;

            let adv = normalized_adv.get(i).copied().unwrap_or(0.0);
            let surr1 = (&ratio * adv)?;
            let clipped_ratio = ratio.clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
            let surr2 = (&clipped_ratio * adv)?;
            let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
            let policy_loss = surr.neg()?.mean_all()?;

            let probs = softmax(&logits, 1)?;
            let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

            let loss = (&policy_loss - (&entropy * cfg.ent_coef)?)?;
            let grads = loss.backward()?;
            policy_grad_norm_sum += named_grad_l2_norm(varmap, "policy", &grads)?;
            opt.step(&grads)?;

            epoch_policy_loss += policy_loss.to_scalar::<f32>()? as f64;
            epoch_entropy += entropy.to_scalar::<f32>()? as f64;
            epoch_kl += (&rollout.logp - &act_logp)?
                .mean_all()?
                .abs()?
                .to_scalar::<f32>()? as f64;
            let ratio_values = ratio.to_vec1::<f32>()?;
            let denom = ratio_values.len().max(1) as f64;
            epoch_clip_frac += ratio_values
                .iter()
                .filter(|ratio_value| {
                    let ratio_value = **ratio_value as f64;
                    ratio_value < 1.0 - cfg.clip || ratio_value > 1.0 + cfg.clip
                })
                .count() as f64
                / denom;
        }

        let denom = group.rollouts.len().max(1) as f64;
        policy_loss_sum += epoch_policy_loss / denom;
        entropy_sum += epoch_entropy / denom;
        kl_sum += epoch_kl / denom;
        clip_frac_sum += epoch_clip_frac / denom;
        total_loss_sum += (epoch_policy_loss - cfg.ent_coef * epoch_entropy) / denom;
    }

    let denom = cfg.grpo_epochs.max(1) as f64;
    Ok(GrpoLossStats {
        policy_loss: policy_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        kl_div: kl_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        clip_frac: clip_frac_sum / denom,
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
        .unwrap_or_else(|_| argmax_index(probs))
}

fn argmax_index(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn named_grad_l2_norm(
    varmap: &VarMap,
    prefix: &str,
    grads: &candle_core::backprop::GradStore,
) -> Result<f64> {
    let vars = varmap.data().lock().unwrap();
    let mut total = 0.0f64;
    for (name, var) in vars.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        let Some(grad) = grads.get(var.as_tensor()) else {
            continue;
        };
        total += grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    }
    Ok(total.sqrt())
}

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
    for value in values {
        ret_mean += value.ret_mean;
        pnl += value.pnl;
        realized_pnl += value.realized_pnl;
        sortino += value.sortino;
        drawdown += value.drawdown;
        commission += value.commission;
        slippage += value.slippage;
        buy_frac += value.buy_frac;
        sell_frac += value.sell_frac;
        hold_frac += value.hold_frac;
        revert_frac += value.revert_frac;
        mean_max_prob += value.mean_max_prob;
        entries += value.entries;
        exits += value.exits;
        flips += value.flips;
        avg_hold += value.avg_hold;
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
    for value in values {
        policy_loss += value.policy_loss;
        value_loss += value.value_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
        policy_grad_norm += value.policy_grad_norm;
        value_grad_norm += value.value_grad_norm;
        approx_kl += value.approx_kl;
        clip_frac += value.clip_frac;
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

fn average_grpo_losses(values: &[GrpoLossStats]) -> GrpoLossStats {
    if values.is_empty() {
        return GrpoLossStats {
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
    for value in values {
        policy_loss += value.policy_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
        kl_div += value.kl_div;
        policy_grad_norm += value.policy_grad_norm;
        clip_frac += value.clip_frac;
    }
    let denom = values.len() as f64;
    GrpoLossStats {
        policy_loss: policy_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        kl_div: kl_div / denom,
        policy_grad_norm: policy_grad_norm / denom,
        clip_frac: clip_frac / denom,
    }
}

fn format_log_f64(value: f64, precision: usize) -> String {
    format!("{value:.precision$}")
}

fn format_log_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|inner| format_log_f64(inner, precision))
        .unwrap_or_default()
}

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
) -> Result<()> {
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
