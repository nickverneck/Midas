#[path = "candle/grpo.rs"]
mod grpo;
#[path = "candle/logging.rs"]
mod logging;
#[path = "candle/model.rs"]
mod model;
#[path = "candle/ppo.rs"]
mod ppo;
#[path = "candle/rollout.rs"]
mod rollout;
#[path = "candle/runtime.rs"]
mod runtime;
#[path = "candle/util.rs"]
mod util;

use anyhow::{Context, bail};
use candle_core::DType;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use midas_env::env::{EnvConfig, MarginMode};
use midas_env::ml;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

use crate::args::Args;
use crate::common;
use grpo::{GrpoConfig, average_grpo_losses, grpo_update};
use logging::{format_duration, write_rl_log_row};
use model::build_value;
use ppo::{PpoConfig, average_losses, ppo_update};
use rollout::{
    RolloutConfig, average_metrics, compute_grpo_advantages, rollout, rollout_group,
    summarize_batch, summarize_group,
};
use runtime::{print_device, resolve_device, runtime_from_device};

#[allow(unused_imports)]
pub(crate) use model::{Mlp, build_policy, load_checkpoint};

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
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
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
