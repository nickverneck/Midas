#[path = "train_rl/args.rs"]
mod args;
#[path = "train_rl/data.rs"]
mod data;
#[path = "train_rl/metrics.rs"]
mod metrics;
#[path = "train_rl/model.rs"]
mod model;
#[path = "train_rl/ppo.rs"]
mod ppo;
#[path = "train_rl/util.rs"]
mod util;

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use midas_env::env::{EnvConfig, MarginMode};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand::rngs::StdRng;
use tch::nn;
use tch::nn::OptimizerConfig;

use args::Args;
use ppo::{LossStats, RolloutConfig, RolloutMetrics};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    std::fs::create_dir_all(&args.outdir)?;

    let device = util::resolve_device(args.device.as_deref());
    util::print_device(&device);

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().r#gen());
    let mut rng = StdRng::seed_from_u64(seed);

    let (train_path, val_path, test_path) = util::resolve_paths(&args)?;
    let train = data::load_dataset(&train_path, args.globex && !args.rth)?;
    let val = data::load_dataset(&val_path, args.globex && !args.rth)?;
    let test = data::load_dataset(&test_path, args.globex && !args.rth)?;

    let (margin_cfg, session_cfg) = util::load_symbol_config(&args.symbol_config, &train.symbol)?;
    let margin_mode = match args.margin_mode.as_str() {
        "per-contract" => MarginMode::PerContract,
        "price" => MarginMode::Price,
        _ => util::infer_margin_mode(&train.symbol, margin_cfg),
    };
    let contract_multiplier = if args.contract_multiplier > 0.0 {
        args.contract_multiplier
    } else {
        1.0
    };
    let margin_per_contract = args
        .margin_per_contract
        .or(margin_cfg)
        .unwrap_or_else(|| util::infer_margin(&train.symbol));
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
    let windows_train = if full_file {
        vec![(0, train.close.len())]
    } else {
        midas_env::sampler::windows(train.close.len(), args.window, args.step)
    };
    let windows_val = if full_file {
        vec![(0, val.close.len())]
    } else {
        midas_env::sampler::windows(val.close.len(), args.window, args.step)
    };
    let windows_test = if full_file {
        vec![(0, test.close.len())]
    } else {
        midas_env::sampler::windows(test.close.len(), args.window, args.step)
    };

    if windows_train.is_empty() {
        anyhow::bail!("no training windows available");
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
        max_hold_bars_positive: args.max_hold_bars_positive,
        max_hold_bars_drawdown: args.max_hold_bars_drawdown,
        hold_duration_penalty: args.hold_duration_penalty,
        hold_duration_penalty_growth: args.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
        invalid_revert_penalty: args.invalid_revert_penalty,
        invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
        flat_hold_penalty: args.flat_hold_penalty,
        flat_hold_penalty_growth: args.flat_hold_penalty_growth,
        max_flat_hold_bars: args.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let obs_dim = train.obs_dim as i64;
    let action_dim = 4i64;

    let mut vs = nn::VarStore::new(device);
    let policy = model::build_policy(&vs.root().sub("policy"), obs_dim, args.hidden as i64, args.layers, action_dim);
    let value = model::build_value(&vs.root().sub("value"), obs_dim, args.hidden as i64, args.layers);

    if let Some(path) = &args.load_checkpoint {
        load_checkpoint(&mut vs, device, path)?;
        println!("info: loaded checkpoint {}", path.display());
    }

    let mut opt = nn::Adam::default()
        .build(&vs, args.lr)
        .context("build optimizer")?;

    let log_path = args.outdir.join("rl_log.csv");
    if !log_path.exists() || std::fs::metadata(&log_path)?.len() == 0 {
        std::fs::write(
            &log_path,
            "epoch,train_ret_mean,train_pnl,train_sortino,train_drawdown,eval_ret_mean,eval_pnl,eval_sortino,eval_drawdown,fitness,policy_loss,value_loss,entropy,total_loss\n",
        )?;
    }

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
        let mut loss_stats = Vec::with_capacity(train_count);

        for window in train_windows.iter().take(train_count) {
            let batch = ppo::rollout(&train, *window, &policy, &value, &env_cfg, &rollout_cfg);
            let metrics = ppo::summarize_batch(&batch, args.sortino_annualization);
            let losses = ppo::ppo_update(&policy, &value, &mut opt, &batch, &ppo_cfg);
            train_metrics.push(metrics);
            loss_stats.push(losses);
        }

        let train_summary = average_metrics(&train_metrics);
        let loss_summary = average_losses(&loss_stats);

        let eval_count = args.eval_windows.min(windows_val.len()).max(1);
        let mut eval_metrics = Vec::with_capacity(eval_count);
        for window in windows_val.iter().take(eval_count) {
            let batch = ppo::rollout(&val, *window, &policy, &value, &env_cfg, &rollout_cfg);
            eval_metrics.push(ppo::summarize_batch(&batch, args.sortino_annualization));
        }
        let eval_summary = average_metrics(&eval_metrics);

        let fitness_source = if args.fitness_use_eval {
            eval_summary
        } else {
            train_summary
        };
        let fitness = (args.w_pnl * fitness_source.pnl)
            + (args.w_sortino * fitness_source.sortino)
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
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)?;
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

        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            let ckpt_path = args.outdir.join(format!("checkpoint_epoch{}.pt", epoch));
            vs.save(&ckpt_path)?;
        }
    }

    let eval_count = args.eval_windows.min(windows_test.len()).max(1);
    let mut test_metrics = Vec::with_capacity(eval_count);
    for window in windows_test.iter().take(eval_count) {
        let batch = ppo::rollout(&test, *window, &policy, &value, &env_cfg, &rollout_cfg);
        test_metrics.push(ppo::summarize_batch(&batch, args.sortino_annualization));
    }
    let test_summary = average_metrics(&test_metrics);
    println!(
        "test | ret {:.4} | pnl {:.4} | sortino {:.4} | mdd {:.4}",
        test_summary.ret_mean,
        test_summary.pnl,
        test_summary.sortino,
        test_summary.drawdown
    );
    println!("total training time: {}", format_duration(training_start.elapsed()));

    let final_path = args.outdir.join("ppo_final.pt");
    vs.save(&final_path)?;
    println!("Saved final PPO checkpoint to {}", final_path.display());

    Ok(())
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
    for v in values {
        ret_mean += v.ret_mean;
        pnl += v.pnl;
        sortino += v.sortino;
        drawdown += v.drawdown;
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
    for v in values {
        policy_loss += v.policy_loss;
        value_loss += v.value_loss;
        entropy += v.entropy;
        total_loss += v.total_loss;
    }
    let denom = values.len() as f64;
    LossStats {
        policy_loss: policy_loss / denom,
        value_loss: value_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
    }
}

fn load_checkpoint(vs: &mut nn::VarStore, device: tch::Device, path: &Path) -> anyhow::Result<()> {
    match vs.load(path) {
        Ok(()) => return Ok(()),
        Err(err) => {
            let err_msg = err.to_string();
            eprintln!(
                "warn: VarStore load failed for {} ({}), trying TorchScript fallback",
                path.display(),
                err_msg
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
    apply_named_parameters(vs, params, path)
}

fn apply_named_parameters(
    vs: &mut nn::VarStore,
    params: Vec<(String, tch::Tensor)>,
    path: &Path,
) -> anyhow::Result<()> {
    let mut by_name = std::collections::HashMap::with_capacity(params.len());
    for (name, tensor) in params {
        let name = name.replace('|', ".");
        by_name.insert(name, tensor);
    }

    let mut variables = vs.variables_.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        let Some(src) = by_name.get(name) else {
            anyhow::bail!("checkpoint missing tensor {name} in {}", path.display());
        };
        tch::no_grad(|| {
            var.set_data(&var.to_kind(src.kind()));
            var.f_copy_(src)
        })
        .with_context(|| format!("copy tensor {name} from {}", path.display()))?;
    }

    Ok(())
}
