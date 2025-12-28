#[cfg(not(feature = "tch"))]
fn main() {
    eprintln!("train_ga requires the 'tch' feature. Rebuild with --features tch and ensure LIBTORCH is set.");
    std::process::exit(1);
}

#[cfg(feature = "tch")]
#[path = "train_ga/args.rs"]
mod args;
#[cfg(feature = "tch")]
#[path = "train_ga/data.rs"]
mod data;
#[cfg(feature = "tch")]
#[path = "train_ga/ga.rs"]
mod ga;
#[cfg(feature = "tch")]
#[path = "train_ga/metrics.rs"]
mod metrics;
#[cfg(feature = "tch")]
#[path = "train_ga/model.rs"]
mod model;
#[cfg(feature = "tch")]
#[path = "train_ga/util.rs"]
mod util;

#[cfg(feature = "tch")]
use clap::Parser;

#[cfg(feature = "tch")]
fn main() -> anyhow::Result<()> {
    let args = args::Args::parse();
    run(args)
}

#[cfg(feature = "tch")]
fn run(args: args::Args) -> anyhow::Result<()> {
    use anyhow::Context;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;
    use std::io::Write;

    if let Some(seed) = args.seed {
        let mut rng = StdRng::seed_from_u64(seed);
        let _ = rng.r#gen::<u64>();
    }

    std::fs::create_dir_all(&args.outdir)?;

    let device = util::resolve_device(args.device.as_deref());
    util::print_device(&device);
    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()
            .context("configure rayon thread pool")?;
        println!("info: using {} worker threads", args.workers);
    }

    let (train_path, val_path, test_path) = util::resolve_paths(&args)?;
    let train = data::load_dataset(&train_path, args.globex && !args.rth)?;
    let val = data::load_dataset(&val_path, args.globex && !args.rth)?;
    let test = data::load_dataset(&test_path, args.globex && !args.rth)?;
    if args.debug_data {
        data::dump_dataset_stats("train", &train);
        data::dump_dataset_stats("val", &val);
        data::dump_dataset_stats("test", &test);
    }

    let (margin_cfg, session_cfg) = util::load_symbol_config(&args.symbol_config, &train.symbol)?;
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

    let obs_dim = train.obs_dim;
    let genome_len = model::param_count(obs_dim, args.hidden, args.layers);

    let mut rng = StdRng::from_entropy();
    let normal = Normal::<f32>::new(0.0, args.init_sigma as f32)?;
    let mut pop: Vec<Vec<f32>> = (0..args.pop_size)
        .map(|_| (0..genome_len).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let log_path = args.outdir.join("ga_log.csv");
    if !log_path.exists() {
        std::fs::write(
            &log_path,
            "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n",
        )?;
    }

    let mut best_overall_fitness = f64::NEG_INFINITY;
    let mut best_overall_genome: Option<Vec<f32>> = None;

    for generation in 0..args.generations {
        let gen_start = std::time::Instant::now();
        println!(
            "\nGeneration {} | Evaluating {} candidates (device={:?})",
            generation,
            pop.len(),
            device
        );

    let base_cfg = ga::CandidateConfig {
        initial_balance: args.initial_balance,
        margin_per_contract,
        disable_margin: args.disable_margin,
        w_pnl: args.w_pnl,
        w_sortino: args.w_sortino,
        w_mdd: args.w_mdd,
        sortino_annualization: args.sortino_annualization,
        hidden: args.hidden,
        layers: args.layers,
        eval_windows: args.eval_windows,
        device,
        ignore_session: args.ignore_session,
        drawdown_penalty: args.drawdown_penalty,
        drawdown_penalty_growth: args.drawdown_penalty_growth,
        session_close_penalty: args.session_close_penalty,
        max_hold_bars_positive: args.max_hold_bars_positive,
        max_hold_bars_drawdown: args.max_hold_bars_drawdown,
        invalid_revert_penalty: args.invalid_revert_penalty,
        flat_hold_penalty: args.flat_hold_penalty,
        max_flat_hold_bars: args.max_flat_hold_bars,
        invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: args.flat_hold_penalty_growth,
    };

        let eval_results: Vec<(usize, ga::CandidateResult, Option<ga::CandidateResult>)> = pop
            .par_iter()
            .enumerate()
            .map(|(idx, genome)| {
                let train_metrics = ga::evaluate_candidate(genome, &train, &windows_train, &base_cfg, false);
                let eval_metrics = if args.skip_val_eval {
                    None
                } else {
                    Some(ga::evaluate_candidate(genome, &val, &windows_val, &base_cfg, true))
                };
                (idx, train_metrics, eval_metrics)
            })
            .collect();

        let mut scored: Vec<(f64, Vec<f32>, Option<ga::CandidateResult>, ga::CandidateResult)> =
            Vec::with_capacity(pop.len());

        let mut results = eval_results;
        results.sort_by_key(|(idx, _, _)| *idx);

        for (idx, train_metrics, eval_metrics) in results.into_iter() {
            let genome = pop[idx].clone();
            scored.push((train_metrics.fitness, genome.clone(), eval_metrics.clone(), train_metrics.clone()));

            let eval_pnl = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_pnl))
                .unwrap_or_default();
            let eval_pnl_realized = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_pnl_realized))
                .unwrap_or_default();
            let eval_pnl_total = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_pnl_total))
                .unwrap_or_default();
            let eval_sortino = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_sortino))
                .unwrap_or_default();
            let eval_dd = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_drawdown))
                .unwrap_or_default();
            let eval_ret = eval_metrics
                .as_ref()
                .map(|m| format!("{:.8}", m.eval_ret_mean))
                .unwrap_or_default();

            let line = format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                generation,
                idx,
                args.w_pnl,
                args.w_sortino,
                args.w_mdd,
                train_metrics.fitness,
                eval_pnl,
                eval_pnl_realized,
                eval_pnl_total,
                eval_sortino,
                eval_dd,
                eval_ret,
                train_metrics.eval_pnl,
                train_metrics.eval_pnl_realized,
                train_metrics.eval_pnl_total,
                train_metrics.eval_sortino,
                train_metrics.eval_drawdown,
                train_metrics.eval_ret_mean
            );
            std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)?
                .write_all(line.as_bytes())
                .context("write ga_log")?;

            if let Some(eval) = eval_metrics {
                println!(
                    "  cand {}/{} | fitness {:.2} | train pnl {:.2} | val pnl {:.2}",
                    idx,
                    pop.len().saturating_sub(1),
                    train_metrics.fitness,
                    train_metrics.eval_pnl,
                    eval.eval_pnl
                );
            } else {
                println!(
                    "  cand {}/{} | fitness {:.2} | train pnl {:.2} | val skipped",
                    idx,
                    pop.len().saturating_sub(1),
                    train_metrics.fitness,
                    train_metrics.eval_pnl
                );
            }
            if args.debug_data && idx == 0 {
                println!(
                    "  debug cand0 | non_hold {} | non_zero_pos {} | mean_abs_pnl {:.6}",
                    train_metrics.debug_non_hold,
                    train_metrics.debug_non_zero_pos,
                    train_metrics.debug_mean_abs_pnl
                );
                println!(
                    "  debug cand0 | pnl realized {:.2} | pnl total {:.2} | dd_penalty {:.4}",
                    train_metrics.eval_pnl_realized,
                    train_metrics.eval_pnl_total,
                    train_metrics.debug_drawdown_penalty
                );
                println!(
                    "  debug cand0 | buy {} | sell {} | hold {} | revert {}",
                    train_metrics.debug_buy,
                    train_metrics.debug_sell,
                    train_metrics.debug_hold,
                    train_metrics.debug_revert
                );
                println!(
                    "  debug cand0 | session_violation {} | margin_violation {} | position_violation {}",
                    train_metrics.debug_session_violations,
                    train_metrics.debug_margin_violations,
                    train_metrics.debug_position_violations
                );
            }
        }

        println!("Generation {} completed in {:.2?}", generation, gen_start.elapsed());

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let top = scored.iter().take(5).collect::<Vec<_>>();
        for (rank, (_fit, genome, eval_metrics, train_metrics)) in top.iter().enumerate() {
            let policy_path = args
                .outdir
                .join(format!("policy_gen{}_rank{}.pt", generation, rank));
            model::save_policy(obs_dim, args.hidden, args.layers, device, genome, &policy_path)?;
            if rank == 0 {
                if let Some(eval) = eval_metrics.as_ref() {
                    println!(
                        "Gen {} Top Performer: train fitness {:.2}, val fitness {:.2}",
                        generation, train_metrics.fitness, eval.fitness
                    );
                } else {
                    println!(
                        "Gen {} Top Performer: train fitness {:.2}",
                        generation, train_metrics.fitness
                    );
                }
                if train_metrics.fitness > best_overall_fitness {
                    best_overall_fitness = train_metrics.fitness;
                    best_overall_genome = Some(genome.clone());
                }
            }
        }

        let elite_n = (args.elite_frac * args.pop_size as f64).round().max(1.0) as usize;
        let elites: Vec<Vec<f32>> = scored.iter().take(elite_n).map(|(_, g, _, _)| g.clone()).collect();
        let mut new_pop = elites.clone();

        let normal_mut = Normal::<f32>::new(0.0, args.mutation_sigma as f32)?;
        while new_pop.len() < args.pop_size {
            let parent_a = elites.choose(&mut rng).unwrap();
            let parent_b = elites.choose(&mut rng).unwrap();
            let mut child = ga::crossover(parent_a, parent_b, &mut rng);
            for v in child.iter_mut() {
                *v += normal_mut.sample(&mut rng);
            }
            new_pop.push(child);
        }
        pop = new_pop;

        let ckpt_path = args.outdir.join(format!("checkpoint_gen{}.bin", generation));
        ga::save_checkpoint(&ckpt_path, &pop)?;
    }

    if let Some(best_genome) = best_overall_genome {
        let base_cfg = ga::CandidateConfig {
            initial_balance: args.initial_balance,
            margin_per_contract,
            disable_margin: args.disable_margin,
            w_pnl: args.w_pnl,
            w_sortino: args.w_sortino,
            w_mdd: args.w_mdd,
            sortino_annualization: args.sortino_annualization,
            hidden: args.hidden,
            layers: args.layers,
            eval_windows: args.eval_windows,
            device,
            ignore_session: args.ignore_session,
            drawdown_penalty: args.drawdown_penalty,
            drawdown_penalty_growth: args.drawdown_penalty_growth,
            session_close_penalty: args.session_close_penalty,
            max_hold_bars_positive: args.max_hold_bars_positive,
            max_hold_bars_drawdown: args.max_hold_bars_drawdown,
            invalid_revert_penalty: args.invalid_revert_penalty,
            flat_hold_penalty: args.flat_hold_penalty,
            max_flat_hold_bars: args.max_flat_hold_bars,
            invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
            flat_hold_penalty_growth: args.flat_hold_penalty_growth,
        };
        let metrics = ga::evaluate_candidate(&best_genome, &test, &windows_test, &base_cfg, false);
        println!(
            "test | fitness {:.2} | pnl {:.2} | sortino {:.2} | mdd {:.2}",
            metrics.fitness, metrics.eval_pnl, metrics.eval_sortino, metrics.eval_drawdown
        );

        let best_path = args.outdir.join("best_overall_policy.pt");
        model::save_policy(obs_dim, args.hidden, args.layers, device, &best_genome, &best_path)?;
        println!("Saved best overall policy to {}", best_path.display());
    }

    Ok(())
}
