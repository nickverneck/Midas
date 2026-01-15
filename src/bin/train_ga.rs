#[path = "train_ga/args.rs"]
mod args;
#[path = "train_ga/data.rs"]
mod data;
#[path = "train_ga/ga.rs"]
mod ga;
#[path = "train_ga/metrics.rs"]
mod metrics;
#[path = "train_ga/model.rs"]
mod model;
#[path = "train_ga/util.rs"]
mod util;

use clap::Parser;
use std::path::Path;
use midas_env::env::MarginMode;

fn write_behavior_csv(
    path: &Path,
    generation: usize,
    candidate_idx: usize,
    split: &str,
    data: &data::DataSet,
    metrics: &ga::CandidateResult,
    rows: &[ga::BehaviorRow],
) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "gen",
        "idx",
        "split",
        "window",
        "step",
        "data_idx",
        "datetime_ns",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "action",
        "action_idx",
        "position_before",
        "position_after",
        "equity_before",
        "equity_after",
        "cash",
        "unrealized_pnl",
        "realized_pnl",
        "pnl_change",
        "realized_pnl_change",
        "reward",
        "commission_paid",
        "slippage_paid",
        "drawdown_penalty",
        "session_close_penalty",
        "invalid_revert_penalty",
        "hold_duration_penalty",
        "flat_hold_penalty",
        "session_open",
        "margin_ok",
        "minutes_to_close",
        "session_closed_violation",
        "margin_call_violation",
        "position_limit_violation",
        "fitness",
        "pnl",
        "pnl_realized",
        "pnl_total",
        "sortino",
        "drawdown",
        "ret_mean",
    ])?;
    let symbol = data.symbol.as_str();
    for row in rows {
        let data_idx = row.data_idx;
        let datetime_ns = data
            .datetime_ns
            .as_ref()
            .and_then(|d| d.get(data_idx))
            .copied();
        let open = data.open.get(data_idx).copied();
        let high = data._high.get(data_idx).copied();
        let low = data._low.get(data_idx).copied();
        let close = data.close.get(data_idx).copied();
        let volume = data
            .volume
            .as_ref()
            .and_then(|v| v.get(data_idx))
            .copied();

        wtr.write_record([
            generation.to_string(),
            candidate_idx.to_string(),
            split.to_string(),
            row.window_idx.to_string(),
            row.step.to_string(),
            row.data_idx.to_string(),
            datetime_ns.map(|v| v.to_string()).unwrap_or_default(),
            symbol.to_string(),
            open.map(|v| v.to_string()).unwrap_or_default(),
            high.map(|v| v.to_string()).unwrap_or_default(),
            low.map(|v| v.to_string()).unwrap_or_default(),
            close.map(|v| v.to_string()).unwrap_or_default(),
            volume.map(|v| v.to_string()).unwrap_or_default(),
            row.action.clone(),
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
            row.session_open.to_string(),
            row.margin_ok.to_string(),
            row.minutes_to_close.map(|v| v.to_string()).unwrap_or_default(),
            row.session_closed_violation.to_string(),
            row.margin_call_violation.to_string(),
            row.position_limit_violation.to_string(),
            metrics.fitness.to_string(),
            metrics.eval_pnl.to_string(),
            metrics.eval_pnl_realized.to_string(),
            metrics.eval_pnl_total.to_string(),
            metrics.eval_sortino.to_string(),
            metrics.eval_drawdown.to_string(),
            metrics.eval_ret_mean.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = args::Args::parse();
    run(args)
}

fn run(args: args::Args) -> anyhow::Result<()> {
    use anyhow::Context;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;
    use std::io::{BufRead, BufReader, Write};

    let overall_start = std::time::Instant::now();

    if let Some(seed) = args.seed {
        let mut rng = StdRng::seed_from_u64(seed);
        let _ = rng.r#gen::<u64>();
    }

    std::fs::create_dir_all(&args.outdir)?;
    let behavior_dir = args.outdir.join("behavior");
    std::fs::create_dir_all(&behavior_dir)?;

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

    let obs_dim = train.obs_dim;
    let genome_len = model::param_count(obs_dim, args.hidden, args.layers);

    let mut rng = StdRng::from_entropy();
    let normal = Normal::<f32>::new(0.0, args.init_sigma as f32)?;
    let mut start_gen = 0usize;
    let mut target_pop_size = args.pop_size;
    let mut pop: Vec<Vec<f32>> = if let Some(checkpoint) = args.load_checkpoint.as_ref() {
        if !checkpoint.exists() {
            anyhow::bail!("checkpoint not found: {}", checkpoint.display());
        }
        let (resume_gen, loaded_pop) = ga::load_checkpoint(checkpoint)
            .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;
        start_gen = resume_gen;
        if loaded_pop.is_empty() {
            anyhow::bail!("checkpoint population is empty");
        }
        if loaded_pop.len() != target_pop_size {
            println!(
                "warn: checkpoint population size {} does not match --pop-size {}; using {}",
                loaded_pop.len(),
                target_pop_size,
                loaded_pop.len()
            );
            target_pop_size = loaded_pop.len();
        }
        println!(
            "info: resuming from checkpoint {} at generation {}",
            checkpoint.display(),
            start_gen
        );
        loaded_pop
    } else {
        (0..target_pop_size)
            .map(|_| (0..genome_len).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    };
    if pop.iter().any(|genome| genome.len() != genome_len) {
        anyhow::bail!(
            "checkpoint genome length mismatch (expected {}, found {})",
            genome_len,
            pop.first().map(|g| g.len()).unwrap_or(0)
        );
    }

    let log_path = args.outdir.join("ga_log.csv");
    let mut log_has_eval_fitness = false;
    let mut log_has_selection_fitness = false;
    if log_path.exists() {
        let meta = std::fs::metadata(&log_path)?;
        if meta.len() == 0 {
            std::fs::write(
                &log_path,
                "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,selection_fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n",
            )?;
            log_has_eval_fitness = true;
            log_has_selection_fitness = true;
        } else {
            let file = std::fs::File::open(&log_path)?;
            let mut reader = BufReader::new(file);
            let mut header = String::new();
            let _ = reader.read_line(&mut header)?;
            log_has_eval_fitness = header.split(',').any(|col| col.trim() == "eval_fitness");
            log_has_selection_fitness = header.split(',').any(|col| col.trim() == "selection_fitness");
            if !log_has_eval_fitness || !log_has_selection_fitness {
                println!(
                    "warn: ga_log.csv missing selection columns; delete the log to enable eval/selection fitness tracking"
                );
            }
        }
    } else {
        std::fs::write(
            &log_path,
            "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,selection_fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n",
        )?;
        log_has_eval_fitness = true;
        log_has_selection_fitness = true;
    }

    let mut best_overall_score = f64::NEG_INFINITY;
    let mut best_overall_genome: Option<Vec<f32>> = None;

    if start_gen >= args.generations {
        println!(
            "info: checkpoint generation {} is >= requested generations {}; nothing to run",
            start_gen, args.generations
        );
        println!(
            "info: total training time {:.2?}",
            overall_start.elapsed()
        );
        return Ok(());
    }

    for generation in start_gen..args.generations {
        let gen_start = std::time::Instant::now();
        println!(
            "\nGeneration {} | Evaluating {} candidates (device={:?})",
            generation,
            pop.len(),
            device
        );

        let base_cfg = ga::CandidateConfig {
            initial_balance: args.initial_balance,
            max_position: args.max_position,
            margin_mode,
            contract_multiplier,
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
            hold_duration_penalty: args.hold_duration_penalty,
            hold_duration_penalty_growth: args.hold_duration_penalty_growth,
            hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
            hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
            invalid_revert_penalty: args.invalid_revert_penalty,
            flat_hold_penalty: args.flat_hold_penalty,
            max_flat_hold_bars: args.max_flat_hold_bars,
            invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
            flat_hold_penalty_growth: args.flat_hold_penalty_growth,
        };

        let mut batch_candidates = if args.batch_candidates > 0 {
            args.batch_candidates
        } else if matches!(device, tch::Device::Cuda(_) | tch::Device::Mps) {
            pop.len()
        } else {
            1
        };
        if batch_candidates == 0 {
            batch_candidates = 1;
        }
        if batch_candidates > pop.len() {
            batch_candidates = pop.len();
        }
        if batch_candidates > 1 {
            println!(
                "info: using batched inference across {} candidates per step",
                batch_candidates
            );
        }

        let eval_results: Vec<(usize, ga::CandidateResult, Option<ga::CandidateResult>)> =
            if batch_candidates > 1 {
                let mut results = Vec::with_capacity(pop.len());
                let mut offset = 0usize;
                for chunk in pop.chunks(batch_candidates) {
                    let train_results =
                        ga::evaluate_candidates_batch(chunk, &train, &windows_train, &base_cfg, false);
                    let mut eval_iter = if args.skip_val_eval {
                        None
                    } else {
                        Some(
                            ga::evaluate_candidates_batch(chunk, &val, &windows_val, &base_cfg, true)
                                .into_iter(),
                        )
                    };
                    for (i, train_metrics) in train_results.into_iter().enumerate() {
                        let eval_metrics = eval_iter.as_mut().and_then(|iter| iter.next());
                        results.push((offset + i, train_metrics, eval_metrics));
                    }
                    offset += chunk.len();
                }
                results
            } else {
                pop.par_iter()
                    .enumerate()
                    .map(|(idx, genome)| {
                        let train_metrics =
                            ga::evaluate_candidate(genome, &train, &windows_train, &base_cfg, false);
                        let eval_metrics = if args.skip_val_eval {
                            None
                        } else {
                            Some(ga::evaluate_candidate(genome, &val, &windows_val, &base_cfg, true))
                        };
                        (idx, train_metrics, eval_metrics)
                    })
                    .collect()
            };

        let mut scored: Vec<(f64, usize, Vec<f32>, Option<ga::CandidateResult>, ga::CandidateResult)> =
            Vec::with_capacity(pop.len());

        let mut results = eval_results;
        results.sort_by_key(|(idx, _, _)| *idx);

        let mut log_buffer = String::new();
        for (idx, train_metrics, eval_metrics) in results.into_iter() {
            let genome = pop[idx].clone();
            let selection_score = eval_metrics
                .as_ref()
                .map(|m| {
                    let gap = (train_metrics.fitness - m.fitness).max(0.0);
                    args.selection_train_weight * train_metrics.fitness
                        + args.selection_eval_weight * m.fitness
                        - args.selection_gap_penalty * gap
                })
                .unwrap_or(train_metrics.fitness);
            scored.push((
                selection_score,
                idx,
                genome.clone(),
                eval_metrics.clone(),
                train_metrics.clone(),
            ));

            let eval_pnl = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.eval_pnl))
                .unwrap_or_default();
            let eval_fitness = eval_metrics
                .as_ref()
                .map(|m| format!("{:.4}", m.fitness))
                .unwrap_or_default();
            let selection_fitness = format!("{:.4}", selection_score);
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

            let line = if log_has_eval_fitness && log_has_selection_fitness {
                format!(
                    "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                    generation,
                    idx,
                    args.w_pnl,
                    args.w_sortino,
                    args.w_mdd,
                    train_metrics.fitness,
                    eval_fitness,
                    selection_fitness,
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
                )
            } else if log_has_eval_fitness {
                format!(
                    "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                    generation,
                    idx,
                    args.w_pnl,
                    args.w_sortino,
                    args.w_mdd,
                    train_metrics.fitness,
                    eval_fitness,
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
                )
            } else {
                format!(
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
                )
            };
            log_buffer.push_str(&line);

            if let Some(eval) = eval_metrics {
                println!(
                    "  cand {}/{} | sel {:.2} | train {:.2} | eval {:.2}",
                    idx,
                    pop.len().saturating_sub(1),
                    selection_score,
                    train_metrics.fitness,
                    eval.fitness
                );
            } else {
                println!(
                    "  cand {}/{} | sel {:.2} | train {:.2} | val skipped",
                    idx,
                    pop.len().saturating_sub(1),
                    selection_score,
                    train_metrics.fitness
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
                    "  debug cand0 | invalid_revert_penalty {:.4} | hold_duration_penalty {:.4} | flat_hold_penalty {:.4} | session_close_penalty {:.4}",
                    train_metrics.debug_invalid_revert_penalty,
                    train_metrics.debug_hold_duration_penalty,
                    train_metrics.debug_flat_hold_penalty,
                    train_metrics.debug_session_close_penalty
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

        if !log_buffer.is_empty() {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)?
                .write_all(log_buffer.as_bytes())
                .context("write ga_log")?;
        }

        println!("Generation {} completed in {:.2?}", generation, gen_start.elapsed());

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((score, best_idx, genome, eval_metrics, train_metrics)) = scored.first() {
            if *score > best_overall_score {
                best_overall_score = *score;
                best_overall_genome = Some(genome.clone());
            }
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

            let (train_behavior_metrics, train_history) =
                ga::evaluate_candidate_with_history(genome, &train, &windows_train, &base_cfg);
            let train_path =
                behavior_dir.join(format!("train_gen{}_idx{}.csv", generation, *best_idx));
            write_behavior_csv(
                &train_path,
                generation,
                *best_idx,
                "train",
                &train,
                &train_behavior_metrics,
                &train_history,
            )?;
            if !args.skip_val_eval {
                let (val_behavior_metrics, val_history) =
                    ga::evaluate_candidate_with_history(genome, &val, &windows_val, &base_cfg);
                let val_path =
                    behavior_dir.join(format!("val_gen{}_idx{}.csv", generation, *best_idx));
                write_behavior_csv(
                    &val_path,
                    generation,
                    *best_idx,
                    "val",
                    &val,
                    &val_behavior_metrics,
                    &val_history,
                )?;
            }
        }

        let should_save = args.save_top_n > 0 && args.save_every > 0 && generation % args.save_every == 0;
        if should_save {
            for (rank, (_fit, _idx, genome, _eval_metrics, _train_metrics)) in
                scored.iter().take(args.save_top_n).enumerate()
            {
                let policy_path = args
                    .outdir
                    .join(format!("policy_gen{}_rank{}.pt", generation, rank));
                model::save_policy(obs_dim, args.hidden, args.layers, device, genome, &policy_path)?;
            }
        }

        let elite_n = (args.elite_frac * target_pop_size as f64).round().max(1.0) as usize;
        let elites: Vec<Vec<f32>> = scored
            .iter()
            .take(elite_n)
            .map(|(_, _, g, _, _)| g.clone())
            .collect();
        let mut new_pop = elites.clone();

        let normal_mut = Normal::<f32>::new(0.0, args.mutation_sigma as f32)?;
        while new_pop.len() < target_pop_size {
            let parent_a = elites.choose(&mut rng).unwrap();
            let parent_b = elites.choose(&mut rng).unwrap();
            let mut child = ga::crossover(parent_a, parent_b, &mut rng);
            for v in child.iter_mut() {
                *v += normal_mut.sample(&mut rng);
            }
            new_pop.push(child);
        }
        pop = new_pop;

        if args.checkpoint_every > 0 && generation % args.checkpoint_every == 0 {
            let ckpt_path = args.outdir.join(format!("checkpoint_gen{}.bin", generation));
            ga::save_checkpoint(&ckpt_path, generation, &pop)?;
        }
    }

    if let Some(best_genome) = best_overall_genome {
        let base_cfg = ga::CandidateConfig {
            initial_balance: args.initial_balance,
            max_position: args.max_position,
            margin_mode,
            contract_multiplier,
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
            hold_duration_penalty: args.hold_duration_penalty,
            hold_duration_penalty_growth: args.hold_duration_penalty_growth,
            hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
            hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
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

    println!(
        "info: total training time {:.2?}",
        overall_start.elapsed()
    );
    Ok(())
}
