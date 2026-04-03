use std::time::Instant;

use midas_env::ml::ResolvedTrainingStack;

use crate::{args::Args, backends, behavior, generation, logging, persistence, setup};

pub(crate) fn run(args: Args, mut stack: ResolvedTrainingStack) -> anyhow::Result<()> {
    let overall_start = Instant::now();

    let (device, behavior_dir) = setup::initialize_runtime(&args, &mut stack)?;
    let resources = setup::load_run_resources(&args, &stack, device, behavior_dir)?;
    let mut population = setup::initialize_population(&args, resources.genome_len)?;
    let log_state = logging::initialize_ga_log(&args.outdir)?;
    let base_cfg = setup::build_candidate_config(&args, resources.device, &resources.margin);

    if population.start_gen >= args.generations {
        println!(
            "info: checkpoint generation {} is >= requested generations {}; nothing to run",
            population.start_gen, args.generations
        );
        println!("info: total training time {:.2?}", overall_start.elapsed());
        return Ok(());
    }

    let evolution_plan = setup::build_evolution_plan(&args, population.target_pop_size);
    if args.immigrant_frac > 0.0 || evolution_plan.parent_pool_n > evolution_plan.elite_n {
        println!(
            "info: reproduction keeps {} elites, breeds from top {} candidates, injects {} immigrants, breeds {} crossover children",
            evolution_plan.elite_n,
            evolution_plan.parent_pool_n,
            evolution_plan.immigrant_n,
            evolution_plan
                .target_pop_size
                .saturating_sub(evolution_plan.elite_n + evolution_plan.immigrant_n)
        );
    }

    let mut best_overall_score = f64::NEG_INFINITY;
    let mut best_overall_genome: Option<Vec<f32>> = None;

    for generation_idx in population.start_gen..args.generations {
        let gen_start = Instant::now();
        println!(
            "\nGeneration {} | Evaluating {} candidates (device={})",
            generation_idx,
            population.pop.len(),
            resources.device.label()
        );

        let batch_candidates =
            setup::determine_batch_candidates(&args, resources.device, population.pop.len());
        if batch_candidates > 1 {
            println!(
                "info: using batched inference across {} candidates per step",
                batch_candidates
            );
        }

        let mut evaluated = generation::evaluate_population(
            &args,
            &stack,
            &resources,
            &population.pop,
            &base_cfg,
            batch_candidates,
        )?;
        logging::append_generation_log(&log_state, &args, generation_idx, &evaluated)?;

        println!(
            "Generation {} completed in {:.2?}",
            generation_idx,
            gen_start.elapsed()
        );

        generation::sort_by_selection(&mut evaluated);
        if let Some(best) = evaluated.first() {
            if best.selection_score > best_overall_score {
                best_overall_score = best.selection_score;
                best_overall_genome = Some(best.genome.clone());
            }
            generation::log_top_performer(generation_idx, best);
            behavior::maybe_capture_behavior(
                &args,
                &stack,
                &resources,
                &base_cfg,
                generation_idx,
                best,
            )?;
        }

        persistence::maybe_save_generation_policies(
            &args,
            &stack,
            generation_idx,
            resources.obs_dim,
            resources.device,
            &evaluated,
        )?;
        population.pop = generation::reproduce_population(
            &args,
            &evaluated,
            &evolution_plan,
            resources.genome_len,
            &mut population.rng,
        )?;
        persistence::maybe_save_checkpoint(&args, generation_idx, &population.pop)?;
    }

    if let Some(best_genome) = best_overall_genome {
        let metrics = backends::evaluate_candidate(
            &stack,
            &best_genome,
            &resources.datasets.test,
            &resources.windows.test,
            &base_cfg,
            false,
        )?;
        println!(
            "test | fitness {:.2} | pnl {:.2} | sortino {:.2} | mdd {:.2}",
            metrics.fitness, metrics.eval_pnl, metrics.eval_sortino, metrics.eval_drawdown
        );
        persistence::save_best_overall_policy(
            &args,
            &stack,
            resources.obs_dim,
            resources.device,
            &best_genome,
        )?;
    }

    println!("info: total training time {:.2?}", overall_start.elapsed());
    Ok(())
}
