use anyhow::Result;
use midas_env::ml::ResolvedTrainingStack;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::{
    args::Args,
    backends,
    config::CandidateConfig,
    evolution,
    setup::{EvolutionPlan, RunResources},
    types::CandidateResult,
};

pub(crate) struct EvaluatedCandidate {
    pub(crate) selection_score: f64,
    pub(crate) idx: usize,
    pub(crate) genome: Vec<f32>,
    pub(crate) eval_metrics: Option<CandidateResult>,
    pub(crate) train_metrics: CandidateResult,
}

pub(crate) fn evaluate_population(
    args: &Args,
    stack: &ResolvedTrainingStack,
    resources: &RunResources,
    pop: &[Vec<f32>],
    base_cfg: &CandidateConfig,
    batch_candidates: usize,
) -> Result<Vec<EvaluatedCandidate>> {
    let eval_results: Vec<(usize, CandidateResult, Option<CandidateResult>)> =
        if batch_candidates > 1 {
            let mut results = Vec::with_capacity(pop.len());
            let mut offset = 0usize;
            for chunk in pop.chunks(batch_candidates) {
                let train_results = backends::evaluate_candidates_batch(
                    stack,
                    chunk,
                    &resources.datasets.train,
                    &resources.windows.train,
                    base_cfg,
                    false,
                )?;
                let mut eval_iter = if args.skip_val_eval {
                    None
                } else {
                    let eval_results = backends::evaluate_candidates_batch(
                        stack,
                        chunk,
                        &resources.datasets.val,
                        &resources.windows.val,
                        base_cfg,
                        true,
                    )?;
                    Some(eval_results.into_iter())
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
                    let train_metrics = backends::evaluate_candidate(
                        stack,
                        genome,
                        &resources.datasets.train,
                        &resources.windows.train,
                        base_cfg,
                        false,
                    )?;
                    let eval_metrics = if args.skip_val_eval {
                        None
                    } else {
                        Some(backends::evaluate_candidate(
                            stack,
                            genome,
                            &resources.datasets.val,
                            &resources.windows.val,
                            base_cfg,
                            true,
                        )?)
                    };
                    Ok::<_, anyhow::Error>((idx, train_metrics, eval_metrics))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

    let mut results = eval_results;
    results.sort_by_key(|(idx, _, _)| *idx);

    let mut evaluated = Vec::with_capacity(pop.len());
    for (idx, train_metrics, eval_metrics) in results {
        let selection_score = selection_score(args, &train_metrics, eval_metrics.as_ref());
        let candidate = EvaluatedCandidate {
            selection_score,
            idx,
            genome: pop[idx].clone(),
            eval_metrics,
            train_metrics,
        };
        print_candidate_summary(pop.len(), &candidate);
        if args.debug_data && candidate.idx == 0 {
            print_candidate_debug(&candidate.train_metrics);
        }
        evaluated.push(candidate);
    }

    Ok(evaluated)
}

pub(crate) fn sort_by_selection(candidates: &mut [EvaluatedCandidate]) {
    candidates.sort_by(|a, b| {
        b.selection_score
            .partial_cmp(&a.selection_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(crate) fn log_top_performer(generation: usize, best: &EvaluatedCandidate) {
    if let Some(eval) = best.eval_metrics.as_ref() {
        println!(
            "Gen {} Top Performer: train fitness {:.2}, val fitness {:.2}",
            generation, best.train_metrics.fitness, eval.fitness
        );
    } else {
        println!(
            "Gen {} Top Performer: train fitness {:.2}",
            generation, best.train_metrics.fitness
        );
    }
}

pub(crate) fn reproduce_population(
    args: &Args,
    candidates: &[EvaluatedCandidate],
    plan: &EvolutionPlan,
    genome_len: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f32>>> {
    let elites: Vec<Vec<f32>> = candidates
        .iter()
        .take(plan.elite_n)
        .map(|candidate| candidate.genome.clone())
        .collect();
    let parent_pool: Vec<Vec<f32>> = candidates
        .iter()
        .take(plan.parent_pool_n)
        .map(|candidate| candidate.genome.clone())
        .collect();
    let mut new_pop = elites.clone();

    let immigrant_dist = Normal::<f32>::new(0.0, args.init_sigma as f32)?;
    for _ in 0..plan.immigrant_n {
        let immigrant = (0..genome_len)
            .map(|_| immigrant_dist.sample(rng))
            .collect();
        new_pop.push(immigrant);
    }

    let mutation_dist = Normal::<f32>::new(0.0, args.mutation_sigma as f32)?;
    while new_pop.len() < plan.target_pop_size {
        let parent_a = parent_pool.choose(rng).unwrap();
        let parent_b = parent_pool.choose(rng).unwrap();
        let mut child = evolution::crossover(parent_a, parent_b, rng);
        for value in &mut child {
            *value += mutation_dist.sample(rng);
        }
        new_pop.push(child);
    }

    Ok(new_pop)
}

fn selection_score(
    args: &Args,
    train_metrics: &CandidateResult,
    eval_metrics: Option<&CandidateResult>,
) -> f64 {
    if args.selection_use_eval {
        eval_metrics
            .map(|metrics| {
                let gap = (train_metrics.fitness - metrics.fitness).max(0.0);
                args.selection_train_weight * train_metrics.fitness
                    + args.selection_eval_weight * metrics.fitness
                    - args.selection_gap_penalty * gap
            })
            .unwrap_or(train_metrics.fitness)
    } else {
        train_metrics.fitness
    }
}

fn print_candidate_summary(pop_len: usize, candidate: &EvaluatedCandidate) {
    if let Some(eval) = candidate.eval_metrics.as_ref() {
        println!(
            "  cand {}/{} | sel {:.2} | train {:.2} | eval {:.2}",
            candidate.idx,
            pop_len.saturating_sub(1),
            candidate.selection_score,
            candidate.train_metrics.fitness,
            eval.fitness
        );
    } else {
        println!(
            "  cand {}/{} | sel {:.2} | train {:.2} | val skipped",
            candidate.idx,
            pop_len.saturating_sub(1),
            candidate.selection_score,
            candidate.train_metrics.fitness
        );
    }
}

fn print_candidate_debug(train_metrics: &CandidateResult) {
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
