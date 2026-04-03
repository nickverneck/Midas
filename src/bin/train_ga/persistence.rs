use anyhow::Result;
use midas_env::ml::ResolvedTrainingStack;

use crate::{
    args::Args, backends, config::ExecutionTarget, evolution, generation::EvaluatedCandidate,
    portable,
};

pub(crate) fn maybe_save_generation_policies(
    args: &Args,
    stack: &ResolvedTrainingStack,
    generation: usize,
    obs_dim: usize,
    device: ExecutionTarget,
    candidates: &[EvaluatedCandidate],
) -> Result<()> {
    let should_save =
        args.save_top_n > 0 && args.save_every > 0 && generation % args.save_every == 0;
    if !should_save {
        return Ok(());
    }

    let policy_ext = backends::policy_extension(stack);
    let needs_portable_sidecar = policy_ext != "portable.json";
    for (rank, candidate) in candidates.iter().take(args.save_top_n).enumerate() {
        let policy_path = args.outdir.join(format!(
            "policy_gen{}_rank{}.{}",
            generation, rank, policy_ext
        ));
        let portable_path = args.outdir.join(format!(
            "policy_gen{}_rank{}.portable.json",
            generation, rank
        ));
        backends::save_policy(
            stack,
            obs_dim,
            args.hidden,
            args.layers,
            device,
            &candidate.genome,
            &policy_path,
        )?;
        if needs_portable_sidecar {
            portable::save_policy_json(
                obs_dim,
                args.hidden,
                args.layers,
                &candidate.genome,
                &portable_path,
            )?;
        }
    }

    Ok(())
}

pub(crate) fn maybe_save_checkpoint(
    args: &Args,
    generation: usize,
    pop: &[Vec<f32>],
) -> Result<()> {
    if args.checkpoint_every > 0 && generation % args.checkpoint_every == 0 {
        let ckpt_path = args
            .outdir
            .join(format!("checkpoint_gen{}.bin", generation));
        evolution::save_checkpoint(&ckpt_path, generation, pop)?;
    }
    Ok(())
}

pub(crate) fn save_best_overall_policy(
    args: &Args,
    stack: &ResolvedTrainingStack,
    obs_dim: usize,
    device: ExecutionTarget,
    genome: &[f32],
) -> Result<()> {
    let best_path = args.outdir.join(format!(
        "best_overall_policy.{}",
        backends::policy_extension(stack)
    ));
    let best_portable_path = args.outdir.join("best_overall_policy.portable.json");
    let needs_portable_sidecar = backends::policy_extension(stack) != "portable.json";
    backends::save_policy(
        stack,
        obs_dim,
        args.hidden,
        args.layers,
        device,
        genome,
        &best_path,
    )?;
    if needs_portable_sidecar {
        portable::save_policy_json(
            obs_dim,
            args.hidden,
            args.layers,
            genome,
            &best_portable_path,
        )?;
    }
    println!("Saved best overall policy to {}", best_path.display());
    Ok(())
}
