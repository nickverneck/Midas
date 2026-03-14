use anyhow::Result;
use std::path::Path;

use crate::config::{CandidateConfig, ExecutionTarget};
use crate::ga;
use crate::types::{BehaviorRow, CandidateResult};
use midas_env::ml::ComputeRuntime;

pub fn resolve_device(requested: ComputeRuntime) -> ExecutionTarget {
    match crate::util::resolve_device(requested) {
        tch::Device::Cuda(idx) => ExecutionTarget::Cuda(idx),
        tch::Device::Mps => ExecutionTarget::Mps,
        _ => ExecutionTarget::Cpu,
    }
}

pub fn print_device(device: ExecutionTarget) {
    crate::util::print_device(&device.as_tch_device());
}

pub fn param_count(input_dim: usize, hidden: usize, layers: usize) -> Result<usize> {
    Ok(crate::model::param_count(input_dim, hidden, layers))
}

pub fn evaluate_candidate(
    genome: &[f32],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> Result<CandidateResult> {
    Ok(ga::evaluate_candidate(
        genome,
        data,
        windows,
        cfg,
        capture_history,
    ))
}

pub fn evaluate_candidate_with_history(
    genome: &[f32],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
) -> Result<(CandidateResult, Vec<BehaviorRow>)> {
    Ok(ga::evaluate_candidate_with_history(
        genome, data, windows, cfg,
    ))
}

pub fn evaluate_candidates_batch(
    genomes: &[Vec<f32>],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> Result<Vec<CandidateResult>> {
    Ok(ga::evaluate_candidates_batch(
        genomes,
        data,
        windows,
        cfg,
        capture_history,
    ))
}

pub fn save_policy(
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    device: ExecutionTarget,
    genome: &[f32],
    path: &Path,
) -> Result<()> {
    crate::model::save_policy(
        obs_dim,
        hidden,
        layers,
        device.as_tch_device(),
        genome,
        path,
    )
}
