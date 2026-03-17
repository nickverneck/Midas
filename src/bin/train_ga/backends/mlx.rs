use anyhow::{Result, bail};
use std::path::Path;

use crate::actions::POLICY_ACTION_DIM;
use crate::config::{CandidateConfig, ExecutionTarget};
use crate::types::{BehaviorRow, CandidateResult};

pub fn param_count(input_dim: usize, hidden: usize, layers: usize) -> Result<usize> {
    let mut count = 0usize;
    let mut in_dim = input_dim;
    for _ in 0..layers {
        count += in_dim * hidden + hidden;
        in_dim = hidden;
    }
    count += in_dim * POLICY_ACTION_DIM + POLICY_ACTION_DIM;
    Ok(count)
}

pub fn evaluate_candidate(
    _genome: &[f32],
    _data: &crate::data::DataSet,
    _windows: &[(usize, usize)],
    _cfg: &CandidateConfig,
    _capture_history: bool,
) -> Result<CandidateResult> {
    bail!(
        "mlx GA backend is reserved for a future runner; use the MLX probe to validate the Mac path for now"
    )
}

pub fn evaluate_candidate_with_history(
    _genome: &[f32],
    _data: &crate::data::DataSet,
    _windows: &[(usize, usize)],
    _cfg: &CandidateConfig,
) -> Result<(CandidateResult, Vec<BehaviorRow>)> {
    bail!(
        "mlx GA backend is reserved for a future runner; use the MLX probe to validate the Mac path for now"
    )
}

pub fn evaluate_candidates_batch(
    _genomes: &[Vec<f32>],
    _data: &crate::data::DataSet,
    _windows: &[(usize, usize)],
    _cfg: &CandidateConfig,
    _capture_history: bool,
) -> Result<Vec<CandidateResult>> {
    bail!(
        "mlx GA backend is reserved for a future runner; use the MLX probe to validate the Mac path for now"
    )
}

pub fn save_policy(
    _obs_dim: usize,
    _hidden: usize,
    _layers: usize,
    _device: ExecutionTarget,
    _genome: &[f32],
    _path: &Path,
) -> Result<()> {
    bail!("mlx GA policy export is not landed yet")
}
