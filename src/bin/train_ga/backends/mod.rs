use anyhow::{Result, bail};
use midas_env::ml::{MlBackend, ResolvedTrainingStack};
use std::path::Path;

use crate::config::{CandidateConfig, ExecutionTarget};
use crate::types::{BehaviorRow, CandidateResult};

#[cfg(feature = "backend-burn")]
mod burn;
#[cfg(feature = "backend-candle")]
mod candle;
mod mlx;
#[cfg(feature = "torch")]
mod tch;

pub fn resolve_device(stack: &ResolvedTrainingStack) -> Result<ExecutionTarget> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                Ok(tch::resolve_device(stack.requested_runtime))
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::resolve_device(stack.requested_runtime)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::resolve_device(stack.requested_runtime)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => {
            bail!("mlx GA device resolution is not implemented yet")
        }
    }
}

pub fn print_device(stack: &ResolvedTrainingStack, device: ExecutionTarget) -> Result<()> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::print_device(device);
                Ok(())
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::print_device(device);
                Ok(())
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::print_device(device);
                Ok(())
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => bail!("mlx GA backend is not implemented yet"),
    }
}

pub fn policy_extension(stack: &ResolvedTrainingStack) -> &'static str {
    match stack.backend {
        MlBackend::Libtorch => "pt",
        MlBackend::Candle => "safetensors",
        MlBackend::Burn => "portable.json",
        MlBackend::Mlx => "bin",
    }
}

pub fn param_count(
    stack: &ResolvedTrainingStack,
    input_dim: usize,
    hidden: usize,
    layers: usize,
) -> Result<usize> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::param_count(input_dim, hidden, layers)
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::param_count(input_dim, hidden, layers)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => mlx::param_count(input_dim, hidden, layers),
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::param_count(input_dim, hidden, layers)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
    }
}

pub fn evaluate_candidate(
    stack: &ResolvedTrainingStack,
    genome: &[f32],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> Result<CandidateResult> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::evaluate_candidate(genome, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::evaluate_candidate(genome, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => mlx::evaluate_candidate(genome, data, windows, cfg, capture_history),
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::evaluate_candidate(genome, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
    }
}

pub fn evaluate_candidate_with_history(
    stack: &ResolvedTrainingStack,
    genome: &[f32],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
) -> Result<(CandidateResult, Vec<BehaviorRow>)> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::evaluate_candidate_with_history(genome, data, windows, cfg)
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::evaluate_candidate_with_history(genome, data, windows, cfg)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => mlx::evaluate_candidate_with_history(genome, data, windows, cfg),
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::evaluate_candidate_with_history(genome, data, windows, cfg)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
    }
}

pub fn evaluate_candidates_batch(
    stack: &ResolvedTrainingStack,
    genomes: &[Vec<f32>],
    data: &crate::data::DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> Result<Vec<CandidateResult>> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::evaluate_candidates_batch(genomes, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::evaluate_candidates_batch(genomes, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => {
            mlx::evaluate_candidates_batch(genomes, data, windows, cfg, capture_history)
        }
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::evaluate_candidates_batch(genomes, data, windows, cfg, capture_history)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
    }
}

pub fn save_policy(
    stack: &ResolvedTrainingStack,
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    device: ExecutionTarget,
    genome: &[f32],
    path: &Path,
) -> Result<()> {
    match stack.backend {
        MlBackend::Libtorch => {
            #[cfg(feature = "torch")]
            {
                tch::save_policy(obs_dim, hidden, layers, device, genome, path)
            }
            #[cfg(not(feature = "torch"))]
            {
                bail!("libtorch GA backend was selected without the 'torch' Cargo feature")
            }
        }
        MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                burn::save_policy(obs_dim, hidden, layers, device, genome, path)
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("burn GA backend was selected without the 'backend-burn' Cargo feature")
            }
        }
        MlBackend::Mlx => mlx::save_policy(obs_dim, hidden, layers, device, genome, path),
        MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                candle::save_policy(obs_dim, hidden, layers, device, genome, path)
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("candle GA backend was selected without the 'backend-candle' Cargo feature")
            }
        }
    }
}
