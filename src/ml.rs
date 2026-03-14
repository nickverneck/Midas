use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainerKind {
    Ga,
    Rl,
}

impl TrainerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ga => "ga",
            Self::Rl => "rl",
        }
    }
}

impl fmt::Display for TrainerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlBackend {
    Libtorch,
    Burn,
    Candle,
    Mlx,
}

impl MlBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "libtorch" | "torch" | "tch" => Ok(Self::Libtorch),
            "burn" => Ok(Self::Burn),
            "candle" => Ok(Self::Candle),
            "mlx" => Ok(Self::Mlx),
            other => Err(anyhow!(
                "unsupported backend '{other}' (expected one of: libtorch, burn, candle, mlx)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Libtorch => "libtorch",
            Self::Burn => "burn",
            Self::Candle => "candle",
            Self::Mlx => "mlx",
        }
    }

    pub fn cargo_feature(self) -> &'static str {
        match self {
            Self::Libtorch => "torch",
            Self::Burn => "backend-burn",
            Self::Candle => "backend-candle",
            Self::Mlx => "backend-mlx",
        }
    }

    pub fn implementation_status(self, trainer: TrainerKind) -> ImplementationStatus {
        match (trainer, self) {
            (_, Self::Libtorch) => ImplementationStatus::Implemented,
            (TrainerKind::Ga, Self::Burn) => ImplementationStatus::Implemented,
            (TrainerKind::Ga, Self::Candle) => ImplementationStatus::Implemented,
            (TrainerKind::Rl, Self::Candle) => ImplementationStatus::Implemented,
            (_, Self::Burn | Self::Mlx) => ImplementationStatus::Planned,
        }
    }
}

impl fmt::Display for MlBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComputeRuntime {
    Auto,
    Cpu,
    Cuda,
    Mps,
}

impl ComputeRuntime {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "cuda:0" => Ok(Self::Cuda),
            "mps" | "metal" => Ok(Self::Mps),
            other => Err(anyhow!(
                "unsupported runtime '{other}' (expected one of: auto, cpu, cuda, mps)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Mps => "mps",
        }
    }
}

impl fmt::Display for ComputeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImplementationStatus {
    Implemented,
    Planned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTrainingStack {
    pub trainer: TrainerKind,
    pub backend: MlBackend,
    pub requested_runtime: ComputeRuntime,
    pub effective_runtime: ComputeRuntime,
    pub implementation_status: ImplementationStatus,
    pub cargo_feature: String,
    pub notes: Vec<String>,
}

impl ResolvedTrainingStack {
    pub fn is_implemented(&self) -> bool {
        matches!(
            self.implementation_status,
            ImplementationStatus::Implemented
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata<'a> {
    pub trainer: TrainerKind,
    pub backend: MlBackend,
    pub requested_runtime: ComputeRuntime,
    pub effective_runtime: ComputeRuntime,
    pub implementation_status: ImplementationStatus,
    pub cargo_feature: &'a str,
    pub algorithm: Option<&'a str>,
    pub os: &'a str,
    pub arch: &'a str,
}

pub fn resolve_training_stack(
    trainer: TrainerKind,
    backend: &str,
    runtime: &str,
) -> Result<ResolvedTrainingStack> {
    let backend = MlBackend::parse(backend)?;
    let requested_runtime = ComputeRuntime::parse(runtime)?;

    if matches!(backend, MlBackend::Mlx) && matches!(requested_runtime, ComputeRuntime::Mps) {
        // Keep the runtime naming consistent with the rest of the repo while
        // preserving the user's intent to target the Apple GPU path.
    }

    let implementation_status = backend.implementation_status(trainer);

    let notes = match (trainer, backend) {
        (_, MlBackend::Libtorch) => vec![
            "Current Rust GA/RL implementation.".to_string(),
            "Use this path for working training runs today.".to_string(),
        ],
        (TrainerKind::Ga, MlBackend::Burn) => vec![
            "Implemented for GA training in this branch.".to_string(),
            "Uses Burn 0.20 with burn-ndarray for CPU by default.".to_string(),
            "Enable native Burn CUDA on the Linux training box with the 'backend-burn-cuda' Cargo feature.".to_string(),
            "Enable burn-mlx on macOS with the 'backend-burn-mlx' Cargo feature when the local MLX toolchain is installed.".to_string(),
        ],
        (TrainerKind::Rl, MlBackend::Burn) => vec![
            "Planned generic backend for RL parity with the GA Burn path.".to_string(),
            "The intended runtime split is Burn CPU, native Burn CUDA on Linux, and burn-mlx on macOS.".to_string(),
        ],
        (TrainerKind::Ga, MlBackend::Candle) => vec![
            "Implemented for GA training in this branch.".to_string(),
            "Runs on CPU today; add the backend-candle-cuda Cargo feature on the Linux training box to benchmark CUDA once that toolchain is available.".to_string(),
            "Apple GPU execution still belongs to the MLX path for now; Candle on macOS is CPU-oriented in this rollout.".to_string(),
        ],
        (TrainerKind::Rl, MlBackend::Candle) => vec![
            "Implemented for PPO and GRPO training in this branch.".to_string(),
            "Runs on CPU today; add the backend-candle-cuda Cargo feature on the Linux training box to benchmark CUDA once that toolchain is available.".to_string(),
            "Apple GPU execution still belongs to the MLX path for now; Candle on macOS is CPU-oriented in this rollout.".to_string(),
        ],
        (_, MlBackend::Mlx) => vec![
            "Planned backend slot for MLX-based experimentation across Apple Silicon dev and Linux CUDA/CPU environments.".to_string(),
            "Expected to land as a separate runner path rather than a drop-in tch replacement.".to_string(),
        ],
    };

    Ok(ResolvedTrainingStack {
        trainer,
        backend,
        requested_runtime,
        effective_runtime: requested_runtime,
        implementation_status,
        cargo_feature: backend.cargo_feature().to_string(),
        notes,
    })
}

pub fn print_training_stack(stack: &ResolvedTrainingStack) {
    println!(
        "info: trainer={} backend={} runtime={} feature={}",
        stack.trainer, stack.backend, stack.requested_runtime, stack.cargo_feature
    );
    for note in &stack.notes {
        println!("info: backend note: {note}");
    }
}

pub fn ensure_backend_is_implemented(stack: &ResolvedTrainingStack) -> Result<()> {
    if stack.is_implemented() {
        return Ok(());
    }
    bail!(
        "backend '{}' is not implemented yet for {} training in this branch. The CLI/UI selection surface and run metadata are ready so Burn, Candle, and MLX runners can plug in next.",
        stack.backend,
        stack.trainer
    );
}

pub fn write_run_metadata(
    path: &Path,
    stack: &ResolvedTrainingStack,
    algorithm: Option<&str>,
) -> Result<()> {
    let metadata = RunMetadata {
        trainer: stack.trainer,
        backend: stack.backend,
        requested_runtime: stack.requested_runtime,
        effective_runtime: stack.effective_runtime,
        implementation_status: stack.implementation_status,
        cargo_feature: stack.cargo_feature.as_str(),
        algorithm,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    };
    let payload = serde_json::to_string_pretty(&metadata).context("serialize run metadata")?;
    std::fs::write(path, payload)
        .with_context(|| format!("write run metadata {}", path.display()))?;
    Ok(())
}
