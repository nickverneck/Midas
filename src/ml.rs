use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainerKind {
    Ga,
    Rl,
    Supervised,
}

impl TrainerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ga => "ga",
            Self::Rl => "rl",
            Self::Supervised => "supervised",
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
    CpuLinear,
}

impl MlBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "libtorch" | "torch" | "tch" => Ok(Self::Libtorch),
            "burn" => Ok(Self::Burn),
            "candle" => Ok(Self::Candle),
            "mlx" => Ok(Self::Mlx),
            "cpu-linear" | "reference-cpu" => Ok(Self::CpuLinear),
            other => Err(anyhow!(
                "unsupported backend '{other}' (expected one of: cpu-linear, libtorch, burn, candle, mlx)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Libtorch => "libtorch",
            Self::Burn => "burn",
            Self::Candle => "candle",
            Self::Mlx => "mlx",
            Self::CpuLinear => "cpu-linear",
        }
    }

    pub fn cargo_feature(self) -> &'static str {
        match self {
            Self::Libtorch => "torch",
            Self::Burn => "backend-burn",
            Self::Candle => "backend-candle",
            Self::Mlx => "backend-mlx",
            Self::CpuLinear => "none",
        }
    }

    pub fn implementation_status(self, trainer: TrainerKind) -> ImplementationStatus {
        match (trainer, self) {
            (TrainerKind::Ga | TrainerKind::Rl, Self::Libtorch) => {
                ImplementationStatus::Implemented
            }
            (TrainerKind::Supervised, Self::Libtorch) => ImplementationStatus::Planned,
            (TrainerKind::Ga, Self::Burn) => ImplementationStatus::Implemented,
            (TrainerKind::Ga, Self::Candle) => ImplementationStatus::Implemented,
            (TrainerKind::Rl, Self::Burn) => ImplementationStatus::Implemented,
            (TrainerKind::Rl, Self::Candle) => ImplementationStatus::Implemented,
            (TrainerKind::Supervised, Self::Burn) => ImplementationStatus::Implemented,
            (TrainerKind::Supervised, Self::CpuLinear) => ImplementationStatus::Implemented,
            (TrainerKind::Supervised, Self::Candle) => ImplementationStatus::Implemented,
            (_, Self::Mlx | Self::CpuLinear) => ImplementationStatus::Planned,
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

/// Host policy for Candle's native CUDA path.
///
/// Candle's current CUDA kernels are not usable on NVIDIA Pascal (sm_61),
/// which includes the GTX 1080 Ti.  Keep this probe independent of Candle so
/// GA and RL can make the same decision before constructing a `Device`.
pub mod candle_cuda {
    use std::process::Command;

    pub const PASCAL_BLOCKED_MESSAGE: &str = "Candle CUDA is disabled for NVIDIA Pascal (compute capability sm_61, including GTX 1080 Ti) because the current Candle CUDA kernels are not compatible; use --device cpu or choose Burn/libtorch for GPU training";
    pub const UNKNOWN_CAPABILITY_MESSAGE: &str = "Candle CUDA requires a verified NVIDIA compute capability, but nvidia-smi could not report one; install/enable nvidia-smi or use --device cpu";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HostPolicy {
        Supported,
        PascalBlocked,
        Unknown,
    }

    pub fn is_pascal_compute_capability(value: &str) -> bool {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "6.1" | "sm_61" | "sm61" | "sm-61"
        )
    }

    pub fn policy_for_compute_capabilities(capabilities: &[String]) -> HostPolicy {
        if capabilities.is_empty() {
            return HostPolicy::Unknown;
        }
        if capabilities
            .iter()
            .any(|value| is_pascal_compute_capability(value))
        {
            HostPolicy::PascalBlocked
        } else {
            HostPolicy::Supported
        }
    }

    /// Query the NVIDIA driver rather than trusting whether the CUDA feature
    /// compiled.  An unverified host is deliberately treated as unavailable:
    /// this prevents auto or explicit Candle selection from silently reaching
    /// an unsupported Pascal device.
    pub fn host_policy() -> HostPolicy {
        let output = match Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
            .output()
        {
            Ok(output) if output.status.success() => output,
            _ => return HostPolicy::Unknown,
        };

        let capabilities = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        policy_for_compute_capabilities(&capabilities)
    }

    pub fn auto_is_allowed() -> bool {
        matches!(host_policy(), HostPolicy::Supported)
    }

    pub fn explicit_block_reason() -> Option<&'static str> {
        match host_policy() {
            HostPolicy::Supported => None,
            HostPolicy::PascalBlocked => Some(PASCAL_BLOCKED_MESSAGE),
            HostPolicy::Unknown => Some(UNKNOWN_CAPABILITY_MESSAGE),
        }
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
    pub observation_schema: Option<&'a str>,
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

    validate_requested_runtime(trainer, backend, requested_runtime)?;

    if matches!(backend, MlBackend::Mlx) && matches!(requested_runtime, ComputeRuntime::Mps) {
        // Keep the runtime naming consistent with the rest of the repo while
        // preserving the user's intent to target the Apple GPU path.
    }

    let implementation_status = backend.implementation_status(trainer);

    let notes = match (trainer, backend) {
        (TrainerKind::Ga | TrainerKind::Rl, MlBackend::Libtorch) => vec![
            "Current Rust GA/RL implementation.".to_string(),
            "Use this path for working training runs today.".to_string(),
        ],
        (TrainerKind::Supervised, MlBackend::Libtorch) => vec![
            "The supervised trainer is currently CPU-linear only; no libtorch supervised runner is wired.".to_string(),
        ],
        (TrainerKind::Ga, MlBackend::Burn) => vec![
            "Implemented for GA training in this branch.".to_string(),
            "Uses Burn 0.20 with the burn-ndarray CPU backend for reliable small-matrix inference.".to_string(),
            "Enable native Burn CUDA on the Linux training box with the 'backend-burn-cuda' Cargo feature.".to_string(),
            "Enable burn-mlx on macOS with the 'backend-burn-mlx' Cargo feature when the local MLX toolchain is installed.".to_string(),
        ],
        (TrainerKind::Rl, MlBackend::Burn) => vec![
            "Implemented for PPO and GRPO training in this branch.".to_string(),
            "Uses the Burn tensor backend on CPU, native Burn CUDA on Linux, or burn-mlx on macOS; the policy update is kept backend-neutral and checkpoints are portable JSON.".to_string(),
        ],
        (TrainerKind::Ga, MlBackend::Candle) => vec![
            "Implemented for GA training in this branch.".to_string(),
            "Runs on CPU today; add the backend-candle-cuda Cargo feature on the Linux training box to benchmark CUDA once that toolchain is available.".to_string(),
            "Apple GPU execution still belongs to the MLX path for now; Candle on macOS is CPU-oriented in this rollout.".to_string(),
        ],
        (TrainerKind::Rl, MlBackend::Candle) => vec![
            "Implemented for PPO and GRPO training in this branch.".to_string(),
            "Runs on CPU by default and uses native Candle CUDA when built with the backend-candle-cuda Cargo feature.".to_string(),
            "Candle Metal is not wired into the training binaries; use Burn MLX or libtorch MPS for Apple GPU experiments.".to_string(),
        ],
        (TrainerKind::Supervised, MlBackend::CpuLinear) => vec![
            "Implemented as the deterministic CPU reference trainer.".to_string(),
            "CUDA and Metal are intentionally rejected until a parity-tested tensor trainer is added.".to_string(),
        ],
        (TrainerKind::Supervised, MlBackend::Burn) => vec![
            "Implemented as the event-level multiclass classifier.".to_string(),
            "Uses Burn autodiff with burn-ndarray on CPU and native Burn CUDA when built with the backend-burn-cuda Cargo feature.".to_string(),
            "Burn MLX is available on macOS builds with the backend-burn-mlx Cargo feature.".to_string(),
        ],
        (TrainerKind::Supervised, MlBackend::Candle) => vec![
            "Implemented as the event-level multiclass classifier.".to_string(),
            "Runs on CPU by default and uses native Candle CUDA when built with the backend-candle-cuda Cargo feature.".to_string(),
            "Candle Metal is not wired into the training binaries; use Burn MLX or libtorch MPS for Apple GPU experiments.".to_string(),
        ],
        (_, MlBackend::Mlx) => vec![
            "Planned backend slot for MLX-based experimentation across Apple Silicon dev and Linux CUDA/CPU environments.".to_string(),
            "Expected to land as a separate runner path rather than a drop-in tch replacement.".to_string(),
        ],
        (_, MlBackend::CpuLinear) => vec![
            "CPU reference backend; it does not provide an accelerator runtime.".to_string(),
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
        "backend '{}' is not implemented for {} training in this branch. {}",
        stack.backend,
        stack.trainer,
        stack.notes.join(" ")
    );
}

/// Validate constraints that can be determined from the selected backend and
/// compiled Cargo features. This intentionally does not probe a physical GPU;
/// the concrete runner performs that check when it creates its device.
pub fn validate_requested_runtime(
    trainer: TrainerKind,
    backend: MlBackend,
    runtime: ComputeRuntime,
) -> Result<()> {
    match runtime {
        ComputeRuntime::Cpu | ComputeRuntime::Auto => Ok(()),
        ComputeRuntime::Cuda => match backend {
            MlBackend::Burn if !cfg!(feature = "backend-burn-cuda") => bail!(
                "Burn CUDA was requested for {trainer}, but this build lacks the `backend-burn-cuda` Cargo feature"
            ),
            MlBackend::Candle if !cfg!(feature = "backend-candle-cuda") => bail!(
                "Candle CUDA was requested for {trainer}, but this build lacks the `backend-candle-cuda` Cargo feature"
            ),
            MlBackend::CpuLinear => bail!(
                "the cpu-linear supervised backend does not support CUDA; use --backend burn or --backend candle after a supervised accelerator runner is implemented"
            ),
            MlBackend::Mlx => bail!(
                "MLX is an Apple Metal backend, not a CUDA backend; use --device mps or select Burn/Candle"
            ),
            MlBackend::Libtorch => Ok(()),
            MlBackend::Burn | MlBackend::Candle => Ok(()),
        },
        ComputeRuntime::Mps => match backend {
            MlBackend::Burn if !cfg!(feature = "backend-burn-mlx") => bail!(
                "Burn Metal was requested for {trainer}, but this build lacks the `backend-burn-mlx` Cargo feature"
            ),
            MlBackend::Burn => Ok(()),
            MlBackend::Libtorch => Ok(()),
            MlBackend::Candle => bail!(
                "Candle Metal is not wired into the training binaries; use Burn with `backend-burn-mlx` or libtorch MPS"
            ),
            MlBackend::Mlx => bail!(
                "the standalone MLX backend is not implemented; use Burn with `backend-burn-mlx`"
            ),
            MlBackend::CpuLinear => {
                bail!("the cpu-linear supervised backend does not support Metal")
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_linear_reference_backend() {
        assert_eq!(
            MlBackend::parse("cpu-linear").unwrap(),
            MlBackend::CpuLinear
        );
        assert_eq!(MlBackend::CpuLinear.cargo_feature(), "none");
    }

    #[test]
    fn burn_rl_is_implemented() {
        assert_eq!(
            MlBackend::Burn.implementation_status(TrainerKind::Rl),
            ImplementationStatus::Implemented
        );
    }

    #[test]
    fn burn_supervised_is_implemented() {
        assert_eq!(
            MlBackend::Burn.implementation_status(TrainerKind::Supervised),
            ImplementationStatus::Implemented
        );
    }

    #[test]
    fn supervised_cpu_reference_is_implemented() {
        let stack = resolve_training_stack(TrainerKind::Supervised, "cpu-linear", "cpu").unwrap();
        assert!(stack.is_implemented());
        assert_eq!(stack.effective_runtime, ComputeRuntime::Cpu);
    }

    #[test]
    fn supervised_candle_is_implemented() {
        assert_eq!(
            MlBackend::Candle.implementation_status(TrainerKind::Supervised),
            ImplementationStatus::Implemented
        );
    }

    #[test]
    fn candle_metal_is_rejected_for_training() {
        let error =
            validate_requested_runtime(TrainerKind::Rl, MlBackend::Candle, ComputeRuntime::Mps)
                .unwrap_err()
                .to_string();
        assert!(error.contains("Candle Metal"));
    }

    #[test]
    fn candle_cuda_policy_blocks_pascal_variants() {
        for value in ["6.1", "sm_61", "SM61", "sm-61"] {
            assert!(candle_cuda::is_pascal_compute_capability(value));
        }

        let capabilities = vec!["6.1".to_string()];
        assert_eq!(
            candle_cuda::policy_for_compute_capabilities(&capabilities),
            candle_cuda::HostPolicy::PascalBlocked
        );
    }

    #[test]
    fn candle_cuda_policy_fails_closed_when_capability_is_unknown() {
        assert_eq!(
            candle_cuda::policy_for_compute_capabilities(&[]),
            candle_cuda::HostPolicy::Unknown
        );
        assert_eq!(
            candle_cuda::policy_for_compute_capabilities(&["8.6".to_string()]),
            candle_cuda::HostPolicy::Supported
        );
    }

    #[cfg(not(feature = "backend-candle-cuda"))]
    #[test]
    fn candle_cuda_requires_the_cuda_feature() {
        let error =
            validate_requested_runtime(TrainerKind::Rl, MlBackend::Candle, ComputeRuntime::Cuda)
                .unwrap_err()
                .to_string();
        assert!(error.contains("backend-candle-cuda"));
    }
}

pub fn write_run_metadata(
    path: &Path,
    stack: &ResolvedTrainingStack,
    algorithm: Option<&str>,
    observation_schema: Option<&str>,
) -> Result<()> {
    let metadata = RunMetadata {
        trainer: stack.trainer,
        backend: stack.backend,
        requested_runtime: stack.requested_runtime,
        effective_runtime: stack.effective_runtime,
        implementation_status: stack.implementation_status,
        cargo_feature: stack.cargo_feature.as_str(),
        algorithm,
        observation_schema,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    };
    let payload = serde_json::to_string_pretty(&metadata).context("serialize run metadata")?;
    std::fs::write(path, payload)
        .with_context(|| format!("write run metadata {}", path.display()))?;
    Ok(())
}
