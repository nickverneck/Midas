//! Burn RL runner.
//!
//! The runner keeps the policy forward pass on the selected Burn device while
//! performing the small MLP backward pass and Adam update explicitly on host
//! vectors. This makes PPO/GRPO available across Burn CPU, native CUDA, and
//! burn-mlx without a silent Candle/libtorch fallback.

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
use burn_ndarray::{NdArray, NdArrayDevice};
use midas_env::bars::BarSelection;
use midas_env::env::{Action, EnvConfig, MarginMode, StepContext, TradingEnv};
use midas_env::ml::{self, ComputeRuntime};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant};

#[cfg(feature = "backend-burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "backend-burn-mlx")]
use burn_mlx::{Mlx, MlxDevice};

use crate::args::Args;
use crate::common;
use crate::data::{self, DataSet, ObservationSchema};
use crate::metrics::{compute_sortino, max_drawdown};

const ACTION_DIM: usize = 4;
const CHECKPOINT_FORMAT_VERSION: u32 = 1;

type CpuBackend = NdArray<f32>;
#[cfg(feature = "backend-burn-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "backend-burn-mlx")]
type MlxBackend = Mlx<f32>;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum BurnTarget {
    Cpu,
    Cuda(usize),
    Mps,
}

impl BurnTarget {
    fn runtime(self) -> ComputeRuntime {
        match self {
            Self::Cpu => ComputeRuntime::Cpu,
            Self::Cuda(_) => ComputeRuntime::Cuda,
            Self::Mps => ComputeRuntime::Mps,
        }
    }
}

#[derive(Clone)]
struct BurnLayer<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    weight_host: Vec<f32>,
    bias_host: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

struct BurnMlp<B: Backend> {
    layers: Vec<BurnLayer<B>>,
    input_dim: usize,
    hidden: usize,
    hidden_layers: usize,
    output_dim: usize,
    host_linear: bool,
}

struct MlpCache {
    inputs: Vec<Vec<f32>>,
    hidden_pre: Vec<Vec<f32>>,
    dropout_masks: Vec<Vec<f32>>,
}

struct ForwardPass {
    logits: Vec<f32>,
    cache: MlpCache,
}

#[derive(Clone, Default)]
struct MlpGrad {
    weight: Vec<Vec<f32>>,
    bias: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerCheckpoint {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCheckpoint {
    input_dim: usize,
    hidden: usize,
    hidden_layers: usize,
    output_dim: usize,
    layers: Vec<LayerCheckpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdamCheckpoint {
    step: u64,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    m_weight: Vec<Vec<f32>>,
    v_weight: Vec<Vec<f32>>,
    m_bias: Vec<Vec<f32>>,
    v_bias: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BurnCheckpoint {
    format_version: u32,
    backend: String,
    algorithm: String,
    epoch: usize,
    policy: ModelCheckpoint,
    value: Option<ModelCheckpoint>,
    policy_optimizer: AdamCheckpoint,
    value_optimizer: Option<AdamCheckpoint>,
}

fn host_linear_forward<B: Backend>(layer: &BurnLayer<B>, input: &[f32]) -> Vec<f32> {
    debug_assert_eq!(input.len(), layer.in_dim);
    (0..layer.out_dim)
        .map(|out_idx| {
            let row = out_idx * layer.in_dim;
            layer.bias_host[out_idx]
                + layer.weight_host[row..row + layer.in_dim]
                    .iter()
                    .zip(input.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
        })
        .collect()
}

impl<B: Backend> BurnMlp<B> {
    fn new(
        input_dim: usize,
        hidden: usize,
        hidden_layers: usize,
        output_dim: usize,
        device: &B::Device,
        rng: &mut StdRng,
        host_linear: bool,
    ) -> Self {
        let mut layers = Vec::with_capacity(hidden_layers + 1);
        let mut in_dim = input_dim;
        for layer_idx in 0..=hidden_layers {
            let out_dim = if layer_idx < hidden_layers {
                hidden
            } else {
                output_dim
            };
            let scale = (2.0 / in_dim.max(1) as f32).sqrt();
            let weight: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            let bias = vec![0.0; out_dim];
            layers.push(BurnLayer {
                weight: tensor_2::<B>(&weight, out_dim, in_dim, device),
                bias: tensor_1::<B>(&bias, out_dim, device),
                weight_host: weight,
                bias_host: bias,
                in_dim,
                out_dim,
            });
            in_dim = hidden;
        }

        Self {
            layers,
            input_dim,
            hidden,
            hidden_layers,
            output_dim,
            host_linear,
        }
    }

    fn from_checkpoint(
        checkpoint: &ModelCheckpoint,
        device: &B::Device,
        host_linear: bool,
    ) -> Result<Self> {
        if checkpoint.layers.len() != checkpoint.hidden_layers + 1 {
            bail!(
                "Burn RL checkpoint has {} layers; expected {}",
                checkpoint.layers.len(),
                checkpoint.hidden_layers + 1
            );
        }

        let mut layers = Vec::with_capacity(checkpoint.layers.len());
        for (idx, layer) in checkpoint.layers.iter().enumerate() {
            if layer.weight.len() != layer.in_dim * layer.out_dim {
                bail!("Burn RL checkpoint layer {idx} has an invalid weight length");
            }
            if layer.bias.len() != layer.out_dim {
                bail!("Burn RL checkpoint layer {idx} has an invalid bias length");
            }
            layers.push(BurnLayer {
                weight: tensor_2::<B>(&layer.weight, layer.out_dim, layer.in_dim, device),
                bias: tensor_1::<B>(&layer.bias, layer.out_dim, device),
                weight_host: layer.weight.clone(),
                bias_host: layer.bias.clone(),
                in_dim: layer.in_dim,
                out_dim: layer.out_dim,
            });
        }

        Ok(Self {
            layers,
            input_dim: checkpoint.input_dim,
            hidden: checkpoint.hidden,
            hidden_layers: checkpoint.hidden_layers,
            output_dim: checkpoint.output_dim,
            host_linear,
        })
    }

    fn checkpoint(&self) -> ModelCheckpoint {
        ModelCheckpoint {
            input_dim: self.input_dim,
            hidden: self.hidden,
            hidden_layers: self.hidden_layers,
            output_dim: self.output_dim,
            layers: self
                .layers
                .iter()
                .map(|layer| LayerCheckpoint {
                    in_dim: layer.in_dim,
                    out_dim: layer.out_dim,
                    weight: layer.weight_host.clone(),
                    bias: layer.bias_host.clone(),
                })
                .collect(),
        }
    }

    fn zero_grad(&self) -> MlpGrad {
        MlpGrad {
            weight: self
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.weight_host.len()])
                .collect(),
            bias: self
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.bias_host.len()])
                .collect(),
        }
    }

    fn forward_with_cache(
        &self,
        input: &[f32],
        device: &B::Device,
        train: bool,
        dropout: f32,
        rng: &mut StdRng,
    ) -> Result<ForwardPass> {
        if input.len() != self.input_dim {
            bail!(
                "Burn RL observation length {} does not match model input {}",
                input.len(),
                self.input_dim
            );
        }

        let mut current = input.to_vec();
        // Burn CPU's CubeCL matmul planner is both fragile and very slow for
        // the tiny 1 x N policy layers used by step-wise RL inference. The
        // CPU target therefore performs only the linear dot products on host
        // vectors and still uses Burn tensors for hidden activations. CUDA and
        // MLX retain native Burn matmul below.
        let mut tensor =
            (!self.host_linear).then(|| tensor_2::<B>(&current, 1, current.len(), device));
        let mut inputs = vec![current.clone()];
        let mut hidden_pre = Vec::with_capacity(self.hidden_layers);
        let mut dropout_masks = Vec::with_capacity(self.hidden_layers);

        for (idx, layer) in self.layers.iter().enumerate() {
            let output = if self.host_linear {
                None
            } else {
                let tensor = tensor
                    .take()
                    .expect("native Burn forward tensor missing before linear layer");
                Some(
                    tensor.matmul(layer.weight.clone().transpose())
                        + layer.bias.clone().reshape([1, layer.out_dim]),
                )
            };
            if idx < self.hidden_layers {
                let (pre, mut hidden) = if self.host_linear {
                    let pre = host_linear_forward(layer, &current);
                    let hidden_tensor = activation::tanh(tensor_2::<B>(&pre, 1, pre.len(), device));
                    let hidden = hidden_tensor
                        .into_data()
                        .to_vec::<f32>()
                        .context("extract Burn hidden activation")?;
                    (pre, hidden)
                } else {
                    let hidden_tensor = activation::tanh(
                        output.expect("native Burn hidden output missing before activation"),
                    );
                    let hidden = hidden_tensor
                        .into_data()
                        .to_vec::<f32>()
                        .context("extract Burn hidden activation")?;
                    let pre = hidden.iter().map(|value| value.atanh()).collect::<Vec<_>>();
                    (pre, hidden)
                };
                hidden.truncate(layer.out_dim);
                let mut mask = vec![1.0; hidden.len()];
                if train && dropout > 0.0 {
                    let keep = (1.0 - dropout).max(1e-6);
                    for value in &mut mask {
                        if rng.r#gen::<f32>() < dropout {
                            *value = 0.0;
                        } else {
                            *value = 1.0 / keep;
                        }
                    }
                    for (value, factor) in hidden.iter_mut().zip(mask.iter()) {
                        *value *= *factor;
                    }
                }
                current = hidden;
                tensor =
                    (!self.host_linear).then(|| tensor_2::<B>(&current, 1, current.len(), device));
                inputs.push(current.clone());
                hidden_pre.push(pre);
                dropout_masks.push(mask);
            } else {
                let logits = if self.host_linear {
                    host_linear_forward(layer, &current)
                } else {
                    output
                        .expect("native Burn policy output missing")
                        .into_data()
                        .to_vec::<f32>()
                        .context("extract Burn policy logits")?
                };
                return Ok(ForwardPass {
                    logits,
                    cache: MlpCache {
                        inputs,
                        hidden_pre,
                        dropout_masks,
                    },
                });
            }
        }

        bail!("Burn RL model has no output layer")
    }

    fn accumulate_backward(&self, cache: &MlpCache, output_grad: &[f32], grads: &mut MlpGrad) {
        let mut delta = output_grad.to_vec();
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];
            let input = &cache.inputs[layer_idx];
            for out_idx in 0..layer.out_dim {
                let delta_value = delta.get(out_idx).copied().unwrap_or(0.0);
                grads.bias[layer_idx][out_idx] += delta_value;
                let row = out_idx * layer.in_dim;
                for in_idx in 0..layer.in_dim {
                    grads.weight[layer_idx][row + in_idx] += delta_value * input[in_idx];
                }
            }

            if layer_idx > 0 {
                let mut previous = vec![0.0; layer.in_dim];
                for in_idx in 0..layer.in_dim {
                    let mut value = 0.0;
                    for out_idx in 0..layer.out_dim {
                        value +=
                            delta[out_idx] * layer.weight_host[out_idx * layer.in_dim + in_idx];
                    }
                    let pre = cache.hidden_pre[layer_idx - 1]
                        .get(in_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let mask = cache.dropout_masks[layer_idx - 1]
                        .get(in_idx)
                        .copied()
                        .unwrap_or(1.0);
                    previous[in_idx] = value * (1.0 - pre.tanh().powi(2)) * mask;
                }
                delta = previous;
            }
        }
    }

    fn replace_from_host(&mut self, device: &B::Device) {
        for layer in &mut self.layers {
            layer.weight = tensor_2::<B>(&layer.weight_host, layer.out_dim, layer.in_dim, device);
            layer.bias = tensor_1::<B>(&layer.bias_host, layer.out_dim, device);
        }
    }
}

impl AdamCheckpoint {
    fn new<B: Backend>(model: &BurnMlp<B>) -> Self {
        Self {
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m_weight: model
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.weight_host.len()])
                .collect(),
            v_weight: model
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.weight_host.len()])
                .collect(),
            m_bias: model
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.bias_host.len()])
                .collect(),
            v_bias: model
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.bias_host.len()])
                .collect(),
        }
    }

    fn update<B: Backend>(
        &mut self,
        model: &mut BurnMlp<B>,
        grads: &MlpGrad,
        lr: f64,
        device: &B::Device,
    ) -> f64 {
        self.step = self.step.saturating_add(1);
        let step = self.step as f32;
        let bias1 = 1.0 - self.beta1.powf(step);
        let bias2 = 1.0 - self.beta2.powf(step);
        let mut squared_norm = 0.0f64;

        for (idx, layer) in model.layers.iter_mut().enumerate() {
            for (param_idx, param) in layer.weight_host.iter_mut().enumerate() {
                let grad = grads.weight[idx][param_idx];
                squared_norm += (grad as f64).powi(2);
                self.m_weight[idx][param_idx] =
                    self.beta1 * self.m_weight[idx][param_idx] + (1.0 - self.beta1) * grad;
                self.v_weight[idx][param_idx] =
                    self.beta2 * self.v_weight[idx][param_idx] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_weight[idx][param_idx] / bias1.max(1e-8);
                let v_hat = self.v_weight[idx][param_idx] / bias2.max(1e-8);
                *param -= lr as f32
                    * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * *param);
            }
            for (param_idx, param) in layer.bias_host.iter_mut().enumerate() {
                let grad = grads.bias[idx][param_idx];
                squared_norm += (grad as f64).powi(2);
                self.m_bias[idx][param_idx] =
                    self.beta1 * self.m_bias[idx][param_idx] + (1.0 - self.beta1) * grad;
                self.v_bias[idx][param_idx] =
                    self.beta2 * self.v_bias[idx][param_idx] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_bias[idx][param_idx] / bias1.max(1e-8);
                let v_hat = self.v_bias[idx][param_idx] / bias2.max(1e-8);
                *param -= lr as f32
                    * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * *param);
            }
        }
        model.replace_from_host(device);
        squared_norm.sqrt()
    }
}

fn tensor_1<B: Backend>(values: &[f32], len: usize, device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [len]), device)
}

fn tensor_2<B: Backend>(
    values: &[f32],
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [rows, cols]), device)
}

fn resolve_device(requested: ComputeRuntime) -> Result<BurnTarget> {
    match requested {
        ComputeRuntime::Cpu => Ok(BurnTarget::Cpu),
        ComputeRuntime::Auto => auto_device(),
        ComputeRuntime::Cuda => explicit_cuda_device(),
        ComputeRuntime::Mps => explicit_mlx_device(),
    }
}

fn auto_device() -> Result<BurnTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        if mlx_device_available() {
            return Ok(BurnTarget::Mps);
        }
        eprintln!(
            "warn: Burn MLX is compiled in, but the Metal device probe failed; falling back to CPU"
        );
    }
    #[cfg(feature = "backend-burn-cuda")]
    {
        if cuda_device_available() {
            return Ok(BurnTarget::Cuda(0));
        }
    }
    Ok(BurnTarget::Cpu)
}

fn explicit_cuda_device() -> Result<BurnTarget> {
    #[cfg(feature = "backend-burn-cuda")]
    {
        if cuda_device_available() {
            return Ok(BurnTarget::Cuda(0));
        }
        bail!("Burn CUDA support is compiled in, but no usable CUDA device is available")
    }
    #[cfg(not(feature = "backend-burn-cuda"))]
    {
        bail!(
            "Burn CUDA support is not compiled into this build; re-run with the 'backend-burn-cuda' Cargo feature"
        )
    }
}

#[cfg(feature = "backend-burn-cuda")]
fn cuda_device_available() -> bool {
    use burn::prelude::DeviceOps;

    CudaDevice::device_count(0) > 0
}

fn explicit_mlx_device() -> Result<BurnTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        if mlx_device_available() {
            return Ok(BurnTarget::Mps);
        }
        bail!("Burn MPS was explicitly requested, but the burn-mlx Metal device probe failed")
    }
    #[cfg(not(all(target_os = "macos", feature = "backend-burn-mlx")))]
    {
        bail!(
            "burn-mlx support is only available on macOS builds with the 'backend-burn-mlx' Cargo feature"
        )
    }
}

#[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
fn mlx_device_available() -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    catch_unwind(AssertUnwindSafe(|| {
        let device = MlxDevice::Gpu;
        let tensor: Tensor<MlxBackend, 1> = Tensor::ones([1], &device);
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("evaluate Burn MLX probe tensor")
            .len()
            == 1
    }))
    .unwrap_or(false)
}

pub fn run(args: Args, mut stack: ml::ResolvedTrainingStack) -> Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    let target = resolve_device(stack.requested_runtime)?;
    stack.effective_runtime = target.runtime();
    ml::write_run_metadata(
        &args.outdir.join("training_stack.json"),
        &stack,
        Some(&args.algorithm),
        Some(data::OBSERVATION_SCHEMA_NORMALIZED),
    )?;
    print_device(target);

    match target {
        BurnTarget::Cpu => run_inner::<CpuBackend>(args, stack, NdArrayDevice::Cpu, true),
        #[cfg(feature = "backend-burn-cuda")]
        BurnTarget::Cuda(index) => {
            run_inner::<CudaBackend>(args, stack, CudaDevice::new(index), false)
        }
        #[cfg(not(feature = "backend-burn-cuda"))]
        BurnTarget::Cuda(_) => bail!("Burn CUDA support is not compiled into this build"),
        #[cfg(feature = "backend-burn-mlx")]
        BurnTarget::Mps => run_inner::<MlxBackend>(args, stack, MlxDevice::Gpu, false),
        #[cfg(not(feature = "backend-burn-mlx"))]
        BurnTarget::Mps => bail!("burn-mlx support is not compiled into this build"),
    }
}

fn print_device(target: BurnTarget) {
    match target {
        BurnTarget::Cpu => println!("info: burn RL backend using cpu (manual PPO/GRPO update)"),
        BurnTarget::Cuda(index) => {
            println!("info: burn RL backend using cuda:{index} (manual PPO/GRPO update)")
        }
        BurnTarget::Mps => {
            println!("info: burn RL backend using apple gpu (burn-mlx; manual PPO/GRPO update)")
        }
    }
}

fn run_inner<B: Backend>(
    args: Args,
    stack: ml::ResolvedTrainingStack,
    device: B::Device,
    host_linear: bool,
) -> Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    println!("info: run directory {}", args.outdir.display());
    if !(0.0..1.0).contains(&args.dropout) {
        bail!("--dropout must be in [0, 1), got {}", args.dropout);
    }

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().r#gen());
    B::seed(&device, seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let (train_path, val_path, test_path) = common::resolve_paths(&args)?;
    let train_symbol = data::read_symbol(&train_path)
        .with_context(|| format!("read symbol from {}", train_path.display()))?;
    let (margin_cfg, session_cfg) = common::load_symbol_config(&args.symbol_config, &train_symbol)?;
    let use_globex = match session_cfg.as_deref() {
        Some("rth") => false,
        Some("globex") => true,
        _ => !args.rth,
    };
    let bar_selection = BarSelection {
        bar_kind: args.bar_kind,
        volume_bar_size: args.volume_bar_size,
        price_source: args.price_source,
    };
    let train = data::load_dataset_with_schema_and_bars(
        &train_path,
        use_globex,
        ObservationSchema::NormalizedV2,
        bar_selection,
    )?
    .with_session(use_globex);
    let val = data::load_dataset_with_schema_and_bars(
        &val_path,
        use_globex,
        ObservationSchema::NormalizedV2,
        bar_selection,
    )?
    .with_session(use_globex);
    let test = data::load_dataset_with_schema_and_bars(
        &test_path,
        use_globex,
        ObservationSchema::NormalizedV2,
        bar_selection,
    )?
    .with_session(use_globex);

    let margin_mode = match args.margin_mode.as_str() {
        "per-contract" => MarginMode::PerContract,
        "price" => MarginMode::Price,
        _ => common::infer_margin_mode(&train_symbol, margin_cfg),
    };
    let margin_per_contract = args
        .margin_per_contract
        .or(margin_cfg)
        .unwrap_or_else(|| common::infer_margin(&train_symbol));
    let full_file = if args.parquet.is_some() {
        args.full_file
    } else {
        args.full_file || !args.windowed
    };
    let raw_windows = |dataset: &DataSet| {
        if full_file {
            vec![(0, dataset.close.len())]
        } else {
            midas_env::sampler::windows(dataset.close.len(), args.window, args.step)
        }
    };
    let min_window_start = midas_env::features::feature_warmup_bars().saturating_sub(1);
    let adjust_windows = |label: &str, windows: Vec<(usize, usize)>| {
        let before = windows.len();
        let windows = midas_env::sampler::enforce_min_start(&windows, min_window_start);
        let dropped = before.saturating_sub(windows.len());
        if dropped > 0 {
            println!(
                "info: dropped {dropped} {label} window(s) before feature warmup ({} bars)",
                min_window_start + 1
            );
        }
        windows
    };
    let mut train_windows = adjust_windows("train", raw_windows(&train));
    let val_windows = adjust_windows("val", raw_windows(&val));
    let test_windows = adjust_windows("test", raw_windows(&test));
    if train_windows.is_empty() {
        bail!("no training windows available after applying feature warmup");
    }

    let env_cfg = EnvConfig {
        max_position: args.max_position,
        margin_mode,
        contract_multiplier: if args.contract_multiplier > 0.0 {
            args.contract_multiplier
        } else {
            1.0
        },
        margin_per_contract,
        enforce_margin: !args.disable_margin,
        drawdown_penalty: args.drawdown_penalty,
        drawdown_penalty_growth: args.drawdown_penalty_growth,
        session_close_penalty: args.session_close_penalty,
        auto_close_minutes_before_close: args.auto_close_minutes_before_close,
        max_hold_bars_positive: args.max_hold_bars_positive,
        max_hold_bars_drawdown: args.max_hold_bars_drawdown,
        hold_duration_penalty: args.hold_duration_penalty,
        hold_duration_penalty_growth: args.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
        min_hold_bars: args.min_hold_bars,
        early_exit_penalty: args.early_exit_penalty,
        early_flip_penalty: args.early_flip_penalty,
        invalid_revert_penalty: args.invalid_revert_penalty,
        invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
        flat_hold_penalty: args.flat_hold_penalty,
        flat_hold_penalty_growth: args.flat_hold_penalty_growth,
        max_flat_hold_bars: args.max_flat_hold_bars,
        ..EnvConfig::default()
    };
    let use_grpo = args.algorithm == "grpo";
    println!(
        "info: Burn RL model input={} hidden={} layers={} action_dim={} backend={}",
        train.obs_dim,
        args.hidden,
        args.layers,
        ACTION_DIM,
        B::name(&device)
    );
    let mut policy: BurnMlp<B> = BurnMlp::new(
        train.obs_dim,
        args.hidden,
        args.layers,
        ACTION_DIM,
        &device,
        &mut rng,
        host_linear,
    );
    let mut value: Option<BurnMlp<B>> = (!use_grpo).then(|| {
        BurnMlp::new(
            train.obs_dim,
            args.hidden,
            args.layers,
            1,
            &device,
            &mut rng,
            host_linear,
        )
    });
    let mut policy_optimizer = AdamCheckpoint::new(&policy);
    let mut value_optimizer = value.as_ref().map(AdamCheckpoint::new);
    let mut start_epoch = 0usize;
    let mut best_eval_fitness = f64::NEG_INFINITY;
    let mut best_checkpoint: Option<BurnCheckpoint> = None;

    if let Some(path) = &args.load_checkpoint {
        let checkpoint = load_checkpoint(path)?;
        validate_checkpoint(&checkpoint, &args, train.obs_dim, use_grpo)?;
        policy = BurnMlp::from_checkpoint(&checkpoint.policy, &device, host_linear)?;
        value = checkpoint
            .value
            .as_ref()
            .map(|model| BurnMlp::from_checkpoint(model, &device, host_linear))
            .transpose()?;
        policy_optimizer = checkpoint.policy_optimizer;
        value_optimizer = checkpoint.value_optimizer;
        start_epoch = checkpoint.epoch.saturating_add(1);
        println!(
            "info: loaded Burn RL checkpoint {} at completed epoch {}",
            path.display(),
            checkpoint.epoch
        );
    }
    if args.dropout > 0.0 {
        println!(
            "info: Burn RL dropout set to {:.3} on hidden layers during training; eval/test disable dropout",
            args.dropout
        );
    }

    let log_path = args.outdir.join("rl_log.csv");
    common::ensure_csv_header(&log_path, common::RL_LOG_HEADER_V2)?;
    let training_start = Instant::now();
    let total_epochs = start_epoch.saturating_add(args.epochs);
    for epoch in start_epoch..total_epochs {
        let epoch_start = Instant::now();
        train_windows.shuffle(&mut rng);
        let train_count = if args.train_windows == 0 {
            train_windows.len()
        } else {
            args.train_windows.min(train_windows.len())
        };
        let mut train_metrics = Vec::with_capacity(train_count);
        let mut ppo_losses = Vec::with_capacity(train_count);
        let mut grpo_losses = Vec::with_capacity(train_count);

        if use_grpo {
            for window in train_windows.iter().take(train_count) {
                let group = rollout_group(
                    &train,
                    *window,
                    &policy,
                    &env_cfg,
                    &device,
                    &args,
                    true,
                    false,
                    args.group_size.max(1),
                    &mut rng,
                )?;
                let advantages = group_advantages(&group);
                let losses = grpo_update(
                    &mut policy,
                    &group,
                    &advantages,
                    &device,
                    &args,
                    &mut policy_optimizer,
                    &mut rng,
                )?;
                train_metrics.push(summarize_group(&group, args.sortino_annualization));
                grpo_losses.push(losses);
            }
        } else {
            for window in train_windows.iter().take(train_count) {
                let batch = rollout_ppo(
                    &train,
                    *window,
                    &policy,
                    value.as_ref().expect("PPO value model"),
                    &env_cfg,
                    &device,
                    &args,
                    true,
                    false,
                    &mut rng,
                )?;
                let losses = ppo_update(
                    &mut policy,
                    value.as_mut().expect("PPO value model"),
                    &batch,
                    &device,
                    &args,
                    &mut policy_optimizer,
                    value_optimizer.as_mut().expect("PPO value optimizer"),
                    &mut rng,
                )?;
                train_metrics.push(summarize_ppo(&batch, args.sortino_annualization));
                ppo_losses.push(losses);
            }
        }

        let train_summary = average_metrics(&train_metrics);
        let eval_summary = evaluate(
            &val,
            &val_windows,
            &policy,
            value.as_ref(),
            &env_cfg,
            &device,
            &args,
            use_grpo,
            true,
            &mut rng,
        )?;
        let probe_summary = if args.log_interval > 0 && epoch % args.log_interval == 0 {
            val_windows.first().map(|window| {
                evaluate_one(
                    &val,
                    *window,
                    &policy,
                    value.as_ref(),
                    &env_cfg,
                    &device,
                    &args,
                    use_grpo,
                    true,
                    &mut rng,
                )
            })
        } else {
            None
        }
        .transpose()?;
        let fitness_source = if args.fitness_use_eval {
            eval_summary
        } else {
            train_summary
        };
        let fitness = args.w_pnl * fitness_source.pnl + args.w_sortino * fitness_source.sortino
            - args.w_mdd * fitness_source.drawdown;
        let eval_fitness = args.w_pnl * eval_summary.pnl + args.w_sortino * eval_summary.sortino
            - args.w_mdd * eval_summary.drawdown;
        if eval_fitness > best_eval_fitness {
            best_eval_fitness = eval_fitness;
            best_checkpoint = Some(BurnCheckpoint {
                format_version: CHECKPOINT_FORMAT_VERSION,
                backend: stack.backend.as_str().to_string(),
                algorithm: args.algorithm.clone(),
                epoch,
                policy: policy.checkpoint(),
                value: value.as_ref().map(BurnMlp::checkpoint),
                policy_optimizer: policy_optimizer.clone(),
                value_optimizer: value_optimizer.clone(),
            });
            let metadata = serde_json::json!({
                "epoch": epoch,
                "eval_fitness": eval_fitness,
                "eval_pnl": eval_summary.pnl,
                "eval_sortino": eval_summary.sortino,
                "eval_drawdown": eval_summary.drawdown,
                "seed": args.seed,
            });
            std::fs::write(
                args.outdir.join("best_validation.json"),
                format!("{}\n", serde_json::to_string_pretty(&metadata)?),
            )?;
            save_checkpoint(
                &args.outdir.join("best_validation.burn.json"),
                &stack,
                &args,
                epoch,
                &policy,
                value.as_ref(),
                &policy_optimizer,
                value_optimizer.as_ref(),
            )?;
        }
        println!(
            "epoch {} | train ret {:.4} | train pnl {:.4} | eval pnl {:.4} | eval sortino {:.4} | eval mdd {:.4} | eval conf {:.3} | eval fitness {:.4} | fitness {:.4} | time {}",
            epoch,
            train_summary.ret_mean,
            train_summary.pnl,
            eval_summary.pnl,
            eval_summary.sortino,
            eval_summary.drawdown,
            eval_summary.mean_max_prob,
            eval_fitness,
            fitness,
            format_duration(epoch_start.elapsed())
        );

        if args.log_interval > 0 && epoch % args.log_interval == 0 {
            let mut file = OpenOptions::new().append(true).open(&log_path)?;
            if use_grpo {
                let loss = average_grpo_losses(&grpo_losses);
                write_rl_log_row(
                    &mut file,
                    epoch,
                    "grpo",
                    &train_summary,
                    &eval_summary,
                    probe_summary.as_ref(),
                    fitness,
                    loss.policy_loss,
                    None,
                    loss.entropy,
                    loss.total_loss,
                    loss.policy_grad_norm,
                    None,
                    None,
                    Some(loss.kl_div),
                    loss.clip_frac,
                )?;
            } else {
                let loss = average_ppo_losses(&ppo_losses);
                write_rl_log_row(
                    &mut file,
                    epoch,
                    "ppo",
                    &train_summary,
                    &eval_summary,
                    probe_summary.as_ref(),
                    fitness,
                    loss.policy_loss,
                    Some(loss.value_loss),
                    loss.entropy,
                    loss.total_loss,
                    loss.policy_grad_norm,
                    Some(loss.value_grad_norm),
                    Some(loss.approx_kl),
                    None,
                    loss.clip_frac,
                )?;
            }
        }

        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            let path = args
                .outdir
                .join(format!("checkpoint_epoch{epoch}.burn.json"));
            save_checkpoint(
                &path,
                &stack,
                &args,
                epoch,
                &policy,
                value.as_ref(),
                &policy_optimizer,
                value_optimizer.as_ref(),
            )?;
        }
    }

    let (
        selected_policy,
        selected_value,
        selected_policy_optimizer,
        selected_value_optimizer,
        selected_epoch,
    ) = if let Some(checkpoint) = best_checkpoint {
        let selected_policy = BurnMlp::from_checkpoint(&checkpoint.policy, &device, host_linear)?;
        let selected_value = checkpoint
            .value
            .as_ref()
            .map(|model| BurnMlp::from_checkpoint(model, &device, host_linear))
            .transpose()?;
        println!(
            "best validation checkpoint: epoch {} | eval fitness {:.4}",
            checkpoint.epoch, best_eval_fitness
        );
        (
            selected_policy,
            selected_value,
            checkpoint.policy_optimizer,
            checkpoint.value_optimizer,
            checkpoint.epoch,
        )
    } else {
        (
            policy,
            value,
            policy_optimizer,
            value_optimizer,
            total_epochs.saturating_sub(1),
        )
    };
    let test_summary = evaluate(
        &test,
        &test_windows,
        &selected_policy,
        selected_value.as_ref(),
        &env_cfg,
        &device,
        &args,
        use_grpo,
        true,
        &mut rng,
    )?;
    println!(
        "test | ret {:.4} | pnl {:.4} | sortino {:.4} | mdd {:.4}",
        test_summary.ret_mean, test_summary.pnl, test_summary.sortino, test_summary.drawdown
    );
    println!(
        "total training time: {}",
        format_duration(training_start.elapsed())
    );
    let final_path = if use_grpo {
        args.outdir.join("grpo_final.burn.json")
    } else {
        args.outdir.join("ppo_final.burn.json")
    };
    save_checkpoint(
        &final_path,
        &stack,
        &args,
        selected_epoch,
        &selected_policy,
        selected_value.as_ref(),
        &selected_policy_optimizer,
        selected_value_optimizer.as_ref(),
    )?;
    println!(
        "Saved final {} checkpoint to {}",
        if use_grpo { "GRPO" } else { "PPO" },
        final_path.display()
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct RolloutMetrics {
    ret_mean: f64,
    pnl: f64,
    realized_pnl: f64,
    sortino: f64,
    drawdown: f64,
    commission: f64,
    slippage: f64,
    buy_frac: f64,
    sell_frac: f64,
    hold_frac: f64,
    revert_frac: f64,
    mean_max_prob: f64,
    entries: f64,
    exits: f64,
    flips: f64,
    avg_hold: f64,
}

struct PpoBatch {
    observations: Vec<Vec<f32>>,
    actions: Vec<usize>,
    old_logp: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    pnl: Vec<f64>,
    returns_series: Vec<f64>,
    equity: Vec<f64>,
    realized_pnl: f64,
    commission: f64,
    slippage: f64,
    action_counts: [usize; ACTION_DIM],
    max_prob_sum: f64,
    entries: usize,
    exits: usize,
    flips: usize,
    total_trade_bars: usize,
}

struct GrpoRollout {
    observations: Vec<Vec<f32>>,
    actions: Vec<usize>,
    old_logp: Vec<f32>,
    reward: f64,
    pnl: f64,
    returns_series: Vec<f64>,
    equity: Vec<f64>,
    realized_pnl: f64,
    commission: f64,
    slippage: f64,
    action_counts: [usize; ACTION_DIM],
    max_prob_sum: f64,
    entries: usize,
    exits: usize,
    flips: usize,
    total_trade_bars: usize,
}

struct GrpoGroup {
    rollouts: Vec<GrpoRollout>,
    mean_reward: f64,
    std_reward: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PpoLossStats {
    policy_loss: f64,
    value_loss: f64,
    entropy: f64,
    total_loss: f64,
    policy_grad_norm: f64,
    value_grad_norm: f64,
    approx_kl: f64,
    clip_frac: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct GrpoLossStats {
    policy_loss: f64,
    entropy: f64,
    total_loss: f64,
    kl_div: f64,
    policy_grad_norm: f64,
    clip_frac: f64,
}

fn rollout_ppo<B: Backend>(
    data: &DataSet,
    window: (usize, usize),
    policy: &BurnMlp<B>,
    value: &BurnMlp<B>,
    env_cfg: &EnvConfig,
    device: &B::Device,
    args: &Args,
    train: bool,
    greedy: bool,
    rng: &mut StdRng,
) -> Result<PpoBatch> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let mut env = TradingEnv::new(data.close[start], args.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = args.initial_balance;
    let mut previous_equity = args.initial_balance;
    let mut observations = Vec::with_capacity(steps);
    let mut actions = Vec::with_capacity(steps);
    let mut old_logp = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(steps);
    let mut rewards = Vec::with_capacity(steps);
    let mut pnl = Vec::with_capacity(steps);
    let mut returns_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);
    let mut realized_pnl = 0.0;
    let mut commission = 0.0;
    let mut slippage = 0.0;
    let mut action_counts = [0usize; ACTION_DIM];
    let mut max_prob_sum = 0.0;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

    for t in (start + 1)..end {
        let observation = data::build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            args.initial_balance,
        );
        let policy_pass =
            policy.forward_with_cache(&observation, device, train, args.dropout as f32, rng)?;
        let probabilities = softmax(&policy_pass.logits);
        let action = if greedy {
            argmax_index(&probabilities)
        } else {
            sample_from_probs(&probabilities, rng)
        };
        let selected_prob = probabilities.get(action).copied().unwrap_or(1e-8).max(1e-8);
        let value_pass =
            value.forward_with_cache(&observation, device, train, args.dropout as f32, rng)?;
        let value_estimate = value_pass.logits.first().copied().unwrap_or(0.0);
        let position_before = position;
        let (reward, info) = env.step(
            action_from_index(action),
            data.close[t],
            step_context(data, args, t),
        );
        position = env.state().position;
        equity = env.state().cash + env.state().unrealized_pnl;
        let step_idx = t.saturating_sub(start + 1);
        observations.push(observation);
        actions.push(action);
        old_logp.push(selected_prob.ln());
        values.push(value_estimate);
        rewards.push(reward);
        pnl.push(info.pnl_change);
        let denominator = if previous_equity.abs() < 1e-8 {
            1e-8
        } else {
            previous_equity
        };
        returns_series.push(info.pnl_change / denominator);
        equity_curve.push(equity);
        previous_equity = equity;
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += probabilities.iter().copied().fold(0.0f32, f32::max) as f64;
        if let Some(count) = action_counts.get_mut(action) {
            *count += 1;
        }
        if position_before == 0 && position != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before != 0 && position != 0 && position_before.signum() != position.signum() {
            flips += 1;
        }
    }

    let (advantages, returns) = compute_gae(&rewards, &values, args.gamma, args.lam);
    Ok(PpoBatch {
        observations,
        actions,
        old_logp,
        advantages,
        returns,
        pnl,
        returns_series,
        equity: equity_curve,
        realized_pnl,
        commission,
        slippage,
        action_counts,
        max_prob_sum,
        entries,
        exits,
        flips,
        total_trade_bars,
    })
}

fn rollout_group<B: Backend>(
    data: &DataSet,
    window: (usize, usize),
    policy: &BurnMlp<B>,
    env_cfg: &EnvConfig,
    device: &B::Device,
    args: &Args,
    train: bool,
    greedy: bool,
    group_size: usize,
    rng: &mut StdRng,
) -> Result<GrpoGroup> {
    let mut rollouts = Vec::with_capacity(group_size);
    for _ in 0..group_size {
        rollouts.push(rollout_grpo(
            data, window, policy, env_cfg, device, args, train, greedy, rng,
        )?);
    }
    let rewards: Vec<f64> = rollouts.iter().map(|rollout| rollout.reward).collect();
    let mean_reward = if rewards.is_empty() {
        0.0
    } else {
        rewards.iter().sum::<f64>() / rewards.len() as f64
    };
    let variance = if rewards.is_empty() {
        0.0
    } else {
        rewards
            .iter()
            .map(|reward| (reward - mean_reward).powi(2))
            .sum::<f64>()
            / rewards.len() as f64
    };
    Ok(GrpoGroup {
        rollouts,
        mean_reward,
        std_reward: variance.sqrt().max(1e-8),
    })
}

fn rollout_grpo<B: Backend>(
    data: &DataSet,
    window: (usize, usize),
    policy: &BurnMlp<B>,
    env_cfg: &EnvConfig,
    device: &B::Device,
    args: &Args,
    train: bool,
    greedy: bool,
    rng: &mut StdRng,
) -> Result<GrpoRollout> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let mut env = TradingEnv::new(data.close[start], args.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = args.initial_balance;
    let mut previous_equity = args.initial_balance;
    let mut observations = Vec::with_capacity(steps);
    let mut actions = Vec::with_capacity(steps);
    let mut old_logp = Vec::with_capacity(steps);
    let mut returns_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);
    let mut total_reward = 0.0;
    let mut total_pnl = 0.0;
    let mut realized_pnl = 0.0;
    let mut commission = 0.0;
    let mut slippage = 0.0;
    let mut action_counts = [0usize; ACTION_DIM];
    let mut max_prob_sum = 0.0;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

    for t in (start + 1)..end {
        let observation = data::build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            args.initial_balance,
        );
        let pass =
            policy.forward_with_cache(&observation, device, train, args.dropout as f32, rng)?;
        let probabilities = softmax(&pass.logits);
        let action = if greedy {
            argmax_index(&probabilities)
        } else {
            sample_from_probs(&probabilities, rng)
        };
        let selected_prob = probabilities.get(action).copied().unwrap_or(1e-8).max(1e-8);
        let position_before = position;
        let (reward, info) = env.step(
            action_from_index(action),
            data.close[t],
            step_context(data, args, t),
        );
        position = env.state().position;
        equity = env.state().cash + env.state().unrealized_pnl;
        let step_idx = t.saturating_sub(start + 1);
        observations.push(observation);
        actions.push(action);
        old_logp.push(selected_prob.ln());
        total_reward += reward;
        total_pnl += info.pnl_change;
        let denominator = if previous_equity.abs() < 1e-8 {
            1e-8
        } else {
            previous_equity
        };
        returns_series.push(info.pnl_change / denominator);
        equity_curve.push(equity);
        previous_equity = equity;
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += probabilities.iter().copied().fold(0.0f32, f32::max) as f64;
        if let Some(count) = action_counts.get_mut(action) {
            *count += 1;
        }
        if position_before == 0 && position != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before != 0 && position != 0 && position_before.signum() != position.signum() {
            flips += 1;
        }
    }

    Ok(GrpoRollout {
        observations,
        actions,
        old_logp,
        reward: total_reward,
        pnl: total_pnl,
        returns_series,
        equity: equity_curve,
        realized_pnl,
        commission,
        slippage,
        action_counts,
        max_prob_sum,
        entries,
        exits,
        flips,
        total_trade_bars,
    })
}

fn step_context(data: &DataSet, args: &Args, t: usize) -> StepContext {
    let session_open = if args.ignore_session {
        true
    } else {
        data.session_open
            .as_ref()
            .and_then(|mask| mask.get(t))
            .copied()
            .unwrap_or(true)
    };
    let minutes_to_close = data
        .minutes_to_close
        .as_ref()
        .and_then(|mask| mask.get(t))
        .copied();
    let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);
    StepContext {
        session_open,
        margin_ok,
        minutes_to_close,
    }
}

fn action_from_index(index: usize) -> Action {
    match index {
        0 => Action::Buy,
        1 => Action::Sell,
        2 => Action::Hold,
        _ => Action::Revert,
    }
}

fn compute_gae(rewards: &[f64], values: &[f32], gamma: f64, lam: f64) -> (Vec<f32>, Vec<f32>) {
    let mut advantages = vec![0.0f32; rewards.len()];
    let mut returns = vec![0.0f32; rewards.len()];
    let mut gae = 0.0;
    let mut next_value = 0.0;
    for index in (0..rewards.len()).rev() {
        let value = values.get(index).copied().unwrap_or(0.0) as f64;
        let delta = rewards[index] + gamma * next_value - value;
        gae = delta + gamma * lam * gae;
        advantages[index] = gae as f32;
        returns[index] = (gae + value) as f32;
        next_value = value;
    }
    (advantages, returns)
}

fn group_advantages(group: &GrpoGroup) -> Vec<f64> {
    group
        .rollouts
        .iter()
        .map(|rollout| (rollout.reward - group.mean_reward) / group.std_reward)
        .collect()
}

fn normalize_advantages(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (*value as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt().max(1e-8);
    values
        .iter()
        .map(|value| ((*value as f64 - mean) / std) as f32)
        .collect()
}

fn normalize_advantages_f64(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt().max(1e-8);
    values.iter().map(|value| (*value - mean) / std).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect();
    let sum = values.iter().sum::<f32>().max(1e-8);
    for value in &mut values {
        *value /= sum;
    }
    values
}

fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
    let weights: Vec<f64> = probs
        .iter()
        .map(|value| {
            if value.is_finite() && *value > 0.0 {
                *value as f64
            } else {
                0.0
            }
        })
        .collect();
    WeightedIndex::new(weights)
        .map(|distribution| distribution.sample(rng))
        .unwrap_or_else(|_| argmax_index(probs))
}

fn summarize_ppo(batch: &PpoBatch, annualization: f64) -> RolloutMetrics {
    let steps = batch.returns_series.len().max(1) as f64;
    RolloutMetrics {
        ret_mean: mean(&batch.returns_series),
        pnl: batch.pnl.iter().sum(),
        realized_pnl: batch.realized_pnl,
        sortino: compute_sortino(&batch.returns_series, annualization, 0.0, 50.0),
        drawdown: max_drawdown(&batch.equity),
        commission: batch.commission,
        slippage: batch.slippage,
        buy_frac: batch.action_counts[0] as f64 / steps,
        sell_frac: batch.action_counts[1] as f64 / steps,
        hold_frac: batch.action_counts[2] as f64 / steps,
        revert_frac: batch.action_counts[3] as f64 / steps,
        mean_max_prob: batch.max_prob_sum / steps,
        entries: batch.entries as f64,
        exits: batch.exits as f64,
        flips: batch.flips as f64,
        avg_hold: if batch.exits > 0 {
            batch.total_trade_bars as f64 / batch.exits as f64
        } else {
            0.0
        },
    }
}

fn summarize_group(group: &GrpoGroup, annualization: f64) -> RolloutMetrics {
    let all_returns: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|rollout| rollout.returns_series.iter().copied())
        .collect();
    let all_equity: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|rollout| rollout.equity.iter().copied())
        .collect();
    let mut action_counts = [0usize; ACTION_DIM];
    let mut max_prob_sum = 0.0;
    let mut steps = 0usize;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut total_trade_bars = 0usize;
    for rollout in &group.rollouts {
        for (index, count) in rollout.action_counts.iter().enumerate() {
            action_counts[index] += *count;
            steps += *count;
        }
        max_prob_sum += rollout.max_prob_sum;
        entries += rollout.entries;
        exits += rollout.exits;
        flips += rollout.flips;
        total_trade_bars += rollout.total_trade_bars;
    }
    RolloutMetrics {
        ret_mean: mean(&all_returns),
        pnl: group.rollouts.iter().map(|rollout| rollout.pnl).sum(),
        realized_pnl: group
            .rollouts
            .iter()
            .map(|rollout| rollout.realized_pnl)
            .sum(),
        sortino: compute_sortino(&all_returns, annualization, 0.0, 50.0),
        drawdown: max_drawdown(&all_equity),
        commission: group
            .rollouts
            .iter()
            .map(|rollout| rollout.commission)
            .sum(),
        slippage: group.rollouts.iter().map(|rollout| rollout.slippage).sum(),
        buy_frac: action_counts[0] as f64 / steps.max(1) as f64,
        sell_frac: action_counts[1] as f64 / steps.max(1) as f64,
        hold_frac: action_counts[2] as f64 / steps.max(1) as f64,
        revert_frac: action_counts[3] as f64 / steps.max(1) as f64,
        mean_max_prob: max_prob_sum / steps.max(1) as f64,
        entries: entries as f64,
        exits: exits as f64,
        flips: flips as f64,
        avg_hold: if exits > 0 {
            total_trade_bars as f64 / exits as f64
        } else {
            0.0
        },
    }
}

fn average_metrics(values: &[RolloutMetrics]) -> RolloutMetrics {
    if values.is_empty() {
        return RolloutMetrics::default();
    }
    let mut total = RolloutMetrics::default();
    for value in values {
        total.ret_mean += value.ret_mean;
        total.pnl += value.pnl;
        total.realized_pnl += value.realized_pnl;
        total.sortino += value.sortino;
        total.drawdown += value.drawdown;
        total.commission += value.commission;
        total.slippage += value.slippage;
        total.buy_frac += value.buy_frac;
        total.sell_frac += value.sell_frac;
        total.hold_frac += value.hold_frac;
        total.revert_frac += value.revert_frac;
        total.mean_max_prob += value.mean_max_prob;
        total.entries += value.entries;
        total.exits += value.exits;
        total.flips += value.flips;
        total.avg_hold += value.avg_hold;
    }
    let denominator = values.len() as f64;
    RolloutMetrics {
        ret_mean: total.ret_mean / denominator,
        pnl: total.pnl / denominator,
        realized_pnl: total.realized_pnl / denominator,
        sortino: total.sortino / denominator,
        drawdown: total.drawdown / denominator,
        commission: total.commission / denominator,
        slippage: total.slippage / denominator,
        buy_frac: total.buy_frac / denominator,
        sell_frac: total.sell_frac / denominator,
        hold_frac: total.hold_frac / denominator,
        revert_frac: total.revert_frac / denominator,
        mean_max_prob: total.mean_max_prob / denominator,
        entries: total.entries / denominator,
        exits: total.exits / denominator,
        flips: total.flips / denominator,
        avg_hold: total.avg_hold / denominator,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn evaluate<B: Backend>(
    data: &DataSet,
    windows: &[(usize, usize)],
    policy: &BurnMlp<B>,
    value: Option<&BurnMlp<B>>,
    env_cfg: &EnvConfig,
    device: &B::Device,
    args: &Args,
    use_grpo: bool,
    greedy: bool,
    rng: &mut StdRng,
) -> Result<RolloutMetrics> {
    let count = args.eval_windows.min(windows.len()).max(1);
    let mut metrics = Vec::with_capacity(count);
    for window in windows.iter().take(count) {
        metrics.push(evaluate_one(
            data, *window, policy, value, env_cfg, device, args, use_grpo, greedy, rng,
        )?);
    }
    Ok(average_metrics(&metrics))
}

fn evaluate_one<B: Backend>(
    data: &DataSet,
    window: (usize, usize),
    policy: &BurnMlp<B>,
    value: Option<&BurnMlp<B>>,
    env_cfg: &EnvConfig,
    device: &B::Device,
    args: &Args,
    use_grpo: bool,
    greedy: bool,
    rng: &mut StdRng,
) -> Result<RolloutMetrics> {
    if use_grpo {
        let group = rollout_group(
            data, window, policy, env_cfg, device, args, false, greedy, 1, rng,
        )?;
        Ok(summarize_group(&group, args.sortino_annualization))
    } else {
        let value = value.context("PPO evaluation requires a value model")?;
        let batch = rollout_ppo(
            data, window, policy, value, env_cfg, device, args, false, greedy, rng,
        )?;
        Ok(summarize_ppo(&batch, args.sortino_annualization))
    }
}

fn ppo_update<B: Backend>(
    policy: &mut BurnMlp<B>,
    value: &mut BurnMlp<B>,
    batch: &PpoBatch,
    device: &B::Device,
    args: &Args,
    policy_optimizer: &mut AdamCheckpoint,
    value_optimizer: &mut AdamCheckpoint,
    rng: &mut StdRng,
) -> Result<PpoLossStats> {
    let normalized_advantages = normalize_advantages(&batch.advantages);
    let sample_count = batch.observations.len().max(1) as f32;
    let epochs = args.ppo_epochs.max(1);
    let mut total = PpoLossStats::default();

    for _ in 0..epochs {
        let mut policy_grad = policy.zero_grad();
        let mut value_grad = value.zero_grad();
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;
        let mut entropy = 0.0;
        let mut approx_kl = 0.0;
        let mut clipped = 0usize;

        for index in 0..batch.observations.len() {
            let policy_pass = policy.forward_with_cache(
                &batch.observations[index],
                device,
                true,
                args.dropout as f32,
                rng,
            )?;
            let probabilities = softmax(&policy_pass.logits);
            let action = batch.actions[index].min(ACTION_DIM - 1);
            let new_logp = probabilities[action].max(1e-8).ln();
            let log_ratio = (new_logp - batch.old_logp[index]).clamp(-20.0, 20.0);
            let ratio = log_ratio.exp();
            let advantage = normalized_advantages[index] as f64;
            let unclipped = ratio as f64 * advantage;
            let clipped_ratio = (ratio as f64).clamp(1.0 - args.clip, 1.0 + args.clip);
            let clipped_objective = clipped_ratio * advantage;
            policy_loss += -unclipped.min(clipped_objective) / sample_count as f64;
            let is_clipped = (advantage >= 0.0 && (ratio as f64) > 1.0 + args.clip)
                || (advantage < 0.0 && (ratio as f64) < 1.0 - args.clip);
            if is_clipped {
                clipped += 1;
            }

            let mut output_grad = vec![0.0f32; ACTION_DIM];
            if !is_clipped {
                let coefficient = -(advantage as f32) * ratio as f32 / sample_count;
                for (class, gradient) in output_grad.iter_mut().enumerate() {
                    *gradient += coefficient
                        * (if class == action {
                            1.0 - probabilities[class]
                        } else {
                            -probabilities[class]
                        });
                }
            }
            let sample_entropy = probabilities
                .iter()
                .map(|probability| {
                    let probability = probability.max(1e-8);
                    -(probability as f64) * probability.ln() as f64
                })
                .sum::<f64>();
            entropy += sample_entropy / sample_count as f64;
            for (class, gradient) in output_grad.iter_mut().enumerate() {
                let probability = probabilities[class];
                *gradient += args.ent_coef as f32
                    * probability
                    * (probability.max(1e-8).ln() + sample_entropy as f32)
                    / sample_count;
            }
            policy.accumulate_backward(&policy_pass.cache, &output_grad, &mut policy_grad);

            let value_pass = value.forward_with_cache(
                &batch.observations[index],
                device,
                true,
                args.dropout as f32,
                rng,
            )?;
            let predicted = value_pass.logits.first().copied().unwrap_or(0.0);
            let difference = predicted - batch.returns[index];
            value_loss += (difference as f64).powi(2) / sample_count as f64;
            let value_output_grad = [2.0 * args.vf_coef as f32 * difference / sample_count];
            value.accumulate_backward(&value_pass.cache, &value_output_grad, &mut value_grad);
            approx_kl += ((ratio as f64 - 1.0) - log_ratio as f64) / sample_count as f64;
        }

        let policy_grad_norm_value = policy_grad_norm(&policy_grad);
        let value_grad_norm_value = policy_grad_norm(&value_grad);
        let total_loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy;
        policy_optimizer.update(policy, &policy_grad, args.lr, device);
        value_optimizer.update(value, &value_grad, args.lr, device);
        total.policy_loss += policy_loss;
        total.value_loss += value_loss;
        total.entropy += entropy;
        total.total_loss += total_loss;
        total.policy_grad_norm += policy_grad_norm_value;
        total.value_grad_norm += value_grad_norm_value;
        total.approx_kl += approx_kl;
        total.clip_frac += clipped as f64 / sample_count as f64;
    }
    let denominator = epochs as f64;
    Ok(PpoLossStats {
        policy_loss: total.policy_loss / denominator,
        value_loss: total.value_loss / denominator,
        entropy: total.entropy / denominator,
        total_loss: total.total_loss / denominator,
        policy_grad_norm: total.policy_grad_norm / denominator,
        value_grad_norm: total.value_grad_norm / denominator,
        approx_kl: total.approx_kl / denominator,
        clip_frac: total.clip_frac / denominator,
    })
}

fn grpo_update<B: Backend>(
    policy: &mut BurnMlp<B>,
    group: &GrpoGroup,
    advantages: &[f64],
    device: &B::Device,
    args: &Args,
    optimizer: &mut AdamCheckpoint,
    rng: &mut StdRng,
) -> Result<GrpoLossStats> {
    let normalized_advantages = normalize_advantages_f64(advantages);
    let epochs = args.grpo_epochs.max(1);
    let mut total = GrpoLossStats::default();
    for _ in 0..epochs {
        let mut epoch = GrpoLossStats::default();
        let mut gradients = policy.zero_grad();
        for (rollout_index, rollout) in group.rollouts.iter().enumerate() {
            let count = rollout.observations.len().max(1) as f32;
            let advantage = normalized_advantages
                .get(rollout_index)
                .copied()
                .unwrap_or(0.0);
            let mut policy_loss = 0.0;
            let mut entropy = 0.0;
            let mut kl = 0.0;
            let mut clipped = 0usize;
            for index in 0..rollout.observations.len() {
                let pass = policy.forward_with_cache(
                    &rollout.observations[index],
                    device,
                    true,
                    args.dropout as f32,
                    rng,
                )?;
                let probabilities = softmax(&pass.logits);
                let action = rollout.actions[index].min(ACTION_DIM - 1);
                let new_logp = probabilities[action].max(1e-8).ln();
                let log_ratio = (new_logp - rollout.old_logp[index]).clamp(-20.0, 20.0);
                let ratio = log_ratio.exp();
                let unclipped = ratio as f64 * advantage;
                let clipped_ratio = (ratio as f64).clamp(1.0 - args.clip, 1.0 + args.clip);
                policy_loss += -unclipped.min(clipped_ratio * advantage) / count as f64;
                let is_clipped = (advantage >= 0.0 && (ratio as f64) > 1.0 + args.clip)
                    || (advantage < 0.0 && (ratio as f64) < 1.0 - args.clip);
                if is_clipped {
                    clipped += 1;
                }
                let mut output_grad = vec![0.0f32; ACTION_DIM];
                if !is_clipped {
                    let coefficient = -(advantage as f32) * ratio as f32 / count;
                    for (class, gradient) in output_grad.iter_mut().enumerate() {
                        *gradient += coefficient
                            * (if class == action {
                                1.0 - probabilities[class]
                            } else {
                                -probabilities[class]
                            });
                    }
                }
                let sample_entropy = probabilities
                    .iter()
                    .map(|probability| {
                        let probability = probability.max(1e-8);
                        -(probability as f64) * probability.ln() as f64
                    })
                    .sum::<f64>();
                entropy += sample_entropy / count as f64;
                for (class, gradient) in output_grad.iter_mut().enumerate() {
                    let probability = probabilities[class];
                    *gradient += args.ent_coef as f32
                        * probability
                        * (probability.max(1e-8).ln() + sample_entropy as f32)
                        / count;
                }
                policy.accumulate_backward(&pass.cache, &output_grad, &mut gradients);
                kl += ((rollout.old_logp[index] as f64 - new_logp as f64).abs()) / count as f64;
            }
            epoch.policy_loss += policy_loss;
            epoch.entropy += entropy;
            epoch.total_loss += policy_loss - args.ent_coef * entropy;
            epoch.kl_div += kl;
            epoch.clip_frac += clipped as f64 / count as f64;
        }
        let group_count = group.rollouts.len().max(1) as f32;
        scale_grad(&mut gradients, 1.0 / group_count);
        let grad_norm = policy_grad_norm(&gradients);
        optimizer.update(policy, &gradients, args.lr, device);
        epoch.policy_grad_norm = grad_norm;
        let denominator = group.rollouts.len().max(1) as f64;
        total.policy_loss += epoch.policy_loss / denominator;
        total.entropy += epoch.entropy / denominator;
        total.total_loss += epoch.total_loss / denominator;
        total.kl_div += epoch.kl_div / denominator;
        total.policy_grad_norm += epoch.policy_grad_norm / denominator;
        total.clip_frac += epoch.clip_frac / denominator;
    }
    let denominator = epochs as f64;
    Ok(GrpoLossStats {
        policy_loss: total.policy_loss / denominator,
        entropy: total.entropy / denominator,
        total_loss: total.total_loss / denominator,
        kl_div: total.kl_div / denominator,
        policy_grad_norm: total.policy_grad_norm / denominator,
        clip_frac: total.clip_frac / denominator,
    })
}

fn policy_grad_norm(grad: &MlpGrad) -> f64 {
    let mut sum = 0.0;
    for layer in &grad.weight {
        for value in layer {
            sum += (*value as f64).powi(2);
        }
    }
    for layer in &grad.bias {
        for value in layer {
            sum += (*value as f64).powi(2);
        }
    }
    sum.sqrt()
}

fn scale_grad(grad: &mut MlpGrad, factor: f32) {
    for layer in &mut grad.weight {
        for value in layer {
            *value *= factor;
        }
    }
    for layer in &mut grad.bias {
        for value in layer {
            *value *= factor;
        }
    }
}

fn average_ppo_losses(values: &[PpoLossStats]) -> PpoLossStats {
    if values.is_empty() {
        return PpoLossStats::default();
    }
    let mut total = PpoLossStats::default();
    for value in values {
        total.policy_loss += value.policy_loss;
        total.value_loss += value.value_loss;
        total.entropy += value.entropy;
        total.total_loss += value.total_loss;
        total.policy_grad_norm += value.policy_grad_norm;
        total.value_grad_norm += value.value_grad_norm;
        total.approx_kl += value.approx_kl;
        total.clip_frac += value.clip_frac;
    }
    let denominator = values.len() as f64;
    PpoLossStats {
        policy_loss: total.policy_loss / denominator,
        value_loss: total.value_loss / denominator,
        entropy: total.entropy / denominator,
        total_loss: total.total_loss / denominator,
        policy_grad_norm: total.policy_grad_norm / denominator,
        value_grad_norm: total.value_grad_norm / denominator,
        approx_kl: total.approx_kl / denominator,
        clip_frac: total.clip_frac / denominator,
    }
}

fn average_grpo_losses(values: &[GrpoLossStats]) -> GrpoLossStats {
    if values.is_empty() {
        return GrpoLossStats::default();
    }
    let mut total = GrpoLossStats::default();
    for value in values {
        total.policy_loss += value.policy_loss;
        total.entropy += value.entropy;
        total.total_loss += value.total_loss;
        total.kl_div += value.kl_div;
        total.policy_grad_norm += value.policy_grad_norm;
        total.clip_frac += value.clip_frac;
    }
    let denominator = values.len() as f64;
    GrpoLossStats {
        policy_loss: total.policy_loss / denominator,
        entropy: total.entropy / denominator,
        total_loss: total.total_loss / denominator,
        kl_div: total.kl_div / denominator,
        policy_grad_norm: total.policy_grad_norm / denominator,
        clip_frac: total.clip_frac / denominator,
    }
}

fn save_checkpoint<B: Backend>(
    path: &std::path::Path,
    stack: &ml::ResolvedTrainingStack,
    args: &Args,
    epoch: usize,
    policy: &BurnMlp<B>,
    value: Option<&BurnMlp<B>>,
    policy_optimizer: &AdamCheckpoint,
    value_optimizer: Option<&AdamCheckpoint>,
) -> Result<()> {
    let checkpoint = BurnCheckpoint {
        format_version: CHECKPOINT_FORMAT_VERSION,
        backend: stack.backend.as_str().to_string(),
        algorithm: args.algorithm.clone(),
        epoch,
        policy: policy.checkpoint(),
        value: value.map(BurnMlp::checkpoint),
        policy_optimizer: policy_optimizer.clone(),
        value_optimizer: value_optimizer.cloned(),
    };
    let json = serde_json::to_string_pretty(&checkpoint)?;
    std::fs::write(path, json)
        .with_context(|| format!("write Burn RL checkpoint {}", path.display()))?;
    Ok(())
}

fn load_checkpoint(path: &std::path::Path) -> Result<BurnCheckpoint> {
    let candidate = if path.exists() {
        path.to_path_buf()
    } else if path.extension().is_none() {
        let mut candidate = path.to_path_buf();
        candidate.set_extension("burn.json");
        candidate
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(&candidate)
        .with_context(|| format!("read Burn RL checkpoint {}", candidate.display()))?;
    let checkpoint: BurnCheckpoint = serde_json::from_str(&text)
        .with_context(|| format!("parse Burn RL checkpoint {}", candidate.display()))?;
    if checkpoint.format_version != CHECKPOINT_FORMAT_VERSION {
        bail!(
            "unsupported Burn RL checkpoint format {}; expected {}",
            checkpoint.format_version,
            CHECKPOINT_FORMAT_VERSION
        );
    }
    Ok(checkpoint)
}

fn validate_checkpoint(
    checkpoint: &BurnCheckpoint,
    args: &Args,
    input_dim: usize,
    use_grpo: bool,
) -> Result<()> {
    let algorithm = if use_grpo { "grpo" } else { "ppo" };
    if checkpoint.algorithm != algorithm {
        bail!(
            "Burn RL checkpoint algorithm '{}' does not match requested '{}'",
            checkpoint.algorithm,
            algorithm
        );
    }
    if checkpoint.policy.input_dim != input_dim
        || checkpoint.policy.hidden != args.hidden
        || checkpoint.policy.hidden_layers != args.layers
        || checkpoint.policy.output_dim != ACTION_DIM
    {
        bail!(
            "Burn RL checkpoint policy architecture does not match the current run (input={}, hidden={}, layers={}, output={})",
            input_dim,
            args.hidden,
            args.layers,
            ACTION_DIM
        );
    }
    if use_grpo != checkpoint.value.is_none() {
        bail!("Burn RL checkpoint value model presence does not match the requested algorithm");
    }
    if let Some(value) = &checkpoint.value
        && (value.input_dim != input_dim
            || value.hidden != args.hidden
            || value.hidden_layers != args.layers
            || value.output_dim != 1)
    {
        bail!("Burn RL checkpoint value architecture does not match the current run");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_cpu_policy_forward_and_update_smoke() {
        let device = NdArrayDevice::Cpu;
        let mut rng = StdRng::seed_from_u64(7);
        let mut model = BurnMlp::<CpuBackend>::new(3, 4, 1, ACTION_DIM, &device, &mut rng, true);
        let forward = model
            .forward_with_cache(&[0.25, -0.5, 0.75], &device, true, 0.0, &mut rng)
            .expect("Burn CPU forward");
        assert_eq!(forward.logits.len(), ACTION_DIM);
        assert!(forward.logits.iter().all(|value| value.is_finite()));

        let mut gradients = model.zero_grad();
        model.accumulate_backward(&forward.cache, &[1.0, -0.5, 0.25, 0.0], &mut gradients);
        let mut optimizer = AdamCheckpoint::new(&model);
        let norm = optimizer.update(&mut model, &gradients, 0.001, &device);
        assert!(norm.is_finite());
        let updated = model
            .forward_with_cache(&[0.25, -0.5, 0.75], &device, false, 0.0, &mut rng)
            .expect("Burn CPU post-update forward");
        assert!(updated.logits.iter().all(|value| value.is_finite()));
    }
}

fn write_rl_log_row(
    file: &mut std::fs::File,
    epoch: usize,
    algorithm: &str,
    train: &RolloutMetrics,
    eval: &RolloutMetrics,
    probe: Option<&RolloutMetrics>,
    fitness: f64,
    policy_loss: f64,
    value_loss: Option<f64>,
    entropy: f64,
    total_loss: f64,
    policy_grad_norm: f64,
    value_grad_norm: Option<f64>,
    approx_kl: Option<f64>,
    kl_div: Option<f64>,
    clip_frac: f64,
) -> Result<()> {
    let mut fields = Vec::with_capacity(60);
    fields.push(epoch.to_string());
    fields.push(algorithm.to_string());
    append_rollout_fields(&mut fields, Some(train));
    append_rollout_fields(&mut fields, Some(eval));
    append_rollout_fields(&mut fields, probe);
    fields.push(format_log_f64(fitness, 4));
    fields.push(format_log_f64(policy_loss, 6));
    fields.push(format_log_opt(value_loss, 6));
    fields.push(format_log_f64(entropy, 6));
    fields.push(format_log_f64(entropy.exp(), 6));
    fields.push(format_log_f64(total_loss, 6));
    fields.push(format_log_f64(policy_grad_norm, 6));
    fields.push(format_log_opt(value_grad_norm, 6));
    fields.push(format_log_opt(approx_kl, 6));
    fields.push(format_log_opt(kl_div, 6));
    fields.push(format_log_f64(clip_frac, 6));
    writeln!(file, "{}", fields.join(","))?;
    Ok(())
}

fn append_rollout_fields(fields: &mut Vec<String>, metrics: Option<&RolloutMetrics>) {
    let Some(metrics) = metrics else {
        for _ in 0..16 {
            fields.push(String::new());
        }
        return;
    };
    fields.push(format_log_f64(metrics.ret_mean, 4));
    fields.push(format_log_f64(metrics.pnl, 4));
    fields.push(format_log_f64(metrics.realized_pnl, 4));
    fields.push(format_log_f64(metrics.sortino, 4));
    fields.push(format_log_f64(metrics.drawdown, 4));
    fields.push(format_log_f64(metrics.commission, 4));
    fields.push(format_log_f64(metrics.slippage, 4));
    fields.push(format_log_f64(metrics.buy_frac, 6));
    fields.push(format_log_f64(metrics.sell_frac, 6));
    fields.push(format_log_f64(metrics.hold_frac, 6));
    fields.push(format_log_f64(metrics.revert_frac, 6));
    fields.push(format_log_f64(metrics.mean_max_prob, 6));
    fields.push(format_log_f64(metrics.entries, 4));
    fields.push(format_log_f64(metrics.exits, 4));
    fields.push(format_log_f64(metrics.flips, 4));
    fields.push(format_log_f64(metrics.avg_hold, 4));
}

fn format_log_f64(value: f64, precision: usize) -> String {
    format!("{value:.precision$}")
}

fn format_log_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|value| format_log_f64(value, precision))
        .unwrap_or_default()
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let millis = duration.subsec_millis();
    let minutes = seconds / 60;
    let seconds = seconds % 60;
    if minutes > 0 {
        format!("{minutes}m{seconds:02}.{millis:03}s")
    } else {
        format!("{seconds}.{millis:03}s")
    }
}
