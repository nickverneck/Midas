use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
use burn_cpu::{Cpu, CpuDevice};
use midas_env::ml::ComputeRuntime;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use std::path::Path;

#[cfg(feature = "backend-burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "backend-burn-mlx")]
use burn_mlx::{Mlx, MlxDevice};
#[cfg(feature = "backend-burn-ndarray")]
use burn_ndarray::NdArray;

use crate::config::{CandidateConfig, ExecutionTarget};
use crate::data::{DataSet, build_observation};
use crate::metrics::{compute_sortino, max_drawdown};
use crate::types::{BehaviorRow, CandidateResult};
use midas_env::env::VIOLATION_PENALTY;

type CpuBackend = Cpu<f32, i32>;
#[cfg(feature = "backend-burn-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "backend-burn-mlx")]
type MlxBackend = Mlx<f32>;
#[cfg(feature = "backend-burn-ndarray")]
type LegacyCpuBackend = NdArray<f32>;

struct LinearLayer<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

struct BurnPolicy<B: Backend> {
    layers: Vec<LinearLayer<B>>,
}

struct CandidateStats {
    eval_pnls: Vec<f64>,
    eval_pnls_realized: Vec<f64>,
    eval_pnls_total: Vec<f64>,
    eval_returns: Vec<f64>,
    eval_equity: Vec<Vec<f64>>,
    non_hold: usize,
    non_zero_pos: usize,
    abs_pnl_sum: f64,
    pnl_steps: usize,
    act_buy: usize,
    act_sell: usize,
    act_hold: usize,
    act_revert: usize,
    session_violations: usize,
    margin_violations: usize,
    position_violations: usize,
    drawdown_penalty_sum: f64,
    invalid_revert_penalty_sum: f64,
    hold_duration_penalty_sum: f64,
    flat_hold_penalty_sum: f64,
    session_close_penalty_sum: f64,
    violation_penalty_sum: f64,
}

impl CandidateStats {
    fn new() -> Self {
        Self {
            eval_pnls: Vec::new(),
            eval_pnls_realized: Vec::new(),
            eval_pnls_total: Vec::new(),
            eval_returns: Vec::new(),
            eval_equity: Vec::new(),
            non_hold: 0,
            non_zero_pos: 0,
            abs_pnl_sum: 0.0,
            pnl_steps: 0,
            act_buy: 0,
            act_sell: 0,
            act_hold: 0,
            act_revert: 0,
            session_violations: 0,
            margin_violations: 0,
            position_violations: 0,
            drawdown_penalty_sum: 0.0,
            invalid_revert_penalty_sum: 0.0,
            hold_duration_penalty_sum: 0.0,
            flat_hold_penalty_sum: 0.0,
            session_close_penalty_sum: 0.0,
            violation_penalty_sum: 0.0,
        }
    }

    fn finish(self, cfg: &CandidateConfig) -> CandidateResult {
        let eval_sortino =
            compute_sortino(&self.eval_returns, cfg.sortino_annualization, 0.0, 50.0);
        let eval_draw = self
            .eval_equity
            .iter()
            .map(|eq| max_drawdown(eq))
            .fold(0.0_f64, |a, b| a.max(b));
        let eval_pnl = if self.eval_pnls.is_empty() {
            0.0
        } else {
            self.eval_pnls.iter().sum::<f64>() / self.eval_pnls.len() as f64
        };
        let eval_pnl_realized = if self.eval_pnls_realized.is_empty() {
            0.0
        } else {
            self.eval_pnls_realized.iter().sum::<f64>() / self.eval_pnls_realized.len() as f64
        };
        let eval_pnl_total = if self.eval_pnls_total.is_empty() {
            0.0
        } else {
            self.eval_pnls_total.iter().sum::<f64>() / self.eval_pnls_total.len() as f64
        };

        let fitness = cfg.w_pnl * eval_pnl + cfg.w_sortino * eval_sortino - cfg.w_mdd * eval_draw;

        CandidateResult {
            fitness,
            eval_pnl,
            eval_pnl_realized,
            eval_pnl_total,
            eval_sortino,
            eval_drawdown: eval_draw,
            eval_ret_mean: if self.eval_returns.is_empty() {
                0.0
            } else {
                self.eval_returns.iter().sum::<f64>() / self.eval_returns.len() as f64
            },
            debug_non_hold: self.non_hold,
            debug_non_zero_pos: self.non_zero_pos,
            debug_mean_abs_pnl: if self.pnl_steps > 0 {
                self.abs_pnl_sum / self.pnl_steps as f64
            } else {
                0.0
            },
            debug_buy: self.act_buy,
            debug_sell: self.act_sell,
            debug_hold: self.act_hold,
            debug_revert: self.act_revert,
            debug_session_violations: self.session_violations,
            debug_margin_violations: self.margin_violations,
            debug_position_violations: self.position_violations,
            debug_drawdown_penalty: self.drawdown_penalty_sum,
            debug_invalid_revert_penalty: self.invalid_revert_penalty_sum,
            debug_hold_duration_penalty: self.hold_duration_penalty_sum,
            debug_flat_hold_penalty: self.flat_hold_penalty_sum,
            debug_session_close_penalty: self.session_close_penalty_sum,
        }
    }
}

impl<B: Backend> BurnPolicy<B> {
    fn from_genome(
        genome: &[f32],
        input_dim: usize,
        hidden: usize,
        layers: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let mut parsed_layers = Vec::with_capacity(layers + 1);
        let mut offset = 0usize;
        let mut in_dim = input_dim;

        for _ in 0..layers {
            let out_dim = hidden;
            let weight_len = in_dim * out_dim;
            let bias_len = out_dim;
            let weight = Tensor::<B, 2>::from_data(
                TensorData::new(
                    genome[offset..offset + weight_len].to_vec(),
                    [out_dim, in_dim],
                ),
                device,
            );
            offset += weight_len;
            let bias = Tensor::<B, 1>::from_data(
                TensorData::new(genome[offset..offset + bias_len].to_vec(), [out_dim]),
                device,
            );
            offset += bias_len;
            parsed_layers.push(LinearLayer { weight, bias });
            in_dim = hidden;
        }

        let out_dim = 4usize;
        let weight_len = in_dim * out_dim;
        let bias_len = out_dim;
        let weight = Tensor::<B, 2>::from_data(
            TensorData::new(
                genome[offset..offset + weight_len].to_vec(),
                [out_dim, in_dim],
            ),
            device,
        );
        offset += weight_len;
        let bias = Tensor::<B, 1>::from_data(
            TensorData::new(genome[offset..offset + bias_len].to_vec(), [out_dim]),
            device,
        );
        offset += bias_len;
        parsed_layers.push(LinearLayer { weight, bias });

        if offset != genome.len() {
            bail!(
                "burn policy genome length mismatch: expected {} values, found {}",
                offset,
                genome.len()
            );
        }

        Ok(Self {
            layers: parsed_layers,
        })
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let last = self.layers.len().saturating_sub(1);
        let mut out = input;
        for (idx, layer) in self.layers.iter().enumerate() {
            out = out.matmul(layer.weight.clone().transpose())
                + layer.bias.clone().reshape([1, layer.bias.dims()[0]]);
            if idx != last {
                out = activation::tanh(out);
            }
        }
        out
    }
}

pub fn resolve_device(requested: ComputeRuntime) -> Result<ExecutionTarget> {
    match requested {
        ComputeRuntime::Cpu => Ok(ExecutionTarget::Cpu),
        ComputeRuntime::Auto => auto_device(),
        ComputeRuntime::Cuda => explicit_cuda_device(),
        ComputeRuntime::Mps => explicit_mlx_device(),
    }
}

pub fn print_device(device: ExecutionTarget) {
    match device {
        ExecutionTarget::Cpu => {
            if use_legacy_ndarray_cpu() {
                println!("info: burn backend using cpu (ndarray legacy)");
            } else {
                println!("info: burn backend using cpu (burn-cpu)");
            }
        }
        ExecutionTarget::Cuda(idx) => println!("info: burn backend using cuda:{idx}"),
        ExecutionTarget::Mps => println!("info: burn backend using apple gpu (burn-mlx)"),
    }
}

pub fn param_count(input_dim: usize, hidden: usize, layers: usize) -> Result<usize> {
    let mut count = 0usize;
    let mut in_dim = input_dim;
    for _ in 0..layers {
        count += in_dim * hidden + hidden;
        in_dim = hidden;
    }
    count += in_dim * 4 + 4;
    Ok(count)
}

pub fn evaluate_candidate(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    _capture_history: bool,
) -> Result<CandidateResult> {
    match cfg.device {
        ExecutionTarget::Cpu => {
            if use_legacy_ndarray_cpu() {
                #[cfg(feature = "backend-burn-ndarray")]
                {
                    let device = <LegacyCpuBackend as Backend>::Device::default();
                    return evaluate_candidate_inner::<LegacyCpuBackend>(
                        genome, data, windows, cfg, &device, None,
                    );
                }
            }
            let device = CpuDevice::default();
            evaluate_candidate_inner::<CpuBackend>(genome, data, windows, cfg, &device, None)
        }
        #[cfg(feature = "backend-burn-cuda")]
        ExecutionTarget::Cuda(_) => {
            let device = CudaDevice::default();
            evaluate_candidate_inner::<CudaBackend>(genome, data, windows, cfg, &device, None)
        }
        #[cfg(not(feature = "backend-burn-cuda"))]
        ExecutionTarget::Cuda(_) => bail!(
            "burn cuda support is not compiled into this build; re-run with the 'backend-burn-cuda' Cargo feature"
        ),
        #[cfg(feature = "backend-burn-mlx")]
        ExecutionTarget::Mps => {
            let device = MlxDevice::default();
            evaluate_candidate_inner::<MlxBackend>(genome, data, windows, cfg, &device, None)
        }
        #[cfg(not(feature = "backend-burn-mlx"))]
        ExecutionTarget::Mps => bail!(
            "burn-mlx support is not compiled into this build; re-run with the 'backend-burn-mlx' Cargo feature"
        ),
    }
}

pub fn evaluate_candidate_with_history(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
) -> Result<(CandidateResult, Vec<BehaviorRow>)> {
    let mut history = Vec::new();
    let metrics = match cfg.device {
        ExecutionTarget::Cpu => {
            if use_legacy_ndarray_cpu() {
                #[cfg(feature = "backend-burn-ndarray")]
                {
                    let device = <LegacyCpuBackend as Backend>::Device::default();
                    let metrics = evaluate_candidate_inner::<LegacyCpuBackend>(
                        genome,
                        data,
                        windows,
                        cfg,
                        &device,
                        Some(&mut history),
                    )?;
                    return Ok((metrics, history));
                }
            }
            let device = CpuDevice::default();
            evaluate_candidate_inner::<CpuBackend>(
                genome,
                data,
                windows,
                cfg,
                &device,
                Some(&mut history),
            )
        }
        #[cfg(feature = "backend-burn-cuda")]
        ExecutionTarget::Cuda(_) => {
            let device = CudaDevice::default();
            evaluate_candidate_inner::<CudaBackend>(
                genome,
                data,
                windows,
                cfg,
                &device,
                Some(&mut history),
            )
        }
        #[cfg(not(feature = "backend-burn-cuda"))]
        ExecutionTarget::Cuda(_) => bail!(
            "burn cuda support is not compiled into this build; re-run with the 'backend-burn-cuda' Cargo feature"
        ),
        #[cfg(feature = "backend-burn-mlx")]
        ExecutionTarget::Mps => {
            let device = MlxDevice::default();
            evaluate_candidate_inner::<MlxBackend>(
                genome,
                data,
                windows,
                cfg,
                &device,
                Some(&mut history),
            )
        }
        #[cfg(not(feature = "backend-burn-mlx"))]
        ExecutionTarget::Mps => bail!(
            "burn-mlx support is not compiled into this build; re-run with the 'backend-burn-mlx' Cargo feature"
        ),
    }?;
    Ok((metrics, history))
}

pub fn evaluate_candidates_batch(
    genomes: &[Vec<f32>],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    _capture_history: bool,
) -> Result<Vec<CandidateResult>> {
    genomes
        .iter()
        .map(|genome| evaluate_candidate(genome, data, windows, cfg, false))
        .collect()
}

pub fn save_policy(
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    _device: ExecutionTarget,
    genome: &[f32],
    path: &Path,
) -> Result<()> {
    crate::portable::save_policy_json(obs_dim, hidden, layers, genome, path)
}

fn evaluate_candidate_inner<B: Backend>(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    device: &B::Device,
    mut history: Option<&mut Vec<BehaviorRow>>,
) -> Result<CandidateResult> {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};

    let policy =
        BurnPolicy::<B>::from_genome(genome, data.obs_dim, cfg.hidden, cfg.layers, device)?;
    let mut rng = rand::thread_rng();

    let env_cfg = EnvConfig {
        max_position: cfg.max_position,
        margin_mode: cfg.margin_mode,
        contract_multiplier: cfg.contract_multiplier,
        margin_per_contract: cfg.margin_per_contract,
        enforce_margin: !cfg.disable_margin,
        drawdown_penalty: cfg.drawdown_penalty,
        drawdown_penalty_growth: cfg.drawdown_penalty_growth,
        session_close_penalty: cfg.session_close_penalty,
        auto_close_minutes_before_close: cfg.auto_close_minutes_before_close,
        max_hold_bars_positive: cfg.max_hold_bars_positive,
        max_hold_bars_drawdown: cfg.max_hold_bars_drawdown,
        hold_duration_penalty: cfg.hold_duration_penalty,
        hold_duration_penalty_growth: cfg.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: cfg.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: cfg.hold_duration_penalty_negative_scale,
        invalid_revert_penalty: cfg.invalid_revert_penalty,
        flat_hold_penalty: cfg.flat_hold_penalty,
        invalid_revert_penalty_growth: cfg.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: cfg.flat_hold_penalty_growth,
        max_flat_hold_bars: cfg.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let mut stats = CandidateStats::new();

    for (window_idx, &(start, end)) in windows.iter().take(cfg.eval_windows).enumerate() {
        if end <= start + 1 {
            continue;
        }
        let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
        let mut position = 0;
        let mut equity = cfg.initial_balance;
        let mut pnl_buf = Vec::with_capacity(end - start - 1);
        let mut eq_curve = Vec::with_capacity(end - start - 1);
        let mut window_drawdown_penalty = 0.0f64;
        let mut window_invalid_revert_penalty = 0.0f64;
        let mut window_hold_duration_penalty = 0.0f64;
        let mut window_flat_hold_penalty = 0.0f64;
        let mut window_session_close_penalty = 0.0f64;
        let mut window_violation_penalty = 0.0f64;
        let mut step_idx = 0usize;

        for t in (start + 1)..end {
            let position_before = position;
            let equity_before = equity;
            let obs = build_observation(
                data,
                t,
                position,
                equity,
                env.state().unrealized_pnl,
                env.state().realized_pnl,
                cfg.initial_balance,
            );
            let action_idx = sample_action::<B>(&policy, device, &obs, &mut rng)?;

            let action = match action_idx {
                0 => {
                    stats.act_buy += 1;
                    Action::Buy
                }
                1 => {
                    stats.act_sell += 1;
                    Action::Sell
                }
                2 => {
                    stats.act_hold += 1;
                    Action::Hold
                }
                _ => {
                    stats.act_revert += 1;
                    Action::Revert
                }
            };

            let session_open = if cfg.ignore_session {
                true
            } else {
                data.session_open
                    .as_ref()
                    .and_then(|m| m.get(t))
                    .copied()
                    .unwrap_or(true)
            };
            let minutes_to_close = data
                .minutes_to_close
                .as_ref()
                .and_then(|m| m.get(t))
                .copied();
            let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);

            let (reward, info) = env.step(
                action,
                data.close[t],
                StepContext {
                    session_open,
                    margin_ok,
                    minutes_to_close,
                },
            );
            let action_label = match info.effective_action {
                Action::Buy => "buy",
                Action::Sell => "sell",
                Action::Hold => "hold",
                Action::Revert => "revert",
            };

            if info.session_closed_violation {
                stats.session_violations += 1;
                window_violation_penalty += VIOLATION_PENALTY;
            }
            if info.margin_call_violation {
                stats.margin_violations += 1;
                window_violation_penalty += VIOLATION_PENALTY;
            }
            if info.position_limit_violation {
                stats.position_violations += 1;
                window_violation_penalty += VIOLATION_PENALTY;
            }
            position = env.state().position;
            equity = env.state().cash + env.state().unrealized_pnl;
            if let Some(hist) = history.as_deref_mut() {
                let state = env.state();
                hist.push(BehaviorRow {
                    window_idx,
                    step: step_idx,
                    data_idx: t,
                    action_idx,
                    action: action_label.to_string(),
                    position_before,
                    position_after: state.position,
                    equity_before,
                    equity_after: equity,
                    cash: state.cash,
                    unrealized_pnl: state.unrealized_pnl,
                    realized_pnl: state.realized_pnl,
                    pnl_change: info.pnl_change,
                    realized_pnl_change: info.realized_pnl_change,
                    reward,
                    commission_paid: info.commission_paid,
                    slippage_paid: info.slippage_paid,
                    drawdown_penalty: info.drawdown_penalty,
                    session_close_penalty: info.session_close_penalty,
                    invalid_revert_penalty: info.invalid_revert_penalty,
                    hold_duration_penalty: info.hold_duration_penalty,
                    flat_hold_penalty: info.flat_hold_penalty,
                    auto_close_executed: info.auto_close_executed,
                    session_open,
                    margin_ok,
                    minutes_to_close,
                    session_closed_violation: info.session_closed_violation,
                    margin_call_violation: info.margin_call_violation,
                    position_limit_violation: info.position_limit_violation,
                });
            }
            pnl_buf.push(info.pnl_change);
            eq_curve.push(equity);
            window_drawdown_penalty += info.drawdown_penalty;
            window_invalid_revert_penalty += info.invalid_revert_penalty;
            window_hold_duration_penalty += info.hold_duration_penalty;
            window_flat_hold_penalty += info.flat_hold_penalty;
            window_session_close_penalty += info.session_close_penalty;
            if !matches!(action, Action::Hold) {
                stats.non_hold += 1;
            }
            if position != 0 {
                stats.non_zero_pos += 1;
            }
            stats.abs_pnl_sum += info.pnl_change.abs();
            stats.pnl_steps += 1;
            step_idx += 1;
        }

        let realized_pnl = env.state().realized_pnl;
        let total_pnl = env.state().cash + env.state().unrealized_pnl - cfg.initial_balance;
        let pnl_sum = realized_pnl
            - window_drawdown_penalty
            - window_invalid_revert_penalty
            - window_hold_duration_penalty
            - window_flat_hold_penalty
            - window_session_close_penalty
            - window_violation_penalty;
        stats.eval_pnls.push(pnl_sum);
        stats.eval_pnls_realized.push(realized_pnl);
        stats.eval_pnls_total.push(total_pnl);
        stats.drawdown_penalty_sum += window_drawdown_penalty;
        stats.invalid_revert_penalty_sum += window_invalid_revert_penalty;
        stats.hold_duration_penalty_sum += window_hold_duration_penalty;
        stats.flat_hold_penalty_sum += window_flat_hold_penalty;
        stats.session_close_penalty_sum += window_session_close_penalty;
        stats.violation_penalty_sum += window_violation_penalty;

        let mut prev_eq = cfg.initial_balance;
        for (idx, &pnl) in pnl_buf.iter().enumerate() {
            let eq = eq_curve.get(idx).copied().unwrap_or(prev_eq);
            let denom = if prev_eq.abs() < 1e-8 { 1e-8 } else { prev_eq };
            stats.eval_returns.push(pnl / denom);
            prev_eq = eq;
        }

        stats.eval_equity.push(eq_curve);
    }

    Ok(stats.finish(cfg))
}

fn sample_action<B: Backend>(
    policy: &BurnPolicy<B>,
    device: &B::Device,
    obs: &[f32],
    rng: &mut rand::rngs::ThreadRng,
) -> Result<i32> {
    let obs_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), device);
    let logits = policy.forward(obs_tensor);
    let values = logits
        .into_data()
        .to_vec::<f32>()
        .context("extract burn logits into host memory")?;
    Ok(sample_from_logits(&values, rng))
}

fn sample_from_logits(logits: &[f32], rng: &mut rand::rngs::ThreadRng) -> i32 {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let weights: Vec<f64> = logits
        .iter()
        .map(|logit| {
            let weight = (*logit - max_logit).exp() as f64;
            if weight.is_finite() && weight > 0.0 {
                weight
            } else {
                0.0
            }
        })
        .collect();

    if weights.iter().all(|weight| *weight <= 0.0) {
        return argmax_index(logits) as i32;
    }

    WeightedIndex::new(&weights)
        .map(|dist| dist.sample(rng) as i32)
        .unwrap_or_else(|_| argmax_index(logits) as i32)
}

fn argmax_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn use_legacy_ndarray_cpu() -> bool {
    #[cfg(feature = "backend-burn-ndarray")]
    {
        matches!(
            std::env::var("MIDAS_BURN_CPU_BACKEND")
                .ok()
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("ndarray") | Some("legacy")
        ) || matches!(
            std::env::var("MIDAS_BURN_NDARRAY").ok().as_deref(),
            Some("1")
        )
    }
    #[cfg(not(feature = "backend-burn-ndarray"))]
    {
        false
    }
}

fn auto_device() -> Result<ExecutionTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        return Ok(ExecutionTarget::Mps);
    }

    #[cfg(all(
        not(all(target_os = "macos", feature = "backend-burn-mlx")),
        feature = "backend-burn-cuda"
    ))]
    {
        return Ok(ExecutionTarget::Cuda(0));
    }

    #[cfg(all(
        not(all(target_os = "macos", feature = "backend-burn-mlx")),
        not(feature = "backend-burn-cuda")
    ))]
    {
        Ok(ExecutionTarget::Cpu)
    }
}

fn explicit_cuda_device() -> Result<ExecutionTarget> {
    #[cfg(feature = "backend-burn-cuda")]
    {
        return Ok(ExecutionTarget::Cuda(0));
    }

    #[cfg(not(feature = "backend-burn-cuda"))]
    {
        bail!(
            "burn cuda support is not compiled into this build; re-run with the 'backend-burn-cuda' Cargo feature"
        )
    }
}

fn explicit_mlx_device() -> Result<ExecutionTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        return Ok(ExecutionTarget::Mps);
    }

    #[cfg(not(all(target_os = "macos", feature = "backend-burn-mlx")))]
    {
        bail!(
            "burn-mlx support is only available on macOS builds with the 'backend-burn-mlx' Cargo feature"
        )
    }
}
