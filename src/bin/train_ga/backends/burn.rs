use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
use midas_env::ml::ComputeRuntime;
use std::path::Path;

#[cfg(feature = "backend-burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "backend-burn-mlx")]
use burn_mlx::{Mlx, MlxDevice};
use burn_ndarray::NdArray;

use crate::actions::{
    POLICY_ACTION_DIM, env_action_for_target, env_action_label, policy_action_label,
    policy_target_position,
};
use crate::config::{CandidateConfig, ExecutionTarget, evaluation_window_count};
use crate::data::{DataSet, build_observation};
use crate::metrics::{
    candidate_fitness, compute_sortino, liquidation_cost_components, max_drawdown,
};
use crate::types::{BehaviorRow, CandidateResult, StepAccounting};
use midas_env::env::VIOLATION_PENALTY;

// Burn's CPU CubeCL matmul backend is not reliable for the 1 x N inference
// matrices used by this step-wise evaluator on every host. NdArray is still a
// Burn backend, and gives the CPU path deterministic small-matrix behavior.
type CpuBackend = NdArray<f32>;
#[cfg(feature = "backend-burn-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "backend-burn-mlx")]
type MlxBackend = Mlx<f32>;

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
    eval_gross_realized_pnls: Vec<f64>,
    eval_execution_costs: Vec<f64>,
    terminal_liquidation_costs: Vec<f64>,
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
            eval_gross_realized_pnls: Vec::new(),
            eval_execution_costs: Vec::new(),
            terminal_liquidation_costs: Vec::new(),
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
        let eval_gross_realized_pnl = if self.eval_gross_realized_pnls.is_empty() {
            0.0
        } else {
            self.eval_gross_realized_pnls.iter().sum::<f64>()
                / self.eval_gross_realized_pnls.len() as f64
        };
        let eval_execution_costs = if self.eval_execution_costs.is_empty() {
            0.0
        } else {
            self.eval_execution_costs.iter().sum::<f64>() / self.eval_execution_costs.len() as f64
        };
        let eval_shaping_penalties =
            eval_gross_realized_pnl - eval_execution_costs - eval_pnl_realized;
        let terminal_liquidation_cost = if self.terminal_liquidation_costs.is_empty() {
            0.0
        } else {
            self.terminal_liquidation_costs.iter().sum::<f64>()
                / self.terminal_liquidation_costs.len() as f64
        };

        let fitness = candidate_fitness(
            eval_pnl,
            eval_sortino,
            eval_draw,
            cfg.w_pnl,
            cfg.w_sortino,
            cfg.w_mdd,
        );

        CandidateResult {
            fitness,
            eval_pnl,
            eval_pnl_realized,
            eval_pnl_total,
            eval_gross_realized_pnl,
            eval_execution_costs,
            eval_shaping_penalties,
            terminal_liquidation_cost,
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

        let out_dim = POLICY_ACTION_DIM;
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
            println!("info: burn backend using cpu (burn-ndarray)");
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
    count += in_dim * POLICY_ACTION_DIM + POLICY_ACTION_DIM;
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
            let device = <CpuBackend as Backend>::Device::default();
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
            let device = <CpuBackend as Backend>::Device::default();
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
        min_hold_bars: cfg.min_hold_bars,
        early_exit_penalty: cfg.early_exit_penalty,
        early_flip_penalty: cfg.early_flip_penalty,
        invalid_revert_penalty: cfg.invalid_revert_penalty,
        flat_hold_penalty: cfg.flat_hold_penalty,
        invalid_revert_penalty_growth: cfg.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: cfg.flat_hold_penalty_growth,
        max_flat_hold_bars: cfg.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let mut stats = CandidateStats::new();

    let eval_count = evaluation_window_count(windows.len(), cfg.eval_windows);
    for (window_idx, &(start, end)) in windows.iter().take(eval_count).enumerate() {
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
        let mut realized_pnl = 0.0f64;
        let mut gross_realized_pnl = 0.0f64;
        let mut execution_costs = 0.0f64;
        let mut step_idx = 0usize;

        for t in (start + 1)..end {
            let position_before = position;
            let equity_before = equity;
            let obs = build_observation(
                data,
                t,
                position,
                env.state()
                    .step
                    .saturating_sub(env.state().position_entry_step),
                env.state().flat_steps,
                cfg.max_position,
                equity,
                env.state().unrealized_pnl,
                env.state().realized_pnl,
                cfg.initial_balance,
            );
            let action_idx = select_action::<B>(&policy, device, &obs)?;
            let policy_label = policy_action_label(action_idx);
            let action = env_action_for_target(position_before, policy_target_position(action_idx));
            match action {
                Action::Buy => stats.act_buy += 1,
                Action::Sell => stats.act_sell += 1,
                Action::Hold => stats.act_hold += 1,
                Action::Revert => stats.act_revert += 1,
            }

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
            let accounting = StepAccounting::from_transition(equity_before, equity, &info);
            if let Some(hist) = history.as_deref_mut() {
                let state = env.state();
                hist.push(BehaviorRow {
                    window_idx,
                    step: step_idx,
                    data_idx: t,
                    action_idx,
                    action: policy_label.to_string(),
                    effective_action: env_action_label(info.effective_action).to_string(),
                    position_before,
                    position_after: state.position,
                    equity_before,
                    equity_after: equity,
                    net_equity_delta: accounting.net_equity_delta,
                    cash: state.cash,
                    unrealized_pnl: state.unrealized_pnl,
                    gross_realized_pnl: state.realized_pnl,
                    gross_mark_to_market_pnl_change: info.pnl_change,
                    gross_realized_pnl_change: info.realized_pnl_change,
                    net_realized_pnl_change: info.realized_pnl_change
                        - info.commission_paid
                        - info.slippage_paid
                        - accounting.penalty_total,
                    reward,
                    commission_paid: info.commission_paid,
                    slippage_paid: info.slippage_paid,
                    drawdown_penalty: info.drawdown_penalty,
                    session_close_penalty: info.session_close_penalty,
                    early_exit_penalty: info.early_exit_penalty,
                    early_flip_penalty: info.early_flip_penalty,
                    invalid_revert_penalty: info.invalid_revert_penalty,
                    hold_duration_penalty: info.hold_duration_penalty,
                    flat_hold_penalty: info.flat_hold_penalty,
                    violation_penalty: accounting.violation_penalty,
                    auto_close_executed: info.auto_close_executed,
                    session_open,
                    margin_ok,
                    minutes_to_close,
                    session_closed_violation: info.session_closed_violation,
                    margin_call_violation: info.margin_call_violation,
                    position_limit_violation: info.position_limit_violation,
                    terminal_liquidation: false,
                    terminal_liquidation_cost: 0.0,
                });
            }
            pnl_buf.push(accounting.net_equity_delta);
            let previous_net_equity = eq_curve.last().copied().unwrap_or(cfg.initial_balance);
            eq_curve.push(previous_net_equity + accounting.net_equity_delta);
            window_drawdown_penalty += info.drawdown_penalty;
            window_invalid_revert_penalty += info.invalid_revert_penalty;
            window_hold_duration_penalty += info.hold_duration_penalty;
            window_flat_hold_penalty += info.flat_hold_penalty;
            window_session_close_penalty += info.session_close_penalty;
            realized_pnl += info.realized_pnl_change
                - info.commission_paid
                - info.slippage_paid
                - accounting.penalty_total;
            gross_realized_pnl += info.realized_pnl_change;
            execution_costs += info.commission_paid + info.slippage_paid;
            if !matches!(action, Action::Hold) {
                stats.non_hold += 1;
            }
            if position != 0 {
                stats.non_zero_pos += 1;
            }
            stats.abs_pnl_sum += accounting.net_equity_delta.abs();
            stats.pnl_steps += 1;
            step_idx += 1;
        }

        let (exit_commission, exit_slippage) = liquidation_cost_components(
            env.state().position,
            env_cfg.commission_round_turn,
            env_cfg.slippage_per_contract,
        );
        let exit_cost = exit_commission + exit_slippage;
        let terminal_position = env.state().position;
        if terminal_position != 0 {
            if let Some(hist) = history.as_deref_mut() {
                let terminal_data_idx = end.saturating_sub(1);
                let terminal_session_open = if cfg.ignore_session {
                    true
                } else {
                    data.session_open
                        .as_ref()
                        .and_then(|values| values.get(terminal_data_idx))
                        .copied()
                        .unwrap_or(true)
                };
                let terminal_minutes_to_close = data
                    .minutes_to_close
                    .as_ref()
                    .and_then(|values| values.get(terminal_data_idx))
                    .copied();
                let terminal_margin_ok = *data.margin_ok.get(terminal_data_idx).unwrap_or(&true);
                let state = env.state();
                hist.push(BehaviorRow::terminal_liquidation(
                    window_idx,
                    step_idx,
                    terminal_data_idx,
                    state.position,
                    equity,
                    state.unrealized_pnl,
                    state.realized_pnl,
                    exit_commission,
                    exit_slippage,
                    terminal_session_open,
                    terminal_margin_ok,
                    terminal_minutes_to_close,
                ));
            }
            let last_eq = eq_curve.last().copied().unwrap_or(cfg.initial_balance);
            eq_curve.push(last_eq - exit_cost);
            pnl_buf.push(-exit_cost);
            // The account-equity curve already contains the final
            // mark-to-market. Transfer it only into the realized diagnostic;
            // do not add it to equity a second time.
            realized_pnl += env.state().unrealized_pnl - exit_cost;
            gross_realized_pnl += env.state().unrealized_pnl;
            execution_costs += exit_cost;
            stats.abs_pnl_sum += exit_cost;
            stats.pnl_steps += 1;
        }
        stats.terminal_liquidation_costs.push(exit_cost);
        let net_pnl = pnl_buf.iter().sum::<f64>();
        stats.eval_pnls.push(net_pnl);
        stats.eval_pnls_realized.push(realized_pnl);
        stats.eval_pnls_total.push(net_pnl);
        stats.eval_gross_realized_pnls.push(gross_realized_pnl);
        stats.eval_execution_costs.push(execution_costs);
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

    let result = stats.finish(cfg);
    if let Some(rows) = history.as_deref() {
        let diagnostics = result.diagnostics(rows);
        debug_assert!(diagnostics.penalties.reconciliation_error.abs() < 1e-8);
        debug_assert!(diagnostics.candidate_reconciliation_error.abs() < 1e-8);
        debug_assert!(diagnostics.terminal_liquidation_reconciliation_error.abs() < 1e-8);
    }
    Ok(result)
}

fn select_action<B: Backend>(
    policy: &BurnPolicy<B>,
    device: &B::Device,
    obs: &[f32],
) -> Result<i32> {
    let obs_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), device);
    let logits = policy.forward(obs_tensor);
    let values = logits
        .into_data()
        .to_vec::<f32>()
        .context("extract burn logits into host memory")?;
    Ok(argmax_index(&values) as i32)
}

fn argmax_index(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;

    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::evaluate_candidate_with_history;
    use crate::config::{ExecutionTarget, test_candidate_config};
    use crate::data::DataSet;

    #[test]
    fn open_terminal_position_is_realized_once_and_emits_terminal_row() {
        let data = DataSet::synthetic_for_test(&[100.0, 101.0, 103.0]);
        let cfg = test_candidate_config(ExecutionTarget::Cpu);

        let mut genome = vec![0.0; data.obs_dim * 3];
        genome.extend([0.0, 0.0, 1.0]);

        let (metrics, rows) =
            evaluate_candidate_with_history(&genome, &data, &[(0, data.close.len())], &cfg)
                .expect("Burn evaluator should handle a terminal open position");

        let terminal_rows: Vec<_> = rows.iter().filter(|row| row.terminal_liquidation).collect();
        assert_eq!(terminal_rows.len(), 1);
        let terminal = terminal_rows[0];
        assert_ne!(terminal.position_before, 0);
        assert_eq!(terminal.position_after, 0);
        assert!((terminal.net_equity_delta + terminal.terminal_liquidation_cost).abs() < 1e-10);
        let diagnostics = metrics.diagnostics(&rows);
        assert!(diagnostics.candidate_reconciliation_error.abs() < 1e-10);
        assert!(diagnostics.terminal_liquidation_reconciliation_error.abs() < 1e-10);
        assert!((metrics.eval_pnl_realized - metrics.eval_pnl_total).abs() < 1e-10);
        assert!((metrics.eval_pnl - metrics.eval_pnl_total).abs() < 1e-10);
        assert!(metrics.eval_gross_realized_pnl.is_finite());
        assert!(metrics.eval_execution_costs >= metrics.terminal_liquidation_cost);
        assert!(
            (metrics.eval_gross_realized_pnl
                - metrics.eval_execution_costs
                - metrics.eval_shaping_penalties
                - metrics.eval_pnl_realized)
                .abs()
                < 1e-10
        );
        assert!(
            (metrics.terminal_liquidation_cost - terminal.terminal_liquidation_cost).abs() < 1e-10
        );
    }
}

fn auto_device() -> Result<ExecutionTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        if mlx_device_available() {
            return Ok(ExecutionTarget::Mps);
        }
        eprintln!(
            "warn: Burn MLX is compiled in, but the Metal device probe failed; falling back to CPU"
        );
    }

    #[cfg(feature = "backend-burn-cuda")]
    {
        if cuda_device_available() {
            return Ok(ExecutionTarget::Cuda(0));
        }
    }

    Ok(ExecutionTarget::Cpu)
}

fn explicit_cuda_device() -> Result<ExecutionTarget> {
    #[cfg(feature = "backend-burn-cuda")]
    {
        if cuda_device_available() {
            return Ok(ExecutionTarget::Cuda(0));
        }
        bail!("burn CUDA support is compiled in, but no usable CUDA device is available")
    }

    #[cfg(not(feature = "backend-burn-cuda"))]
    {
        bail!(
            "burn cuda support is not compiled into this build; re-run with the 'backend-burn-cuda' Cargo feature"
        )
    }
}

#[cfg(feature = "backend-burn-cuda")]
fn cuda_device_available() -> bool {
    use burn::prelude::DeviceOps;

    CudaDevice::device_count(0) > 0
}

fn explicit_mlx_device() -> Result<ExecutionTarget> {
    #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
    {
        if mlx_device_available() {
            return Ok(ExecutionTarget::Mps);
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
