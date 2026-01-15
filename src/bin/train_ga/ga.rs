use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::{kind::Kind, no_grad, Device, Tensor};

use crate::data::{build_observation, DataSet};
use crate::metrics::{compute_sortino, max_drawdown};
use crate::model::{build_batched_policy, build_mlp, load_params_from_vec};
use midas_env::env::{MarginMode, VIOLATION_PENALTY};

#[derive(Clone)]
pub struct CandidateConfig {
    pub initial_balance: f64,
    pub max_position: i32,
    pub margin_mode: MarginMode,
    pub contract_multiplier: f64,
    pub margin_per_contract: f64,
    pub disable_margin: bool,
    pub w_pnl: f64,
    pub w_sortino: f64,
    pub w_mdd: f64,
    pub sortino_annualization: f64,
    pub hidden: usize,
    pub layers: usize,
    pub eval_windows: usize,
    pub device: tch::Device,
    pub ignore_session: bool,
    pub drawdown_penalty: f64,
    pub drawdown_penalty_growth: f64,
    pub session_close_penalty: f64,
    pub max_hold_bars_positive: usize,
    pub max_hold_bars_drawdown: usize,
    pub invalid_revert_penalty: f64,
    pub invalid_revert_penalty_growth: f64,
    pub flat_hold_penalty: f64,
    pub flat_hold_penalty_growth: f64,
    pub max_flat_hold_bars: usize,
}

#[derive(Clone)]
pub struct CandidateResult {
    pub fitness: f64,
    pub eval_pnl: f64,
    pub eval_pnl_realized: f64,
    pub eval_pnl_total: f64,
    pub eval_sortino: f64,
    pub eval_drawdown: f64,
    pub eval_ret_mean: f64,
    pub debug_non_hold: usize,
    pub debug_non_zero_pos: usize,
    pub debug_mean_abs_pnl: f64,
    pub debug_buy: usize,
    pub debug_sell: usize,
    pub debug_hold: usize,
    pub debug_revert: usize,
    pub debug_session_violations: usize,
    pub debug_margin_violations: usize,
    pub debug_position_violations: usize,
    pub debug_drawdown_penalty: f64,
    pub debug_invalid_revert_penalty: f64,
    pub debug_flat_hold_penalty: f64,
    pub debug_session_close_penalty: f64,
}

#[derive(Clone)]
pub struct BehaviorRow {
    pub window_idx: usize,
    pub step: usize,
    pub data_idx: usize,
    pub action_idx: i32,
    pub action: String,
    pub position_before: i32,
    pub position_after: i32,
    pub equity_before: f64,
    pub equity_after: f64,
    pub cash: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub pnl_change: f64,
    pub realized_pnl_change: f64,
    pub reward: f64,
    pub commission_paid: f64,
    pub slippage_paid: f64,
    pub drawdown_penalty: f64,
    pub session_close_penalty: f64,
    pub invalid_revert_penalty: f64,
    pub flat_hold_penalty: f64,
    pub session_open: bool,
    pub margin_ok: bool,
    pub minutes_to_close: Option<f64>,
    pub session_closed_violation: bool,
    pub margin_call_violation: bool,
    pub position_limit_violation: bool,
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
            flat_hold_penalty_sum: 0.0,
            session_close_penalty_sum: 0.0,
            violation_penalty_sum: 0.0,
        }
    }

    fn finish(self, cfg: &CandidateConfig) -> CandidateResult {
        let eval_sortino = compute_sortino(&self.eval_returns, cfg.sortino_annualization, 0.0, 50.0);
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
            debug_flat_hold_penalty: self.flat_hold_penalty_sum,
        debug_session_close_penalty: self.session_close_penalty_sum,
    }
}
}

fn evaluate_candidate_internal(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    mut history: Option<&mut Vec<BehaviorRow>>,
) -> CandidateResult {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
    use tch::nn::Module;

    let vs = tch::nn::VarStore::new(cfg.device);
    let policy = build_mlp(&vs.root(), data.obs_dim as i64, cfg.hidden as i64, cfg.layers);
    load_params_from_vec(&vs, genome);

    let env_cfg = EnvConfig {
        max_position: cfg.max_position,
        margin_mode: cfg.margin_mode,
        contract_multiplier: cfg.contract_multiplier,
        margin_per_contract: cfg.margin_per_contract,
        enforce_margin: !cfg.disable_margin,
        drawdown_penalty: cfg.drawdown_penalty,
        drawdown_penalty_growth: cfg.drawdown_penalty_growth,
        session_close_penalty: cfg.session_close_penalty,
        max_hold_bars_positive: cfg.max_hold_bars_positive,
        max_hold_bars_drawdown: cfg.max_hold_bars_drawdown,
        invalid_revert_penalty: cfg.invalid_revert_penalty,
        flat_hold_penalty: cfg.flat_hold_penalty,
        invalid_revert_penalty_growth: cfg.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: cfg.flat_hold_penalty_growth,
        max_flat_hold_bars: cfg.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let obs_dim = data.obs_dim as i64;
    let mut obs_device = Tensor::zeros(&[1, obs_dim], (Kind::Float, cfg.device));

    let mut eval_pnls = Vec::new();
    let mut eval_pnls_realized = Vec::new();
    let mut eval_pnls_total = Vec::new();
    let mut eval_returns = Vec::new();
    let mut eval_equity: Vec<Vec<f64>> = Vec::new();
    let mut non_hold = 0usize;
    let mut non_zero_pos = 0usize;
    let mut abs_pnl_sum = 0.0f64;
    let mut pnl_steps = 0usize;
    let mut act_buy = 0usize;
    let mut act_sell = 0usize;
    let mut act_hold = 0usize;
    let mut act_revert = 0usize;
    let mut session_violations = 0usize;
    let mut margin_violations = 0usize;
    let mut position_violations = 0usize;
    let mut drawdown_penalty_sum = 0.0f64;
    let mut invalid_revert_penalty_sum = 0.0f64;
    let mut violation_penalty_sum = 0.0f64;
    let mut flat_hold_penalty_sum = 0.0f64;
    let mut session_close_penalty_sum = 0.0f64;

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
            let obs_cpu = Tensor::f_from_slice(&obs)
                .expect("tensor from obs")
                .reshape(&[1, obs_dim]);
            obs_device.copy_(&obs_cpu);

            let action_idx = no_grad(|| {
                let logits = policy.forward(&obs_device);
                let probs = logits.softmax(-1, Kind::Float);
                let sample = probs.multinomial(1, true);
                sample.int64_value(&[0, 0]) as i32
            });

            let (action, action_label) = match action_idx {
                0 => {
                    act_buy += 1;
                    (Action::Buy, "buy")
                }
                1 => {
                    act_sell += 1;
                    (Action::Sell, "sell")
                }
                2 => {
                    act_hold += 1;
                    (Action::Hold, "hold")
                }
                _ => {
                    act_revert += 1;
                    (Action::Revert, "revert")
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
            if info.session_closed_violation {
                session_violations += 1;
                violation_penalty_sum += VIOLATION_PENALTY;
            }
            if info.margin_call_violation {
                margin_violations += 1;
                violation_penalty_sum += VIOLATION_PENALTY;
            }
            if info.position_limit_violation {
                position_violations += 1;
                violation_penalty_sum += VIOLATION_PENALTY;
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
                    flat_hold_penalty: info.flat_hold_penalty,
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
            invalid_revert_penalty_sum += info.invalid_revert_penalty;
            flat_hold_penalty_sum += info.flat_hold_penalty;
            session_close_penalty_sum += info.session_close_penalty;
            if !matches!(action, Action::Hold) {
                non_hold += 1;
            }
            if position != 0 {
                non_zero_pos += 1;
            }
            abs_pnl_sum += info.pnl_change.abs();
            pnl_steps += 1;
            step_idx += 1;
        }

        let realized_pnl = env.state().realized_pnl;
        let total_pnl = env.state().cash + env.state().unrealized_pnl - cfg.initial_balance;
        let pnl_sum = realized_pnl
            - window_drawdown_penalty
            - invalid_revert_penalty_sum
            - flat_hold_penalty_sum
            - session_close_penalty_sum
            - violation_penalty_sum;
        eval_pnls.push(pnl_sum);
        eval_pnls_realized.push(realized_pnl);
        eval_pnls_total.push(total_pnl);
        drawdown_penalty_sum += window_drawdown_penalty;

        let mut prev_eq = cfg.initial_balance;
        for (i, &pnl) in pnl_buf.iter().enumerate() {
            let eq = eq_curve.get(i).copied().unwrap_or(prev_eq);
            let denom = if prev_eq.abs() < 1e-8 { 1e-8 } else { prev_eq };
            eval_returns.push(pnl / denom);
            prev_eq = eq;
        }

        eval_equity.push(eq_curve);

    }

    let eval_sortino = compute_sortino(&eval_returns, cfg.sortino_annualization, 0.0, 50.0);
    let eval_draw = eval_equity
        .iter()
        .map(|eq| max_drawdown(eq))
        .fold(0.0_f64, |a, b| a.max(b));
    let eval_pnl = if eval_pnls.is_empty() {
        0.0
    } else {
        eval_pnls.iter().sum::<f64>() / eval_pnls.len() as f64
    };
    let eval_pnl_realized = if eval_pnls_realized.is_empty() {
        0.0
    } else {
        eval_pnls_realized.iter().sum::<f64>() / eval_pnls_realized.len() as f64
    };
    let eval_pnl_total = if eval_pnls_total.is_empty() {
        0.0
    } else {
        eval_pnls_total.iter().sum::<f64>() / eval_pnls_total.len() as f64
    };

    let fitness = cfg.w_pnl * eval_pnl + cfg.w_sortino * eval_sortino - cfg.w_mdd * eval_draw;

    CandidateResult {
        fitness,
        eval_pnl,
        eval_pnl_realized,
        eval_pnl_total,
        eval_sortino,
        eval_drawdown: eval_draw,
        eval_ret_mean: if eval_returns.is_empty() {
            0.0
        } else {
            eval_returns.iter().sum::<f64>() / eval_returns.len() as f64
        },
        debug_non_hold: non_hold,
        debug_non_zero_pos: non_zero_pos,
        debug_mean_abs_pnl: if pnl_steps > 0 {
            abs_pnl_sum / pnl_steps as f64
        } else {
            0.0
        },
        debug_buy: act_buy,
        debug_sell: act_sell,
        debug_hold: act_hold,
        debug_revert: act_revert,
        debug_session_violations: session_violations,
        debug_margin_violations: margin_violations,
        debug_position_violations: position_violations,
        debug_drawdown_penalty: drawdown_penalty_sum,
        debug_invalid_revert_penalty: invalid_revert_penalty_sum,
        debug_flat_hold_penalty: flat_hold_penalty_sum,
        debug_session_close_penalty: session_close_penalty_sum,
    }
}

pub fn evaluate_candidate(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    _capture_history: bool,
) -> CandidateResult {
    evaluate_candidate_internal(genome, data, windows, cfg, None)
}

pub fn evaluate_candidate_with_history(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
) -> (CandidateResult, Vec<BehaviorRow>) {
    let mut history = Vec::new();
    let metrics = evaluate_candidate_internal(genome, data, windows, cfg, Some(&mut history));
    (metrics, history)
}

pub fn evaluate_candidates_batch(
    genomes: &[Vec<f32>],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> Vec<CandidateResult> {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};

    if genomes.is_empty() {
        return Vec::new();
    }

    let batch = genomes.len();
    let policy = build_batched_policy(genomes, data.obs_dim, cfg.hidden, cfg.layers, cfg.device);

    let env_cfg = EnvConfig {
        max_position: cfg.max_position,
        margin_mode: cfg.margin_mode,
        contract_multiplier: cfg.contract_multiplier,
        margin_per_contract: cfg.margin_per_contract,
        enforce_margin: !cfg.disable_margin,
        drawdown_penalty: cfg.drawdown_penalty,
        drawdown_penalty_growth: cfg.drawdown_penalty_growth,
        session_close_penalty: cfg.session_close_penalty,
        max_hold_bars_positive: cfg.max_hold_bars_positive,
        max_hold_bars_drawdown: cfg.max_hold_bars_drawdown,
        invalid_revert_penalty: cfg.invalid_revert_penalty,
        flat_hold_penalty: cfg.flat_hold_penalty,
        invalid_revert_penalty_growth: cfg.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: cfg.flat_hold_penalty_growth,
        max_flat_hold_bars: cfg.max_flat_hold_bars,
        ..EnvConfig::default()
    };

    let mut stats: Vec<CandidateStats> = (0..batch).map(|_| CandidateStats::new()).collect();
    let obs_dim = data.obs_dim as i64;
    let batch_dim = batch as i64;
    let mut obs_device = Tensor::zeros(&[batch_dim, obs_dim], (Kind::Float, cfg.device));

    for &(start, end) in windows.iter().take(cfg.eval_windows) {
        if end <= start + 1 {
            continue;
        }

        let mut envs: Vec<TradingEnv> = (0..batch)
            .map(|_| TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone()))
            .collect();
        let mut positions = vec![0i32; batch];
        let mut equities = vec![cfg.initial_balance; batch];
        let mut pnl_bufs: Vec<Vec<f64>> = (0..batch)
            .map(|_| Vec::with_capacity(end - start - 1))
            .collect();
        let mut eq_curves: Vec<Vec<f64>> = (0..batch)
            .map(|_| Vec::with_capacity(end - start - 1))
            .collect();
        let mut window_drawdown_penalty = vec![0.0f64; batch];

        for t in (start + 1)..end {
            let mut obs_batch: Vec<f32> = Vec::with_capacity(batch * data.obs_dim);
            for i in 0..batch {
                let env = &envs[i];
                let obs = build_observation(
                    data,
                    t,
                    positions[i],
                    equities[i],
                    env.state().unrealized_pnl,
                    env.state().realized_pnl,
                    cfg.initial_balance,
                );
                obs_batch.extend_from_slice(&obs);
            }

            let obs_cpu = Tensor::f_from_slice(&obs_batch)
                .expect("tensor from obs batch")
                .reshape(&[batch_dim, obs_dim]);
            obs_device.copy_(&obs_cpu);

            let logits = no_grad(|| policy.forward(&obs_device));
            let probs = logits.softmax(-1, Kind::Float);
            let sample = probs.multinomial(1, true);
            let sample_cpu = sample.squeeze_dim(-1).to_device(Device::Cpu);

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

            for i in 0..batch {
                let action_idx = sample_cpu.int64_value(&[i as i64]) as i32;
                let action = match action_idx {
                    0 => {
                        stats[i].act_buy += 1;
                        Action::Buy
                    }
                    1 => {
                        stats[i].act_sell += 1;
                        Action::Sell
                    }
                    2 => {
                        stats[i].act_hold += 1;
                        Action::Hold
                    }
                    _ => {
                        stats[i].act_revert += 1;
                        Action::Revert
                    }
                };

                let (_reward, info) = envs[i].step(
                    action,
                    data.close[t],
                    StepContext {
                        session_open,
                        margin_ok,
                        minutes_to_close,
                    },
                );
                if info.session_closed_violation {
                    stats[i].session_violations += 1;
                    stats[i].violation_penalty_sum += VIOLATION_PENALTY;
                }
                if info.margin_call_violation {
                    stats[i].margin_violations += 1;
                    stats[i].violation_penalty_sum += VIOLATION_PENALTY;
                }
                if info.position_limit_violation {
                    stats[i].position_violations += 1;
                    stats[i].violation_penalty_sum += VIOLATION_PENALTY;
                }

                positions[i] = envs[i].state().position;
                equities[i] = envs[i].state().cash + envs[i].state().unrealized_pnl;
                pnl_bufs[i].push(info.pnl_change);
                eq_curves[i].push(equities[i]);
                window_drawdown_penalty[i] += info.drawdown_penalty;
                stats[i].invalid_revert_penalty_sum += info.invalid_revert_penalty;
                stats[i].flat_hold_penalty_sum += info.flat_hold_penalty;
                stats[i].session_close_penalty_sum += info.session_close_penalty;
                if !matches!(action, Action::Hold) {
                    stats[i].non_hold += 1;
                }
                if positions[i] != 0 {
                    stats[i].non_zero_pos += 1;
                }
                stats[i].abs_pnl_sum += info.pnl_change.abs();
                stats[i].pnl_steps += 1;
            }
        }

        for i in 0..batch {
            let realized_pnl = envs[i].state().realized_pnl;
            let total_pnl = envs[i].state().cash + envs[i].state().unrealized_pnl - cfg.initial_balance;
            let pnl_sum = realized_pnl
                - window_drawdown_penalty[i]
                - stats[i].invalid_revert_penalty_sum
                - stats[i].flat_hold_penalty_sum
                - stats[i].session_close_penalty_sum
                - stats[i].violation_penalty_sum;
            stats[i].eval_pnls.push(pnl_sum);
            stats[i].eval_pnls_realized.push(realized_pnl);
            stats[i].eval_pnls_total.push(total_pnl);
            stats[i].drawdown_penalty_sum += window_drawdown_penalty[i];

            let mut prev_eq = cfg.initial_balance;
            for (idx, pnl) in pnl_bufs[i].iter().enumerate() {
                let eq = eq_curves[i].get(idx).copied().unwrap_or(prev_eq);
                let denom = if prev_eq.abs() < 1e-8 { 1e-8 } else { prev_eq };
                stats[i].eval_returns.push(pnl / denom);
                prev_eq = eq;
            }

            stats[i].eval_equity.push(eq_curves[i].clone());

            if capture_history {
                let _ = capture_history;
            }
        }
    }

    stats.into_iter().map(|s| s.finish(cfg)).collect()
}

pub fn crossover(a: &[f32], b: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if rng.gen_bool(0.5) {
            out.push(a[i]);
        } else {
            out.push(b[i]);
        }
    }
    out
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    generation: usize,
    pop: Vec<Vec<f32>>,
}

fn parse_generation_from_path(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_string_lossy();
    let marker = "checkpoint_gen";
    let start = stem.find(marker)? + marker.len();
    let digits: String = stem[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub fn load_checkpoint(path: &Path) -> Result<(usize, Vec<Vec<f32>>)> {
    let data = std::fs::read(path)?;
    if let Ok(ckpt) = bincode::deserialize::<Checkpoint>(&data) {
        return Ok((ckpt.generation.saturating_add(1), ckpt.pop));
    }
    let pop: Vec<Vec<f32>> = bincode::deserialize(&data)?;
    let start_gen = parse_generation_from_path(path).unwrap_or(0).saturating_add(1);
    Ok((start_gen, pop))
}

pub fn save_checkpoint(path: &Path, generation: usize, pop: &[Vec<f32>]) -> Result<()> {
    let data = bincode::serialize(&Checkpoint {
        generation,
        pop: pop.to_vec(),
    })?;
    std::fs::write(path, data)?;
    Ok(())
}
