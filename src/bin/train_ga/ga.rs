use tch::{Device, Kind, Tensor, no_grad};

use crate::actions::{
    POLICY_ACTION_DIM, env_action_for_target, env_action_label, policy_action_label,
    policy_target_position,
};
use crate::config::CandidateConfig;
use crate::data::{DataSet, build_observation};
use crate::metrics::{candidate_fitness, compute_sortino, liquidation_cost, max_drawdown};
use crate::model::{build_batched_policy, build_mlp, load_params_from_vec};
use crate::types::{BehaviorRow, CandidateResult};
use midas_env::env::VIOLATION_PENALTY;

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

fn evaluate_candidate_internal(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    mut history: Option<&mut Vec<BehaviorRow>>,
) -> CandidateResult {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
    use tch::nn::Module;

    let device = cfg.device.as_tch_device();
    let vs = tch::nn::VarStore::new(device);
    let policy = build_mlp(
        &vs.root(),
        data.obs_dim as i64,
        cfg.hidden as i64,
        cfg.layers,
    );
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

    let obs_dim = data.obs_dim as i64;
    let mut obs_device = Tensor::zeros(&[1, obs_dim], (Kind::Float, device));

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
    let mut hold_duration_penalty_sum = 0.0f64;
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
            let obs_cpu = Tensor::f_from_slice(&obs)
                .expect("tensor from obs")
                .reshape(&[1, obs_dim]);
            obs_device.copy_(&obs_cpu);

            let action_idx = no_grad(|| {
                let logits = policy.forward(&obs_device);
                select_action_from_logits(&logits)
            });

            let policy_label = policy_action_label(action_idx);
            let target_position = policy_target_position(action_idx);
            let action = env_action_for_target(position_before, target_position);
            match action {
                Action::Buy => act_buy += 1,
                Action::Sell => act_sell += 1,
                Action::Hold => act_hold += 1,
                Action::Revert => act_revert += 1,
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
                session_violations += 1;
                window_violation_penalty += VIOLATION_PENALTY;
            }
            if info.margin_call_violation {
                margin_violations += 1;
                window_violation_penalty += VIOLATION_PENALTY;
            }
            if info.position_limit_violation {
                position_violations += 1;
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
                    action: policy_label.to_string(),
                    effective_action: env_action_label(info.effective_action).to_string(),
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
        let mut total_pnl = env.state().cash + env.state().unrealized_pnl - cfg.initial_balance;
        let exit_cost = liquidation_cost(
            env.state().position,
            env_cfg.commission_round_turn,
            env_cfg.slippage_per_contract,
        );
        if exit_cost > 0.0 {
            total_pnl -= exit_cost;
            let last_eq = eq_curve.last().copied().unwrap_or(cfg.initial_balance);
            eq_curve.push(last_eq - exit_cost);
            pnl_buf.push(-exit_cost);
        }
        let pnl_sum = total_pnl
            - window_drawdown_penalty
            - window_invalid_revert_penalty
            - window_hold_duration_penalty
            - window_flat_hold_penalty
            - window_session_close_penalty
            - window_violation_penalty;
        eval_pnls.push(pnl_sum);
        eval_pnls_realized.push(realized_pnl);
        eval_pnls_total.push(total_pnl);
        drawdown_penalty_sum += window_drawdown_penalty;
        invalid_revert_penalty_sum += window_invalid_revert_penalty;
        hold_duration_penalty_sum += window_hold_duration_penalty;
        flat_hold_penalty_sum += window_flat_hold_penalty;
        session_close_penalty_sum += window_session_close_penalty;

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
        debug_hold_duration_penalty: hold_duration_penalty_sum,
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
    let device = cfg.device.as_tch_device();
    let policy = build_batched_policy(genomes, data.obs_dim, cfg.hidden, cfg.layers, device);

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

    let mut stats: Vec<CandidateStats> = (0..batch).map(|_| CandidateStats::new()).collect();
    let obs_dim = data.obs_dim as i64;
    let batch_dim = batch as i64;
    let mut obs_device = Tensor::zeros(&[batch_dim, obs_dim], (Kind::Float, device));

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
        let mut window_invalid_revert_penalty = vec![0.0f64; batch];
        let mut window_hold_duration_penalty = vec![0.0f64; batch];
        let mut window_flat_hold_penalty = vec![0.0f64; batch];
        let mut window_session_close_penalty = vec![0.0f64; batch];
        let mut window_violation_penalty = vec![0.0f64; batch];

        for t in (start + 1)..end {
            let mut obs_batch: Vec<f32> = Vec::with_capacity(batch * data.obs_dim);
            for i in 0..batch {
                let env = &envs[i];
                let obs = build_observation(
                    data,
                    t,
                    positions[i],
                    env.state()
                        .step
                        .saturating_sub(env.state().position_entry_step),
                    env.state().flat_steps,
                    cfg.max_position,
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
            let action_indices = select_actions_from_batch_logits(&logits, batch);

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
                let action_idx = action_indices[i];
                let target_position = policy_target_position(action_idx);
                let action = env_action_for_target(positions[i], target_position);
                match action {
                    Action::Buy => stats[i].act_buy += 1,
                    Action::Sell => stats[i].act_sell += 1,
                    Action::Hold => stats[i].act_hold += 1,
                    Action::Revert => stats[i].act_revert += 1,
                }

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
                    window_violation_penalty[i] += VIOLATION_PENALTY;
                }
                if info.margin_call_violation {
                    stats[i].margin_violations += 1;
                    window_violation_penalty[i] += VIOLATION_PENALTY;
                }
                if info.position_limit_violation {
                    stats[i].position_violations += 1;
                    window_violation_penalty[i] += VIOLATION_PENALTY;
                }

                positions[i] = envs[i].state().position;
                equities[i] = envs[i].state().cash + envs[i].state().unrealized_pnl;
                pnl_bufs[i].push(info.pnl_change);
                eq_curves[i].push(equities[i]);
                window_drawdown_penalty[i] += info.drawdown_penalty;
                window_invalid_revert_penalty[i] += info.invalid_revert_penalty;
                window_hold_duration_penalty[i] += info.hold_duration_penalty;
                window_flat_hold_penalty[i] += info.flat_hold_penalty;
                window_session_close_penalty[i] += info.session_close_penalty;
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
            let mut total_pnl =
                envs[i].state().cash + envs[i].state().unrealized_pnl - cfg.initial_balance;
            let exit_cost = liquidation_cost(
                envs[i].state().position,
                env_cfg.commission_round_turn,
                env_cfg.slippage_per_contract,
            );
            if exit_cost > 0.0 {
                total_pnl -= exit_cost;
                let last_eq = eq_curves[i].last().copied().unwrap_or(cfg.initial_balance);
                eq_curves[i].push(last_eq - exit_cost);
                pnl_bufs[i].push(-exit_cost);
            }
            let pnl_sum = total_pnl
                - window_drawdown_penalty[i]
                - window_invalid_revert_penalty[i]
                - window_hold_duration_penalty[i]
                - window_flat_hold_penalty[i]
                - window_session_close_penalty[i]
                - window_violation_penalty[i];
            stats[i].eval_pnls.push(pnl_sum);
            stats[i].eval_pnls_realized.push(realized_pnl);
            stats[i].eval_pnls_total.push(total_pnl);
            stats[i].drawdown_penalty_sum += window_drawdown_penalty[i];
            stats[i].invalid_revert_penalty_sum += window_invalid_revert_penalty[i];
            stats[i].hold_duration_penalty_sum += window_hold_duration_penalty[i];
            stats[i].flat_hold_penalty_sum += window_flat_hold_penalty[i];
            stats[i].session_close_penalty_sum += window_session_close_penalty[i];
            stats[i].violation_penalty_sum += window_violation_penalty[i];

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

fn select_action_from_logits(logits: &Tensor) -> i32 {
    let logits_cpu = logits.squeeze_dim(0).to_device(Device::Cpu);
    let mut values = vec![0f32; logits_cpu.numel()];
    let len = values.len();
    logits_cpu.copy_data(&mut values, len);
    argmax_index(&values) as i32
}

fn select_actions_from_batch_logits(logits: &Tensor, batch: usize) -> Vec<i32> {
    let logits_cpu = logits.to_device(Device::Cpu);
    let mut values = vec![0f32; logits_cpu.numel()];
    let len = values.len();
    logits_cpu.copy_data(&mut values, len);

    values
        .chunks_exact(POLICY_ACTION_DIM)
        .take(batch)
        .map(|row| argmax_index(row) as i32)
        .collect()
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
