use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
use rand::rngs::StdRng;

use super::model::Mlp;
use super::util::{argmax_index, sample_from_probs};
use crate::data::{DataSet, build_observation};
use crate::metrics::{compute_sortino, max_drawdown};

#[derive(Debug, Clone, Copy)]
pub(crate) struct RolloutMetrics {
    pub(crate) ret_mean: f64,
    pub(crate) pnl: f64,
    pub(crate) realized_pnl: f64,
    pub(crate) sortino: f64,
    pub(crate) drawdown: f64,
    pub(crate) commission: f64,
    pub(crate) slippage: f64,
    pub(crate) buy_frac: f64,
    pub(crate) sell_frac: f64,
    pub(crate) hold_frac: f64,
    pub(crate) revert_frac: f64,
    pub(crate) mean_max_prob: f64,
    pub(crate) entries: f64,
    pub(crate) exits: f64,
    pub(crate) flips: f64,
    pub(crate) avg_hold: f64,
}

pub(crate) struct RolloutBatch {
    pub(crate) obs: Tensor,
    pub(crate) actions: Tensor,
    pub(crate) logp: Tensor,
    pub(crate) adv: Vec<f32>,
    pub(crate) ret: Tensor,
    pub(crate) rewards: Vec<f64>,
    pub(crate) pnl: Vec<f64>,
    pub(crate) returns: Vec<f64>,
    pub(crate) equity: Vec<f64>,
    pub(crate) realized_pnl: f64,
    pub(crate) commission: f64,
    pub(crate) slippage: f64,
    pub(crate) action_counts: [usize; 4],
    pub(crate) max_prob_sum: f64,
    pub(crate) entries: usize,
    pub(crate) exits: usize,
    pub(crate) flips: usize,
    pub(crate) total_trade_bars: usize,
}

pub(crate) struct GrpoRollout {
    pub(crate) obs: Tensor,
    pub(crate) actions: Tensor,
    pub(crate) logp: Tensor,
    pub(crate) reward: f64,
    pub(crate) pnl: f64,
    pub(crate) returns: Vec<f64>,
    pub(crate) equity: Vec<f64>,
    pub(crate) realized_pnl: f64,
    pub(crate) commission: f64,
    pub(crate) slippage: f64,
    pub(crate) action_counts: [usize; 4],
    pub(crate) max_prob_sum: f64,
    pub(crate) entries: usize,
    pub(crate) exits: usize,
    pub(crate) flips: usize,
    pub(crate) total_trade_bars: usize,
}

pub(crate) struct GrpoGroup {
    pub(crate) rollouts: Vec<GrpoRollout>,
    pub(crate) mean_reward: f64,
    pub(crate) std_reward: f64,
}

pub(crate) struct RolloutConfig {
    pub(crate) gamma: f64,
    pub(crate) lam: f64,
    pub(crate) initial_balance: f64,
    pub(crate) ignore_session: bool,
}

pub(crate) fn rollout(
    data: &DataSet,
    window: (usize, usize),
    policy: &Mlp,
    value: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
) -> Result<RolloutBatch> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let obs_dim = data.obs_dim;

    let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = cfg.initial_balance;
    let mut prev_equity = cfg.initial_balance;

    let mut obs_buf = Vec::with_capacity(steps * obs_dim);
    let mut act_buf = Vec::with_capacity(steps);
    let mut logp_buf = Vec::with_capacity(steps);
    let mut val_buf = Vec::with_capacity(steps);
    let mut rew_buf = Vec::with_capacity(steps);
    let mut pnl_buf = Vec::with_capacity(steps);
    let mut ret_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);
    let mut realized_pnl = 0.0f64;
    let mut commission = 0.0f64;
    let mut slippage = 0.0f64;
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

    for t in (start + 1)..end {
        let obs = build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            cfg.initial_balance,
        );
        obs_buf.extend_from_slice(&obs);
        let obs_tensor = Tensor::from_vec(obs, (1, obs_dim), device)?;

        let logits = policy.forward(&obs_tensor, train)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = if greedy {
            argmax_index(&probs) as u32
        } else {
            sample_from_probs(&probs, rng) as u32
        };
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();
        let max_prob = probs
            .iter()
            .copied()
            .fold(0.0f32, |acc, value| acc.max(value)) as f64;
        let value_val = value
            .forward(&obs_tensor, train)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()?;

        let position_before = position;
        let (reward, info) = env.step(
            action_from_index(action_idx),
            data.close[t],
            step_context(data, cfg, t),
        );
        position = env.state().position;
        equity = env.state().cash + env.state().unrealized_pnl;
        let position_after = position;
        let step_idx = t.saturating_sub(start + 1);

        act_buf.push(action_idx);
        logp_buf.push(logp_val);
        val_buf.push(value_val);
        rew_buf.push(reward);
        pnl_buf.push(info.pnl_change);
        let denom = if prev_equity.abs() < 1e-8 {
            1e-8
        } else {
            prev_equity
        };
        ret_series.push(info.pnl_change / denom);
        equity_curve.push(equity);
        prev_equity = equity;
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += max_prob;
        if let Some(count) = action_counts.get_mut(action_idx as usize) {
            *count += 1;
        }
        if position_before == 0 && position_after != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position_after == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before.signum() != position_after.signum()
            && position_before != 0
            && position_after != 0
        {
            flips += 1;
        }
    }

    let (adv_buf, ret_buf) = compute_gae(&rew_buf, &val_buf, cfg.gamma, cfg.lam);

    Ok(RolloutBatch {
        obs: Tensor::from_vec(obs_buf, (steps, obs_dim), device)?,
        actions: Tensor::from_vec(act_buf, steps, device)?,
        logp: Tensor::from_vec(logp_buf, steps, device)?,
        adv: adv_buf,
        ret: Tensor::from_vec(ret_buf, steps, device)?,
        rewards: rew_buf,
        pnl: pnl_buf,
        returns: ret_series,
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

pub(crate) fn summarize_batch(batch: &RolloutBatch, annualization: f64) -> RolloutMetrics {
    let ret_mean = if batch.returns.is_empty() {
        0.0
    } else {
        batch.returns.iter().sum::<f64>() / batch.returns.len() as f64
    };
    let pnl_total = batch.pnl.iter().sum::<f64>();
    let sortino = compute_sortino(&batch.returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&batch.equity);
    let _ = batch.rewards.len();
    let steps = batch.returns.len().max(1) as f64;
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        realized_pnl: batch.realized_pnl,
        sortino,
        drawdown,
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

pub(crate) fn rollout_group(
    data: &DataSet,
    windows: &[(usize, usize)],
    policy: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    group_size: usize,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
) -> Result<GrpoGroup> {
    let mut rollouts = Vec::with_capacity(group_size);
    for i in 0..group_size {
        let window = windows[i % windows.len()];
        rollouts.push(rollout_single(
            data, window, policy, env_cfg, cfg, train, device, rng, greedy,
        )?);
    }

    let rewards: Vec<f64> = rollouts.iter().map(|r| r.reward).collect();
    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards
        .iter()
        .map(|r| (r - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    let std_reward = variance.sqrt().max(1e-8);

    Ok(GrpoGroup {
        rollouts,
        mean_reward,
        std_reward,
    })
}

pub(crate) fn compute_grpo_advantages(group: &GrpoGroup) -> Vec<f64> {
    group
        .rollouts
        .iter()
        .map(|rollout| (rollout.reward - group.mean_reward) / group.std_reward)
        .collect()
}

pub(crate) fn summarize_group(group: &GrpoGroup, annualization: f64) -> RolloutMetrics {
    let all_returns: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|rollout| rollout.returns.iter().copied())
        .collect();
    let ret_mean = if all_returns.is_empty() {
        0.0
    } else {
        all_returns.iter().sum::<f64>() / all_returns.len() as f64
    };
    let pnl_total = group
        .rollouts
        .iter()
        .map(|rollout| rollout.pnl)
        .sum::<f64>();
    let realized_pnl = group
        .rollouts
        .iter()
        .map(|rollout| rollout.realized_pnl)
        .sum::<f64>();
    let commission = group
        .rollouts
        .iter()
        .map(|rollout| rollout.commission)
        .sum::<f64>();
    let slippage = group
        .rollouts
        .iter()
        .map(|rollout| rollout.slippage)
        .sum::<f64>();
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut steps = 0usize;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut total_trade_bars = 0usize;
    for rollout in &group.rollouts {
        for (idx, count) in rollout.action_counts.iter().enumerate() {
            action_counts[idx] += *count;
            steps += *count;
        }
        max_prob_sum += rollout.max_prob_sum;
        entries += rollout.entries;
        exits += rollout.exits;
        flips += rollout.flips;
        total_trade_bars += rollout.total_trade_bars;
    }
    let all_equity: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|rollout| rollout.equity.iter().copied())
        .collect();
    let sortino = compute_sortino(&all_returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&all_equity);
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        realized_pnl,
        sortino,
        drawdown,
        commission,
        slippage,
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

pub(crate) fn average_metrics(values: &[RolloutMetrics]) -> RolloutMetrics {
    if values.is_empty() {
        return RolloutMetrics {
            ret_mean: 0.0,
            pnl: 0.0,
            realized_pnl: 0.0,
            sortino: 0.0,
            drawdown: 0.0,
            commission: 0.0,
            slippage: 0.0,
            buy_frac: 0.0,
            sell_frac: 0.0,
            hold_frac: 0.0,
            revert_frac: 0.0,
            mean_max_prob: 0.0,
            entries: 0.0,
            exits: 0.0,
            flips: 0.0,
            avg_hold: 0.0,
        };
    }

    let mut ret_mean = 0.0;
    let mut pnl = 0.0;
    let mut realized_pnl = 0.0;
    let mut sortino = 0.0;
    let mut drawdown = 0.0;
    let mut commission = 0.0;
    let mut slippage = 0.0;
    let mut buy_frac = 0.0;
    let mut sell_frac = 0.0;
    let mut hold_frac = 0.0;
    let mut revert_frac = 0.0;
    let mut mean_max_prob = 0.0;
    let mut entries = 0.0;
    let mut exits = 0.0;
    let mut flips = 0.0;
    let mut avg_hold = 0.0;
    for value in values {
        ret_mean += value.ret_mean;
        pnl += value.pnl;
        realized_pnl += value.realized_pnl;
        sortino += value.sortino;
        drawdown += value.drawdown;
        commission += value.commission;
        slippage += value.slippage;
        buy_frac += value.buy_frac;
        sell_frac += value.sell_frac;
        hold_frac += value.hold_frac;
        revert_frac += value.revert_frac;
        mean_max_prob += value.mean_max_prob;
        entries += value.entries;
        exits += value.exits;
        flips += value.flips;
        avg_hold += value.avg_hold;
    }

    let denom = values.len() as f64;
    RolloutMetrics {
        ret_mean: ret_mean / denom,
        pnl: pnl / denom,
        realized_pnl: realized_pnl / denom,
        sortino: sortino / denom,
        drawdown: drawdown / denom,
        commission: commission / denom,
        slippage: slippage / denom,
        buy_frac: buy_frac / denom,
        sell_frac: sell_frac / denom,
        hold_frac: hold_frac / denom,
        revert_frac: revert_frac / denom,
        mean_max_prob: mean_max_prob / denom,
        entries: entries / denom,
        exits: exits / denom,
        flips: flips / denom,
        avg_hold: avg_hold / denom,
    }
}

fn rollout_single(
    data: &DataSet,
    window: (usize, usize),
    policy: &Mlp,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    device: &Device,
    rng: &mut StdRng,
    greedy: bool,
) -> Result<GrpoRollout> {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let obs_dim = data.obs_dim;

    let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = cfg.initial_balance;
    let mut prev_equity = cfg.initial_balance;

    let mut obs_buf = Vec::with_capacity(steps * obs_dim);
    let mut act_buf = Vec::with_capacity(steps);
    let mut logp_buf = Vec::with_capacity(steps);
    let mut total_reward = 0.0;
    let mut total_pnl = 0.0;
    let mut ret_series = Vec::with_capacity(steps);
    let mut equity_curve = Vec::with_capacity(steps);
    let mut realized_pnl = 0.0f64;
    let mut commission = 0.0f64;
    let mut slippage = 0.0f64;
    let mut action_counts = [0usize; 4];
    let mut max_prob_sum = 0.0f64;
    let mut entries = 0usize;
    let mut exits = 0usize;
    let mut flips = 0usize;
    let mut entry_step = None;
    let mut total_trade_bars = 0usize;

    for t in (start + 1)..end {
        let obs = build_observation(
            data,
            t,
            position,
            equity,
            env.state().unrealized_pnl,
            env.state().realized_pnl,
            cfg.initial_balance,
        );
        obs_buf.extend_from_slice(&obs);
        let obs_tensor = Tensor::from_vec(obs, (1, obs_dim), device)?;

        let logits = policy.forward(&obs_tensor, train)?;
        let probs = softmax(&logits, 1)?.squeeze(0)?.to_vec1::<f32>()?;
        let action_idx = if greedy {
            argmax_index(&probs) as u32
        } else {
            sample_from_probs(&probs, rng) as u32
        };
        let logp_val = probs
            .get(action_idx as usize)
            .copied()
            .unwrap_or(1e-8)
            .max(1e-8)
            .ln();
        let max_prob = probs
            .iter()
            .copied()
            .fold(0.0f32, |acc, value| acc.max(value)) as f64;

        let position_before = position;
        let (reward, info) = env.step(
            action_from_index(action_idx),
            data.close[t],
            step_context(data, cfg, t),
        );
        position = env.state().position;
        equity = env.state().cash + env.state().unrealized_pnl;
        let position_after = position;
        let step_idx = t.saturating_sub(start + 1);

        act_buf.push(action_idx);
        logp_buf.push(logp_val);
        total_reward += reward;
        total_pnl += info.pnl_change;
        let denom = if prev_equity.abs() < 1e-8 {
            1e-8
        } else {
            prev_equity
        };
        ret_series.push(info.pnl_change / denom);
        equity_curve.push(equity);
        prev_equity = equity;
        realized_pnl += info.realized_pnl_change;
        commission += info.commission_paid;
        slippage += info.slippage_paid;
        max_prob_sum += max_prob;
        if let Some(count) = action_counts.get_mut(action_idx as usize) {
            *count += 1;
        }
        if position_before == 0 && position_after != 0 {
            entries += 1;
            entry_step = Some(step_idx);
        }
        if position_before != 0 && position_after == 0 {
            exits += 1;
            if let Some(start_step) = entry_step.take() {
                total_trade_bars += step_idx.saturating_sub(start_step) + 1;
            }
        }
        if position_before.signum() != position_after.signum()
            && position_before != 0
            && position_after != 0
        {
            flips += 1;
        }
    }

    Ok(GrpoRollout {
        obs: Tensor::from_vec(obs_buf, (steps, obs_dim), device)?,
        actions: Tensor::from_vec(act_buf, steps, device)?,
        logp: Tensor::from_vec(logp_buf, steps, device)?,
        reward: total_reward,
        pnl: total_pnl,
        returns: ret_series,
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

fn compute_gae(rewards: &[f64], values: &[f32], gamma: f64, lam: f64) -> (Vec<f32>, Vec<f32>) {
    let mut adv = vec![0.0f32; rewards.len()];
    let mut ret = vec![0.0f32; rewards.len()];
    let mut gae = 0.0f64;
    let mut next_value = 0.0f64;

    for t in (0..rewards.len()).rev() {
        let v = values.get(t).copied().unwrap_or(0.0) as f64;
        let delta = rewards[t] + gamma * next_value - v;
        gae = delta + gamma * lam * gae;
        adv[t] = gae as f32;
        ret[t] = (gae + v) as f32;
        next_value = v;
    }

    (adv, ret)
}

fn action_from_index(action_idx: u32) -> Action {
    match action_idx {
        0 => Action::Buy,
        1 => Action::Sell,
        2 => Action::Hold,
        _ => Action::Revert,
    }
}

fn step_context(data: &DataSet, cfg: &RolloutConfig, t: usize) -> StepContext {
    let session_open = if cfg.ignore_session {
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
