use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
use rand::rngs::StdRng;
use tch::{Kind, Tensor, nn, no_grad};

use crate::data::{DataSet, build_observation};
use crate::metrics::{compute_sortino, max_drawdown};
use crate::util;

pub struct GrpoRollout {
    pub obs: Tensor,
    pub actions: Tensor,
    pub logp: Tensor,
    pub reward: f64,
    pub pnl: f64,
    pub returns: Vec<f64>,
    pub equity: Vec<f64>,
    pub realized_pnl: f64,
    pub commission: f64,
    pub slippage: f64,
    pub action_counts: [usize; 4],
    pub max_prob_sum: f64,
    pub entries: usize,
    pub exits: usize,
    pub flips: usize,
    pub total_trade_bars: usize,
}

pub struct GrpoGroup {
    pub rollouts: Vec<GrpoRollout>,
    pub mean_reward: f64,
    pub std_reward: f64,
}

// Reuse RolloutMetrics from ppo for compatibility
pub use crate::ppo::RolloutMetrics as GrpoMetrics;

#[derive(Debug, Clone, Copy)]
pub struct GrpoLossStats {
    pub policy_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
    pub kl_div: f64,
    pub policy_grad_norm: f64,
    pub clip_frac: f64,
}

pub struct GrpoConfig {
    pub group_size: usize,
    pub clip: f64,
    pub ent_coef: f64,
    pub grpo_epochs: usize,
}

pub fn rollout_single<P: nn::ModuleT>(
    data: &DataSet,
    window: (usize, usize),
    policy: &P,
    env_cfg: &EnvConfig,
    device: tch::Device,
    train: bool,
    rng: &mut StdRng,
    greedy: bool,
) -> GrpoRollout {
    let (start, end) = window;
    let steps = end.saturating_sub(start + 1);
    let obs_dim = data.obs_dim;

    let mut env = TradingEnv::new(data.close[start], 10000.0, env_cfg.clone());
    let mut position = 0i32;
    let mut equity = 10000.0;
    let mut prev_equity = 10000.0;

    let mut obs_buf = Vec::with_capacity(steps * obs_dim);
    let mut act_buf = Vec::with_capacity(steps);
    let mut logp_buf = Vec::with_capacity(steps);
    let mut total_reward = 0.0f64;
    let mut total_pnl = 0.0f64;
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
            10000.0,
        );
        obs_buf.extend_from_slice(&obs);
        let obs_tensor = Tensor::f_from_slice(&obs)
            .expect("tensor from obs slice")
            .reshape(&[1, obs_dim as i64])
            .to_device(device);

        let probs = no_grad(|| {
            let logits = policy.forward_t(&obs_tensor, train);
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let probs = log_probs.exp();
            util::tensor_to_vec_f32(&probs)
        });
        let action_idx = if greedy {
            util::argmax_index(&probs)
        } else {
            util::sample_from_probs(&probs, rng)
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

        let action = match action_idx {
            0 => Action::Buy,
            1 => Action::Sell,
            2 => Action::Hold,
            _ => Action::Revert,
        };
        let session_open = data
            .session_open
            .as_ref()
            .and_then(|m| m.get(t))
            .copied()
            .unwrap_or(true);
        let minutes_to_close = data
            .minutes_to_close
            .as_ref()
            .and_then(|m| m.get(t))
            .copied();
        let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);

        let position_before = position;
        let (reward, info) = env.step(
            action,
            data.close[t],
            StepContext {
                session_open,
                margin_ok,
                minutes_to_close,
            },
        );
        position = env.state().position;
        equity = env.state().cash + env.state().unrealized_pnl;
        let position_after = position;
        let step_idx = t.saturating_sub(start + 1);

        act_buf.push(action_idx);
        logp_buf.push(logp_val as f32);
        total_reward += reward;
        total_pnl += info.pnl_change;
        let denom = if (prev_equity as f64).abs() < 1e-8 {
            1e-8f64
        } else {
            prev_equity as f64
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

    let obs_tensor = Tensor::f_from_slice(&obs_buf)
        .expect("tensor from obs buffer")
        .reshape(&[steps as i64, obs_dim as i64])
        .to_device(device);
    let act_tensor = Tensor::f_from_slice(&act_buf)
        .expect("tensor from action buffer")
        .to_device(device);
    let logp_tensor = Tensor::f_from_slice(&logp_buf)
        .expect("tensor from logp buffer")
        .to_device(device);

    GrpoRollout {
        obs: obs_tensor,
        actions: act_tensor,
        logp: logp_tensor,
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
    }
}

pub fn rollout_group<P: nn::ModuleT>(
    data: &DataSet,
    windows: &[(usize, usize)],
    policy: &P,
    env_cfg: &EnvConfig,
    group_size: usize,
    device: tch::Device,
    train: bool,
    rng: &mut StdRng,
    greedy: bool,
) -> GrpoGroup {
    let mut rollouts = Vec::with_capacity(group_size);

    for i in 0..group_size {
        let window = windows[i % windows.len()];
        let rollout = rollout_single(data, window, policy, env_cfg, device, train, rng, greedy);
        rollouts.push(rollout);
    }

    let rewards: Vec<f64> = rollouts.iter().map(|r| r.reward).collect();
    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let variance = rewards
        .iter()
        .map(|r| (r - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    let std_reward = variance.sqrt().max(1e-8);

    GrpoGroup {
        rollouts,
        mean_reward,
        std_reward,
    }
}

pub fn compute_grpo_advantages(group: &GrpoGroup) -> Vec<f64> {
    group
        .rollouts
        .iter()
        .map(|r| (r.reward - group.mean_reward) / group.std_reward)
        .collect()
}

pub fn summarize_group(group: &GrpoGroup, annualization: f64) -> GrpoMetrics {
    let all_returns: Vec<f64> = group
        .rollouts
        .iter()
        .flat_map(|r| r.returns.clone())
        .collect();

    let ret_mean = if all_returns.is_empty() {
        0.0
    } else {
        all_returns.iter().sum::<f64>() / all_returns.len() as f64
    };

    let pnl_total = group.rollouts.iter().map(|r| r.pnl).sum::<f64>();
    let realized_pnl = group.rollouts.iter().map(|r| r.realized_pnl).sum::<f64>();
    let commission = group.rollouts.iter().map(|r| r.commission).sum::<f64>();
    let slippage = group.rollouts.iter().map(|r| r.slippage).sum::<f64>();
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
        .flat_map(|r| r.equity.clone())
        .collect();

    let sortino = compute_sortino(&all_returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&all_equity);

    GrpoMetrics {
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

pub fn grpo_update<P: nn::ModuleT>(
    policy: &P,
    vs: &nn::VarStore,
    opt: &mut nn::Optimizer,
    group: &GrpoGroup,
    advantages: &[f64],
    cfg: &GrpoConfig,
) -> GrpoLossStats {
    let mut policy_loss_sum = 0.0f64;
    let mut entropy_sum = 0.0f64;
    let mut total_loss_sum = 0.0f64;
    let mut kl_sum = 0.0f64;
    let mut policy_grad_norm_sum = 0.0f64;
    let mut clip_frac_sum = 0.0f64;

    let adv_tensor = Tensor::f_from_slice(advantages)
        .expect("tensor from advantages")
        .to_device(group.rollouts[0].obs.device());

    let adv_mean = adv_tensor.mean(Kind::Float);
    let diff = &adv_tensor - &adv_mean;
    let squared = &diff * &diff;
    let adv_std = squared.mean(Kind::Float).sqrt() + 1e-8;
    let normalized_adv = (&adv_tensor - &adv_mean) / &adv_std;

    for _epoch in 0..cfg.grpo_epochs {
        let mut epoch_policy_loss = 0.0f64;
        let mut epoch_entropy = 0.0f64;
        let mut epoch_kl = 0.0f64;
        let mut epoch_clip_frac = 0.0f64;

        for (i, rollout) in group.rollouts.iter().enumerate() {
            let logits = policy.forward_t(&rollout.obs, true);
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let act_logp = log_probs
                .gather(1, &rollout.actions.unsqueeze(-1), false)
                .squeeze();
            let ratio = (&act_logp - &rollout.logp).exp();

            let adv = normalized_adv.get(i as i64);
            let surr1 = &ratio * &adv;
            let clipped_ratio = ratio.shallow_clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip);
            let surr2 = &clipped_ratio * &adv;
            let stacked = Tensor::stack(&[surr1, surr2], 0);
            let surr = stacked.min_dim(0, false).0;
            let policy_loss = -surr.mean(Kind::Float);

            let probs = log_probs.exp();
            let entropy =
                (-(probs * log_probs).sum_dim_intlist([-1_i64].as_ref(), false, Kind::Float))
                    .mean(Kind::Float);

            let loss = &policy_loss - cfg.ent_coef * &entropy;

            opt.zero_grad();
            loss.backward();
            policy_grad_norm_sum += util::named_grad_l2_norm(vs, "policy.");
            opt.step();

            epoch_policy_loss += policy_loss.double_value(&[]);
            epoch_entropy += entropy.double_value(&[]);

            let kl = (&rollout.logp - &act_logp)
                .mean(Kind::Float)
                .double_value(&[])
                .abs();
            epoch_kl += kl;
            epoch_clip_frac += (&ratio - &clipped_ratio)
                .abs()
                .gt(1e-6)
                .to_kind(Kind::Float)
                .mean(Kind::Float)
                .double_value(&[]);
        }

        policy_loss_sum += epoch_policy_loss / group.rollouts.len() as f64;
        entropy_sum += epoch_entropy / group.rollouts.len() as f64;
        kl_sum += epoch_kl / group.rollouts.len() as f64;
        clip_frac_sum += epoch_clip_frac / group.rollouts.len() as f64;
        total_loss_sum +=
            (epoch_policy_loss - cfg.ent_coef * epoch_entropy) / group.rollouts.len() as f64;
    }

    let denom = cfg.grpo_epochs.max(1) as f64;
    GrpoLossStats {
        policy_loss: policy_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        kl_div: kl_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        clip_frac: clip_frac_sum / denom,
    }
}
