use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
use rand::rngs::StdRng;
use tch::{Kind, Tensor, nn, no_grad};

use crate::data::{DataSet, build_observation};
use crate::metrics::{compute_sortino, max_drawdown};
use crate::util;

pub struct RolloutBatch {
    pub obs: Tensor,
    pub actions: Tensor,
    pub logp: Tensor,
    pub adv: Tensor,
    pub ret: Tensor,
    pub values: Tensor,
    pub rewards: Vec<f64>,
    pub pnl: Vec<f64>,
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

#[derive(Debug, Clone, Copy)]
pub struct RolloutMetrics {
    pub ret_mean: f64,
    pub pnl: f64,
    pub realized_pnl: f64,
    pub sortino: f64,
    pub drawdown: f64,
    pub commission: f64,
    pub slippage: f64,
    pub buy_frac: f64,
    pub sell_frac: f64,
    pub hold_frac: f64,
    pub revert_frac: f64,
    pub mean_max_prob: f64,
    pub entries: f64,
    pub exits: f64,
    pub flips: f64,
    pub avg_hold: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LossStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
    pub policy_grad_norm: f64,
    pub value_grad_norm: f64,
    pub approx_kl: f64,
    pub clip_frac: f64,
}

pub struct RolloutConfig {
    pub gamma: f64,
    pub lam: f64,
    pub initial_balance: f64,
    pub device: tch::Device,
    pub ignore_session: bool,
}

pub struct PpoConfig {
    pub clip: f64,
    pub vf_coef: f64,
    pub ent_coef: f64,
    pub ppo_epochs: usize,
}

pub fn rollout<P: nn::ModuleT, V: nn::ModuleT>(
    data: &DataSet,
    window: (usize, usize),
    policy: &P,
    value: &V,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
    train: bool,
    rng: &mut StdRng,
    greedy: bool,
) -> RolloutBatch {
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
        let obs_tensor = Tensor::f_from_slice(&obs)
            .expect("tensor from obs slice")
            .reshape(&[1, obs_dim as i64])
            .to_device(cfg.device);

        let (probs, value_val) = no_grad(|| {
            let logits = policy.forward_t(&obs_tensor, train);
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let probs = log_probs.exp();
            let value_val = value.forward_t(&obs_tensor, train).double_value(&[0, 0]) as f32;
            (util::tensor_to_vec_f32(&probs), value_val)
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

    let obs_tensor = Tensor::f_from_slice(&obs_buf)
        .expect("tensor from obs buffer")
        .reshape(&[steps as i64, obs_dim as i64])
        .to_device(cfg.device);
    let act_tensor = Tensor::f_from_slice(&act_buf)
        .expect("tensor from action buffer")
        .to_device(cfg.device);
    let logp_tensor = Tensor::f_from_slice(&logp_buf)
        .expect("tensor from logp buffer")
        .to_device(cfg.device);
    let adv_tensor = Tensor::f_from_slice(&adv_buf)
        .expect("tensor from advantage buffer")
        .to_device(cfg.device);
    let ret_tensor = Tensor::f_from_slice(&ret_buf)
        .expect("tensor from return buffer")
        .to_device(cfg.device);
    let val_tensor = Tensor::f_from_slice(&val_buf)
        .expect("tensor from value buffer")
        .to_device(cfg.device);

    RolloutBatch {
        obs: obs_tensor,
        actions: act_tensor,
        logp: logp_tensor,
        adv: adv_tensor,
        ret: ret_tensor,
        values: val_tensor,
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
    }
}

pub fn summarize_batch(batch: &RolloutBatch, annualization: f64) -> RolloutMetrics {
    let ret_mean = if batch.returns.is_empty() {
        0.0
    } else {
        batch.returns.iter().sum::<f64>() / batch.returns.len() as f64
    };
    let pnl_total = batch.pnl.iter().sum::<f64>();
    let sortino = compute_sortino(&batch.returns, annualization, 0.0, 50.0);
    let drawdown = max_drawdown(&batch.equity);
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

pub fn ppo_update<P: nn::ModuleT, V: nn::ModuleT>(
    policy: &P,
    value: &V,
    vs: &nn::VarStore,
    opt: &mut nn::Optimizer,
    batch: &RolloutBatch,
    cfg: &PpoConfig,
) -> LossStats {
    let mut policy_loss_sum = 0.0f64;
    let mut value_loss_sum = 0.0f64;
    let mut entropy_sum = 0.0f64;
    let mut total_loss_sum = 0.0f64;
    let mut policy_grad_norm_sum = 0.0f64;
    let mut value_grad_norm_sum = 0.0f64;
    let mut approx_kl_sum = 0.0f64;
    let mut clip_frac_sum = 0.0f64;

    for _ in 0..cfg.ppo_epochs {
        let logits = policy.forward_t(&batch.obs, true);
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let act_logp = log_probs
            .gather(1, &batch.actions.unsqueeze(-1), false)
            .squeeze();
        let log_ratio = &act_logp - &batch.logp;
        let ratio = log_ratio.exp();

        let adv_mean = batch.adv.mean(Kind::Float);
        let adv_diff = &batch.adv - &adv_mean;
        let adv_var = (&adv_diff * &adv_diff).mean(Kind::Float);
        let adv_std = adv_var.sqrt() + 1e-8;
        let adv = adv_diff / adv_std;

        let clipped_ratio = ratio.shallow_clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip);
        let surr1 = &ratio * &adv;
        let surr2 = &clipped_ratio * &adv;
        let stacked = Tensor::stack(&[surr1, surr2], 0);
        let surr = stacked.min_dim(0, false).0;
        let policy_loss = -surr.mean(Kind::Float);

        let value_pred = value.forward_t(&batch.obs, true).flatten(0, -1);
        let value_diff = &value_pred - &batch.ret;
        let value_loss = (&value_diff * &value_diff).mean(Kind::Float);

        let probs = log_probs.exp();
        let entropy = (-(probs * log_probs).sum_dim_intlist([-1_i64].as_ref(), false, Kind::Float))
            .mean(Kind::Float);

        let loss = &policy_loss + cfg.vf_coef * &value_loss - cfg.ent_coef * &entropy;
        let approx_kl = ((&ratio - 1.0) - &log_ratio).mean(Kind::Float);
        let clip_frac = (&ratio - &clipped_ratio)
            .abs()
            .gt(1e-6)
            .to_kind(Kind::Float)
            .mean(Kind::Float);

        opt.zero_grad();
        loss.backward();
        policy_grad_norm_sum += util::named_grad_l2_norm(vs, "policy.");
        value_grad_norm_sum += util::named_grad_l2_norm(vs, "value.");
        opt.step();

        policy_loss_sum += policy_loss.double_value(&[]);
        value_loss_sum += value_loss.double_value(&[]);
        entropy_sum += entropy.double_value(&[]);
        total_loss_sum += loss.double_value(&[]);
        approx_kl_sum += approx_kl.double_value(&[]);
        clip_frac_sum += clip_frac.double_value(&[]);
    }

    let denom = cfg.ppo_epochs.max(1) as f64;
    LossStats {
        policy_loss: policy_loss_sum / denom,
        value_loss: value_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        value_grad_norm: value_grad_norm_sum / denom,
        approx_kl: approx_kl_sum / denom,
        clip_frac: clip_frac_sum / denom,
    }
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
