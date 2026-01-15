use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
use tch::{nn, no_grad, Kind, Tensor};

use crate::data::{build_observation, DataSet};
use crate::metrics::{compute_sortino, max_drawdown};

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
}

#[derive(Debug, Clone, Copy)]
pub struct RolloutMetrics {
    pub ret_mean: f64,
    pub pnl: f64,
    pub sortino: f64,
    pub drawdown: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LossStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
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

pub fn rollout<P: nn::Module, V: nn::Module>(
    data: &DataSet,
    window: (usize, usize),
    policy: &P,
    value: &V,
    env_cfg: &EnvConfig,
    cfg: &RolloutConfig,
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
        let obs_tensor = Tensor::of_slice(&obs)
            .reshape(&[1, obs_dim as i64])
            .to_device(cfg.device);

        let (action_idx, logp_val, value_val) = no_grad(|| {
            let logits = policy.forward(&obs_tensor);
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let probs = log_probs.exp();
            let action_tensor = probs.multinomial(1, true);
            let action_idx = action_tensor.int64_value(&[0, 0]);
            let logp_val = log_probs
                .gather(1, &action_tensor, false)
                .double_value(&[0, 0]) as f32;
            let value_val = value.forward(&obs_tensor).double_value(&[0, 0]) as f32;
            (action_idx, logp_val, value_val)
        });

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

        act_buf.push(action_idx);
        logp_buf.push(logp_val);
        val_buf.push(value_val);
        rew_buf.push(reward);
        pnl_buf.push(info.pnl_change);
        let denom = if prev_equity.abs() < 1e-8 { 1e-8 } else { prev_equity };
        ret_series.push(info.pnl_change / denom);
        equity_curve.push(equity);
        prev_equity = equity;
    }

    let (adv_buf, ret_buf) = compute_gae(&rew_buf, &val_buf, cfg.gamma, cfg.lam);

    let obs_tensor = Tensor::of_slice(&obs_buf)
        .reshape(&[steps as i64, obs_dim as i64])
        .to_device(cfg.device);
    let act_tensor = Tensor::of_slice(&act_buf).to_device(cfg.device);
    let logp_tensor = Tensor::of_slice(&logp_buf).to_device(cfg.device);
    let adv_tensor = Tensor::of_slice(&adv_buf).to_device(cfg.device);
    let ret_tensor = Tensor::of_slice(&ret_buf).to_device(cfg.device);
    let val_tensor = Tensor::of_slice(&val_buf).to_device(cfg.device);

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
    RolloutMetrics {
        ret_mean,
        pnl: pnl_total,
        sortino,
        drawdown,
    }
}

pub fn ppo_update<P: nn::Module, V: nn::Module>(
    policy: &P,
    value: &V,
    opt: &mut nn::Optimizer,
    batch: &RolloutBatch,
    cfg: &PpoConfig,
) -> LossStats {
    let mut policy_loss_sum = 0.0f64;
    let mut value_loss_sum = 0.0f64;
    let mut entropy_sum = 0.0f64;
    let mut total_loss_sum = 0.0f64;

    for _ in 0..cfg.ppo_epochs {
        let logits = policy.forward(&batch.obs);
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let act_logp = log_probs
            .gather(1, &batch.actions.unsqueeze(-1), false)
            .squeeze();
        let ratio = (&act_logp - &batch.logp).exp();

        let adv_mean = batch.adv.mean(Kind::Float);
        let adv_diff = &batch.adv - &adv_mean;
        let adv_var = (&adv_diff * &adv_diff).mean(Kind::Float);
        let adv_std = adv_var.sqrt() + 1e-8;
        let adv = adv_diff / adv_std;

        let surr1 = &ratio * &adv;
        let surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * &adv;
        let stacked = Tensor::stack(&[surr1, surr2], 0);
        let surr = stacked.min_dim(0, false).0;
        let policy_loss = -surr.mean(Kind::Float);

        let value_pred = value.forward(&batch.obs).flatten(0, -1);
        let value_diff = &value_pred - &batch.ret;
        let value_loss = (&value_diff * &value_diff).mean(Kind::Float);

        let probs = log_probs.exp();
        let entropy = (-(probs * log_probs)
            .sum_dim_intlist([-1_i64].as_ref(), false, Kind::Float))
        .mean(Kind::Float);

        let loss = &policy_loss + cfg.vf_coef * &value_loss - cfg.ent_coef * &entropy;

        opt.zero_grad();
        loss.backward();
        opt.step();

        policy_loss_sum += f64::from(&policy_loss);
        value_loss_sum += f64::from(&value_loss);
        entropy_sum += f64::from(&entropy);
        total_loss_sum += f64::from(&loss);
    }

    let denom = cfg.ppo_epochs.max(1) as f64;
    LossStats {
        policy_loss: policy_loss_sum / denom,
        value_loss: value_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
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
