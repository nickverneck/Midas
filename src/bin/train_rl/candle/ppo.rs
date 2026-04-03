use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{
    AdamW, Optimizer, VarMap,
    ops::{log_softmax, softmax},
};

use super::model::Mlp;
use super::rollout::RolloutBatch;
use super::util::{named_grad_l2_norm, normalize_advantages};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LossStats {
    pub(crate) policy_loss: f64,
    pub(crate) value_loss: f64,
    pub(crate) entropy: f64,
    pub(crate) total_loss: f64,
    pub(crate) policy_grad_norm: f64,
    pub(crate) value_grad_norm: f64,
    pub(crate) approx_kl: f64,
    pub(crate) clip_frac: f64,
}

pub(crate) struct PpoConfig {
    pub(crate) clip: f64,
    pub(crate) vf_coef: f64,
    pub(crate) ent_coef: f64,
    pub(crate) ppo_epochs: usize,
}

pub(crate) fn ppo_update(
    policy: &Mlp,
    value: &Mlp,
    varmap: &VarMap,
    opt: &mut AdamW,
    batch: &RolloutBatch,
    device: &Device,
    cfg: &PpoConfig,
) -> Result<LossStats> {
    let normalized_adv = normalize_advantages(&batch.adv);
    let adv_tensor = Tensor::from_vec(normalized_adv, batch.adv.len(), device)?;

    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut total_loss_sum = 0.0;
    let mut policy_grad_norm_sum = 0.0;
    let mut value_grad_norm_sum = 0.0;
    let mut approx_kl_sum = 0.0;
    let mut clip_frac_sum = 0.0;

    for _ in 0..cfg.ppo_epochs {
        let logits = policy.forward(&batch.obs, true)?;
        let log_probs = log_softmax(&logits, 1)?;
        let actions = batch.actions.unsqueeze(1)?;
        let act_logp = log_probs.gather(&actions, 1)?.squeeze(1)?;
        let log_ratio = (&act_logp - &batch.logp)?;
        let ratio = log_ratio.exp()?;

        let surr1 = (&ratio * &adv_tensor)?;
        let clipped_ratio = ratio.clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
        let surr2 = (&clipped_ratio * &adv_tensor)?;
        let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
        let policy_loss = surr.neg()?.mean_all()?;

        let value_pred = value.forward(&batch.obs, true)?.squeeze(1)?;
        let value_loss = (&value_pred - &batch.ret)?.sqr()?.mean_all()?;

        let probs = softmax(&logits, 1)?;
        let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

        let loss = (&policy_loss + (&value_loss * cfg.vf_coef)?)?;
        let loss = (&loss - (&entropy * cfg.ent_coef)?)?;

        let grads = loss.backward()?;
        policy_grad_norm_sum += named_grad_l2_norm(varmap, "policy", &grads)?;
        value_grad_norm_sum += named_grad_l2_norm(varmap, "value", &grads)?;
        opt.step(&grads)?;

        let ratio_values = ratio.to_vec1::<f32>()?;
        let log_ratio_values = log_ratio.to_vec1::<f32>()?;
        let denom = ratio_values.len().max(1) as f64;
        let approx_kl = ratio_values
            .iter()
            .zip(log_ratio_values.iter())
            .map(|(ratio_value, log_ratio_value)| {
                (*ratio_value as f64 - 1.0) - *log_ratio_value as f64
            })
            .sum::<f64>()
            / denom;
        let clip_frac = ratio_values
            .iter()
            .filter(|ratio_value| {
                let ratio_value = **ratio_value as f64;
                ratio_value < 1.0 - cfg.clip || ratio_value > 1.0 + cfg.clip
            })
            .count() as f64
            / denom;

        policy_loss_sum += policy_loss.to_scalar::<f32>()? as f64;
        value_loss_sum += value_loss.to_scalar::<f32>()? as f64;
        entropy_sum += entropy.to_scalar::<f32>()? as f64;
        total_loss_sum += loss.to_scalar::<f32>()? as f64;
        approx_kl_sum += approx_kl;
        clip_frac_sum += clip_frac;
    }

    let denom = cfg.ppo_epochs.max(1) as f64;
    Ok(LossStats {
        policy_loss: policy_loss_sum / denom,
        value_loss: value_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        value_grad_norm: value_grad_norm_sum / denom,
        approx_kl: approx_kl_sum / denom,
        clip_frac: clip_frac_sum / denom,
    })
}

pub(crate) fn average_losses(values: &[LossStats]) -> LossStats {
    if values.is_empty() {
        return LossStats {
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            policy_grad_norm: 0.0,
            value_grad_norm: 0.0,
            approx_kl: 0.0,
            clip_frac: 0.0,
        };
    }

    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    let mut policy_grad_norm = 0.0;
    let mut value_grad_norm = 0.0;
    let mut approx_kl = 0.0;
    let mut clip_frac = 0.0;
    for value in values {
        policy_loss += value.policy_loss;
        value_loss += value.value_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
        policy_grad_norm += value.policy_grad_norm;
        value_grad_norm += value.value_grad_norm;
        approx_kl += value.approx_kl;
        clip_frac += value.clip_frac;
    }

    let denom = values.len() as f64;
    LossStats {
        policy_loss: policy_loss / denom,
        value_loss: value_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        policy_grad_norm: policy_grad_norm / denom,
        value_grad_norm: value_grad_norm / denom,
        approx_kl: approx_kl / denom,
        clip_frac: clip_frac / denom,
    }
}
