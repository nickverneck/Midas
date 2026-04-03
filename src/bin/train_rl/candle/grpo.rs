use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    AdamW, Optimizer, VarMap,
    ops::{log_softmax, softmax},
};

use super::model::Mlp;
use super::rollout::GrpoGroup;
use super::util::{named_grad_l2_norm, normalize_advantages_f64};

#[derive(Debug, Clone, Copy)]
pub(crate) struct GrpoLossStats {
    pub(crate) policy_loss: f64,
    pub(crate) entropy: f64,
    pub(crate) total_loss: f64,
    pub(crate) kl_div: f64,
    pub(crate) policy_grad_norm: f64,
    pub(crate) clip_frac: f64,
}

pub(crate) struct GrpoConfig {
    pub(crate) group_size: usize,
    pub(crate) clip: f64,
    pub(crate) ent_coef: f64,
    pub(crate) grpo_epochs: usize,
}

pub(crate) fn grpo_update(
    policy: &Mlp,
    varmap: &VarMap,
    opt: &mut AdamW,
    group: &GrpoGroup,
    advantages: &[f64],
    cfg: &GrpoConfig,
) -> Result<GrpoLossStats> {
    let normalized_adv = normalize_advantages_f64(advantages);
    let mut policy_loss_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut total_loss_sum = 0.0;
    let mut kl_sum = 0.0;
    let mut policy_grad_norm_sum = 0.0;
    let mut clip_frac_sum = 0.0;

    for _ in 0..cfg.grpo_epochs {
        let mut epoch_policy_loss = 0.0;
        let mut epoch_entropy = 0.0;
        let mut epoch_kl = 0.0;
        let mut epoch_clip_frac = 0.0;

        for (i, rollout) in group.rollouts.iter().enumerate() {
            let logits = policy.forward(&rollout.obs, true)?;
            let log_probs = log_softmax(&logits, 1)?;
            let act_logp = log_probs
                .gather(&rollout.actions.unsqueeze(1)?, 1)?
                .squeeze(1)?;
            let ratio = (&act_logp - &rollout.logp)?.exp()?;

            let adv = normalized_adv.get(i).copied().unwrap_or(0.0);
            let surr1 = (&ratio * adv)?;
            let clipped_ratio = ratio.clone().clamp(1.0 - cfg.clip, 1.0 + cfg.clip)?;
            let surr2 = (&clipped_ratio * adv)?;
            let surr = Tensor::stack(&[&surr1, &surr2], 0)?.min(0)?;
            let policy_loss = surr.neg()?.mean_all()?;

            let probs = softmax(&logits, 1)?;
            let entropy = (&probs * &log_probs)?.sum(1)?.neg()?.mean_all()?;

            let loss = (&policy_loss - (&entropy * cfg.ent_coef)?)?;
            let grads = loss.backward()?;
            policy_grad_norm_sum += named_grad_l2_norm(varmap, "policy", &grads)?;
            opt.step(&grads)?;

            epoch_policy_loss += policy_loss.to_scalar::<f32>()? as f64;
            epoch_entropy += entropy.to_scalar::<f32>()? as f64;
            epoch_kl += (&rollout.logp - &act_logp)?
                .mean_all()?
                .abs()?
                .to_scalar::<f32>()? as f64;
            let ratio_values = ratio.to_vec1::<f32>()?;
            let denom = ratio_values.len().max(1) as f64;
            epoch_clip_frac += ratio_values
                .iter()
                .filter(|ratio_value| {
                    let ratio_value = **ratio_value as f64;
                    ratio_value < 1.0 - cfg.clip || ratio_value > 1.0 + cfg.clip
                })
                .count() as f64
                / denom;
        }

        let denom = group.rollouts.len().max(1) as f64;
        policy_loss_sum += epoch_policy_loss / denom;
        entropy_sum += epoch_entropy / denom;
        kl_sum += epoch_kl / denom;
        clip_frac_sum += epoch_clip_frac / denom;
        total_loss_sum += (epoch_policy_loss - cfg.ent_coef * epoch_entropy) / denom;
    }

    let denom = cfg.grpo_epochs.max(1) as f64;
    Ok(GrpoLossStats {
        policy_loss: policy_loss_sum / denom,
        entropy: entropy_sum / denom,
        total_loss: total_loss_sum / denom,
        kl_div: kl_sum / denom,
        policy_grad_norm: policy_grad_norm_sum / denom,
        clip_frac: clip_frac_sum / denom,
    })
}

pub(crate) fn average_grpo_losses(values: &[GrpoLossStats]) -> GrpoLossStats {
    if values.is_empty() {
        return GrpoLossStats {
            policy_loss: 0.0,
            entropy: 0.0,
            total_loss: 0.0,
            kl_div: 0.0,
            policy_grad_norm: 0.0,
            clip_frac: 0.0,
        };
    }

    let mut policy_loss = 0.0;
    let mut entropy = 0.0;
    let mut total_loss = 0.0;
    let mut kl_div = 0.0;
    let mut policy_grad_norm = 0.0;
    let mut clip_frac = 0.0;
    for value in values {
        policy_loss += value.policy_loss;
        entropy += value.entropy;
        total_loss += value.total_loss;
        kl_div += value.kl_div;
        policy_grad_norm += value.policy_grad_norm;
        clip_frac += value.clip_frac;
    }

    let denom = values.len() as f64;
    GrpoLossStats {
        policy_loss: policy_loss / denom,
        entropy: entropy / denom,
        total_loss: total_loss / denom,
        kl_div: kl_div / denom,
        policy_grad_norm: policy_grad_norm / denom,
        clip_frac: clip_frac / denom,
    }
}
