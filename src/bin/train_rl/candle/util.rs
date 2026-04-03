use anyhow::Result;
use candle_nn::VarMap;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

pub(crate) fn normalize_advantages(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std = var.sqrt().max(1e-8);
    values
        .iter()
        .map(|v| ((*v as f64 - mean) / std) as f32)
        .collect()
}

pub(crate) fn normalize_advantages_f64(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (*v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std = var.sqrt().max(1e-8);
    values.iter().map(|v| (*v - mean) / std).collect()
}

pub(crate) fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
    let weights: Vec<f64> = probs
        .iter()
        .map(|prob| {
            if prob.is_finite() && *prob > 0.0 {
                *prob as f64
            } else {
                0.0
            }
        })
        .collect();

    if weights.iter().all(|weight| *weight <= 0.0) {
        return probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    WeightedIndex::new(&weights)
        .map(|dist| dist.sample(rng))
        .unwrap_or_else(|_| argmax_index(probs))
}

pub(crate) fn argmax_index(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

pub(crate) fn named_grad_l2_norm(
    varmap: &VarMap,
    prefix: &str,
    grads: &candle_core::backprop::GradStore,
) -> Result<f64> {
    let vars = varmap.data().lock().unwrap();
    let mut total = 0.0f64;
    for (name, var) in vars.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        let Some(grad) = grads.get(var.as_tensor()) else {
            continue;
        };
        total += grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    }
    Ok(total.sqrt())
}
