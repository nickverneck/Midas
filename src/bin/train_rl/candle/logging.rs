use anyhow::Result;
use std::io::Write;

use super::rollout::RolloutMetrics;

fn format_log_f64(value: f64, precision: usize) -> String {
    format!("{value:.precision$}")
}

fn format_log_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|inner| format_log_f64(inner, precision))
        .unwrap_or_default()
}

fn append_rollout_fields(fields: &mut Vec<String>, metrics: Option<&RolloutMetrics>) {
    let Some(metrics) = metrics else {
        for _ in 0..16 {
            fields.push(String::new());
        }
        return;
    };
    fields.push(format_log_f64(metrics.ret_mean, 4));
    fields.push(format_log_f64(metrics.pnl, 4));
    fields.push(format_log_f64(metrics.realized_pnl, 4));
    fields.push(format_log_f64(metrics.sortino, 4));
    fields.push(format_log_f64(metrics.drawdown, 4));
    fields.push(format_log_f64(metrics.commission, 4));
    fields.push(format_log_f64(metrics.slippage, 4));
    fields.push(format_log_f64(metrics.buy_frac, 6));
    fields.push(format_log_f64(metrics.sell_frac, 6));
    fields.push(format_log_f64(metrics.hold_frac, 6));
    fields.push(format_log_f64(metrics.revert_frac, 6));
    fields.push(format_log_f64(metrics.mean_max_prob, 6));
    fields.push(format_log_f64(metrics.entries, 4));
    fields.push(format_log_f64(metrics.exits, 4));
    fields.push(format_log_f64(metrics.flips, 4));
    fields.push(format_log_f64(metrics.avg_hold, 4));
}

pub(crate) fn write_rl_log_row(
    file: &mut std::fs::File,
    epoch: usize,
    algorithm: &str,
    train: &RolloutMetrics,
    eval: &RolloutMetrics,
    probe: Option<&RolloutMetrics>,
    fitness: f64,
    policy_loss: f64,
    value_loss: Option<f64>,
    entropy: f64,
    total_loss: f64,
    policy_grad_norm: f64,
    value_grad_norm: Option<f64>,
    approx_kl: Option<f64>,
    kl_div: Option<f64>,
    clip_frac: f64,
) -> Result<()> {
    let mut fields = Vec::with_capacity(60);
    fields.push(epoch.to_string());
    fields.push(algorithm.to_string());
    append_rollout_fields(&mut fields, Some(train));
    append_rollout_fields(&mut fields, Some(eval));
    append_rollout_fields(&mut fields, probe);
    fields.push(format_log_f64(fitness, 4));
    fields.push(format_log_f64(policy_loss, 6));
    fields.push(format_log_opt(value_loss, 6));
    fields.push(format_log_f64(entropy, 6));
    fields.push(format_log_f64(entropy.exp(), 6));
    fields.push(format_log_f64(total_loss, 6));
    fields.push(format_log_f64(policy_grad_norm, 6));
    fields.push(format_log_opt(value_grad_norm, 6));
    fields.push(format_log_opt(approx_kl, 6));
    fields.push(format_log_opt(kl_div, 6));
    fields.push(format_log_f64(clip_frac, 6));
    writeln!(file, "{}", fields.join(","))?;
    Ok(())
}

pub(crate) fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    let minutes = secs / 60;
    let seconds = secs % 60;
    if minutes > 0 {
        format!("{minutes}m{seconds:02}.{millis:03}s")
    } else {
        format!("{seconds}.{millis:03}s")
    }
}
