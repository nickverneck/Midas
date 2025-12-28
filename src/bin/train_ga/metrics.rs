pub fn compute_sortino(returns: &[f64], annualization: f64, target: f64, cap: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for r in returns {
        sum += r - target;
    }
    let mean = sum / returns.len() as f64;
    let mut downside = 0.0;
    for r in returns {
        let ex = r - target;
        let d = if ex < 0.0 { ex } else { 0.0 };
        downside += d * d;
    }
    let downside_std = (downside / returns.len() as f64).sqrt();
    if downside_std < 1e-6 {
        return if mean > 0.0 { cap } else { 0.0 };
    }
    let mut ratio = mean / (downside_std + 1e-8);
    ratio *= annualization.sqrt();
    ratio.min(cap)
}

pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0;
    for &eq in equity {
        if eq > peak {
            peak = eq;
        }
        let dd = peak - eq;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}
