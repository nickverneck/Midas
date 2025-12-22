//! Feature computation: SMA, EMA, HMA for a fixed set of periods.
use std::collections::HashMap;

const PERIODS: [usize; 14] = [3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53];
pub const ATR_PERIODS: [usize; 3] = [7, 14, 21];

pub fn periods() -> &'static [usize] {
    &PERIODS
}

/// Returns a map of column name -> vector of feature values (NaN during warmup).
pub fn compute_features(prices: &[f64]) -> HashMap<String, Vec<f64>> {
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();

    for &p in PERIODS.iter() {
        out.insert(format!("sma_{p}"), sma(prices, p));
        out.insert(format!("ema_{p}"), ema(prices, p));
        out.insert(format!("hma_{p}"), hma(prices, p));
    }

    out
}

/// Compute features and optionally add t-1 volume (shifted volume).
pub fn compute_features_with_volume(prices: &[f64], volume: Option<&[f64]>) -> HashMap<String, Vec<f64>> {
    let mut out = compute_features(prices);
    if let Some(vol) = volume {
        let mut vol_t1 = vec![f64::NAN; vol.len()];
        if vol.len() > 1 {
            vol_t1[1..].copy_from_slice(&vol[..vol.len() - 1]);
        }
        out.insert("vol_t1".to_string(), vol_t1);
    }
    out
}

/// Compute features from OHLC arrays (close required) and optional volume:
/// - SMA/EMA/HMA over PERIODS
/// - ATR over ATR_PERIODS
/// - vol_t1 if volume provided
pub fn compute_features_ohlcv(
    close: &[f64],
    high: Option<&[f64]>,
    low: Option<&[f64]>,
    volume: Option<&[f64]>,
) -> HashMap<String, Vec<f64>> {
    let mut out = compute_features_with_volume(close, volume);

    if let (Some(h), Some(l)) = (high, low) {
        let tr = true_range(h, l, close);
        for &p in ATR_PERIODS.iter() {
            let atr = atr_wilder(&tr, p);
            out.insert(format!("atr_{p}"), atr);
        }
    }

    out
}

fn sma(prices: &[f64], period: usize) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period == 0 || prices.len() < period {
        return res;
    }
    let mut sum: f64 = prices[..period].iter().sum();
    res[period - 1] = sum / period as f64;
    for i in period..prices.len() {
        sum += prices[i] - prices[i - period];
        res[i] = sum / period as f64;
    }
    res
}

fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period == 0 || prices.len() < period {
        return res;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    // seed with SMA of first period
    let mut prev = prices[..period].iter().sum::<f64>() / period as f64;
    res[period - 1] = prev;
    for i in period..prices.len() {
        let val = alpha * prices[i] + (1.0 - alpha) * prev;
        res[i] = val;
        prev = val;
    }
    res
}

fn wma(prices: &[f64], period: usize) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period == 0 || prices.len() < period {
        return res;
    }
    let denom = (period * (period + 1) / 2) as f64;
    for end in period..=prices.len() {
        let start = end - period;
        let mut num = 0.0;
        for (w, &v) in prices[start..end].iter().enumerate() {
            num += (w + 1) as f64 * v;
        }
        res[end - 1] = num / denom;
    }
    res
}

fn hma(prices: &[f64], period: usize) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period < 2 || prices.len() < period {
        return res;
    }
    let half = period / 2;
    let sqrt_p = (period as f64).sqrt().round() as usize;

    let wma_full = wma(prices, period);
    let wma_half = wma(prices, half);

    // construct series: 2 * WMA(half) - WMA(full)
    let mut diff = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        if wma_half[i].is_nan() || wma_full[i].is_nan() {
            continue;
        }
        diff[i] = 2.0 * wma_half[i] - wma_full[i];
    }

    // HMA is WMA of diff with sqrt(period)
    let hma_vec = wma(&diff, sqrt_p);
    res.copy_from_slice(&hma_vec);
    res
}

fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = close.len();
    let mut tr = vec![f64::NAN; len];
    for i in 0..len {
        let hl = high.get(i).copied().unwrap_or(f64::NAN) - low.get(i).copied().unwrap_or(f64::NAN);
        if i == 0 {
            tr[i] = hl.abs();
        } else {
            let h = high[i];
            let l = low[i];
            let pc = close[i - 1];
            let tr_i = hl
                .abs()
                .max((h - pc).abs())
                .max((l - pc).abs());
            tr[i] = tr_i;
        }
    }
    tr
}

fn atr_wilder(tr: &[f64], period: usize) -> Vec<f64> {
    let mut atr = vec![f64::NAN; tr.len()];
    if tr.len() < period || period == 0 {
        return atr;
    }
    let seed: f64 = tr[..period].iter().sum::<f64>() / period as f64;
    atr[period - 1] = seed;
    for i in period..tr.len() {
        let prev = atr[i - 1];
        atr[i] = (prev * (period as f64 - 1.0) + tr[i]) / period as f64;
    }
    atr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sma_basic() {
        let prices = [1.0, 2.0, 3.0, 4.0];
        let s = sma(&prices, 2);
        assert!(s[0].is_nan());
        assert_eq!(s[1], 1.5);
        assert_eq!(s[2], 2.5);
        assert_eq!(s[3], 3.5);
    }

    #[test]
    fn ema_monotone() {
        let prices = [1.0, 2.0, 3.0, 4.0, 5.0];
        let e = ema(&prices, 3);
        assert!(e.iter().any(|v| !v.is_nan()));
        assert!(e[4] > e[3]);
    }

    #[test]
    fn atr_basic() {
        // Simple ranges: TR should be high-low on first bar, then use prev close impacts
        let high = [5.0, 6.0, 7.0];
        let low = [1.0, 2.0, 3.0];
        let close = [3.0, 4.0, 5.0];
        let tr = true_range(&high, &low, &close);
        assert_eq!(tr.len(), 3);
        assert!(!tr[0].is_nan());
        let atr = atr_wilder(&tr, 2);
        assert_eq!(atr[1].is_nan(), false);
    }
}
