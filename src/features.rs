//! Feature computation: SMA, EMA, HMA, KAMA, and ALMA for a fixed set of periods.
use std::collections::HashMap;

const PERIODS: [usize; 19] = [
    3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53, 100, 150, 200, 250, 300,
];
pub const ATR_PERIODS: [usize; 3] = [7, 14, 21];
pub const KAMA_FAST_PERIOD: usize = 2;
pub const KAMA_SLOW_PERIOD: usize = 30;
pub const ALMA_OFFSET: f64 = 0.85;
pub const ALMA_SIGMA: f64 = 6.0;
pub const RVOL_PERIOD: usize = 20;
pub const CMF_PERIOD: usize = 20;
pub const VWAP_PERIOD: usize = 20;

pub fn periods() -> &'static [usize] {
    &PERIODS
}

/// Minimum observation index `idx` such that all configured features at `idx - 1` are warmed.
pub fn feature_warmup_bars() -> usize {
    let ma_warmup = PERIODS
        .iter()
        .copied()
        .map(hma_warmup_bars)
        .max()
        .unwrap_or(1);
    let atr_warmup = ATR_PERIODS.iter().copied().max().unwrap_or(1);
    let volume_warmup = RVOL_PERIOD.max(CMF_PERIOD).max(VWAP_PERIOD);
    ma_warmup.max(atr_warmup).max(volume_warmup).max(1)
}

fn hma_warmup_bars(period: usize) -> usize {
    if period < 2 {
        return period.max(1);
    }
    let sqrt_p = (period as f64).sqrt().round() as usize;
    period.saturating_add(sqrt_p.saturating_sub(1))
}

/// Returns a map of column name -> vector of feature values (NaN during warmup).
pub fn compute_features(prices: &[f64]) -> HashMap<String, Vec<f64>> {
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();

    for &p in PERIODS.iter() {
        out.insert(format!("sma_{p}"), sma(prices, p));
        out.insert(format!("ema_{p}"), ema(prices, p));
        out.insert(format!("hma_{p}"), hma(prices, p));
        out.insert(
            format!("kama_{p}"),
            kama(prices, p, KAMA_FAST_PERIOD, KAMA_SLOW_PERIOD),
        );
        out.insert(
            format!("alma_{p}"),
            alma(prices, p, ALMA_OFFSET, ALMA_SIGMA),
        );
    }

    out
}

/// Compute features and optionally add t-1 volume (shifted volume).
pub fn compute_features_with_volume(
    prices: &[f64],
    volume: Option<&[f64]>,
) -> HashMap<String, Vec<f64>> {
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
/// - RVOL/CMF/VWAP distance/OBV from volume (NaN if volume unavailable)
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

    let len = close.len();
    if let Some(vol) = volume {
        out.insert(format!("rvol_{RVOL_PERIOD}"), rvol(vol, RVOL_PERIOD));
        out.insert("obv".to_string(), obv(close, vol));
        out.insert(
            format!("vwap_dist_{VWAP_PERIOD}"),
            rolling_vwap_distance(close, vol, VWAP_PERIOD),
        );

        let cmf_values = if let (Some(h), Some(l)) = (high, low) {
            cmf(h, l, close, vol, CMF_PERIOD)
        } else {
            vec![f64::NAN; len]
        };
        out.insert(format!("cmf_{CMF_PERIOD}"), cmf_values);
    } else {
        out.insert(format!("rvol_{RVOL_PERIOD}"), vec![f64::NAN; len]);
        out.insert(format!("cmf_{CMF_PERIOD}"), vec![f64::NAN; len]);
        out.insert(format!("vwap_dist_{VWAP_PERIOD}"), vec![f64::NAN; len]);
        out.insert("obv".to_string(), vec![f64::NAN; len]);
    }

    out
}

pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
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

pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
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

pub fn wma(prices: &[f64], period: usize) -> Vec<f64> {
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

pub fn hma(prices: &[f64], period: usize) -> Vec<f64> {
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

/// Kaufman's Adaptive Moving Average (KAMA).
/// - `period` controls the efficiency-ratio lookback.
/// - `fast_period` and `slow_period` control smoothing limits.
pub fn kama(prices: &[f64], period: usize, fast_period: usize, slow_period: usize) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period == 0 || fast_period == 0 || slow_period == 0 || slow_period <= fast_period {
        return res;
    }
    if prices.len() < period {
        return res;
    }

    let fast_sc = 2.0 / (fast_period as f64 + 1.0);
    let slow_sc = 2.0 / (slow_period as f64 + 1.0);

    let seed = prices[..period].iter().sum::<f64>() / period as f64;
    res[period - 1] = seed;
    let mut prev_kama = seed;

    for i in period..prices.len() {
        let change = (prices[i] - prices[i - period]).abs();
        let mut volatility = 0.0;
        for j in (i - period + 1)..=i {
            volatility += (prices[j] - prices[j - 1]).abs();
        }
        let efficiency_ratio = if volatility > 0.0 {
            change / volatility
        } else {
            0.0
        };
        let smoothing = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc).powi(2);
        let current = prev_kama + smoothing * (prices[i] - prev_kama);
        res[i] = current;
        prev_kama = current;
    }

    res
}

/// Arnaud Legoux Moving Average (ALMA).
/// - `period` is the moving window length.
/// - `offset` should be in [0, 1], typically 0.85.
/// - `sigma` controls Gaussian width, typically 6.0.
pub fn alma(prices: &[f64], period: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let mut res = vec![f64::NAN; prices.len()];
    if period == 0 || prices.len() < period {
        return res;
    }
    if sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
        return res;
    }

    let m = offset * (period as f64 - 1.0);
    let s = period as f64 / sigma;
    if s <= 0.0 {
        return res;
    }

    let mut weights = vec![0.0; period];
    let mut weight_sum = 0.0;
    for (i, w) in weights.iter_mut().enumerate() {
        let dist = i as f64 - m;
        let weight = (-(dist * dist) / (2.0 * s * s)).exp();
        *w = weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return res;
    }

    for end in period..=prices.len() {
        let start = end - period;
        let mut acc = 0.0;
        for i in 0..period {
            acc += weights[i] * prices[start + i];
        }
        res[end - 1] = acc / weight_sum;
    }

    res
}

/// Relative volume: current volume divided by SMA(volume, period).
pub fn rvol(volume: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; volume.len()];
    let vol_sma = sma(volume, period);
    for i in 0..volume.len() {
        let denom = vol_sma[i];
        if denom.is_finite() && denom.abs() > 1e-12 && volume[i].is_finite() {
            out[i] = volume[i] / denom;
        }
    }
    out
}

/// Chaikin Money Flow over `period`.
pub fn cmf(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if period == 0 || len < period || high.len() < len || low.len() < len || volume.len() < len {
        return out;
    }

    let mut mfv = vec![f64::NAN; len];
    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];
        if !(h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite()) {
            continue;
        }
        let range = h - l;
        if range.abs() <= 1e-12 {
            mfv[i] = 0.0;
            continue;
        }
        let multiplier = ((c - l) - (h - c)) / range;
        mfv[i] = multiplier * v;
    }

    let mut mfv_sum = 0.0;
    let mut vol_sum = 0.0;
    for i in 0..len {
        mfv_sum += mfv[i];
        vol_sum += volume[i];
        if i >= period {
            mfv_sum -= mfv[i - period];
            vol_sum -= volume[i - period];
        }
        if i + 1 >= period && vol_sum.abs() > 1e-12 {
            out[i] = mfv_sum / vol_sum;
        }
    }
    out
}

/// Rolling VWAP distance as a normalized percentage delta from rolling VWAP:
/// (close - rolling_vwap) / rolling_vwap.
pub fn rolling_vwap_distance(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if period == 0 || len < period || volume.len() < len {
        return out;
    }

    let mut pv_sum = 0.0;
    let mut vol_sum = 0.0;
    for i in 0..len {
        let c = close[i];
        let v = volume[i];
        if c.is_finite() && v.is_finite() {
            pv_sum += c * v;
            vol_sum += v;
        }
        if i >= period {
            let c_prev = close[i - period];
            let v_prev = volume[i - period];
            if c_prev.is_finite() && v_prev.is_finite() {
                pv_sum -= c_prev * v_prev;
                vol_sum -= v_prev;
            }
        }

        if i + 1 >= period && vol_sum.abs() > 1e-12 {
            let vwap = pv_sum / vol_sum;
            if vwap.abs() > 1e-12 {
                out[i] = (c - vwap) / vwap;
            }
        }
    }
    out
}

/// On-Balance Volume.
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if len == 0 || volume.len() < len {
        return out;
    }

    let mut cumulative = 0.0;
    out[0] = 0.0;
    for i in 1..len {
        let c = close[i];
        let prev_c = close[i - 1];
        let v = volume[i];
        if !(c.is_finite() && prev_c.is_finite() && v.is_finite()) {
            out[i] = cumulative;
            continue;
        }
        if c > prev_c {
            cumulative += v;
        } else if c < prev_c {
            cumulative -= v;
        }
        out[i] = cumulative;
    }
    out
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
            let tr_i = hl.abs().max((h - pc).abs()).max((l - pc).abs());
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

    #[test]
    fn warmup_covers_long_hma_periods() {
        assert_eq!(feature_warmup_bars(), 316);
    }

    #[test]
    fn kama_warms_and_tracks_prices() {
        let prices = [1.0, 1.2, 1.1, 1.3, 1.4, 1.6, 1.7, 1.9];
        let out = kama(&prices, 3, 2, 30);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!(!out[2].is_nan());
        assert!(!out[7].is_nan());
        assert!(out[7].is_finite());
    }

    #[test]
    fn alma_warms_and_tracks_prices() {
        let prices = [1.0, 1.2, 1.1, 1.3, 1.4, 1.6, 1.7, 1.9];
        let out = alma(&prices, 4, 0.85, 6.0);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!(out[2].is_nan());
        assert!(!out[3].is_nan());
        assert!(!out[7].is_nan());
        assert!(out[7].is_finite());
    }

    #[test]
    fn compute_features_includes_kama_and_alma_columns() {
        let prices = [1.0, 1.2, 1.1, 1.3, 1.4, 1.6, 1.7, 1.9];
        let out = compute_features(&prices);
        assert!(out.contains_key("kama_3"));
        assert!(out.contains_key("alma_3"));
    }

    #[test]
    fn compute_features_ohlcv_includes_volume_indicators() {
        let close: Vec<f64> = (1..=40).map(|v| v as f64).collect();
        let high: Vec<f64> = close.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 1.0).collect();
        let volume: Vec<f64> = (1..=40).map(|v| 100.0 + v as f64).collect();
        let out = compute_features_ohlcv(&close, Some(&high), Some(&low), Some(&volume));

        assert!(out.contains_key("rvol_20"));
        assert!(out.contains_key("cmf_20"));
        assert!(out.contains_key("vwap_dist_20"));
        assert!(out.contains_key("obv"));

        assert!(out["rvol_20"][39].is_finite());
        assert!(out["cmf_20"][39].is_finite());
        assert!(out["vwap_dist_20"][39].is_finite());
        assert!(out["obv"][39].is_finite());
    }

    #[test]
    fn compute_features_ohlcv_volume_indicators_nan_without_volume() {
        let close: Vec<f64> = (1..=40).map(|v| v as f64).collect();
        let high: Vec<f64> = close.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 1.0).collect();
        let out = compute_features_ohlcv(&close, Some(&high), Some(&low), None);

        assert!(out["rvol_20"].iter().all(|v| v.is_nan()));
        assert!(out["cmf_20"].iter().all(|v| v.is_nan()));
        assert!(out["vwap_dist_20"].iter().all(|v| v.is_nan()));
        assert!(out["obv"].iter().all(|v| v.is_nan()));
    }
}
