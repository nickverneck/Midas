use crate::strategies::{PositionSide, StrategySignal};
use crate::tradovate::Bar;

#[derive(Debug, Clone)]
pub struct HmaAngleConfig {
    pub hma_length: usize,
    pub min_angle: f64,
    pub angle_lookback: usize,
    pub bars_required_to_trade: usize,
    pub longs_only: bool,
    pub inverted: bool,
}

impl Default for HmaAngleConfig {
    fn default() -> Self {
        Self {
            hma_length: 255,
            min_angle: 7.0,
            angle_lookback: 7,
            bars_required_to_trade: 50,
            longs_only: false,
            inverted: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HmaAngleEvaluation {
    pub signal: StrategySignal,
    pub angle: Option<f64>,
    pub latest_close: Option<f64>,
    pub latest_hma: Option<f64>,
}

impl HmaAngleEvaluation {
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Signal: {}", self.signal.label())];
        if let Some(angle) = self.angle {
            parts.push(format!("Angle: {:.2}", angle));
        }
        if let Some(close) = self.latest_close {
            parts.push(format!("Close: {:.2}", close));
        }
        if let Some(hma) = self.latest_hma {
            parts.push(format!("HMA: {:.2}", hma));
        }
        parts.join(" | ")
    }
}

impl HmaAngleConfig {
    pub fn warmup_bars(&self) -> usize {
        let sqrt_length = (self.hma_length as f64).sqrt().round() as usize;
        (self.hma_length.saturating_add(sqrt_length)).max(self.bars_required_to_trade)
    }

    pub fn evaluate(&self, bars: &[Bar], current_side: Option<PositionSide>) -> HmaAngleEvaluation {
        let warmup_bars = self.warmup_bars();
        let Some(last_bar) = bars.last() else {
            return HmaAngleEvaluation {
                signal: StrategySignal::Hold,
                angle: None,
                latest_close: None,
                latest_hma: None,
            };
        };

        if bars.len() < 2 {
            return HmaAngleEvaluation {
                signal: StrategySignal::Hold,
                angle: None,
                latest_close: Some(last_bar.close),
                latest_hma: None,
            };
        }

        let close = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let high = bars.iter().map(|bar| bar.high).collect::<Vec<_>>();
        let low = bars.iter().map(|bar| bar.low).collect::<Vec<_>>();
        let atr_14 = atr_wilder(&true_range(&high, &low, &close), 14);
        let zero_hma = compute_zero_lag_hma(&close, self.hma_length);
        let idx = close.len() - 1;

        if idx < warmup_bars
            || idx < self.angle_lookback
            || !close[idx].is_finite()
            || !zero_hma[idx].is_finite()
            || !zero_hma[idx.saturating_sub(1)].is_finite()
        {
            return HmaAngleEvaluation {
                signal: StrategySignal::Hold,
                angle: None,
                latest_close: Some(last_bar.close),
                latest_hma: zero_hma.get(idx).copied().filter(|value| value.is_finite()),
            };
        }

        let atr = atr_14.get(idx).copied().unwrap_or(f64::NAN);
        let lookback_hma = zero_hma[idx - self.angle_lookback];
        if !atr.is_finite()
            || atr.abs() < f64::EPSILON
            || !lookback_hma.is_finite()
            || lookback_hma.abs() < f64::EPSILON
        {
            return HmaAngleEvaluation {
                signal: StrategySignal::Hold,
                angle: None,
                latest_close: Some(last_bar.close),
                latest_hma: Some(zero_hma[idx]),
            };
        }

        let prev_close = close[idx - 1];
        let curr_close = close[idx];
        let prev_hma = zero_hma[idx - 1];
        let curr_hma = zero_hma[idx];

        let price_change = curr_hma - lookback_hma;
        let slope = price_change / (atr * self.angle_lookback as f64);
        let angle = slope.atan().to_degrees();
        let is_steep_enough = angle.abs() >= self.min_angle;

        let mut buy_signal = cross_above(prev_close, prev_hma, curr_close, curr_hma)
            && is_steep_enough
            && angle > 0.0;
        let mut sell_signal = cross_below(prev_close, prev_hma, curr_close, curr_hma)
            && is_steep_enough
            && angle < 0.0;

        if self.inverted {
            std::mem::swap(&mut buy_signal, &mut sell_signal);
        }

        let signal = resolve_signal(buy_signal, sell_signal, current_side, self.longs_only);

        HmaAngleEvaluation {
            signal,
            angle: Some(angle),
            latest_close: Some(curr_close),
            latest_hma: Some(curr_hma),
        }
    }
}

fn resolve_signal(
    buy_signal: bool,
    sell_signal: bool,
    current_side: Option<PositionSide>,
    longs_only: bool,
) -> StrategySignal {
    if buy_signal && current_side != Some(PositionSide::Long) {
        return StrategySignal::EnterLong;
    }

    if sell_signal {
        if longs_only {
            if current_side == Some(PositionSide::Long) {
                return StrategySignal::ExitLongOnShortSignal;
            }
        } else if current_side != Some(PositionSide::Short) {
            return StrategySignal::EnterShort;
        }
    }

    StrategySignal::Hold
}

fn cross_above(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a <= prev_b && curr_a > curr_b
}

fn cross_below(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a >= prev_b && curr_a < curr_b
}

fn compute_zero_lag_hma(close: &[f64], hma_length: usize) -> Vec<f64> {
    if close.is_empty() || hma_length == 0 {
        return Vec::new();
    }

    let half_length = ((hma_length as f64) / 2.0).ceil() as usize;
    let sqrt_length = (hma_length as f64).sqrt().round() as usize;
    let wma_half = wma(close, half_length.max(1));
    let wma_full = wma(close, hma_length.max(1));

    let mut out = vec![f64::NAN; close.len()];
    for idx in 0..close.len() {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..sqrt_length {
            if idx < i {
                break;
            }
            let hist_idx = idx - i;
            let wma1 = wma_half[hist_idx];
            let wma2 = wma_full[hist_idx];
            if !wma1.is_finite() || !wma2.is_finite() {
                continue;
            }

            let weight = (sqrt_length - i) as f64;
            let diff = 2.0 * wma1 - wma2;
            sum += diff * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            out[idx] = sum / weight_sum;
        }
    }

    out
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
        for (weight, &value) in prices[start..end].iter().enumerate() {
            num += (weight + 1) as f64 * value;
        }
        res[end - 1] = num / denom;
    }
    res
}

fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = close.len();
    let mut tr = vec![f64::NAN; len];
    for idx in 0..len {
        let hl =
            high.get(idx).copied().unwrap_or(f64::NAN) - low.get(idx).copied().unwrap_or(f64::NAN);
        if idx == 0 {
            tr[idx] = hl.abs();
        } else {
            let high_value = high[idx];
            let low_value = low[idx];
            let prev_close = close[idx - 1];
            tr[idx] = hl
                .abs()
                .max((high_value - prev_close).abs())
                .max((low_value - prev_close).abs());
        }
    }
    tr
}

fn atr_wilder(tr: &[f64], period: usize) -> Vec<f64> {
    let mut atr = vec![f64::NAN; tr.len()];
    if tr.len() < period || period == 0 {
        return atr;
    }

    let seed = tr[..period].iter().sum::<f64>() / period as f64;
    atr[period - 1] = seed;
    for idx in period..tr.len() {
        let prev = atr[idx - 1];
        atr[idx] = (prev * (period as f64 - 1.0) + tr[idx]) / period as f64;
    }
    atr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_signal_enters_short_when_inverted_sell_is_active() {
        let signal = resolve_signal(false, true, None, false);
        assert_eq!(signal, StrategySignal::EnterShort);
    }

    #[test]
    fn resolve_signal_exits_long_when_longs_only_and_sell_signal_hits() {
        let signal = resolve_signal(false, true, Some(PositionSide::Long), true);
        assert_eq!(signal, StrategySignal::ExitLongOnShortSignal);
    }

    #[test]
    fn warmup_uses_hma_length_plus_sqrt() {
        let config = HmaAngleConfig {
            hma_length: 16,
            bars_required_to_trade: 5,
            ..HmaAngleConfig::default()
        };

        assert_eq!(config.warmup_bars(), 20);
    }
}
