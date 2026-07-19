use crate::broker::Bar;
use crate::strategies::{PositionSide, StrategySignal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HmaCrossConfig {
    pub fast_length: usize,
    pub slow_length: usize,
    pub inverted: bool,
    pub take_profit_ticks: f64,
    pub stop_loss_ticks: f64,
    pub use_trailing_stop: bool,
    pub trail_trigger_ticks: f64,
    pub trail_offset_ticks: f64,
}

impl Default for HmaCrossConfig {
    fn default() -> Self {
        Self {
            fast_length: 21,
            slow_length: 55,
            inverted: false,
            take_profit_ticks: 0.0,
            stop_loss_ticks: 0.0,
            use_trailing_stop: false,
            trail_trigger_ticks: 12.0,
            trail_offset_ticks: 8.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HmaCrossEvaluation {
    pub signal: StrategySignal,
    pub latest_close: Option<f64>,
    pub previous_fast_hma: Option<f64>,
    pub previous_slow_hma: Option<f64>,
    pub fast_hma: Option<f64>,
    pub slow_hma: Option<f64>,
    pub previous_observed_side: Option<HmaCrossSide>,
    pub observed_side: Option<HmaCrossSide>,
    pub bars_len: usize,
    pub warmup_bars: usize,
    pub current_side: Option<PositionSide>,
    pub inverted: bool,
    pub raw_buy_signal: bool,
    pub raw_sell_signal: bool,
    pub effective_buy_signal: bool,
    pub effective_sell_signal: bool,
    pub current_bar_side_edge: bool,
    pub hold_reason: Option<&'static str>,
}

impl HmaCrossEvaluation {
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Signal: {}", self.signal.label())];
        if let Some(close) = self.latest_close {
            parts.push(format!("Close: {:.2}", close));
        }
        if let (Some(prev_fast), Some(prev_slow), Some(fast), Some(slow)) = (
            self.previous_fast_hma,
            self.previous_slow_hma,
            self.fast_hma,
            self.slow_hma,
        ) {
            parts.push(format!(
                "HMA Delta: {:.2}->{:.2}",
                prev_fast - prev_slow,
                fast - slow
            ));
            parts.push(format!("Prev Fast HMA: {:.2}", prev_fast));
            parts.push(format!("Prev Slow HMA: {:.2}", prev_slow));
        }
        if let Some(fast) = self.fast_hma {
            parts.push(format!("Fast HMA: {:.2}", fast));
        }
        if let Some(slow) = self.slow_hma {
            parts.push(format!("Slow HMA: {:.2}", slow));
        }
        if let Some(side) = self.observed_side {
            parts.push(format!("HMA Side: {}", side.label()));
            parts.push(format!(
                "Prior HMA Side: {}",
                self.previous_observed_side
                    .map(HmaCrossSide::label)
                    .unwrap_or("unset")
            ));
        }
        parts.join(" | ")
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "Signal: {} | reason: {} | bars: {}/{} | side: {:?} | inverted: {} | close: {} | prev_fast_hma: {} | prev_slow_hma: {} | fast_hma: {} | slow_hma: {} | delta: {}->{} | raw_cross buy={} sell={} | effective_cross buy={} sell={} | prior_hma_side: {} | hma_side: {} | Prior HMA Side: {} | HMA Side: {} | current_bar_side_edge: {}",
            self.signal.label(),
            self.hold_reason.unwrap_or("signal_ready"),
            self.bars_len,
            self.warmup_bars,
            self.current_side,
            self.inverted,
            fmt_price(self.latest_close),
            fmt_price(self.previous_fast_hma),
            fmt_price(self.previous_slow_hma),
            fmt_price(self.fast_hma),
            fmt_price(self.slow_hma),
            fmt_delta(self.previous_fast_hma, self.previous_slow_hma),
            fmt_delta(self.fast_hma, self.slow_hma),
            self.raw_buy_signal,
            self.raw_sell_signal,
            self.effective_buy_signal,
            self.effective_sell_signal,
            self.previous_observed_side
                .map(HmaCrossSide::label)
                .unwrap_or("unset"),
            self.observed_side
                .map(HmaCrossSide::label)
                .unwrap_or("unset"),
            self.previous_observed_side
                .map(HmaCrossSide::label)
                .unwrap_or("unset"),
            self.observed_side
                .map(HmaCrossSide::label)
                .unwrap_or("unset"),
            self.current_bar_side_edge
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmaCrossSide {
    Above,
    Below,
}

impl HmaCrossSide {
    pub fn label(self) -> &'static str {
        match self {
            Self::Above => "fast>slow",
            Self::Below => "fast<slow",
        }
    }

    fn from_values(fast: f64, slow: f64) -> Option<Self> {
        if fast > slow {
            Some(Self::Above)
        } else if fast < slow {
            Some(Self::Below)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HmaCrossExecutionState {
    pub position: Option<HmaCrossManagedPosition>,
    pub last_observed_side: Option<HmaCrossSide>,
    pub last_observed_bar_ts: Option<i64>,
    pub last_observed_delta: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct HmaCrossManagedPosition {
    pub side: PositionSide,
    pub qty: i32,
    pub entry_price: f64,
    pub best_price: f64,
    pub current_stop_price: Option<f64>,
    pub trailing_active: bool,
}

impl HmaCrossConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.take_profit_ticks > 0.0 || self.stop_loss_ticks > 0.0 || self.use_trailing_stop
    }

    pub fn warmup_bars(&self) -> usize {
        hma_warmup_bars(self.fast_length).max(hma_warmup_bars(self.slow_length)) + 1
    }

    pub fn evaluate(&self, bars: &[Bar], current_side: Option<PositionSide>) -> HmaCrossEvaluation {
        let Some(last_bar) = bars.last() else {
            return HmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: None,
                previous_fast_hma: None,
                previous_slow_hma: None,
                fast_hma: None,
                slow_hma: None,
                previous_observed_side: None,
                observed_side: None,
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                current_bar_side_edge: false,
                hold_reason: Some("no_bars"),
            };
        };

        if bars.len() < self.warmup_bars() {
            return HmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                previous_fast_hma: None,
                previous_slow_hma: None,
                fast_hma: None,
                slow_hma: None,
                previous_observed_side: None,
                observed_side: None,
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                current_bar_side_edge: false,
                hold_reason: Some("warming_up"),
            };
        }

        let close = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let fast = hma_series(&close, self.fast_length.max(1));
        let slow = hma_series(&close, self.slow_length.max(1));
        let idx = close.len() - 1;
        let prev_idx = idx.saturating_sub(1);

        let prev_fast = fast.get(prev_idx).copied().unwrap_or(f64::NAN);
        let curr_fast = fast.get(idx).copied().unwrap_or(f64::NAN);
        let prev_slow = slow.get(prev_idx).copied().unwrap_or(f64::NAN);
        let curr_slow = slow.get(idx).copied().unwrap_or(f64::NAN);

        if !prev_fast.is_finite()
            || !curr_fast.is_finite()
            || !prev_slow.is_finite()
            || !curr_slow.is_finite()
        {
            return HmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                previous_fast_hma: fast
                    .get(prev_idx)
                    .copied()
                    .filter(|value| value.is_finite()),
                previous_slow_hma: slow
                    .get(prev_idx)
                    .copied()
                    .filter(|value| value.is_finite()),
                fast_hma: fast.get(idx).copied().filter(|value| value.is_finite()),
                slow_hma: slow.get(idx).copied().filter(|value| value.is_finite()),
                previous_observed_side: None,
                observed_side: None,
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                current_bar_side_edge: false,
                hold_reason: Some("non_finite_indicator"),
            };
        }

        let raw_buy_signal = prev_fast <= prev_slow && curr_fast > curr_slow;
        let raw_sell_signal = prev_fast >= prev_slow && curr_fast < curr_slow;
        let mut buy_signal = raw_buy_signal;
        let mut sell_signal = raw_sell_signal;
        if self.inverted {
            std::mem::swap(&mut buy_signal, &mut sell_signal);
        }

        let signal = resolve_signal(buy_signal, sell_signal, current_side);
        let hold_reason = if signal != StrategySignal::Hold {
            None
        } else if buy_signal && current_side == Some(PositionSide::Long) {
            Some("buy_cross_already_long")
        } else if sell_signal && current_side == Some(PositionSide::Short) {
            Some("sell_cross_already_short")
        } else if !buy_signal && !sell_signal {
            Some("no_effective_cross")
        } else {
            Some("hold")
        };
        HmaCrossEvaluation {
            signal,
            latest_close: Some(last_bar.close),
            previous_fast_hma: Some(prev_fast),
            previous_slow_hma: Some(prev_slow),
            fast_hma: Some(curr_fast),
            slow_hma: Some(curr_slow),
            previous_observed_side: None,
            observed_side: None,
            bars_len: bars.len(),
            warmup_bars: self.warmup_bars(),
            current_side,
            inverted: self.inverted,
            raw_buy_signal,
            raw_sell_signal,
            effective_buy_signal: buy_signal,
            effective_sell_signal: sell_signal,
            current_bar_side_edge: false,
            hold_reason,
        }
    }

    pub fn evaluate_current_cross(
        &self,
        runtime: &mut HmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> HmaCrossEvaluation {
        let mut evaluation = self.evaluate(bars, current_side);

        let Some(last_bar) = bars.last() else {
            return evaluation;
        };
        let (Some(fast), Some(slow)) = (evaluation.fast_hma, evaluation.slow_hma) else {
            return evaluation;
        };
        let Some(observed_side) = HmaCrossSide::from_values(fast, slow) else {
            return evaluation;
        };

        let expected_previous_bar_ts = bars.get(bars.len().saturating_sub(2)).map(|bar| bar.ts_ns);
        let stored_side_is_previous_bar = runtime.last_observed_bar_ts == expected_previous_bar_ts;
        let stored_side_is_current_bar = runtime.last_observed_bar_ts == Some(last_bar.ts_ns);
        let previous_side = runtime
            .last_observed_side
            .filter(|_| stored_side_is_previous_bar || stored_side_is_current_bar);
        let desired_side = self.desired_position_side(observed_side);
        let current_bar_side_edge = previous_side.is_some_and(|side| side != observed_side);

        evaluation.previous_observed_side = previous_side;
        evaluation.observed_side = Some(observed_side);
        evaluation.current_bar_side_edge = current_bar_side_edge;
        if current_bar_side_edge {
            evaluation.signal = match desired_side {
                PositionSide::Long => resolve_signal(true, false, current_side),
                PositionSide::Short => resolve_signal(false, true, current_side),
            };
            evaluation.hold_reason = if evaluation.signal == StrategySignal::Hold {
                Some("stateful_side_edge_already_current")
            } else {
                None
            };
        }

        runtime.last_observed_side = Some(observed_side);
        runtime.last_observed_bar_ts = Some(last_bar.ts_ns);
        runtime.last_observed_delta = Some(fast - slow);
        evaluation
    }

    fn desired_position_side(&self, hma_side: HmaCrossSide) -> PositionSide {
        match (self.inverted, hma_side) {
            (false, HmaCrossSide::Above) | (true, HmaCrossSide::Below) => PositionSide::Long,
            (false, HmaCrossSide::Below) | (true, HmaCrossSide::Above) => PositionSide::Short,
        }
    }

    pub fn sync_position(
        &self,
        runtime: &mut HmaCrossExecutionState,
        signed_qty: i32,
        entry_price: Option<f64>,
    ) {
        let Some(side) = signed_qty_to_side(signed_qty) else {
            runtime.position = None;
            return;
        };

        let qty = signed_qty.abs().max(1);
        let broker_entry = entry_price.filter(|price| price.is_finite() && *price > 0.0);

        match runtime.position.as_mut() {
            Some(position) if position.side == side => {
                position.qty = qty;
                if let Some(price) = broker_entry {
                    if (position.entry_price - price).abs() > 1e-6 {
                        position.entry_price = price;
                        position.best_price = price;
                        position.current_stop_price = None;
                        position.trailing_active = false;
                    }
                }
            }
            _ => {
                let Some(price) = broker_entry else {
                    runtime.position = None;
                    return;
                };
                runtime.position = Some(HmaCrossManagedPosition {
                    side,
                    qty,
                    entry_price: price,
                    best_price: price,
                    current_stop_price: None,
                    trailing_active: false,
                });
            }
        }
    }

    pub fn take_profit_offset(&self, tick_size: Option<f64>) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        if self.take_profit_ticks <= 0.0 {
            return None;
        }
        Some(self.take_profit_ticks * tick_size)
    }

    pub fn desired_trailing_stop_price(
        &self,
        runtime: &mut HmaCrossExecutionState,
        bar: &Bar,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_mut()?;
        if !self.use_trailing_stop
            || self.trail_trigger_ticks <= 0.0
            || self.trail_offset_ticks < 0.0
        {
            return None;
        }

        let favorable_price = match position.side {
            PositionSide::Long => bar.high,
            PositionSide::Short => bar.low,
        };
        if favorable_price.is_finite() {
            match position.side {
                PositionSide::Long => {
                    if favorable_price > position.best_price {
                        position.best_price = favorable_price;
                    }
                }
                PositionSide::Short => {
                    if favorable_price < position.best_price {
                        position.best_price = favorable_price;
                    }
                }
            }
        }

        let current_pnl_ticks = match position.side {
            PositionSide::Long => (bar.close - position.entry_price) / tick_size,
            PositionSide::Short => (position.entry_price - bar.close) / tick_size,
        };
        let favorable_ticks = match position.side {
            PositionSide::Long => (position.best_price - position.entry_price) / tick_size,
            PositionSide::Short => (position.entry_price - position.best_price) / tick_size,
        };
        if favorable_ticks < self.trail_trigger_ticks {
            if current_pnl_ticks < 0.0 {
                position.best_price = position.entry_price;
            }
            return None;
        }

        position.trailing_active = true;
        let candidate = match position.side {
            PositionSide::Long => position.best_price - self.trail_offset_ticks * tick_size,
            PositionSide::Short => position.best_price + self.trail_offset_ticks * tick_size,
        };
        let next_stop = match (position.side, position.current_stop_price) {
            (PositionSide::Long, Some(current)) => current.max(candidate),
            (PositionSide::Short, Some(current)) => current.min(candidate),
            (_, None) => candidate,
        };
        position.current_stop_price = Some(next_stop);
        Some(next_stop)
    }

    pub fn current_effective_stop_price(
        &self,
        runtime: &HmaCrossExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_ref()?;
        self.effective_stop_price(position, tick_size)
    }

    fn stop_loss_price(&self, position: &HmaCrossManagedPosition, tick_size: f64) -> Option<f64> {
        if self.stop_loss_ticks <= 0.0 {
            return None;
        }
        Some(match position.side {
            PositionSide::Long => position.entry_price - self.stop_loss_ticks * tick_size,
            PositionSide::Short => position.entry_price + self.stop_loss_ticks * tick_size,
        })
    }

    fn effective_stop_price(
        &self,
        position: &HmaCrossManagedPosition,
        tick_size: f64,
    ) -> Option<f64> {
        let fixed = self.stop_loss_price(position, tick_size);
        let trailing = position.current_stop_price;
        match (fixed, trailing, position.side) {
            (None, None, _) => None,
            (Some(price), None, _) => Some(price),
            (None, Some(price), _) => Some(price),
            (Some(fixed_price), Some(trail_price), PositionSide::Long) => {
                Some(trail_price.max(fixed_price))
            }
            (Some(fixed_price), Some(trail_price), PositionSide::Short) => {
                Some(trail_price.min(fixed_price))
            }
        }
    }
}

fn hma_warmup_bars(length: usize) -> usize {
    let length = length.max(1);
    let sqrt_len = (length as f64).sqrt().floor().max(1.0) as usize;
    length + sqrt_len
}

pub(crate) fn hma_series(values: &[f64], length: usize) -> Vec<f64> {
    let length = length.max(1);
    let half_length = (length / 2).max(1);
    let sqrt_length = (length as f64).sqrt().floor().max(1.0) as usize;
    let wma_half = wma(values, half_length);
    let wma_full = wma(values, length);
    let diff = wma_half
        .iter()
        .zip(wma_full.iter())
        .map(|(half, full)| {
            if half.is_finite() && full.is_finite() {
                2.0 * half - full
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    wma(&diff, sqrt_length)
}

fn wma(values: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if period == 0 || values.len() < period {
        return out;
    }

    let denom = (period * (period + 1) / 2) as f64;
    for end in period..=values.len() {
        let start = end - period;
        let mut weighted_sum = 0.0;
        let mut valid = true;
        for (weight, value) in values[start..end].iter().copied().enumerate() {
            if !value.is_finite() {
                valid = false;
                break;
            }
            weighted_sum += (weight + 1) as f64 * value;
        }
        if valid {
            out[end - 1] = weighted_sum / denom;
        }
    }
    out
}

fn signed_qty_to_side(signed_qty: i32) -> Option<PositionSide> {
    if signed_qty > 0 {
        Some(PositionSide::Long)
    } else if signed_qty < 0 {
        Some(PositionSide::Short)
    } else {
        None
    }
}

fn fmt_price(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_delta(fast: Option<f64>, slow: Option<f64>) -> String {
    match (fast, slow) {
        (Some(fast), Some(slow)) => format!("{:.6}", fast - slow),
        _ => "n/a".to_string(),
    }
}

fn resolve_signal(
    buy_signal: bool,
    sell_signal: bool,
    current_side: Option<PositionSide>,
) -> StrategySignal {
    if buy_signal && current_side != Some(PositionSide::Long) {
        return StrategySignal::EnterLong;
    }

    if sell_signal && current_side != Some(PositionSide::Short) {
        return StrategySignal::EnterShort;
    }

    StrategySignal::Hold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
        }
    }

    #[test]
    fn hma_cross_emits_buy_after_fast_crosses_above_slow() {
        let config = HmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..HmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 10.0),
            bar(6, 8.0),
            bar(7, 12.0),
        ];

        let evaluation = config.evaluate(&bars, None);
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
    }

    #[test]
    fn current_cross_does_not_align_late_against_opposite_position() {
        let config = HmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..HmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 10.0),
            bar(6, 8.0),
            bar(7, 12.0),
            bar(8, 13.0),
        ];
        let mut runtime = HmaCrossExecutionState::default();

        let evaluation =
            config.evaluate_current_cross(&mut runtime, &bars, Some(PositionSide::Short));

        assert_eq!(evaluation.observed_side, Some(HmaCrossSide::Above));
        assert_eq!(evaluation.signal, StrategySignal::Hold);
        assert_eq!(runtime.last_observed_side, Some(HmaCrossSide::Above));
    }

    #[test]
    fn current_cross_uses_immediately_previous_observed_side_for_next_edge() {
        let config = HmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..HmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 10.0),
            bar(6, 8.0),
            bar(7, 12.0),
        ];
        let mut runtime = HmaCrossExecutionState {
            last_observed_side: Some(HmaCrossSide::Below),
            last_observed_bar_ts: Some(6),
            ..HmaCrossExecutionState::default()
        };

        let evaluation =
            config.evaluate_current_cross(&mut runtime, &bars, Some(PositionSide::Short));

        assert_eq!(evaluation.previous_observed_side, Some(HmaCrossSide::Below));
        assert_eq!(evaluation.observed_side, Some(HmaCrossSide::Above));
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
        assert_eq!(runtime.last_observed_side, Some(HmaCrossSide::Above));

        let debug = evaluation.debug_summary();
        assert!(debug.contains("Signal: Buy"));
        assert!(debug.contains("prior_hma_side: fast<slow"));
        assert!(debug.contains("hma_side: fast>slow"));
        assert!(debug.contains("current_bar_side_edge: true"));
        assert!(debug.contains("delta:"));
        assert!(debug.contains("raw_cross buy=true"));
    }

    #[test]
    fn current_cross_uses_same_bar_observed_side_for_revised_edge() {
        let config = HmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..HmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 10.0),
            bar(6, 8.0),
            bar(7, 12.0),
        ];
        let mut runtime = HmaCrossExecutionState {
            last_observed_side: Some(HmaCrossSide::Below),
            last_observed_bar_ts: Some(7),
            ..HmaCrossExecutionState::default()
        };

        let evaluation =
            config.evaluate_current_cross(&mut runtime, &bars, Some(PositionSide::Short));

        assert_eq!(evaluation.previous_observed_side, Some(HmaCrossSide::Below));
        assert_eq!(evaluation.observed_side, Some(HmaCrossSide::Above));
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
        assert_eq!(runtime.last_observed_side, Some(HmaCrossSide::Above));
        assert_eq!(runtime.last_observed_bar_ts, Some(7));
    }

    #[test]
    fn current_cross_ignores_stale_observed_side() {
        let config = HmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..HmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 10.0),
            bar(6, 8.0),
            bar(7, 12.0),
            bar(8, 13.0),
        ];
        let mut runtime = HmaCrossExecutionState {
            last_observed_side: Some(HmaCrossSide::Below),
            last_observed_bar_ts: Some(5),
            ..HmaCrossExecutionState::default()
        };

        let evaluation =
            config.evaluate_current_cross(&mut runtime, &bars, Some(PositionSide::Short));

        assert_eq!(evaluation.previous_observed_side, None);
        assert_eq!(evaluation.observed_side, Some(HmaCrossSide::Above));
        assert_eq!(evaluation.signal, StrategySignal::Hold);
        assert_eq!(runtime.last_observed_side, Some(HmaCrossSide::Above));
    }

    #[test]
    fn protection_offsets_follow_tick_size() {
        let config = HmaCrossConfig {
            take_profit_ticks: 8.0,
            stop_loss_ticks: 6.0,
            ..HmaCrossConfig::default()
        };
        let mut runtime = HmaCrossExecutionState::default();
        config.sync_position(&mut runtime, 1, Some(100.0));

        assert_eq!(config.take_profit_offset(Some(0.25)), Some(2.0));
        assert_eq!(
            config.current_effective_stop_price(&runtime, Some(0.25)),
            Some(98.5)
        );
    }

    #[test]
    fn hma_series_uses_integer_half_and_floor_sqrt_lengths() {
        let values = (1..=30).map(|value| value as f64).collect::<Vec<_>>();
        let hma = hma_series(&values, 21);
        let expected = wma(
            &wma(&values, 10)
                .iter()
                .zip(wma(&values, 21).iter())
                .map(|(half, full)| {
                    if half.is_finite() && full.is_finite() {
                        2.0 * half - full
                    } else {
                        f64::NAN
                    }
                })
                .collect::<Vec<_>>(),
            4,
        );

        assert_eq!(hma.len(), expected.len());
        for (actual, expected) in hma.iter().zip(expected.iter()) {
            if expected.is_finite() {
                assert!((actual - expected).abs() < 1e-12);
            } else {
                assert!(actual.is_nan());
            }
        }
        assert!(hma[22].is_nan());
        assert!(hma[23].is_finite());
    }
}
