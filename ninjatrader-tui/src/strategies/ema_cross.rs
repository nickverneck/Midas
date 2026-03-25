use crate::strategies::{PositionSide, StrategySignal};
use crate::tradovate::Bar;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmaCrossConfig {
    pub fast_length: usize,
    pub slow_length: usize,
    pub inverted: bool,
    pub take_profit_ticks: f64,
    pub stop_loss_ticks: f64,
    pub use_trailing_stop: bool,
    pub trail_trigger_ticks: f64,
    pub trail_offset_ticks: f64,
}

impl Default for EmaCrossConfig {
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
pub struct EmaCrossEvaluation {
    pub signal: StrategySignal,
    pub latest_close: Option<f64>,
    pub fast_ema: Option<f64>,
    pub slow_ema: Option<f64>,
}

impl EmaCrossEvaluation {
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Signal: {}", self.signal.label())];
        if let Some(close) = self.latest_close {
            parts.push(format!("Close: {:.2}", close));
        }
        if let Some(fast) = self.fast_ema {
            parts.push(format!("Fast EMA: {:.2}", fast));
        }
        if let Some(slow) = self.slow_ema {
            parts.push(format!("Slow EMA: {:.2}", slow));
        }
        parts.join(" | ")
    }
}

#[derive(Debug, Clone, Default)]
pub struct EmaCrossExecutionState {
    pub position: Option<EmaManagedPosition>,
}

#[derive(Debug, Clone)]
pub struct EmaManagedPosition {
    pub side: PositionSide,
    pub qty: i32,
    pub entry_price: f64,
    pub best_price: f64,
    pub current_stop_price: Option<f64>,
    pub trailing_active: bool,
}

impl EmaCrossConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.take_profit_ticks > 0.0 || self.stop_loss_ticks > 0.0 || self.use_trailing_stop
    }

    pub fn warmup_bars(&self) -> usize {
        self.fast_length.max(self.slow_length).max(2) + 1
    }

    pub fn evaluate(&self, bars: &[Bar], current_side: Option<PositionSide>) -> EmaCrossEvaluation {
        let Some(last_bar) = bars.last() else {
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: None,
                fast_ema: None,
                slow_ema: None,
            };
        };

        if bars.len() < self.warmup_bars() {
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                fast_ema: None,
                slow_ema: None,
            };
        }

        let close = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let fast = ema(&close, self.fast_length.max(1));
        let slow = ema(&close, self.slow_length.max(1));
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
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                fast_ema: fast.get(idx).copied().filter(|value| value.is_finite()),
                slow_ema: slow.get(idx).copied().filter(|value| value.is_finite()),
            };
        }

        let mut buy_signal = prev_fast <= prev_slow && curr_fast > curr_slow;
        let mut sell_signal = prev_fast >= prev_slow && curr_fast < curr_slow;
        if self.inverted {
            std::mem::swap(&mut buy_signal, &mut sell_signal);
        }

        let signal = resolve_signal(buy_signal, sell_signal, current_side);
        EmaCrossEvaluation {
            signal,
            latest_close: Some(last_bar.close),
            fast_ema: Some(curr_fast),
            slow_ema: Some(curr_slow),
        }
    }

    pub fn sync_position(
        &self,
        runtime: &mut EmaCrossExecutionState,
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
                runtime.position = Some(EmaManagedPosition {
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
        runtime: &mut EmaCrossExecutionState,
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
        runtime: &EmaCrossExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_ref()?;
        self.effective_stop_price(position, tick_size)
    }

    fn stop_loss_price(&self, position: &EmaManagedPosition, tick_size: f64) -> Option<f64> {
        if self.stop_loss_ticks <= 0.0 {
            return None;
        }
        Some(match position.side {
            PositionSide::Long => position.entry_price - self.stop_loss_ticks * tick_size,
            PositionSide::Short => position.entry_price + self.stop_loss_ticks * tick_size,
        })
    }

    fn effective_stop_price(&self, position: &EmaManagedPosition, tick_size: f64) -> Option<f64> {
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

fn ema(values: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if values.is_empty() || period == 0 {
        return out;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut prev = values[0];
    if prev.is_finite() {
        out[0] = prev;
    }
    for (idx, value) in values.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            continue;
        }
        prev = alpha * value + (1.0 - alpha) * prev;
        out[idx] = prev;
    }
    out
}

pub(crate) fn ema_series(values: &[f64], period: usize) -> Vec<f64> {
    ema(values, period)
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
    fn ema_cross_emits_buy_after_fast_crosses_above_slow() {
        let config = EmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..EmaCrossConfig::default()
        };
        let bars = vec![
            bar(1, 10.0),
            bar(2, 10.0),
            bar(3, 10.0),
            bar(4, 10.0),
            bar(5, 8.0),
            bar(6, 12.0),
        ];

        let evaluation = config.evaluate(&bars, None);
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
    }

    #[test]
    fn broker_protection_offsets_follow_tick_size() {
        let config = EmaCrossConfig {
            take_profit_ticks: 8.0,
            stop_loss_ticks: 6.0,
            ..EmaCrossConfig::default()
        };

        assert_eq!(config.take_profit_offset(Some(0.25)), Some(2.0));
        assert_eq!(
            config.current_effective_stop_price(&EmaCrossExecutionState::default(), Some(0.25)),
            None
        );
    }

    #[test]
    fn desired_trailing_stop_price_advances_after_trigger() {
        let config = EmaCrossConfig {
            stop_loss_ticks: 8.0,
            use_trailing_stop: true,
            trail_trigger_ticks: 4.0,
            trail_offset_ticks: 2.0,
            ..EmaCrossConfig::default()
        };
        let mut runtime = EmaCrossExecutionState::default();
        config.sync_position(&mut runtime, 1, Some(100.0));

        let stop_price = config
            .desired_trailing_stop_price(
                &mut runtime,
                &Bar {
                    ts_ns: 1,
                    open: 100.0,
                    high: 101.5,
                    low: 99.75,
                    close: 101.0,
                },
                Some(0.25),
            )
            .expect("trailing stop should activate");

        assert_eq!(stop_price, 101.0);
        assert_eq!(
            config.current_effective_stop_price(&runtime, Some(0.25)),
            Some(101.0)
        );
    }
}
