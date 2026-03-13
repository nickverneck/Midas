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
    pub take_profit_ticks: f64,
    pub stop_loss_ticks: f64,
    pub use_trailing_stop: bool,
    pub trail_trigger_ticks: f64,
    pub trail_offset_ticks: f64,
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
            take_profit_ticks: 0.0,
            stop_loss_ticks: 0.0,
            use_trailing_stop: false,
            trail_trigger_ticks: 12.0,
            trail_offset_ticks: 8.0,
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

#[derive(Debug, Clone, Default)]
pub struct HmaAngleExecutionState {
    pub position: Option<HmaManagedPosition>,
}

#[derive(Debug, Clone)]
pub struct HmaManagedPosition {
    pub side: PositionSide,
    pub qty: i32,
    pub entry_price: f64,
    pub best_price: f64,
    pub current_stop_price: Option<f64>,
    pub trailing_active: bool,
}

#[derive(Debug, Clone)]
pub struct HmaProtectiveExit {
    pub reason: &'static str,
    pub trigger_price: f64,
}

#[derive(Debug, Clone)]
pub struct HmaBrokerProtection {
    pub take_profit_offset: Option<f64>,
    pub stop_loss_offset: Option<f64>,
}

impl HmaAngleConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.take_profit_ticks > 0.0 || self.stop_loss_ticks > 0.0 || self.use_trailing_stop
    }

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

    pub fn sync_position(
        &self,
        runtime: &mut HmaAngleExecutionState,
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
                runtime.position = Some(HmaManagedPosition {
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

    pub fn broker_protection(&self, tick_size: Option<f64>) -> Option<HmaBrokerProtection> {
        let take_profit_offset = self.take_profit_offset(tick_size);
        let stop_loss_offset = self.stop_loss_offset(tick_size);
        if take_profit_offset.is_none() && stop_loss_offset.is_none() {
            return None;
        }
        Some(HmaBrokerProtection {
            take_profit_offset,
            stop_loss_offset,
        })
    }

    pub fn take_profit_offset(&self, tick_size: Option<f64>) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        if self.take_profit_ticks <= 0.0 {
            return None;
        }
        Some(self.take_profit_ticks * tick_size)
    }

    pub fn stop_loss_offset(&self, tick_size: Option<f64>) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        if self.stop_loss_ticks <= 0.0 {
            return None;
        }
        Some(self.stop_loss_ticks * tick_size)
    }

    pub fn desired_trailing_stop_price(
        &self,
        runtime: &mut HmaAngleExecutionState,
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
        runtime: &HmaAngleExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_ref()?;
        self.effective_stop_price(position, tick_size).0
    }

    pub fn evaluate_protective_exit(
        &self,
        runtime: &mut HmaAngleExecutionState,
        bar: &Bar,
        tick_size: Option<f64>,
    ) -> Option<HmaProtectiveExit> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_mut()?;

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
        let mut trailing_changed = false;

        if self.use_trailing_stop
            && self.trail_trigger_ticks > 0.0
            && self.trail_offset_ticks >= 0.0
        {
            let favorable_ticks = match position.side {
                PositionSide::Long => (position.best_price - position.entry_price) / tick_size,
                PositionSide::Short => (position.entry_price - position.best_price) / tick_size,
            };
            if favorable_ticks >= self.trail_trigger_ticks {
                position.trailing_active = true;
                let candidate = match position.side {
                    PositionSide::Long => position.best_price - self.trail_offset_ticks * tick_size,
                    PositionSide::Short => {
                        position.best_price + self.trail_offset_ticks * tick_size
                    }
                };
                let next_stop = match (position.side, position.current_stop_price) {
                    (PositionSide::Long, Some(current)) => Some(current.max(candidate)),
                    (PositionSide::Short, Some(current)) => Some(current.min(candidate)),
                    (_, None) => Some(candidate),
                };
                trailing_changed = next_stop != position.current_stop_price;
                position.current_stop_price = next_stop;
            } else if current_pnl_ticks < 0.0 {
                position.best_price = position.entry_price;
            }
        }

        let take_profit_price = self.take_profit_price(position, tick_size);
        let (stop_price, stop_reason) = self.effective_stop_price(position, tick_size);

        let stop_hit = match (position.side, stop_price, stop_reason) {
            (_, Some(_), Some("trail_stop")) if trailing_changed => false,
            (PositionSide::Long, Some(price), _) if bar.low.is_finite() => bar.low <= price,
            (PositionSide::Short, Some(price), _) if bar.high.is_finite() => bar.high >= price,
            _ => false,
        };
        let take_profit_hit = match (position.side, take_profit_price) {
            (PositionSide::Long, Some(price)) if bar.high.is_finite() => bar.high >= price,
            (PositionSide::Short, Some(price)) if bar.low.is_finite() => bar.low <= price,
            _ => false,
        };

        if stop_hit {
            return stop_price.map(|price| HmaProtectiveExit {
                reason: stop_reason.unwrap_or("stop_loss"),
                trigger_price: price,
            });
        }

        if take_profit_hit {
            return take_profit_price.map(|price| HmaProtectiveExit {
                reason: "take_profit",
                trigger_price: price,
            });
        }

        None
    }

    fn take_profit_price(&self, position: &HmaManagedPosition, tick_size: f64) -> Option<f64> {
        if self.take_profit_ticks <= 0.0 {
            return None;
        }
        Some(match position.side {
            PositionSide::Long => position.entry_price + self.take_profit_ticks * tick_size,
            PositionSide::Short => position.entry_price - self.take_profit_ticks * tick_size,
        })
    }

    fn stop_loss_price(&self, position: &HmaManagedPosition, tick_size: f64) -> Option<f64> {
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
        position: &HmaManagedPosition,
        tick_size: f64,
    ) -> (Option<f64>, Option<&'static str>) {
        let fixed = self.stop_loss_price(position, tick_size);
        let trailing = position.current_stop_price;
        match (fixed, trailing, position.side) {
            (None, None, _) => (None, None),
            (Some(price), None, _) => (Some(price), Some("stop_loss")),
            (None, Some(price), _) => (Some(price), Some("trail_stop")),
            (Some(fixed_price), Some(trail_price), PositionSide::Long) => {
                if trail_price >= fixed_price {
                    (Some(trail_price), Some("trail_stop"))
                } else {
                    (Some(fixed_price), Some("stop_loss"))
                }
            }
            (Some(fixed_price), Some(trail_price), PositionSide::Short) => {
                if trail_price <= fixed_price {
                    (Some(trail_price), Some("trail_stop"))
                } else {
                    (Some(fixed_price), Some("stop_loss"))
                }
            }
        }
    }
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

    #[test]
    fn broker_protection_offsets_follow_tick_size() {
        let config = HmaAngleConfig {
            take_profit_ticks: 8.0,
            stop_loss_ticks: 6.0,
            ..HmaAngleConfig::default()
        };

        let protection = config.broker_protection(Some(0.25)).expect("protection");
        assert_eq!(protection.take_profit_offset, Some(2.0));
        assert_eq!(protection.stop_loss_offset, Some(1.5));
    }

    #[test]
    fn protective_exit_hits_take_profit_for_long_position() {
        let config = HmaAngleConfig {
            take_profit_ticks: 4.0,
            ..HmaAngleConfig::default()
        };
        let mut runtime = HmaAngleExecutionState::default();
        config.sync_position(&mut runtime, 1, Some(100.0));

        let exit = config
            .evaluate_protective_exit(
                &mut runtime,
                &Bar {
                    ts_ns: 0,
                    open: 100.0,
                    high: 101.25,
                    low: 99.75,
                    close: 101.0,
                },
                Some(0.25),
            )
            .expect("take profit should trigger");

        assert_eq!(exit.reason, "take_profit");
        assert_eq!(exit.trigger_price, 101.0);
    }

    #[test]
    fn desired_trailing_stop_price_advances_after_trigger() {
        let config = HmaAngleConfig {
            stop_loss_ticks: 8.0,
            use_trailing_stop: true,
            trail_trigger_ticks: 4.0,
            trail_offset_ticks: 2.0,
            ..HmaAngleConfig::default()
        };
        let mut runtime = HmaAngleExecutionState::default();
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
    }

    #[test]
    fn protective_exit_trails_long_position_by_ticks() {
        let config = HmaAngleConfig {
            use_trailing_stop: true,
            trail_trigger_ticks: 4.0,
            trail_offset_ticks: 2.0,
            ..HmaAngleConfig::default()
        };
        let mut runtime = HmaAngleExecutionState::default();
        config.sync_position(&mut runtime, 1, Some(100.0));

        let no_exit = config.evaluate_protective_exit(
            &mut runtime,
            &Bar {
                ts_ns: 1,
                open: 100.0,
                high: 101.25,
                low: 100.0,
                close: 101.0,
            },
            Some(0.25),
        );
        assert!(no_exit.is_none());
        let stop = runtime
            .position
            .as_ref()
            .and_then(|position| position.current_stop_price);
        assert_eq!(stop, Some(100.75));

        let exit = config
            .evaluate_protective_exit(
                &mut runtime,
                &Bar {
                    ts_ns: 2,
                    open: 101.0,
                    high: 101.0,
                    low: 100.70,
                    close: 100.80,
                },
                Some(0.25),
            )
            .expect("trailing stop should trigger");

        assert_eq!(exit.reason, "trail_stop");
        assert_eq!(exit.trigger_price, 100.75);
    }
}
