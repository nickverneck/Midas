use crate::broker::Bar;
use crate::strategies::{PositionSide, StrategySignal};
use serde::{Deserialize, Serialize};

/// Native ADX directional-regime strategy.
///
/// ADX is used as a hysteretic trend-strength gate.  Direction comes from
/// the signed DI imbalance, while a completed-bar breakout provides price
/// confirmation.  Set `breakout_lookback` to zero to disable that final
/// confirmation when using ADX as a filter for another entry system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdxConfig {
    pub adx_length: usize,
    pub adx_entry_threshold: f64,
    pub adx_exit_threshold: f64,
    pub di_imbalance_threshold: f64,
    pub slope_lookback: usize,
    pub dominance_bars: usize,
    pub breakout_lookback: usize,
    pub inverted: bool,
    pub take_profit_ticks: f64,
    pub stop_loss_ticks: f64,
    pub use_trailing_stop: bool,
    pub trail_trigger_ticks: f64,
    pub trail_offset_ticks: f64,
}

impl Default for AdxConfig {
    fn default() -> Self {
        Self {
            adx_length: 14,
            adx_entry_threshold: 25.0,
            adx_exit_threshold: 20.0,
            di_imbalance_threshold: 0.10,
            slope_lookback: 3,
            dominance_bars: 2,
            breakout_lookback: 20,
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
pub struct AdxEvaluation {
    pub signal: StrategySignal,
    pub latest_close: Option<f64>,
    pub adx: Option<f64>,
    pub previous_adx: Option<f64>,
    pub plus_di: Option<f64>,
    pub minus_di: Option<f64>,
    pub di_imbalance: Option<f64>,
    pub signed_trend_score: Option<f64>,
    pub adx_slope: Option<f64>,
    pub trend_mode: bool,
    pub long_breakout: bool,
    pub short_breakout: bool,
    pub bars_len: usize,
    pub warmup_bars: usize,
    pub current_side: Option<PositionSide>,
    pub inverted: bool,
    pub raw_buy_signal: bool,
    pub raw_sell_signal: bool,
    pub effective_buy_signal: bool,
    pub effective_sell_signal: bool,
    pub hold_reason: Option<&'static str>,
}

impl AdxEvaluation {
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Signal: {}", self.signal.label())];
        if let Some(close) = self.latest_close {
            parts.push(format!("Close: {close:.2}"));
        }
        if let Some(adx) = self.adx {
            parts.push(format!("ADX: {adx:.2}"));
        }
        if let Some(score) = self.signed_trend_score {
            parts.push(format!("Trend score: {score:.2}"));
        }
        if let (Some(plus), Some(minus)) = (self.plus_di, self.minus_di) {
            parts.push(format!("+DI/-DI: {plus:.2}/{minus:.2}"));
        }
        parts.push(format!(
            "Regime: {}",
            if self.trend_mode { "on" } else { "off" }
        ));
        parts.join(" | ")
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "Signal: {} | reason: {} | bars: {}/{} | side: {:?} | inverted: {} | close: {} | adx: {} | prev_adx: {} | +di: {} | -di: {} | imbalance: {} | score: {} | slope: {} | trend_mode: {} | breakout long={} short={} | raw buy={} sell={} | effective buy={} sell={}",
            self.signal.label(),
            self.hold_reason.unwrap_or("signal_ready"),
            self.bars_len,
            self.warmup_bars,
            self.current_side,
            self.inverted,
            fmt_price(self.latest_close),
            fmt_price(self.adx),
            fmt_price(self.previous_adx),
            fmt_price(self.plus_di),
            fmt_price(self.minus_di),
            fmt_price(self.di_imbalance),
            fmt_price(self.signed_trend_score),
            fmt_price(self.adx_slope),
            self.trend_mode,
            self.long_breakout,
            self.short_breakout,
            self.raw_buy_signal,
            self.raw_sell_signal,
            self.effective_buy_signal,
            self.effective_sell_signal,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct AdxExecutionState {
    pub position: Option<AdxManagedPosition>,
}

#[derive(Debug, Clone)]
pub struct AdxManagedPosition {
    pub side: PositionSide,
    pub qty: i32,
    pub entry_price: f64,
    pub best_price: f64,
    pub current_stop_price: Option<f64>,
    pub trailing_active: bool,
}

impl AdxConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.take_profit_ticks > 0.0 || self.stop_loss_ticks > 0.0 || self.use_trailing_stop
    }

    /// Number of bars needed for Wilder ADX, slope, dominance, and breakout
    /// inputs.  The extra bar makes the prior/current ADX pair available for
    /// a closed-bar signal without lookahead.
    pub fn warmup_bars(&self) -> usize {
        let period = self.adx_length.max(1);
        let adx_seed = period.saturating_mul(2).saturating_sub(1);
        adx_seed
            .saturating_add(
                self.slope_lookback
                    .max(self.dominance_bars)
                    .max(self.breakout_lookback),
            )
            .saturating_add(1)
    }

    pub fn evaluate(&self, bars: &[Bar], current_side: Option<PositionSide>) -> AdxEvaluation {
        let warmup_bars = self.warmup_bars();
        let Some(_last_bar) = bars.last() else {
            return self.empty_evaluation(bars.len(), warmup_bars, current_side, "no_bars");
        };

        let series = adx_series(bars, self.adx_length.max(1));
        let idx = bars.len().saturating_sub(1);
        let trend_modes = hysteretic_regime(
            &series.adx,
            self.adx_entry_threshold,
            self.adx_exit_threshold,
        );
        let trend_mode = trend_modes.get(idx).copied().unwrap_or(false);
        let adx = series.adx.get(idx).copied().filter(|v| v.is_finite());
        let previous_adx = idx
            .checked_sub(1)
            .and_then(|index| series.adx.get(index).copied())
            .filter(|v| v.is_finite());
        let plus_di = series.plus_di.get(idx).copied().filter(|v| v.is_finite());
        let minus_di = series.minus_di.get(idx).copied().filter(|v| v.is_finite());
        let di_imbalance = di_imbalance(plus_di, minus_di);
        let signed_trend_score = match (adx, di_imbalance) {
            (Some(adx), Some(imbalance)) => Some(adx * imbalance),
            _ => None,
        };
        let adx_slope = slope_at(&series.adx, idx, self.slope_lookback);
        let (long_breakout, short_breakout) = breakout_at(bars, idx, self.breakout_lookback);

        if bars.len() < warmup_bars {
            return self.evaluation_with_values(
                bars,
                current_side,
                warmup_bars,
                adx,
                previous_adx,
                plus_di,
                minus_di,
                di_imbalance,
                signed_trend_score,
                adx_slope,
                trend_mode,
                long_breakout,
                short_breakout,
                false,
                false,
                "warming_up",
            );
        }

        let long_dominance = dominance_count(
            &series.plus_di,
            &series.minus_di,
            idx,
            self.di_imbalance_threshold,
            true,
        );
        let short_dominance = dominance_count(
            &series.plus_di,
            &series.minus_di,
            idx,
            self.di_imbalance_threshold,
            false,
        );
        let slope_positive = adx_slope.is_some_and(|slope| slope > 0.0);
        let long_direction =
            di_imbalance.is_some_and(|imbalance| imbalance >= self.di_imbalance_threshold);
        let short_direction =
            di_imbalance.is_some_and(|imbalance| imbalance <= -self.di_imbalance_threshold);
        let raw_buy_signal = trend_mode
            && slope_positive
            && long_direction
            && long_dominance >= self.dominance_bars.max(1)
            && long_breakout;
        let raw_sell_signal = trend_mode
            && slope_positive
            && short_direction
            && short_dominance >= self.dominance_bars.max(1)
            && short_breakout;
        self.evaluation_with_values(
            bars,
            current_side,
            warmup_bars,
            adx,
            previous_adx,
            plus_di,
            minus_di,
            di_imbalance,
            signed_trend_score,
            adx_slope,
            trend_mode,
            long_breakout,
            short_breakout,
            raw_buy_signal,
            raw_sell_signal,
            if !trend_mode {
                "trend_mode_inactive"
            } else if !slope_positive {
                "adx_slope_not_positive"
            } else if !long_direction && !short_direction {
                "di_imbalance_below_threshold"
            } else if long_dominance < self.dominance_bars.max(1)
                && short_dominance < self.dominance_bars.max(1)
            {
                "di_dominance_not_confirmed"
            } else if !long_breakout && !short_breakout {
                "price_confirmation_missing"
            } else {
                "no_effective_signal"
            },
        )
    }

    fn empty_evaluation(
        &self,
        bars_len: usize,
        warmup_bars: usize,
        current_side: Option<PositionSide>,
        reason: &'static str,
    ) -> AdxEvaluation {
        AdxEvaluation {
            signal: StrategySignal::Hold,
            latest_close: None,
            adx: None,
            previous_adx: None,
            plus_di: None,
            minus_di: None,
            di_imbalance: None,
            signed_trend_score: None,
            adx_slope: None,
            trend_mode: false,
            long_breakout: false,
            short_breakout: false,
            bars_len,
            warmup_bars,
            current_side,
            inverted: self.inverted,
            raw_buy_signal: false,
            raw_sell_signal: false,
            effective_buy_signal: false,
            effective_sell_signal: false,
            hold_reason: Some(reason),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluation_with_values(
        &self,
        bars: &[Bar],
        current_side: Option<PositionSide>,
        warmup_bars: usize,
        adx: Option<f64>,
        previous_adx: Option<f64>,
        plus_di: Option<f64>,
        minus_di: Option<f64>,
        di_imbalance: Option<f64>,
        signed_trend_score: Option<f64>,
        adx_slope: Option<f64>,
        trend_mode: bool,
        long_breakout: bool,
        short_breakout: bool,
        raw_buy_signal: bool,
        raw_sell_signal: bool,
        reason: &'static str,
    ) -> AdxEvaluation {
        let latest_close = bars.last().map(|bar| bar.close);
        if bars.len() < warmup_bars {
            return AdxEvaluation {
                signal: StrategySignal::Hold,
                latest_close,
                adx,
                previous_adx,
                plus_di,
                minus_di,
                di_imbalance,
                signed_trend_score,
                adx_slope,
                trend_mode,
                long_breakout,
                short_breakout,
                bars_len: bars.len(),
                warmup_bars,
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some(reason),
            };
        }

        let mut buy_signal = raw_buy_signal;
        let mut sell_signal = raw_sell_signal;
        if self.inverted {
            std::mem::swap(&mut buy_signal, &mut sell_signal);
        }
        let signal = resolve_signal(buy_signal, sell_signal, current_side);
        let hold_reason = if signal != StrategySignal::Hold {
            None
        } else if buy_signal && current_side == Some(PositionSide::Long) {
            Some("buy_signal_already_long")
        } else if sell_signal && current_side == Some(PositionSide::Short) {
            Some("sell_signal_already_short")
        } else {
            Some(reason)
        };
        AdxEvaluation {
            signal,
            latest_close,
            adx,
            previous_adx,
            plus_di,
            minus_di,
            di_imbalance,
            signed_trend_score,
            adx_slope,
            trend_mode,
            long_breakout,
            short_breakout,
            bars_len: bars.len(),
            warmup_bars,
            current_side,
            inverted: self.inverted,
            raw_buy_signal,
            raw_sell_signal,
            effective_buy_signal: buy_signal,
            effective_sell_signal: sell_signal,
            hold_reason,
        }
    }

    pub fn sync_position(
        &self,
        runtime: &mut AdxExecutionState,
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
                runtime.position = Some(AdxManagedPosition {
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
        (self.take_profit_ticks > 0.0).then_some(self.take_profit_ticks * tick_size)
    }

    pub fn desired_trailing_stop_price(
        &self,
        runtime: &mut AdxExecutionState,
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
                    position.best_price = position.best_price.max(favorable_price)
                }
                PositionSide::Short => {
                    position.best_price = position.best_price.min(favorable_price)
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
        runtime: &AdxExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
        let position = runtime.position.as_ref()?;
        let fixed = (self.stop_loss_ticks > 0.0).then_some(match position.side {
            PositionSide::Long => position.entry_price - self.stop_loss_ticks * tick_size,
            PositionSide::Short => position.entry_price + self.stop_loss_ticks * tick_size,
        });
        match (fixed, position.current_stop_price, position.side) {
            (None, None, _) => None,
            (Some(price), None, _) | (None, Some(price), _) => Some(price),
            (Some(fixed), Some(trail), PositionSide::Long) => Some(fixed.max(trail)),
            (Some(fixed), Some(trail), PositionSide::Short) => Some(fixed.min(trail)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AdxSeries {
    pub(crate) adx: Vec<f64>,
    pub(crate) plus_di: Vec<f64>,
    pub(crate) minus_di: Vec<f64>,
}

pub(crate) fn adx_series(bars: &[Bar], period: usize) -> AdxSeries {
    let period = period.max(1);
    let len = bars.len();
    let mut tr = vec![f64::NAN; len];
    let mut plus_dm = vec![f64::NAN; len];
    let mut minus_dm = vec![f64::NAN; len];
    for idx in 0..len {
        let bar = &bars[idx];
        if idx == 0 {
            tr[idx] = (bar.high - bar.low).abs();
            plus_dm[idx] = 0.0;
            minus_dm[idx] = 0.0;
            continue;
        }
        let previous = &bars[idx - 1];
        let values = [
            bar.high,
            bar.low,
            bar.close,
            previous.high,
            previous.low,
            previous.close,
        ];
        if values.iter().all(|value| value.is_finite()) {
            tr[idx] = (bar.high - bar.low)
                .abs()
                .max((bar.high - previous.close).abs())
                .max((bar.low - previous.close).abs());
            let up_move = bar.high - previous.high;
            let down_move = previous.low - bar.low;
            plus_dm[idx] = if up_move > down_move && up_move > 0.0 {
                up_move
            } else {
                0.0
            };
            minus_dm[idx] = if down_move > up_move && down_move > 0.0 {
                down_move
            } else {
                0.0
            };
        }
    }

    let mut smooth_tr = vec![f64::NAN; len];
    let mut smooth_plus = vec![f64::NAN; len];
    let mut smooth_minus = vec![f64::NAN; len];
    if len >= period {
        for (source, output) in [
            (&tr, &mut smooth_tr),
            (&plus_dm, &mut smooth_plus),
            (&minus_dm, &mut smooth_minus),
        ] {
            let seed = &source[..period];
            if seed.iter().all(|value| value.is_finite()) {
                output[period - 1] = seed.iter().sum::<f64>();
            }
        }
        for idx in period..len {
            if smooth_tr[idx - 1].is_finite() && tr[idx].is_finite() {
                smooth_tr[idx] = smooth_tr[idx - 1] - smooth_tr[idx - 1] / period as f64 + tr[idx];
            }
            if smooth_plus[idx - 1].is_finite() && plus_dm[idx].is_finite() {
                smooth_plus[idx] =
                    smooth_plus[idx - 1] - smooth_plus[idx - 1] / period as f64 + plus_dm[idx];
            }
            if smooth_minus[idx - 1].is_finite() && minus_dm[idx].is_finite() {
                smooth_minus[idx] =
                    smooth_minus[idx - 1] - smooth_minus[idx - 1] / period as f64 + minus_dm[idx];
            }
        }
    }

    let mut plus_di = vec![f64::NAN; len];
    let mut minus_di = vec![f64::NAN; len];
    let mut dx = vec![f64::NAN; len];
    for idx in 0..len {
        let (tr, plus, minus) = (smooth_tr[idx], smooth_plus[idx], smooth_minus[idx]);
        if tr.is_finite() && tr > f64::EPSILON && plus.is_finite() && minus.is_finite() {
            plus_di[idx] = 100.0 * plus / tr;
            minus_di[idx] = 100.0 * minus / tr;
            let denominator = plus_di[idx] + minus_di[idx];
            if denominator > f64::EPSILON {
                dx[idx] = 100.0 * (plus_di[idx] - minus_di[idx]).abs() / denominator;
            }
        }
    }

    let mut adx = vec![f64::NAN; len];
    let adx_seed_start = period.saturating_sub(1);
    let adx_seed_end = adx_seed_start.saturating_add(period);
    if adx_seed_end <= len {
        let seed = &dx[adx_seed_start..adx_seed_end];
        if seed.iter().all(|value| value.is_finite()) {
            adx[adx_seed_end - 1] = seed.iter().sum::<f64>() / period as f64;
            for idx in adx_seed_end..len {
                if adx[idx - 1].is_finite() && dx[idx].is_finite() {
                    adx[idx] = (adx[idx - 1] * (period as f64 - 1.0) + dx[idx]) / period as f64;
                }
            }
        }
    }
    AdxSeries {
        adx,
        plus_di,
        minus_di,
    }
}

fn hysteretic_regime(adx: &[f64], entry: f64, exit: f64) -> Vec<bool> {
    let mut active = false;
    let mut output = Vec::with_capacity(adx.len());
    for value in adx {
        if value.is_finite() {
            if !active && *value >= entry {
                active = true;
            } else if active && *value <= exit {
                active = false;
            }
        }
        output.push(active);
    }
    output
}

fn di_imbalance(plus_di: Option<f64>, minus_di: Option<f64>) -> Option<f64> {
    let (plus_di, minus_di) = (plus_di?, minus_di?);
    let denominator = plus_di + minus_di;
    (denominator > f64::EPSILON).then_some((plus_di - minus_di) / denominator)
}

fn slope_at(series: &[f64], idx: usize, lookback: usize) -> Option<f64> {
    let lookback = lookback.max(1);
    let previous_idx = idx.checked_sub(lookback)?;
    let current = series.get(idx).copied()?.filter_finite()?;
    let previous = series.get(previous_idx).copied()?.filter_finite()?;
    Some((current - previous) / lookback as f64)
}

trait FiniteValue {
    fn filter_finite(self) -> Option<f64>;
}

impl FiniteValue for f64 {
    fn filter_finite(self) -> Option<f64> {
        self.is_finite().then_some(self)
    }
}

fn dominance_count(
    plus_di: &[f64],
    minus_di: &[f64],
    idx: usize,
    threshold: f64,
    bullish: bool,
) -> usize {
    let mut count = 0;
    for cursor in (0..=idx).rev() {
        let Some(imbalance) = di_imbalance(
            plus_di.get(cursor).copied().filter(|v| v.is_finite()),
            minus_di.get(cursor).copied().filter(|v| v.is_finite()),
        ) else {
            break;
        };
        if imbalance.abs() < threshold
            || (bullish && imbalance < 0.0)
            || (!bullish && imbalance > 0.0)
        {
            break;
        }
        count += 1;
    }
    count
}

fn breakout_at(bars: &[Bar], idx: usize, lookback: usize) -> (bool, bool) {
    if lookback == 0 {
        return (true, true);
    }
    if idx < lookback {
        return (false, false);
    }
    let window = &bars[idx - lookback..idx];
    let highest = window
        .iter()
        .map(|bar| bar.high)
        .filter(|value| value.is_finite())
        .reduce(f64::max);
    let lowest = window
        .iter()
        .map(|bar| bar.low)
        .filter(|value| value.is_finite())
        .reduce(f64::min);
    let close = bars[idx].close;
    (
        close.is_finite() && highest.is_some_and(|high| close > high),
        close.is_finite() && lowest.is_some_and(|low| close < low),
    )
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

fn fmt_price(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, high: f64, low: f64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open: close,
            high,
            low,
            close,
            volume: None,
        }
    }

    #[test]
    fn adx_wilder_series_is_finite_after_two_periods() {
        let bars = (0..80)
            .map(|index| {
                let close = 100.0 + index as f64;
                bar(index, close + 1.0, close - 1.0, close)
            })
            .collect::<Vec<_>>();
        let series = adx_series(&bars, 14);
        assert!(series.plus_di[13].is_finite());
        assert!(series.adx[26].is_finite());
        assert!(series.adx[26] > 99.0);
        assert!(series.plus_di[40] > series.minus_di[40]);
    }

    #[test]
    fn hysteresis_keeps_regime_on_between_thresholds() {
        let mode = hysteretic_regime(&[10.0, 26.0, 23.0, 21.0, 19.0, 24.0], 25.0, 20.0);
        assert_eq!(mode, vec![false, true, true, true, false, false]);
    }

    #[test]
    fn signed_score_penalizes_weak_di_separation() {
        let score = 30.0 * di_imbalance(Some(55.0), Some(45.0)).unwrap();
        assert!((score - 3.0).abs() < 1e-12);
    }

    #[test]
    fn adx_requires_breakout_and_positive_slope_for_entry() {
        let mut bars = Vec::new();
        for index in 0..100 {
            let close = if index < 50 {
                100.0 + (index as f64 * 0.9).sin() * 2.0
            } else {
                100.0 + (index - 50) as f64 * 0.25
            };
            bars.push(bar(index, close + 0.1, close - 0.1, close));
        }
        let config = AdxConfig {
            breakout_lookback: 5,
            slope_lookback: 2,
            dominance_bars: 1,
            adx_entry_threshold: 10.0,
            adx_exit_threshold: 5.0,
            ..AdxConfig::default()
        };
        let evaluation = config.evaluate(&bars, None);
        assert!(evaluation.trend_mode);
        assert!(evaluation.adx_slope.is_some_and(|slope| slope >= 0.0));
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
        assert!(evaluation.signed_trend_score.unwrap() > 0.0);
    }

    #[test]
    fn adx_warmup_and_protection_follow_strategy_contract() {
        let config = AdxConfig::default();
        assert_eq!(config.evaluate(&[], None).hold_reason, Some("no_bars"));
        assert_eq!(config.take_profit_offset(Some(0.25)), None);
        let mut runtime = AdxExecutionState::default();
        config.sync_position(&mut runtime, 1, Some(100.0));
        let with_stop = AdxConfig {
            stop_loss_ticks: 6.0,
            ..config
        };
        assert_eq!(
            with_stop.current_effective_stop_price(&runtime, Some(0.25)),
            Some(98.5)
        );
    }
}
