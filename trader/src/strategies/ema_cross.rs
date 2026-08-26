use crate::broker::{Bar, MarketHistoryUpdate};
use crate::strategies::{PositionSide, StrategySignal};
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
    pub previous_fast_ema: Option<f64>,
    pub previous_slow_ema: Option<f64>,
    pub fast_ema: Option<f64>,
    pub slow_ema: Option<f64>,
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

    pub fn debug_summary(&self) -> String {
        format!(
            "Signal: {} | reason: {} | bars: {}/{} | side: {:?} | inverted: {} | close: {} | prev_fast: {} | prev_slow: {} | fast: {} | slow: {} | delta: {}->{} | raw_cross buy={} sell={} | effective_cross buy={} sell={}",
            self.signal.label(),
            self.hold_reason.unwrap_or("signal_ready"),
            self.bars_len,
            self.warmup_bars,
            self.current_side,
            self.inverted,
            fmt_price(self.latest_close),
            fmt_price(self.previous_fast_ema),
            fmt_price(self.previous_slow_ema),
            fmt_price(self.fast_ema),
            fmt_price(self.slow_ema),
            fmt_delta(self.previous_fast_ema, self.previous_slow_ema),
            fmt_delta(self.fast_ema, self.slow_ema),
            self.raw_buy_signal,
            self.raw_sell_signal,
            self.effective_buy_signal,
            self.effective_sell_signal
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct EmaCrossExecutionState {
    pub position: Option<EmaManagedPosition>,
    indicator: EmaCrossIndicatorState,
}

/// Recursive EMA state used by the replay streaming evaluator.
///
/// The legacy evaluator intentionally remains available on
/// [`EmaCrossConfig::evaluate`] for compatibility.  This state is kept next
/// to the execution state so a replay engine can advance indicators in lock
/// step with the ordered bars it is already processing.
#[derive(Debug, Clone, Default)]
struct EmaCrossIndicatorState {
    fast_length: usize,
    slow_length: usize,
    first_bar: Option<Bar>,
    last_bar: Option<Bar>,
    window_len: usize,
    previous_fast_ema: Option<f64>,
    previous_slow_ema: Option<f64>,
    fast_ema: Option<f64>,
    slow_ema: Option<f64>,
    fast_recurrence: Option<f64>,
    slow_recurrence: Option<f64>,
    fast_weighted_tail: Option<f64>,
    slow_weighted_tail: Option<f64>,
    /// Rolling digest of the retained bar window.  The digest lets the
    /// streaming path distinguish a genuine append/slide from an older-bar
    /// correction without ever trusting an unchanged latest bar by itself.
    history_fingerprint: Option<u64>,
    history_leading_power: u64,
    source_update_sequence: Option<u64>,
}

const HISTORY_HASH_OFFSET: u64 = 0xcbf29ce484222325;
const HISTORY_HASH_BASE: u64 = 0x100000001b3;

fn bar_fingerprint(bar: &Bar) -> u64 {
    let mut hash = HISTORY_HASH_OFFSET;
    for value in [
        bar.ts_ns as u64,
        bar.open.to_bits(),
        bar.high.to_bits(),
        bar.low.to_bits(),
        bar.close.to_bits(),
        bar.volume.map(f64::to_bits).unwrap_or(0),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(HISTORY_HASH_BASE);
    }
    hash
}

fn bars_fingerprint(bars: &[Bar]) -> u64 {
    bars.iter().fold(0, |hash, bar| {
        hash.wrapping_mul(HISTORY_HASH_BASE)
            .wrapping_add(bar_fingerprint(bar))
    })
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

impl EmaCrossIndicatorState {
    fn reset(&mut self, fast_length: usize, slow_length: usize) {
        self.fast_length = fast_length;
        self.slow_length = slow_length;
        self.first_bar = None;
        self.last_bar = None;
        self.window_len = 0;
        self.previous_fast_ema = None;
        self.previous_slow_ema = None;
        self.fast_ema = None;
        self.slow_ema = None;
        self.fast_recurrence = None;
        self.slow_recurrence = None;
        self.fast_weighted_tail = None;
        self.slow_weighted_tail = None;
        self.history_fingerprint = None;
        self.history_leading_power = 1;
        self.source_update_sequence = None;
    }

    fn rebuild(&mut self, bars: &[Bar]) {
        let fast_length = self.fast_length;
        let slow_length = self.slow_length;
        self.reset(fast_length, slow_length);
        for bar in bars {
            self.push(bar);
        }
    }

    fn push(&mut self, bar: &Bar) {
        if self.first_bar.is_none() {
            self.first_bar = Some(bar.clone());
        }
        self.previous_fast_ema = self.fast_ema;
        self.previous_slow_ema = self.slow_ema;
        let previous_window_len = self.window_len;
        let (fast_recurrence, fast_output) =
            next_ema(self.fast_recurrence, bar.close, self.fast_length.max(1));
        let (slow_recurrence, slow_output) =
            next_ema(self.slow_recurrence, bar.close, self.slow_length.max(1));
        self.fast_recurrence = Some(fast_recurrence);
        self.slow_recurrence = Some(slow_recurrence);
        self.fast_ema = fast_output;
        self.slow_ema = slow_output;
        if bar.close.is_finite() {
            if previous_window_len == 0 {
                self.fast_weighted_tail = Some(0.0);
                self.slow_weighted_tail = Some(0.0);
            } else {
                self.fast_weighted_tail = self.fast_weighted_tail.map(|tail| {
                    let alpha = 2.0 / (self.fast_length.max(1) as f64 + 1.0);
                    (1.0 - alpha) * tail + alpha * bar.close
                });
                self.slow_weighted_tail = self.slow_weighted_tail.map(|tail| {
                    let alpha = 2.0 / (self.slow_length.max(1) as f64 + 1.0);
                    (1.0 - alpha) * tail + alpha * bar.close
                });
            }
        } else {
            self.fast_weighted_tail = None;
            self.slow_weighted_tail = None;
        }
        self.window_len = self.window_len.saturating_add(1);
        self.last_bar = Some(bar.clone());
        self.history_fingerprint = Some(match self.history_fingerprint {
            Some(history) => history
                .wrapping_mul(HISTORY_HASH_BASE)
                .wrapping_add(bar_fingerprint(bar)),
            None => bar_fingerprint(bar),
        });
        if self.window_len > 1 {
            self.history_leading_power = self.history_leading_power.wrapping_mul(HISTORY_HASH_BASE);
        }
    }

    /// Slide a fixed-size finite window by one bar and append the new close.
    /// The weighted-tail accumulators let this preserve the legacy
    /// first-value EMA seed without replaying the whole retained window.
    fn slide_and_push(&mut self, new_first: &Bar, latest: &Bar) -> bool {
        if self.window_len <= 1
            || !new_first.close.is_finite()
            || !latest.close.is_finite()
            || self.first_bar.is_none()
        {
            return false;
        }
        let n = self.window_len;
        let transform = |tail: Option<f64>, period: usize| {
            let Some(tail) = tail.filter(|value| value.is_finite()) else {
                return None;
            };
            let alpha = 2.0 / (period.max(1) as f64 + 1.0);
            let retain = 1.0 - alpha;
            let removed = alpha * retain.powi((n.saturating_sub(2)) as i32) * new_first.close;
            let after_drop = tail - removed;
            let previous = retain.powi((n.saturating_sub(2)) as i32) * new_first.close + after_drop;
            let next_tail = retain * after_drop + alpha * latest.close;
            let next = retain.powi((n.saturating_sub(1)) as i32) * new_first.close + next_tail;
            Some((previous, next, next_tail))
        };
        let Some((previous_fast, fast, fast_tail)) =
            transform(self.fast_weighted_tail, self.fast_length)
        else {
            return false;
        };
        let Some((previous_slow, slow, slow_tail)) =
            transform(self.slow_weighted_tail, self.slow_length)
        else {
            return false;
        };
        if !previous_fast.is_finite()
            || !fast.is_finite()
            || !previous_slow.is_finite()
            || !slow.is_finite()
        {
            return false;
        }
        self.previous_fast_ema = Some(previous_fast);
        self.previous_slow_ema = Some(previous_slow);
        self.fast_ema = Some(fast);
        self.slow_ema = Some(slow);
        self.fast_recurrence = Some(fast);
        self.slow_recurrence = Some(slow);
        self.fast_weighted_tail = Some(fast_tail);
        self.slow_weighted_tail = Some(slow_tail);
        if let (Some(history), Some(old_first)) =
            (self.history_fingerprint, self.first_bar.as_ref())
        {
            self.history_fingerprint = Some(
                history
                    .wrapping_sub(
                        bar_fingerprint(old_first).wrapping_mul(self.history_leading_power),
                    )
                    .wrapping_mul(HISTORY_HASH_BASE)
                    .wrapping_add(bar_fingerprint(latest)),
            );
        } else {
            return false;
        }
        self.first_bar = Some(new_first.clone());
        self.last_bar = Some(latest.clone());
        true
    }

    /// Replace only the latest bar when a forming/range bar is revised in
    /// place.  The EMA recurrence before the latest bar is already cached, so
    /// this avoids rebuilding the retained history on every quote update.
    ///
    /// If the preceding recurrence is unavailable (for example because the
    /// retained history contains a non-finite close), callers should fall
    /// back to `rebuild`; that keeps the correction path exactly equivalent to
    /// the legacy evaluator.
    fn revise_last(&mut self, latest: &Bar) -> bool {
        if self.window_len <= 1
            || self.first_bar.is_none()
            || self.last_bar.is_none()
            || !latest.close.is_finite()
        {
            return false;
        }

        let Some(previous_fast) = self.previous_fast_ema.filter(|value| value.is_finite()) else {
            return false;
        };
        let Some(previous_slow) = self.previous_slow_ema.filter(|value| value.is_finite()) else {
            return false;
        };
        let Some(first_close) = self
            .first_bar
            .as_ref()
            .map(|bar| bar.close)
            .filter(|value| value.is_finite())
        else {
            return false;
        };

        let (fast_recurrence, fast) =
            next_ema(Some(previous_fast), latest.close, self.fast_length.max(1));
        let (slow_recurrence, slow) =
            next_ema(Some(previous_slow), latest.close, self.slow_length.max(1));
        let (Some(fast), Some(slow)) = (fast, slow) else {
            return false;
        };

        let update_tail = |previous: f64, period: usize| {
            let alpha = 2.0 / (period.max(1) as f64 + 1.0);
            let retain = 1.0 - alpha;
            let prior_seed = retain.powi((self.window_len.saturating_sub(2)) as i32) * first_close;
            let prior_tail = previous - prior_seed;
            (retain * prior_tail + alpha * latest.close)
                .is_finite()
                .then_some(retain * prior_tail + alpha * latest.close)
        };
        let Some(fast_tail) = update_tail(previous_fast, self.fast_length) else {
            return false;
        };
        let Some(slow_tail) = update_tail(previous_slow, self.slow_length) else {
            return false;
        };

        let Some(old_last) = self.last_bar.as_ref() else {
            return false;
        };
        let Some(history) = self.history_fingerprint else {
            return false;
        };
        self.history_fingerprint = Some(
            history
                .wrapping_sub(bar_fingerprint(old_last))
                .wrapping_add(bar_fingerprint(latest)),
        );
        self.fast_recurrence = Some(fast_recurrence);
        self.slow_recurrence = Some(slow_recurrence);
        self.previous_fast_ema = Some(previous_fast);
        self.previous_slow_ema = Some(previous_slow);
        self.fast_ema = Some(fast);
        self.slow_ema = Some(slow);
        self.fast_weighted_tail = Some(fast_tail);
        self.slow_weighted_tail = Some(slow_tail);
        self.last_bar = Some(latest.clone());
        true
    }
}

fn next_ema(previous: Option<f64>, value: f64, period: usize) -> (f64, Option<f64>) {
    let alpha = 2.0 / (period as f64 + 1.0);
    match previous {
        Some(previous) if value.is_finite() => {
            let next = alpha * value + (1.0 - alpha) * previous;
            (next, next.is_finite().then_some(next))
        }
        Some(previous) => (previous, None),
        None => (value, value.is_finite().then_some(value)),
    }
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
                previous_fast_ema: None,
                previous_slow_ema: None,
                fast_ema: None,
                slow_ema: None,
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some("no_bars"),
            };
        };

        if bars.len() < self.warmup_bars() {
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                previous_fast_ema: None,
                previous_slow_ema: None,
                fast_ema: None,
                slow_ema: None,
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some("warming_up"),
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
                previous_fast_ema: fast
                    .get(prev_idx)
                    .copied()
                    .filter(|value| value.is_finite()),
                previous_slow_ema: slow
                    .get(prev_idx)
                    .copied()
                    .filter(|value| value.is_finite()),
                fast_ema: fast.get(idx).copied().filter(|value| value.is_finite()),
                slow_ema: slow.get(idx).copied().filter(|value| value.is_finite()),
                bars_len: bars.len(),
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
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
        EmaCrossEvaluation {
            signal,
            latest_close: Some(last_bar.close),
            previous_fast_ema: Some(prev_fast),
            previous_slow_ema: Some(prev_slow),
            fast_ema: Some(curr_fast),
            slow_ema: Some(curr_slow),
            bars_len: bars.len(),
            warmup_bars: self.warmup_bars(),
            current_side,
            inverted: self.inverted,
            raw_buy_signal,
            raw_sell_signal,
            effective_buy_signal: buy_signal,
            effective_sell_signal: sell_signal,
            hold_reason,
        }
    }

    /// Evaluate the latest bar using recursive indicator state.
    ///
    /// The first call seeds the state from the supplied history (one linear
    /// pass). The replay path uses [`Self::evaluate_streaming_with_market_update`]
    /// with a publisher sequence/hint, making verified appends constant-time.
    /// This hint-free convenience method defensively computes a retained-window
    /// digest on each call so callers that do not own the bar publisher still
    /// detect older-bar corrections. Corrections, out-of-order bars, and
    /// configuration changes fall back to a linear rebuild so the observable
    /// signal remains equivalent to the legacy batch evaluator.
    pub fn evaluate_streaming(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> EmaCrossEvaluation {
        self.evaluate_streaming_with_market_update(
            runtime,
            bars,
            current_side,
            None,
            MarketHistoryUpdate::Snapshot,
        )
    }

    /// Evaluate with a market-update hint supplied by the retained bar
    /// publisher. A sequence number lets repeated broker/account callbacks
    /// reuse the already-advanced state; append/correction hints avoid
    /// rescanning the full window on the normal replay path. Callers without
    /// a trustworthy hint should use [`Self::evaluate_streaming`], which
    /// validates a bounded digest instead.
    pub fn evaluate_streaming_with_market_update(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
        source_update_sequence: Option<u64>,
        market_update: MarketHistoryUpdate,
    ) -> EmaCrossEvaluation {
        if bars.is_empty() {
            runtime.indicator.reset(self.fast_length, self.slow_length);
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: None,
                previous_fast_ema: None,
                previous_slow_ema: None,
                fast_ema: None,
                slow_ema: None,
                bars_len: 0,
                warmup_bars: self.warmup_bars(),
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some("no_bars"),
            };
        }

        let state = &mut runtime.indicator;
        if source_update_sequence.is_none() {
            state.source_update_sequence = None;
        }
        let hinted_update_already_applied = source_update_sequence
            .is_some_and(|sequence| state.source_update_sequence == Some(sequence));
        let config_changed = state.fast_length != self.fast_length
            || state.slow_length != self.slow_length
            || state.last_bar.is_none();
        if !hinted_update_already_applied
            && (config_changed
                || (source_update_sequence.is_some()
                    && matches!(
                        market_update,
                        MarketHistoryUpdate::Snapshot | MarketHistoryUpdate::Correction
                    )))
        {
            state.reset(self.fast_length, self.slow_length);
            state.rebuild(bars);
        } else if !hinted_update_already_applied {
            let latest = bars.last().expect("bars is not empty");
            let same_latest = state.last_bar.as_ref() == Some(latest);
            let incoming_fingerprint = source_update_sequence
                .is_none()
                .then(|| bars_fingerprint(bars));
            let history_matches = incoming_fingerprint
                .is_none_or(|fingerprint| state.history_fingerprint == Some(fingerprint));
            let appended = bars.len() >= 2
                && (bars.len() == state.window_len.saturating_add(1)
                    || (bars.len() == state.window_len
                        && state.first_bar.as_ref() != bars.first()))
                && state.last_bar.as_ref() == bars.get(bars.len() - 2)
                && latest.ts_ns
                    > state
                        .last_bar
                        .as_ref()
                        .map(|bar| bar.ts_ns)
                        .unwrap_or(i64::MIN);
            let same_timestamp_revision = !same_latest
                && bars.len() == state.window_len
                && state.first_bar.as_ref() == bars.first()
                && state
                    .last_bar
                    .as_ref()
                    .is_some_and(|previous| previous.ts_ns == latest.ts_ns)
                && if source_update_sequence.is_some() {
                    matches!(market_update, MarketHistoryUpdate::Unchanged)
                } else {
                    state
                        .history_fingerprint
                        .zip(state.last_bar.as_ref())
                        .zip(incoming_fingerprint)
                        .is_some_and(|((history, previous), incoming)| {
                            history
                                .wrapping_sub(bar_fingerprint(previous))
                                .wrapping_add(bar_fingerprint(latest))
                                == incoming
                        })
                };
            if same_latest && !history_matches {
                state.rebuild(bars);
            } else if same_timestamp_revision {
                if !state.revise_last(latest) {
                    state.rebuild(bars);
                }
            } else if !same_latest && appended {
                let first_changed = state.first_bar.as_ref() != bars.first();
                let expected_fingerprint = if source_update_sequence.is_none() {
                    if first_changed {
                        state.history_fingerprint.zip(state.first_bar.as_ref()).map(
                            |(history, old_first)| {
                                history
                                    .wrapping_sub(
                                        bar_fingerprint(old_first)
                                            .wrapping_mul(state.history_leading_power),
                                    )
                                    .wrapping_mul(HISTORY_HASH_BASE)
                                    .wrapping_add(bar_fingerprint(latest))
                            },
                        )
                    } else {
                        state.history_fingerprint.map(|history| {
                            history
                                .wrapping_mul(HISTORY_HASH_BASE)
                                .wrapping_add(bar_fingerprint(latest))
                        })
                    }
                } else {
                    None
                };
                if expected_fingerprint
                    .is_some_and(|expected| Some(expected) != incoming_fingerprint)
                {
                    state.rebuild(bars);
                } else if first_changed {
                    let Some(first) = bars.first() else {
                        state.rebuild(bars);
                        return self.evaluate_streaming(runtime, bars, current_side);
                    };
                    if !state.slide_and_push(first, latest) {
                        state.rebuild(bars);
                    }
                } else {
                    state.push(latest);
                }
            } else if !same_latest {
                state.rebuild(bars);
            }
        }
        if let Some(sequence) = source_update_sequence {
            state.source_update_sequence = Some(sequence);
        }

        let Some(last_bar) = bars.last() else {
            unreachable!("bars is not empty");
        };
        let warmup_bars = self.warmup_bars();
        if bars.len() < warmup_bars {
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                previous_fast_ema: None,
                previous_slow_ema: None,
                fast_ema: None,
                slow_ema: None,
                bars_len: bars.len(),
                warmup_bars,
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some("warming_up"),
            };
        }

        let prev_fast = state.previous_fast_ema.filter(|value| value.is_finite());
        let prev_slow = state.previous_slow_ema.filter(|value| value.is_finite());
        let curr_fast = state.fast_ema.filter(|value| value.is_finite());
        let curr_slow = state.slow_ema.filter(|value| value.is_finite());
        let (Some(prev_fast), Some(prev_slow), Some(curr_fast), Some(curr_slow)) =
            (prev_fast, prev_slow, curr_fast, curr_slow)
        else {
            return EmaCrossEvaluation {
                signal: StrategySignal::Hold,
                latest_close: Some(last_bar.close),
                previous_fast_ema: prev_fast,
                previous_slow_ema: prev_slow,
                fast_ema: curr_fast,
                slow_ema: curr_slow,
                bars_len: bars.len(),
                warmup_bars,
                current_side,
                inverted: self.inverted,
                raw_buy_signal: false,
                raw_sell_signal: false,
                effective_buy_signal: false,
                effective_sell_signal: false,
                hold_reason: Some("non_finite_indicator"),
            };
        };

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
        EmaCrossEvaluation {
            signal,
            latest_close: Some(last_bar.close),
            previous_fast_ema: Some(prev_fast),
            previous_slow_ema: Some(prev_slow),
            fast_ema: Some(curr_fast),
            slow_ema: Some(curr_slow),
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
            volume: None,
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
    fn streaming_ema_matches_batch_values_and_signals() {
        let config = EmaCrossConfig {
            fast_length: 3,
            slow_length: 7,
            inverted: false,
            ..EmaCrossConfig::default()
        };
        let bars = (0..48)
            .map(|idx| {
                let close = 100.0 + (idx as f64 * 0.37).sin() * 4.0 + idx as f64 * 0.03;
                bar(idx + 1, close)
            })
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();

        for end in 1..=bars.len() {
            let window = &bars[..end];
            let legacy = config.evaluate(window, None);
            let streaming = config.evaluate_streaming(&mut runtime, window, None);
            assert_eq!(streaming.signal, legacy.signal, "bar {end}");
            assert_eq!(streaming.raw_buy_signal, legacy.raw_buy_signal, "bar {end}");
            assert_eq!(
                streaming.raw_sell_signal, legacy.raw_sell_signal,
                "bar {end}"
            );
            assert_eq!(
                streaming.previous_fast_ema, legacy.previous_fast_ema,
                "bar {end}"
            );
            assert_eq!(
                streaming.previous_slow_ema, legacy.previous_slow_ema,
                "bar {end}"
            );
            assert_eq!(streaming.fast_ema, legacy.fast_ema, "bar {end}");
            assert_eq!(streaming.slow_ema, legacy.slow_ema, "bar {end}");
        }
    }

    #[test]
    fn hinted_append_path_matches_batch_without_window_scan() {
        let config = EmaCrossConfig {
            fast_length: 3,
            slow_length: 7,
            ..EmaCrossConfig::default()
        };
        let bars = (0..256)
            .map(|idx| bar(idx + 1, 100.0 + (idx as f64 * 0.21).sin() * 5.0))
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let seed = &bars[..32];
        let _ = config.evaluate_streaming_with_market_update(
            &mut runtime,
            seed,
            None,
            Some(1),
            MarketHistoryUpdate::Snapshot,
        );
        for end in 33..=bars.len() {
            let window = &bars[..end];
            let expected = config.evaluate(window, None);
            let actual = config.evaluate_streaming_with_market_update(
                &mut runtime,
                window,
                None,
                Some(end as u64),
                MarketHistoryUpdate::Append,
            );
            assert_eq!(actual.signal, expected.signal, "bar {end}");
            assert_eq!(actual.previous_fast_ema, expected.previous_fast_ema);
            assert_eq!(actual.previous_slow_ema, expected.previous_slow_ema);
            assert_eq!(actual.fast_ema, expected.fast_ema);
            assert_eq!(actual.slow_ema, expected.slow_ema);
        }
    }

    #[test]
    fn streaming_ema_rebuilds_after_a_bar_correction() {
        let config = EmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..EmaCrossConfig::default()
        };
        let mut bars = (0..12)
            .map(|idx| bar(idx + 1, 20.0 + idx as f64))
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let _ = config.evaluate_streaming(&mut runtime, &bars, None);

        // A revised latest bar is common for live/forming streams. It is
        // detected by the cached bar fingerprint and triggers a rebuild.
        bars[11].close += 3.0;
        let expected = config.evaluate(&bars, None);
        let actual = config.evaluate_streaming(&mut runtime, &bars, None);
        assert_eq!(actual.signal, expected.signal);
        assert_eq!(actual.previous_fast_ema, expected.previous_fast_ema);
        assert_eq!(actual.previous_slow_ema, expected.previous_slow_ema);
        assert_eq!(actual.fast_ema, expected.fast_ema);
        assert_eq!(actual.slow_ema, expected.slow_ema);
    }

    #[test]
    fn hinted_forming_bar_revision_matches_batch_and_preserves_next_append() {
        let config = EmaCrossConfig {
            fast_length: 10,
            slow_length: 30,
            ..EmaCrossConfig::default()
        };
        let mut bars = (0..96)
            .map(|idx| bar(idx + 1, 100.0 + (idx as f64 * 0.13).sin() * 3.0))
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let _ = config.evaluate_streaming_with_market_update(
            &mut runtime,
            &bars,
            None,
            Some(1),
            MarketHistoryUpdate::Snapshot,
        );

        bars.last_mut().expect("latest bar").close += 1.25;
        let expected_revision = config.evaluate(&bars, None);
        let actual_revision = config.evaluate_streaming_with_market_update(
            &mut runtime,
            &bars,
            None,
            Some(2),
            MarketHistoryUpdate::Unchanged,
        );
        assert_eq!(actual_revision.signal, expected_revision.signal);
        assert_eq!(
            actual_revision.previous_fast_ema,
            expected_revision.previous_fast_ema
        );
        assert_eq!(
            actual_revision.previous_slow_ema,
            expected_revision.previous_slow_ema
        );
        assert_eq!(actual_revision.fast_ema, expected_revision.fast_ema);
        assert_eq!(actual_revision.slow_ema, expected_revision.slow_ema);

        bars.push(bar(97, 101.0));
        let expected_append = config.evaluate(&bars, None);
        let actual_append = config.evaluate_streaming_with_market_update(
            &mut runtime,
            &bars,
            None,
            Some(3),
            MarketHistoryUpdate::Append,
        );
        assert_eq!(actual_append.signal, expected_append.signal);
        assert_eq!(
            actual_append.previous_fast_ema,
            expected_append.previous_fast_ema
        );
        assert_eq!(
            actual_append.previous_slow_ema,
            expected_append.previous_slow_ema
        );
        assert_eq!(actual_append.fast_ema, expected_append.fast_ema);
        assert_eq!(actual_append.slow_ema, expected_append.slow_ema);
    }

    #[test]
    fn streaming_ema_rebuilds_after_an_interior_bar_correction() {
        let config = EmaCrossConfig {
            fast_length: 2,
            slow_length: 4,
            ..EmaCrossConfig::default()
        };
        let mut bars = (0..24)
            .map(|idx| bar(idx + 1, 20.0 + (idx as f64 * 0.7).sin() * 3.0))
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let _ = config.evaluate_streaming(&mut runtime, &bars, None);

        // Keep the latest timestamp/value unchanged while revising a retained
        // historical bar. The digest must invalidate the incremental state.
        bars[7].close += 4.0;
        let expected = config.evaluate(&bars, None);
        let actual = config.evaluate_streaming(&mut runtime, &bars, None);
        assert_eq!(actual.signal, expected.signal);
        assert_eq!(actual.raw_buy_signal, expected.raw_buy_signal);
        assert_eq!(actual.raw_sell_signal, expected.raw_sell_signal);
        assert_eq!(actual.previous_fast_ema, expected.previous_fast_ema);
        assert_eq!(actual.previous_slow_ema, expected.previous_slow_ema);
        assert_eq!(actual.fast_ema, expected.fast_ema);
        assert_eq!(actual.slow_ema, expected.slow_ema);
    }

    #[test]
    fn streaming_ema_matches_batch_when_the_history_window_slides() {
        let config = EmaCrossConfig {
            fast_length: 3,
            slow_length: 9,
            ..EmaCrossConfig::default()
        };
        let bars = (0..512)
            .map(|idx| {
                let close = 100.0 + (idx as f64 * 0.17).sin() * 3.0 + idx as f64 * 0.01;
                bar(idx + 1, close)
            })
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let window_len = 64;
        for end in window_len..=bars.len() {
            let window = &bars[end - window_len..end];
            let expected = config.evaluate(window, None);
            let actual = config.evaluate_streaming(&mut runtime, window, None);
            assert_eq!(actual.signal, expected.signal, "window ending at {end}");
            assert_eq!(actual.raw_buy_signal, expected.raw_buy_signal);
            assert_eq!(actual.raw_sell_signal, expected.raw_sell_signal);
            for (actual, expected) in [
                (actual.previous_fast_ema, expected.previous_fast_ema),
                (actual.previous_slow_ema, expected.previous_slow_ema),
                (actual.fast_ema, expected.fast_ema),
                (actual.slow_ema, expected.slow_ema),
            ] {
                match (actual, expected) {
                    (Some(actual), Some(expected)) => {
                        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}")
                    }
                    (None, None) => {}
                    _ => panic!("streaming/batch EMA presence mismatch"),
                }
            }
        }
    }

    #[test]
    fn streaming_ema_sliding_window_preserves_crosses_on_volatile_values() {
        let config = EmaCrossConfig {
            fast_length: 10,
            slow_length: 30,
            ..EmaCrossConfig::default()
        };
        let mut seed = 0x1234_5678_u64;
        let bars = (0..12_000)
            .map(|idx| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let normalized = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
                bar(idx + 1, 7_500.0 + (normalized - 0.5) * 250.0)
            })
            .collect::<Vec<_>>();
        let mut runtime = EmaCrossExecutionState::default();
        let window_len = 4_096;
        for end in window_len..=bars.len() {
            let window = &bars[end - window_len..end];
            let expected = config.evaluate(window, None);
            let actual = config.evaluate_streaming(&mut runtime, window, None);
            assert_eq!(actual.signal, expected.signal, "window ending at {end}");
            assert_eq!(
                actual.raw_buy_signal, expected.raw_buy_signal,
                "window ending at {end}"
            );
            assert_eq!(
                actual.raw_sell_signal, expected.raw_sell_signal,
                "window ending at {end}"
            );
        }
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
                    volume: None,
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
