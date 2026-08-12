use crate::broker::{Bar, HeikinAshiState};
use crate::strategies::{PositionSide, StrategySignal};
use serde::{Deserialize, Serialize};

/// The direction of a completed Heikin-Ashi candle.
///
/// A candle whose Heikin-Ashi close equals its open is deliberately neutral;
/// it does not count as either a green or red transition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeikinAshiColor {
    Green,
    Red,
    Neutral,
}

impl HeikinAshiColor {
    pub fn label(self) -> &'static str {
        match self {
            Self::Green => "green",
            Self::Red => "red",
            Self::Neutral => "neutral",
        }
    }
}

/// Native Heikin-Ashi color-transition strategy configuration.
///
/// `evaluate` expects completed Heikin-Ashi bars, such as the closed-bar
/// series produced by the market layer when Heikin-Ashi candle mode is
/// selected. Use [`Self::evaluate_from_source_bars`] when the caller has raw
/// completed OHLC bars instead.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HeikinAshiConfig {
    /// Swap green-transition buys and red-transition sells when enabled.
    pub inverted: bool,
}

impl Default for HeikinAshiConfig {
    fn default() -> Self {
        Self { inverted: false }
    }
}

/// Diagnostics for one completed-bar Heikin-Ashi evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct HeikinAshiEvaluation {
    pub signal: StrategySignal,
    pub previous_color: Option<HeikinAshiColor>,
    pub current_color: Option<HeikinAshiColor>,
    pub previous_ha_open: Option<f64>,
    pub previous_ha_close: Option<f64>,
    pub current_ha_open: Option<f64>,
    pub current_ha_close: Option<f64>,
    pub latest_ts_ns: Option<i64>,
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

impl HeikinAshiEvaluation {
    pub fn summary(&self) -> String {
        format!(
            "Signal: {} | HA: {} -> {} | close: {}",
            self.signal.label(),
            color_label(self.previous_color),
            color_label(self.current_color),
            fmt_price(self.current_ha_close),
        )
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "Signal: {} | reason: {} | bars: {}/{} | side: {:?} | inverted: {} | colors: {} -> {} | prev_open: {} | prev_close: {} | current_open: {} | current_close: {} | raw_transition buy={} sell={} | effective_transition buy={} sell={}",
            self.signal.label(),
            self.hold_reason.unwrap_or("signal_ready"),
            self.bars_len,
            self.warmup_bars,
            self.current_side,
            self.inverted,
            color_label(self.previous_color),
            color_label(self.current_color),
            fmt_price(self.previous_ha_open),
            fmt_price(self.previous_ha_close),
            fmt_price(self.current_ha_open),
            fmt_price(self.current_ha_close),
            self.raw_buy_signal,
            self.raw_sell_signal,
            self.effective_buy_signal,
            self.effective_sell_signal,
        )
    }
}

impl HeikinAshiConfig {
    /// Two completed candles are required to establish a color transition.
    pub fn warmup_bars(&self) -> usize {
        2
    }

    /// Evaluate the latest completed Heikin-Ashi candle.
    ///
    /// The slice must contain completed Heikin-Ashi bars only. A green
    /// transition is a non-green previous candle followed by a green current
    /// candle. A red transition is defined symmetrically. Same-color candles,
    /// doji candles, and insufficient history produce `Hold`.
    pub fn evaluate(
        &self,
        completed_heikin_ashi_bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> HeikinAshiEvaluation {
        let latest = completed_heikin_ashi_bars.last();
        let previous = completed_heikin_ashi_bars
            .len()
            .checked_sub(2)
            .and_then(|index| completed_heikin_ashi_bars.get(index));
        let warmup_bars = self.warmup_bars();

        let mut evaluation = HeikinAshiEvaluation {
            signal: StrategySignal::Hold,
            previous_color: previous.and_then(heikin_ashi_color),
            current_color: latest.and_then(heikin_ashi_color),
            previous_ha_open: previous.and_then(|bar| finite(bar.open)),
            previous_ha_close: previous.and_then(|bar| finite(bar.close)),
            current_ha_open: latest.and_then(|bar| finite(bar.open)),
            current_ha_close: latest.and_then(|bar| finite(bar.close)),
            latest_ts_ns: latest.map(|bar| bar.ts_ns),
            bars_len: completed_heikin_ashi_bars.len(),
            warmup_bars,
            current_side,
            inverted: self.inverted,
            raw_buy_signal: false,
            raw_sell_signal: false,
            effective_buy_signal: false,
            effective_sell_signal: false,
            hold_reason: None,
        };

        if completed_heikin_ashi_bars.is_empty() {
            evaluation.hold_reason = Some("no_bars");
            return evaluation;
        }

        if completed_heikin_ashi_bars.len() < warmup_bars {
            evaluation.hold_reason = Some("warming_up");
            return evaluation;
        }

        let Some(current_color) = evaluation.current_color else {
            evaluation.hold_reason = Some("non_finite_current_heikin_ashi");
            return evaluation;
        };
        let Some(previous_color) = evaluation.previous_color else {
            evaluation.hold_reason = Some("non_finite_previous_heikin_ashi");
            return evaluation;
        };

        evaluation.raw_buy_signal =
            current_color == HeikinAshiColor::Green && previous_color != HeikinAshiColor::Green;
        evaluation.raw_sell_signal =
            current_color == HeikinAshiColor::Red && previous_color != HeikinAshiColor::Red;

        if self.inverted {
            evaluation.effective_buy_signal = evaluation.raw_sell_signal;
            evaluation.effective_sell_signal = evaluation.raw_buy_signal;
        } else {
            evaluation.effective_buy_signal = evaluation.raw_buy_signal;
            evaluation.effective_sell_signal = evaluation.raw_sell_signal;
        }

        evaluation.signal = resolve_signal(
            evaluation.effective_buy_signal,
            evaluation.effective_sell_signal,
            current_side,
        );
        evaluation.hold_reason = if evaluation.signal != StrategySignal::Hold {
            None
        } else if current_color == HeikinAshiColor::Neutral {
            Some("neutral_bar")
        } else if evaluation.effective_buy_signal && current_side == Some(PositionSide::Long) {
            Some("buy_transition_already_long")
        } else if evaluation.effective_sell_signal && current_side == Some(PositionSide::Short) {
            Some("sell_transition_already_short")
        } else {
            Some("no_color_transition")
        };

        evaluation
    }

    /// Derive Heikin-Ashi candles from completed source OHLC bars and then
    /// evaluate the resulting completed candle transition.
    ///
    /// This helper owns a fresh [`HeikinAshiState`] and is intended for callers
    /// that do not already have the market layer's cached transformed series.
    /// It never includes a forming bar because the input contract is a slice
    /// of completed source bars.
    pub fn evaluate_from_source_bars(
        &self,
        completed_source_bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> HeikinAshiEvaluation {
        let mut state = HeikinAshiState::new();
        let completed_heikin_ashi_bars = state.transform_all(completed_source_bars);
        self.evaluate(&completed_heikin_ashi_bars, current_side)
    }
}

fn heikin_ashi_color(bar: &Bar) -> Option<HeikinAshiColor> {
    if !bar.open.is_finite() || !bar.close.is_finite() {
        return None;
    }

    Some(if bar.close > bar.open {
        HeikinAshiColor::Green
    } else if bar.close < bar.open {
        HeikinAshiColor::Red
    } else {
        HeikinAshiColor::Neutral
    })
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
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

fn color_label(color: Option<HeikinAshiColor>) -> &'static str {
    color.map(HeikinAshiColor::label).unwrap_or("n/a")
}

fn fmt_price(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heikin_ashi_bar(ts_ns: i64, open: f64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open,
            high: open.max(close) + 1.0,
            low: open.min(close) - 1.0,
            close,
            volume: None,
        }
    }

    fn source_bar(ts_ns: i64, open: f64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open,
            high: open.max(close),
            low: open.min(close),
            close,
            volume: None,
        }
    }

    #[test]
    fn completed_color_transitions_are_causal_and_only_fire_on_edges() {
        let config = HeikinAshiConfig::default();
        let bars = vec![
            heikin_ashi_bar(1, 11.0, 9.0),  // red
            heikin_ashi_bar(2, 9.0, 11.0),  // red -> green
            heikin_ashi_bar(3, 10.0, 12.0), // green -> green
            heikin_ashi_bar(4, 12.0, 10.0), // green -> red
        ];

        let buy = config.evaluate(&bars[..2], None);
        assert_eq!(buy.signal, StrategySignal::EnterLong);
        assert_eq!(buy.previous_color, Some(HeikinAshiColor::Red));
        assert_eq!(buy.current_color, Some(HeikinAshiColor::Green));
        assert!(buy.raw_buy_signal);
        assert!(!buy.raw_sell_signal);

        let same_color = config.evaluate(&bars[..3], None);
        assert_eq!(same_color.signal, StrategySignal::Hold);
        assert_eq!(same_color.hold_reason, Some("no_color_transition"));
        assert!(!same_color.raw_buy_signal);
        assert!(!same_color.raw_sell_signal);

        let sell = config.evaluate(&bars, None);
        assert_eq!(sell.signal, StrategySignal::EnterShort);
        assert_eq!(sell.previous_color, Some(HeikinAshiColor::Green));
        assert_eq!(sell.current_color, Some(HeikinAshiColor::Red));
        assert!(sell.raw_sell_signal);
    }

    #[test]
    fn doji_bars_hold_but_do_not_block_a_later_transition() {
        let config = HeikinAshiConfig::default();
        let bars = vec![
            heikin_ashi_bar(1, 9.0, 11.0),  // green
            heikin_ashi_bar(2, 10.0, 10.0), // doji
            heikin_ashi_bar(3, 9.0, 11.0),  // doji -> green
        ];

        let doji = config.evaluate(&bars[..2], None);
        assert_eq!(doji.signal, StrategySignal::Hold);
        assert_eq!(doji.current_color, Some(HeikinAshiColor::Neutral));
        assert_eq!(doji.hold_reason, Some("neutral_bar"));

        let after_doji = config.evaluate(&bars, None);
        assert_eq!(after_doji.signal, StrategySignal::EnterLong);
        assert_eq!(after_doji.previous_color, Some(HeikinAshiColor::Neutral));
        assert!(after_doji.raw_buy_signal);
    }

    #[test]
    fn inversion_swaps_transition_directions() {
        let bars = vec![heikin_ashi_bar(1, 11.0, 9.0), heikin_ashi_bar(2, 9.0, 11.0)];

        let normal = HeikinAshiConfig::default().evaluate(&bars, None);
        assert_eq!(normal.signal, StrategySignal::EnterLong);
        assert!(normal.effective_buy_signal);
        assert!(!normal.effective_sell_signal);

        let inverted = HeikinAshiConfig { inverted: true }.evaluate(&bars, None);
        assert_eq!(inverted.signal, StrategySignal::EnterShort);
        assert!(!inverted.effective_buy_signal);
        assert!(inverted.effective_sell_signal);
        assert!(inverted.raw_buy_signal);
    }

    #[test]
    fn insufficient_history_holds_without_a_transition() {
        let config = HeikinAshiConfig::default();

        let empty = config.evaluate(&[], None);
        assert_eq!(empty.signal, StrategySignal::Hold);
        assert_eq!(empty.hold_reason, Some("no_bars"));
        assert_eq!(empty.current_color, None);

        let one = config.evaluate(&[heikin_ashi_bar(1, 9.0, 11.0)], None);
        assert_eq!(one.signal, StrategySignal::Hold);
        assert_eq!(one.hold_reason, Some("warming_up"));
        assert_eq!(one.current_color, Some(HeikinAshiColor::Green));
        assert_eq!(one.previous_color, None);
    }

    #[test]
    fn source_bar_helper_uses_recursive_completed_heikin_ashi_state() {
        let config = HeikinAshiConfig::default();
        let source_bars = vec![source_bar(1, 10.0, 10.0), source_bar(2, 10.0, 12.0)];

        let evaluation = config.evaluate_from_source_bars(&source_bars, None);
        assert_eq!(evaluation.previous_color, Some(HeikinAshiColor::Neutral));
        assert_eq!(evaluation.current_color, Some(HeikinAshiColor::Green));
        assert_eq!(evaluation.signal, StrategySignal::EnterLong);
    }
}
