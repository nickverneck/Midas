use crate::broker::Bar;
use crate::strategies::ema_cross::ema_series;
use serde::{Deserialize, Serialize};

/// Optional same-direction entry filter driven by a slower EMA crossover.
///
/// The current fast/slow relationship defines the gate bias. It does not
/// create entries or alter protection for an open position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EmaCrossoverGateConfig {
    pub enabled: bool,
    pub fast_length: usize,
    pub slow_length: usize,
    /// Only enter on execution crosses while flat. Crosses during an open
    /// position are ignored so TP, SL, or trailing protection exits it.
    pub flat_only_entries: bool,
}

impl Default for EmaCrossoverGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fast_length: 50,
            slow_length: 200,
            flat_only_entries: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EmaCrossoverGateEvaluation {
    pub ready: bool,
    pub fast_ema: Option<f64>,
    pub slow_ema: Option<f64>,
    pub bullish: bool,
    pub bearish: bool,
}

impl EmaCrossoverGateConfig {
    pub fn warmup_bars(&self) -> usize {
        self.fast_length.max(self.slow_length).max(2) + 1
    }

    pub fn evaluate(&self, bars: &[Bar]) -> EmaCrossoverGateEvaluation {
        if !self.enabled || self.fast_length == 0 || self.slow_length == 0 {
            return EmaCrossoverGateEvaluation::default();
        }
        let Some(last_bar) = bars.last() else {
            return EmaCrossoverGateEvaluation::default();
        };
        if bars.len() < self.warmup_bars() {
            return EmaCrossoverGateEvaluation::default();
        }

        let closes: Vec<f64> = bars.iter().map(|bar| bar.close).collect();
        let fast = ema_series(&closes, self.fast_length).last().copied();
        let slow = ema_series(&closes, self.slow_length).last().copied();
        let (Some(fast_ema), Some(slow_ema)) = (fast, slow) else {
            return EmaCrossoverGateEvaluation::default();
        };
        if !last_bar.close.is_finite() || !fast_ema.is_finite() || !slow_ema.is_finite() {
            return EmaCrossoverGateEvaluation::default();
        }
        EmaCrossoverGateEvaluation {
            ready: fast_ema != slow_ema,
            fast_ema: Some(fast_ema),
            slow_ema: Some(slow_ema),
            bullish: fast_ema > slow_ema,
            bearish: fast_ema < slow_ema,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.enabled && (self.fast_length == 0 || self.slow_length == 0) {
            return Err(
                "directional EMA gate fast_length and slow_length must be greater than zero"
                    .to_string(),
            );
        }
        if self.enabled && self.fast_length >= self.slow_length {
            return Err(
                "directional EMA gate fast_length must be less than slow_length".to_string(),
            );
        }
        Ok(())
    }
}

/// Apply the gate to already-oriented execution cross signals. `buy` and
/// `sell` must reflect the execution strategy's inversion setting first.
pub fn filter_entry_signals(
    config: &EmaCrossoverGateConfig,
    gate: EmaCrossoverGateEvaluation,
    buy: bool,
    sell: bool,
) -> (bool, bool, Option<&'static str>) {
    if !config.enabled {
        return (buy, sell, None);
    }
    if !gate.ready {
        return (false, false, Some("directional_ema_gate_warming_up"));
    }
    match (gate.bullish, gate.bearish) {
        (true, false) => (
            buy,
            false,
            (sell && !buy).then_some("directional_ema_gate_blocks_short"),
        ),
        (false, true) => (
            false,
            sell,
            (buy && !sell).then_some("directional_ema_gate_blocks_long"),
        ),
        _ => (false, false, Some("directional_ema_gate_unresolved")),
    }
}

/// Apply the optional gate-specific rule that ignores execution crosses while
/// a position is open. Ordinary crossover strategies keep their existing
/// reverse-on-opposite-cross behavior when this option is disabled.
pub fn filter_position_entry_signals(
    config: &EmaCrossoverGateConfig,
    gate: EmaCrossoverGateEvaluation,
    buy: bool,
    sell: bool,
    position_open: bool,
) -> (bool, bool, Option<&'static str>) {
    let (buy, sell, reason) = filter_entry_signals(config, gate, buy, sell);
    if config.enabled && config.flat_only_entries && position_open && (buy || sell) {
        (
            false,
            false,
            Some("directional_ema_gate_ignores_cross_while_position_open"),
        )
    } else {
        (buy, sell, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts: i64, close: f64) -> Bar {
        Bar {
            ts_ns: ts,
            open: close,
            high: close,
            low: close,
            close,
            volume: Some(1.0),
        }
    }

    #[test]
    fn bullish_gate_allows_only_long_entries() {
        let config = EmaCrossoverGateConfig {
            enabled: true,
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossoverGateConfig::default()
        };
        let bars = [bar(1, 10.0), bar(2, 11.0), bar(3, 12.0), bar(4, 13.0)];
        let gate = config.evaluate(&bars);
        assert!(gate.ready && gate.bullish);
        assert_eq!(
            filter_entry_signals(&config, gate, true, false),
            (true, false, None)
        );
        assert_eq!(
            filter_entry_signals(&config, gate, false, true),
            (false, false, Some("directional_ema_gate_blocks_short"))
        );
    }

    #[test]
    fn bearish_gate_allows_only_short_entries_and_inversion_is_applied_before_filtering() {
        let config = EmaCrossoverGateConfig {
            enabled: true,
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossoverGateConfig::default()
        };
        let bars = [bar(1, 13.0), bar(2, 12.0), bar(3, 11.0), bar(4, 10.0)];
        let gate = config.evaluate(&bars);
        assert!(gate.ready && gate.bearish);
        // A raw buy cross on an inverted execution strategy is an effective
        // short, and the bearish gate permits that effective direction.
        assert_eq!(
            filter_entry_signals(&config, gate, false, true),
            (false, true, None)
        );
        assert_eq!(
            filter_entry_signals(&config, gate, true, false),
            (false, false, Some("directional_ema_gate_blocks_long"))
        );
    }

    #[test]
    fn gate_blocks_entries_until_both_ema_periods_are_warm() {
        let config = EmaCrossoverGateConfig {
            enabled: true,
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossoverGateConfig::default()
        };
        let bars = [bar(1, 10.0), bar(2, 11.0), bar(3, 12.0)];
        let gate = config.evaluate(&bars);
        assert!(!gate.ready);
        assert_eq!(
            filter_entry_signals(&config, gate, true, false),
            (false, false, Some("directional_ema_gate_warming_up"))
        );
    }

    #[test]
    fn flat_only_mode_ignores_crosses_in_position_but_allows_entries_when_flat() {
        let config = EmaCrossoverGateConfig {
            enabled: true,
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossoverGateConfig::default()
        };
        let bullish_gate = EmaCrossoverGateEvaluation {
            ready: true,
            bullish: true,
            ..EmaCrossoverGateEvaluation::default()
        };
        assert_eq!(
            filter_position_entry_signals(&config, bullish_gate, true, false, true),
            (
                false,
                false,
                Some("directional_ema_gate_ignores_cross_while_position_open")
            )
        );
        assert_eq!(
            filter_position_entry_signals(&config, bullish_gate, true, false, false),
            (true, false, None)
        );

        let reverse_on_cross = EmaCrossoverGateConfig {
            flat_only_entries: false,
            ..config
        };
        assert_eq!(
            filter_position_entry_signals(&reverse_on_cross, bullish_gate, true, false, true),
            (true, false, None)
        );
    }
}
