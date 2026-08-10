use crate::broker::Bar;
use crate::strategies::ema_cross::ema_series;
use serde::{Deserialize, Serialize};

/// Optional price-versus-EMA orientation gate.
///
/// When enabled, a close above the reference EMA marks the wrapped strategy
/// as inverted; a close below it keeps the wrapped strategy in its normal
/// orientation.  The gate only supplies an orientation decision.  It never
/// creates an entry or forces a position exit by itself, which makes it safe
/// to layer over HMA, EMA, ADX, or future native strategies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmaOrientationGateConfig {
    /// Disabled by default so existing strategies retain their exact behavior.
    pub enabled: bool,
    /// Number of bars used by the reference EMA.
    pub ema_length: usize,
    /// If true, close > EMA inverts.  This is the requested default rule.
    #[serde(default = "default_invert_when_above")]
    pub invert_when_above: bool,
}

fn default_invert_when_above() -> bool {
    true
}

impl Default for EmaOrientationGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ema_length: 500,
            invert_when_above: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EmaOrientationGateEvaluation {
    pub latest_close: Option<f64>,
    pub ema: Option<f64>,
    pub inverted: bool,
    pub ready: bool,
}

impl EmaOrientationGateConfig {
    pub fn warmup_bars(&self) -> usize {
        if self.enabled {
            self.ema_length.max(1) + 1
        } else {
            0
        }
    }

    pub fn evaluate(&self, bars: &[Bar]) -> EmaOrientationGateEvaluation {
        let latest_close = bars.last().map(|bar| bar.close).filter(|v| v.is_finite());
        if !self.enabled || self.ema_length == 0 || bars.len() < self.warmup_bars() {
            return EmaOrientationGateEvaluation {
                latest_close,
                ..Default::default()
            };
        }

        let closes = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let ema = ema_series(&closes, self.ema_length)
            .last()
            .copied()
            .filter(|value| value.is_finite());
        let Some((close, ema_value)) = latest_close.zip(ema) else {
            return EmaOrientationGateEvaluation {
                latest_close,
                ..Default::default()
            };
        };

        let above = close > ema_value;
        let below = close < ema_value;
        let inverted = if self.invert_when_above { above } else { below };
        EmaOrientationGateEvaluation {
            latest_close,
            ema: Some(ema_value),
            inverted,
            ready: above || below,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.enabled && self.ema_length == 0 {
            return Err(
                "EMA orientation gate ema_length must be greater than zero when enabled"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open: close,
            high: close,
            low: close,
            close,
            volume: Some(1.0),
        }
    }

    #[test]
    fn disabled_gate_is_a_noop() {
        let config = EmaOrientationGateConfig::default();
        let evaluation = config.evaluate(&[bar(1, 10.0), bar(2, 11.0)]);
        assert!(!evaluation.ready);
        assert!(!evaluation.inverted);
        assert_eq!(evaluation.ema, None);
    }

    #[test]
    fn close_above_reference_ema_inverts_after_warmup() {
        let config = EmaOrientationGateConfig {
            enabled: true,
            ema_length: 2,
            invert_when_above: true,
        };
        let evaluation = config.evaluate(&[bar(1, 10.0), bar(2, 10.0), bar(3, 12.0)]);
        assert!(evaluation.ready);
        assert!(evaluation.inverted);
        assert!(evaluation.ema.is_some());
    }

    #[test]
    fn close_below_reference_ema_keeps_normal_orientation() {
        let config = EmaOrientationGateConfig {
            enabled: true,
            ema_length: 2,
            invert_when_above: true,
        };
        let evaluation = config.evaluate(&[bar(1, 12.0), bar(2, 12.0), bar(3, 10.0)]);
        assert!(evaluation.ready);
        assert!(!evaluation.inverted);
    }
}
