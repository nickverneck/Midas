use crate::broker::{Bar, MarketHistoryUpdate};
use crate::strategies::adaptive_gate::{RegimeAdaptiveGateConfig, RegimeAdaptiveGateEvaluation};
use crate::strategies::ema_cross::{EmaCrossConfig, EmaCrossEvaluation, EmaCrossExecutionState};
use crate::strategies::hma_cross::{HmaCrossConfig, HmaCrossEvaluation, HmaCrossExecutionState};
use crate::strategies::orientation_gate::{EmaOrientationGateConfig, EmaOrientationGateEvaluation};
use crate::strategies::{PositionSide, StrategySignal};
use serde::{Deserialize, Serialize};

/// A small, reusable market-volume regime gate.
///
/// The current bar is compared with the mean volume of the preceding
/// `lookback_bars` bars.  A ratio below `invert_below_relative_volume` marks a
/// thin-liquidity regime and inverts the wrapped strategy's directional
/// signal.  The comparison deliberately excludes the current bar from the
/// baseline so a partial/live bar cannot raise its own reference volume.
///
/// This is an opt-in layer.  Strategies that do not wrap it, or wrappers with
/// no usable volume, retain their existing signal behavior.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VolumeRegimeConfig {
    pub lookback_bars: usize,
    pub invert_below_relative_volume: f64,
}

impl Default for VolumeRegimeConfig {
    fn default() -> Self {
        Self {
            lookback_bars: 30,
            invert_below_relative_volume: 0.60,
        }
    }
}

impl VolumeRegimeConfig {
    pub fn warmup_bars(&self) -> usize {
        self.lookback_bars.max(1) + 1
    }

    /// Return current volume divided by the mean of the preceding bars.
    /// Missing, non-positive, or non-finite volume is treated as unavailable.
    pub fn relative_volume(&self, bars: &[Bar]) -> Option<f64> {
        let current = bars.last()?.volume?;
        if !current.is_finite() || current <= 0.0 {
            return None;
        }

        let lookback = self.lookback_bars.max(1);
        let previous = bars
            .iter()
            .rev()
            .skip(1)
            .take(lookback)
            .filter_map(|bar| bar.volume)
            .filter(|volume| volume.is_finite() && *volume > 0.0)
            .collect::<Vec<_>>();
        if previous.len() < lookback {
            return None;
        }

        let mean = previous.iter().sum::<f64>() / previous.len() as f64;
        if !mean.is_finite() || mean <= 0.0 {
            return None;
        }
        Some(current / mean)
    }

    pub fn should_invert(&self, bars: &[Bar]) -> bool {
        self.relative_volume(bars)
            .is_some_and(|ratio| ratio < self.invert_below_relative_volume)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lookback_bars == 0 {
            return Err("volume lookback_bars must be greater than zero".to_string());
        }
        if !self.invert_below_relative_volume.is_finite() || self.invert_below_relative_volume < 0.0
        {
            return Err(
                "volume invert_below_relative_volume must be finite and non-negative".to_string(),
            );
        }
        Ok(())
    }
}

/// Separate native strategy that wraps the existing HMA crossover with the
/// volume gate.  The embedded HMA config is intentionally independent from
/// `native_hma_cross`, so enabling this strategy cannot alter the existing HMA
/// strategy or its live/replay runtime state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VolumeAdaptiveHmaCrossConfig {
    #[serde(flatten)]
    pub hma_cross: HmaCrossConfig,
    pub volume_regime: VolumeRegimeConfig,
    #[serde(default)]
    pub ema_gate: EmaOrientationGateConfig,
    /// Optional multi-feature causal orientation layer. Disabled by default
    /// so existing HMA configurations retain their exact behavior.
    #[serde(default)]
    pub adaptive_gate: RegimeAdaptiveGateConfig,
}

/// Separate native strategy that wraps the existing EMA crossover with the
/// optional relative-volume and price-versus-EMA orientation gates.  The
/// embedded EMA config is intentionally independent from `native_ema`, so
/// enabling this strategy cannot alter the existing EMA strategy or its
/// live/replay runtime state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VolumeAdaptiveEmaCrossConfig {
    #[serde(flatten)]
    pub ema_cross: EmaCrossConfig,
    pub volume_regime: VolumeRegimeConfig,
    #[serde(default)]
    pub ema_gate: EmaOrientationGateConfig,
    /// Optional multi-feature causal orientation layer.  It is disabled by
    /// default so existing live/replay configurations retain their behavior.
    #[serde(default)]
    pub adaptive_gate: RegimeAdaptiveGateConfig,
}

impl Default for VolumeAdaptiveEmaCrossConfig {
    fn default() -> Self {
        Self {
            ema_cross: EmaCrossConfig::default(),
            volume_regime: VolumeRegimeConfig::default(),
            ema_gate: EmaOrientationGateConfig::default(),
            adaptive_gate: RegimeAdaptiveGateConfig::default(),
        }
    }
}

impl Default for VolumeAdaptiveHmaCrossConfig {
    fn default() -> Self {
        Self {
            hma_cross: HmaCrossConfig::default(),
            volume_regime: VolumeRegimeConfig::default(),
            ema_gate: EmaOrientationGateConfig::default(),
            adaptive_gate: RegimeAdaptiveGateConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeAdaptiveHmaCrossEvaluation {
    pub hma: HmaCrossEvaluation,
    pub relative_volume: Option<f64>,
    pub volume_inverted: bool,
    pub ema_value: Option<f64>,
    pub ema_inverted: bool,
    pub ema_ready: bool,
    pub adaptive_gate: RegimeAdaptiveGateEvaluation,
}

impl VolumeAdaptiveHmaCrossEvaluation {
    pub fn signal(&self) -> StrategySignal {
        self.hma.signal
    }

    pub fn summary(&self) -> String {
        format!(
            "{} | Relative volume: {} | Volume gate: {} | EMA gate: {} | {}",
            self.hma.summary(),
            format_ratio(self.relative_volume),
            if self.volume_inverted {
                "inverted"
            } else {
                "normal"
            },
            if self.ema_inverted {
                "inverted"
            } else if self.ema_ready {
                "normal"
            } else {
                "warming_up"
            },
            self.adaptive_gate.summary(),
        )
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "{} | relative_volume={} | volume_gate_inverted={} | ema={} | ema_gate_inverted={} | ema_ready={} | {}",
            self.hma.debug_summary(),
            format_ratio(self.relative_volume),
            self.volume_inverted,
            self.ema_value
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            self.ema_inverted,
            self.ema_ready,
            self.adaptive_gate.summary(),
        )
    }

    pub fn gate_summary(&self) -> String {
        format!(
            "relative_volume={} volume_gate_inverted={} ema={} ema_gate_inverted={} ema_ready={} | {}",
            format_ratio(self.relative_volume),
            self.volume_inverted,
            self.ema_value
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            self.ema_inverted,
            self.ema_ready,
            self.adaptive_gate.summary(),
        )
    }
}

impl VolumeAdaptiveHmaCrossConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.hma_cross.uses_native_protection()
    }

    pub fn warmup_bars(&self) -> usize {
        self.hma_cross
            .warmup_bars()
            .max(self.volume_regime.warmup_bars())
            .max(self.ema_gate.warmup_bars())
            .max(
                self.adaptive_gate
                    .warmup_bars(self.hma_cross.fast_length, self.hma_cross.slow_length),
            )
    }

    pub fn evaluate(
        &self,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> VolumeAdaptiveHmaCrossEvaluation {
        if self.adaptive_gate.enabled {
            let base = self.hma_cross.evaluate(bars, current_side);
            if !base.raw_buy_signal && !base.raw_sell_signal {
                return self.wrap_without_adaptive(base, bars);
            }
        }
        let (effective, relative_volume, volume_inverted, ema_gate, adaptive_gate) =
            self.effective_hma(bars);
        let mut hma = effective.evaluate(bars, current_side);
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_hma(&mut hma);
        }
        if adaptive_gate.should_hold() {
            hold_for_adaptive_gate_hma(&mut hma, adaptive_gate.hold_reason);
        }
        VolumeAdaptiveHmaCrossEvaluation {
            hma,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }

    pub fn evaluate_current_cross(
        &self,
        runtime: &mut HmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> VolumeAdaptiveHmaCrossEvaluation {
        if self.adaptive_gate.enabled {
            // Probe on a clone so the wrapper can preserve the stateful HMA
            // side tracker on non-crossing bars without advancing it twice
            // on a real cross.
            let mut probe_runtime = runtime.clone();
            let base =
                self.hma_cross
                    .evaluate_current_cross(&mut probe_runtime, bars, current_side);
            if !base.raw_buy_signal && !base.raw_sell_signal {
                *runtime = probe_runtime;
                return self.wrap_without_adaptive(base, bars);
            }
        }
        let (effective, relative_volume, volume_inverted, ema_gate, adaptive_gate) =
            self.effective_hma(bars);
        let mut hma = effective.evaluate_current_cross(runtime, bars, current_side);
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_hma(&mut hma);
        }
        if adaptive_gate.should_hold() {
            hold_for_adaptive_gate_hma(&mut hma, adaptive_gate.hold_reason);
        }
        VolumeAdaptiveHmaCrossEvaluation {
            hma,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }

    pub fn sync_position(
        &self,
        runtime: &mut HmaCrossExecutionState,
        signed_qty: i32,
        entry_price: Option<f64>,
    ) {
        self.hma_cross
            .sync_position(runtime, signed_qty, entry_price);
    }

    pub fn take_profit_offset(&self, tick_size: Option<f64>) -> Option<f64> {
        self.hma_cross.take_profit_offset(tick_size)
    }

    pub fn desired_trailing_stop_price(
        &self,
        runtime: &mut HmaCrossExecutionState,
        bar: &Bar,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        self.hma_cross
            .desired_trailing_stop_price(runtime, bar, tick_size)
    }

    pub fn current_effective_stop_price(
        &self,
        runtime: &HmaCrossExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        self.hma_cross
            .current_effective_stop_price(runtime, tick_size)
    }

    fn effective_hma(
        &self,
        bars: &[Bar],
    ) -> (
        HmaCrossConfig,
        Option<f64>,
        bool,
        EmaOrientationGateEvaluation,
        RegimeAdaptiveGateEvaluation,
    ) {
        let relative_volume = self.volume_regime.relative_volume(bars);
        let volume_inverted = relative_volume
            .is_some_and(|ratio| ratio < self.volume_regime.invert_below_relative_volume);
        let ema_gate = self.ema_gate.evaluate(bars);
        let adaptive_gate = self.adaptive_gate.evaluate(
            bars,
            self.hma_cross.fast_length,
            self.hma_cross.slow_length,
        );
        let mut effective = self.hma_cross.clone();
        effective.inverted ^= volume_inverted ^ ema_gate.inverted ^ adaptive_gate.inverted;
        (
            effective,
            relative_volume,
            volume_inverted,
            ema_gate,
            adaptive_gate,
        )
    }

    fn wrap_without_adaptive(
        &self,
        mut hma: HmaCrossEvaluation,
        bars: &[Bar],
    ) -> VolumeAdaptiveHmaCrossEvaluation {
        let relative_volume = self.volume_regime.relative_volume(bars);
        let volume_inverted = relative_volume
            .is_some_and(|ratio| ratio < self.volume_regime.invert_below_relative_volume);
        let ema_gate = if self.ema_gate.enabled {
            self.ema_gate.evaluate(bars)
        } else {
            EmaOrientationGateEvaluation::default()
        };
        let adaptive_gate = self.adaptive_gate.evaluate(
            &[],
            self.hma_cross.fast_length,
            self.hma_cross.slow_length,
        );
        hma.inverted ^= volume_inverted ^ ema_gate.inverted;
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_hma(&mut hma);
        }
        VolumeAdaptiveHmaCrossEvaluation {
            hma,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeAdaptiveEmaCrossEvaluation {
    pub ema: EmaCrossEvaluation,
    pub relative_volume: Option<f64>,
    pub volume_inverted: bool,
    pub ema_value: Option<f64>,
    pub ema_inverted: bool,
    pub ema_ready: bool,
    pub adaptive_gate: RegimeAdaptiveGateEvaluation,
}

impl VolumeAdaptiveEmaCrossEvaluation {
    pub fn signal(&self) -> StrategySignal {
        self.ema.signal
    }

    pub fn summary(&self) -> String {
        format!(
            "{} | Relative volume: {} | Volume gate: {} | EMA gate: {} | {}",
            self.ema.summary(),
            format_ratio(self.relative_volume),
            if self.volume_inverted {
                "inverted"
            } else {
                "normal"
            },
            if self.ema_inverted {
                "inverted"
            } else if self.ema_ready {
                "normal"
            } else {
                "warming_up"
            },
            self.adaptive_gate.summary(),
        )
    }

    pub fn debug_summary(&self) -> String {
        format!(
            "{} | relative_volume={} | volume_gate_inverted={} | ema={} | ema_gate_inverted={} | ema_ready={} | {}",
            self.ema.debug_summary(),
            format_ratio(self.relative_volume),
            self.volume_inverted,
            self.ema_value
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            self.ema_inverted,
            self.ema_ready,
            self.adaptive_gate.summary(),
        )
    }

    pub fn gate_summary(&self) -> String {
        format!(
            "relative_volume={} volume_gate_inverted={} ema={} ema_gate_inverted={} ema_ready={} | {}",
            format_ratio(self.relative_volume),
            self.volume_inverted,
            self.ema_value
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            self.ema_inverted,
            self.ema_ready,
            self.adaptive_gate.summary(),
        )
    }
}

impl VolumeAdaptiveEmaCrossConfig {
    pub fn uses_native_protection(&self) -> bool {
        self.ema_cross.uses_native_protection()
    }

    pub fn warmup_bars(&self) -> usize {
        self.ema_cross
            .warmup_bars()
            .max(self.volume_regime.warmup_bars())
            .max(self.ema_gate.warmup_bars())
            .max(
                self.adaptive_gate
                    .warmup_bars(self.ema_cross.fast_length, self.ema_cross.slow_length),
            )
    }

    pub fn evaluate(
        &self,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> VolumeAdaptiveEmaCrossEvaluation {
        // The adaptive layer is intentionally evaluated only at a decision
        // point.  A full-prefix feature reconstruction on every non-crossing
        // bar would make an enabled research gate needlessly quadratic.
        if self.adaptive_gate.enabled {
            let base = self.ema_cross.evaluate(bars, current_side);
            if !base.raw_buy_signal && !base.raw_sell_signal {
                return self.wrap_without_adaptive(base, bars);
            }
        }
        let (effective, relative_volume, volume_inverted, ema_gate, adaptive_gate) =
            self.effective_ema(bars);
        let mut ema = effective.evaluate(bars, current_side);
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_ema(&mut ema);
        }
        if adaptive_gate.should_hold() {
            hold_for_adaptive_gate_ema(&mut ema, adaptive_gate.hold_reason);
        }
        VolumeAdaptiveEmaCrossEvaluation {
            ema,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }

    pub fn evaluate_current_cross(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> VolumeAdaptiveEmaCrossEvaluation {
        if self.adaptive_gate.enabled {
            let base = self
                .ema_cross
                .evaluate_streaming(runtime, bars, current_side);
            if !base.raw_buy_signal && !base.raw_sell_signal {
                return self.wrap_without_adaptive(base, bars);
            }
        }
        let (effective, relative_volume, volume_inverted, ema_gate, adaptive_gate) =
            self.effective_ema(bars);
        let mut ema = effective.evaluate_streaming(runtime, bars, current_side);
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_ema(&mut ema);
        }
        if adaptive_gate.should_hold() {
            hold_for_adaptive_gate_ema(&mut ema, adaptive_gate.hold_reason);
        }
        VolumeAdaptiveEmaCrossEvaluation {
            ema,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }

    pub fn evaluate_streaming(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
    ) -> VolumeAdaptiveEmaCrossEvaluation {
        self.evaluate_streaming_with_market_update(
            runtime,
            bars,
            current_side,
            None,
            MarketHistoryUpdate::Snapshot,
        )
    }

    pub fn evaluate_streaming_with_market_update(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bars: &[Bar],
        current_side: Option<PositionSide>,
        source_update_sequence: Option<u64>,
        market_update: MarketHistoryUpdate,
    ) -> VolumeAdaptiveEmaCrossEvaluation {
        if self.adaptive_gate.enabled {
            let base = self.ema_cross.evaluate_streaming_with_market_update(
                runtime,
                bars,
                current_side,
                source_update_sequence,
                market_update,
            );
            if !base.raw_buy_signal && !base.raw_sell_signal {
                return self.wrap_without_adaptive(base, bars);
            }
        }
        let (effective, relative_volume, volume_inverted, ema_gate, adaptive_gate) =
            self.effective_ema(bars);
        let mut ema = effective.evaluate_streaming_with_market_update(
            runtime,
            bars,
            current_side,
            source_update_sequence,
            market_update,
        );
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_ema(&mut ema);
        }
        if adaptive_gate.should_hold() {
            hold_for_adaptive_gate_ema(&mut ema, adaptive_gate.hold_reason);
        }
        VolumeAdaptiveEmaCrossEvaluation {
            ema,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }

    pub fn sync_position(
        &self,
        runtime: &mut EmaCrossExecutionState,
        signed_qty: i32,
        entry_price: Option<f64>,
    ) {
        self.ema_cross
            .sync_position(runtime, signed_qty, entry_price);
    }

    pub fn take_profit_offset(&self, tick_size: Option<f64>) -> Option<f64> {
        self.ema_cross.take_profit_offset(tick_size)
    }

    pub fn desired_trailing_stop_price(
        &self,
        runtime: &mut EmaCrossExecutionState,
        bar: &Bar,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        self.ema_cross
            .desired_trailing_stop_price(runtime, bar, tick_size)
    }

    pub fn current_effective_stop_price(
        &self,
        runtime: &EmaCrossExecutionState,
        tick_size: Option<f64>,
    ) -> Option<f64> {
        self.ema_cross
            .current_effective_stop_price(runtime, tick_size)
    }

    fn effective_ema(
        &self,
        bars: &[Bar],
    ) -> (
        EmaCrossConfig,
        Option<f64>,
        bool,
        EmaOrientationGateEvaluation,
        RegimeAdaptiveGateEvaluation,
    ) {
        let relative_volume = self.volume_regime.relative_volume(bars);
        let volume_inverted = relative_volume
            .is_some_and(|ratio| ratio < self.volume_regime.invert_below_relative_volume);
        let ema_gate = self.ema_gate.evaluate(bars);
        let adaptive_gate = self.adaptive_gate.evaluate(
            bars,
            self.ema_cross.fast_length,
            self.ema_cross.slow_length,
        );
        let mut effective = self.ema_cross.clone();
        effective.inverted ^= volume_inverted ^ ema_gate.inverted ^ adaptive_gate.inverted;
        (
            effective,
            relative_volume,
            volume_inverted,
            ema_gate,
            adaptive_gate,
        )
    }

    fn wrap_without_adaptive(
        &self,
        mut ema: EmaCrossEvaluation,
        bars: &[Bar],
    ) -> VolumeAdaptiveEmaCrossEvaluation {
        let relative_volume = self.volume_regime.relative_volume(bars);
        let volume_inverted = relative_volume
            .is_some_and(|ratio| ratio < self.volume_regime.invert_below_relative_volume);
        let ema_gate = if self.ema_gate.enabled {
            self.ema_gate.evaluate(bars)
        } else {
            EmaOrientationGateEvaluation::default()
        };
        let adaptive_gate = self.adaptive_gate.evaluate(
            &[],
            self.ema_cross.fast_length,
            self.ema_cross.slow_length,
        );
        ema.inverted ^= volume_inverted ^ ema_gate.inverted;
        if self.ema_gate.enabled && !ema_gate.ready {
            hold_for_ema_gate_ema(&mut ema);
        }
        VolumeAdaptiveEmaCrossEvaluation {
            ema,
            relative_volume,
            volume_inverted,
            ema_value: ema_gate.ema,
            ema_inverted: ema_gate.inverted,
            ema_ready: ema_gate.ready,
            adaptive_gate,
        }
    }
}

fn hold_for_ema_gate_hma(evaluation: &mut HmaCrossEvaluation) {
    evaluation.signal = StrategySignal::Hold;
    evaluation.effective_buy_signal = false;
    evaluation.effective_sell_signal = false;
    evaluation.hold_reason = Some("ema_gate_warming_up");
}

fn hold_for_ema_gate_ema(evaluation: &mut EmaCrossEvaluation) {
    evaluation.signal = StrategySignal::Hold;
    evaluation.effective_buy_signal = false;
    evaluation.effective_sell_signal = false;
    evaluation.hold_reason = Some("ema_gate_warming_up");
}

fn hold_for_adaptive_gate_hma(evaluation: &mut HmaCrossEvaluation, reason: Option<&'static str>) {
    evaluation.signal = StrategySignal::Hold;
    evaluation.effective_buy_signal = false;
    evaluation.effective_sell_signal = false;
    evaluation.hold_reason = Some(reason.unwrap_or("adaptive_gate"));
}

fn hold_for_adaptive_gate_ema(evaluation: &mut EmaCrossEvaluation, reason: Option<&'static str>) {
    evaluation.signal = StrategySignal::Hold;
    evaluation.effective_buy_signal = false;
    evaluation.effective_sell_signal = false;
    evaluation.hold_reason = Some(reason.unwrap_or("adaptive_gate"));
}

fn format_ratio(value: Option<f64>) -> String {
    value
        .map(|ratio| format!("{ratio:.3}x"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, close: f64, volume: Option<f64>) -> Bar {
        Bar {
            ts_ns,
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume,
        }
    }

    #[test]
    fn relative_volume_excludes_current_bar_from_baseline() {
        let config = VolumeRegimeConfig {
            lookback_bars: 3,
            invert_below_relative_volume: 0.6,
        };
        let bars = vec![
            bar(1, 1.0, Some(100.0)),
            bar(2, 1.0, Some(100.0)),
            bar(3, 1.0, Some(100.0)),
            bar(4, 1.0, Some(50.0)),
        ];
        assert_eq!(config.relative_volume(&bars), Some(0.5));
        assert!(config.should_invert(&bars));
    }

    #[test]
    fn missing_volume_keeps_normal_orientation() {
        let config = VolumeRegimeConfig {
            lookback_bars: 2,
            ..VolumeRegimeConfig::default()
        };
        let bars = vec![
            bar(1, 1.0, Some(100.0)),
            bar(2, 1.0, None),
            bar(3, 1.0, Some(50.0)),
        ];
        assert_eq!(config.relative_volume(&bars), None);
        assert!(!config.should_invert(&bars));
    }

    #[test]
    fn disabled_ema_wrapper_matches_the_native_ema_evaluation() {
        let ema_cross = EmaCrossConfig {
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossConfig::default()
        };
        let wrapper = VolumeAdaptiveEmaCrossConfig {
            ema_cross: ema_cross.clone(),
            volume_regime: VolumeRegimeConfig {
                lookback_bars: 2,
                invert_below_relative_volume: 0.0,
            },
            ema_gate: EmaOrientationGateConfig::default(),
            adaptive_gate: RegimeAdaptiveGateConfig::default(),
        };
        let bars = vec![
            bar(1, 1.0, Some(100.0)),
            bar(2, 2.0, Some(100.0)),
            bar(3, 1.0, Some(100.0)),
            bar(4, 3.0, Some(100.0)),
            bar(5, 4.0, Some(100.0)),
        ];

        let expected = ema_cross.evaluate(&bars, None);
        let actual = wrapper.evaluate(&bars, None).ema;
        assert_eq!(actual.signal, expected.signal);
        assert_eq!(actual.raw_buy_signal, expected.raw_buy_signal);
        assert_eq!(actual.raw_sell_signal, expected.raw_sell_signal);
        assert_eq!(actual.effective_buy_signal, expected.effective_buy_signal);
        assert_eq!(actual.effective_sell_signal, expected.effective_sell_signal);
        assert_eq!(actual.fast_ema, expected.fast_ema);
        assert_eq!(actual.slow_ema, expected.slow_ema);
        assert!(!wrapper.evaluate(&bars, None).volume_inverted);
        assert!(!wrapper.evaluate(&bars, None).ema_ready);
    }

    #[test]
    fn low_relative_volume_inverts_only_the_wrapper_signal() {
        let ema_cross = EmaCrossConfig {
            fast_length: 2,
            slow_length: 3,
            ..EmaCrossConfig::default()
        };
        let wrapper = VolumeAdaptiveEmaCrossConfig {
            ema_cross: ema_cross.clone(),
            volume_regime: VolumeRegimeConfig {
                lookback_bars: 2,
                invert_below_relative_volume: 0.5,
            },
            ema_gate: EmaOrientationGateConfig::default(),
            adaptive_gate: RegimeAdaptiveGateConfig::default(),
        };
        let bars = vec![
            bar(1, 1.0, Some(100.0)),
            bar(2, 2.0, Some(100.0)),
            bar(3, 1.0, Some(100.0)),
            bar(4, 3.0, Some(100.0)),
            bar(5, 4.0, Some(10.0)),
        ];

        let expected = ema_cross.evaluate(&bars, None);
        let actual = wrapper.evaluate(&bars, None);
        assert_eq!(actual.relative_volume, Some(0.1));
        assert!(actual.volume_inverted);
        assert_eq!(actual.ema.inverted, !expected.inverted);
        assert_eq!(actual.ema.raw_buy_signal, expected.raw_buy_signal);
        assert_eq!(actual.ema.raw_sell_signal, expected.raw_sell_signal);
        let expected_inverted_signal = match expected.signal {
            StrategySignal::EnterLong => StrategySignal::EnterShort,
            StrategySignal::EnterShort => StrategySignal::EnterLong,
            other => other,
        };
        assert_eq!(actual.ema.signal, expected_inverted_signal);
    }
}
