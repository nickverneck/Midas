//! Causal, opt-in regime orientation gates for the EMA crossover wrapper.
//!
//! This module intentionally only decides whether the wrapped crossover should
//! be inverted.  It does not create orders, flatten positions, or mutate any
//! of the existing strategy defaults.  Every feature is evaluated at the
//! latest supplied bar and only uses bars at or before that bar, which makes
//! the gate usable by both replay and live closed-bar execution.

use crate::broker::Bar;
use crate::strategies::adx::adx_series;
use crate::strategies::ema_cross::ema_series;
use crate::strategies::hma_cross::hma_series_incremental;
use chrono::{DateTime, Datelike, Timelike, Utc};
use chrono_tz::America::New_York;
use serde::{Deserialize, Serialize};

/// How independent feature votes are combined into one orientation vote.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveGateCombine {
    Any,
    All,
    /// Invert only when an odd number of configured feature votes are true.
    /// This is useful for explicitly testing complementary signals without
    /// silently treating a two-feature tie as inversion.
    Xor,
    Majority,
}

impl Default for AdaptiveGateCombine {
    fn default() -> Self {
        Self::Majority
    }
}

/// Action when a raw lower-timeframe crossover conflicts with a strong,
/// persistent higher-timeframe direction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HigherTimeframeConflictAction {
    /// Reverse the raw crossover so it follows the higher-timeframe regime.
    Invert,
    /// Suppress the crossover while the conflict remains unresolved.
    Abstain,
}

impl Default for HigherTimeframeConflictAction {
    fn default() -> Self {
        Self::Invert
    }
}

/// Smoothing family used by the optional secondary higher-timeframe regime
/// pair.  EMA is the compatibility default; HMA is opt-in for faster regime
/// response while retaining the same causal completed-bucket mapping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HigherTimeframeAverageKind {
    Ema,
    Hma,
}

impl Default for HigherTimeframeAverageKind {
    fn default() -> Self {
        Self::Ema
    }
}

/// A causal orientation gate that can combine liquidity, volatility, trend,
/// crossover geometry, and New York session features.
///
/// The gate is deliberately disabled by default.  When enabled, each
/// `use_*` feature contributes a vote only when its current-bar inputs are
/// available.  Missing inputs are neutral (no inversion); callers may opt in
/// to holding entries while a required feature is unavailable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegimeAdaptiveGateConfig {
    /// Opt-in switch.  False is a strict no-op for existing strategies.
    pub enabled: bool,
    /// Combination rule for available feature votes.
    #[serde(default)]
    pub combine: AdaptiveGateCombine,
    /// Minimum number of usable feature votes required before a decision is
    /// considered ready.  Unavailable features are excluded, not treated as
    /// bullish or bearish.
    #[serde(default = "default_minimum_feature_votes")]
    pub minimum_feature_votes: usize,
    /// Consecutive inversion votes required before switching to inverted.
    #[serde(default = "default_confirmation_bars")]
    pub invert_confirmation_bars: usize,
    /// Consecutive normal votes required before switching back to normal.
    #[serde(default = "default_confirmation_bars")]
    pub normal_confirmation_bars: usize,
    /// If true, missing features produce a Hold at the wrapper.  If false,
    /// missing features are neutral and the wrapper keeps its base orientation.
    #[serde(default)]
    pub hold_on_missing_features: bool,
    /// If true, an unconfirmed orientation change produces a Hold.  The
    /// default keeps the previous orientation during the dwell period.
    #[serde(default)]
    pub hold_during_dwell: bool,
    /// If true, an unavailable feature resets the prior orientation to normal
    /// instead of retaining the previous confirmed inversion.
    #[serde(default = "default_reset_on_missing")]
    pub reset_on_missing_features: bool,

    /// Enable current volume / preceding-volume-mean comparison.
    #[serde(default)]
    pub use_relative_volume: bool,
    #[serde(default = "default_volume_lookback")]
    pub volume_lookback_bars: usize,
    /// Low relative volume votes for inversion.
    #[serde(default = "default_volume_cutoff")]
    pub invert_below_relative_volume: f64,
    /// Optional high relative-volume inversion threshold.  Keeping this
    /// separate from the low cutoff lets a sweep express either tail without
    /// turning the whole middle of the distribution into an inversion.
    #[serde(default)]
    pub invert_above_relative_volume: Option<f64>,

    /// Enable current ATR / preceding-ATR-mean comparison.
    #[serde(default)]
    pub use_atr_ratio: bool,
    #[serde(default = "default_atr_length")]
    pub atr_length: usize,
    #[serde(default = "default_atr_lookback")]
    pub atr_lookback_bars: usize,
    /// Low relative ATR votes for inversion.
    #[serde(default = "default_atr_cutoff")]
    pub invert_below_atr_ratio: f64,
    #[serde(default)]
    pub invert_above_atr_ratio: Option<f64>,

    /// Enable preceding-window Choppiness Index as an orientation feature.
    /// The window deliberately excludes the current bar, so a closed-bar
    /// decision cannot use the bar that triggered the crossover as its own
    /// regime reference.
    #[serde(default)]
    pub use_choppiness: bool,
    #[serde(default = "default_choppiness_length")]
    pub choppiness_length: usize,
    /// Low Choppiness Index votes for inversion.
    #[serde(default)]
    pub invert_below_choppiness: f64,
    /// Optional high Choppiness Index inversion threshold.
    #[serde(default)]
    pub invert_above_choppiness: Option<f64>,

    /// Enable a raw-cross-direction-aligned prior-return vote.  The return is
    /// measured over completed bars before the decision bar and normalized by
    /// the ATR at the end of that prior window.
    #[serde(default)]
    pub use_directional_return: bool,
    #[serde(default = "default_directional_return_length")]
    pub directional_return_length: usize,
    /// Optional lower aligned-return inversion threshold.
    #[serde(default)]
    pub invert_when_aligned_return_below: Option<f64>,
    /// Optional upper aligned-return inversion threshold.
    #[serde(default)]
    pub invert_when_aligned_return_above: Option<f64>,

    /// Enable ADX strength as an orientation feature.  Low ADX votes for
    /// inversion, which is useful for testing the observed chop/low-liquidity
    /// hypothesis.
    #[serde(default)]
    pub use_adx: bool,
    #[serde(default = "default_adx_length")]
    pub adx_length: usize,
    #[serde(default = "default_adx_cutoff")]
    pub invert_below_adx: f64,
    #[serde(default)]
    pub invert_above_adx: Option<f64>,
    /// Enable signed +DI/-DI imbalance.  Negative imbalance votes for
    /// inversion; this can be disabled independently from ADX strength.
    #[serde(default)]
    pub use_di_imbalance: bool,
    #[serde(default = "default_di_threshold")]
    pub invert_when_di_imbalance_below: f64,
    #[serde(default)]
    pub invert_when_di_imbalance_above: Option<f64>,

    /// Enable normalized fast/slow EMA spread.  The base EMA lengths are
    /// supplied by `VolumeAdaptiveEmaCrossConfig` at evaluation time and the
    /// spread is normalized by current ATR.
    #[serde(default)]
    pub use_ema_spread: bool,
    #[serde(default)]
    pub invert_when_normalized_spread_below: f64,
    #[serde(default)]
    pub invert_when_normalized_spread_above: Option<f64>,
    /// Enable normalized slow EMA slope over this many bars.
    #[serde(default)]
    pub use_ema_slope: bool,
    #[serde(default = "default_slope_lookback")]
    pub ema_slope_lookback: usize,
    #[serde(default)]
    pub invert_when_normalized_slope_below: f64,
    #[serde(default)]
    pub invert_when_normalized_slope_above: Option<f64>,

    /// Enable a raw-crossover-direction-aware long-EMA gap vote.  The raw
    /// EMA cross is computed before any orientation inversion; the gap is
    /// sampled from a prior completed bar so this feature cannot simply
    /// restate the current cross.  Bullish and bearish thresholds are
    /// independent, allowing a symmetric conflict rule or a deliberately
    /// one-sided research screen.
    #[serde(default)]
    pub use_directional_ema_gap: bool,
    #[serde(default = "default_directional_ema_length")]
    pub directional_ema_length: usize,
    #[serde(default = "default_directional_gap_lookback")]
    pub directional_gap_lookback_bars: usize,
    #[serde(default)]
    pub invert_when_bullish_gap_below: Option<f64>,
    #[serde(default)]
    pub invert_when_bullish_gap_above: Option<f64>,
    #[serde(default)]
    pub invert_when_bearish_gap_below: Option<f64>,
    #[serde(default)]
    pub invert_when_bearish_gap_above: Option<f64>,

    /// Enable a New York local-time session vote.  Start/end are minutes after
    /// midnight ET.  Equal values disable the window; start > end means an
    /// overnight window.  Outside the configured window the feature votes
    /// normal, so the session feature remains a complete causal signal.
    #[serde(default)]
    pub use_session_window: bool,
    #[serde(default)]
    pub session_start_minute_et: u16,
    #[serde(default)]
    pub session_end_minute_et: u16,
    #[serde(default)]
    pub invert_inside_session_window: bool,
    /// Monday bit 0 through Sunday bit 6.  Default is Monday-Friday.
    #[serde(default = "default_weekday_mask")]
    pub session_weekdays_mask: u8,

    /// Enable a causal higher-timeframe secondary regime context feature. The
    /// context is built from completed time buckets of the supplied
    /// lower-timeframe bars; the currently forming bucket is never used.
    #[serde(default)]
    pub use_higher_timeframe_context: bool,
    /// Smoothing family for the secondary regime pair.  EMA preserves the
    /// original behavior; HMA is a separate opt-in regime selector and does
    /// not alter the EMA 10/30 trigger.
    #[serde(default)]
    pub higher_timeframe_average_kind: HigherTimeframeAverageKind,
    #[serde(default = "default_higher_timeframe_minutes")]
    pub higher_timeframe_minutes: usize,
    #[serde(default = "default_higher_timeframe_fast_length")]
    pub higher_timeframe_fast_length: usize,
    #[serde(default = "default_higher_timeframe_slow_length")]
    pub higher_timeframe_slow_length: usize,
    #[serde(default = "default_higher_timeframe_atr_length")]
    pub higher_timeframe_atr_length: usize,
    #[serde(default = "default_higher_timeframe_slope_lookback")]
    pub higher_timeframe_slope_lookback: usize,
    #[serde(default = "default_higher_timeframe_persistence_bars")]
    pub higher_timeframe_persistence_bars: usize,
    /// Minimum absolute fast/slow separation in HTF ATR units required for a
    /// directional context state.
    #[serde(default)]
    pub higher_timeframe_spread_atr_floor: f64,
    /// Optional maximum absolute HTF separation for a directional context
    /// state. This is useful when testing whether a very strong, lagging HTF
    /// trend should be treated as unreliable during a lower-timeframe regime
    /// transition instead of automatically inverting every conflict.
    #[serde(default)]
    pub higher_timeframe_spread_atr_ceiling: Option<f64>,
    /// Minimum absolute slow-EMA slope in HTF ATR units per HTF bar required
    /// for a directional context state.
    #[serde(default)]
    pub higher_timeframe_slope_atr_floor: f64,
    #[serde(default)]
    pub higher_timeframe_conflict_action: HigherTimeframeConflictAction,
}

fn default_minimum_feature_votes() -> usize {
    1
}

fn default_confirmation_bars() -> usize {
    1
}

fn default_reset_on_missing() -> bool {
    true
}

fn default_volume_lookback() -> usize {
    30
}

fn default_volume_cutoff() -> f64 {
    0.60
}

fn default_atr_length() -> usize {
    14
}

fn default_atr_lookback() -> usize {
    30
}

fn default_atr_cutoff() -> f64 {
    0.60
}

fn default_choppiness_length() -> usize {
    30
}

fn default_directional_return_length() -> usize {
    60
}

fn default_adx_length() -> usize {
    14
}

fn default_adx_cutoff() -> f64 {
    20.0
}

fn default_di_threshold() -> f64 {
    0.10
}

fn default_slope_lookback() -> usize {
    3
}

fn default_directional_ema_length() -> usize {
    120
}

fn default_directional_gap_lookback() -> usize {
    1
}

fn default_weekday_mask() -> u8 {
    0b0001_1111
}

fn default_higher_timeframe_minutes() -> usize {
    15
}

fn default_higher_timeframe_fast_length() -> usize {
    8
}

fn default_higher_timeframe_slow_length() -> usize {
    21
}

fn default_higher_timeframe_atr_length() -> usize {
    14
}

fn default_higher_timeframe_slope_lookback() -> usize {
    3
}

fn default_higher_timeframe_persistence_bars() -> usize {
    2
}

impl Default for RegimeAdaptiveGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            combine: AdaptiveGateCombine::Majority,
            minimum_feature_votes: default_minimum_feature_votes(),
            invert_confirmation_bars: default_confirmation_bars(),
            normal_confirmation_bars: default_confirmation_bars(),
            hold_on_missing_features: false,
            hold_during_dwell: false,
            reset_on_missing_features: default_reset_on_missing(),
            use_relative_volume: false,
            volume_lookback_bars: default_volume_lookback(),
            invert_below_relative_volume: default_volume_cutoff(),
            invert_above_relative_volume: None,
            use_atr_ratio: false,
            atr_length: default_atr_length(),
            atr_lookback_bars: default_atr_lookback(),
            invert_below_atr_ratio: default_atr_cutoff(),
            invert_above_atr_ratio: None,
            use_choppiness: false,
            choppiness_length: default_choppiness_length(),
            invert_below_choppiness: 0.0,
            invert_above_choppiness: None,
            use_directional_return: false,
            directional_return_length: default_directional_return_length(),
            invert_when_aligned_return_below: None,
            invert_when_aligned_return_above: None,
            use_adx: false,
            adx_length: default_adx_length(),
            invert_below_adx: default_adx_cutoff(),
            invert_above_adx: None,
            use_di_imbalance: false,
            invert_when_di_imbalance_below: default_di_threshold(),
            invert_when_di_imbalance_above: None,
            use_ema_spread: false,
            invert_when_normalized_spread_below: 0.0,
            invert_when_normalized_spread_above: None,
            use_ema_slope: false,
            ema_slope_lookback: default_slope_lookback(),
            invert_when_normalized_slope_below: 0.0,
            invert_when_normalized_slope_above: None,
            use_directional_ema_gap: false,
            directional_ema_length: default_directional_ema_length(),
            directional_gap_lookback_bars: default_directional_gap_lookback(),
            invert_when_bullish_gap_below: None,
            invert_when_bullish_gap_above: None,
            invert_when_bearish_gap_below: None,
            invert_when_bearish_gap_above: None,
            use_session_window: false,
            session_start_minute_et: 0,
            session_end_minute_et: 0,
            invert_inside_session_window: true,
            session_weekdays_mask: default_weekday_mask(),
            use_higher_timeframe_context: false,
            higher_timeframe_average_kind: HigherTimeframeAverageKind::default(),
            higher_timeframe_minutes: default_higher_timeframe_minutes(),
            higher_timeframe_fast_length: default_higher_timeframe_fast_length(),
            higher_timeframe_slow_length: default_higher_timeframe_slow_length(),
            higher_timeframe_atr_length: default_higher_timeframe_atr_length(),
            higher_timeframe_slope_lookback: default_higher_timeframe_slope_lookback(),
            higher_timeframe_persistence_bars: default_higher_timeframe_persistence_bars(),
            higher_timeframe_spread_atr_floor: 0.0,
            higher_timeframe_spread_atr_ceiling: None,
            higher_timeframe_slope_atr_floor: 0.0,
            higher_timeframe_conflict_action: HigherTimeframeConflictAction::default(),
        }
    }
}

impl RegimeAdaptiveGateConfig {
    pub fn warmup_bars(&self, fast_length: usize, slow_length: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        let mut warmup = slow_length.max(fast_length).max(1);
        if self.use_relative_volume {
            warmup = warmup.max(self.volume_lookback_bars.max(1) + 1);
        }
        if self.use_atr_ratio {
            warmup = warmup.max(
                self.atr_length
                    .max(1)
                    .saturating_add(self.atr_lookback_bars.max(1)),
            );
        }
        if self.use_choppiness {
            warmup = warmup.max(self.choppiness_length.max(1));
        }
        if self.use_directional_return {
            warmup = warmup.max(
                self.directional_return_length
                    .max(1)
                    .saturating_add(1)
                    .max(self.atr_length.max(1)),
            );
        }
        if self.use_adx || self.use_di_imbalance {
            warmup = warmup.max(self.adx_length.max(1).saturating_mul(2));
        }
        if self.use_ema_slope {
            warmup = warmup.max(slow_length.max(1) + self.ema_slope_lookback.max(1));
        }
        if self.use_directional_ema_gap {
            warmup = warmup.max(
                self.directional_ema_length
                    .max(1)
                    .saturating_add(self.directional_gap_lookback_bars.max(1))
                    .max(self.atr_length.max(1)),
            );
        }
        if self.use_higher_timeframe_context {
            let average_length = self
                .higher_timeframe_slow_length
                .max(self.higher_timeframe_fast_length)
                .max(1);
            let average_warmup = match self.higher_timeframe_average_kind {
                HigherTimeframeAverageKind::Ema => average_length,
                HigherTimeframeAverageKind::Hma => {
                    average_length.saturating_add((average_length as f64).sqrt().floor() as usize)
                }
            };
            let context_bars = average_warmup
                .saturating_add(self.higher_timeframe_atr_length)
                .saturating_add(self.higher_timeframe_slope_lookback)
                .saturating_add(self.higher_timeframe_persistence_bars)
                .max(1);
            warmup = warmup.max(
                self.higher_timeframe_minutes
                    .max(1)
                    .saturating_mul(context_bars),
            );
        }
        warmup
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.minimum_feature_votes == 0 {
            return Err("adaptive gate minimum_feature_votes must be greater than zero".into());
        }
        if self.invert_confirmation_bars == 0 || self.normal_confirmation_bars == 0 {
            return Err("adaptive gate confirmation bars must be greater than zero".into());
        }
        if self.use_relative_volume
            && (self.volume_lookback_bars == 0
                || !self.invert_below_relative_volume.is_finite()
                || self.invert_below_relative_volume < 0.0
                || !optional_non_negative(self.invert_above_relative_volume))
        {
            return Err("adaptive gate relative-volume settings are invalid".into());
        }
        if self.use_atr_ratio
            && (self.atr_length == 0
                || self.atr_lookback_bars == 0
                || !self.invert_below_atr_ratio.is_finite()
                || self.invert_below_atr_ratio < 0.0
                || !optional_non_negative(self.invert_above_atr_ratio))
        {
            return Err("adaptive gate ATR settings are invalid".into());
        }
        if self.use_choppiness
            && (self.choppiness_length < 2
                || !self.invert_below_choppiness.is_finite()
                || self.invert_below_choppiness < 0.0
                || !optional_non_negative(self.invert_above_choppiness))
        {
            return Err("adaptive gate choppiness settings are invalid".into());
        }
        if self.use_directional_return {
            let has_threshold = self.invert_when_aligned_return_below.is_some()
                || self.invert_when_aligned_return_above.is_some();
            if self.directional_return_length == 0
                || !has_threshold
                || !optional_finite(self.invert_when_aligned_return_below)
                || !optional_finite(self.invert_when_aligned_return_above)
            {
                return Err("adaptive gate directional-return settings are invalid".into());
            }
        }
        if (self.use_adx || self.use_di_imbalance)
            && (self.adx_length == 0
                || !self.invert_below_adx.is_finite()
                || !optional_non_negative(self.invert_above_adx))
        {
            return Err("adaptive gate ADX settings are invalid".into());
        }
        if self.use_di_imbalance
            && (!self.invert_when_di_imbalance_below.is_finite()
                || self.invert_when_di_imbalance_below < 0.0)
        {
            return Err("adaptive gate DI threshold is invalid".into());
        }
        if !optional_non_negative(self.invert_when_di_imbalance_above) {
            return Err("adaptive gate DI upper threshold is invalid".into());
        }
        if self.use_ema_slope && self.ema_slope_lookback == 0 {
            return Err("adaptive gate EMA slope lookback must be greater than zero".into());
        }
        if !self.invert_when_normalized_spread_below.is_finite()
            || !self.invert_when_normalized_slope_below.is_finite()
            || !optional_finite(self.invert_when_normalized_spread_above)
            || !optional_finite(self.invert_when_normalized_slope_above)
        {
            return Err("adaptive gate normalized EMA thresholds must be finite".into());
        }
        if self.use_directional_ema_gap {
            if self.directional_ema_length == 0 {
                return Err("adaptive gate directional EMA settings are invalid".into());
            }
            let has_threshold = self.invert_when_bullish_gap_below.is_some()
                || self.invert_when_bullish_gap_above.is_some()
                || self.invert_when_bearish_gap_below.is_some()
                || self.invert_when_bearish_gap_above.is_some();
            if !has_threshold
                || !optional_finite(self.invert_when_bullish_gap_below)
                || !optional_finite(self.invert_when_bullish_gap_above)
                || !optional_finite(self.invert_when_bearish_gap_below)
                || !optional_finite(self.invert_when_bearish_gap_above)
            {
                return Err("adaptive gate directional EMA thresholds are invalid".into());
            }
        }
        if self.use_higher_timeframe_context
            && (self.higher_timeframe_minutes == 0
                || self.higher_timeframe_fast_length == 0
                || self.higher_timeframe_slow_length == 0
                || self.higher_timeframe_fast_length >= self.higher_timeframe_slow_length
                || self.higher_timeframe_atr_length == 0
                || self.higher_timeframe_slope_lookback == 0
                || self.higher_timeframe_persistence_bars == 0
                || !self.higher_timeframe_spread_atr_floor.is_finite()
                || self.higher_timeframe_spread_atr_floor < 0.0
                || !optional_non_negative(self.higher_timeframe_spread_atr_ceiling)
                || !self.higher_timeframe_slope_atr_floor.is_finite()
                || self.higher_timeframe_slope_atr_floor < 0.0)
        {
            return Err("adaptive gate higher-timeframe settings are invalid".into());
        }
        if self.use_session_window
            && (self.session_start_minute_et >= 1440
                || self.session_end_minute_et >= 1440
                || self.session_weekdays_mask == 0)
        {
            return Err("adaptive gate session window is invalid".into());
        }
        Ok(())
    }

    /// Evaluate the gate at the latest bar.  Every intermediate decision is
    /// reconstructed from the prefix ending at that bar, allowing dwell and
    /// hysteresis without storing hidden mutable state and preventing future
    /// bars from influencing the result.
    pub fn evaluate(
        &self,
        bars: &[Bar],
        fast_length: usize,
        slow_length: usize,
    ) -> RegimeAdaptiveGateEvaluation {
        let Some(last_bar) = bars.last() else {
            return RegimeAdaptiveGateEvaluation::disabled_or_empty(self.enabled);
        };
        if !self.enabled {
            return RegimeAdaptiveGateEvaluation::disabled_or_empty(false);
        }

        let need_ema_spread = self.use_ema_spread;
        let need_ema_slope = self.use_ema_slope;
        let need_directional_gap = self.use_directional_ema_gap;
        let need_directional_return = self.use_directional_return;
        let need_directional_context = need_directional_gap || need_directional_return;
        // The HTF gate still needs the raw 1-minute EMA relationship to know
        // whether the current cross agrees with or conflicts with the
        // completed higher-timeframe context.
        let need_base_ema = self.use_higher_timeframe_context;
        let closes =
            if need_ema_spread || need_ema_slope || need_directional_context || need_base_ema {
                bars.iter().map(|bar| bar.close).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
        let fast_ema = if need_ema_spread || need_directional_context || need_base_ema {
            ema_series(&closes, fast_length.max(1))
        } else {
            Vec::new()
        };
        let slow_ema =
            if need_ema_spread || need_ema_slope || need_directional_context || need_base_ema {
                ema_series(&closes, slow_length.max(1))
            } else {
                Vec::new()
            };
        let directional_ema = if need_directional_gap {
            ema_series(&closes, self.directional_ema_length.max(1))
        } else {
            Vec::new()
        };
        let need_atr = self.use_atr_ratio
            || self.use_ema_spread
            || self.use_ema_slope
            || self.use_directional_ema_gap
            || self.use_directional_return;
        let atr = if need_atr {
            atr_series(bars, self.atr_length.max(1))
        } else {
            vec![f64::NAN; bars.len()]
        };
        let atr_ratio = if self.use_atr_ratio {
            relative_series_value(&atr, self.atr_lookback_bars.max(1))
        } else {
            vec![None; bars.len()]
        };
        let adx_values = if self.use_adx || self.use_di_imbalance {
            Some(adx_series(bars, self.adx_length.max(1)))
        } else {
            None
        };
        let choppiness = if self.use_choppiness {
            choppiness_series(bars, self.choppiness_length)
        } else {
            vec![None; bars.len()]
        };
        let higher_timeframe_context = if self.use_higher_timeframe_context {
            Some(higher_timeframe_context_series(bars, self))
        } else {
            None
        };

        let mut orientations = Vec::with_capacity(bars.len());
        let mut vote_history = Vec::with_capacity(bars.len());
        let mut previous_orientation = false;
        let mut latest_snapshot = FeatureSnapshot::default();
        for idx in 0..bars.len() {
            let snapshot = self.snapshot_at_with_direction_cached(
                bars,
                idx,
                &fast_ema,
                &slow_ema,
                &directional_ema,
                &atr,
                &atr_ratio,
                adx_values.as_ref(),
                Some(&choppiness),
                higher_timeframe_context.as_deref(),
            );
            let vote = self.combine_snapshot(&snapshot);
            let orientation = match vote {
                Some(vote) => {
                    let required = if vote {
                        self.invert_confirmation_bars.max(1)
                    } else {
                        self.normal_confirmation_bars.max(1)
                    };
                    let consecutive = consecutive_vote_count(&vote_history, vote);
                    if consecutive >= required {
                        vote
                    } else {
                        previous_orientation
                    }
                }
                None if self.reset_on_missing_features => false,
                None => previous_orientation,
            };
            previous_orientation = orientation;
            orientations.push(orientation);
            vote_history.push(vote);
            if idx + 1 == bars.len() {
                latest_snapshot = snapshot;
            }
        }

        let current_vote = self.combine_snapshot(&latest_snapshot);
        let latest_votes = self.feature_votes(&latest_snapshot);
        let available_features = latest_votes.iter().filter(|vote| vote.is_some()).count();
        let inverted_votes = latest_votes
            .iter()
            .filter(|vote| **vote == Some(true))
            .count();
        let normal_votes = latest_votes
            .iter()
            .filter(|vote| **vote == Some(false))
            .count();
        let confirmed = match current_vote {
            Some(vote) => {
                let required = if vote {
                    self.invert_confirmation_bars.max(1)
                } else {
                    self.normal_confirmation_bars.max(1)
                };
                consecutive_vote_count(&vote_history, vote) >= required
            }
            None => false,
        };
        let ready = current_vote.is_some();
        let hold_reason = if self.higher_timeframe_conflict_action
            == HigherTimeframeConflictAction::Abstain
            && latest_snapshot.higher_timeframe_conflict == Some(true)
        {
            Some("adaptive_gate_higher_timeframe_conflict")
        } else if self.hold_on_missing_features && !ready {
            Some("adaptive_gate_missing_features")
        } else if self.hold_during_dwell && ready && !confirmed {
            Some("adaptive_gate_dwell")
        } else {
            None
        };
        RegimeAdaptiveGateEvaluation {
            inverted: orientations.last().copied().unwrap_or(false),
            ready,
            confirmed,
            current_vote,
            available_features,
            inverted_votes,
            normal_votes,
            relative_volume: latest_snapshot.relative_volume,
            atr: latest_snapshot.atr,
            atr_ratio: latest_snapshot.atr_ratio,
            choppiness: latest_snapshot.choppiness,
            aligned_return: latest_snapshot.aligned_return,
            adx: latest_snapshot.adx,
            di_imbalance: latest_snapshot.di_imbalance,
            normalized_spread: latest_snapshot.normalized_spread,
            normalized_slope: latest_snapshot.normalized_slope,
            directional_ema_gap: latest_snapshot.directional_ema_gap,
            raw_signal_direction: latest_snapshot.raw_signal_direction,
            in_session_window: latest_snapshot.in_session_window,
            higher_timeframe_direction: latest_snapshot.higher_timeframe_direction,
            higher_timeframe_spread: latest_snapshot.higher_timeframe_spread,
            higher_timeframe_slope: latest_snapshot.higher_timeframe_slope,
            higher_timeframe_conflict: latest_snapshot.higher_timeframe_conflict,
            hold_reason,
            latest_ts_ns: last_bar.ts_ns,
        }
    }

    fn snapshot_at(
        &self,
        bars: &[Bar],
        idx: usize,
        fast_ema: &[f64],
        slow_ema: &[f64],
        atr: &[f64],
        atr_ratio: &[Option<f64>],
        adx_values: Option<&crate::strategies::adx::AdxSeries>,
    ) -> FeatureSnapshot {
        self.snapshot_at_with_direction(
            bars,
            idx,
            fast_ema,
            slow_ema,
            &[],
            atr,
            atr_ratio,
            adx_values,
        )
    }

    fn snapshot_at_with_direction(
        &self,
        bars: &[Bar],
        idx: usize,
        fast_ema: &[f64],
        slow_ema: &[f64],
        directional_ema: &[f64],
        atr: &[f64],
        atr_ratio: &[Option<f64>],
        adx_values: Option<&crate::strategies::adx::AdxSeries>,
    ) -> FeatureSnapshot {
        self.snapshot_at_with_direction_cached(
            bars,
            idx,
            fast_ema,
            slow_ema,
            directional_ema,
            atr,
            atr_ratio,
            adx_values,
            None,
            None,
        )
    }

    fn snapshot_at_with_direction_cached(
        &self,
        bars: &[Bar],
        idx: usize,
        fast_ema: &[f64],
        slow_ema: &[f64],
        directional_ema: &[f64],
        atr: &[f64],
        atr_ratio: &[Option<f64>],
        adx_values: Option<&crate::strategies::adx::AdxSeries>,
        choppiness_values: Option<&[Option<f64>]>,
        higher_timeframe_values: Option<&[Option<HigherTimeframeSnapshot>]>,
    ) -> FeatureSnapshot {
        let mut snapshot = FeatureSnapshot {
            relative_volume: self.relative_volume_at(bars, idx),
            atr: atr.get(idx).copied().filter(|v| v.is_finite()),
            atr_ratio: atr_ratio.get(idx).copied().flatten(),
            choppiness: if self.use_choppiness {
                choppiness_values
                    .and_then(|values| values.get(idx).copied())
                    .flatten()
                    .or_else(|| choppiness_at(bars, idx, self.choppiness_length))
            } else {
                None
            },
            aligned_return: None,
            ..FeatureSnapshot::default()
        };
        if self.use_higher_timeframe_context {
            if let Some(context) = higher_timeframe_values
                .and_then(|values| values.get(idx).copied())
                .flatten()
            {
                snapshot.higher_timeframe_direction = Some(context.direction);
                snapshot.higher_timeframe_spread = Some(context.spread);
                snapshot.higher_timeframe_slope = Some(context.slope);
            }
        }
        if let Some(series) = adx_values {
            snapshot.adx = series.adx.get(idx).copied().filter(|v| v.is_finite());
            let plus = series.plus_di.get(idx).copied().filter(|v| v.is_finite());
            let minus = series.minus_di.get(idx).copied().filter(|v| v.is_finite());
            snapshot.di_imbalance = di_imbalance(plus, minus);
        }
        if self.use_ema_spread || self.use_ema_slope {
            let close = bars.get(idx).map(|bar| bar.close).filter(|v| v.is_finite());
            let denominator = snapshot.atr.or(close.map(f64::abs));
            if self.use_ema_spread {
                snapshot.normalized_spread = fast_ema
                    .get(idx)
                    .copied()
                    .filter(|v| v.is_finite())
                    .zip(slow_ema.get(idx).copied().filter(|v| v.is_finite()))
                    .zip(denominator)
                    .and_then(|((fast, slow), scale)| {
                        (scale > f64::EPSILON).then_some((fast - slow) / scale)
                    });
            }
            if self.use_ema_slope {
                let lookback = self.ema_slope_lookback.max(1);
                snapshot.normalized_slope = idx
                    .checked_sub(lookback)
                    .and_then(|previous| {
                        slow_ema
                            .get(idx)
                            .copied()
                            .filter(|v| v.is_finite())
                            .zip(slow_ema.get(previous).copied().filter(|v| v.is_finite()))
                    })
                    .zip(denominator)
                    .and_then(|((current, previous), scale)| {
                        (scale > f64::EPSILON)
                            .then_some((current - previous) / scale / lookback as f64)
                    });
            }
        }
        if self.use_directional_ema_gap || self.use_directional_return {
            snapshot.raw_signal_direction = raw_signal_direction_at(fast_ema, slow_ema, idx);
        }
        if self.use_higher_timeframe_context {
            snapshot.raw_signal_direction = snapshot
                .raw_signal_direction
                .or_else(|| raw_signal_direction_at(fast_ema, slow_ema, idx));
            snapshot.higher_timeframe_conflict = snapshot
                .higher_timeframe_direction
                .zip(snapshot.raw_signal_direction)
                .map(|(context, raw)| context != raw);
        }
        if self.use_directional_return {
            snapshot.aligned_return = snapshot.raw_signal_direction.and_then(|direction| {
                directional_return_at(bars, idx, self.directional_return_length, atr)
                    .map(|value| value * direction as f64)
            });
        }
        if self.use_directional_ema_gap {
            // A zero lookback deliberately samples the current completed
            // decision bar.  It remains causal (the bar is closed before the
            // signal is submitted) and is useful for testing price-vs-long-EMA
            // context.  The default stays one bar behind the cross so callers
            // can avoid a feature that is mechanically correlated with the
            // crossover itself.
            let context_idx = idx.checked_sub(self.directional_gap_lookback_bars);
            snapshot.directional_ema_gap = context_idx
                .and_then(|context_idx| {
                    bars.get(context_idx)
                        .map(|bar| (bar.close, atr.get(context_idx).copied(), context_idx))
                })
                .and_then(|(close, context_atr, context_idx)| {
                    let scale = context_atr
                        .filter(|value| value.is_finite() && *value > f64::EPSILON)
                        .or_else(|| close.is_finite().then_some(close.abs()));
                    directional_ema
                        .get(context_idx)
                        .copied()
                        .filter(|value| value.is_finite())
                        .zip(scale)
                        .and_then(|(ema, scale)| {
                            (scale > f64::EPSILON && close.is_finite())
                                .then_some((close - ema) / scale)
                        })
                });
        }
        snapshot.in_session_window = if self.use_session_window {
            session_window_at(bars.get(idx).map(|bar| bar.ts_ns).unwrap_or_default(), self)
        } else {
            None
        };
        snapshot
    }

    fn relative_volume_at(&self, bars: &[Bar], idx: usize) -> Option<f64> {
        if !self.use_relative_volume {
            return None;
        }
        let current = bars.get(idx)?.volume?;
        if !current.is_finite() || current <= 0.0 {
            return None;
        }
        let lookback = self.volume_lookback_bars.max(1);
        if idx < lookback {
            return None;
        }
        let previous = bars[idx - lookback..idx]
            .iter()
            .filter_map(|bar| bar.volume)
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect::<Vec<_>>();
        if previous.len() != lookback {
            return None;
        }
        let mean = previous.iter().sum::<f64>() / lookback as f64;
        (mean.is_finite() && mean > 0.0).then_some(current / mean)
    }

    fn combine_snapshot(&self, snapshot: &FeatureSnapshot) -> Option<bool> {
        let votes = self.feature_votes(snapshot);
        let available = votes.iter().filter(|vote| vote.is_some()).count();
        if available < self.minimum_feature_votes.max(1) {
            return None;
        }
        let inverted = votes.iter().filter(|vote| **vote == Some(true)).count();
        let normal = votes.iter().filter(|vote| **vote == Some(false)).count();
        match self.combine {
            AdaptiveGateCombine::Any => Some(inverted > 0),
            AdaptiveGateCombine::All => Some(inverted == available),
            AdaptiveGateCombine::Xor => Some(inverted % 2 == 1),
            AdaptiveGateCombine::Majority => {
                if inverted > normal {
                    Some(true)
                } else if normal > inverted {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    fn feature_votes(&self, snapshot: &FeatureSnapshot) -> Vec<Option<bool>> {
        let mut votes = Vec::new();
        if self.use_relative_volume {
            votes.push(snapshot.relative_volume.map(|value| {
                is_below_or_above(
                    value,
                    self.invert_below_relative_volume,
                    self.invert_above_relative_volume,
                )
            }));
        }
        if self.use_atr_ratio {
            votes.push(snapshot.atr_ratio.map(|value| {
                is_below_or_above(
                    value,
                    self.invert_below_atr_ratio,
                    self.invert_above_atr_ratio,
                )
            }));
        }
        if self.use_choppiness {
            votes.push(snapshot.choppiness.map(|value| {
                is_below_or_above(
                    value,
                    self.invert_below_choppiness,
                    self.invert_above_choppiness,
                )
            }));
        }
        if self.use_directional_return {
            votes.push(snapshot.aligned_return.map(|value| {
                is_optional_below_or_above(
                    value,
                    self.invert_when_aligned_return_below,
                    self.invert_when_aligned_return_above,
                )
            }));
        }
        if self.use_adx {
            votes.push(snapshot.adx.map(|value| {
                is_below_or_above(value, self.invert_below_adx, self.invert_above_adx)
            }));
        }
        if self.use_di_imbalance {
            let threshold = self.invert_when_di_imbalance_below.abs();
            votes.push(snapshot.di_imbalance.map(|value| {
                value < -threshold
                    || self
                        .invert_when_di_imbalance_above
                        .is_some_and(|upper| value > upper)
            }));
        }
        if self.use_ema_spread {
            votes.push(snapshot.normalized_spread.map(|value| {
                is_below_or_above(
                    value,
                    self.invert_when_normalized_spread_below,
                    self.invert_when_normalized_spread_above,
                )
            }));
        }
        if self.use_ema_slope {
            votes.push(snapshot.normalized_slope.map(|value| {
                is_below_or_above(
                    value,
                    self.invert_when_normalized_slope_below,
                    self.invert_when_normalized_slope_above,
                )
            }));
        }
        if self.use_directional_ema_gap {
            votes.push(self.directional_ema_gap_vote(snapshot));
        }
        if self.use_higher_timeframe_context {
            votes.push(
                snapshot
                    .higher_timeframe_direction
                    .zip(snapshot.raw_signal_direction)
                    .map(|(context, raw)| context != raw),
            );
        }
        if self.use_session_window {
            votes.push(
                snapshot
                    .in_session_window
                    .map(|inside| inside == self.invert_inside_session_window),
            );
        }
        votes
    }

    fn directional_ema_gap_vote(&self, snapshot: &FeatureSnapshot) -> Option<bool> {
        let direction = snapshot.raw_signal_direction?;
        let gap = snapshot.directional_ema_gap?;
        let (below, above) = if direction > 0 {
            (
                self.invert_when_bullish_gap_below,
                self.invert_when_bullish_gap_above,
            )
        } else {
            (
                self.invert_when_bearish_gap_below,
                self.invert_when_bearish_gap_above,
            )
        };
        Some(
            below.is_some_and(|threshold| gap < threshold)
                || above.is_some_and(|threshold| gap > threshold),
        )
    }
}

/// Causal feature values exposed in the replay/TUI diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RegimeAdaptiveGateEvaluation {
    pub inverted: bool,
    pub ready: bool,
    pub confirmed: bool,
    pub current_vote: Option<bool>,
    pub available_features: usize,
    pub inverted_votes: usize,
    pub normal_votes: usize,
    pub relative_volume: Option<f64>,
    pub atr: Option<f64>,
    pub atr_ratio: Option<f64>,
    pub choppiness: Option<f64>,
    pub aligned_return: Option<f64>,
    pub adx: Option<f64>,
    pub di_imbalance: Option<f64>,
    pub normalized_spread: Option<f64>,
    pub normalized_slope: Option<f64>,
    pub directional_ema_gap: Option<f64>,
    pub raw_signal_direction: Option<i8>,
    pub in_session_window: Option<bool>,
    pub higher_timeframe_direction: Option<i8>,
    pub higher_timeframe_spread: Option<f64>,
    pub higher_timeframe_slope: Option<f64>,
    pub higher_timeframe_conflict: Option<bool>,
    pub hold_reason: Option<&'static str>,
    pub latest_ts_ns: i64,
}

impl RegimeAdaptiveGateEvaluation {
    fn disabled_or_empty(enabled: bool) -> Self {
        Self {
            ready: !enabled,
            confirmed: !enabled,
            ..Self::default()
        }
    }

    pub fn should_hold(&self) -> bool {
        self.hold_reason.is_some()
    }

    pub fn summary(&self) -> String {
        format!(
            "adaptive_gate={} ready={} confirmed={} vote={} features={}/{} rv={} atr={} atr_ratio={} chop={} aligned_return={} adx={} di={} spread={} slope={} dir_gap={} raw_dir={} session={} htf_dir={} htf_spread={} htf_slope={} htf_conflict={} reason={}",
            if self.inverted { "inverted" } else { "normal" },
            self.ready,
            self.confirmed,
            self.current_vote
                .map(|v| if v { "invert" } else { "normal" })
                .unwrap_or("neutral"),
            self.available_features,
            self.inverted_votes + self.normal_votes,
            fmt(self.relative_volume),
            fmt(self.atr),
            fmt(self.atr_ratio),
            fmt(self.choppiness),
            fmt(self.aligned_return),
            fmt(self.adx),
            fmt(self.di_imbalance),
            fmt(self.normalized_spread),
            fmt(self.normalized_slope),
            fmt(self.directional_ema_gap),
            self.raw_signal_direction
                .map(|direction| direction.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.in_session_window
                .map(|inside| if inside { "inside" } else { "outside" })
                .unwrap_or("n/a"),
            self.higher_timeframe_direction
                .map(|direction| direction.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            fmt(self.higher_timeframe_spread),
            fmt(self.higher_timeframe_slope),
            self.higher_timeframe_conflict
                .map(|conflict| conflict.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.hold_reason.unwrap_or("none"),
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct HigherTimeframeSnapshot {
    direction: i8,
    spread: f64,
    slope: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct FeatureSnapshot {
    relative_volume: Option<f64>,
    atr: Option<f64>,
    atr_ratio: Option<f64>,
    choppiness: Option<f64>,
    aligned_return: Option<f64>,
    adx: Option<f64>,
    di_imbalance: Option<f64>,
    normalized_spread: Option<f64>,
    normalized_slope: Option<f64>,
    directional_ema_gap: Option<f64>,
    raw_signal_direction: Option<i8>,
    in_session_window: Option<bool>,
    higher_timeframe_direction: Option<i8>,
    higher_timeframe_spread: Option<f64>,
    higher_timeframe_slope: Option<f64>,
    higher_timeframe_conflict: Option<bool>,
}

fn consecutive_vote_count(history: &[Option<bool>], vote: bool) -> usize {
    history
        .iter()
        .rev()
        .take_while(|previous| **previous == Some(vote))
        .count()
        .saturating_add(1)
}

fn relative_series_value(series: &[f64], lookback: usize) -> Vec<Option<f64>> {
    let lookback = lookback.max(1);
    let mut output = vec![None; series.len()];
    for idx in 0..series.len() {
        let Some(current) = series.get(idx).copied().filter(|v| v.is_finite()) else {
            continue;
        };
        if idx < lookback {
            continue;
        }
        let previous = &series[idx - lookback..idx];
        if previous.len() != lookback || !previous.iter().all(|v| v.is_finite()) {
            continue;
        }
        let mean = previous.iter().sum::<f64>() / lookback as f64;
        if mean.is_finite() && mean > f64::EPSILON {
            output[idx] = Some(current / mean);
        }
    }
    output
}

/// Build a causal higher-timeframe context snapshot for every lower-timeframe
/// bar. The snapshot at a lower-timeframe index uses only HTF buckets strictly
/// before that bar's bucket, so the currently forming HTF candle cannot leak
/// into the decision.
fn higher_timeframe_context_series(
    bars: &[Bar],
    config: &RegimeAdaptiveGateConfig,
) -> Vec<Option<HigherTimeframeSnapshot>> {
    let mut output = vec![None; bars.len()];
    if bars.is_empty() || config.higher_timeframe_minutes == 0 {
        return output;
    }

    let interval_ns = (config.higher_timeframe_minutes as i64)
        .saturating_mul(60)
        .saturating_mul(1_000_000_000);
    if interval_ns <= 0 {
        return output;
    }

    let mut higher_bars = Vec::new();
    let mut current_bucket: Option<i64> = None;
    let mut current_bar: Option<Bar> = None;
    for bar in bars.iter().filter(|bar| bar.ts_ns > 0) {
        let bucket = bar.ts_ns.div_euclid(interval_ns);
        if current_bucket == Some(bucket) {
            if let Some(aggregate) = current_bar.as_mut() {
                aggregate.high = aggregate.high.max(bar.high);
                aggregate.low = aggregate.low.min(bar.low);
                aggregate.close = bar.close;
                aggregate.volume = match (aggregate.volume, bar.volume) {
                    (Some(previous), Some(current))
                        if previous.is_finite() && current.is_finite() =>
                    {
                        Some(previous + current)
                    }
                    _ => None,
                };
            }
            continue;
        }

        if let Some(aggregate) = current_bar.take() {
            higher_bars.push(aggregate);
        }
        current_bucket = Some(bucket);
        current_bar = Some(Bar {
            ts_ns: bucket.saturating_mul(interval_ns),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        });
    }
    if let Some(aggregate) = current_bar {
        higher_bars.push(aggregate);
    }
    if higher_bars.is_empty() {
        return output;
    }

    let closes = higher_bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
    let (fast, slow) = match config.higher_timeframe_average_kind {
        HigherTimeframeAverageKind::Ema => (
            ema_series(&closes, config.higher_timeframe_fast_length.max(1)),
            ema_series(&closes, config.higher_timeframe_slow_length.max(1)),
        ),
        HigherTimeframeAverageKind::Hma => (
            hma_series_incremental(&closes, config.higher_timeframe_fast_length.max(1)),
            hma_series_incremental(&closes, config.higher_timeframe_slow_length.max(1)),
        ),
    };
    let atr = atr_series(&higher_bars, config.higher_timeframe_atr_length.max(1));
    let slope_lookback = config.higher_timeframe_slope_lookback.max(1);
    let persistence = config.higher_timeframe_persistence_bars.max(1);

    for (idx, bar) in bars.iter().enumerate() {
        if bar.ts_ns <= 0 {
            continue;
        }
        let bucket = bar.ts_ns.div_euclid(interval_ns);
        let closed_count =
            higher_bars.partition_point(|higher| higher.ts_ns.div_euclid(interval_ns) < bucket);
        let Some(context_idx) = closed_count.checked_sub(1) else {
            continue;
        };
        if context_idx < slope_lookback || context_idx + 1 < persistence {
            continue;
        }
        let Some(scale) = atr
            .get(context_idx)
            .copied()
            .filter(|value| value.is_finite() && *value > f64::EPSILON)
        else {
            continue;
        };
        let Some((fast_value, slow_value)) = fast
            .get(context_idx)
            .copied()
            .filter(|value| value.is_finite())
            .zip(
                slow.get(context_idx)
                    .copied()
                    .filter(|value| value.is_finite()),
            )
        else {
            continue;
        };
        let Some(previous_slow) = slow
            .get(context_idx - slope_lookback)
            .copied()
            .filter(|value| value.is_finite())
        else {
            continue;
        };
        let spread = (fast_value - slow_value) / scale;
        let slope = (slow_value - previous_slow) / scale / slope_lookback as f64;
        if !spread.is_finite() || !slope.is_finite() {
            continue;
        }

        let spread_within_ceiling = config
            .higher_timeframe_spread_atr_ceiling
            .is_none_or(|ceiling| spread.abs() <= ceiling);

        let persistence_start = context_idx + 1 - persistence;
        let persistent_up = (persistence_start..=context_idx).all(|position| {
            fast.get(position).copied().is_some_and(|value| {
                value.is_finite()
                    && slow
                        .get(position)
                        .copied()
                        .is_some_and(|slow_value| slow_value.is_finite() && value > slow_value)
            })
        });
        let persistent_down = (persistence_start..=context_idx).all(|position| {
            fast.get(position).copied().is_some_and(|value| {
                value.is_finite()
                    && slow
                        .get(position)
                        .copied()
                        .is_some_and(|slow_value| slow_value.is_finite() && value < slow_value)
            })
        });
        let direction = if spread > config.higher_timeframe_spread_atr_floor
            && spread_within_ceiling
            && slope > config.higher_timeframe_slope_atr_floor
            && persistent_up
        {
            1
        } else if spread < -config.higher_timeframe_spread_atr_floor
            && slope < -config.higher_timeframe_slope_atr_floor
            && persistent_down
        {
            -1
        } else {
            continue;
        };
        output[idx] = Some(HigherTimeframeSnapshot {
            direction,
            spread,
            slope,
        });
    }
    output
}

fn atr_series(bars: &[Bar], period: usize) -> Vec<f64> {
    let period = period.max(1);
    let mut tr = vec![f64::NAN; bars.len()];
    for idx in 0..bars.len() {
        let bar = &bars[idx];
        if ![bar.high, bar.low, bar.close]
            .iter()
            .all(|value| value.is_finite())
        {
            continue;
        }
        tr[idx] = if idx == 0 {
            (bar.high - bar.low).abs()
        } else {
            let previous = &bars[idx - 1];
            if !previous.close.is_finite() {
                f64::NAN
            } else {
                (bar.high - bar.low)
                    .abs()
                    .max((bar.high - previous.close).abs())
                    .max((bar.low - previous.close).abs())
            }
        };
    }
    let mut atr = vec![f64::NAN; bars.len()];
    if bars.len() < period {
        return atr;
    }
    let seed = &tr[..period];
    if seed.iter().all(|value| value.is_finite()) {
        atr[period - 1] = seed.iter().sum::<f64>() / period as f64;
        for idx in period..bars.len() {
            if atr[idx - 1].is_finite() && tr[idx].is_finite() {
                atr[idx] = (atr[idx - 1] * (period as f64 - 1.0) + tr[idx]) / period as f64;
            }
        }
    }
    atr
}

/// Compute the Choppiness Index at `idx` from the strictly preceding window.
/// This mirrors the offline research definition: summed true range divided by
/// the high-low span, normalized by the log of the window length.
fn choppiness_at(bars: &[Bar], idx: usize, length: usize) -> Option<f64> {
    if length < 2 || idx < length {
        return None;
    }
    let start = idx - length;
    let mut travel = 0.0;
    let mut highest = f64::NEG_INFINITY;
    let mut lowest = f64::INFINITY;
    for position in start..idx {
        let bar = bars.get(position)?;
        if ![bar.high, bar.low, bar.close]
            .iter()
            .all(|value| value.is_finite())
        {
            return None;
        }
        highest = highest.max(bar.high);
        lowest = lowest.min(bar.low);
        let true_range = if position == 0 {
            (bar.high - bar.low).abs()
        } else {
            let previous_close = bars.get(position - 1)?.close;
            if !previous_close.is_finite() {
                return None;
            }
            (bar.high - bar.low)
                .abs()
                .max((bar.high - previous_close).abs())
                .max((bar.low - previous_close).abs())
        };
        if !true_range.is_finite() {
            return None;
        }
        travel += true_range;
    }
    let span = highest - lowest;
    if !travel.is_finite() || travel <= 0.0 || !span.is_finite() || span <= 0.0 {
        return None;
    }
    let length_log = (length as f64).ln();
    let value = 100.0 * (travel / span).ln() / length_log;
    value.is_finite().then_some(value)
}

fn choppiness_series(bars: &[Bar], length: usize) -> Vec<Option<f64>> {
    (0..bars.len())
        .map(|idx| choppiness_at(bars, idx, length))
        .collect()
}

fn directional_return_at(bars: &[Bar], idx: usize, length: usize, atr: &[f64]) -> Option<f64> {
    if length == 0 || idx < length.saturating_add(1) {
        return None;
    }
    let prior_idx = idx - 1;
    let start_idx = idx - length - 1;
    let prior_close = bars.get(prior_idx)?.close;
    let start_close = bars.get(start_idx)?.close;
    let scale = atr
        .get(prior_idx)
        .copied()
        .filter(|value| value.is_finite() && *value > f64::EPSILON)?;
    if !prior_close.is_finite() || !start_close.is_finite() {
        return None;
    }
    let value = (prior_close - start_close) / scale;
    value.is_finite().then_some(value)
}

fn raw_signal_direction_at(fast_ema: &[f64], slow_ema: &[f64], idx: usize) -> Option<i8> {
    if idx == 0 {
        return None;
    }
    let previous_fast = fast_ema.get(idx - 1).copied()?;
    let previous_slow = slow_ema.get(idx - 1).copied()?;
    let current_fast = fast_ema.get(idx).copied()?;
    let current_slow = slow_ema.get(idx).copied()?;
    if ![previous_fast, previous_slow, current_fast, current_slow]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }
    if previous_fast <= previous_slow && current_fast > current_slow {
        Some(1)
    } else if previous_fast >= previous_slow && current_fast < current_slow {
        Some(-1)
    } else {
        None
    }
}

fn di_imbalance(plus: Option<f64>, minus: Option<f64>) -> Option<f64> {
    let (plus, minus) = (plus?, minus?);
    let denominator = plus + minus;
    (denominator > f64::EPSILON).then_some((plus - minus) / denominator)
}

fn session_window_at(ts_ns: i64, config: &RegimeAdaptiveGateConfig) -> Option<bool> {
    if ts_ns <= 0
        || !config.use_session_window
        || config.session_start_minute_et == config.session_end_minute_et
    {
        return None;
    }
    let timestamp = DateTime::<Utc>::from_timestamp_nanos(ts_ns).with_timezone(&New_York);
    let weekday_bit = match timestamp.weekday() {
        chrono::Weekday::Mon => 1 << 0,
        chrono::Weekday::Tue => 1 << 1,
        chrono::Weekday::Wed => 1 << 2,
        chrono::Weekday::Thu => 1 << 3,
        chrono::Weekday::Fri => 1 << 4,
        chrono::Weekday::Sat => 1 << 5,
        chrono::Weekday::Sun => 1 << 6,
    };
    if config.session_weekdays_mask & weekday_bit == 0 {
        return Some(false);
    }
    let minute = timestamp.hour() as u16 * 60 + timestamp.minute() as u16;
    let start = config.session_start_minute_et;
    let end = config.session_end_minute_et;
    let inside = if start < end {
        minute >= start && minute < end
    } else {
        minute >= start || minute < end
    };
    Some(inside)
}

fn fmt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn optional_non_negative(value: Option<f64>) -> bool {
    value.is_none_or(|value| value.is_finite() && value >= 0.0)
}

fn optional_finite(value: Option<f64>) -> bool {
    value.is_none_or(f64::is_finite)
}

fn is_below_or_above(value: f64, below: f64, above: Option<f64>) -> bool {
    value < below || above.is_some_and(|threshold| value > threshold)
}

fn is_optional_below_or_above(value: f64, below: Option<f64>, above: Option<f64>) -> bool {
    below.is_some_and(|threshold| value < threshold)
        || above.is_some_and(|threshold| value > threshold)
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
    fn disabled_gate_is_an_exact_neutral_noop() {
        let config = RegimeAdaptiveGateConfig::default();
        let evaluation = config.evaluate(&[bar(1, 100.0, Some(10.0))], 2, 3);
        assert!(!evaluation.inverted);
        assert!(evaluation.ready);
        assert!(!evaluation.should_hold());
    }

    #[test]
    fn missing_volume_is_neutral_without_lookahead() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_relative_volume: true,
            volume_lookback_bars: 2,
            invert_below_relative_volume: 0.8,
            ..RegimeAdaptiveGateConfig::default()
        };
        let prefix = vec![
            bar(1, 100.0, Some(10.0)),
            bar(2, 100.0, Some(10.0)),
            bar(3, 100.0, None),
        ];
        let mut with_future = prefix.clone();
        with_future.push(bar(4, 100.0, Some(1.0)));
        let first = config.evaluate(&prefix, 2, 3);
        let second = config.evaluate(&with_future[..prefix.len()], 2, 3);
        assert_eq!(first, second);
        assert!(!first.inverted);
        assert!(!first.ready);
        assert!(!first.should_hold());
    }

    #[test]
    fn future_bars_do_not_change_a_prefix_feature_snapshot() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_atr_ratio: true,
            use_adx: true,
            use_di_imbalance: true,
            use_ema_spread: true,
            use_ema_slope: true,
            atr_length: 2,
            atr_lookback_bars: 2,
            adx_length: 2,
            ema_slope_lookback: 1,
            ..RegimeAdaptiveGateConfig::default()
        };
        let prefix = vec![
            bar(1, 100.0, Some(10.0)),
            bar(2, 101.0, Some(10.0)),
            bar(3, 100.5, Some(10.0)),
            bar(4, 101.5, Some(10.0)),
            bar(5, 101.0, Some(10.0)),
        ];
        let mut full = prefix.clone();
        full.push(bar(6, 10_000.0, Some(1_000_000.0)));

        fn snapshot(
            config: &RegimeAdaptiveGateConfig,
            bars: &[Bar],
            idx: usize,
        ) -> FeatureSnapshot {
            let closes = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
            let fast = ema_series(&closes, 2);
            let slow = ema_series(&closes, 3);
            let atr = atr_series(bars, config.atr_length);
            let ratio = relative_series_value(&atr, config.atr_lookback_bars);
            let adx = adx_series(bars, config.adx_length);
            config.snapshot_at(bars, idx, &fast, &slow, &atr, &ratio, Some(&adx))
        }

        assert_eq!(
            snapshot(&config, &prefix, prefix.len() - 1),
            snapshot(&config, &full, prefix.len() - 1)
        );
    }

    #[test]
    fn low_volume_can_invert_only_after_causal_dwell() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_relative_volume: true,
            volume_lookback_bars: 2,
            invert_below_relative_volume: 0.8,
            invert_confirmation_bars: 2,
            normal_confirmation_bars: 2,
            ..RegimeAdaptiveGateConfig::default()
        };
        let bars = vec![
            bar(1, 100.0, Some(100.0)),
            bar(2, 100.0, Some(100.0)),
            bar(3, 100.0, Some(100.0)),
            bar(4, 100.0, Some(50.0)),
            bar(5, 100.0, Some(50.0)),
        ];
        let evaluation = config.evaluate(&bars, 2, 3);
        assert!(evaluation.inverted);
        assert!(evaluation.confirmed);
    }

    #[test]
    fn session_feature_uses_et_and_is_causal() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_session_window: true,
            session_start_minute_et: 12 * 60,
            session_end_minute_et: 13 * 60,
            invert_inside_session_window: true,
            ..RegimeAdaptiveGateConfig::default()
        };
        let ts = chrono::NaiveDate::from_ymd_opt(2026, 8, 10)
            .unwrap()
            .and_hms_opt(16, 30, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap();
        let evaluation = config.evaluate(&[bar(ts, 100.0, None)], 2, 3);
        assert!(evaluation.in_session_window == Some(true));
        assert!(evaluation.inverted);
    }

    #[test]
    fn majority_tie_is_neutral() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_session_window: true,
            session_start_minute_et: 12 * 60,
            session_end_minute_et: 13 * 60,
            use_ema_slope: true,
            ema_slope_lookback: 1,
            ..RegimeAdaptiveGateConfig::default()
        };
        // The helper is intentionally tested directly so a two-feature tie
        // cannot silently become an inversion.
        let snapshot = FeatureSnapshot {
            in_session_window: Some(true),
            normalized_slope: Some(1.0),
            ..FeatureSnapshot::default()
        };
        assert_eq!(config.combine_snapshot(&snapshot), None);
    }

    #[test]
    fn xor_combination_is_explicit() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            combine: AdaptiveGateCombine::Xor,
            use_session_window: true,
            session_start_minute_et: 12 * 60,
            session_end_minute_et: 13 * 60,
            use_ema_slope: true,
            ema_slope_lookback: 1,
            ..RegimeAdaptiveGateConfig::default()
        };
        assert_eq!(
            config.combine_snapshot(&FeatureSnapshot {
                in_session_window: Some(true),
                normalized_slope: Some(1.0),
                ..FeatureSnapshot::default()
            }),
            Some(true)
        );
        assert_eq!(
            config.combine_snapshot(&FeatureSnapshot {
                in_session_window: Some(true),
                normalized_slope: Some(-1.0),
                ..FeatureSnapshot::default()
            }),
            Some(false)
        );
    }

    #[test]
    fn high_relative_volume_can_be_an_explicit_inversion_tail() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_relative_volume: true,
            volume_lookback_bars: 2,
            invert_below_relative_volume: 0.2,
            invert_above_relative_volume: Some(2.0),
            ..RegimeAdaptiveGateConfig::default()
        };
        let bars = vec![
            bar(1, 100.0, Some(100.0)),
            bar(2, 100.0, Some(100.0)),
            bar(3, 100.0, Some(250.0)),
        ];
        let evaluation = config.evaluate(&bars, 2, 3);
        assert_eq!(evaluation.relative_volume, Some(2.5));
        assert!(evaluation.inverted);
    }

    #[test]
    fn choppiness_uses_only_the_preceding_window() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_choppiness: true,
            choppiness_length: 3,
            invert_above_choppiness: Some(35.0),
            ..RegimeAdaptiveGateConfig::default()
        };
        let prefix = vec![
            bar(1, 100.0, None),
            bar(2, 101.0, None),
            bar(3, 102.0, None),
            bar(4, 103.0, None),
        ];
        let mut different_current_bar = prefix[..3].to_vec();
        different_current_bar.push(bar(4, 10_000.0, None));

        let first = config.evaluate(&prefix, 2, 3);
        let second = config.evaluate(&different_current_bar, 2, 3);
        assert_eq!(first.choppiness, second.choppiness);
        assert!(first.choppiness.is_some_and(|value| value > 35.0));
        assert!(first.inverted);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn invalid_choppiness_settings_are_rejected_only_when_enabled() {
        let invalid = RegimeAdaptiveGateConfig {
            enabled: true,
            use_choppiness: true,
            choppiness_length: 1,
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(invalid.validate().is_err());

        let disabled = RegimeAdaptiveGateConfig {
            use_choppiness: false,
            choppiness_length: 1,
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(disabled.validate().is_ok());
    }

    #[test]
    fn directional_return_is_aligned_to_the_raw_cross_and_lagged() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_return: true,
            directional_return_length: 2,
            atr_length: 2,
            invert_when_aligned_return_below: Some(0.0),
            ..RegimeAdaptiveGateConfig::default()
        };
        let prefix = vec![
            bar(1, 100.0, None),
            bar(2, 90.0, None),
            bar(3, 90.0, None),
            bar(4, 110.0, None),
        ];
        let mut different_current_bar = prefix[..3].to_vec();
        different_current_bar.push(bar(4, 10_000.0, None));

        let first = config.evaluate(&prefix, 2, 3);
        let second = config.evaluate(&different_current_bar, 2, 3);
        assert_eq!(first.raw_signal_direction, Some(1));
        assert_eq!(first.aligned_return, second.aligned_return);
        assert!(first.aligned_return.is_some_and(|value| value < 0.0));
        assert!(first.inverted);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn directional_return_requires_a_threshold_when_enabled() {
        let invalid = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_return: true,
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(invalid.validate().is_err());

        let disabled = RegimeAdaptiveGateConfig {
            use_directional_return: false,
            invert_when_aligned_return_below: Some(f64::NAN),
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(disabled.validate().is_ok());
    }

    #[test]
    fn directional_gap_is_available_only_on_a_raw_cross() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_ema_gap: true,
            directional_ema_length: 2,
            directional_gap_lookback_bars: 1,
            atr_length: 2,
            invert_when_bullish_gap_below: Some(0.1),
            ..RegimeAdaptiveGateConfig::default()
        };
        let bars = vec![
            bar(1, 100.0, None),
            bar(2, 100.0, None),
            bar(3, 100.0, None),
            bar(4, 110.0, None),
            bar(5, 111.0, None),
        ];
        let closes = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let fast = ema_series(&closes, 2);
        let slow = ema_series(&closes, 3);
        let directional = ema_series(&closes, config.directional_ema_length);
        let atr = atr_series(&bars, config.atr_length);
        let atr_ratio = vec![None; bars.len()];

        let cross = config.snapshot_at_with_direction(
            &bars,
            3,
            &fast,
            &slow,
            &directional,
            &atr,
            &atr_ratio,
            None,
        );
        assert_eq!(cross.raw_signal_direction, Some(1));
        assert_eq!(cross.directional_ema_gap, Some(0.0));
        assert_eq!(config.combine_snapshot(&cross), Some(true));

        let later = config.snapshot_at_with_direction(
            &bars,
            4,
            &fast,
            &slow,
            &directional,
            &atr,
            &atr_ratio,
            None,
        );
        assert_eq!(later.raw_signal_direction, None);
        assert_eq!(config.combine_snapshot(&later), None);
    }

    #[test]
    fn directional_gap_uses_only_the_prior_completed_bar() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_ema_gap: true,
            directional_ema_length: 2,
            directional_gap_lookback_bars: 1,
            atr_length: 2,
            invert_when_bullish_gap_below: Some(0.1),
            ..RegimeAdaptiveGateConfig::default()
        };
        let prefix = vec![
            bar(1, 100.0, None),
            bar(2, 100.0, None),
            bar(3, 100.0, None),
            bar(4, 110.0, None),
        ];
        let mut full = prefix.clone();
        full.push(bar(5, 10_000.0, None));

        fn snapshot(
            config: &RegimeAdaptiveGateConfig,
            bars: &[Bar],
            idx: usize,
        ) -> FeatureSnapshot {
            let closes = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
            let fast = ema_series(&closes, 2);
            let slow = ema_series(&closes, 3);
            let directional = ema_series(&closes, config.directional_ema_length);
            let atr = atr_series(bars, config.atr_length);
            let atr_ratio = vec![None; bars.len()];
            config.snapshot_at_with_direction(
                bars,
                idx,
                &fast,
                &slow,
                &directional,
                &atr,
                &atr_ratio,
                None,
            )
        }

        assert_eq!(snapshot(&config, &prefix, 3), snapshot(&config, &full, 3));
    }

    #[test]
    fn directional_gap_zero_lookback_uses_the_current_closed_bar() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_ema_gap: true,
            directional_ema_length: 2,
            directional_gap_lookback_bars: 0,
            atr_length: 2,
            invert_when_bullish_gap_below: Some(1.0),
            ..RegimeAdaptiveGateConfig::default()
        };
        let bars = vec![
            bar(1, 100.0, None),
            bar(2, 100.0, None),
            bar(3, 100.0, None),
            bar(4, 110.0, None),
        ];
        let closes = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let fast = ema_series(&closes, 2);
        let slow = ema_series(&closes, 3);
        let directional = ema_series(&closes, config.directional_ema_length);
        let atr = atr_series(&bars, config.atr_length);
        let atr_ratio = vec![None; bars.len()];
        let snapshot = config.snapshot_at_with_direction(
            &bars,
            3,
            &fast,
            &slow,
            &directional,
            &atr,
            &atr_ratio,
            None,
        );
        let prior_snapshot = RegimeAdaptiveGateConfig {
            directional_gap_lookback_bars: 1,
            ..config.clone()
        }
        .snapshot_at_with_direction(
            &bars,
            3,
            &fast,
            &slow,
            &directional,
            &atr,
            &atr_ratio,
            None,
        );

        assert_eq!(snapshot.raw_signal_direction, Some(1));
        let current_gap = snapshot.directional_ema_gap.expect("current gap");
        assert!(current_gap.is_finite() && current_gap > 0.0);
        assert!(
            prior_snapshot
                .directional_ema_gap
                .is_some_and(|prior_gap| current_gap > prior_gap)
        );
        assert_eq!(config.combine_snapshot(&snapshot), Some(true));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn directional_gap_requires_a_finite_threshold_when_enabled() {
        let no_threshold = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_ema_gap: true,
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(no_threshold.validate().is_err());

        let invalid_threshold = RegimeAdaptiveGateConfig {
            enabled: true,
            use_directional_ema_gap: true,
            invert_when_bullish_gap_below: Some(f64::NAN),
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(invalid_threshold.validate().is_err());

        let feature_disabled = RegimeAdaptiveGateConfig {
            invert_when_bullish_gap_below: Some(f64::NAN),
            ..RegimeAdaptiveGateConfig::default()
        };
        assert!(feature_disabled.validate().is_ok());
    }

    #[test]
    fn higher_timeframe_context_uses_only_completed_buckets() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_higher_timeframe_context: true,
            higher_timeframe_minutes: 5,
            higher_timeframe_fast_length: 2,
            higher_timeframe_slow_length: 3,
            higher_timeframe_atr_length: 2,
            higher_timeframe_slope_lookback: 1,
            higher_timeframe_persistence_bars: 1,
            higher_timeframe_spread_atr_floor: 0.0,
            higher_timeframe_slope_atr_floor: 0.0,
            ..RegimeAdaptiveGateConfig::default()
        };
        let start = 1_000_000_000_i64;
        let mut bars = Vec::new();
        for group in 0..5 {
            for offset in 0..5 {
                let close = 100.0 + group as f64;
                bars.push(bar(
                    start + (group * 5 + offset) as i64 * 60 * 1_000_000_000,
                    close,
                    None,
                ));
            }
        }

        let prefix = higher_timeframe_context_series(&bars, &config);
        let mut with_future = bars.clone();
        with_future.push(bar(start + 25 * 60 * 1_000_000_000, 10_000.0, None));
        with_future.push(bar(start + 26 * 60 * 1_000_000_000, 10_000.0, None));

        assert_eq!(
            prefix[bars.len() - 1],
            Some(HigherTimeframeSnapshot {
                direction: 1,
                spread: prefix[bars.len() - 1].unwrap().spread,
                slope: prefix[bars.len() - 1].unwrap().slope,
            })
        );
        assert_eq!(
            prefix[bars.len() - 1],
            higher_timeframe_context_series(&with_future, &config)[bars.len() - 1]
        );
        assert!(higher_timeframe_context_series(&bars[..5], &config)[4].is_none());
        assert!(config.validate().is_ok());

        let mut capped = config.clone();
        capped.higher_timeframe_spread_atr_ceiling = Some(0.0001);
        assert!(higher_timeframe_context_series(&bars, &capped)[bars.len() - 1].is_none());
        assert!(capped.validate().is_ok());
    }

    #[test]
    fn hma_secondary_context_is_causal_and_serializes_as_opt_in_kind() {
        let config = RegimeAdaptiveGateConfig {
            enabled: true,
            use_higher_timeframe_context: true,
            higher_timeframe_average_kind: HigherTimeframeAverageKind::Hma,
            higher_timeframe_minutes: 5,
            higher_timeframe_fast_length: 3,
            higher_timeframe_slow_length: 5,
            higher_timeframe_atr_length: 2,
            higher_timeframe_slope_lookback: 1,
            higher_timeframe_persistence_bars: 1,
            higher_timeframe_spread_atr_floor: 0.0,
            higher_timeframe_slope_atr_floor: 0.0,
            ..RegimeAdaptiveGateConfig::default()
        };
        let start = 1_000_000_000_i64;
        let mut bars = Vec::new();
        for group in 0..8 {
            for offset in 0..5 {
                let close = 100.0 + group as f64;
                bars.push(bar(
                    start + (group * 5 + offset) as i64 * 60 * 1_000_000_000,
                    close,
                    None,
                ));
            }
        }

        let prefix = higher_timeframe_context_series(&bars, &config);
        let mut with_future = bars.clone();
        with_future.push(bar(start + 40 * 60 * 1_000_000_000, 10_000.0, None));
        with_future.push(bar(start + 41 * 60 * 1_000_000_000, 10_000.0, None));

        assert!(prefix[bars.len() - 1].is_some());
        assert_eq!(
            prefix[bars.len() - 1],
            higher_timeframe_context_series(&with_future, &config)[bars.len() - 1]
        );
        assert_eq!(
            serde_json::to_value(config.higher_timeframe_average_kind).unwrap(),
            serde_json::Value::String("hma".to_string())
        );
        assert!(config.validate().is_ok());
    }
}
