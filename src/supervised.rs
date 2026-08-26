//! Event-level supervised-learning dataset preparation.
//!
//! The dataset is intentionally sparse: one row is emitted for each causal
//! trigger crossover, not for every bar.  Rolling indicators and all feature
//! values are evaluated at the closed crossover bar.  The labeler is allowed
//! to look forward, because its output is a target, but those future prices
//! are kept in explicitly named label/audit columns and are never part of the
//! feature registry.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveTime, Timelike, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

pub const SUPERVISED_DATASET_SCHEMA: &str = "supervised-event-v1";
pub const SUPERVISED_LABEL_SCHEMA: &str = "backward-dynamic-program-v1";
/// Full bar-level artifact shared by the supervised, GA, and RL pipelines.
///
/// The event parquet remains the authoritative sparse supervised input.  This
/// schema is the dense companion: it keeps every source bar, its causal
/// feature snapshot, and nullable event/label audit columns on the same row.
pub const TRAINING_BAR_DATASET_SCHEMA: &str = "training-bar-v1";

/// Actions available to the hindsight event labeler.
///
/// `NormalInvert` is intentionally a two-class target: it still evaluates the
/// skip counterfactual for audit/debug columns, but skip can never win the
/// dynamic program and no event is emitted with a skip label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LabelMode {
    #[default]
    #[serde(rename = "normal-skip-invert", alias = "three-way")]
    NormalSkipInvert,
    #[serde(rename = "normal-invert", alias = "normal-reverse")]
    NormalInvert,
}

impl LabelMode {
    pub fn class_names(self) -> &'static [&'static str] {
        match self {
            Self::NormalSkipInvert => &["normal", "skip", "invert"],
            Self::NormalInvert => &["normal", "invert"],
        }
    }

    pub fn class_count(self) -> usize {
        self.class_names().len()
    }

    pub fn includes_skip(self) -> bool {
        matches!(self, Self::NormalSkipInvert)
    }

    pub fn class_index(self, action: i8) -> Option<usize> {
        match (self, action) {
            (Self::NormalSkipInvert, 1) => Some(0),
            (Self::NormalSkipInvert, 0) => Some(1),
            (Self::NormalSkipInvert, -1) => Some(2),
            (Self::NormalInvert, 1) => Some(0),
            (Self::NormalInvert, -1) => Some(1),
            _ => None,
        }
    }

    pub fn action_for_class(self, class: usize, raw_direction: i8, current: i8) -> Option<i8> {
        match (self, class) {
            (Self::NormalSkipInvert, 0) => Some(raw_direction),
            (Self::NormalSkipInvert, 1) => Some(current),
            (Self::NormalSkipInvert, 2) => Some(-raw_direction),
            (Self::NormalInvert, 0) => Some(raw_direction),
            (Self::NormalInvert, 1) => Some(-raw_direction),
            _ => None,
        }
    }
}

impl fmt::Display for LabelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NormalSkipInvert => "normal-skip-invert",
            Self::NormalInvert => "normal-invert",
        })
    }
}

impl FromStr for LabelMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "normal-skip-invert" | "three-way" | "normal-skip-reverse" => {
                Ok(Self::NormalSkipInvert)
            }
            "normal-invert" | "normal-reverse" | "normal-reverse-only" => Ok(Self::NormalInvert),
            other => bail!("unknown label mode `{other}`; use normal-skip-invert or normal-invert"),
        }
    }
}

fn label_mode_is_default(value: &LabelMode) -> bool {
    *value == LabelMode::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndicatorKind {
    Sma,
    Ema,
    Hma,
    Kama,
    Alma,
    Atr,
    Adx,
    Rsi,
    Stoch,
    Rvol,
    Cmf,
    Vwap,
    #[serde(rename = "di_spread", alias = "dispread")]
    DiSpread,
    #[serde(rename = "di_plus", alias = "diplus", alias = "plus_di")]
    DiPlus,
    #[serde(rename = "di_minus", alias = "diminus", alias = "minus_di")]
    DiMinus,
    Chop,
    Vhf,
    Er,
    Close,
}

impl fmt::Display for IndicatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sma => "sma",
            Self::Ema => "ema",
            Self::Hma => "hma",
            Self::Kama => "kama",
            Self::Alma => "alma",
            Self::Atr => "atr",
            Self::Adx => "adx",
            Self::Rsi => "rsi",
            Self::Stoch => "stoch",
            Self::Rvol => "rvol",
            Self::Cmf => "cmf",
            Self::Vwap => "vwap",
            Self::DiSpread => "di_spread",
            Self::DiPlus => "di_plus",
            Self::DiMinus => "di_minus",
            Self::Chop => "chop",
            Self::Vhf => "vhf",
            Self::Er => "er",
            Self::Close => "close",
        })
    }
}

impl FromStr for IndicatorKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sma" => Ok(Self::Sma),
            "ema" => Ok(Self::Ema),
            "hma" => Ok(Self::Hma),
            "kama" | "kaufman" => Ok(Self::Kama),
            "alma" => Ok(Self::Alma),
            "atr" => Ok(Self::Atr),
            "adx" => Ok(Self::Adx),
            "rsi" => Ok(Self::Rsi),
            "stoch" | "stochastic" | "stochastic_k" => Ok(Self::Stoch),
            "rvol" | "relative_volume" => Ok(Self::Rvol),
            "cmf" | "chaikin_money_flow" => Ok(Self::Cmf),
            "vwap" | "vwap_distance" | "rolling_vwap_distance" => Ok(Self::Vwap),
            "di_spread" | "di" | "directional_imbalance" => Ok(Self::DiSpread),
            "di_plus" | "diplus" | "plus_di" => Ok(Self::DiPlus),
            "di_minus" | "diminus" | "minus_di" => Ok(Self::DiMinus),
            "chop" | "choppiness" | "choppiness_index" => Ok(Self::Chop),
            "vhf" | "vertical_horizontal_filter" => Ok(Self::Vhf),
            "er" | "efficiency_ratio" | "kaufman_er" => Ok(Self::Er),
            "close" | "price" => Ok(Self::Close),
            other => bail!("unknown indicator `{other}`"),
        }
    }
}

fn default_lookbacks() -> Vec<usize> {
    vec![1, 3, 5]
}

fn is_empty_usize_vec(values: &Vec<usize>) -> bool {
    values.is_empty()
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// A selectable indicator and the causal values/deltas to expose around an
/// event. A delta is `(value[t] - value[t-lookback]) / ATR[t]` when
/// normalization is enabled. `normalize_value_by_atr` applies the same
/// price-scale normalization to the current value, which prevents absolute
/// futures price levels from becoming model inputs. RSI and stochastic %K are
/// exceptions: they are already bounded, so their values are emitted as
/// `(value - 50) / 50` and their deltas as
/// `(value[t] - value[t-lookback]) / 100`. Lookbacks are evaluated strictly at
/// or before the event row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureSpec {
    pub indicator: IndicatorKind,
    pub period: usize,
    #[serde(default = "default_lookbacks")]
    pub lookbacks: Vec<usize>,
    #[serde(default = "default_true")]
    pub include_value: bool,
    #[serde(default = "default_true")]
    pub include_delta: bool,
    #[serde(default = "default_true")]
    pub normalize_by_atr: bool,
    /// Normalize the current indicator value by ATR and suffix its feature
    /// name with `_atr`. This is separate from `normalize_by_atr` so legacy
    /// feature registries remain reproducible.
    #[serde(default, skip_serializing_if = "is_false")]
    pub normalize_value_by_atr: bool,
}

fn default_true() -> bool {
    true
}

/// Optional causal features that describe the context around a crossover
/// without adding another moving-average family.  These are kept separate
/// from `FeatureSpec` because they are derived from event history, timestamps,
/// and rolling OHLC rather than from one indicator series.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DerivedFeatureConfig {
    /// Encode local time as a continuous 24-hour phase.
    #[serde(default, skip_serializing_if = "is_false")]
    pub include_time_cycle: bool,
    /// Previous-trigger history windows, measured in source bars.  Each
    /// window emits a causal crossover count; the first previous crossover
    /// also emits bars-since and ATR-normalized displacement features.
    #[serde(default, skip_serializing_if = "is_empty_usize_vec")]
    pub cross_lookbacks: Vec<usize>,
    /// Previous context-pair history windows, measured in source bars.  These
    /// describe the configured context fast/slow pair independently from the
    /// trigger-pair history above.  The history also emits the current
    /// context-cross direction, the previous context-cross direction, bars
    /// since that cross, and ATR-normalized displacement since it.
    #[serde(default, skip_serializing_if = "is_empty_usize_vec")]
    pub context_cross_lookbacks: Vec<usize>,
    /// Rolling high-low range windows, measured in source bars.  Each window
    /// emits range width in ATRs and the close's position within that range.
    #[serde(default, skip_serializing_if = "is_empty_usize_vec")]
    pub range_lookbacks: Vec<usize>,
    /// Add `ATR(config.atr_period) / ATR(slow_period)` as a normalized
    /// volatility-regime feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atr_ratio_slow_period: Option<usize>,
    /// Add explicit cross-direction interactions for the linear baseline.
    /// These let a logistic model learn that the same regime state can imply
    /// different normal/invert choices for bullish and bearish crosses.
    #[serde(default, skip_serializing_if = "is_false")]
    pub include_direction_interactions: bool,
    /// Add causal persistence and transition-timing features. Each lookback
    /// measures the state immediately before the current bar/event so a
    /// crossover cannot contribute its own post-transition bars.
    #[serde(default, skip_serializing_if = "is_empty_usize_vec")]
    pub regime_persistence_lookbacks: Vec<usize>,
}

impl DerivedFeatureConfig {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

fn derived_features_are_default(value: &DerivedFeatureConfig) -> bool {
    value.is_default()
}

fn direction_interaction_names() -> &'static [&'static str] {
    &[
        "direction_x_trigger_spread_atr",
        "direction_x_context_spread_atr",
        "direction_x_trigger_fast_slope_atr",
        "direction_x_trigger_slow_slope_atr",
        "direction_x_context_fast_slope_atr",
        "direction_x_context_slow_slope_atr",
        "direction_x_price_distance_trigger_slow_atr",
        "direction_x_adx_14",
        "direction_x_er_14",
        "direction_x_rvol_20",
        "direction_x_cmf_20",
        "direction_x_vwap_20",
        "direction_x_di_spread_14",
        "direction_x_chop_30",
        "direction_x_vhf_30",
    ]
}

impl FeatureSpec {
    pub fn new(indicator: IndicatorKind, period: usize) -> Self {
        let normalize_by_atr = !matches!(
            indicator,
            IndicatorKind::Rsi
                | IndicatorKind::Stoch
                | IndicatorKind::Cmf
                | IndicatorKind::Vwap
                | IndicatorKind::DiSpread
                | IndicatorKind::DiPlus
                | IndicatorKind::DiMinus
                | IndicatorKind::Chop
                | IndicatorKind::Vhf
        );
        Self {
            indicator,
            period,
            lookbacks: default_lookbacks(),
            include_value: true,
            include_delta: true,
            normalize_by_atr,
            normalize_value_by_atr: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.period == 0 {
            bail!(
                "feature {} period must be greater than zero",
                self.indicator
            );
        }
        if self.lookbacks.iter().any(|value| *value == 0) {
            bail!(
                "feature {}_{} lookbacks must be greater than zero",
                self.indicator,
                self.period
            );
        }
        if !self.include_value && !self.include_delta {
            bail!(
                "feature {}_{} must include a value or a delta",
                self.indicator,
                self.period
            );
        }
        if matches!(self.indicator, IndicatorKind::Hma) && self.period < 2 {
            bail!("HMA feature periods must be at least 2");
        }
        if matches!(self.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch)
            && (self.normalize_by_atr || self.normalize_value_by_atr)
        {
            bail!(
                "bounded oscillator features must use centered/scaled output; disable ATR normalization"
            );
        }
        Ok(())
    }

    pub fn base_name(&self) -> String {
        format!("{}_{}", self.indicator, self.period)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if matches!(self.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch) {
            if self.include_value {
                names.push(format!("{}_centered", self.base_name()));
            }
            if self.include_delta {
                for lookback in &self.lookbacks {
                    names.push(format!("{}_delta_{}_scaled", self.base_name(), lookback));
                }
            }
            return names;
        }
        if self.include_value {
            names.push(format!(
                "{}{}",
                self.base_name(),
                if self.normalize_value_by_atr {
                    "_atr"
                } else {
                    ""
                }
            ));
        }
        if self.include_delta {
            for lookback in &self.lookbacks {
                let suffix = if self.normalize_by_atr { "_atr" } else { "" };
                names.push(format!("{}_delta_{}{}", self.base_name(), lookback, suffix));
            }
        }
        names
    }
}

fn default_features() -> Vec<FeatureSpec> {
    [
        (IndicatorKind::Ema, 10),
        (IndicatorKind::Ema, 30),
        (IndicatorKind::Ema, 210),
        (IndicatorKind::Ema, 240),
        (IndicatorKind::Hma, 10),
        (IndicatorKind::Hma, 30),
        (IndicatorKind::Kama, 30),
        (IndicatorKind::Alma, 30),
        (IndicatorKind::Sma, 30),
        (IndicatorKind::Atr, 14),
        (IndicatorKind::Adx, 14),
        (IndicatorKind::Rvol, 20),
        (IndicatorKind::Er, 14),
    ]
    .into_iter()
    .map(|(kind, period)| FeatureSpec::new(kind, period))
    .collect()
}

fn default_trigger_kind() -> IndicatorKind {
    IndicatorKind::Ema
}

fn default_context_kind() -> IndicatorKind {
    IndicatorKind::Ema
}

fn default_timezone() -> String {
    "America/New_York".to_string()
}

fn default_bar_kind() -> String {
    "minute".to_string()
}

fn default_bar_value() -> f64 {
    1.0
}

fn default_start_hour() -> u32 {
    18
}

fn default_end_hour() -> u32 {
    17
}

fn default_contract_multiplier() -> f64 {
    1.0
}

/// Return the standard USD value of a one-point move for the supported
/// outright futures symbols.  This is a PnL conversion, not a margin or
/// leverage setting.  The generic `SupervisedConfig` intentionally keeps a
/// neutral 1.0 default for library callers; the supervised CLI resolves this
/// table before preparing a futures dataset and refuses to fall back to raw
/// price units for an unknown symbol.
pub fn standard_contract_multiplier(instrument: &str, contract: &str) -> Option<f64> {
    for symbol in [instrument, contract] {
        let symbol = symbol
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_uppercase)
            .collect::<String>();
        if symbol.is_empty() || symbol == "UNKNOWN" {
            continue;
        }
        let multiplier = if symbol.starts_with("MGC") {
            10.0
        } else if symbol.starts_with("GC") {
            100.0
        } else if symbol.starts_with("MES") {
            5.0
        } else if symbol.starts_with("ES") {
            50.0
        } else if symbol.starts_with("MNQ") {
            2.0
        } else if symbol.starts_with("NQ") {
            20.0
        } else {
            continue;
        };
        return Some(multiplier);
    }
    None
}

/// Standard outright futures minimum price increment used when deriving raw
/// trade range bars.  `bar_value` for a range dataset is expressed in ticks,
/// matching Trader's replay bar type (for example GC range 10 = 10 * 0.10).
pub fn standard_tick_size(instrument: &str, contract: &str) -> Option<f64> {
    for symbol in [instrument, contract] {
        let symbol = symbol
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_uppercase)
            .collect::<String>();
        if symbol.is_empty() || symbol == "UNKNOWN" {
            continue;
        }
        let tick_size = if symbol.starts_with("GC") || symbol.starts_with("MGC") {
            0.10
        } else if symbol.starts_with("ES")
            || symbol.starts_with("MES")
            || symbol.starts_with("NQ")
            || symbol.starts_with("MNQ")
        {
            0.25
        } else {
            continue;
        };
        return Some(tick_size);
    }
    None
}

/// Configurable event extraction and label semantics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SupervisedConfig {
    #[serde(default = "default_trigger_kind")]
    pub trigger_kind: IndicatorKind,
    #[serde(default = "default_trigger_fast")]
    pub trigger_fast: usize,
    #[serde(default = "default_trigger_slow")]
    pub trigger_slow: usize,
    #[serde(default = "default_context_kind")]
    pub context_kind: IndicatorKind,
    #[serde(default = "default_context_fast")]
    pub context_fast: usize,
    #[serde(default = "default_context_slow")]
    pub context_slow: usize,
    #[serde(default = "default_atr_period")]
    pub atr_period: usize,
    #[serde(default = "default_slope_lookback")]
    pub slope_lookback: usize,
    #[serde(default = "default_features")]
    pub features: Vec<FeatureSpec>,
    /// Optional causal context features used by feature ablation experiments.
    /// The field is omitted from serialized legacy configs when disabled so
    /// existing supervised artifacts retain their exact config registry.
    #[serde(default, skip_serializing_if = "derived_features_are_default")]
    pub derived_features: DerivedFeatureConfig,
    #[serde(default = "default_timezone")]
    pub session_timezone: String,
    #[serde(default = "default_start_hour")]
    pub session_start_hour: u32,
    #[serde(default = "default_end_hour")]
    pub session_end_hour: u32,
    #[serde(default = "default_bar_kind")]
    pub bar_kind: String,
    #[serde(default = "default_bar_value")]
    pub bar_value: f64,
    #[serde(default = "default_contract_multiplier")]
    pub contract_multiplier: f64,
    /// Cost of one complete round trip.  An entry or exit costs half of this;
    /// reversing from -1 to +1 costs one complete round trip.
    #[serde(default)]
    pub round_trip_cost: f64,
    /// Include raw trigger direction as a causal model feature.
    #[serde(default = "default_true")]
    pub include_raw_direction: bool,
    /// Require every price-scale feature value and delta to be expressed as
    /// geometry (ATR-normalized) rather than an absolute market price. This
    /// is opt-in so legacy datasets remain reproducible, while new
    /// cross-contract training configs can fail closed if a raw price feature
    /// is accidentally added.
    #[serde(default, skip_serializing_if = "is_false")]
    pub require_dimensionless_features: bool,
    /// Hindsight action set. The default preserves the original three-class
    /// normal/skip/invert labeler; normal-invert excludes skip from selection.
    #[serde(default, skip_serializing_if = "label_mode_is_default")]
    pub label_mode: LabelMode,
}

fn default_trigger_fast() -> usize {
    10
}
fn default_trigger_slow() -> usize {
    30
}
fn default_context_fast() -> usize {
    210
}
fn default_context_slow() -> usize {
    240
}
fn default_atr_period() -> usize {
    14
}
fn default_slope_lookback() -> usize {
    5
}

impl Default for SupervisedConfig {
    fn default() -> Self {
        Self {
            trigger_kind: default_trigger_kind(),
            trigger_fast: default_trigger_fast(),
            trigger_slow: default_trigger_slow(),
            context_kind: default_context_kind(),
            context_fast: default_context_fast(),
            context_slow: default_context_slow(),
            atr_period: default_atr_period(),
            slope_lookback: default_slope_lookback(),
            features: default_features(),
            derived_features: DerivedFeatureConfig::default(),
            session_timezone: default_timezone(),
            session_start_hour: default_start_hour(),
            session_end_hour: default_end_hour(),
            bar_kind: default_bar_kind(),
            bar_value: default_bar_value(),
            contract_multiplier: default_contract_multiplier(),
            round_trip_cost: 0.0,
            include_raw_direction: true,
            require_dimensionless_features: false,
            label_mode: LabelMode::default(),
        }
    }
}

impl SupervisedConfig {
    pub fn validate(&self) -> Result<()> {
        if self.trigger_fast == 0 || self.trigger_slow == 0 {
            bail!("trigger periods must be greater than zero");
        }
        if self.trigger_fast >= self.trigger_slow {
            bail!("trigger_fast must be less than trigger_slow");
        }
        if self.context_fast == 0 || self.context_slow == 0 {
            bail!("context periods must be greater than zero");
        }
        if self.context_fast >= self.context_slow {
            bail!("context_fast must be less than context_slow");
        }
        if self.atr_period == 0 || self.slope_lookback == 0 {
            bail!("atr_period and slope_lookback must be greater than zero");
        }
        if self.session_start_hour > 23 || self.session_end_hour > 23 {
            bail!("session hours must be in the range 0..23");
        }
        if self.session_start_hour <= self.session_end_hour {
            bail!("session window must cross midnight (start hour must be after end hour)");
        }
        if !matches!(
            self.bar_kind.to_ascii_lowercase().as_str(),
            "minute"
                | "minutes"
                | "1m"
                | "second"
                | "seconds"
                | "1s"
                | "tick"
                | "ticks"
                | "volume"
                | "range"
        ) || !self.bar_value.is_finite()
            || self.bar_value <= 0.0
        {
            bail!(
                "bar_kind must be minute, second, tick, volume, or range and bar_value must be positive"
            );
        }
        if !self.contract_multiplier.is_finite() || self.contract_multiplier <= 0.0 {
            bail!("contract_multiplier must be finite and positive");
        }
        if !self.round_trip_cost.is_finite() || self.round_trip_cost < 0.0 {
            bail!("round_trip_cost must be finite and non-negative");
        }
        Tz::from_str(&self.session_timezone)
            .with_context(|| format!("invalid session timezone `{}`", self.session_timezone))?;
        for spec in &self.features {
            spec.validate()?;
            if self.require_dimensionless_features
                && matches!(
                    spec.indicator,
                    IndicatorKind::Sma
                        | IndicatorKind::Ema
                        | IndicatorKind::Hma
                        | IndicatorKind::Kama
                        | IndicatorKind::Alma
                        | IndicatorKind::Atr
                        | IndicatorKind::Close
                )
            {
                if spec.include_value && !spec.normalize_value_by_atr {
                    bail!(
                        "dimensionless feature mode rejects raw {}_{} value; set normalize_value_by_atr=true or disable include_value",
                        spec.indicator,
                        spec.period
                    );
                }
                if spec.include_delta && !spec.normalize_by_atr {
                    bail!(
                        "dimensionless feature mode rejects raw {}_{} delta; set normalize_by_atr=true or disable include_delta",
                        spec.indicator,
                        spec.period
                    );
                }
            }
        }
        for lookback in self
            .derived_features
            .cross_lookbacks
            .iter()
            .chain(self.derived_features.context_cross_lookbacks.iter())
            .chain(self.derived_features.range_lookbacks.iter())
            .chain(self.derived_features.regime_persistence_lookbacks.iter())
        {
            if *lookback == 0 {
                bail!("derived feature lookbacks must be greater than zero");
            }
        }
        if self
            .derived_features
            .atr_ratio_slow_period
            .is_some_and(|period| period == 0)
        {
            bail!("atr_ratio_slow_period must be greater than zero");
        }
        let names = self.feature_names();
        let unique = names.iter().collect::<BTreeSet<_>>();
        if unique.len() != names.len() {
            bail!("feature selection produces duplicate feature names");
        }
        Ok(())
    }

    pub fn trigger_name(&self, fast: bool) -> String {
        format!(
            "{}_{}",
            self.trigger_kind,
            if fast {
                self.trigger_fast
            } else {
                self.trigger_slow
            }
        )
    }

    pub fn context_name(&self, fast: bool) -> String {
        format!(
            "{}_{}",
            self.context_kind,
            if fast {
                self.context_fast
            } else {
                self.context_slow
            }
        )
    }

    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.include_raw_direction {
            names.push("raw_direction".to_string());
        }
        for spec in &self.features {
            names.extend(spec.names());
        }
        names.push("trigger_spread_atr".to_string());
        names.push("context_spread_atr".to_string());
        names.push("trigger_fast_slope_atr".to_string());
        names.push("trigger_slow_slope_atr".to_string());
        names.push("context_fast_slope_atr".to_string());
        names.push("context_slow_slope_atr".to_string());
        names.push("price_distance_trigger_slow_atr".to_string());
        if self.derived_features.include_time_cycle {
            names.push("tod_sin".to_string());
            names.push("tod_cos".to_string());
        }
        if !self.derived_features.cross_lookbacks.is_empty() {
            names.push("has_previous_cross".to_string());
            names.push("bars_since_last_cross".to_string());
            names.push("displacement_since_last_cross_atr".to_string());
            for lookback in &self.derived_features.cross_lookbacks {
                names.push(format!("crosses_last_{lookback}_bars"));
            }
        }
        if !self.derived_features.context_cross_lookbacks.is_empty() {
            names.push("context_cross_direction".to_string());
            names.push("context_has_previous_cross".to_string());
            names.push("context_bars_since_last_cross".to_string());
            names.push("context_last_cross_direction".to_string());
            names.push("context_displacement_since_last_cross_atr".to_string());
            for lookback in &self.derived_features.context_cross_lookbacks {
                names.push(format!("context_crosses_last_{lookback}_bars"));
            }
        }
        for lookback in &self.derived_features.range_lookbacks {
            names.push(format!("range_{lookback}_atr"));
            names.push(format!("position_in_range_{lookback}"));
        }
        if !self
            .derived_features
            .regime_persistence_lookbacks
            .is_empty()
        {
            names.push("trigger_pre_run_bars".to_string());
            names.push("context_pre_run_bars".to_string());
            for lookback in &self.derived_features.regime_persistence_lookbacks {
                names.push(format!("trigger_pre_persistence_{lookback}"));
                names.push(format!("trigger_pre_flip_rate_{lookback}"));
                names.push(format!("trigger_pre_return_aligned_{lookback}_atr"));
                names.push(format!("context_pre_persistence_{lookback}"));
                names.push(format!("context_pre_flip_rate_{lookback}"));
                names.push(format!("trigger_context_pre_agreement_{lookback}"));
            }
        }
        if let Some(period) = self.derived_features.atr_ratio_slow_period {
            names.push(format!("atr_{}_over_atr_{}", self.atr_period, period));
        }
        if self.derived_features.include_direction_interactions {
            for name in direction_interaction_names() {
                names.push(name.to_string());
            }
        }
        names
    }

    pub fn feature_schema(&self) -> String {
        self.feature_names().join(",")
    }
}

#[derive(Debug, Clone)]
pub struct SupervisedBars {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamp_ns: Vec<i64>,
}

/// Causal feature values for one source bar.  Values are empty during
/// indicator warmup or outside the configured session; the parquet writer
/// represents those unavailable values as NaN and exposes `ready` explicitly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SupervisedBarFeatureRow {
    pub row_idx: usize,
    pub timestamp_ns: i64,
    pub session_id: Option<String>,
    pub ready: bool,
    /// Direction of a crossover observed on this bar, if any.  A value of
    /// zero in the dense parquet means that this row is not a crossover.
    pub cross_direction: Option<i8>,
    pub features: BTreeMap<String, f64>,
}

impl SupervisedBars {
    pub fn validate(&self) -> Result<()> {
        let len = self.close.len();
        if len < 2 {
            bail!("at least two bars are required");
        }
        for (name, actual) in [
            ("open", self.open.len()),
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("volume", self.volume.len()),
            ("timestamp_ns", self.timestamp_ns.len()),
        ] {
            if actual != len {
                bail!("{name} length {actual} does not match close length {len}");
            }
        }
        for (name, values) in [
            ("open", self.open.as_slice()),
            ("high", self.high.as_slice()),
            ("low", self.low.as_slice()),
            ("close", self.close.as_slice()),
            ("volume", self.volume.as_slice()),
        ] {
            if values.iter().any(|value| !value.is_finite()) {
                bail!("{name} contains non-finite values");
            }
        }
        if self
            .timestamp_ns
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            bail!("timestamp_ns must be strictly increasing");
        }
        if self
            .high
            .iter()
            .zip(self.low.iter())
            .zip(self.open.iter().zip(self.close.iter()))
            .any(|((high, low), (open, close))| {
                *high < *low || *open < *low || *open > *high || *close < *low || *close > *high
            })
        {
            bail!("OHLC values contain a bar outside its high/low range");
        }
        Ok(())
    }
}

/// One row in the supervised dataset.  `features` is the only vector a
/// trainer should consume.  The action values, chosen label, and future price
/// columns are deliberately separate audit/target data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SupervisedEvent {
    pub event_id: usize,
    pub session_id: String,
    pub session_event_index: usize,
    pub row_idx: usize,
    pub timestamp_ns: i64,
    pub raw_direction: i8,
    pub decision_price: f64,
    pub interval_end_price: f64,
    pub interval_end_row_idx: usize,
    pub terminal_event: bool,
    pub features: BTreeMap<String, f64>,
    pub action_value_normal: f64,
    pub action_value_skip: f64,
    pub action_value_invert: f64,
    pub label_action: i8,
    pub label_name: String,
    pub oracle_position_before: i8,
    pub oracle_position_after: i8,
    pub oracle_value: f64,
}

impl SupervisedEvent {
    pub fn feature_values(&self, names: &[String]) -> Result<Vec<f64>> {
        names
            .iter()
            .map(|name| {
                let value = self
                    .features
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("event is missing feature `{name}`"))?;
                if !value.is_finite() {
                    bail!("feature `{name}` is non-finite at event {}", self.event_id);
                }
                Ok(value)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct IndicatorSeries {
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CandidateEvent {
    event_id: usize,
    session_id: String,
    session_event_index: usize,
    row_idx: usize,
    timestamp_ns: i64,
    raw_direction: i8,
    decision_price: f64,
    features: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
struct PreviousCross {
    row_idx: usize,
    close: f64,
    direction: i8,
}

/// Build all causal event rows and assign a backward dynamic-programming
/// normal/skip/invert label independently for each session.
pub fn prepare_events(
    bars: &SupervisedBars,
    config: &SupervisedConfig,
) -> Result<Vec<SupervisedEvent>> {
    bars.validate()?;
    config.validate()?;

    let timezone = Tz::from_str(&config.session_timezone)?;
    let session_ids = bars
        .timestamp_ns
        .iter()
        .map(|timestamp| session_id(*timestamp, timezone, config))
        .collect::<Result<Vec<_>>>()?;
    let indicator_series = build_indicator_series_map(bars, config);
    let mut series = indicator_series
        .values()
        .cloned()
        .map(|values| IndicatorSeries { values })
        .collect::<Vec<_>>();
    let atr = atr_wilder(&bars.high, &bars.low, &bars.close, config.atr_period);
    series.push(IndicatorSeries {
        values: atr.clone(),
    });
    let atr_slow = config
        .derived_features
        .atr_ratio_slow_period
        .map(|period| atr_wilder(&bars.high, &bars.low, &bars.close, period));
    if let Some(values) = &atr_slow {
        series.push(IndicatorSeries {
            values: values.clone(),
        });
    }
    let trigger_fast = indicator_series
        .get(&indicator_key(config.trigger_kind, config.trigger_fast))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("trigger fast series was not prepared"))?;
    let trigger_slow = indicator_series
        .get(&indicator_key(config.trigger_kind, config.trigger_slow))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("trigger slow series was not prepared"))?;
    let context_fast = indicator_series
        .get(&indicator_key(config.context_kind, config.context_fast))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("context fast series was not prepared"))?;
    let context_slow = indicator_series
        .get(&indicator_key(config.context_kind, config.context_slow))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("context slow series was not prepared"))?;
    let trigger_spreads = trigger_fast
        .iter()
        .zip(&trigger_slow)
        .map(|(fast, slow)| fast - slow)
        .collect::<Vec<_>>();
    let context_spreads = context_fast
        .iter()
        .zip(&context_slow)
        .map(|(fast, slow)| fast - slow)
        .collect::<Vec<_>>();

    let mut candidates = Vec::new();
    let mut previous_trigger_crosses = BTreeMap::<String, Vec<PreviousCross>>::new();
    let mut previous_context_crosses = BTreeMap::<String, Vec<PreviousCross>>::new();
    let mut event_index = 0usize;
    for i in 1..bars.close.len() {
        let Some(session_id) = session_ids[i].as_ref() else {
            continue;
        };
        if session_ids[i - 1].as_ref() != Some(session_id) {
            continue;
        }
        let context_cross_direction = crossing_direction(
            context_fast[i - 1] - context_slow[i - 1],
            context_fast[i] - context_slow[i],
        );
        let previous_spread = trigger_fast[i - 1] - trigger_slow[i - 1];
        let spread = trigger_fast[i] - trigger_slow[i];
        let raw_direction = crossing_direction(previous_spread, spread);
        let entry_idx = i + 1;
        let executable =
            entry_idx < bars.close.len() && session_ids[entry_idx].as_ref() == Some(session_id);
        let features = causal_feature_values(
            i,
            raw_direction.unwrap_or(0),
            context_cross_direction.unwrap_or(0),
            bars,
            &session_ids,
            config,
            timezone,
            &indicator_series,
            &atr,
            atr_slow.as_ref(),
            &trigger_fast,
            &trigger_slow,
            &context_fast,
            &context_slow,
            &trigger_spreads,
            &context_spreads,
            &series,
            &previous_trigger_crosses,
            &previous_context_crosses,
        )?;
        let ready = features.is_some();

        if executable {
            if let (Some(raw_direction), Some(features)) = (raw_direction, features) {
                candidates.push(CandidateEvent {
                    event_id: event_index,
                    session_id: session_id.clone(),
                    session_event_index: 0,
                    row_idx: i,
                    timestamp_ns: bars.timestamp_ns[i],
                    raw_direction,
                    decision_price: bars.open[entry_idx],
                    features,
                });
                event_index += 1;
            }
        }

        if executable && ready {
            if let Some(raw_direction) = raw_direction {
                previous_trigger_crosses
                    .entry(session_id.clone())
                    .or_default()
                    .push(PreviousCross {
                        row_idx: i,
                        close: bars.close[i],
                        direction: raw_direction,
                    });
            }
            if let Some(direction) = context_cross_direction {
                previous_context_crosses
                    .entry(session_id.clone())
                    .or_default()
                    .push(PreviousCross {
                        row_idx: i,
                        close: bars.close[i],
                        direction,
                    });
            }
        }
    }

    Ok(label_candidates(
        &mut candidates,
        bars,
        &session_ids,
        config,
    )?)
}

/// Materialize the same causal feature registry used by [`prepare_events`]
/// for every source bar.  This is the dense side of the training artifact:
/// warmup/maintenance rows are retained for GA/RL episode context, while
/// `ready` identifies rows that have a complete finite feature vector.
///
/// No label, interval endpoint, or future price is consulted here.  The
/// optional `cross_direction` is determined from the current and previous
/// trigger spreads only.
pub fn prepare_bar_features(
    bars: &SupervisedBars,
    config: &SupervisedConfig,
) -> Result<Vec<SupervisedBarFeatureRow>> {
    bars.validate()?;
    config.validate()?;

    let timezone = Tz::from_str(&config.session_timezone)?;
    let session_ids = bars
        .timestamp_ns
        .iter()
        .map(|timestamp| session_id(*timestamp, timezone, config))
        .collect::<Result<Vec<_>>>()?;
    let indicator_series = build_indicator_series_map(bars, config);
    let mut series = indicator_series
        .values()
        .cloned()
        .map(|values| IndicatorSeries { values })
        .collect::<Vec<_>>();
    let atr = atr_wilder(&bars.high, &bars.low, &bars.close, config.atr_period);
    series.push(IndicatorSeries {
        values: atr.clone(),
    });
    let atr_slow = config
        .derived_features
        .atr_ratio_slow_period
        .map(|period| atr_wilder(&bars.high, &bars.low, &bars.close, period));
    if let Some(values) = &atr_slow {
        series.push(IndicatorSeries {
            values: values.clone(),
        });
    }
    let trigger_fast = indicator_series
        .get(&indicator_key(config.trigger_kind, config.trigger_fast))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("trigger fast series was not prepared"))?;
    let trigger_slow = indicator_series
        .get(&indicator_key(config.trigger_kind, config.trigger_slow))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("trigger slow series was not prepared"))?;
    let context_fast = indicator_series
        .get(&indicator_key(config.context_kind, config.context_fast))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("context fast series was not prepared"))?;
    let context_slow = indicator_series
        .get(&indicator_key(config.context_kind, config.context_slow))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("context slow series was not prepared"))?;
    let trigger_spreads = trigger_fast
        .iter()
        .zip(&trigger_slow)
        .map(|(fast, slow)| fast - slow)
        .collect::<Vec<_>>();
    let context_spreads = context_fast
        .iter()
        .zip(&context_slow)
        .map(|(fast, slow)| fast - slow)
        .collect::<Vec<_>>();

    let mut previous_trigger_crosses = BTreeMap::<String, Vec<PreviousCross>>::new();
    let mut previous_context_crosses = BTreeMap::<String, Vec<PreviousCross>>::new();
    let mut output = Vec::with_capacity(bars.close.len());
    for i in 0..bars.close.len() {
        let session_id = session_ids[i].clone();
        let cross_direction = if i > 0
            && session_ids[i - 1].as_ref() == session_ids[i].as_ref()
            && session_id.is_some()
        {
            crossing_direction(
                trigger_fast[i - 1] - trigger_slow[i - 1],
                trigger_fast[i] - trigger_slow[i],
            )
        } else {
            None
        };
        let context_cross_direction = if i > 0
            && session_ids[i - 1].as_ref() == session_ids[i].as_ref()
            && session_id.is_some()
        {
            crossing_direction(
                context_fast[i - 1] - context_slow[i - 1],
                context_fast[i] - context_slow[i],
            )
        } else {
            None
        };

        let features = if session_id.is_some() {
            causal_feature_values(
                i,
                cross_direction.unwrap_or(0),
                context_cross_direction.unwrap_or(0),
                bars,
                &session_ids,
                config,
                timezone,
                &indicator_series,
                &atr,
                atr_slow.as_ref(),
                &trigger_fast,
                &trigger_slow,
                &context_fast,
                &context_slow,
                &trigger_spreads,
                &context_spreads,
                &series,
                &previous_trigger_crosses,
                &previous_context_crosses,
            )?
        } else {
            None
        };
        let (features, ready) = match features {
            Some(features) => (features, true),
            None => (BTreeMap::new(), false),
        };

        output.push(SupervisedBarFeatureRow {
            row_idx: i,
            timestamp_ns: bars.timestamp_ns[i],
            session_id: session_id.clone(),
            ready,
            cross_direction,
            features,
        });

        // Match the sparse event builder's history semantics: only a cross
        // that could become an executable event participates in later
        // crossover-history features.  This keeps the dense row at a future
        // event byte-for-byte aligned with the sparse event artifact.
        let eligible_cross =
            i + 1 < bars.close.len() && session_ids[i + 1].as_ref() == session_id.as_ref() && ready;
        if eligible_cross {
            if let Some(session_id) = session_id.as_ref() {
                if let Some(direction) = cross_direction {
                    previous_trigger_crosses
                        .entry(session_id.clone())
                        .or_default()
                        .push(PreviousCross {
                            row_idx: i,
                            close: bars.close[i],
                            direction,
                        });
                }
                if let Some(direction) = context_cross_direction {
                    previous_context_crosses
                        .entry(session_id.clone())
                        .or_default()
                        .push(PreviousCross {
                            row_idx: i,
                            close: bars.close[i],
                            direction,
                        });
                }
            }
        }
    }
    Ok(output)
}

/// Construct one complete causal feature map.  `raw_direction` is the
/// current crossover direction for sparse event rows and zero for ordinary
/// dense bars.
fn causal_feature_values(
    i: usize,
    raw_direction: i8,
    context_cross_direction: i8,
    bars: &SupervisedBars,
    session_ids: &[Option<String>],
    config: &SupervisedConfig,
    timezone: Tz,
    indicator_series: &BTreeMap<String, Vec<f64>>,
    atr: &[f64],
    atr_slow: Option<&Vec<f64>>,
    trigger_fast: &[f64],
    trigger_slow: &[f64],
    context_fast: &[f64],
    context_slow: &[f64],
    trigger_spreads: &[f64],
    context_spreads: &[f64],
    series: &[IndicatorSeries],
    previous_trigger_crosses: &BTreeMap<String, Vec<PreviousCross>>,
    previous_context_crosses: &BTreeMap<String, Vec<PreviousCross>>,
) -> Result<Option<BTreeMap<String, f64>>> {
    if i == 0
        || config
            .derived_features
            .range_lookbacks
            .iter()
            .any(|lookback| *lookback > i + 1)
        || config
            .derived_features
            .regime_persistence_lookbacks
            .iter()
            .any(|lookback| *lookback > i + 1)
        || !causal_values_ready(
            i,
            config,
            atr,
            trigger_fast,
            trigger_slow,
            context_fast,
            context_slow,
            series,
        )
    {
        return Ok(None);
    }

    let spread = trigger_fast[i] - trigger_slow[i];
    let current_session_id = session_id(bars.timestamp_ns[i], timezone, config)?;
    let prior_trigger = current_session_id
        .as_ref()
        .and_then(|value| previous_trigger_crosses.get(value))
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let prior_context = current_session_id
        .as_ref()
        .and_then(|value| previous_context_crosses.get(value))
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut features = BTreeMap::new();
    if config.include_raw_direction {
        features.insert("raw_direction".to_string(), raw_direction as f64);
    }
    for spec in &config.features {
        let values = indicator_series
            .get(&spec.base_name())
            .ok_or_else(|| anyhow::anyhow!("feature series was not prepared"))?;
        let base_name = spec.base_name();
        let value = values[i];
        if spec.include_value {
            let feature_name =
                if matches!(spec.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch) {
                    format!("{base_name}_centered")
                } else if spec.normalize_value_by_atr {
                    format!("{base_name}_atr")
                } else {
                    base_name.clone()
                };
            let feature_value =
                if matches!(spec.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch) {
                    (value - 50.0) / 50.0
                } else if spec.normalize_value_by_atr {
                    value / atr[i]
                } else {
                    value
                };
            features.insert(feature_name, feature_value);
        }
        if spec.include_delta {
            for lookback in &spec.lookbacks {
                let raw_delta = values[i] - values[i - lookback];
                let feature_name =
                    if matches!(spec.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch) {
                        format!("{}_delta_{}_scaled", base_name, lookback)
                    } else {
                        format!(
                            "{}_delta_{}{}",
                            base_name,
                            lookback,
                            if spec.normalize_by_atr { "_atr" } else { "" }
                        )
                    };
                let delta = if matches!(spec.indicator, IndicatorKind::Rsi | IndicatorKind::Stoch) {
                    raw_delta / 100.0
                } else if spec.normalize_by_atr {
                    raw_delta / atr[i]
                } else {
                    raw_delta
                };
                features.insert(feature_name, delta);
            }
        }
    }
    features.insert("trigger_spread_atr".to_string(), spread / atr[i]);
    features.insert(
        "context_spread_atr".to_string(),
        (context_fast[i] - context_slow[i]) / atr[i],
    );
    let slope = config.slope_lookback as f64;
    features.insert(
        "trigger_fast_slope_atr".to_string(),
        (trigger_fast[i] - trigger_fast[i - config.slope_lookback]) / (slope * atr[i]),
    );
    features.insert(
        "trigger_slow_slope_atr".to_string(),
        (trigger_slow[i] - trigger_slow[i - config.slope_lookback]) / (slope * atr[i]),
    );
    features.insert(
        "context_fast_slope_atr".to_string(),
        (context_fast[i] - context_fast[i - config.slope_lookback]) / (slope * atr[i]),
    );
    features.insert(
        "context_slow_slope_atr".to_string(),
        (context_slow[i] - context_slow[i - config.slope_lookback]) / (slope * atr[i]),
    );
    features.insert(
        "price_distance_trigger_slow_atr".to_string(),
        (bars.close[i] - trigger_slow[i]) / atr[i],
    );
    if config.derived_features.include_time_cycle {
        let (tod_sin, tod_cos) = time_of_day_cycle(bars.timestamp_ns[i], timezone)?;
        features.insert("tod_sin".to_string(), tod_sin);
        features.insert("tod_cos".to_string(), tod_cos);
    }
    if !config.derived_features.cross_lookbacks.is_empty() {
        let previous = prior_trigger.last().copied();
        features.insert(
            "has_previous_cross".to_string(),
            if previous.is_some() { 1.0 } else { 0.0 },
        );
        features.insert(
            "bars_since_last_cross".to_string(),
            previous
                .map(|cross| (i - cross.row_idx) as f64)
                .unwrap_or(0.0),
        );
        features.insert(
            "displacement_since_last_cross_atr".to_string(),
            previous
                .map(|cross| (bars.close[i] - cross.close) / atr[i])
                .unwrap_or(0.0),
        );
        for lookback in &config.derived_features.cross_lookbacks {
            let count = prior_trigger
                .iter()
                .filter(|cross| i.saturating_sub(cross.row_idx) <= *lookback)
                .count();
            features.insert(format!("crosses_last_{lookback}_bars"), count as f64);
        }
    }
    if !config.derived_features.context_cross_lookbacks.is_empty() {
        let previous = prior_context.last().copied();
        features.insert(
            "context_cross_direction".to_string(),
            context_cross_direction as f64,
        );
        features.insert(
            "context_has_previous_cross".to_string(),
            if previous.is_some() { 1.0 } else { 0.0 },
        );
        features.insert(
            "context_bars_since_last_cross".to_string(),
            previous
                .map(|cross| (i - cross.row_idx) as f64)
                .unwrap_or(0.0),
        );
        features.insert(
            "context_last_cross_direction".to_string(),
            previous.map(|cross| cross.direction as f64).unwrap_or(0.0),
        );
        features.insert(
            "context_displacement_since_last_cross_atr".to_string(),
            previous
                .map(|cross| (bars.close[i] - cross.close) / atr[i])
                .unwrap_or(0.0),
        );
        for lookback in &config.derived_features.context_cross_lookbacks {
            let count = prior_context
                .iter()
                .filter(|cross| i.saturating_sub(cross.row_idx) <= *lookback)
                .count();
            features.insert(
                format!("context_crosses_last_{lookback}_bars"),
                count as f64,
            );
        }
    }
    for lookback in &config.derived_features.range_lookbacks {
        let start = i + 1 - lookback;
        let rolling_high = bars.high[start..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let rolling_low = bars.low[start..=i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let width = rolling_high - rolling_low;
        features.insert(format!("range_{lookback}_atr"), width / atr[i]);
        features.insert(
            format!("position_in_range_{lookback}"),
            if width > f64::EPSILON {
                ((bars.close[i] - rolling_low) / width).clamp(0.0, 1.0)
            } else {
                0.5
            },
        );
    }
    if !config
        .derived_features
        .regime_persistence_lookbacks
        .is_empty()
    {
        let current_session_id = current_session_id.as_ref();
        let trigger_reference = finite_sign(trigger_spreads[i.saturating_sub(1)]);
        let context_reference = finite_sign(context_spreads[i.saturating_sub(1)]);
        features.insert(
            "trigger_pre_run_bars".to_string(),
            consecutive_sign_run(&trigger_spreads, i, current_session_id, session_ids) as f64,
        );
        features.insert(
            "context_pre_run_bars".to_string(),
            consecutive_sign_run(&context_spreads, i, current_session_id, session_ids) as f64,
        );
        let direction = if raw_direction != 0 {
            raw_direction
        } else {
            trigger_reference
        } as f64;
        for lookback in &config.derived_features.regime_persistence_lookbacks {
            let start = session_window_start(i, *lookback, current_session_id, session_ids);
            let end = i;
            features.insert(
                format!("trigger_pre_persistence_{lookback}"),
                sign_persistence(&trigger_spreads, start, end, trigger_reference),
            );
            features.insert(
                format!("trigger_pre_flip_rate_{lookback}"),
                sign_flip_rate(&trigger_spreads, start, end),
            );
            features.insert(
                format!("trigger_pre_return_aligned_{lookback}_atr"),
                if start < end {
                    direction * (bars.close[end - 1] - bars.close[start]) / atr[i]
                } else {
                    0.0
                },
            );
            features.insert(
                format!("context_pre_persistence_{lookback}"),
                sign_persistence(&context_spreads, start, end, context_reference),
            );
            features.insert(
                format!("context_pre_flip_rate_{lookback}"),
                sign_flip_rate(&context_spreads, start, end),
            );
            features.insert(
                format!("trigger_context_pre_agreement_{lookback}"),
                sign_agreement(&trigger_spreads, &context_spreads, start, end),
            );
        }
    }
    if let (Some(period), Some(slow_atr)) =
        (config.derived_features.atr_ratio_slow_period, atr_slow)
    {
        features.insert(
            format!("atr_{}_over_atr_{}", config.atr_period, period),
            atr[i] / slow_atr[i],
        );
    }
    if config.derived_features.include_direction_interactions {
        let direction = raw_direction as f64;
        for (interaction, source) in [
            ("direction_x_trigger_spread_atr", "trigger_spread_atr"),
            ("direction_x_context_spread_atr", "context_spread_atr"),
            (
                "direction_x_trigger_fast_slope_atr",
                "trigger_fast_slope_atr",
            ),
            (
                "direction_x_trigger_slow_slope_atr",
                "trigger_slow_slope_atr",
            ),
            (
                "direction_x_context_fast_slope_atr",
                "context_fast_slope_atr",
            ),
            (
                "direction_x_context_slow_slope_atr",
                "context_slow_slope_atr",
            ),
            (
                "direction_x_price_distance_trigger_slow_atr",
                "price_distance_trigger_slow_atr",
            ),
            ("direction_x_adx_14", "adx_14"),
            ("direction_x_er_14", "er_14"),
            ("direction_x_rvol_20", "rvol_20"),
            ("direction_x_cmf_20", "cmf_20"),
            ("direction_x_vwap_20", "vwap_20"),
            ("direction_x_di_spread_14", "di_spread_14"),
            ("direction_x_chop_30", "chop_30"),
            ("direction_x_vhf_30", "vhf_30"),
        ] {
            let value = features.get(source).copied().unwrap_or(0.0);
            features.insert(interaction.to_string(), direction * value);
        }
    }
    if features.len() != config.feature_names().len()
        || features.values().any(|value| !value.is_finite())
    {
        return Ok(None);
    }
    Ok(Some(features))
}

fn finite_sign(value: f64) -> i8 {
    if value.is_finite() && value > 0.0 {
        1
    } else if value.is_finite() && value < 0.0 {
        -1
    } else {
        0
    }
}

fn session_window_start(
    i: usize,
    lookback: usize,
    current_session_id: Option<&String>,
    session_ids: &[Option<String>],
) -> usize {
    let mut start = i.saturating_sub(lookback);
    while start < i && session_ids[start].as_ref() != current_session_id {
        start += 1;
    }
    start
}

fn consecutive_sign_run(
    values: &[f64],
    i: usize,
    current_session_id: Option<&String>,
    session_ids: &[Option<String>],
) -> usize {
    if i == 0 || session_ids[i].as_ref() != current_session_id {
        return 0;
    }
    let reference = finite_sign(values[i - 1]);
    if reference == 0 {
        return 0;
    }
    let mut start = i;
    while start > 0 {
        let row = start - 1;
        if session_ids[row].as_ref() != current_session_id || finite_sign(values[row]) != reference
        {
            break;
        }
        start = row;
    }
    i - start
}

fn sign_persistence(values: &[f64], start: usize, end: usize, reference: i8) -> f64 {
    if reference == 0 || start >= end {
        return 0.0;
    }
    let mut total = 0usize;
    let mut matching = 0usize;
    for value in &values[start..end] {
        let sign = finite_sign(*value);
        if sign == 0 {
            continue;
        }
        total += 1;
        if sign == reference {
            matching += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        matching as f64 / total as f64
    }
}

fn sign_flip_rate(values: &[f64], start: usize, end: usize) -> f64 {
    if start >= end {
        return 0.0;
    }
    let mut previous = 0i8;
    let mut transitions = 0usize;
    let mut flips = 0usize;
    for value in &values[start..end] {
        let sign = finite_sign(*value);
        if sign == 0 {
            continue;
        }
        if previous != 0 {
            transitions += 1;
            if sign != previous {
                flips += 1;
            }
        }
        previous = sign;
    }
    if transitions == 0 {
        0.0
    } else {
        flips as f64 / transitions as f64
    }
}

fn sign_agreement(left: &[f64], right: &[f64], start: usize, end: usize) -> f64 {
    if start >= end {
        return 0.0;
    }
    let mut total = 0usize;
    let mut matching = 0usize;
    for (left_value, right_value) in left[start..end].iter().zip(&right[start..end]) {
        let left_sign = finite_sign(*left_value);
        let right_sign = finite_sign(*right_value);
        if left_sign == 0 || right_sign == 0 {
            continue;
        }
        total += 1;
        if left_sign == right_sign {
            matching += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        matching as f64 / total as f64
    }
}

fn time_of_day_cycle(timestamp_ns: i64, timezone: Tz) -> Result<(f64, f64)> {
    let utc = DateTime::<Utc>::from_timestamp(
        timestamp_ns.div_euclid(1_000_000_000),
        timestamp_ns.rem_euclid(1_000_000_000) as u32,
    )
    .ok_or_else(|| anyhow::anyhow!("timestamp {timestamp_ns} is outside chrono range"))?;
    let local = utc.with_timezone(&timezone);
    let seconds = local.time().num_seconds_from_midnight() as f64
        + f64::from(local.nanosecond()) / 1_000_000_000.0;
    let angle = std::f64::consts::TAU * seconds / 86_400.0;
    Ok((angle.sin(), angle.cos()))
}

fn crossing_direction(previous_spread: f64, spread: f64) -> Option<i8> {
    if !previous_spread.is_finite() || !spread.is_finite() {
        return None;
    }
    if previous_spread <= 0.0 && spread > 0.0 {
        Some(1)
    } else if previous_spread >= 0.0 && spread < 0.0 {
        Some(-1)
    } else {
        None
    }
}

fn causal_values_ready(
    i: usize,
    config: &SupervisedConfig,
    atr: &[f64],
    trigger_fast: &[f64],
    trigger_slow: &[f64],
    context_fast: &[f64],
    context_slow: &[f64],
    series: &[IndicatorSeries],
) -> bool {
    if i < config.slope_lookback
        || config
            .features
            .iter()
            .flat_map(|spec| spec.lookbacks.iter())
            .any(|lookback| *lookback > i)
        || !atr[i].is_finite()
        || atr[i] <= f64::EPSILON
        || [
            trigger_fast[i],
            trigger_fast[i - 1],
            trigger_slow[i],
            trigger_slow[i - 1],
            context_fast[i],
            context_slow[i],
        ]
        .iter()
        .all(|value| value.is_finite())
            == false
    {
        return false;
    }
    if series.iter().any(|value| {
        !value.values[i].is_finite()
            || value.values[..=i]
                .iter()
                .rev()
                .take(config.slope_lookback + 1)
                .any(|sample| !sample.is_finite())
    }) {
        return false;
    }
    true
}

fn label_candidates(
    candidates: &mut [CandidateEvent],
    bars: &SupervisedBars,
    session_ids: &[Option<String>],
    config: &SupervisedConfig,
) -> Result<Vec<SupervisedEvent>> {
    let mut by_session: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, event) in candidates.iter().enumerate() {
        by_session
            .entry(event.session_id.clone())
            .or_default()
            .push(index);
    }

    let mut output = Vec::with_capacity(candidates.len());
    for (session_id, indices) in by_session {
        if indices.is_empty() {
            continue;
        }
        for (session_event_index, index) in indices.iter().enumerate() {
            candidates[*index].session_event_index = session_event_index;
        }
        let session_close_idx = bars
            .timestamp_ns
            .iter()
            .enumerate()
            .filter(|(row, _)| session_ids[*row].as_deref() == Some(session_id.as_str()))
            .map(|(row, _)| row)
            .max()
            .ok_or_else(|| anyhow::anyhow!("session {session_id} has no bars"))?;

        let n = indices.len();
        let mut values = vec![[0.0_f64; 3]; n + 1];
        for position in [-1_i8, 0, 1] {
            values[n][position_index(position)] = transition_cost(position, 0, config);
        }
        let mut action_values = vec![[[0.0_f64; 3]; 3]; n];
        let mut interval_end_price = vec![0.0; n];
        let mut interval_end_row_idx = vec![session_close_idx; n];
        let mut terminal = vec![false; n];

        for j in (0..n).rev() {
            let event = &candidates[indices[j]];
            let (end_price, end_row, is_terminal) = if j + 1 < n {
                let next = &candidates[indices[j + 1]];
                (next.decision_price, next.row_idx + 1, false)
            } else {
                (bars.close[session_close_idx], session_close_idx, true)
            };
            interval_end_price[j] = end_price;
            interval_end_row_idx[j] = end_row;
            terminal[j] = is_terminal;
            let delta = end_price - event.decision_price;
            for position in [-1_i8, 0, 1] {
                let pidx = position_index(position);
                let actions = [1_i8, 0, -1_i8]; // normal, skip, invert
                for (action_idx, action) in actions.into_iter().enumerate() {
                    let next_position = target_position(position, event.raw_direction, action);
                    let immediate = next_position as f64 * delta * config.contract_multiplier
                        + transition_cost(position, next_position, config);
                    action_values[j][pidx][action_idx] =
                        immediate + values[j + 1][position_index(next_position)];
                }
                values[j][pidx] = permitted_action_indices(&config.label_mode)
                    .iter()
                    .map(|action_idx| action_values[j][pidx][*action_idx])
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }

        let mut position = 0_i8;
        for (j, index) in indices.into_iter().enumerate() {
            let candidate = &candidates[index];
            let pidx = position_index(position);
            let action_values_for_position = action_values[j][pidx];
            let action_idx = best_action_index(action_values_for_position, config.label_mode);
            let action = [1_i8, 0, -1][action_idx];
            let next_position = target_position(position, candidate.raw_direction, action);
            let oracle_value = action_values_for_position[action_idx];
            let (label_action, label_name) = match action {
                1 => (1, "normal"),
                -1 => (-1, "invert"),
                _ => (0, "skip"),
            };
            output.push(SupervisedEvent {
                event_id: candidate.event_id,
                session_id: candidate.session_id.clone(),
                session_event_index: candidate.session_event_index,
                row_idx: candidate.row_idx,
                timestamp_ns: candidate.timestamp_ns,
                raw_direction: candidate.raw_direction,
                decision_price: candidate.decision_price,
                interval_end_price: interval_end_price[j],
                interval_end_row_idx: interval_end_row_idx[j],
                terminal_event: terminal[j],
                features: candidate.features.clone(),
                action_value_normal: action_values_for_position[0],
                action_value_skip: action_values_for_position[1],
                action_value_invert: action_values_for_position[2],
                label_action,
                label_name: label_name.to_string(),
                oracle_position_before: position,
                oracle_position_after: next_position,
                oracle_value,
            });
            position = next_position;
        }
    }
    output.sort_by_key(|event| event.event_id);
    Ok(output)
}

fn permitted_action_indices(mode: &LabelMode) -> &'static [usize] {
    match mode {
        LabelMode::NormalSkipInvert => &[0, 1, 2],
        LabelMode::NormalInvert => &[0, 2],
    }
}

fn best_action_index(values: [f64; 3], mode: LabelMode) -> usize {
    // Deterministic tie preference: skip, then normal, then invert for the
    // legacy three-way mode. In two-way mode, prefer normal, then invert.
    let order = match mode {
        LabelMode::NormalSkipInvert => &[1usize, 0, 2][..],
        LabelMode::NormalInvert => &[0usize, 2][..],
    };
    let mut best = order[0];
    for idx in order.iter().copied().skip(1) {
        if values[idx] > values[best] + 1e-9 {
            best = idx;
        }
    }
    best
}

fn position_index(position: i8) -> usize {
    match position {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 1,
    }
}

fn target_position(current: i8, raw_direction: i8, action: i8) -> i8 {
    match action {
        1 => raw_direction.clamp(-1, 1),
        -1 => (-raw_direction).clamp(-1, 1),
        _ => current.clamp(-1, 1),
    }
}

fn transition_cost(from: i8, to: i8, config: &SupervisedConfig) -> f64 {
    -((from - to).unsigned_abs() as f64 * config.round_trip_cost / 2.0)
}

fn session_id(
    timestamp_ns: i64,
    timezone: Tz,
    config: &SupervisedConfig,
) -> Result<Option<String>> {
    let utc = DateTime::<Utc>::from_timestamp(
        timestamp_ns.div_euclid(1_000_000_000),
        timestamp_ns.rem_euclid(1_000_000_000) as u32,
    )
    .ok_or_else(|| anyhow::anyhow!("timestamp {timestamp_ns} is outside chrono range"))?;
    let local = utc.with_timezone(&timezone);
    let time = local.time();
    let start = NaiveTime::from_hms_opt(config.session_start_hour, 0, 0).unwrap();
    let end = NaiveTime::from_hms_opt(config.session_end_hour, 0, 0).unwrap();
    if time >= start {
        return Ok(Some(local.date_naive().to_string()));
    }
    if time <= end {
        let date = local
            .date_naive()
            .pred_opt()
            .ok_or_else(|| anyhow::anyhow!("timestamp date underflow"))?;
        return Ok(Some(date.to_string()));
    }
    Ok(None)
}

fn indicator_key(kind: IndicatorKind, period: usize) -> String {
    format!("{kind}_{period}")
}

fn build_indicator_series_map(
    bars: &SupervisedBars,
    config: &SupervisedConfig,
) -> BTreeMap<String, Vec<f64>> {
    let mut unique = BTreeMap::<String, (IndicatorKind, usize)>::new();
    unique.insert(
        indicator_key(config.trigger_kind, config.trigger_fast),
        (config.trigger_kind, config.trigger_fast),
    );
    unique.insert(
        indicator_key(config.trigger_kind, config.trigger_slow),
        (config.trigger_kind, config.trigger_slow),
    );
    unique.insert(
        indicator_key(config.context_kind, config.context_fast),
        (config.context_kind, config.context_fast),
    );
    unique.insert(
        indicator_key(config.context_kind, config.context_slow),
        (config.context_kind, config.context_slow),
    );
    for spec in &config.features {
        unique.insert(spec.base_name(), (spec.indicator, spec.period));
    }
    unique
        .into_iter()
        .map(|(name, (kind, period))| {
            (
                name,
                series_for_kind(
                    kind,
                    &bars.close,
                    &bars.high,
                    &bars.low,
                    &bars.volume,
                    period,
                ),
            )
        })
        .collect()
}

fn series_for_kind(
    kind: IndicatorKind,
    close: &[f64],
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    period: usize,
) -> Vec<f64> {
    match kind {
        IndicatorKind::Sma => crate::features::sma(close, period),
        IndicatorKind::Ema => crate::features::ema(close, period),
        IndicatorKind::Hma => crate::features::hma(close, period),
        IndicatorKind::Kama => crate::features::kama(close, period, 2, 30),
        IndicatorKind::Alma => crate::features::alma(close, period, 0.85, 6.0),
        IndicatorKind::Atr => atr_wilder(high, low, close, period),
        IndicatorKind::Adx => adx(high, low, close, period),
        IndicatorKind::Rsi => rsi_wilder(close, period),
        IndicatorKind::Stoch => stochastic_k(high, low, close, period),
        IndicatorKind::Rvol => crate::features::rvol(volume, period),
        IndicatorKind::Cmf => crate::features::cmf(high, low, close, volume, period),
        IndicatorKind::Vwap => crate::features::rolling_vwap_distance(close, volume, period),
        IndicatorKind::DiSpread => di_components(high, low, close, period).2,
        IndicatorKind::DiPlus => di_components(high, low, close, period).0,
        IndicatorKind::DiMinus => di_components(high, low, close, period).1,
        IndicatorKind::Chop => choppiness_index(high, low, close, period),
        IndicatorKind::Vhf => vertical_horizontal_filter(close, period),
        IndicatorKind::Er => efficiency_ratio_series(close, period),
        IndicatorKind::Close => close.to_vec(),
    }
}

fn true_ranges(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut tr = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if i == 0 {
            tr[i] = high[i] - low[i];
        } else {
            tr[i] = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
        }
    }
    tr
}

fn atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let tr = true_ranges(high, low, close);
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 || close.len() < period {
        return out;
    }
    let mut current = tr[..period].iter().sum::<f64>() / period as f64;
    out[period - 1] = current;
    for i in period..close.len() {
        current = (current * (period as f64 - 1.0) + tr[i]) / period as f64;
        out[i] = current;
    }
    out
}

/// A rolling ADX implementation.  It uses only completed bars through `i`;
/// exact Wilder-vs-rolling smoothing is part of the feature schema and is
/// therefore stable/reproducible for a prepared dataset.
fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if period == 0 || len < period * 2 {
        return out;
    }
    let tr = true_ranges(high, low, close);
    let mut dx = vec![f64::NAN; len];
    for i in period..len {
        let mut tr_sum = 0.0;
        let mut plus_sum = 0.0;
        let mut minus_sum = 0.0;
        for j in (i + 1 - period)..=i {
            tr_sum += tr[j];
            if j == 0 {
                continue;
            }
            let up = high[j] - high[j - 1];
            let down = low[j - 1] - low[j];
            if up > down && up > 0.0 {
                plus_sum += up;
            } else if down > up && down > 0.0 {
                minus_sum += down;
            }
        }
        if tr_sum <= f64::EPSILON {
            continue;
        }
        let plus_di = 100.0 * plus_sum / tr_sum;
        let minus_di = 100.0 * minus_sum / tr_sum;
        let denom = plus_di + minus_di;
        if denom > f64::EPSILON {
            dx[i] = 100.0 * (plus_di - minus_di).abs() / denom;
        }
    }
    for i in (period * 2 - 1)..len {
        let start = i + 1 - period;
        let values = &dx[start..=i];
        if values.iter().all(|value| value.is_finite()) {
            out[i] = values.iter().sum::<f64>() / period as f64;
        }
    }
    out
}

/// Directional movement components over a rolling window.  The first two
/// outputs are the conventional +DI and -DI levels; the third is their
/// signed spread.  All three are calculated only from bars through the
/// current index.
fn di_components(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut plus = vec![f64::NAN; len];
    let mut minus = vec![f64::NAN; len];
    let mut spread = vec![f64::NAN; len];
    if period == 0 || len < period {
        return (plus, minus, spread);
    }
    let tr = true_ranges(high, low, close);
    for i in period..len {
        let start = i + 1 - period;
        let mut tr_sum = 0.0;
        let mut plus_sum = 0.0;
        let mut minus_sum = 0.0;
        for j in start..=i {
            tr_sum += tr[j];
            if j == 0 {
                continue;
            }
            let up = high[j] - high[j - 1];
            let down = low[j - 1] - low[j];
            if up > down && up > 0.0 {
                plus_sum += up;
            } else if down > up && down > 0.0 {
                minus_sum += down;
            }
        }
        let di_total = plus_sum + minus_sum;
        if tr_sum > f64::EPSILON {
            plus[i] = 100.0 * plus_sum / tr_sum;
            minus[i] = 100.0 * minus_sum / tr_sum;
            if di_total > f64::EPSILON {
                spread[i] = 100.0 * (plus_sum - minus_sum) / di_total;
            }
        }
    }
    (plus, minus, spread)
}

/// Choppiness Index.  Larger values describe a longer, more two-sided path
/// relative to its net high/low displacement.  The implementation uses only
/// the current and preceding bars and emits NaN during warmup.
fn choppiness_index(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if period < 2 || len < period {
        return out;
    }
    let tr = true_ranges(high, low, close);
    let log_period = (period as f64).log10();
    for i in period - 1..len {
        let start = i + 1 - period;
        let tr_sum = tr[start..=i].iter().sum::<f64>();
        let highest = high[start..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest = low[start..=i].iter().copied().fold(f64::INFINITY, f64::min);
        let range = highest - lowest;
        if tr_sum > f64::EPSILON && range > f64::EPSILON && log_period > 0.0 {
            out[i] = (100.0 * (tr_sum / range).log10() / log_period).clamp(0.0, 100.0);
        }
    }
    out
}

/// Vertical Horizontal Filter.  High values indicate that net displacement
/// dominates the total absolute close-to-close path.  This is a compact,
/// causal alternative to adding another moving-average family.
fn vertical_horizontal_filter(close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if period == 0 || len <= period {
        return out;
    }
    for i in period..len {
        let start = i + 1 - period;
        let highest = close[start..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest = close[start..=i]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let path = (start + 1..=i)
            .map(|j| (close[j] - close[j - 1]).abs())
            .sum::<f64>();
        if path > f64::EPSILON {
            out[i] = (highest - lowest) / path;
        }
    }
    out
}

fn efficiency_ratio_series(close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 {
        return out;
    }
    for i in period..close.len() {
        let change = (close[i] - close[i - period]).abs();
        let volatility = (i - period + 1..=i)
            .map(|j| (close[j] - close[j - 1]).abs())
            .sum::<f64>();
        out[i] = if volatility > f64::EPSILON {
            (change / volatility).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    out
}

/// Standard Wilder RSI.  The first finite value is available after `period`
/// completed close-to-close changes.  It is kept in the native 0..100 scale
/// internally; feature emission centers/scales it without using ATR because
/// RSI is already dimensionless and bounded.
fn rsi_wilder(close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 || close.len() <= period {
        return out;
    }

    let mut average_gain = 0.0;
    let mut average_loss = 0.0;
    for i in 1..=period {
        let change = close[i] - close[i - 1];
        if change >= 0.0 {
            average_gain += change;
        } else {
            average_loss += -change;
        }
    }
    average_gain /= period as f64;
    average_loss /= period as f64;

    let value = if average_loss <= f64::EPSILON {
        if average_gain <= f64::EPSILON {
            50.0
        } else {
            100.0
        }
    } else {
        100.0 - 100.0 / (1.0 + average_gain / average_loss)
    };
    out[period] = value;

    for i in (period + 1)..close.len() {
        let change = close[i] - close[i - 1];
        let gain = change.max(0.0);
        let loss = (-change).max(0.0);
        average_gain = (average_gain * (period as f64 - 1.0) + gain) / period as f64;
        average_loss = (average_loss * (period as f64 - 1.0) + loss) / period as f64;
        out[i] = if average_loss <= f64::EPSILON {
            if average_gain <= f64::EPSILON {
                50.0
            } else {
                100.0
            }
        } else {
            100.0 - 100.0 / (1.0 + average_gain / average_loss)
        };
    }
    out
}

/// Causal fast stochastic oscillator (%K).  The current bar is the newest
/// sample in the rolling high/low window, so no future bar contributes to the
/// value.  Flat windows use 50 as the neutral midpoint.
fn stochastic_k(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 || high.len() != close.len() || low.len() != close.len() {
        return out;
    }
    for i in period.saturating_sub(1)..close.len() {
        let start = i + 1 - period;
        let highest = high[start..=i]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest = low[start..=i].iter().copied().fold(f64::INFINITY, f64::min);
        let width = highest - lowest;
        out[i] = if width > f64::EPSILON {
            (100.0 * (close[i] - lowest) / width).clamp(0.0, 100.0)
        } else {
            50.0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(closes: &[f64]) -> SupervisedBars {
        SupervisedBars {
            open: closes.to_vec(),
            high: closes.iter().map(|value| value + 0.5).collect(),
            low: closes.iter().map(|value| value - 0.5).collect(),
            close: closes.to_vec(),
            volume: vec![10.0; closes.len()],
            timestamp_ns: (0..closes.len())
                .map(|index| 1_720_000_000_000_000_000_i64 + index as i64 * 60_000_000_000)
                .collect(),
        }
    }

    fn bars_at(closes: &[f64], start_ns: i64) -> SupervisedBars {
        let mut result = bars(closes);
        result.timestamp_ns = (0..closes.len())
            .map(|index| start_ns + index as i64 * 60_000_000_000)
            .collect();
        result
    }

    fn small_config() -> SupervisedConfig {
        SupervisedConfig {
            trigger_fast: 2,
            trigger_slow: 4,
            context_fast: 3,
            context_slow: 5,
            atr_period: 2,
            slope_lookback: 1,
            features: vec![FeatureSpec {
                indicator: IndicatorKind::Ema,
                period: 2,
                lookbacks: vec![1],
                include_value: true,
                include_delta: true,
                normalize_by_atr: true,
                normalize_value_by_atr: false,
            }],
            ..SupervisedConfig::default()
        }
    }

    #[test]
    fn rsi_is_causal_bounded_and_centered_without_atr_normalization() {
        let values = rsi_wilder(&[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 3.0], 3);
        assert!(values[..3].iter().all(|value| value.is_nan()));
        assert_eq!(values[3], 100.0);
        assert!(values[5] < 100.0);
        assert!(values[6] > 0.0 && values[6] < 100.0);

        let spec = FeatureSpec {
            indicator: IndicatorKind::Rsi,
            period: 14,
            lookbacks: vec![1, 3, 5],
            include_value: true,
            include_delta: true,
            normalize_by_atr: false,
            normalize_value_by_atr: false,
        };
        assert_eq!(
            spec.names(),
            vec![
                "rsi_14_centered",
                "rsi_14_delta_1_scaled",
                "rsi_14_delta_3_scaled",
                "rsi_14_delta_5_scaled"
            ]
        );
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn stochastic_k_is_causal_and_uses_bounded_feature_names() {
        let high = [10.0, 11.0, 12.0, 13.0, 14.0];
        let low = [8.0, 9.0, 10.0, 11.0, 12.0];
        let close = [9.0, 10.0, 11.0, 12.0, 13.5];
        let values = stochastic_k(&high, &low, &close, 3);
        assert!(values[..2].iter().all(|value| value.is_nan()));
        assert!(values[2] > 0.0 && values[2] < 100.0);
        assert_eq!(values[4], 87.5);

        let spec = FeatureSpec {
            indicator: IndicatorKind::Stoch,
            period: 14,
            lookbacks: vec![5, 10],
            include_value: true,
            include_delta: true,
            normalize_by_atr: false,
            normalize_value_by_atr: false,
        };
        assert_eq!(
            spec.names(),
            vec![
                "stoch_14_centered",
                "stoch_14_delta_5_scaled",
                "stoch_14_delta_10_scaled"
            ]
        );
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn dynamic_labeler_emits_only_causal_features_and_valid_actions() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let mut config = small_config();
        config.session_start_hour = 18;
        config.session_end_hour = 17;
        let result = prepare_events(&bars(&closes), &config).unwrap();
        for event in &result {
            assert!([-1, 0, 1].contains(&event.label_action));
            assert!(event.features.values().all(|value| value.is_finite()));
            assert!(event.action_value_normal.is_finite());
            assert!(event.action_value_skip.is_finite());
            assert!(event.action_value_invert.is_finite());
        }
    }

    #[test]
    fn normal_invert_mode_never_selects_skip() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let mut config = small_config();
        config.label_mode = LabelMode::NormalInvert;
        let result = prepare_events(&bars(&closes), &config).unwrap();
        assert!(!result.is_empty());
        assert!(result.iter().all(|event| event.label_action != 0));
        assert!(
            result
                .iter()
                .all(|event| matches!(event.label_name.as_str(), "normal" | "invert"))
        );
        assert_eq!(config.label_mode.class_names(), &["normal", "invert"]);
    }

    #[test]
    fn normalized_values_and_atr_ratio_are_emitted_causally() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let mut config = small_config();
        config.features[0].normalize_value_by_atr = true;
        config.derived_features.atr_ratio_slow_period = Some(5);
        let result = prepare_events(&bars(&closes), &config).unwrap();
        assert!(!result.is_empty());
        assert!(config.feature_names().contains(&"ema_2_atr".to_string()));
        assert!(
            config
                .feature_names()
                .contains(&"atr_2_over_atr_5".to_string())
        );
        for event in result {
            assert!(event.features["ema_2_atr"].is_finite());
            assert!(event.features["atr_2_over_atr_5"].is_finite());
        }
    }

    #[test]
    fn dimensionless_mode_rejects_raw_price_scale_features() {
        let mut config = SupervisedConfig::default();
        config.require_dimensionless_features = true;
        config.features = vec![FeatureSpec {
            indicator: IndicatorKind::Kama,
            period: 30,
            lookbacks: vec![1],
            include_value: true,
            include_delta: true,
            normalize_by_atr: true,
            normalize_value_by_atr: false,
        }];
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("rejects raw kama_30 value"));

        config.features[0].normalize_value_by_atr = true;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn standard_contract_multiplier_uses_point_value_not_margin() {
        assert_eq!(standard_contract_multiplier("GC", "GCZ6"), Some(100.0));
        assert_eq!(standard_contract_multiplier("ES", "ESU6"), Some(50.0));
        assert_eq!(standard_contract_multiplier("NQ", "NQZ6"), Some(20.0));
        assert_eq!(standard_contract_multiplier("MGC", "MGCZ6"), Some(10.0));
        assert_eq!(standard_contract_multiplier("MES", "MESU6"), Some(5.0));
        assert_eq!(standard_contract_multiplier("MNQ", "MNQZ6"), Some(2.0));
        assert_eq!(standard_contract_multiplier("UNKNOWN", "UNKNOWN"), None);
    }

    #[test]
    fn standard_tick_size_matches_common_contracts() {
        assert_eq!(standard_tick_size("GC", "GCZ6"), Some(0.1));
        assert_eq!(standard_tick_size("ES", "ESU6"), Some(0.25));
        assert_eq!(standard_tick_size("NQ", "NQZ6"), Some(0.25));
        assert_eq!(standard_tick_size("UNKNOWN", "UNKNOWN"), None);
    }

    #[test]
    fn derived_context_features_are_causal_and_time_is_cyclic() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let mut config = small_config();
        config.derived_features = DerivedFeatureConfig {
            include_time_cycle: true,
            cross_lookbacks: vec![5],
            context_cross_lookbacks: vec![5, 10],
            range_lookbacks: vec![3],
            atr_ratio_slow_period: None,
            include_direction_interactions: false,
            regime_persistence_lookbacks: vec![3, 5],
        };
        let result = prepare_events(&bars(&closes), &config).unwrap();
        assert!(!result.is_empty());
        for event in &result {
            let sin = event.features["tod_sin"];
            let cos = event.features["tod_cos"];
            assert!((sin * sin + cos * cos - 1.0).abs() < 1e-9);
            assert!(event.features["bars_since_last_cross"] >= 0.0);
            assert!(event.features["crosses_last_5_bars"] >= 0.0);
            assert!(event.features["context_bars_since_last_cross"] >= 0.0);
            assert!(event.features["context_crosses_last_5_bars"] >= 0.0);
            assert!((0.0..=1.0).contains(&event.features["position_in_range_3"]));
            assert!(event.features["trigger_pre_run_bars"] >= 0.0);
            assert!(event.features["context_pre_run_bars"] >= 0.0);
            assert!((0.0..=1.0).contains(&event.features["trigger_pre_persistence_3"]));
            assert!((0.0..=1.0).contains(&event.features["context_pre_persistence_5"]));
            assert!((0.0..=1.0).contains(&event.features["trigger_context_pre_agreement_3"]));
        }
        assert!(
            result
                .iter()
                .any(|event| event.features["has_previous_cross"] > 0.0)
        );
    }

    #[test]
    fn volume_regime_features_and_direction_interactions_are_emitted() {
        let closes = (0..128)
            .map(|index| {
                let trend = (index as f64 * 0.17).sin() * 2.0 + index as f64 * 0.03;
                100.0 + trend
            })
            .collect::<Vec<_>>();
        let mut config = small_config();
        config.features.extend([
            FeatureSpec::new(IndicatorKind::Rvol, 20),
            FeatureSpec::new(IndicatorKind::Cmf, 20),
            FeatureSpec::new(IndicatorKind::Vwap, 20),
            FeatureSpec::new(IndicatorKind::DiSpread, 14),
            FeatureSpec::new(IndicatorKind::DiPlus, 14),
            FeatureSpec::new(IndicatorKind::DiMinus, 14),
            FeatureSpec::new(IndicatorKind::Chop, 30),
            FeatureSpec::new(IndicatorKind::Vhf, 30),
        ]);
        config.derived_features = DerivedFeatureConfig {
            include_direction_interactions: true,
            ..DerivedFeatureConfig::default()
        };

        let events = prepare_events(&bars(&closes), &config).unwrap();
        assert!(!events.is_empty());
        for name in [
            "rvol_20",
            "cmf_20",
            "vwap_20",
            "di_spread_14",
            "di_plus_14",
            "di_minus_14",
            "chop_30",
            "vhf_30",
            "direction_x_rvol_20",
            "direction_x_cmf_20",
            "direction_x_di_spread_14",
            "direction_x_chop_30",
            "direction_x_vhf_30",
        ] {
            assert!(
                config.feature_names().contains(&name.to_string()),
                "missing {name}"
            );
        }
        for event in events {
            assert!(event.features.values().all(|value| value.is_finite()));
            let direction = f64::from(event.raw_direction);
            for (interaction, source) in [
                ("direction_x_rvol_20", "rvol_20"),
                ("direction_x_cmf_20", "cmf_20"),
                ("direction_x_di_spread_14", "di_spread_14"),
                ("direction_x_chop_30", "chop_30"),
                ("direction_x_vhf_30", "vhf_30"),
            ] {
                assert_eq!(
                    event.features[interaction],
                    direction * event.features[source],
                    "interaction {interaction} must be direction times {source}"
                );
            }
        }
        for event in prepare_events(&bars(&closes), &config).unwrap() {
            assert!(event.features["di_plus_14"] >= 0.0);
            assert!(event.features["di_minus_14"] >= 0.0);
            assert!(event.features["di_plus_14"] <= 100.0);
            assert!(event.features["di_minus_14"] <= 100.0);
        }
    }

    #[test]
    fn session_id_uses_previous_local_date_before_five_pm() {
        let config = SupervisedConfig::default();
        let tz = Tz::from_str("America/New_York").unwrap();
        let at_noon = chrono::DateTime::parse_from_rfc3339("2026-08-13T12:00:00-04:00")
            .unwrap()
            .with_timezone(&Utc)
            .timestamp_nanos_opt()
            .unwrap();
        let key = session_id(at_noon, tz, &config).unwrap().unwrap();
        assert_eq!(key, "2026-08-12");
    }

    #[test]
    fn transition_cost_charges_entry_exit_and_reversal_consistently() {
        let mut config = SupervisedConfig::default();
        config.round_trip_cost = 4.0;
        assert_eq!(transition_cost(0, 1, &config), -2.0);
        assert_eq!(transition_cost(1, 0, &config), -2.0);
        assert_eq!(transition_cost(-1, 1, &config), -4.0);
        assert_eq!(transition_cost(1, 1, &config), 0.0);
    }

    #[test]
    fn appending_future_bars_cannot_change_existing_feature_values() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let start = DateTime::parse_from_rfc3339("2026-08-02T22:00:00Z")
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let prefix = bars_at(&closes, start);
        let mut extended_closes = closes.to_vec();
        extended_closes.extend([102.0, 98.0, 97.0, 101.0, 105.0]);
        let extended = bars_at(&extended_closes, start);
        let mut config = small_config();
        config.features.extend([
            FeatureSpec {
                indicator: IndicatorKind::Rsi,
                period: 3,
                lookbacks: vec![1, 2],
                include_value: true,
                include_delta: true,
                normalize_by_atr: false,
                normalize_value_by_atr: false,
            },
            FeatureSpec {
                indicator: IndicatorKind::Stoch,
                period: 3,
                lookbacks: vec![1, 2],
                include_value: true,
                include_delta: true,
                normalize_by_atr: false,
                normalize_value_by_atr: false,
            },
        ]);
        config.derived_features = DerivedFeatureConfig {
            include_time_cycle: true,
            cross_lookbacks: vec![5],
            context_cross_lookbacks: vec![],
            range_lookbacks: vec![3],
            atr_ratio_slow_period: None,
            include_direction_interactions: false,
            regime_persistence_lookbacks: vec![3, 5],
        };
        let original = prepare_events(&prefix, &config).unwrap();
        let appended = prepare_events(&extended, &config).unwrap();
        assert!(!original.is_empty());
        for event in original {
            let matching = appended
                .iter()
                .find(|candidate| candidate.row_idx == event.row_idx)
                .expect("event must remain identifiable after append");
            assert_eq!(matching.raw_direction, event.raw_direction);
            assert_eq!(matching.features, event.features);
        }
    }

    #[test]
    fn dense_features_match_sparse_events_and_are_causal() {
        let closes = [
            100.0, 99.0, 98.0, 99.0, 101.0, 100.0, 98.0, 97.0, 99.0, 102.0, 101.0, 99.0, 98.0,
            100.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0,
        ];
        let start = DateTime::parse_from_rfc3339("2026-08-02T22:00:00Z")
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let prefix = bars_at(&closes, start);
        let mut extended_closes = closes.to_vec();
        extended_closes.extend([102.0, 98.0, 97.0, 101.0, 105.0]);
        let extended = bars_at(&extended_closes, start);
        let mut config = small_config();
        config.derived_features = DerivedFeatureConfig {
            include_time_cycle: true,
            cross_lookbacks: vec![5],
            context_cross_lookbacks: vec![5, 10],
            range_lookbacks: vec![3],
            atr_ratio_slow_period: None,
            include_direction_interactions: false,
            regime_persistence_lookbacks: vec![3, 5],
        };

        let dense = prepare_bar_features(&prefix, &config).unwrap();
        let dense_extended = prepare_bar_features(&extended, &config).unwrap();
        let sparse = prepare_events(&prefix, &config).unwrap();

        assert_eq!(dense.len(), prefix.close.len());
        assert_eq!(dense.len(), dense_extended[..dense.len()].len());
        for (row, appended) in dense.iter().zip(&dense_extended) {
            assert_eq!(row.row_idx, appended.row_idx);
            assert_eq!(row.session_id, appended.session_id);
            assert_eq!(row.ready, appended.ready);
            assert_eq!(row.cross_direction, appended.cross_direction);
            assert_eq!(row.features, appended.features);
        }

        for event in sparse {
            let row = dense
                .get(event.row_idx)
                .expect("sparse event row must exist in dense output");
            assert!(row.ready);
            assert_eq!(row.cross_direction, Some(event.raw_direction));
            assert_eq!(row.features, event.features);
        }
    }
}
