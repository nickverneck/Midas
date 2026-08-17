//! Event-level supervised-learning dataset preparation.
//!
//! The dataset is intentionally sparse: one row is emitted for each causal
//! trigger crossover, not for every bar.  Rolling indicators and all feature
//! values are evaluated at the closed crossover bar.  The labeler is allowed
//! to look forward, because its output is a target, but those future prices
//! are kept in explicitly named label/audit columns and are never part of the
//! feature registry.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveTime, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

pub const SUPERVISED_DATASET_SCHEMA: &str = "supervised-event-v1";
pub const SUPERVISED_LABEL_SCHEMA: &str = "backward-dynamic-program-v1";

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
    Rvol,
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
            Self::Rvol => "rvol",
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
            "rvol" | "relative_volume" => Ok(Self::Rvol),
            "er" | "efficiency_ratio" | "kaufman_er" => Ok(Self::Er),
            "close" | "price" => Ok(Self::Close),
            other => bail!("unknown indicator `{other}`"),
        }
    }
}

fn default_lookbacks() -> Vec<usize> {
    vec![1, 3, 5]
}

/// A selectable indicator and the causal deltas to expose around an event.
/// A delta is `(value[t] - value[t-lookback]) / ATR[t]` when normalization is
/// enabled.  Lookbacks are evaluated strictly at or before the event row.
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
}

fn default_true() -> bool {
    true
}

impl FeatureSpec {
    pub fn new(indicator: IndicatorKind, period: usize) -> Self {
        Self {
            indicator,
            period,
            lookbacks: default_lookbacks(),
            include_value: true,
            include_delta: true,
            normalize_by_atr: true,
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
        Ok(())
    }

    pub fn base_name(&self) -> String {
        format!("{}_{}", self.indicator, self.period)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.include_value {
            names.push(self.base_name());
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
            session_timezone: default_timezone(),
            session_start_hour: default_start_hour(),
            session_end_hour: default_end_hour(),
            bar_kind: default_bar_kind(),
            bar_value: default_bar_value(),
            contract_multiplier: default_contract_multiplier(),
            round_trip_cost: 0.0,
            include_raw_direction: true,
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

    let mut candidates = Vec::new();
    let mut event_index = 0usize;
    for i in 1..bars.close.len() {
        let Some(session_id) = session_ids[i].as_ref() else {
            continue;
        };
        if session_ids[i - 1].as_ref() != Some(session_id) {
            continue;
        }
        let previous_spread = trigger_fast[i - 1] - trigger_slow[i - 1];
        let spread = trigger_fast[i] - trigger_slow[i];
        let Some(raw_direction) = crossing_direction(previous_spread, spread) else {
            continue;
        };
        let entry_idx = i + 1;
        if entry_idx >= bars.close.len() || session_ids[entry_idx].as_ref() != Some(session_id) {
            continue;
        }
        if !causal_values_ready(
            i,
            config,
            &atr,
            &trigger_fast,
            &trigger_slow,
            &context_fast,
            &context_slow,
            &series,
        ) {
            continue;
        }

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
                features.insert(base_name.clone(), value);
            }
            if spec.include_delta {
                for lookback in &spec.lookbacks {
                    let raw_delta = values[i] - values[i - lookback];
                    let delta = if spec.normalize_by_atr {
                        raw_delta / atr[i]
                    } else {
                        raw_delta
                    };
                    features.insert(
                        format!(
                            "{}_delta_{}{}",
                            base_name,
                            lookback,
                            if spec.normalize_by_atr { "_atr" } else { "" }
                        ),
                        delta,
                    );
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
        if features.values().any(|value| !value.is_finite()) {
            continue;
        }

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

    Ok(label_candidates(
        &mut candidates,
        bars,
        &session_ids,
        config,
    )?)
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
                values[j][pidx] = action_values[j][pidx]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }

        let mut position = 0_i8;
        for (j, index) in indices.into_iter().enumerate() {
            let candidate = &candidates[index];
            let pidx = position_index(position);
            let action_values_for_position = action_values[j][pidx];
            let action_idx = best_action_index(action_values_for_position);
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

fn best_action_index(values: [f64; 3]) -> usize {
    // Deterministic tie preference: skip, then normal, then invert.  This
    // avoids a fee-neutral churn label when two actions are equivalent.
    let order = [1usize, 0, 2];
    let mut best = order[0];
    for idx in order.into_iter().skip(1) {
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
        IndicatorKind::Rvol => crate::features::rvol(volume, period),
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
            }],
            ..SupervisedConfig::default()
        }
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
        let config = small_config();
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
}
