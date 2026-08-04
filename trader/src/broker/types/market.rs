use super::InstrumentSessionProfile;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarKind {
    Minute,
    Second,
    Tick,
    Volume,
    Range,
}

impl BarKind {
    const ALL: [Self; 5] = [
        Self::Minute,
        Self::Second,
        Self::Tick,
        Self::Volume,
        Self::Range,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Minute => "Minute",
            Self::Second => "Seconds",
            Self::Tick => "Tick Count",
            Self::Volume => "Volume",
            Self::Range => "Range",
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            Self::Minute => "Min",
            Self::Second => "Sec",
            Self::Tick => "Tick",
            Self::Volume => "Vol",
            Self::Range => "Range",
        }
    }

    fn next(self) -> Self {
        let index = Self::ALL.iter().position(|kind| *kind == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    fn previous(self) -> Self {
        let index = Self::ALL.iter().position(|kind| *kind == self).unwrap_or(0);
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarType {
    kind: BarKind,
    value: u32,
}

impl BarType {
    pub fn new(kind: BarKind, value: u32) -> Self {
        Self {
            kind,
            value: value.max(1),
        }
    }

    pub fn minute(value: u32) -> Self {
        Self::new(BarKind::Minute, value)
    }

    pub fn second(value: u32) -> Self {
        Self::new(BarKind::Second, value)
    }

    pub fn tick(value: u32) -> Self {
        Self::new(BarKind::Tick, value)
    }

    pub fn volume(value: u32) -> Self {
        Self::new(BarKind::Volume, value)
    }

    pub fn range(value: u32) -> Self {
        Self::new(BarKind::Range, value)
    }

    pub fn kind(self) -> BarKind {
        self.kind
    }

    pub fn value(self) -> u32 {
        self.value
    }

    pub fn is_range(self) -> bool {
        self.kind == BarKind::Range
    }

    pub fn is_one_minute(self) -> bool {
        self.kind == BarKind::Minute && self.value == 1
    }

    pub fn is_time_based(self) -> bool {
        matches!(self.kind, BarKind::Minute | BarKind::Second)
    }

    pub fn supports_candle_mode(self) -> bool {
        !self.is_range()
    }

    pub fn effective_candle_mode(self, candle_mode: CandleMode) -> CandleMode {
        if self.supports_candle_mode() {
            candle_mode
        } else {
            CandleMode::Standard
        }
    }

    pub fn with_value(self, value: u32) -> Self {
        Self::new(self.kind, value)
    }

    pub fn next_kind(self) -> Self {
        match self.kind.next() {
            BarKind::Minute => Self::minute(self.value),
            BarKind::Second => Self::second(self.value),
            BarKind::Tick => Self::tick(self.value),
            BarKind::Volume => Self::volume(self.value),
            BarKind::Range => Self::range(self.value),
        }
    }

    pub fn previous_kind(self) -> Self {
        match self.kind.previous() {
            BarKind::Minute => Self::minute(self.value),
            BarKind::Second => Self::second(self.value),
            BarKind::Tick => Self::tick(self.value),
            BarKind::Volume => Self::volume(self.value),
            BarKind::Range => Self::range(self.value),
        }
    }

    pub fn label(self) -> String {
        format!("{} {}", self.value, self.kind.short_label())
    }

    pub fn mode_label(self, candle_mode: CandleMode) -> String {
        if self.supports_candle_mode() {
            format!("{} {}", candle_mode.label(), self.label())
        } else {
            self.label()
        }
    }

    pub fn chart_description(self) -> Value {
        match self.kind {
            BarKind::Minute => json!({
                "underlyingType": "MinuteBar",
                "elementSize": self.value,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }),
            BarKind::Second => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Seconds",
                "withHistogram": false
            }),
            BarKind::Tick => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }),
            BarKind::Volume => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Volume",
                "withHistogram": false
            }),
            BarKind::Range => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Range",
                "withHistogram": false
            }),
        }
    }
}

impl Default for BarType {
    fn default() -> Self {
        Self::minute(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandleMode {
    Standard,
    HeikinAshi,
}

impl CandleMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "OHLC",
            Self::HeikinAshi => "Heikin Ashi",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Standard => Self::HeikinAshi,
            Self::HeikinAshi => Self::Standard,
        }
    }
}

impl Default for CandleMode {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
}

pub fn transform_bars_for_candle_mode(bars: &[Bar], candle_mode: CandleMode) -> Vec<Bar> {
    match candle_mode {
        CandleMode::Standard => bars.to_vec(),
        CandleMode::HeikinAshi => heikin_ashi_bars(bars),
    }
}

/// Incremental state for deriving Heikin Ashi candles.
///
/// Heikin Ashi opens are recursive: the current open depends on the previous
/// transformed open/close.  Keeping those two values means a new source bar
/// can be transformed in constant time instead of rebuilding the complete
/// history on every market update.  The state is intentionally independent of
/// replay/live plumbing so callers can use it for either stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HeikinAshiState {
    previous_open: Option<f64>,
    previous_close: Option<f64>,
}

impl HeikinAshiState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Transform one raw OHLC bar and advance the recursive state.
    pub fn push(&mut self, bar: &Bar) -> Bar {
        let ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0;
        let ha_open = match (self.previous_open, self.previous_close) {
            (Some(open), Some(close)) => (open + close) / 2.0,
            _ => (bar.open + bar.close) / 2.0,
        };
        let transformed = Bar {
            ts_ns: bar.ts_ns,
            open: ha_open,
            high: bar.high.max(ha_open).max(ha_close),
            low: bar.low.min(ha_open).min(ha_close),
            close: ha_close,
            volume: bar.volume,
        };
        self.previous_open = Some(transformed.open);
        self.previous_close = Some(transformed.close);
        transformed
    }

    /// Transform a forming source bar without mutating state.
    pub fn forming(&self, bar: &Bar) -> Bar {
        let mut next = self.clone();
        next.push(bar)
    }

    /// Rebuild the state from a source slice and return transformed bars.
    /// This is used only after a correction/out-of-order insertion; the
    /// normal append path should call [`Self::push`].
    pub fn transform_all(&mut self, bars: &[Bar]) -> Vec<Bar> {
        self.reset();
        let mut transformed = Vec::with_capacity(bars.len());
        for bar in bars {
            transformed.push(self.push(bar));
        }
        transformed
    }

    pub fn previous_open(&self) -> Option<f64> {
        self.previous_open
    }

    pub fn previous_close(&self) -> Option<f64> {
        self.previous_close
    }
}

fn heikin_ashi_bars(bars: &[Bar]) -> Vec<Bar> {
    let mut state = HeikinAshiState::new();
    state.transform_all(bars)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketSnapshot {
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub candle_mode: CandleMode,
    pub bars: Vec<Bar>,
    pub trade_markers: Vec<TradeMarker>,
    pub session_profile: Option<InstrumentSessionProfile>,
    pub value_per_point: Option<f64>,
    pub tick_size: Option<f64>,
    pub history_loaded: usize,
    pub live_bars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_window: Option<super::ReplayWindowSnapshot>,
    pub status: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TradeMarkerSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMarker {
    pub fill_id: Option<i64>,
    pub account_id: Option<i64>,
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub ts_ns: i64,
    pub price: f64,
    pub qty: i32,
    pub side: TradeMarkerSide,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_candle_mode_preserves_raw_bars() {
        let bars = vec![Bar {
            ts_ns: 1,
            open: 10.0,
            high: 12.0,
            low: 9.0,
            close: 11.0,
            volume: Some(125.0),
        }];

        assert_eq!(
            transform_bars_for_candle_mode(&bars, CandleMode::Standard),
            bars
        );
    }

    #[test]
    fn heikin_ashi_candle_mode_transforms_ohlc_recursively() {
        let bars = vec![
            Bar {
                ts_ns: 1,
                open: 10.0,
                high: 14.0,
                low: 8.0,
                close: 12.0,
                volume: Some(100.0),
            },
            Bar {
                ts_ns: 2,
                open: 12.0,
                high: 16.0,
                low: 11.0,
                close: 15.0,
                volume: Some(150.0),
            },
        ];

        let transformed = transform_bars_for_candle_mode(&bars, CandleMode::HeikinAshi);

        assert_eq!(
            transformed,
            vec![
                Bar {
                    ts_ns: 1,
                    open: 11.0,
                    high: 14.0,
                    low: 8.0,
                    close: 11.0,
                    volume: Some(100.0),
                },
                Bar {
                    ts_ns: 2,
                    open: 11.0,
                    high: 16.0,
                    low: 11.0,
                    close: 13.5,
                    volume: Some(150.0),
                },
            ]
        );
    }

    #[test]
    fn incremental_heikin_ashi_matches_batch_transform() {
        let bars = (0..64)
            .map(|idx| Bar {
                ts_ns: idx + 1,
                open: 100.0 + idx as f64 * 0.25,
                high: 101.5 + idx as f64 * 0.2,
                low: 98.5 + idx as f64 * 0.15,
                close: 100.5 + (idx as f64 * 0.31).sin(),
                volume: Some(idx as f64),
            })
            .collect::<Vec<_>>();
        let expected = transform_bars_for_candle_mode(&bars, CandleMode::HeikinAshi);
        let mut state = HeikinAshiState::new();
        let actual = bars.iter().map(|bar| state.push(bar)).collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn heikin_ashi_forming_bar_does_not_advance_state() {
        let first = Bar {
            ts_ns: 1,
            open: 10.0,
            high: 12.0,
            low: 8.0,
            close: 11.0,
            volume: None,
        };
        let forming = Bar {
            ts_ns: 2,
            open: 11.0,
            high: 13.0,
            low: 10.0,
            close: 12.0,
            volume: None,
        };
        let mut state = HeikinAshiState::new();
        let _ = state.push(&first);
        let before = state.clone();
        let derived = state.forming(&forming);
        assert_eq!(state, before);
        let mut expected_state = before;
        assert_eq!(derived, expected_state.push(&forming));
    }

    #[test]
    fn bar_type_chart_descriptions_match_tradovate_units() {
        assert_eq!(
            BarType::minute(3).chart_description(),
            json!({
                "underlyingType": "MinuteBar",
                "elementSize": 3,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::second(15).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 15,
                "elementSizeUnit": "Seconds",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::tick(100).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 100,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::volume(1000).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 1000,
                "elementSizeUnit": "Volume",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::range(4).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 4,
                "elementSizeUnit": "Range",
                "withHistogram": false
            })
        );
    }

    #[test]
    fn range_bars_force_standard_effective_candle_mode() {
        assert_eq!(
            BarType::range(1).effective_candle_mode(CandleMode::HeikinAshi),
            CandleMode::Standard
        );
        assert_eq!(
            BarType::volume(1000).effective_candle_mode(CandleMode::HeikinAshi),
            CandleMode::HeikinAshi
        );
    }
}
