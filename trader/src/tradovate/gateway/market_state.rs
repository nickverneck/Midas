use super::*;
use crate::broker::{HeikinAshiState, MarketHistoryUpdate, transform_bars_for_candle_mode};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MARKET_UPDATE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub(crate) struct LiveSeries {
    pub(crate) closed_bars: Vec<Bar>,
    pub(crate) forming_bar: Option<Bar>,
    /// Cached transformed candles.  The raw series remains the source of
    /// truth for fills; this cache prevents rebuilding the entire Heikin Ashi
    /// history for every incoming bar.
    heikin_ashi_closed_bars: Vec<Bar>,
    heikin_ashi_state: HeikinAshiState,
    closed_revision: Cell<u64>,
    published_closed_revision: Cell<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct MarketUpdate {
    pub(crate) contract_id: i64,
    pub(crate) contract_name: String,
    pub(crate) candle_mode: CandleMode,
    pub(crate) session_profile: Option<InstrumentSessionProfile>,
    pub(crate) value_per_point: Option<f64>,
    pub(crate) tick_size: Option<f64>,
    pub(crate) history_loaded: usize,
    pub(crate) live_bars: usize,
    pub(crate) replay_window: Option<ReplayWindowSnapshot>,
    pub(crate) history_update: MarketHistoryUpdate,
    pub(crate) history_sequence: u64,
    pub(crate) status: String,
    pub(crate) bars: MarketBarsUpdate,
}

#[derive(Debug, Clone)]
pub(crate) enum MarketBarsUpdate {
    Snapshot {
        closed_bars: Vec<Bar>,
        forming_bar: Option<Bar>,
    },
    Forming {
        forming_bar: Bar,
    },
    Closed {
        closed_bar: Bar,
        forming_bar: Option<Bar>,
    },
}

impl LiveSeries {
    pub(crate) fn new() -> Self {
        Self {
            closed_bars: Vec::new(),
            forming_bar: None,
            heikin_ashi_closed_bars: Vec::new(),
            heikin_ashi_state: HeikinAshiState::new(),
            closed_revision: Cell::new(0),
            published_closed_revision: Cell::new(0),
        }
    }

    pub(crate) fn push_closed_bar(&mut self, bar: &Bar) {
        let insertion = self
            .closed_bars
            .binary_search_by_key(&bar.ts_ns, |current| current.ts_ns);
        match insertion {
            Ok(index) => {
                self.closed_bars[index] = bar.clone();
                // A correction can affect every subsequent HA open, so
                // rebuild only on this uncommon non-append path.
                self.heikin_ashi_closed_bars =
                    self.heikin_ashi_state.transform_all(&self.closed_bars);
            }
            Err(index) if index == self.closed_bars.len() => {
                self.closed_bars.push(bar.clone());
                self.heikin_ashi_closed_bars
                    .push(self.heikin_ashi_state.push(bar));
            }
            Err(index) => {
                self.closed_bars.insert(index, bar.clone());
                self.heikin_ashi_closed_bars =
                    self.heikin_ashi_state.transform_all(&self.closed_bars);
            }
        }
        self.closed_revision
            .set(self.closed_revision.get().saturating_add(1));
    }

    pub(crate) fn push_closed_bar_capped(&mut self, bar: &Bar, max_closed_bars: usize) {
        self.push_closed_bar(bar);
        if max_closed_bars == 0 {
            self.closed_bars.clear();
            self.heikin_ashi_closed_bars.clear();
            self.heikin_ashi_state.reset();
            return;
        }
        if self.closed_bars.len() > max_closed_bars {
            let overflow = self.closed_bars.len() - max_closed_bars;
            self.closed_bars.drain(0..overflow);
            self.heikin_ashi_closed_bars.drain(0..overflow);
        }
    }

    pub(crate) fn mark_closed_revision_published(&self) {
        self.published_closed_revision
            .set(self.closed_revision.get());
    }

    fn heikin_ashi_closed_bars(&self) -> &[Bar] {
        &self.heikin_ashi_closed_bars
    }

    fn heikin_ashi_forming_bar(&self, bar: &Bar) -> Bar {
        self.heikin_ashi_state.forming(bar)
    }
}

fn market_last_closed_ts(market: &MarketSnapshot) -> Option<i64> {
    let closed_len = market.history_loaded.min(market.bars.len());
    closed_len
        .checked_sub(1)
        .and_then(|idx| market.bars.get(idx))
        .map(|bar| bar.ts_ns)
}

fn trim_market_closed_bars(market: &mut MarketSnapshot, limit: usize) {
    let closed_len = market.history_loaded.min(market.bars.len());
    if closed_len <= limit {
        market.history_loaded = closed_len;
        return;
    }

    let overflow = closed_len - limit;
    market.bars.drain(0..overflow);
    market.history_loaded = limit;
}

pub(crate) fn display_market_snapshot(market: &MarketSnapshot) -> MarketSnapshot {
    let closed_len = market.history_loaded.min(market.bars.len());
    let retained_closed = closed_len.min(UI_MARKET_BAR_LIMIT);
    let closed_start = closed_len.saturating_sub(retained_closed);
    let mut bars = market.bars[closed_start..closed_len].to_vec();

    if let Some(forming_bar) = market.bars.get(closed_len).cloned() {
        bars.push(forming_bar);
    }

    MarketSnapshot {
        contract_id: market.contract_id,
        contract_name: market.contract_name.clone(),
        candle_mode: market.candle_mode,
        bars,
        trade_markers: market.trade_markers.clone(),
        session_profile: market.session_profile,
        value_per_point: market.value_per_point,
        tick_size: market.tick_size,
        history_loaded: retained_closed,
        live_bars: market.live_bars,
        replay_window: market.replay_window.clone(),
        status: market.status.clone(),
    }
}

pub(crate) fn build_market_update(
    contract: &ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    candle_mode: CandleMode,
    history_loaded: usize,
    live_bars: usize,
    status: String,
    before_closed_len: usize,
    before_last_closed: Option<Bar>,
    before_forming: Option<Bar>,
    series: &LiveSeries,
) -> Option<MarketUpdate> {
    let bars = if before_closed_len == 0 && history_loaded > 0 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len + 1 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.closed_bars.last() != before_last_closed.as_ref() {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.closed_revision.get() != series.published_closed_revision.get() {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if series.forming_bar != before_forming {
        series
            .forming_bar
            .clone()
            .map(|forming_bar| MarketBarsUpdate::Forming { forming_bar })
    } else {
        None
    }?;
    let bars = transform_market_bars_update(bars, candle_mode, series);

    let revision_delta = series
        .closed_revision
        .get()
        .saturating_sub(series.published_closed_revision.get());
    let history_update = match &bars {
        MarketBarsUpdate::Snapshot { .. } => MarketHistoryUpdate::Snapshot,
        MarketBarsUpdate::Forming { .. } => MarketHistoryUpdate::Unchanged,
        MarketBarsUpdate::Closed { closed_bar, .. } => {
            if revision_delta == 1
                && before_last_closed
                    .as_ref()
                    .is_none_or(|previous| closed_bar.ts_ns > previous.ts_ns)
            {
                MarketHistoryUpdate::Append
            } else {
                MarketHistoryUpdate::Correction
            }
        }
    };
    series.mark_closed_revision_published();
    Some(MarketUpdate {
        contract_id: contract.id,
        contract_name: contract.name.clone(),
        candle_mode,
        session_profile: market_specs.and_then(|specs| specs.session_profile),
        value_per_point: market_specs.and_then(|specs| specs.value_per_point),
        tick_size: market_specs.and_then(|specs| specs.tick_size),
        history_loaded,
        live_bars,
        replay_window: None,
        history_update,
        history_sequence: NEXT_MARKET_UPDATE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        status,
        bars,
    })
}

fn transform_market_bars_update(
    update: MarketBarsUpdate,
    candle_mode: CandleMode,
    series: &LiveSeries,
) -> MarketBarsUpdate {
    match candle_mode {
        CandleMode::Standard => update,
        CandleMode::HeikinAshi => {
            let transformed_closed =
                if series.heikin_ashi_closed_bars().len() == series.closed_bars.len() {
                    series.heikin_ashi_closed_bars().to_vec()
                } else {
                    // Defensive fallback for a future LiveSeries constructor or
                    // an out-of-band mutation. The normal append path uses the
                    // cached state above.
                    transform_bars_for_candle_mode(&series.closed_bars, candle_mode)
                };
            let transformed_forming = series
                .forming_bar
                .as_ref()
                .map(|forming_bar| series.heikin_ashi_forming_bar(forming_bar));

            match update {
                MarketBarsUpdate::Snapshot { .. } => MarketBarsUpdate::Snapshot {
                    closed_bars: transformed_closed,
                    forming_bar: transformed_forming,
                },
                MarketBarsUpdate::Forming { forming_bar } => MarketBarsUpdate::Forming {
                    forming_bar: transformed_forming.unwrap_or(forming_bar),
                },
                MarketBarsUpdate::Closed {
                    closed_bar,
                    forming_bar: _,
                } => MarketBarsUpdate::Closed {
                    closed_bar: transformed_closed.last().cloned().unwrap_or(closed_bar),
                    forming_bar: transformed_forming,
                },
            }
        }
    }
}

pub(crate) fn apply_market_update(market: &mut MarketSnapshot, update: MarketUpdate) -> bool {
    let prev_last_closed_ts = market_last_closed_ts(market);
    market.contract_id = Some(update.contract_id);
    market.contract_name = Some(update.contract_name);
    market.candle_mode = update.candle_mode;
    market.session_profile = update.session_profile;
    market.value_per_point = update.value_per_point;
    market.tick_size = update.tick_size;
    market.live_bars = update.live_bars;
    market.replay_window = update.replay_window;
    market.status = update.status;

    let closed_bar_advanced = match update.bars {
        MarketBarsUpdate::Snapshot {
            closed_bars,
            forming_bar,
        } => {
            let next_last_closed_ts = closed_bars.last().map(|bar| bar.ts_ns);
            market.history_loaded = update.history_loaded.min(closed_bars.len());
            market.bars = closed_bars;
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            next_last_closed_ts.is_some_and(|ts| prev_last_closed_ts.is_none_or(|prev| ts > prev))
        }
        MarketBarsUpdate::Forming { forming_bar } => {
            let closed_len = update.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            market.history_loaded = closed_len;
            market.bars.push(forming_bar);
            false
        }
        MarketBarsUpdate::Closed {
            closed_bar,
            forming_bar,
        } => {
            let closed_len = market.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            match market.bars.last_mut() {
                Some(last) if last.ts_ns == closed_bar.ts_ns => {
                    *last = closed_bar.clone();
                }
                Some(last) if closed_bar.ts_ns > last.ts_ns => {
                    market.bars.push(closed_bar.clone());
                }
                None => market.bars.push(closed_bar.clone()),
                _ => {}
            }
            let retained_closed = update.history_loaded.min(market.bars.len());
            if market.bars.len() > retained_closed {
                let overflow = market.bars.len() - retained_closed;
                market.bars.drain(0..overflow);
            }
            market.history_loaded = market.bars.len();
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            prev_last_closed_ts.is_none_or(|prev| closed_bar.ts_ns > prev)
        }
    };

    trim_market_closed_bars(market, ENGINE_MARKET_BAR_LIMIT);

    closed_bar_advanced
}
