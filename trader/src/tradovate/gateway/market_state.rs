use super::*;

#[derive(Debug, Clone)]
pub(crate) struct LiveSeries {
    pub(crate) closed_bars: Vec<Bar>,
    pub(crate) forming_bar: Option<Bar>,
}

#[derive(Debug, Clone)]
pub(crate) struct MarketUpdate {
    pub(crate) contract_id: i64,
    pub(crate) contract_name: String,
    pub(crate) session_profile: Option<InstrumentSessionProfile>,
    pub(crate) value_per_point: Option<f64>,
    pub(crate) tick_size: Option<f64>,
    pub(crate) history_loaded: usize,
    pub(crate) live_bars: usize,
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
        }
    }

    pub(crate) fn push_closed_bar(&mut self, bar: &Bar) {
        if let Some(last) = self.closed_bars.last_mut() {
            if bar.ts_ns == last.ts_ns {
                *last = bar.clone();
                return;
            }
            if bar.ts_ns < last.ts_ns {
                return;
            }
        }
        self.closed_bars.push(bar.clone());
    }

    pub(crate) fn push_closed_bar_capped(&mut self, bar: &Bar, max_closed_bars: usize) {
        self.push_closed_bar(bar);
        trim_recent_bars(&mut self.closed_bars, max_closed_bars);
    }
}

fn market_last_closed_ts(market: &MarketSnapshot) -> Option<i64> {
    let closed_len = market.history_loaded.min(market.bars.len());
    closed_len
        .checked_sub(1)
        .and_then(|idx| market.bars.get(idx))
        .map(|bar| bar.ts_ns)
}

fn trim_recent_bars(bars: &mut Vec<Bar>, limit: usize) {
    if bars.len() <= limit {
        return;
    }
    let overflow = bars.len() - limit;
    bars.drain(0..overflow);
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
        bars,
        trade_markers: market.trade_markers.clone(),
        session_profile: market.session_profile,
        value_per_point: market.value_per_point,
        tick_size: market.tick_size,
        history_loaded: retained_closed,
        live_bars: market.live_bars,
        status: market.status.clone(),
    }
}

pub(crate) fn build_market_update(
    contract: &ContractSuggestion,
    market_specs: Option<MarketSpecs>,
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
    } else if series.forming_bar != before_forming {
        series
            .forming_bar
            .clone()
            .map(|forming_bar| MarketBarsUpdate::Forming { forming_bar })
    } else {
        None
    }?;

    Some(MarketUpdate {
        contract_id: contract.id,
        contract_name: contract.name.clone(),
        session_profile: market_specs.and_then(|specs| specs.session_profile),
        value_per_point: market_specs.and_then(|specs| specs.value_per_point),
        tick_size: market_specs.and_then(|specs| specs.tick_size),
        history_loaded,
        live_bars,
        status,
        bars,
    })
}

pub(crate) fn apply_market_update(market: &mut MarketSnapshot, update: MarketUpdate) -> bool {
    let prev_last_closed_ts = market_last_closed_ts(market);
    market.contract_id = Some(update.contract_id);
    market.contract_name = Some(update.contract_name);
    market.session_profile = update.session_profile;
    market.value_per_point = update.value_per_point;
    market.tick_size = update.tick_size;
    market.live_bars = update.live_bars;
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
