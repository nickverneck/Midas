use super::*;

#[cfg(feature = "replay")]
use super::bars::{
    build_range_bars, build_tick_count_bars, build_time_bars, build_volume_bars,
    derive_cached_raw_tick_bars, make_bar_timestamps_strictly_increasing,
};
#[cfg(feature = "replay")]
use super::ticks::ReplayTick;
#[cfg(feature = "replay")]
use crate::replay_cache::ReplayCacheResolvedRawTicks;
#[cfg(feature = "replay")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct ReplayState {
    #[cfg(feature = "replay")]
    pub(super) evaluation_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    #[cfg(feature = "replay")]
    pub(super) replay_window: Option<ReplayWindowSnapshot>,
    #[cfg(feature = "replay")]
    pub(super) contract: ContractSuggestion,
    #[cfg(feature = "replay")]
    pub(super) account: AccountInfo,
    #[cfg(feature = "replay")]
    pub(super) market_specs: MarketSpecs,
    #[cfg(feature = "replay")]
    pub(super) dom_updates: Arc<[ReplayMarketDom]>,
    #[cfg(feature = "replay")]
    pub(super) data: ReplayDataSource,
}

pub(crate) fn replay_accounts(state: &ReplayState) -> Vec<AccountInfo> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = state;
        Vec::new()
    }

    #[cfg(feature = "replay")]
    {
        vec![state.account.clone()]
    }
}

pub(crate) fn replay_contract(state: &ReplayState) -> ContractSuggestion {
    #[cfg(not(feature = "replay"))]
    {
        let _ = state;
        ContractSuggestion {
            id: 0,
            name: "Replay".to_string(),
            description: "Replay".to_string(),
            raw: json!({}),
        }
    }

    #[cfg(feature = "replay")]
    {
        state.contract.clone()
    }
}

pub(crate) fn search_replay_contracts(
    state: &ReplayState,
    query: &str,
    limit: usize,
) -> Vec<ContractSuggestion> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (state, query, limit);
        Vec::new()
    }

    #[cfg(feature = "replay")]
    {
        let needle = query.trim().to_ascii_lowercase();
        let haystack = format!(
            "{} {}",
            state.contract.name.to_ascii_lowercase(),
            state.contract.description.to_ascii_lowercase()
        );
        if needle.is_empty() || haystack.contains(&needle) {
            vec![state.contract.clone()]
                .into_iter()
                .take(limit.max(1))
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(feature = "replay")]
impl ReplayState {
    pub(super) fn evaluation_start_ns(&self) -> Result<Option<i64>> {
        self.evaluation_range
            .map(|range| range.bounds_ns().map(|(start, _)| start))
            .transpose()
    }

    pub(super) fn bars_for_type(&self, bar_type: BarType) -> Result<Vec<Bar>> {
        match &self.data {
            ReplayDataSource::PriceTicks(ticks) | ReplayDataSource::RawTicks(ticks) => {
                let ticks = ticks.as_ref();
                match bar_type.kind() {
                    BarKind::Minute => {
                        let interval = i64::from(bar_type.value()) * 60 * 1_000_000_000;
                        Ok(build_time_bars(ticks, interval.max(1)))
                    }
                    BarKind::Second => {
                        let interval = i64::from(bar_type.value()) * 1_000_000_000;
                        Ok(build_time_bars(ticks, interval.max(1)))
                    }
                    BarKind::Tick => {
                        let mut bars = build_tick_count_bars(ticks, bar_type.value());
                        make_bar_timestamps_strictly_increasing(&mut bars);
                        Ok(bars)
                    }
                    BarKind::Range => {
                        let mut bars = build_range_bars(
                            ticks,
                            self.market_specs.tick_size.unwrap_or(0.25).max(0.01)
                                * f64::from(bar_type.value()),
                        );
                        make_bar_timestamps_strictly_increasing(&mut bars);
                        Ok(bars)
                    }
                    BarKind::Volume => {
                        if matches!(&self.data, ReplayDataSource::PriceTicks(_)) {
                            bail!(
                                "volume bars require trade size; local replay file only has trusted last prices"
                            )
                        }
                        Ok(build_volume_bars(ticks, bar_type.value()))
                    }
                }
            }
            ReplayDataSource::CachedRawTicks {
                resolved,
                timestamp_range,
                ..
            } => derive_cached_raw_tick_bars(
                resolved,
                timestamp_range.as_ref(),
                bar_type,
                self.market_specs.tick_size.unwrap_or(0.25).max(0.01),
            ),
            ReplayDataSource::CachedServerBars {
                bars,
                bar_type: cached_bar_type,
                source_label,
            } => {
                if *cached_bar_type != bar_type {
                    bail!(
                        "cached server-bar replay from {source_label} contains {}; requested {}",
                        cached_bar_type.label(),
                        bar_type.label()
                    );
                }
                Ok(bars.as_ref().to_vec())
            }
        }
    }
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
pub(super) struct ReplayBarFrame {
    pub(super) bar: Bar,
    pub(super) ticks: Arc<[ReplayMarketTick]>,
    pub(super) dom_updates: Arc<[ReplayMarketDom]>,
}

#[cfg(feature = "replay")]
impl ReplayState {
    pub(super) fn frames_for_type(&self, bar_type: BarType) -> Result<Vec<ReplayBarFrame>> {
        let bars = self.bars_for_type(bar_type)?;
        let ticks = self.execution_ticks()?;
        let groups = group_execution_ticks(&ticks, &bars, bar_type, self.market_specs.tick_size);
        let dom_groups = group_dom_updates(&self.dom_updates, &bars, bar_type);
        Ok(bars
            .into_iter()
            .enumerate()
            .map(|(index, bar)| ReplayBarFrame {
                bar,
                ticks: Arc::from(
                    groups
                        .get(index)
                        .cloned()
                        .unwrap_or_default()
                        .into_boxed_slice(),
                ),
                dom_updates: Arc::from(
                    dom_groups
                        .get(index)
                        .cloned()
                        .unwrap_or_default()
                        .into_boxed_slice(),
                ),
            })
            .collect())
    }

    fn execution_ticks(&self) -> Result<Vec<ReplayMarketTick>> {
        match &self.data {
            ReplayDataSource::PriceTicks(ticks) | ReplayDataSource::RawTicks(ticks) => Ok(ticks
                .iter()
                .map(|tick| ReplayMarketTick {
                    ts_ns: tick.ts_ns,
                    last: tick.last,
                    size: tick.size,
                    bid_price: None,
                    bid_size: None,
                    ask_price: None,
                    ask_size: None,
                })
                .collect()),
            ReplayDataSource::CachedRawTicks {
                execution_ticks, ..
            } => Ok(execution_ticks.to_vec()),
            ReplayDataSource::CachedServerBars { .. } => Ok(Vec::new()),
        }
    }
}

#[cfg(feature = "replay")]
fn group_dom_updates(
    dom_updates: &[ReplayMarketDom],
    bars: &[Bar],
    bar_type: BarType,
) -> Vec<Vec<ReplayMarketDom>> {
    let mut groups = vec![Vec::new(); bars.len()];
    if bars.is_empty() {
        return groups;
    }

    for dom in dom_updates {
        let index = match bar_type.kind() {
            BarKind::Minute => {
                let interval_ns = i64::from(bar_type.value()) * 60 * 1_000_000_000;
                let period = dom.ts_ns - dom.ts_ns.rem_euclid(interval_ns.max(1));
                bars.partition_point(|bar| bar.ts_ns <= period)
                    .saturating_sub(1)
            }
            BarKind::Second => {
                let interval_ns = i64::from(bar_type.value()) * 1_000_000_000;
                let period = dom.ts_ns - dom.ts_ns.rem_euclid(interval_ns.max(1));
                bars.partition_point(|bar| bar.ts_ns <= period)
                    .saturating_sub(1)
            }
            // Non-time bars use the first completed source bar at or after
            // the snapshot timestamp. This keeps the DOM update with the
            // bar whose close made it visible without inventing a second
            // clock.
            BarKind::Tick | BarKind::Volume | BarKind::Range => bars
                .partition_point(|bar| bar.ts_ns < dom.ts_ns)
                .min(bars.len().saturating_sub(1)),
        };
        groups[index].push(dom.clone());
    }
    groups
}

#[cfg(feature = "replay")]
fn group_execution_ticks(
    ticks: &[ReplayMarketTick],
    bars: &[Bar],
    bar_type: BarType,
    tick_size: Option<f64>,
) -> Vec<Vec<ReplayMarketTick>> {
    let mut groups = vec![Vec::new(); bars.len()];
    if bars.is_empty() {
        return groups;
    }
    match bar_type.kind() {
        BarKind::Minute | BarKind::Second => {
            let interval_ns = match bar_type.kind() {
                BarKind::Minute => i64::from(bar_type.value()) * 60 * 1_000_000_000,
                BarKind::Second => i64::from(bar_type.value()) * 1_000_000_000,
                _ => unreachable!(),
            }
            .max(1);
            for tick in ticks {
                let period = tick.ts_ns - tick.ts_ns.rem_euclid(interval_ns);
                let index = bars
                    .partition_point(|bar| bar.ts_ns <= period)
                    .saturating_sub(1)
                    .min(groups.len() - 1);
                groups[index].push(*tick);
            }
        }
        BarKind::Tick => {
            let per_bar = bar_type.value().max(1) as usize;
            let last_group = groups.len() - 1;
            for (index, tick) in ticks.iter().enumerate() {
                groups[(index / per_bar).min(last_group)].push(*tick);
            }
        }
        BarKind::Volume => {
            let threshold = f64::from(bar_type.value().max(1));
            let mut cumulative = 0.0;
            for tick in ticks {
                let index = ((cumulative / threshold).floor() as usize).min(groups.len() - 1);
                groups[index].push(*tick);
                cumulative += tick.size.unwrap_or(0.0).max(0.0);
            }
        }
        BarKind::Range => {
            let range = tick_size.unwrap_or(0.25).max(0.01) * f64::from(bar_type.value());
            let mut index = 0usize;
            let mut open = ticks[0].last;
            let mut high = open;
            let mut low = open;
            for tick in ticks {
                groups[index].push(*tick);
                high = high.max(tick.last);
                low = low.min(tick.last);
                if (high - low) + f64::EPSILON >= range && index + 1 < groups.len() {
                    index += 1;
                    open = tick.last;
                    high = open;
                    low = open;
                }
            }
        }
    }
    groups
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
pub(super) enum ReplayDataSource {
    PriceTicks(Arc<[ReplayTick]>),
    RawTicks(Arc<[ReplayTick]>),
    CachedRawTicks {
        resolved: ReplayCacheResolvedRawTicks,
        timestamp_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
        execution_ticks: Arc<[ReplayMarketTick]>,
    },
    CachedServerBars {
        bars: Arc<[Bar]>,
        bar_type: BarType,
        source_label: String,
    },
}
