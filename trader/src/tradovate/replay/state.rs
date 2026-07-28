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
pub(super) enum ReplayDataSource {
    PriceTicks(Arc<[ReplayTick]>),
    RawTicks(Arc<[ReplayTick]>),
    CachedRawTicks {
        resolved: ReplayCacheResolvedRawTicks,
        timestamp_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    },
    CachedServerBars {
        bars: Arc<[Bar]>,
        bar_type: BarType,
        source_label: String,
    },
}
