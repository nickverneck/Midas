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
use crate::replay_cache::{ReplayCacheRawTickRow, stream_resolved_raw_ticks};
#[cfg(feature = "replay")]
use std::sync::Arc;
#[cfg(feature = "replay")]
use tokio::sync::mpsc;
#[cfg(feature = "replay")]
use tokio::task::JoinHandle;

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
    #[cfg(feature = "replay")]
    pub(super) shared_frames: Option<Arc<ReplayFrameSet>>,
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
    pub(super) fn is_cached_server_bars(&self) -> bool {
        matches!(self.data, ReplayDataSource::CachedServerBars { .. })
    }

    pub(super) fn market_tick_size(&self) -> Option<f64> {
        self.market_specs.tick_size
    }

    pub(super) fn market_value_per_point(&self) -> Option<f64> {
        self.market_specs.value_per_point
    }

    pub(super) fn market_session_profile(&self) -> Option<InstrumentSessionProfile> {
        self.market_specs.session_profile
    }

    pub(super) fn evaluation_start_ns(&self) -> Result<Option<i64>> {
        self.evaluation_range
            .map(|range| range.bounds_ns().map(|(start, _)| start))
            .transpose()
    }

    pub(super) fn has_execution_tick_data(&self) -> bool {
        match &self.data {
            ReplayDataSource::PriceTicks(ticks) | ReplayDataSource::RawTicks(ticks) => {
                !ticks.is_empty()
            }
            ReplayDataSource::CachedRawTicks { .. } => true,
            ReplayDataSource::CachedServerBars { .. } => false,
        }
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
impl ReplayState {
    pub(super) fn with_shared_frames(
        mut self,
        shared_frames: Arc<ReplayFrameSet>,
        bar_type: BarType,
        candle_mode: CandleMode,
    ) -> Result<Self> {
        if !shared_frames.matches(bar_type, candle_mode) {
            bail!(
                "shared replay frames are for {}, not {}",
                shared_frames.bar_type.mode_label(shared_frames.candle_mode),
                bar_type.mode_label(candle_mode)
            );
        }
        self.shared_frames = Some(shared_frames);
        Ok(self)
    }

    pub(super) fn frames_for_type(&self, bar_type: BarType) -> Result<Vec<ReplayBarFrame>> {
        if let Some(shared) = self.shared_frames.as_ref() {
            if shared.bar_type != bar_type {
                bail!(
                    "shared replay frames are for {}, not {}",
                    shared.bar_type.mode_label(shared.candle_mode),
                    bar_type.label()
                );
            }
            return Ok(shared.frames.iter().cloned().collect());
        }
        let bars = self.bars_for_type(bar_type)?;
        self.frames_for_bars(&bars, bar_type)
    }

    pub(super) fn shared_frame_set_for_type(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
    ) -> Result<Arc<ReplayFrameSet>> {
        let bars: Arc<[Bar]> = Arc::from(self.bars_for_type(bar_type)?.into_boxed_slice());
        let frames: Arc<[ReplayBarFrame]> =
            Arc::from(self.frames_for_bars(&bars, bar_type)?.into_boxed_slice());
        Ok(Arc::new(ReplayFrameSet {
            bar_type,
            candle_mode,
            bars,
            frames,
        }))
    }

    fn frames_for_bars(&self, bars: &[Bar], bar_type: BarType) -> Result<Vec<ReplayBarFrame>> {
        let ticks = self.execution_ticks()?;
        let groups = group_execution_ticks(&ticks, bars, bar_type, self.market_specs.tick_size);
        let dom_groups = group_dom_updates(&self.dom_updates, bars, bar_type);
        Ok(bars
            .iter()
            .enumerate()
            .map(|(index, bar)| ReplayBarFrame {
                bar: bar.clone(),
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

    /// Build a bounded frame stream for replay workers. Cached raw ticks are
    /// decoded on a blocking task and delivered through a small channel so the
    /// worker never holds the complete tick dataset in memory. Other replay
    /// sources retain the existing buffered path.
    pub(super) fn frame_stream_for_type(&self, bar_type: BarType) -> Result<ReplayFrameStream> {
        if let Some(shared) = self.shared_frames.as_ref() {
            if shared.bar_type != bar_type {
                bail!(
                    "shared replay frames are for {}, not {}",
                    shared.bar_type.mode_label(shared.candle_mode),
                    bar_type.label()
                );
            }
            return Ok(ReplayFrameStream::Buffered {
                bars: shared.bars.clone(),
                frames: shared
                    .frames
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter(),
            });
        }
        let bars: Arc<[Bar]> = Arc::from(self.bars_for_type(bar_type)?.into_boxed_slice());
        let ReplayDataSource::CachedRawTicks {
            resolved,
            timestamp_range,
        } = &self.data
        else {
            return Ok(ReplayFrameStream::Buffered {
                bars: bars.clone(),
                frames: self.frames_for_bars(&bars, bar_type)?.into_iter(),
            });
        };

        let (sender, receiver) = mpsc::channel(REPLAY_FRAME_CHANNEL_CAPACITY);
        let resolved = resolved.clone();
        let timestamp_range = timestamp_range.clone();
        let producer_bars = bars.clone();
        let dom_updates = self.dom_updates.clone();
        let tick_size = self.market_specs.tick_size.unwrap_or(0.25).max(0.01);
        let producer = tokio::task::spawn_blocking(move || {
            if let Err(error) = stream_cached_raw_tick_frames(
                resolved,
                timestamp_range,
                producer_bars,
                bar_type,
                tick_size,
                dom_updates,
                &sender,
            ) {
                let _ = sender.blocking_send(Err(error.to_string()));
            }
        });
        Ok(ReplayFrameStream::Streaming {
            bars,
            receiver,
            producer,
        })
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
                resolved,
                timestamp_range,
            } => {
                let mut ticks = Vec::new();
                stream_resolved_raw_ticks(resolved, timestamp_range.as_ref(), |row| {
                    ticks.push(replay_market_tick_from_row(&row));
                    Ok(())
                })?;
                Ok(ticks)
            }
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
    // Cached server-bar replay deliberately has no execution-tick stream.
    // There is still a valid bar frame for each cached bar, so leave the
    // per-bar tick groups empty instead of indexing an absent first tick.
    if bars.is_empty() || ticks.is_empty() {
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
    },
    CachedServerBars {
        bars: Arc<[Bar]>,
        bar_type: BarType,
        source_label: String,
    },
}

#[cfg(feature = "replay")]
const REPLAY_FRAME_CHANNEL_CAPACITY: usize = 4;

#[cfg(feature = "replay")]
pub(super) enum ReplayFrameStream {
    Buffered {
        bars: Arc<[Bar]>,
        frames: std::vec::IntoIter<ReplayBarFrame>,
    },
    Streaming {
        bars: Arc<[Bar]>,
        receiver: mpsc::Receiver<Result<ReplayBarFrame, String>>,
        producer: JoinHandle<()>,
    },
}

#[cfg(feature = "replay")]
impl ReplayFrameStream {
    pub(super) fn bars(&self) -> &[Bar] {
        match self {
            Self::Buffered { bars, .. } | Self::Streaming { bars, .. } => bars,
        }
    }

    pub(super) async fn next(&mut self) -> Result<Option<ReplayBarFrame>> {
        match self {
            Self::Buffered { frames, .. } => Ok(frames.next()),
            Self::Streaming { receiver, .. } => match receiver.recv().await {
                Some(Ok(frame)) => Ok(Some(frame)),
                Some(Err(error)) => bail!("stream replay frames: {error}"),
                None => Ok(None),
            },
        }
    }

    pub(super) async fn finish(&mut self) -> Result<()> {
        match self {
            Self::Buffered { frames, .. } => {
                if frames.next().is_some() {
                    bail!("replay frame stream produced more frames than scheduled bars");
                }
                Ok(())
            }
            Self::Streaming {
                bars,
                receiver,
                producer,
            } => {
                let mut extra = false;
                while let Some(item) = receiver.recv().await {
                    match item {
                        Ok(_) => extra = true,
                        Err(error) => bail!("stream replay frames: {error}"),
                    }
                }
                producer
                    .await
                    .map_err(|error| anyhow::anyhow!("join replay frame producer: {error}"))?;
                if extra {
                    bail!("replay frame stream produced more frames than scheduled bars");
                }
                let _ = bars;
                Ok(())
            }
        }
    }
}

#[cfg(feature = "replay")]
impl Drop for ReplayFrameStream {
    fn drop(&mut self) {
        if let Self::Streaming { producer, .. } = self {
            producer.abort();
        }
    }
}

#[cfg(feature = "replay")]
fn stream_cached_raw_tick_frames(
    resolved: ReplayCacheResolvedRawTicks,
    timestamp_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    bars: Arc<[Bar]>,
    bar_type: BarType,
    tick_size: f64,
    dom_updates: Arc<[ReplayMarketDom]>,
    sender: &mpsc::Sender<Result<ReplayBarFrame, String>>,
) -> Result<()> {
    let dom_groups = group_dom_updates(&dom_updates, &bars, bar_type);
    let mut assembler = RawTickFrameAssembler::new(bars, bar_type, tick_size, dom_groups, sender);
    stream_resolved_raw_ticks(&resolved, timestamp_range.as_ref(), |row| {
        assembler.push(&row)
    })?;
    assembler.finish()
}

#[cfg(feature = "replay")]
fn replay_market_tick_from_row(row: &ReplayCacheRawTickRow) -> ReplayMarketTick {
    ReplayMarketTick {
        ts_ns: row.ts_ns,
        last: row.price,
        size: Some(row.size),
        bid_price: row.bid_price,
        bid_size: row.bid_size,
        ask_price: row.ask_price,
        ask_size: row.ask_size,
    }
}

#[cfg(feature = "replay")]
struct RawTickFrameAssembler<'a> {
    bars: Arc<[Bar]>,
    bar_type: BarType,
    tick_size: f64,
    dom_groups: Vec<Vec<ReplayMarketDom>>,
    sender: &'a mpsc::Sender<Result<ReplayBarFrame, String>>,
    next_bar_index: usize,
    current_ticks: Vec<ReplayMarketTick>,
    tick_ordinal: usize,
    cumulative_volume: f64,
    range_initialized: bool,
    range_high: f64,
    range_low: f64,
    range_index: usize,
}

#[cfg(feature = "replay")]
impl<'a> RawTickFrameAssembler<'a> {
    fn new(
        bars: Arc<[Bar]>,
        bar_type: BarType,
        tick_size: f64,
        dom_groups: Vec<Vec<ReplayMarketDom>>,
        sender: &'a mpsc::Sender<Result<ReplayBarFrame, String>>,
    ) -> Self {
        Self {
            bars,
            bar_type,
            tick_size,
            dom_groups,
            sender,
            next_bar_index: 0,
            current_ticks: Vec::new(),
            tick_ordinal: 0,
            cumulative_volume: 0.0,
            range_initialized: false,
            range_high: 0.0,
            range_low: 0.0,
            range_index: 0,
        }
    }

    fn push(&mut self, row: &ReplayCacheRawTickRow) -> Result<()> {
        if self.bars.is_empty() {
            return Ok(());
        }
        let index = self.group_index(row).min(self.bars.len() - 1);
        if index < self.next_bar_index {
            bail!("raw-tick frame groups are not monotonic");
        }
        if self.next_bar_index < index {
            self.emit_current()?;
            while self.next_bar_index < index {
                self.emit_empty()?;
            }
        }
        self.current_ticks.push(replay_market_tick_from_row(row));
        Ok(())
    }

    fn group_index(&mut self, row: &ReplayCacheRawTickRow) -> usize {
        let index = match self.bar_type.kind() {
            BarKind::Minute => {
                let interval_ns = i64::from(self.bar_type.value()) * 60 * 1_000_000_000;
                let period = row.ts_ns - row.ts_ns.rem_euclid(interval_ns.max(1));
                self.bars
                    .partition_point(|bar| bar.ts_ns <= period)
                    .saturating_sub(1)
            }
            BarKind::Second => {
                let interval_ns = i64::from(self.bar_type.value()) * 1_000_000_000;
                let period = row.ts_ns - row.ts_ns.rem_euclid(interval_ns.max(1));
                self.bars
                    .partition_point(|bar| bar.ts_ns <= period)
                    .saturating_sub(1)
            }
            BarKind::Tick => self.tick_ordinal / self.bar_type.value().max(1) as usize,
            BarKind::Volume => {
                let threshold = f64::from(self.bar_type.value().max(1));
                let index = (self.cumulative_volume / threshold).floor() as usize;
                self.cumulative_volume += row.size.max(0.0);
                index
            }
            BarKind::Range => {
                if !self.range_initialized {
                    self.range_initialized = true;
                    self.range_high = row.price;
                    self.range_low = row.price;
                }
                self.range_high = self.range_high.max(row.price);
                self.range_low = self.range_low.min(row.price);
                let index = self.range_index;
                let range = self.bar_type.value().max(1) as f64 * self.tick_size;
                if (self.range_high - self.range_low) + f64::EPSILON >= range
                    && self.range_index + 1 < self.bars.len()
                {
                    self.range_index += 1;
                    self.range_high = row.price;
                    self.range_low = row.price;
                }
                index
            }
        };
        self.tick_ordinal = self.tick_ordinal.saturating_add(1);
        index
    }

    fn emit_current(&mut self) -> Result<()> {
        if self.next_bar_index >= self.bars.len() {
            self.current_ticks.clear();
            return Ok(());
        }
        let index = self.next_bar_index;
        let frame = ReplayBarFrame {
            bar: self.bars[index].clone(),
            ticks: Arc::from(std::mem::take(&mut self.current_ticks).into_boxed_slice()),
            dom_updates: Arc::from(self.dom_groups[index].clone().into_boxed_slice()),
        };
        self.sender
            .blocking_send(Ok(frame))
            .map_err(|_| anyhow::anyhow!("replay frame consumer closed"))?;
        self.next_bar_index += 1;
        Ok(())
    }

    fn emit_empty(&mut self) -> Result<()> {
        if self.next_bar_index >= self.bars.len() {
            return Ok(());
        }
        let index = self.next_bar_index;
        let frame = ReplayBarFrame {
            bar: self.bars[index].clone(),
            ticks: Arc::from(Vec::new().into_boxed_slice()),
            dom_updates: Arc::from(self.dom_groups[index].clone().into_boxed_slice()),
        };
        self.sender
            .blocking_send(Ok(frame))
            .map_err(|_| anyhow::anyhow!("replay frame consumer closed"))?;
        self.next_bar_index += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.next_bar_index < self.bars.len() {
            self.emit_current()?;
            while self.next_bar_index < self.bars.len() {
                self.emit_empty()?;
            }
        }
        Ok(())
    }
}
