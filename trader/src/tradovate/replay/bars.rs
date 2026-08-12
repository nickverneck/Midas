use super::*;

use super::ticks::ReplayTick;
use crate::replay_cache::{
    ReplayCacheResolvedRawTicks, ReplayCacheTimeRange, stream_resolved_raw_ticks,
};

pub(super) fn derive_cached_raw_tick_bars(
    resolved: &ReplayCacheResolvedRawTicks,
    timestamp_range: Option<&ReplayCacheTimeRange>,
    bar_type: BarType,
    tick_size: f64,
) -> Result<Vec<Bar>> {
    let mut bars = match bar_type.kind() {
        BarKind::Minute => {
            let interval_ns = i64::from(bar_type.value()) * 60 * 1_000_000_000;
            let mut builder = TimeBarBuilder::new(interval_ns.max(1));
            stream_resolved_raw_ticks(resolved, timestamp_range, |row| {
                builder.push_tick_with_size(row.ts_ns, row.price, Some(row.size));
                Ok(())
            })?;
            builder.finish()
        }
        BarKind::Second => {
            let interval_ns = i64::from(bar_type.value()) * 1_000_000_000;
            let mut builder = TimeBarBuilder::new(interval_ns.max(1));
            stream_resolved_raw_ticks(resolved, timestamp_range, |row| {
                builder.push_tick_with_size(row.ts_ns, row.price, Some(row.size));
                Ok(())
            })?;
            builder.finish()
        }
        BarKind::Tick => {
            let mut builder = TickCountBarBuilder::new(bar_type.value());
            stream_resolved_raw_ticks(resolved, timestamp_range, |row| {
                builder.push_tick(row.ts_ns, row.price);
                Ok(())
            })?;
            builder.finish()
        }
        BarKind::Volume => {
            let mut builder = VolumeBarBuilder::new(f64::from(bar_type.value()));
            stream_resolved_raw_ticks(resolved, timestamp_range, |row| {
                builder.push_tick(row.ts_ns, row.price, row.size);
                Ok(())
            })?;
            builder.finish()
        }
        BarKind::Range => {
            let mut builder = RangeBarBuilder::new(tick_size * f64::from(bar_type.value()));
            stream_resolved_raw_ticks(resolved, timestamp_range, |row| {
                builder.push_tick(row.ts_ns, row.price);
                Ok(())
            })?;
            builder.finish()
        }
    };
    if matches!(bar_type.kind(), BarKind::Tick) {
        make_bar_timestamps_strictly_increasing(&mut bars);
    }
    Ok(bars)
}

pub(super) fn build_time_bars(ticks: &[ReplayTick], interval_ns: i64) -> Vec<Bar> {
    let mut builder = TimeBarBuilder::new(interval_ns);
    for tick in ticks {
        builder.push_tick_with_size(tick.ts_ns, tick.last, tick.size);
    }
    builder.finish()
}

pub(super) fn build_tick_count_bars(ticks: &[ReplayTick], ticks_per_bar: u32) -> Vec<Bar> {
    let mut builder = TickCountBarBuilder::new(ticks_per_bar);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last);
    }
    builder.finish()
}

pub(super) fn build_range_bars(ticks: &[ReplayTick], range_size: f64) -> Vec<Bar> {
    let mut builder = RangeBarBuilder::new(range_size);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last);
    }
    builder.finish()
}

pub(super) fn build_volume_bars(ticks: &[ReplayTick], volume_per_bar: u32) -> Vec<Bar> {
    let mut builder = VolumeBarBuilder::new(volume_per_bar as f64);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last, tick.size.unwrap_or(0.0).max(0.0));
    }
    builder.finish()
}

pub(super) fn make_bar_timestamps_strictly_increasing(bars: &mut [Bar]) {
    let mut last_ts = None::<i64>;
    for bar in bars {
        if let Some(last) = last_ts
            && bar.ts_ns <= last
        {
            bar.ts_ns = last.saturating_add(1);
        }
        last_ts = Some(bar.ts_ns);
    }
}

pub(super) struct TimeBarBuilder {
    interval_ns: i64,
    current_period_ts_ns: Option<i64>,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

impl TimeBarBuilder {
    pub(super) fn new(interval_ns: i64) -> Self {
        Self {
            interval_ns: interval_ns.max(1),
            current_period_ts_ns: None,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    pub(super) fn push_tick(&mut self, ts_ns: i64, price: f64) {
        self.push_tick_with_size(ts_ns, price, None);
    }

    pub(super) fn push_tick_with_size(&mut self, ts_ns: i64, price: f64, volume: Option<f64>) {
        let volume = volume.filter(|value| value.is_finite() && *value >= 0.0);
        let period_ts_ns = ts_ns - ts_ns.rem_euclid(self.interval_ns);
        match self.current_bar.as_mut() {
            Some(current) if self.current_period_ts_ns == Some(period_ts_ns) => {
                current.high = current.high.max(price);
                current.low = current.low.min(price);
                current.close = price;
                if let Some(size) = volume {
                    current.volume = Some(current.volume.unwrap_or(0.0) + size);
                }
            }
            Some(_) => {
                if let Some(current) = self.current_bar.take() {
                    self.bars.push(current);
                }
                self.current_period_ts_ns = Some(period_ts_ns);
                self.current_bar = Some(new_bar(period_ts_ns, price, volume));
            }
            None => {
                self.current_period_ts_ns = Some(period_ts_ns);
                self.current_bar = Some(new_bar(period_ts_ns, price, volume));
            }
        }
    }

    pub(super) fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

pub(super) struct TickCountBarBuilder {
    ticks_per_bar: usize,
    current_tick_count: usize,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

impl TickCountBarBuilder {
    pub(super) fn new(ticks_per_bar: u32) -> Self {
        Self {
            ticks_per_bar: ticks_per_bar.max(1) as usize,
            current_tick_count: 0,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    pub(super) fn push_tick(&mut self, ts_ns: i64, price: f64) {
        if self.current_bar.is_none() || self.current_tick_count >= self.ticks_per_bar {
            if let Some(current) = self.current_bar.take() {
                self.bars.push(current);
            }
            self.current_tick_count = 0;
            self.current_bar = Some(new_bar(ts_ns, price, None));
        }

        if let Some(current) = self.current_bar.as_mut() {
            current.ts_ns = ts_ns;
            current.high = current.high.max(price);
            current.low = current.low.min(price);
            current.close = price;
            self.current_tick_count = self.current_tick_count.saturating_add(1);
        }
    }

    pub(super) fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

pub(super) struct VolumeBarBuilder {
    volume_per_bar: f64,
    current_volume: f64,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

impl VolumeBarBuilder {
    pub(super) fn new(volume_per_bar: f64) -> Self {
        Self {
            volume_per_bar: volume_per_bar.max(1.0),
            current_volume: 0.0,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    pub(super) fn push_tick(&mut self, ts_ns: i64, price: f64, size: f64) {
        let mut remaining = size;
        if remaining <= 0.0 {
            return;
        }
        while remaining > 0.0 {
            if self.current_bar.is_none() {
                self.current_bar = Some(new_bar(ts_ns, price, Some(0.0)));
            }
            let capacity = self.volume_per_bar - self.current_volume;
            let consumed = remaining.min(capacity);
            if let Some(current) = self.current_bar.as_mut() {
                current.ts_ns = ts_ns;
                current.high = current.high.max(price);
                current.low = current.low.min(price);
                current.close = price;
                current.volume = Some(self.current_volume + consumed);
            }
            self.current_volume += consumed;
            remaining -= consumed;
            if self.current_volume >= self.volume_per_bar - f64::EPSILON {
                if let Some(current) = self.current_bar.take() {
                    self.bars.push(current);
                }
                self.current_volume = 0.0;
            }
        }
    }

    pub(super) fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

/// Stateful range-bar boundary engine shared by bar aggregation and raw-tick
/// frame construction. A source tick can close more than one range bar; the
/// caller receives each completed bar in source-event order.
///
/// Keeping this state machine in one place is important: replay frames must
/// assign the closing source tick to the same derived-bar boundary that the
/// aggregation pass used. Do not replace this with a high/low-only indexer.
pub(super) struct RangeBarBoundaryTracker {
    range_size: f64,
    current_bar: Option<Bar>,
}

impl RangeBarBoundaryTracker {
    pub(super) fn new(range_size: f64) -> Self {
        Self {
            range_size: range_size.max(f64::EPSILON),
            current_bar: None,
        }
    }

    /// Applies one source tick and invokes `on_complete` once for every range
    /// bar that tick closes. The returned count is deliberately available to
    /// streaming frame assembly, which needs to emit the corresponding empty
    /// synthetic-bar frames after the one frame that owns the source tick.
    pub(super) fn push_tick<F>(&mut self, ts_ns: i64, price: f64, mut on_complete: F) -> usize
    where
        F: FnMut(Bar),
    {
        const EPSILON: f64 = 1e-9;
        let mut completed = 0usize;
        let mut current = self
            .current_bar
            .take()
            .unwrap_or_else(|| new_bar(ts_ns, price, None));

        loop {
            let tentative_high = current.high.max(price);
            let tentative_low = current.low.min(price);
            let breaks_up =
                price > current.high && (price - tentative_low) >= self.range_size - EPSILON;
            let breaks_down =
                price < current.low && (tentative_high - price) >= self.range_size - EPSILON;

            if breaks_up {
                let close = tentative_low + self.range_size;
                current.high = close;
                current.close = close;
                current.ts_ns = ts_ns;
                on_complete(current);
                completed = completed.saturating_add(1);
                current = new_bar(ts_ns, close, None);
                if price <= close + EPSILON {
                    break;
                }
                continue;
            }

            if breaks_down {
                let close = tentative_high - self.range_size;
                current.low = close;
                current.close = close;
                current.ts_ns = ts_ns;
                on_complete(current);
                completed = completed.saturating_add(1);
                current = new_bar(ts_ns, close, None);
                if price >= close - EPSILON {
                    break;
                }
                continue;
            }

            current.high = tentative_high;
            current.low = tentative_low;
            current.close = price;
            current.ts_ns = ts_ns;
            break;
        }

        self.current_bar = Some(current);
        completed
    }

    pub(super) fn finish(mut self) -> Option<Bar> {
        self.current_bar.take()
    }
}

pub(super) struct RangeBarBuilder {
    tracker: RangeBarBoundaryTracker,
    bars: Vec<Bar>,
}

impl RangeBarBuilder {
    pub(super) fn new(range_size: f64) -> Self {
        Self {
            tracker: RangeBarBoundaryTracker::new(range_size),
            bars: Vec::new(),
        }
    }

    pub(super) fn push_tick(&mut self, ts_ns: i64, price: f64) {
        self.tracker
            .push_tick(ts_ns, price, |bar| self.bars.push(bar));
    }

    pub(super) fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.tracker.finish() {
            self.bars.push(current);
        }
        self.bars
    }
}

fn new_bar(ts_ns: i64, price: f64, volume: Option<f64>) -> Bar {
    Bar {
        ts_ns,
        open: price,
        high: price,
        low: price,
        close: price,
        volume,
    }
}
