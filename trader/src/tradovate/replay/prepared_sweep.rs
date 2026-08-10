//! Prepared, replay-only EMA sweep evaluation.
//!
//! This module deliberately does not participate in live execution.  It owns
//! the immutable indicator/protection preparation shared by a `prepared_cpu`
//! sweep and a small deterministic candidate evaluator for cached server bars.
//! Unsupported candidates are routed to the existing service simulator by the
//! sweep runner.

use super::ReplayState;
use super::sweep::ReplaySweepChildSpec;
use crate::broker::{
    Bar, CandleMode, REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION, ReplayBarProtectionPolicy,
    ReplayEngineMode, ReplayExecutionFill, ReplayExecutionLedgerSnapshot, ReplayExecutionPrecision,
    ReplayFillModel, ReplayFillPriceSource, ReplayFrameSet, ReplayLatencyModel,
    ReplaySignalDiagnostic, transform_bars_for_candle_mode,
};
use crate::config::AppConfig;
use crate::strategies::ema_cross::{EmaCrossConfig, EmaCrossExecutionState};
use crate::strategies::hma_cross::{hma_series_incremental, hma_warmup_bars};
use crate::strategy::{NativeExecutionPath, NativeReversalMode, NativeSignalTiming, StrategyKind};
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone)]
pub(crate) struct PreparedEmaSweepInputs {
    pub(crate) frames: Arc<ReplayFrameSet>,
    pub(crate) replay: Arc<ReplayState>,
    /// The service-backed view exposes warmup bars to the replay market
    /// worker, but it starts the strategy's retained market series at the
    /// evaluation boundary.  Keep the raw/frame index so prepared EMA
    /// traces can use the same boundary without copying the frame set.
    pub(crate) signal_start: usize,
    pub(crate) signal_bars: Arc<[Bar]>,
    traces: Arc<BTreeMap<(usize, usize), Arc<EmaCrossTrace>>>,
    hma_traces: Arc<BTreeMap<(usize, usize), Arc<EmaCrossTrace>>>,
    pub(crate) protection: Arc<BarProtectionIndex>,
}

#[derive(Debug, Clone)]
struct EmaCrossTrace {
    fast_length: usize,
    slow_length: usize,
    previous_fast: Arc<[f64]>,
    previous_slow: Arc<[f64]>,
    fast: Arc<[f64]>,
    slow: Arc<[f64]>,
    raw_buy: Arc<[bool]>,
    raw_sell: Arc<[bool]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtectionSide {
    Long,
    Short,
}

impl ProtectionSide {
    fn from_qty(qty: i32) -> Option<Self> {
        match qty.signum() {
            1 => Some(Self::Long),
            -1 => Some(Self::Short),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtectionReason {
    TakeProfit,
    StopLoss,
}

impl ProtectionReason {
    fn label(self) -> &'static str {
        match self {
            Self::TakeProfit => "take_profit",
            Self::StopLoss => "stop_loss",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ProtectionHit {
    index: usize,
    reason: ProtectionReason,
    price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProtectionQueryKey {
    entry_index: usize,
    side: ProtectionSideKey,
    take_profit_bits: u64,
    stop_loss_bits: u64,
    policy: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ProtectionSideKey {
    Long,
    Short,
}

impl From<ProtectionSide> for ProtectionSideKey {
    fn from(value: ProtectionSide) -> Self {
        match value {
            ProtectionSide::Long => Self::Long,
            ProtectionSide::Short => Self::Short,
        }
    }
}

fn protection_policy_key(policy: ReplayBarProtectionPolicy) -> u8 {
    match policy {
        ReplayBarProtectionPolicy::Conservative => 0,
        ReplayBarProtectionPolicy::Optimistic => 1,
        ReplayBarProtectionPolicy::NearestOpen => 2,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BarProtectionIndex {
    opens: Arc<[f64]>,
    highs: Arc<[f64]>,
    lows: Arc<[f64]>,
    max_tree: Arc<[f64]>,
    min_tree: Arc<[f64]>,
    tree_base: usize,
    cache: Arc<Mutex<HashMap<ProtectionQueryKey, Option<ProtectionHit>>>>,
}

impl BarProtectionIndex {
    fn new(bars: &[Bar]) -> Self {
        let opens = bars.iter().map(|bar| bar.open).collect::<Vec<_>>();
        let highs = bars.iter().map(|bar| bar.high).collect::<Vec<_>>();
        let lows = bars.iter().map(|bar| bar.low).collect::<Vec<_>>();
        let tree_base = bars.len().max(1).next_power_of_two();
        let mut max_tree = vec![f64::NEG_INFINITY; tree_base * 2];
        let mut min_tree = vec![f64::INFINITY; tree_base * 2];
        for (index, (&high, &low)) in highs.iter().zip(lows.iter()).enumerate() {
            max_tree[tree_base + index] = if high.is_finite() {
                high
            } else {
                f64::NEG_INFINITY
            };
            min_tree[tree_base + index] = if low.is_finite() { low } else { f64::INFINITY };
        }
        for index in (1..tree_base).rev() {
            max_tree[index] = max_tree[index * 2].max(max_tree[index * 2 + 1]);
            min_tree[index] = min_tree[index * 2].min(min_tree[index * 2 + 1]);
        }
        Self {
            opens: Arc::from(opens.into_boxed_slice()),
            highs: Arc::from(highs.into_boxed_slice()),
            lows: Arc::from(lows.into_boxed_slice()),
            max_tree: Arc::from(max_tree.into_boxed_slice()),
            min_tree: Arc::from(min_tree.into_boxed_slice()),
            tree_base,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn first_fixed_hit(
        &self,
        entry_index: usize,
        side: ProtectionSide,
        take_profit_price: Option<f64>,
        stop_loss_price: Option<f64>,
        policy: ReplayBarProtectionPolicy,
    ) -> Option<ProtectionHit> {
        let key = ProtectionQueryKey {
            entry_index,
            side: side.into(),
            take_profit_bits: take_profit_price.map(f64::to_bits).unwrap_or_default(),
            stop_loss_bits: stop_loss_price.map(f64::to_bits).unwrap_or_default(),
            policy: protection_policy_key(policy),
        };
        if let Ok(cache) = self.cache.lock() {
            if let Some(hit) = cache.get(&key) {
                return *hit;
            }
        }

        let target_index = take_profit_price.and_then(|price| match side {
            ProtectionSide::Long => self.first_high_at_or_above(entry_index, price),
            ProtectionSide::Short => self.first_low_at_or_below(entry_index, price),
        });
        let stop_index = stop_loss_price.and_then(|price| match side {
            ProtectionSide::Long => self.first_low_at_or_below(entry_index, price),
            ProtectionSide::Short => self.first_high_at_or_above(entry_index, price),
        });
        let hit = choose_fixed_hit(
            policy,
            target_index.map(|index| (index, ProtectionReason::TakeProfit, take_profit_price)),
            stop_index.map(|index| (index, ProtectionReason::StopLoss, stop_loss_price)),
            &self.opens,
        );
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, hit);
        }
        hit
    }

    fn first_high_at_or_above(&self, from: usize, threshold: f64) -> Option<usize> {
        if !threshold.is_finite() || from >= self.highs.len() {
            return None;
        }
        first_tree_match(&self.max_tree, 1, 0, self.tree_base, from, threshold, true)
            .filter(|index| *index < self.highs.len())
    }

    fn first_low_at_or_below(&self, from: usize, threshold: f64) -> Option<usize> {
        if !threshold.is_finite() || from >= self.lows.len() {
            return None;
        }
        first_tree_match(&self.min_tree, 1, 0, self.tree_base, from, threshold, false)
            .filter(|index| *index < self.lows.len())
    }
}

fn first_tree_match(
    tree: &[f64],
    node: usize,
    left: usize,
    right: usize,
    from: usize,
    threshold: f64,
    is_max: bool,
) -> Option<usize> {
    if right <= from {
        return None;
    }
    let aggregate = tree.get(node).copied().unwrap_or(if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    });
    let matches = if is_max {
        aggregate >= threshold
    } else {
        aggregate <= threshold
    };
    if !matches {
        return None;
    }
    if right.saturating_sub(left) <= 1 {
        return Some(left);
    }
    let mid = left + (right - left) / 2;
    first_tree_match(tree, node * 2, left, mid, from, threshold, is_max)
        .or_else(|| first_tree_match(tree, node * 2 + 1, mid, right, from, threshold, is_max))
}

fn choose_fixed_hit(
    policy: ReplayBarProtectionPolicy,
    target: Option<(usize, ProtectionReason, Option<f64>)>,
    stop: Option<(usize, ProtectionReason, Option<f64>)>,
    opens: &[f64],
) -> Option<ProtectionHit> {
    let target =
        target.and_then(|(index, reason, price)| price.map(|price| (index, reason, price)));
    let stop = stop.and_then(|(index, reason, price)| price.map(|price| (index, reason, price)));
    match (target, stop) {
        (None, None) => None,
        (Some((index, reason, price)), None) | (None, Some((index, reason, price))) => {
            Some(ProtectionHit {
                index,
                reason,
                price,
            })
        }
        (Some(target), Some(stop)) if target.0 < stop.0 => Some(ProtectionHit {
            index: target.0,
            reason: target.1,
            price: target.2,
        }),
        (Some(target), Some(stop)) if stop.0 < target.0 => Some(ProtectionHit {
            index: stop.0,
            reason: stop.1,
            price: stop.2,
        }),
        (Some(target), Some(stop)) => {
            let index = target.0;
            let reason = match policy {
                ReplayBarProtectionPolicy::Conservative => ProtectionReason::StopLoss,
                ReplayBarProtectionPolicy::Optimistic => ProtectionReason::TakeProfit,
                ReplayBarProtectionPolicy::NearestOpen => {
                    let open = opens.get(index).copied().unwrap_or_default();
                    let target_distance = (open - target.2).abs();
                    let stop_distance = (open - stop.2).abs();
                    if target_distance + EPSILON < stop_distance {
                        ProtectionReason::TakeProfit
                    } else {
                        ProtectionReason::StopLoss
                    }
                }
            };
            let price = if reason == target.1 { target.2 } else { stop.2 };
            Some(ProtectionHit {
                index,
                reason,
                price,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSweepRun {
    pub(crate) ledger: ReplayExecutionLedgerSnapshot,
    pub(crate) diagnostics: Vec<ReplaySignalDiagnostic>,
    pub(crate) history_loaded: usize,
    pub(crate) evaluation_rows_total: usize,
    pub(crate) evaluation_rows_processed: usize,
}

pub(crate) fn prepare_ema_sweep_inputs(
    replay: Arc<ReplayState>,
    frames: Arc<ReplayFrameSet>,
    children: &[ReplaySweepChildSpec],
) -> Result<PreparedEmaSweepInputs> {
    if frames.bars.is_empty() {
        bail!("prepared replay sweep requires at least one server bar");
    }
    let mut periods = HashSet::new();
    for child in children {
        if child.resolved_strategy.kind == StrategyKind::Native
            && child.resolved_strategy.native_strategy
                == crate::strategy::NativeStrategyKind::EmaCross
        {
            periods.insert(child.resolved_strategy.native_ema.fast_length.max(1));
            periods.insert(child.resolved_strategy.native_ema.slow_length.max(1));
        }
    }
    let mut hma_periods = HashSet::new();
    for child in children {
        if child.resolved_strategy.kind == StrategyKind::Native
            && child.resolved_strategy.native_strategy
                == crate::strategy::NativeStrategyKind::HmaCross
        {
            hma_periods.insert(child.resolved_strategy.native_hma_cross.fast_length.max(1));
            hma_periods.insert(child.resolved_strategy.native_hma_cross.slow_length.max(1));
        }
    }
    if periods.is_empty() && hma_periods.is_empty() {
        bail!("prepared replay sweep contains no native EMA or HMA candidates");
    }

    let signal_bars: Arc<[Bar]> = Arc::from(
        transform_bars_for_candle_mode(&frames.bars, frames.candle_mode).into_boxed_slice(),
    );
    let signal_start = replay
        .evaluation_start_ns()
        .ok()
        .flatten()
        .map(|start_ns| frames.bars.partition_point(|bar| bar.ts_ns < start_ns))
        .unwrap_or(0);
    let mut ema_series: BTreeMap<usize, Arc<[f64]>> = BTreeMap::new();
    for period in periods {
        ema_series.insert(
            period,
            Arc::from(build_ema_series(&signal_bars, period, signal_start).into_boxed_slice()),
        );
    }
    let ema_series = Arc::new(ema_series);
    let mut traces = BTreeMap::new();
    let mut pairs = HashSet::new();
    for child in children {
        if child.resolved_strategy.kind == StrategyKind::Native
            && child.resolved_strategy.native_strategy
                == crate::strategy::NativeStrategyKind::EmaCross
        {
            let config = &child.resolved_strategy.native_ema;
            pairs.insert((config.fast_length.max(1), config.slow_length.max(1)));
        }
    }
    for (fast_length, slow_length) in pairs {
        let fast = ema_series
            .get(&fast_length)
            .context("prepared EMA fast series missing")?
            .clone();
        let slow = ema_series
            .get(&slow_length)
            .context("prepared EMA slow series missing")?
            .clone();
        traces.insert(
            (fast_length, slow_length),
            Arc::new(build_trace(
                fast_length,
                slow_length,
                fast,
                slow,
                signal_start,
                None,
            )),
        );
    }
    let mut hma_series: BTreeMap<usize, Arc<[f64]>> = BTreeMap::new();
    for period in hma_periods {
        let start = signal_start.min(signal_bars.len());
        let closes = signal_bars[start..]
            .iter()
            .map(|bar| bar.close)
            .collect::<Vec<_>>();
        let local = hma_series_incremental(&closes, period);
        let mut values = vec![f64::NAN; signal_bars.len()];
        values[start..].copy_from_slice(&local);
        hma_series.insert(period, Arc::from(values.into_boxed_slice()));
    }
    let hma_series = Arc::new(hma_series);
    let mut hma_traces = BTreeMap::new();
    let mut hma_pairs = HashSet::new();
    for child in children {
        if child.resolved_strategy.kind == StrategyKind::Native
            && child.resolved_strategy.native_strategy
                == crate::strategy::NativeStrategyKind::HmaCross
        {
            let config = &child.resolved_strategy.native_hma_cross;
            hma_pairs.insert((config.fast_length.max(1), config.slow_length.max(1)));
        }
    }
    for (fast_length, slow_length) in hma_pairs {
        let fast = hma_series
            .get(&fast_length)
            .context("prepared HMA fast series missing")?
            .clone();
        let slow = hma_series
            .get(&slow_length)
            .context("prepared HMA slow series missing")?
            .clone();
        let warmup = hma_warmup_bars(fast_length.max(1))
            .max(hma_warmup_bars(slow_length.max(1)))
            .saturating_add(1);
        hma_traces.insert(
            (fast_length, slow_length),
            Arc::new(build_trace(
                fast_length,
                slow_length,
                fast,
                slow,
                signal_start,
                Some(warmup),
            )),
        );
    }
    Ok(PreparedEmaSweepInputs {
        frames: frames.clone(),
        replay,
        signal_start,
        signal_bars,
        traces: Arc::new(traces),
        hma_traces: Arc::new(hma_traces),
        protection: Arc::new(BarProtectionIndex::new(&frames.bars)),
    })
}

fn build_ema_series(bars: &[Bar], period: usize, signal_start: usize) -> Vec<f64> {
    let config = EmaCrossConfig {
        fast_length: period,
        slow_length: period,
        ..EmaCrossConfig::default()
    };
    let mut runtime = EmaCrossExecutionState::default();
    let mut values = vec![f64::NAN; bars.len()];
    let signal_start = signal_start.min(bars.len());
    for end in (signal_start + 1)..=bars.len() {
        let start = signal_start.max(end.saturating_sub(crate::tradovate::ENGINE_MARKET_BAR_LIMIT));
        // Supplying a monotonic append sequence avoids the defensive
        // retained-window fingerprint scan in the hint-free evaluator.  The
        // streaming state still performs the exact finite-window slide when
        // the engine bar limit is reached, so this remains parity-safe while
        // making preparation O(rows) per EMA period instead of O(rows *
        // retained_window).
        let evaluation = config.evaluate_streaming_with_market_update(
            &mut runtime,
            &bars[start..end],
            None,
            Some(end as u64),
            crate::broker::MarketHistoryUpdate::Append,
        );
        values[end - 1] = evaluation.fast_ema.unwrap_or(f64::NAN);
    }
    values
}

fn build_trace(
    fast_length: usize,
    slow_length: usize,
    fast: Arc<[f64]>,
    slow: Arc<[f64]>,
    signal_start: usize,
    warmup_override: Option<usize>,
) -> EmaCrossTrace {
    let mut previous_fast = vec![f64::NAN; fast.len()];
    let mut previous_slow = vec![f64::NAN; slow.len()];
    let mut raw_buy = vec![false; fast.len()];
    let mut raw_sell = vec![false; fast.len()];
    let warmup = warmup_override.unwrap_or_else(|| fast_length.max(slow_length).max(2) + 1);
    let signal_start = signal_start.min(fast.len().min(slow.len()));
    for index in (signal_start + 1)..fast.len().min(slow.len()) {
        previous_fast[index] = fast[index - 1];
        previous_slow[index] = slow[index - 1];
        if index.saturating_sub(signal_start) + 1 < warmup {
            continue;
        }
        let (pf, ps, cf, cs) = (
            previous_fast[index],
            previous_slow[index],
            fast[index],
            slow[index],
        );
        if pf.is_finite() && ps.is_finite() && cf.is_finite() && cs.is_finite() {
            raw_buy[index] = pf <= ps && cf > cs;
            raw_sell[index] = pf >= ps && cf < cs;
        }
    }
    EmaCrossTrace {
        fast_length,
        slow_length,
        previous_fast: Arc::from(previous_fast.into_boxed_slice()),
        previous_slow: Arc::from(previous_slow.into_boxed_slice()),
        fast,
        slow,
        raw_buy: Arc::from(raw_buy.into_boxed_slice()),
        raw_sell: Arc::from(raw_sell.into_boxed_slice()),
    }
}

impl PreparedEmaSweepInputs {
    pub(crate) fn supports(
        &self,
        replay: &ReplayState,
        config: &AppConfig,
        child: &ReplaySweepChildSpec,
    ) -> Result<(), String> {
        if !replay.is_cached_server_bars() {
            return Err("dataset is not cached server bars".to_string());
        }
        if child.engine_mode != ReplayEngineMode::Deterministic {
            return Err("prepared kernel requires deterministic replay".to_string());
        }
        if child.fill_model != ReplayFillModel::RawBarOpen {
            return Err("prepared kernel currently requires raw_bar_open fills".to_string());
        }
        if child.latency.model != ReplayLatencyModel::Fixed {
            return Err("prepared kernel currently requires fixed latency".to_string());
        }
        if child.latency.fixed_latency_ms != 0 {
            return Err(
                "prepared kernel currently requires zero fixed latency; nonzero latency uses the reference simulator"
                    .to_string(),
            );
        }
        if child.resolved_strategy.kind != StrategyKind::Native
            || child.resolved_strategy.native_strategy
                != crate::strategy::NativeStrategyKind::EmaCross
        {
            return Err("prepared kernel currently supports native EMA only".to_string());
        }
        if child.resolved_strategy.native_signal_timing != NativeSignalTiming::ClosedBar {
            return Err("prepared kernel currently requires closed-bar signals".to_string());
        }
        if child.resolved_strategy.native_signal_delay_bars != 0 {
            return Err(
                "prepared kernel currently requires zero signal delay bars; delayed signals use the reference simulator"
                    .to_string(),
            );
        }
        if child.resolved_strategy.native_execution_path != NativeExecutionPath::Guarded {
            return Err("prepared kernel currently requires the guarded native path".to_string());
        }
        if child.resolved_strategy.native_reversal_mode == NativeReversalMode::FlattenConfirmEnter {
            return Err(
                "prepared kernel does not emulate staged flatten-confirm-enter transitions; use the reference simulator"
                    .to_string(),
            );
        }
        if child.margin.is_some() || config.replay_liquidation_enabled {
            return Err("margin/liquidation candidates use the reference simulator".to_string());
        }
        if replay
            .market_tick_size()
            .is_none_or(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("prepared kernel requires a valid market tick size".to_string());
        }
        let ema = &child.resolved_strategy.native_ema;
        if ema.use_trailing_stop
            && (ema.trail_trigger_ticks <= 0.0 || ema.trail_offset_ticks <= 0.0)
        {
            return Err("prepared kernel requires valid trailing-stop offsets".to_string());
        }
        if child.candle_mode != CandleMode::Standard && child.candle_mode != CandleMode::HeikinAshi
        {
            return Err("unsupported candle mode".to_string());
        }
        Ok(())
    }
}

/// The prepared kernel's execution/ledger code is indicator-agnostic once a
/// crossover trace has been materialized.  Reuse that kernel for HMA by
/// mapping the HMA protection fields into the existing EMA-shaped config;
/// this adapter is replay-only and never reaches live strategy dispatch.
pub(crate) fn prepared_hma_child_as_ema(child: &ReplaySweepChildSpec) -> ReplaySweepChildSpec {
    let hma = &child.resolved_strategy.native_hma_cross;
    let mut prepared = child.clone();
    prepared.resolved_strategy.native_strategy = crate::strategy::NativeStrategyKind::EmaCross;
    prepared.resolved_strategy.native_ema = EmaCrossConfig {
        fast_length: hma.fast_length,
        slow_length: hma.slow_length,
        inverted: hma.inverted,
        take_profit_ticks: hma.take_profit_ticks,
        stop_loss_ticks: hma.stop_loss_ticks,
        use_trailing_stop: hma.use_trailing_stop,
        trail_trigger_ticks: hma.trail_trigger_ticks,
        trail_offset_ticks: hma.trail_offset_ticks,
    };
    prepared
}

#[derive(Debug, Clone, Copy)]
struct KernelPosition {
    qty: i32,
    entry_price: f64,
    protection: Option<KernelProtection>,
}

#[derive(Debug, Clone, Copy)]
struct KernelProtection {
    side: ProtectionSide,
    order_strategy_id: Option<i64>,
    take_profit_price: Option<f64>,
    current_stop_price: Option<f64>,
    trail_trigger_ticks: f64,
    trail_offset_ticks: f64,
    trail_frequency: f64,
    best_price: f64,
    trailing_active: bool,
    fixed_hit: Option<ProtectionHit>,
}

#[derive(Debug, Clone, Copy)]
enum PendingTransitionKind {
    Single,
    AtomicReverse,
    StagedFlatten,
    StagedEntry,
}

#[derive(Debug, Clone, Copy)]
struct PendingTransition {
    kind: PendingTransitionKind,
    target_qty: i32,
    submitted_index: usize,
    signal_index: usize,
    signal_ts_ns: i64,
    arrival_ts_ns: i64,
}

#[derive(Debug, Default)]
struct KernelState {
    position: Option<KernelPosition>,
    pending: Option<PendingTransition>,
    /// Closed-bar guarded execution remembers the last entry side that was
    /// dispatched while flat.  If that order is later flattened by a
    /// protection or blockout event, another same-side crossover is consumed
    /// until the opposite entry signal appears.  Keep this state in the
    /// prepared kernel so it follows the reference service's gate exactly.
    last_dispatched_entry_signal: Option<i8>,
    fills: Vec<ReplayExecutionFill>,
    diagnostics: Vec<ReplaySignalDiagnostic>,
    next_fill_id: i64,
    next_order_id: i64,
    next_strategy_id: i64,
    next_lifecycle_sequence: u64,
}

impl KernelState {
    fn new() -> Self {
        Self {
            next_fill_id: 10_000,
            next_order_id: 1_000,
            next_strategy_id: 40_000,
            next_lifecycle_sequence: 1,
            ..Self::default()
        }
    }

    fn next_fill_id(&mut self) -> i64 {
        self.next_fill_id = self.next_fill_id.saturating_add(1).max(10_000);
        self.next_fill_id
    }

    fn next_order_id(&mut self) -> i64 {
        self.next_order_id = self.next_order_id.saturating_add(1).max(1_000);
        self.next_order_id
    }

    fn next_strategy_id(&mut self) -> i64 {
        self.next_strategy_id = self.next_strategy_id.saturating_add(1).max(40_000);
        self.next_strategy_id
    }

    fn next_lifecycle_sequence(&mut self) -> u64 {
        let next = self.next_lifecycle_sequence;
        self.next_lifecycle_sequence = self.next_lifecycle_sequence.saturating_add(1);
        next
    }
}

pub(crate) fn run_prepared_ema_candidate(
    inputs: &PreparedEmaSweepInputs,
    replay: &ReplayState,
    config: &AppConfig,
    child: &ReplaySweepChildSpec,
) -> Result<PreparedSweepRun> {
    inputs
        .supports(replay, config, child)
        .map_err(|reason| anyhow::anyhow!(reason))?;
    let strategy = &child.resolved_strategy;
    let ema = &strategy.native_ema;
    let trace = inputs
        .traces
        .get(&(ema.fast_length.max(1), ema.slow_length.max(1)))
        .context("prepared EMA crossover trace missing")?;
    let tick_size = replay
        .market_tick_size()
        .filter(|value| value.is_finite() && *value > 0.0)
        .context("prepared EMA replay requires a valid tick size")?;
    if ema.use_trailing_stop && (ema.trail_trigger_ticks <= 0.0 || ema.trail_offset_ticks <= 0.0) {
        bail!("prepared EMA replay requires valid trailing-stop offsets");
    }

    let bars = &inputs.frames.bars;
    let signal_bars = &inputs.signal_bars;
    let history_loaded = if replay.evaluation_start_ns().ok().flatten().is_some() {
        inputs.signal_start
    } else {
        prepared_history_loaded(replay, config, bars)
    };
    let diagnostic_history_offset = replay.replay_window.as_ref().map_or(0, |_| history_loaded);
    let evaluation_rows_total = bars.len().saturating_sub(history_loaded);
    let latency_ns = (child.latency.fixed_latency_ms as i64).saturating_mul(1_000_000);
    let mut state = KernelState::new();
    let mut anchored = false;
    let mut processed = 0usize;

    for bar_index in 0..bars.len() {
        let bar = &bars[bar_index];
        execute_pending_transition(
            &mut state,
            bar_index,
            bar,
            strategy,
            child,
            replay,
            tick_size,
            latency_ns,
            &inputs.protection,
        )?;
        if let Some(position) = state.position
            && let Some(protection) = position.protection
        {
            if let Some(hit) = protection_hit_for_bar(
                protection,
                position,
                bar_index,
                bar,
                child.bar_protection_policy,
            ) {
                append_fill(
                    &mut state,
                    replay,
                    child,
                    bar,
                    if position.qty > 0 { "Sell" } else { "Buy" },
                    position.qty.abs(),
                    hit.price,
                    None,
                    Some(hit.reason.label()),
                    protection.order_strategy_id,
                    0,
                    ReplayFillPriceSource::RawBarOhlc,
                );
                state.position = None;
            } else if let Some(position) = state.position.as_mut() {
                update_trailing_stop(position, bar, tick_size);
            }
        }

        if bar_index < history_loaded {
            continue;
        }
        processed = processed.saturating_add(1);
        let Some(signal_index) = bar_index.checked_sub(strategy.native_signal_delay_bars) else {
            continue;
        };
        if signal_index >= trace.raw_buy.len() {
            continue;
        }
        if !anchored {
            anchored = true;
            continue;
        }
        if signal_index < history_loaded {
            continue;
        }

        let current_qty = state
            .position
            .map(|position| position.qty)
            .unwrap_or_default();
        let raw_buy = trace.raw_buy[signal_index];
        let raw_sell = trace.raw_sell[signal_index];
        let (effective_buy, effective_sell) = if ema.inverted {
            (raw_sell, raw_buy)
        } else {
            (raw_buy, raw_sell)
        };
        let signal = if effective_buy && current_qty <= 0 {
            Some(1)
        } else if effective_sell && current_qty >= 0 {
            Some(-1)
        } else {
            None
        };
        let target_qty = signal.map(|direction| direction * strategy.order_qty.max(1));
        let session_window = strategy
            .blockout_enabled
            .then(|| {
                replay.market_session_profile().map(|profile| {
                    profile.evaluate_with_blockout(
                        signal_bars[signal_index].ts_ns,
                        strategy.blockout_minutes_before_close,
                    )
                })
            })
            .flatten();

        if session_window.is_some_and(|window| window.hold_entries) {
            if current_qty != 0 && state.pending.is_none() {
                schedule_transition(
                    &mut state,
                    PendingTransitionKind::Single,
                    0,
                    bar_index,
                    signal_bars[signal_index].ts_ns,
                    bar_index,
                    latency_ns,
                );
                append_diagnostic(
                    &mut state,
                    trace,
                    signal_bars,
                    signal_index,
                    diagnostic_history_offset,
                    current_qty,
                    strategy,
                    Some(0),
                    "dispatching",
                    "session hold flattening",
                );
            } else {
                append_diagnostic(
                    &mut state,
                    trace,
                    signal_bars,
                    signal_index,
                    diagnostic_history_offset,
                    current_qty,
                    strategy,
                    None,
                    "blocked",
                    "session hold blocks entries",
                );
            }
            continue;
        }

        let Some(target_qty) = target_qty else {
            append_diagnostic(
                &mut state,
                trace,
                signal_bars,
                signal_index,
                diagnostic_history_offset,
                current_qty,
                strategy,
                None,
                "no_target",
                "no actionable crossover",
            );
            continue;
        };
        if current_qty == 0 && state.last_dispatched_entry_signal == Some(target_qty.signum() as i8)
        {
            append_diagnostic(
                &mut state,
                trace,
                signal_bars,
                signal_index,
                diagnostic_history_offset,
                current_qty,
                strategy,
                Some(target_qty),
                "flat_entry_already_consumed",
                "entry side was already consumed while flat",
            );
            continue;
        }
        let pending = state.pending.is_some();
        if target_qty == current_qty || pending {
            append_diagnostic(
                &mut state,
                trace,
                signal_bars,
                signal_index,
                diagnostic_history_offset,
                current_qty,
                strategy,
                Some(target_qty),
                "blocked",
                if pending {
                    "waiting for pending transition"
                } else {
                    "target already current"
                },
            );
            continue;
        }

        let reversal_mode = normalized_reversal_mode(strategy, ema.uses_native_protection());
        let kind = if current_qty != 0 && target_qty != 0 {
            match reversal_mode {
                NativeReversalMode::FlattenConfirmEnter => PendingTransitionKind::StagedFlatten,
                NativeReversalMode::CloseAllEnter | NativeReversalMode::Direct => {
                    PendingTransitionKind::AtomicReverse
                }
            }
        } else {
            PendingTransitionKind::Single
        };
        schedule_transition(
            &mut state,
            kind,
            target_qty,
            bar_index,
            signal_bars[signal_index].ts_ns,
            bar_index,
            latency_ns,
        );
        state.last_dispatched_entry_signal = Some(target_qty.signum() as i8);
        append_diagnostic(
            &mut state,
            trace,
            signal_bars,
            signal_index,
            diagnostic_history_offset,
            current_qty,
            strategy,
            Some(target_qty),
            "dispatching",
            "crossover passed prepared execution gates",
        );
    }

    let gross_realized_pnl = state
        .fills
        .iter()
        .filter_map(|fill| fill.gross_realized_pnl_delta)
        .sum();
    let ledger = ReplayExecutionLedgerSnapshot {
        schema_version: REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION,
        fee_neutral: true,
        engine_mode: child.engine_mode,
        fill_model: child.fill_model,
        latency_model: child.latency.model,
        fixed_latency_ms: child.latency.fixed_latency_ms,
        latency_seed: None,
        observed_latency_sample_count: 0,
        bar_protection_policy: child.bar_protection_policy,
        signal_source: child.bar_type.mode_label(child.candle_mode),
        gross_realized_pnl,
        fills: state.fills,
    };
    Ok(PreparedSweepRun {
        ledger,
        diagnostics: state.diagnostics,
        history_loaded,
        evaluation_rows_total,
        evaluation_rows_processed: processed.min(evaluation_rows_total),
    })
}

/// Run one HMA crossover candidate through the replay-only prepared kernel.
/// HMA signals come from the immutable incremental traces prepared above; the
/// fill, session blockout, and ledger behavior is shared with the established
/// prepared EMA implementation.
pub(crate) fn run_prepared_hma_candidate(
    inputs: &PreparedEmaSweepInputs,
    replay: &ReplayState,
    config: &AppConfig,
    child: &ReplaySweepChildSpec,
) -> Result<PreparedSweepRun> {
    let hma = &child.resolved_strategy.native_hma_cross;
    inputs
        .hma_traces
        .get(&(hma.fast_length.max(1), hma.slow_length.max(1)))
        .context("prepared HMA crossover trace missing")?;
    let prepared_child = prepared_hma_child_as_ema(child);
    let mut prepared_inputs = inputs.clone();
    prepared_inputs.traces = inputs.hma_traces.clone();
    run_prepared_ema_candidate(&prepared_inputs, replay, config, &prepared_child)
}

fn prepared_history_loaded(replay: &ReplayState, config: &AppConfig, bars: &[Bar]) -> usize {
    if let Ok(Some(start_ns)) = replay.evaluation_start_ns() {
        return bars.partition_point(|bar| bar.ts_ns < start_ns);
    }
    if bars.len() > 1 {
        config.history_bars.max(1).min(bars.len() - 1)
    } else {
        bars.len()
    }
}

fn normalized_reversal_mode(
    strategy: &crate::strategy::ExecutionStrategyConfig,
    uses_protection: bool,
) -> NativeReversalMode {
    if uses_protection && strategy.native_reversal_mode == NativeReversalMode::Direct {
        NativeReversalMode::CloseAllEnter
    } else {
        strategy.native_reversal_mode
    }
}

fn schedule_transition(
    state: &mut KernelState,
    kind: PendingTransitionKind,
    target_qty: i32,
    submitted_index: usize,
    signal_ts_ns: i64,
    current_index: usize,
    latency_ns: i64,
) {
    let arrival_ts_ns = signal_ts_ns.saturating_add(latency_ns.max(0));
    state.pending = Some(PendingTransition {
        kind,
        target_qty,
        submitted_index: current_index.max(submitted_index),
        signal_index: submitted_index,
        signal_ts_ns,
        arrival_ts_ns,
    });
}

fn execute_pending_transition(
    state: &mut KernelState,
    bar_index: usize,
    bar: &Bar,
    strategy: &crate::strategy::ExecutionStrategyConfig,
    child: &ReplaySweepChildSpec,
    replay: &ReplayState,
    tick_size: f64,
    latency_ns: i64,
    protection_index: &BarProtectionIndex,
) -> Result<()> {
    let Some(pending) = state.pending else {
        return Ok(());
    };
    if bar_index <= pending.submitted_index || bar.ts_ns < pending.arrival_ts_ns {
        return Ok(());
    }
    match pending.kind {
        PendingTransitionKind::StagedFlatten => {
            if state.position.is_some() {
                let qty = state
                    .position
                    .map(|position| position.qty)
                    .unwrap_or_default();
                append_fill(
                    state,
                    replay,
                    child,
                    bar,
                    if qty > 0 { "Sell" } else { "Buy" },
                    qty.abs(),
                    bar.open,
                    Some(pending.signal_ts_ns),
                    None,
                    None,
                    child.latency.fixed_latency_ms,
                    ReplayFillPriceSource::RawBarOpen,
                );
                state.position = None;
                state.pending = Some(PendingTransition {
                    kind: PendingTransitionKind::StagedEntry,
                    target_qty: pending.target_qty,
                    submitted_index: bar_index,
                    signal_index: pending.signal_index,
                    signal_ts_ns: pending.signal_ts_ns,
                    arrival_ts_ns: bar.ts_ns.saturating_add(latency_ns.max(0)),
                });
            } else {
                state.pending = Some(PendingTransition {
                    kind: PendingTransitionKind::StagedEntry,
                    target_qty: pending.target_qty,
                    submitted_index: bar_index,
                    signal_index: pending.signal_index,
                    signal_ts_ns: pending.signal_ts_ns,
                    arrival_ts_ns: bar.ts_ns.saturating_add(latency_ns.max(0)),
                });
            }
        }
        PendingTransitionKind::StagedEntry => {
            fill_target_position(
                state,
                replay,
                child,
                strategy,
                protection_index,
                bar,
                bar_index,
                pending.target_qty,
                pending.signal_ts_ns,
                tick_size,
            )?;
            state.pending = None;
        }
        PendingTransitionKind::Single => {
            fill_target_position(
                state,
                replay,
                child,
                strategy,
                protection_index,
                bar,
                bar_index,
                pending.target_qty,
                pending.signal_ts_ns,
                tick_size,
            )?;
            state.pending = None;
        }
        PendingTransitionKind::AtomicReverse => {
            let old_qty = state
                .position
                .map(|position| position.qty)
                .unwrap_or_default();
            if old_qty != 0 {
                append_fill(
                    state,
                    replay,
                    child,
                    bar,
                    if old_qty > 0 { "Sell" } else { "Buy" },
                    old_qty.abs(),
                    bar.open,
                    Some(pending.signal_ts_ns),
                    None,
                    None,
                    child.latency.fixed_latency_ms,
                    ReplayFillPriceSource::RawBarOpen,
                );
                state.position = None;
            }
            fill_target_position(
                state,
                replay,
                child,
                strategy,
                protection_index,
                bar,
                bar_index,
                pending.target_qty,
                pending.signal_ts_ns,
                tick_size,
            )?;
            state.pending = None;
        }
    }
    Ok(())
}

fn fill_target_position(
    state: &mut KernelState,
    replay: &ReplayState,
    child: &ReplaySweepChildSpec,
    strategy: &crate::strategy::ExecutionStrategyConfig,
    protection_index: &BarProtectionIndex,
    bar: &Bar,
    bar_index: usize,
    target_qty: i32,
    signal_ts_ns: i64,
    tick_size: f64,
) -> Result<()> {
    let current_qty = state
        .position
        .map(|position| position.qty)
        .unwrap_or_default();
    if target_qty == 0 {
        if current_qty != 0 {
            append_fill(
                state,
                replay,
                child,
                bar,
                if current_qty > 0 { "Sell" } else { "Buy" },
                current_qty.abs(),
                bar.open,
                Some(signal_ts_ns),
                None,
                None,
                child.latency.fixed_latency_ms,
                ReplayFillPriceSource::RawBarOpen,
            );
            state.position = None;
        }
        return Ok(());
    }
    if current_qty != 0 && current_qty.signum() != target_qty.signum() {
        append_fill(
            state,
            replay,
            child,
            bar,
            if current_qty > 0 { "Sell" } else { "Buy" },
            current_qty.abs(),
            bar.open,
            Some(signal_ts_ns),
            None,
            None,
            child.latency.fixed_latency_ms,
            ReplayFillPriceSource::RawBarOpen,
        );
        state.position = None;
    }
    let order_strategy_id = strategy
        .native_ema
        .uses_native_protection()
        .then(|| state.next_strategy_id());
    append_fill(
        state,
        replay,
        child,
        bar,
        if target_qty > 0 { "Buy" } else { "Sell" },
        target_qty.abs(),
        bar.open,
        Some(signal_ts_ns),
        None,
        order_strategy_id,
        child.latency.fixed_latency_ms,
        ReplayFillPriceSource::RawBarOpen,
    );
    let side = ProtectionSide::from_qty(target_qty).context("target position has no side")?;
    let protection = make_protection(
        strategy,
        side,
        bar.open,
        bar_index,
        tick_size,
        child.bar_protection_policy,
        protection_index,
        order_strategy_id,
    );
    state.position = Some(KernelPosition {
        qty: target_qty,
        entry_price: bar.open,
        protection,
    });
    Ok(())
}

fn make_protection(
    strategy: &crate::strategy::ExecutionStrategyConfig,
    side: ProtectionSide,
    entry_price: f64,
    entry_index: usize,
    tick_size: f64,
    policy: ReplayBarProtectionPolicy,
    protection_index: &BarProtectionIndex,
    order_strategy_id: Option<i64>,
) -> Option<KernelProtection> {
    let config = &strategy.native_ema;
    if !config.uses_native_protection() {
        return None;
    }
    let signed = |ticks: f64| match side {
        ProtectionSide::Long => entry_price + ticks * tick_size,
        ProtectionSide::Short => entry_price - ticks * tick_size,
    };
    let take_profit_price =
        (config.take_profit_ticks > 0.0).then(|| signed(config.take_profit_ticks));
    let fixed_stop_ticks = if config.stop_loss_ticks > 0.0 {
        config.stop_loss_ticks
    } else if config.use_trailing_stop {
        config.trail_trigger_ticks + config.trail_offset_ticks
    } else {
        0.0
    };
    let current_stop_price = (fixed_stop_ticks > 0.0).then(|| match side {
        ProtectionSide::Long => entry_price - fixed_stop_ticks * tick_size,
        ProtectionSide::Short => entry_price + fixed_stop_ticks * tick_size,
    });
    let fixed_hit = if config.use_trailing_stop {
        None
    } else {
        let stop = (config.stop_loss_ticks > 0.0).then(|| match side {
            ProtectionSide::Long => entry_price - config.stop_loss_ticks * tick_size,
            ProtectionSide::Short => entry_price + config.stop_loss_ticks * tick_size,
        });
        // Deterministic replay applies deferred entries at the beginning of
        // the next raw bar, before that bar's protection trigger pass.  The
        // same bar can therefore hit a bracket immediately after entry.
        protection_index.first_fixed_hit(entry_index, side, take_profit_price, stop, policy)
    };
    Some(KernelProtection {
        side,
        order_strategy_id,
        take_profit_price,
        current_stop_price,
        trail_trigger_ticks: config.trail_trigger_ticks,
        trail_offset_ticks: config.trail_offset_ticks,
        trail_frequency: tick_size,
        best_price: entry_price,
        trailing_active: false,
        fixed_hit,
    })
}

fn protection_hit_for_bar(
    protection: KernelProtection,
    _position: KernelPosition,
    bar_index: usize,
    bar: &Bar,
    policy: ReplayBarProtectionPolicy,
) -> Option<ProtectionHit> {
    if let Some(fixed_hit) = protection.fixed_hit {
        return (fixed_hit.index == bar_index).then_some(fixed_hit);
    }
    let target = protection
        .take_profit_price
        .and_then(|price| match protection.side {
            ProtectionSide::Long => {
                (bar.high >= price).then_some((price, ProtectionReason::TakeProfit))
            }
            ProtectionSide::Short => {
                (bar.low <= price).then_some((price, ProtectionReason::TakeProfit))
            }
        });
    let stop = protection
        .current_stop_price
        .and_then(|price| match protection.side {
            ProtectionSide::Long => {
                (bar.low <= price).then_some((price, ProtectionReason::StopLoss))
            }
            ProtectionSide::Short => {
                (bar.high >= price).then_some((price, ProtectionReason::StopLoss))
            }
        });
    match (target, stop) {
        (None, None) => None,
        (Some((price, reason)), None) | (None, Some((price, reason))) => Some(ProtectionHit {
            index: bar_index,
            reason,
            price,
        }),
        (Some((target_price, _)), Some((stop_price, _))) => {
            let reason = match policy {
                ReplayBarProtectionPolicy::Conservative => ProtectionReason::StopLoss,
                ReplayBarProtectionPolicy::Optimistic => ProtectionReason::TakeProfit,
                ReplayBarProtectionPolicy::NearestOpen => {
                    if (bar.open - target_price).abs() + EPSILON < (bar.open - stop_price).abs() {
                        ProtectionReason::TakeProfit
                    } else {
                        ProtectionReason::StopLoss
                    }
                }
            };
            Some(ProtectionHit {
                index: bar_index,
                reason,
                price: if reason == ProtectionReason::TakeProfit {
                    target_price
                } else {
                    stop_price
                },
            })
        }
    }
}

fn update_trailing_stop(position: &mut KernelPosition, bar: &Bar, tick_size: f64) {
    let Some(protection) = position.protection.as_mut() else {
        return;
    };
    if protection.trail_trigger_ticks <= 0.0 || protection.trail_offset_ticks <= 0.0 {
        return;
    }
    let favorable = match protection.side {
        ProtectionSide::Long => bar.high,
        ProtectionSide::Short => bar.low,
    };
    if favorable.is_finite() {
        match protection.side {
            ProtectionSide::Long => protection.best_price = protection.best_price.max(favorable),
            ProtectionSide::Short => protection.best_price = protection.best_price.min(favorable),
        }
    }
    let favorable_ticks = match protection.side {
        ProtectionSide::Long => (protection.best_price - position.entry_price) / tick_size,
        ProtectionSide::Short => (position.entry_price - protection.best_price) / tick_size,
    };
    if favorable_ticks + EPSILON < protection.trail_trigger_ticks {
        if match protection.side {
            ProtectionSide::Long => (bar.close - position.entry_price) / tick_size < 0.0,
            ProtectionSide::Short => (position.entry_price - bar.close) / tick_size < 0.0,
        } {
            protection.best_price = position.entry_price;
        }
        return;
    }
    protection.trailing_active = true;
    let raw_candidate = match protection.side {
        ProtectionSide::Long => protection.best_price - protection.trail_offset_ticks * tick_size,
        ProtectionSide::Short => protection.best_price + protection.trail_offset_ticks * tick_size,
    };
    let Some(current) = protection.current_stop_price else {
        protection.current_stop_price = Some(raw_candidate);
        return;
    };
    let frequency = protection.trail_frequency.max(EPSILON);
    let tightened = match protection.side {
        ProtectionSide::Long => {
            current + (((raw_candidate - current) / frequency).floor().max(0.0) * frequency)
        }
        ProtectionSide::Short => {
            current - (((current - raw_candidate) / frequency).floor().max(0.0) * frequency)
        }
    };
    let improves = match protection.side {
        ProtectionSide::Long => tightened > current + EPSILON,
        ProtectionSide::Short => tightened < current - EPSILON,
    };
    if improves {
        protection.current_stop_price = Some(tightened);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_fill(
    state: &mut KernelState,
    replay: &ReplayState,
    _child: &ReplaySweepChildSpec,
    bar: &Bar,
    action: &str,
    quantity: i32,
    price: f64,
    signal_timestamp_ns: Option<i64>,
    exit_reason: Option<&str>,
    order_strategy_id: Option<i64>,
    latency_ms: u64,
    fill_source: ReplayFillPriceSource,
) {
    let signed = if action.eq_ignore_ascii_case("Buy") {
        quantity as f64
    } else {
        -(quantity as f64)
    };
    let prior = state.position;
    let gross = match prior {
        None => Some(0.0),
        Some(position) if (position.qty as f64).signum() == signed.signum() => Some(0.0),
        Some(position) => {
            let close_qty = (position.qty.abs() as f64).min(signed.abs());
            let points = if position.qty > 0 {
                price - position.entry_price
            } else {
                position.entry_price - price
            };
            replay
                .market_value_per_point()
                .map(|value| points * close_qty * value)
        }
    };
    let fill_id = state.next_fill_id();
    let order_id = state.next_order_id();
    let lifecycle_sequence = Some(state.next_lifecycle_sequence());
    state.fills.push(ReplayExecutionFill {
        sequence: state.fills.len() as u64,
        lifecycle_sequence,
        fill_id,
        order_id,
        order_strategy_id,
        protection_order_id: exit_reason.map(|_| order_id),
        account_id: replay.account.id,
        contract_id: replay.contract.id,
        contract_name: replay.contract.name.clone(),
        side: action.to_string(),
        quantity: quantity as f64,
        price,
        signal_timestamp_ns,
        submission_timestamp_ns: signal_timestamp_ns,
        exchange_arrival_timestamp_ns: signal_timestamp_ns
            .map(|ts| ts.saturating_add((latency_ms as i64).saturating_mul(1_000_000))),
        acknowledgement_timestamp_ns: Some(bar.ts_ns),
        fill_timestamp_ns: bar.ts_ns,
        fill_price_source: fill_source,
        execution_precision: ReplayExecutionPrecision::BarApproximate,
        exit_reason: exit_reason.map(ToString::to_string),
        latency_ms,
        tick_size: replay.market_tick_size(),
        value_per_point: replay.market_value_per_point(),
        gross_realized_pnl_delta: gross,
    });
}

fn append_diagnostic(
    state: &mut KernelState,
    trace: &EmaCrossTrace,
    bars: &[Bar],
    signal_index: usize,
    history_loaded: usize,
    current_qty: i32,
    strategy: &crate::strategy::ExecutionStrategyConfig,
    target_qty: Option<i32>,
    decision: &str,
    gate_reason: &str,
) {
    let Some(bar) = bars.get(signal_index) else {
        return;
    };
    let raw_buy = trace.raw_buy.get(signal_index).copied().unwrap_or(false);
    let raw_sell = trace.raw_sell.get(signal_index).copied().unwrap_or(false);
    let (effective_buy, effective_sell) = if strategy.native_ema.inverted {
        (raw_sell, raw_buy)
    } else {
        (raw_buy, raw_sell)
    };
    let signal = if effective_buy {
        "Enter Long"
    } else if effective_sell {
        "Enter Short"
    } else {
        "Hold"
    };
    let raw_signal = if raw_buy {
        "buy"
    } else if raw_sell {
        "sell"
    } else {
        "none"
    };
    let effective_signal = if effective_buy {
        "buy"
    } else if effective_sell {
        "sell"
    } else {
        "none"
    };
    let relative_index = signal_index.saturating_sub(history_loaded);
    state.diagnostics.push(ReplaySignalDiagnostic {
        bar_timestamp_ns: bar.ts_ns,
        bar_open: bar.open,
        bar_high: bar.high,
        bar_low: bar.low,
        bar_close: bar.close,
        bar_index: Some(relative_index + 1),
        bar_count: relative_index + strategy.native_signal_delay_bars + 1,
        strategy: strategy.native_strategy.slug().to_string(),
        execution_path: "guarded".to_string(),
        signal_timing: "closed bar".to_string(),
        signal_delay_bars: strategy.native_signal_delay_bars,
        signal: signal.to_string(),
        raw_signal: raw_signal.to_string(),
        effective_signal: effective_signal.to_string(),
        raw_buy_signal: raw_buy,
        raw_sell_signal: raw_sell,
        effective_buy_signal: effective_buy,
        effective_sell_signal: effective_sell,
        current_position_qty: current_qty,
        effective_position_qty: current_qty,
        target_qty,
        decision: decision.to_string(),
        gate_reason: gate_reason.to_string(),
        order_action: target_qty.map(|target| {
            if target > current_qty {
                "Buy".to_string()
            } else {
                "Sell".to_string()
            }
        }),
        order_qty: target_qty.map(|target| target.saturating_sub(current_qty).abs()),
        indicator_name: "EMA".to_string(),
        previous_fast_indicator: trace
            .previous_fast
            .get(signal_index)
            .copied()
            .filter(|v| v.is_finite()),
        previous_slow_indicator: trace
            .previous_slow
            .get(signal_index)
            .copied()
            .filter(|v| v.is_finite()),
        fast_indicator: trace
            .fast
            .get(signal_index)
            .copied()
            .filter(|v| v.is_finite()),
        slow_indicator: trace
            .slow
            .get(signal_index)
            .copied()
            .filter(|v| v.is_finite()),
        auxiliary_name: None,
        auxiliary_value: None,
        hold_reason: (signal == "Hold").then(|| gate_reason.to_string()),
        strategy_detail: format!(
            "EMA {} / {} | inverted {}",
            trace.fast_length, trace.slow_length, strategy.native_ema.inverted
        ),
        fingerprint: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, open: f64, high: f64, low: f64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open,
            high,
            low,
            close,
            volume: Some(1.0),
        }
    }

    #[test]
    fn prepared_ema_series_matches_batch_window_values() {
        let bars = (0..5_000)
            .map(|index| {
                let close = 100.0 + (index as f64 * 0.17).sin() * 3.0 + index as f64 * 0.01;
                bar(index as i64 + 1, close, close + 0.5, close - 0.5, close)
            })
            .collect::<Vec<_>>();
        let actual = build_ema_series(&bars, 17, 0);
        let config = EmaCrossConfig {
            fast_length: 17,
            slow_length: 17,
            ..EmaCrossConfig::default()
        };
        for end in [1, 17, 64, 4_096, 4_097, 4_500, bars.len()] {
            let start = end.saturating_sub(crate::tradovate::ENGINE_MARKET_BAR_LIMIT);
            let expected = config.evaluate(&bars[start..end], None).fast_ema;
            match expected {
                Some(expected) => assert_eq!(
                    actual[end - 1],
                    expected,
                    "EMA mismatch at retained-window endpoint {end}"
                ),
                None => assert!(
                    actual[end - 1].is_nan(),
                    "expected warmup NaN at retained-window endpoint {end}"
                ),
            }
        }
    }

    #[test]
    fn prepared_ema_series_resets_at_dataset_view_evaluation_boundary() {
        let bars = (0..128)
            .map(|index| {
                let close = 100.0 + (index as f64 * 0.31).sin() * 2.0;
                bar(index as i64 + 1, close, close + 0.5, close - 0.5, close)
            })
            .collect::<Vec<_>>();
        let signal_start = 64;
        let actual = build_ema_series(&bars, 17, signal_start);
        let config = EmaCrossConfig {
            fast_length: 17,
            slow_length: 17,
            ..EmaCrossConfig::default()
        };
        assert!(actual[..signal_start].iter().all(|value| value.is_nan()));
        for end in (signal_start + 1)..=bars.len() {
            let expected = config.evaluate(&bars[signal_start..end], None).fast_ema;
            match expected {
                Some(expected) => assert_eq!(
                    actual[end - 1],
                    expected,
                    "EMA mismatch at evaluation-window endpoint {end}"
                ),
                None => assert!(
                    actual[end - 1].is_nan(),
                    "expected warmup NaN at evaluation-window endpoint {end}"
                ),
            }
        }
    }

    #[test]
    fn protection_index_matches_bar_policy_when_both_legs_hit() {
        let index = BarProtectionIndex::new(&[
            bar(1, 100.0, 103.0, 97.0, 100.0),
            bar(2, 100.0, 101.0, 99.0, 100.0),
        ]);
        let conservative = index.first_fixed_hit(
            0,
            ProtectionSide::Long,
            Some(102.0),
            Some(98.0),
            ReplayBarProtectionPolicy::Conservative,
        );
        assert_eq!(
            conservative.map(|hit| hit.reason),
            Some(ProtectionReason::StopLoss)
        );
        let optimistic = index.first_fixed_hit(
            0,
            ProtectionSide::Long,
            Some(102.0),
            Some(98.0),
            ReplayBarProtectionPolicy::Optimistic,
        );
        assert_eq!(
            optimistic.map(|hit| hit.reason),
            Some(ProtectionReason::TakeProfit)
        );
        let nearest = index.first_fixed_hit(
            0,
            ProtectionSide::Long,
            Some(101.0),
            Some(96.0),
            ReplayBarProtectionPolicy::NearestOpen,
        );
        assert_eq!(
            nearest.map(|hit| hit.reason),
            Some(ProtectionReason::TakeProfit)
        );
    }

    #[test]
    fn protection_index_returns_first_reachable_bar_and_caches_query() {
        let index = BarProtectionIndex::new(&[
            bar(1, 100.0, 100.5, 99.5, 100.0),
            bar(2, 100.0, 101.0, 99.0, 100.0),
            bar(3, 100.0, 102.0, 98.0, 100.0),
        ]);
        let first = index.first_fixed_hit(
            0,
            ProtectionSide::Short,
            Some(99.0),
            Some(101.0),
            ReplayBarProtectionPolicy::Conservative,
        );
        let second = index.first_fixed_hit(
            0,
            ProtectionSide::Short,
            Some(99.0),
            Some(101.0),
            ReplayBarProtectionPolicy::Conservative,
        );
        assert_eq!(first, second);
        assert_eq!(first.map(|hit| hit.index), Some(1));
    }

    #[test]
    fn protection_index_includes_bar_where_entry_becomes_active() {
        let index = BarProtectionIndex::new(&[
            // The preceding bar is unrelated; the query starts at the bar
            // where the deferred entry becomes active.
            bar(1, 100.0, 103.0, 97.0, 100.0),
            bar(2, 100.0, 102.0, 99.0, 100.0),
            bar(3, 100.0, 100.5, 99.5, 100.0),
        ]);
        let hit = index.first_fixed_hit(
            1,
            ProtectionSide::Long,
            Some(101.0),
            Some(98.0),
            ReplayBarProtectionPolicy::Conservative,
        );
        assert_eq!(hit.map(|value| value.index), Some(1));
        assert_eq!(
            hit.map(|value| value.reason),
            Some(ProtectionReason::TakeProfit)
        );
    }
}
