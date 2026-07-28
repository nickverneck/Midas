use super::*;
use crate::broker::ReplayEngineMode;
use std::collections::BTreeMap;

/// A deterministic replay event. Market timestamps and semantic phase own
/// ordering; insertion sequence breaks ties without consulting wall time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReplayVirtualEvent {
    pub market_ts_ns: i64,
    pub sequence: u64,
    pub kind: ReplayVirtualEventKind,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ReplayVirtualEventKind {
    Tick { tick_index: usize },
    QuoteUpdate { quote_index: usize },
    DomUpdate { dom_index: usize },
    BarClose { bar_index: usize },
    StrategyEvaluation { evaluation_id: u64 },
    OrderSubmitted { order_id: u64 },
    OrderArrivesAtExchange { order_id: u64 },
    OrderAck { order_id: u64 },
    Fill { fill_id: u64 },
    ProtectionUpdate { order_id: u64 },
}

impl ReplayVirtualEventKind {
    fn phase(&self) -> u8 {
        match self {
            // Raw market events with identical provider timestamps preserve
            // source insertion order instead of inventing quote/tick priority.
            Self::Tick { .. } | Self::QuoteUpdate { .. } | Self::DomUpdate { .. } => 0,
            Self::BarClose { .. } => 1,
            Self::StrategyEvaluation { .. } => 2,
            Self::OrderSubmitted { .. } => 3,
            Self::OrderArrivesAtExchange { .. } => 4,
            Self::OrderAck { .. } => 5,
            Self::Fill { .. } => 6,
            Self::ProtectionUpdate { .. } => 7,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ReplayVirtualClock {
    now_ns: Option<i64>,
    phase: u8,
}

impl ReplayVirtualClock {
    pub fn now_ns(self) -> Option<i64> {
        self.now_ns
    }
}

#[derive(Debug, Default)]
pub(super) struct ReplayVirtualEventQueue {
    events: BTreeMap<(i64, u8, u64), ReplayVirtualEventKind>,
    next_sequence: u64,
    clock: ReplayVirtualClock,
}

impl ReplayVirtualEventQueue {
    pub fn schedule(&mut self, market_ts_ns: i64, kind: ReplayVirtualEventKind) -> Result<u64> {
        let phase = kind.phase();
        if let Some(now_ns) = self.clock.now_ns() {
            if market_ts_ns < now_ns || (market_ts_ns == now_ns && phase < self.clock.phase) {
                bail!(
                    "cannot schedule replay event at {market_ts_ns} phase {phase} behind virtual clock {now_ns} phase {}",
                    self.clock.phase
                );
            }
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("replay virtual event sequence exhausted")?;
        self.events.insert((market_ts_ns, phase, sequence), kind);
        Ok(sequence)
    }

    pub fn pop_next(&mut self) -> Option<ReplayVirtualEvent> {
        let ((market_ts_ns, phase, sequence), kind) = self.events.pop_first()?;
        self.clock = ReplayVirtualClock {
            now_ns: Some(market_ts_ns),
            phase,
        };
        Some(ReplayVirtualEvent {
            market_ts_ns,
            sequence,
            kind,
        })
    }

    #[allow(dead_code)]
    pub fn clock(&self) -> ReplayVirtualClock {
        self.clock
    }
}

pub(super) enum ReplayBarSchedule {
    Legacy { next_index: usize, len: usize },
    Deterministic { queue: ReplayVirtualEventQueue },
}

impl ReplayBarSchedule {
    pub fn new(mode: ReplayEngineMode, bars: &[Bar]) -> Result<Self> {
        match mode {
            ReplayEngineMode::Legacy => Ok(Self::Legacy {
                next_index: 0,
                len: bars.len(),
            }),
            ReplayEngineMode::Deterministic => {
                let mut queue = ReplayVirtualEventQueue::default();
                for (bar_index, bar) in bars.iter().enumerate() {
                    queue.schedule(bar.ts_ns, ReplayVirtualEventKind::BarClose { bar_index })?;
                }
                Ok(Self::Deterministic { queue })
            }
        }
    }

    pub fn next_bar(&mut self, bars: &[Bar]) -> Option<ReplayVirtualEvent> {
        match self {
            Self::Legacy { next_index, len } => {
                if *next_index >= *len {
                    return None;
                }
                let bar_index = *next_index;
                *next_index += 1;
                Some(ReplayVirtualEvent {
                    market_ts_ns: bars[bar_index].ts_ns,
                    sequence: bar_index as u64,
                    kind: ReplayVirtualEventKind::BarClose { bar_index },
                })
            }
            Self::Deterministic { queue } => queue.pop_next(),
        }
    }
}
