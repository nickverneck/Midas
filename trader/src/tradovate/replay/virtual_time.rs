use super::*;
use crate::broker::ReplayEngineMode;
use std::collections::BTreeMap;

/// A deterministic replay event. Market timestamps and semantic phase own
/// ordering; insertion sequence breaks ties without consulting wall time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReplayVirtualEvent {
    pub market_ts_ns: i64,
    pub logical_step: u64,
    pub sequence: u64,
    pub kind: ReplayVirtualEventKind,
}

impl ReplayVirtualEvent {
    pub fn strategy_evaluation_after_bar(&self) -> Result<Self> {
        let ReplayVirtualEventKind::BarClose { bar_index } = self.kind else {
            bail!("strategy evaluation requires a replay bar-close event")
        };
        Ok(Self {
            market_ts_ns: self.market_ts_ns,
            logical_step: self.logical_step,
            sequence: self.sequence,
            kind: ReplayVirtualEventKind::StrategyEvaluation {
                evaluation_id: bar_index as u64,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReplayVirtualEventKind {
    Tick { tick_index: usize },
    QuoteUpdate { quote_index: usize },
    DomUpdate { dom_index: usize },
    RawBarOpen { bar_sequence: u64 },
    BarClose { bar_index: usize },
    StrategyEvaluation { evaluation_id: u64 },
    OrderSubmitted { order_id: u64 },
    OrderArrivesAtExchange { order_id: u64 },
    OrderAck { order_id: u64 },
    Fill { fill_id: u64 },
    ProtectionUpdate { order_id: u64 },
}

impl ReplayVirtualEventKind {
    pub(crate) fn phase(&self) -> u8 {
        match self {
            // Raw market events with identical provider timestamps preserve
            // source insertion order instead of inventing quote/tick priority.
            Self::Tick { .. }
            | Self::QuoteUpdate { .. }
            | Self::DomUpdate { .. }
            | Self::RawBarOpen { .. } => 0,
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
    logical_step: u64,
    phase: u8,
}

impl ReplayVirtualClock {
    pub fn now_ns(self) -> Option<i64> {
        self.now_ns
    }
}

#[derive(Debug, Default)]
pub(crate) struct ReplayVirtualEventQueue {
    events: BTreeMap<(i64, u64, u8, u64), ReplayVirtualEventKind>,
    next_sequence: u64,
    clock: ReplayVirtualClock,
}

impl ReplayVirtualEventQueue {
    pub fn schedule(&mut self, market_ts_ns: i64, kind: ReplayVirtualEventKind) -> Result<u64> {
        self.schedule_at_step(market_ts_ns, 0, kind)
    }

    pub(crate) fn schedule_at_step(
        &mut self,
        market_ts_ns: i64,
        logical_step: u64,
        kind: ReplayVirtualEventKind,
    ) -> Result<u64> {
        let phase = kind.phase();
        if let Some(now_ns) = self.clock.now_ns() {
            if (market_ts_ns, logical_step, phase)
                < (now_ns, self.clock.logical_step, self.clock.phase)
            {
                bail!(
                    "cannot schedule replay event at {market_ts_ns} step {logical_step} phase {phase} behind virtual clock {now_ns} step {} phase {}",
                    self.clock.logical_step,
                    self.clock.phase,
                );
            }
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("replay virtual event sequence exhausted")?;
        self.events
            .insert((market_ts_ns, logical_step, phase, sequence), kind);
        Ok(sequence)
    }

    pub(crate) fn pop_next(&mut self) -> Option<ReplayVirtualEvent> {
        let ((market_ts_ns, logical_step, phase, sequence), kind) = self.events.pop_first()?;
        self.clock = ReplayVirtualClock {
            now_ns: Some(market_ts_ns),
            logical_step,
            phase,
        };
        Some(ReplayVirtualEvent {
            market_ts_ns,
            logical_step,
            sequence,
            kind,
        })
    }

    pub(crate) fn pop_next_through(
        &mut self,
        market_ts_ns: i64,
        logical_step: u64,
        phase: u8,
    ) -> Option<ReplayVirtualEvent> {
        let &(event_ts_ns, event_step, event_phase, _) = self.events.first_key_value()?.0;
        if (event_ts_ns, event_step, event_phase) > (market_ts_ns, logical_step, phase) {
            return None;
        }
        self.pop_next()
    }

    #[allow(dead_code)]
    pub(super) fn clock(&self) -> ReplayVirtualClock {
        self.clock
    }
}

pub(super) enum ReplayBarSchedule {
    Legacy { next_index: usize, len: usize },
    Deterministic { next_index: usize, len: usize },
}

impl ReplayBarSchedule {
    pub fn new(mode: ReplayEngineMode, bars: &[Bar]) -> Result<Self> {
        match mode {
            ReplayEngineMode::Legacy => Ok(Self::Legacy {
                next_index: 0,
                len: bars.len(),
            }),
            ReplayEngineMode::Deterministic => Ok(Self::Deterministic {
                next_index: 0,
                len: bars.len(),
            }),
        }
    }

    pub fn next_bar(&mut self, bars: &[Bar]) -> Option<ReplayVirtualEvent> {
        match self {
            Self::Legacy { next_index, len } | Self::Deterministic { next_index, len } => {
                if *next_index >= *len {
                    return None;
                }
                let bar_index = *next_index;
                *next_index += 1;
                Some(ReplayVirtualEvent {
                    market_ts_ns: bars[bar_index].ts_ns,
                    logical_step: bar_index as u64,
                    sequence: bar_index as u64,
                    kind: ReplayVirtualEventKind::BarClose { bar_index },
                })
            }
        }
    }
}
