use super::super::virtual_time::{
    ReplayBarSchedule, ReplayVirtualEventKind, ReplayVirtualEventQueue,
};
use super::super::worker::scale_duration;
use crate::broker::{Bar, ReplayEngineMode, ReplaySpeed};
use std::time::Duration;

fn bar(ts_ns: i64) -> Bar {
    Bar {
        ts_ns,
        open: 1.0,
        high: 2.0,
        low: 0.5,
        close: 1.5,
        volume: Some(10.0),
    }
}

#[test]
fn same_timestamp_events_follow_semantic_phase_then_insertion_order() {
    let mut queue = ReplayVirtualEventQueue::default();
    let timestamp = 42;

    queue
        .schedule(timestamp, ReplayVirtualEventKind::Fill { fill_id: 8 })
        .unwrap();
    queue
        .schedule(
            timestamp,
            ReplayVirtualEventKind::QuoteUpdate { quote_index: 1 },
        )
        .unwrap();
    queue
        .schedule(timestamp, ReplayVirtualEventKind::Tick { tick_index: 2 })
        .unwrap();
    queue
        .schedule(
            timestamp,
            ReplayVirtualEventKind::StrategyEvaluation { evaluation_id: 3 },
        )
        .unwrap();
    queue
        .schedule(
            timestamp,
            ReplayVirtualEventKind::ProtectionUpdate { order_id: 9 },
        )
        .unwrap();
    queue
        .schedule(
            timestamp,
            ReplayVirtualEventKind::OrderSubmitted { order_id: 4 },
        )
        .unwrap();
    queue
        .schedule(timestamp, ReplayVirtualEventKind::OrderAck { order_id: 6 })
        .unwrap();
    queue
        .schedule(timestamp, ReplayVirtualEventKind::BarClose { bar_index: 0 })
        .unwrap();
    queue
        .schedule(
            timestamp,
            ReplayVirtualEventKind::OrderArrivesAtExchange { order_id: 5 },
        )
        .unwrap();

    let mut events = Vec::new();
    while let Some(event) = queue.pop_next() {
        events.push(event);
    }

    assert!(matches!(
        events[0].kind,
        ReplayVirtualEventKind::QuoteUpdate { quote_index: 1 }
    ));
    assert!(matches!(
        events[1].kind,
        ReplayVirtualEventKind::Tick { tick_index: 2 }
    ));
    assert!(matches!(
        events[2].kind,
        ReplayVirtualEventKind::BarClose { bar_index: 0 }
    ));
    assert!(matches!(
        events[3].kind,
        ReplayVirtualEventKind::StrategyEvaluation { evaluation_id: 3 }
    ));
    assert!(matches!(
        events[4].kind,
        ReplayVirtualEventKind::OrderSubmitted { order_id: 4 }
    ));
    assert!(matches!(
        events[5].kind,
        ReplayVirtualEventKind::OrderArrivesAtExchange { order_id: 5 }
    ));
    assert!(matches!(
        events[6].kind,
        ReplayVirtualEventKind::OrderAck { order_id: 6 }
    ));
    assert!(matches!(
        events[7].kind,
        ReplayVirtualEventKind::Fill { fill_id: 8 }
    ));
    assert!(matches!(
        events[8].kind,
        ReplayVirtualEventKind::ProtectionUpdate { order_id: 9 }
    ));
}

#[test]
fn events_follow_market_time_even_when_scheduled_out_of_order() {
    let mut queue = ReplayVirtualEventQueue::default();
    for timestamp in [30, 10, 20] {
        queue
            .schedule(
                timestamp,
                ReplayVirtualEventKind::Tick {
                    tick_index: timestamp as usize,
                },
            )
            .unwrap();
    }

    let timestamps = std::iter::from_fn(|| queue.pop_next())
        .map(|event| event.market_ts_ns)
        .collect::<Vec<_>>();

    assert_eq!(timestamps, vec![10, 20, 30]);
}

#[test]
fn virtual_clock_rejects_events_scheduled_behind_it() {
    let mut queue = ReplayVirtualEventQueue::default();
    queue
        .schedule(
            100,
            ReplayVirtualEventKind::StrategyEvaluation { evaluation_id: 1 },
        )
        .unwrap();
    queue.pop_next().unwrap();

    let earlier_timestamp = queue
        .schedule(99, ReplayVirtualEventKind::Fill { fill_id: 2 })
        .unwrap_err();
    assert!(
        earlier_timestamp
            .to_string()
            .contains("behind virtual clock")
    );

    let earlier_phase = queue
        .schedule(100, ReplayVirtualEventKind::BarClose { bar_index: 0 })
        .unwrap_err();
    assert!(earlier_phase.to_string().contains("behind virtual clock"));

    queue
        .schedule(100, ReplayVirtualEventKind::OrderSubmitted { order_id: 3 })
        .unwrap();
}

#[test]
fn legacy_and_deterministic_schedules_match_for_ordered_bars() {
    let bars = vec![bar(10), bar(20), bar(30)];

    let legacy = drain_bar_schedule(ReplayEngineMode::Legacy, &bars);
    let deterministic = drain_bar_schedule(ReplayEngineMode::Deterministic, &bars);

    assert_eq!(legacy, deterministic);
}

#[test]
fn duplicate_bar_timestamps_preserve_source_order() {
    let bars = vec![bar(10), bar(10), bar(20)];

    assert_eq!(
        drain_bar_schedule(ReplayEngineMode::Deterministic, &bars),
        vec![(10, 0), (10, 1), (20, 2)]
    );
}

#[test]
fn replay_speed_changes_wall_pacing_not_virtual_event_trace() {
    let bars = vec![bar(0), bar(60_000_000_000), bar(120_000_000_000)];
    let expected_trace = drain_bar_schedule(ReplayEngineMode::Deterministic, &bars);
    let market_gap = Duration::from_secs(60);

    let speeds = [
        ReplaySpeed::Realtime,
        ReplaySpeed::X2,
        ReplaySpeed::X5,
        ReplaySpeed::X10,
        ReplaySpeed::X25,
    ];
    let wall_delays = speeds
        .into_iter()
        .map(|speed| {
            assert_eq!(
                drain_bar_schedule(ReplayEngineMode::Deterministic, &bars),
                expected_trace
            );
            scale_duration(market_gap, 1.0 / speed.multiplier())
        })
        .collect::<Vec<_>>();

    assert_eq!(wall_delays[0], Duration::from_secs(60));
    assert_eq!(wall_delays[1], Duration::from_secs(30));
    assert_eq!(wall_delays[4], Duration::from_millis(2_400));
}

fn drain_bar_schedule(mode: ReplayEngineMode, bars: &[Bar]) -> Vec<(i64, usize)> {
    let mut schedule = ReplayBarSchedule::new(mode, bars).unwrap();
    let mut trace = Vec::new();
    while let Some(event) = schedule.next_bar(bars) {
        let ReplayVirtualEventKind::BarClose { bar_index } = event.kind else {
            panic!("bar schedule returned a non-bar event")
        };
        trace.push((event.market_ts_ns, bar_index));
    }
    trace
}
