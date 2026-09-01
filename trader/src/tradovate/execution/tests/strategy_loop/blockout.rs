use super::super::*;
use super::support::*;

#[test]
fn strategy_blockout_flattens_inside_configured_preclose_window() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let bar_ts = et_ts_ns(2026, 3, 9, 16, 20);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.blockout_enabled = true;
    session.execution_config.blockout_minutes_before_close = 45.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(bar_ts - 60_000_000_000);
    session.market.session_profile = Some(InstrumentSessionProfile::FuturesGlobex);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: bar_ts,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.0,
        volume: None,
    }];
    seed_long_position(&mut session);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("blockout should flatten open position");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(0));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Session hold active")
    );
    match broker_rx.try_recv().expect("flatten command queued") {
        BrokerCommand::LiquidatePosition { liquidation, .. } => {
            assert_eq!(liquidation.target_qty, Some(0));
            assert_eq!(liquidation.contract_name, "ESH6");
        }
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(0));
            assert_eq!(order.contract_name, "ESH6");
            assert_eq!(order.order_action, "Sell");
            assert_eq!(order.order_qty, 1);
        }
        _ => panic!("expected liquidation command"),
    }
}

#[test]
fn disabled_strategy_blockout_does_not_flatten_preclose_position() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let bar_ts = et_ts_ns(2026, 3, 9, 16, 20);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.blockout_enabled = false;
    session.execution_config.blockout_minutes_before_close = 45.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(bar_ts - 60_000_000_000);
    session.market.session_profile = Some(InstrumentSessionProfile::FuturesGlobex);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: bar_ts,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.0,
        volume: None,
    }];
    seed_long_position(&mut session);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("disabled blockout should allow strategy evaluation");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(broker_rx.try_recv().is_err(), "no flatten should be queued");
    assert!(
        !session
            .execution_runtime
            .last_summary
            .contains("Session hold active")
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | ema_cross | no target")
                && message.contains("signal Hold")
                && message.contains("target_qty none")
    )));
}

#[test]
fn live_session_hold_skips_incremental_hma_evaluation() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let latest_ts = et_ts_ns(2026, 3, 9, 16, 20);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_hma_cross.calculation_mode =
        crate::strategies::hma_cross::HmaCalculationMode::Incremental;
    session.execution_config.blockout_enabled = true;
    session.execution_config.blockout_minutes_before_close = 45.0;
    session.execution_runtime.armed = true;
    session.market.session_profile = Some(InstrumentSessionProfile::FuturesGlobex);
    session.market.bars = (0..32)
        .map(|index| Bar {
            ts_ns: latest_ts - (31 - index) as i64 * 60_000_000_000,
            open: 5000.0 + index as f64,
            high: 5001.0 + index as f64,
            low: 4999.0 + index as f64,
            close: 5000.0 + index as f64,
            volume: None,
        })
        .collect();
    session.market.history_loaded = session.market.bars.len();

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("session hold should return without evaluating the strategy");

    let audit = session
        .execution_runtime
        .hma_cross_execution
        .audit_counters();
    assert_eq!(audit.incremental_pushes, 0);
    assert_eq!(audit.full_rebuilds, 0);
    assert_eq!(audit.idle_evaluations, 0);
    assert_eq!(
        session.execution_runtime.last_closed_bar_ts,
        Some(latest_ts)
    );
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Session hold active")
    );
    assert!(
        broker_rx.try_recv().is_err(),
        "flat session hold must not order"
    );
}
