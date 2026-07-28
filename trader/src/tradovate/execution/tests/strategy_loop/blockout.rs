use super::super::*;
use super::support::*;

#[test]
fn strategy_blockout_flattens_inside_configured_preclose_window() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
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
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
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
