use super::super::*;
use super::support::*;

#[test]
fn guarded_closed_bar_delay_waits_extra_completed_bar_before_entry() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_config.native_signal_delay_bars = 1;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("delay should wait until the crossed bar becomes eligible");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(broker_rx.try_recv().is_err(), "delay should suppress entry");

    session.market.bars.push(Bar {
        ts_ns: 8,
        open: 13.0,
        high: 13.5,
        low: 12.5,
        close: 13.0,
        volume: None,
    });
    session.market.history_loaded = 8;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("delayed crossed bar should queue an entry");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    match broker_rx
        .try_recv()
        .expect("delayed closed-bar signal should queue a buy")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from delayed buy signal"),
    }
}

#[test]
fn guarded_closed_bar_signal_dispatches_once_even_if_position_returns_flat() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("first closed-bar signal should queue an entry");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    broker_rx
        .try_recv()
        .expect("first closed-bar signal should queue an order");
    let _ = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();

    session.execution_runtime.pending_target_qty = None;
    session.order_submit_in_flight = false;
    session.order_latency_tracker = None;
    session.execution_runtime.last_closed_bar_ts = Some(6);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("same closed-bar signal should be blocked after a flat return");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        broker_rx.try_recv().is_err(),
        "same closed-bar signal should not queue a second order"
    );
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("already dispatched")
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | ema_cross | closed-bar already dispatched")
                && message.contains("signal Buy")
                && message.contains("bar_ts 7")
                && message.starts_with("strategy decision |")
                && message.contains("path=guarded")
                && message.contains("decision=closed-bar already dispatched")
                && message.contains("strategy_detail=Signal: Buy")
                && message.contains("delta:")
    )));
}

#[test]
fn guarded_closed_bar_blocks_repeat_flat_entry_side_until_opposite_dispatch() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(5);
    session.execution_runtime.last_dispatched_entry_signal = Some(StrategySignal::EnterShort);
    session.market.history_loaded = 6;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 12.0, 8.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("same-side flat signal should be consumed");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        broker_rx.try_recv().is_err(),
        "same-side flat signal should not queue another entry"
    );
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Sell entry side already dispatched while flat")
    );

    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("opposite flat signal should still be allowed");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(
        session.execution_runtime.last_dispatched_entry_signal,
        Some(StrategySignal::EnterLong)
    );
    match broker_rx
        .try_recv()
        .expect("opposite signal should queue a buy")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from opposite buy signal"),
    }
}

#[test]
fn hma_direct_path_executes_current_closed_bar_cross() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::HmaDirect;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();
    seed_short_position(&mut session);
    let precheck = session.execution_config.native_hma_cross.evaluate(
        &session.market.bars,
        Some(crate::strategies::PositionSide::Short),
    );
    assert!(
        precheck
            .fast_hma
            .zip(precheck.slow_hma)
            .is_some_and(|(fast, slow)| fast > slow),
        "{}",
        precheck.summary()
    );

    maybe_run_hma_direct_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("HMA direct should queue current-bar crossover");

    assert_eq!(
        session.execution_runtime.pending_target_qty,
        Some(1),
        "{}",
        session.execution_runtime.last_summary
    );
    match broker_rx
        .try_recv()
        .expect("HMA direct should queue a buy reversal")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from HMA direct buy signal"),
    }

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("hma direct dispatch")
                && message.contains("Prior HMA Side: fast<slow")
                && message.contains("HMA Side: fast>slow")
                && message.starts_with("strategy decision |")
                && message.contains("path=hma direct")
                && message.contains("decision=dispatching")
                && message.contains("prior_hma_side: fast<slow")
    )));
}

#[test]
fn hma_direct_path_does_not_enter_from_stale_cross_state() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::HmaDirect;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(7);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(5);
    session.market.history_loaded = 8;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0, 13.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();
    seed_short_position(&mut session);

    maybe_run_hma_direct_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("HMA direct should evaluate stale crossover state without ordering");

    assert_eq!(
        session.execution_runtime.pending_target_qty, None,
        "{}",
        session.execution_runtime.last_summary
    );
    assert!(
        broker_rx.try_recv().is_err(),
        "stale crossover state should not queue an order"
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("hma direct eval")
                && message.contains("signal Hold")
                && message.contains("Prior HMA Side: unset")
                && message.starts_with("strategy decision |")
                && message.contains("strategy_detail=Signal: Hold")
                && message.contains("current_bar_side_edge: false")
    )));
}

#[test]
fn guarded_hma_cross_blocks_same_bar_revised_side_change_after_dispatch() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::Guarded;
    session.execution_config.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_config.native_hma_cross.take_profit_ticks = 5.0;
    session.execution_config.native_hma_cross.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.contract_id = Some(3570918);
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(7);
    session.execution_runtime.last_closed_bar_fingerprint = Some(0);
    session.execution_runtime.last_dispatched_signal_bar_ts = Some(7);
    session.execution_runtime.last_dispatched_entry_signal = Some(StrategySignal::EnterShort);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(7);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
            volume: None,
        })
        .collect();
    seed_short_position(&mut session);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            1001,
            json!({
                "id": 1001,
                "accountId": 42,
                "contractId": 3570918,
                "ordStatus": "Working"
            }),
        )]),
    );
    session.user_store.order_strategy_links.insert(
        1,
        json!({
            "id": 1,
            "orderStrategyId": 77,
            "orderId": 1001
        }),
    );

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("guarded HMA cross should block a revised side change on the same bar");

    assert_eq!(
        session.execution_runtime.pending_target_qty, None,
        "{}",
        session.execution_runtime.last_summary
    );
    assert!(session.execution_runtime.pending_reversal_entry.is_none());
    assert!(
        broker_rx.try_recv().is_err(),
        "same-bar revised side change should not queue a broker command"
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy closed-bar revision | hma_cross")
                && message.contains("same timestamp fingerprint changed")
                && message.contains("previous_fingerprint Some(0)")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | hma_cross | closed-bar already dispatched")
                && message.contains("signal Buy")
                && message.contains("Prior HMA Side: fast<slow")
                && message.contains("HMA Side: fast>slow")
    )));
}
