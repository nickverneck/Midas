use super::*;

#[test]
fn stale_pending_target_clears_when_broker_has_no_live_order_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(100);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(4),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-stale-entry".to_string(),
        strategy_owned_protection: false,
        order_id: Some(77),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 100,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("stale pending target should reconcile");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(99));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target 1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("Pending target 1 cleared")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-stale-entry")
                && message.contains("pending target 1")
    )));
}

#[test]
fn strategy_loop_clears_stale_pending_target_when_broker_path_missing() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(11),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 0 -> -1)".to_string()),
        cl_ord_id: "midas-stale-strategy-loop".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("strategy loop should clear stale pending target");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(199));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("Pending target -1 cleared")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-stale-strategy-loop")
                && message.contains("pending target -1")
    )));
}

#[test]
fn strategy_loop_keeps_pending_target_during_broker_path_grace_after_ack() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(300);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-fresh-strategy-loop".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 300,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("fresh order-strategy ack should keep pending target briefly");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.order_latency_tracker.is_some());
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for prior automated order to settle")
    );
}

#[test]
fn strategy_loop_keeps_market_order_pending_target_during_position_sync_grace() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(320);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1
            }),
        )]),
    );
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-market-sync".to_string(),
        strategy_owned_protection: false,
        order_id: Some(88),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 320,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("market-order reversal should wait for position sync");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(-1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(320));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for position sync after automated order settle")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution market position sync wait")
                || message.contains("Waiting for position sync after automated order settle")
    )));
}

#[test]
fn strategy_loop_keeps_order_strategy_pending_target_during_position_sync_grace() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(325);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(3),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-strategy-position-sync".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 325,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("order-strategy entry should wait for position sync");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(325));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for position sync after automated order settle")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution market position sync wait")
                || message.contains("Waiting for position sync after automated order settle")
    )));
}

#[test]
fn strategy_loop_clears_market_order_pending_target_after_position_sync_grace_expires() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(330);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1
            }),
        )]),
    );
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(4),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-market-sync-stale".to_string(),
        strategy_owned_protection: false,
        order_id: Some(99),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 330,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("stale market-order pending target should eventually clear");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(329));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-market-sync-stale")
    )));
}
