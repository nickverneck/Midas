use super::super::*;

#[test]
fn strategy_loop_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(100);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -2,
                "netPrice": 5000.0
            }),
        )]),
    );
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
        .expect("active strategy path should suppress transient drift disarm");

    assert!(session.execution_runtime.armed);
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync")
    );
}

#[test]
fn strategy_loop_waits_when_flat_qty_conflicts_with_live_broker_path() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 10;
    session.execution_config.native_ema.slow_length = 30;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.contract_id = Some(3570918);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 0,
            }),
        )]),
    );
    session.market.history_loaded = 31;
    session.market.bars = (0..31)
        .map(|idx| Bar {
            ts_ns: idx + 1,
            open: 6410.0,
            high: 6411.0,
            low: 6409.0,
            close: if idx < 30 { 6410.0 } else { 6420.0 },
            volume: None,
        })
        .collect();
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(30);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: None,
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-hydrating-flat-strategy".to_string(),
        order_id: Some(1001),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: false,
    });
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
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
        .expect("flat/local qty with live broker path should wait instead of opening");

    assert!(broker_rx.try_recv().is_err(), "no order should be queued");
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("flat position reported while an order path is still active")
    );
}

#[test]
fn strategy_loop_clears_stale_flat_broker_path_after_sync_grace() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 10;
    session.execution_config.native_ema.slow_length = 30;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.contract_id = Some(3570918);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 0,
            }),
        )]),
    );
    session.market.history_loaded = 31;
    session.market.bars = (0..31)
        .map(|idx| Bar {
            ts_ns: idx + 1,
            open: 6410.0,
            high: 6411.0,
            low: 6409.0,
            close: if idx < 30 { 6410.0 } else { 6420.0 },
            volume: None,
        })
        .collect();
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(30);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now()
            - Duration::from_millis((ORDER_STRATEGY_POSITION_SYNC_GRACE_MS + 1) as u64),
        signal_started_at: None,
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-stale-flat-strategy".to_string(),
        order_id: Some(1001),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.user_store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working",
            "uuid": "midas-stale-flat-strategy"
        }),
    );
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            1001,
            json!({
                "id": 1001,
                "accountId": 42,
                "contractId": 3570918,
                "orderStrategyId": 77,
                "ordStatus": "Working",
                "orderType": "Limit",
                "price": 6422.50,
                "action": "Sell",
                "clOrdId": "midas-stale-tp"
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
        .expect("stale flat broker path should be cleared by the next entry");

    match broker_rx
        .try_recv()
        .expect("entry strategy should be queued")
    {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.interrupt_order_strategy_id, Some(77));
            assert!(strategy.cancel_order_ids.is_empty());
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.order_action, "Buy");
        }
        _ => panic!("expected order strategy entry"),
    }
}
