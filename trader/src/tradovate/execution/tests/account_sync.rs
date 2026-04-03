use super::*;

#[test]
fn account_sync_waits_when_position_is_temporarily_flat_but_strategy_orders_are_live() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaAngle;
    session.execution_config.native_hma.take_profit_ticks = 30.0;
    session.execution_config.native_hma.stop_loss_ticks = 30.0;
    session.execution_runtime.armed = true;
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
        }),
    );
    session.user_store.orders.insert(
        42,
        BTreeMap::from([
            (
                1001,
                json!({
                    "id": 1001,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderStrategyId": 77,
                    "ordStatus": "Working",
                    "orderType": "Limit",
                    "price": 5030.0,
                    "action": "Sell"
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderStrategyId": 77,
                    "ordStatus": "Working",
                    "orderType": "Stop",
                    "stopPrice": 4970.0,
                    "action": "Sell"
                }),
            ),
        ]),
    );
    session.user_store.order_strategy_links.insert(
        1,
        json!({
            "id": 1,
            "orderStrategyId": 77,
            "orderId": 1001
        }),
    );
    session.user_store.order_strategy_links.insert(
        2,
        json!({
            "id": 2,
            "orderStrategyId": 77,
            "orderId": 1002
        }),
    );

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("flat hydration gap should wait for broker sync");

    assert!(
        broker_rx.try_recv().is_err(),
        "account sync should not clear broker-owned protection during the hydration gap"
    );
    assert_eq!(
        session.execution_runtime.last_summary,
        "Waiting for broker sync: flat position reported while an order path is still active."
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::ExecutionState(state)
            if state.runtime.last_summary
                == "Waiting for broker sync: flat position reported while an order path is still active."
    )));
}

#[test]
fn account_sync_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_runtime.armed = true;
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

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("live strategy path should suppress transient drift disarm");

    assert!(session.execution_runtime.armed);
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync")
    );
}

#[test]
fn broker_sync_wait_logs_once_with_observability_context() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: -2,
            take_profit_price: Some(4998.0),
            stop_price: Some(5002.0),
            last_requested_take_profit_price: Some(4998.0),
            last_requested_stop_price: Some(5002.0),
            take_profit_cl_ord_id: Some("midas-tp".to_string()),
            stop_cl_ord_id: Some("midas-sl".to_string()),
            take_profit_order_id: Some(1002),
            stop_order_id: Some(1003),
        },
    );
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

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("first wait transition should emit debug context");
    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("repeated wait transition should not duplicate debug context");

    let wait_logs = std::iter::from_fn(|| event_rx.try_recv().ok())
        .filter_map(|event| match event {
            ServiceEvent::DebugLog(message) if message.contains("execution broker sync wait") => {
                Some(message)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let unique_wait_logs = wait_logs
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(unique_wait_logs.len(), 1);
    let wait_log = unique_wait_logs
        .first()
        .expect("expected broker sync wait log");
    assert!(wait_log.contains("tracked strategy 77 (1 active linked)"));
    assert!(wait_log.contains("broker strategy none (0 active linked)"));
    assert!(wait_log.contains("tp 4998.00 [order 1002 clOrdId midas-tp]"));
    assert!(wait_log.contains("sl 5002.00 [order 1003 clOrdId midas-sl]"));
}

#[test]
fn duplicate_symbol_position_record_does_not_trigger_false_drift_disarm() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.order_qty = 1;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_summary = "Armed".to_string();
    session.user_store.positions.insert(
        42,
        BTreeMap::from([
            (
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 6659.75
                }),
            ),
            (
                2,
                json!({
                    "id": 2,
                    "accountId": 42,
                    "symbol": "ESH6",
                    "netPos": 1,
                    "avgPrice": 6659.75
                }),
            ),
        ]),
    );

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("duplicate mirror records should not disarm");

    assert!(session.execution_runtime.armed);
    assert_eq!(selected_market_position_qty(&session), 1);
    assert_eq!(selected_market_entry_price(&session), Some(6659.75));
}
