use super::*;

#[test]
fn reconcile_keeps_known_strategy_id_while_position_is_open() {
    let mut session = test_session();
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
                "netPos": -1
            }),
        )]),
    );

    reconcile_selected_active_order_strategy(&mut session);

    assert_eq!(
        session
            .active_order_strategy
            .as_ref()
            .map(|tracked| tracked.order_strategy_id),
        Some(77)
    );
}

#[test]
fn waits_for_strategy_owned_protection_during_hydration_grace_without_linked_orders() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
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
                "netPos": -1
            }),
        )]),
    );
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-hydrating-strategy".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    assert!(should_wait_for_strategy_owned_protection(&session));
}

#[test]
fn waits_for_strategy_owned_protection_when_linked_orders_exist() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
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
                "netPos": -1
            }),
        )]),
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
                    "ordStatus": "Working"
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working",
                    "orderType": "Limit",
                    "price": 4998.0,
                    "clOrdId": "midas-tp"
                }),
            ),
            (
                1003,
                json!({
                    "id": 1003,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working",
                    "orderType": "Stop",
                    "stopPrice": 5002.0,
                    "clOrdId": "midas-sl"
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

    assert!(should_wait_for_strategy_owned_protection(&session));
}
