use super::*;

#[test]
fn trailing_stop_update_does_not_modify_broker_owned_stop() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: Some(1001),
            stop_order_id: None,
        },
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
                    "orderType": "Limit",
                    "price": 6639.0,
                    "ordStatus": "Working",
                    "clOrdId": "midas-broker-tp"
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Stop",
                    "stopPrice": 6644.25,
                    "ordStatus": "Working",
                    "clOrdId": "midas-broker-sl"
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

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.5),
            reason: "ema_cross bar sync".to_string(),
        },
    )
    .expect("planner should succeed");

    assert!(
        sync.is_none(),
        "broker-owned stop updates must not be modified through app-managed protection sync"
    );
}

#[test]
fn trailing_stop_update_does_not_replace_when_live_stop_order_id_is_missing() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: Some(1001),
            stop_order_id: None,
        },
    );

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.5),
            reason: "ema_cross bar sync".to_string(),
        },
    )
    .expect("planner should succeed");

    assert!(
        sync.is_none(),
        "missing observed leg IDs must not trigger app-managed protection replacement"
    );
}

#[test]
fn protection_update_does_not_resubmit_when_live_take_profit_order_id_is_missing() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: None,
            stop_order_id: Some(1002),
        },
    );

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            reason: "ema_cross bar sync".to_string(),
        },
    )
    .expect("planner should succeed");

    assert!(
        sync.is_none(),
        "unchanged protection should not resubmit just because the live TP id is missing"
    );
}

#[test]
fn unchanged_native_protection_does_not_resubmit_when_ids_are_missing() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: None,
            stop_order_id: None,
        },
    );

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            reason: "ema_cross position sync".to_string(),
        },
    )
    .expect("planner should succeed");

    assert!(sync.is_none(), "unchanged protection should not resubmit");
}

#[test]
fn flat_native_protection_sync_clears_known_app_managed_orders() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6639.0),
            stop_price: Some(6644.25),
            take_profit_cl_ord_id: Some("midas-old-tp".to_string()),
            stop_cl_ord_id: Some("midas-old-sl".to_string()),
            take_profit_order_id: Some(1001),
            stop_order_id: Some(1002),
        },
    );

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 0,
            take_profit_price: None,
            stop_price: None,
            reason: "ema_cross flat".to_string(),
        },
    )
    .expect("planner should succeed")
    .expect("flat sync should clear known stale protection");

    let ProtectionSyncOperation::Clear { cancel_order_ids } = sync.operation;
    assert_eq!(cancel_order_ids, vec![1002, 1001]);
    assert!(sync.next_state.is_none());
}

#[test]
fn flat_native_protection_sync_does_not_cancel_broker_owned_linked_orders() {
    let mut session = test_session();
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
        BTreeMap::from([
            (
                1001,
                json!({
                    "id": 1001,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Limit",
                    "price": 6639.0,
                    "ordStatus": "Working",
                    "orderStrategyId": 77
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Stop",
                    "stopPrice": 6644.25,
                    "ordStatus": "Working",
                    "orderStrategyId": 77
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

    let sync = plan_native_protection_sync(
        &mut session,
        DesiredNativeProtection {
            key,
            account_name: "SIM".to_string(),
            contract_name: "ESM6".to_string(),
            signed_qty: 0,
            take_profit_price: None,
            stop_price: None,
            reason: "ema_cross flat".to_string(),
        },
    )
    .expect("planner should succeed");

    assert!(
        sync.is_none(),
        "flat clearing must not cancel broker-owned orderStrategy TP/SL children"
    );
}
