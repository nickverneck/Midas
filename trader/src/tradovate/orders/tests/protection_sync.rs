use super::*;

#[test]
fn trailing_stop_update_recovers_live_stop_order_id() {
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
            last_requested_take_profit_price: Some(6639.0),
            last_requested_stop_price: Some(6644.25),
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
                    "ordStatus": "Working"
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
                    "ordStatus": "Working"
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
    .expect("planner should succeed")
    .expect("stop update should queue a modify request");

    match sync.operation {
        ProtectionSyncOperation::ModifyStop { payload } => {
            assert_eq!(payload.get("orderId").and_then(Value::as_i64), Some(1002));
            assert_eq!(
                payload.get("stopPrice").and_then(Value::as_f64),
                Some(6644.5)
            );
        }
        _ => panic!("expected stop modification"),
    }
}

#[test]
fn trailing_stop_update_resyncs_when_live_stop_order_id_is_missing() {
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
            last_requested_take_profit_price: Some(6639.0),
            last_requested_stop_price: Some(6644.25),
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

    assert!(matches!(
        sync,
        Some(PendingProtectionSync {
            operation: ProtectionSyncOperation::Replace { .. },
            ..
        })
    ));
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
            last_requested_take_profit_price: Some(6639.0),
            last_requested_stop_price: Some(6644.25),
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
            last_requested_take_profit_price: Some(6639.0),
            last_requested_stop_price: Some(6644.25),
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
