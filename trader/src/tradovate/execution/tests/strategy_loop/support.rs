use super::super::*;

pub(super) fn et_ts_ns(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
    chrono_tz::America::New_York
        .with_ymd_and_hms(year, month, day, hour, minute, 0)
        .single()
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap()
}

pub(super) fn seed_long_position(session: &mut SessionState) {
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "avgPrice": 5000.0
            }),
        )]),
    );
}

pub(super) fn seed_short_position(session: &mut SessionState) {
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -1,
                "avgPrice": 5000.0
            }),
        )]),
    );
}

pub(super) fn seed_live_strategy_child(session: &mut SessionState, order_strategy_id: i64) {
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id,
        target_qty: 0,
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
            "orderStrategyId": order_strategy_id,
            "orderId": 1001
        }),
    );
}
