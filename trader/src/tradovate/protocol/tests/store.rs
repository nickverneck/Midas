use super::*;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn user_store_tracks_active_order_strategy_and_linked_orders() {
    let mut store = UserSyncStore::default();
    store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "ActiveStrategy",
            "uuid": "midas-1710546400000-1-strategy"
        }),
    );
    store.orders.insert(
        42,
        BTreeMap::from([
            (
                101,
                json!({
                    "id": 101,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Limit",
                    "price": 5010.0
                }),
            ),
            (
                102,
                json!({
                    "id": 102,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Stop",
                    "stopPrice": 4990.0
                }),
            ),
        ]),
    );
    store.order_strategy_links.insert(
        1,
        json!({
            "id": 1,
            "orderStrategyId": 77,
            "orderId": 101,
            "label": "tp"
        }),
    );
    store.order_strategy_links.insert(
        2,
        json!({
            "id": 2,
            "orderStrategyId": 77,
            "orderId": 102,
            "label": "sl"
        }),
    );

    let strategy = store
        .find_active_order_strategy(42, 3570918)
        .expect("strategy should be tracked");
    let linked = store.linked_strategy_orders(42, 77);

    assert_eq!(extract_entity_id(strategy), Some(77));
    assert_eq!(linked.len(), 2);
    assert_eq!(linked[0].get("id").and_then(Value::as_i64), Some(101));
    assert_eq!(linked[1].get("id").and_then(Value::as_i64), Some(102));
}

#[test]
fn user_store_treats_nonterminal_midas_order_strategy_status_as_active() {
    let mut store = UserSyncStore::default();
    store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working",
            "uuid": "midas-1710546400000-1-strategy"
        }),
    );

    let strategy = store
        .find_active_order_strategy(42, 3570918)
        .expect("working strategy should be tracked");

    assert_eq!(extract_entity_id(strategy), Some(77));
}

#[test]
fn user_store_recovers_midas_strategy_from_linked_active_orders() {
    let mut store = UserSyncStore::default();
    store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "status": "Working",
            "uuid": "midas-1710546400000-1-strategy"
        }),
    );
    store.orders.insert(
        42,
        BTreeMap::from([(
            101,
            json!({
                "id": 101,
                "accountId": 42,
                "contractId": 3570918,
                "orderType": "Stop",
                "ordStatus": "Working"
            }),
        )]),
    );
    store.order_strategy_links.insert(
        1,
        json!({
            "id": 1,
            "orderStrategyId": 77,
            "orderId": 101,
            "label": "sl"
        }),
    );

    let strategy = store
        .find_active_order_strategy(42, 3570918)
        .expect("linked active order should recover the strategy");

    assert_eq!(extract_entity_id(strategy), Some(77));
}

#[test]
fn user_store_ignores_terminal_midas_order_strategy_status() {
    let mut store = UserSyncStore::default();
    store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "InterruptedStrategy",
            "uuid": "midas-1710546400000-1-strategy"
        }),
    );

    assert!(store.find_active_order_strategy(42, 3570918).is_none());
}

#[test]
fn user_store_skips_live_fills_but_keeps_replay_fills() {
    let mut store = UserSyncStore::default();
    store.apply(EntityEnvelope {
        entity_type: "fill".to_string(),
        deleted: false,
        entity: json!({
            "id": 11,
            "accountId": 42,
            "contractId": 3570918,
            "buySell": "Buy",
            "price": 5000.0,
            "qty": 1,
            "timestamp": 1
        }),
    });
    assert!(store.fills.get(&42).is_none());

    store.apply(EntityEnvelope {
        entity_type: "fill".to_string(),
        deleted: false,
        entity: json!({
            "id": 12,
            "accountId": 42,
            "contractId": 3570918,
            "source": "replay",
            "buySell": "Buy",
            "price": 5000.0,
            "qty": 1,
            "timestamp": 2
        }),
    });
    assert_eq!(store.fills.get(&42).map(BTreeMap::len), Some(1));
}

#[test]
fn build_snapshots_include_realized_pnl_and_protection_prices() {
    let mut store = UserSyncStore::default();
    store.risk.insert(
        42,
        json!({
            "accountId": 42,
            "balance": 10000.0,
            "realizedPnL": 125.5
        }),
    );
    store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "netPrice": 5000.0
            }),
        )]),
    );
    let market = MarketSnapshot {
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        bars: vec![Bar {
            ts_ns: 0,
            open: 4999.0,
            high: 5002.0,
            low: 4998.0,
            close: 5001.0,
        }],
        trade_markers: Vec::new(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 1,
        live_bars: 0,
        status: String::new(),
    };
    let managed_protection = BTreeMap::from([(
        StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(5004.0),
            stop_price: Some(4998.0),
            last_requested_take_profit_price: Some(5004.0),
            last_requested_stop_price: Some(4998.0),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: None,
            stop_order_id: None,
        },
    )]);
    let accounts = vec![AccountInfo {
        id: 42,
        name: "sim".to_string(),
        raw: json!({ "id": 42, "name": "sim" }),
    }];

    let snapshots = store.build_snapshots(&accounts, Some(&market), &managed_protection);
    let snapshot = snapshots.first().expect("snapshot should exist");

    assert_eq!(snapshot.realized_pnl, Some(125.5));
    assert_eq!(snapshot.market_entry_price, Some(5000.0));
    assert_eq!(snapshot.selected_contract_take_profit_price, Some(5004.0));
    assert_eq!(snapshot.selected_contract_stop_price, Some(4998.0));
}

#[test]
fn replay_snapshots_mark_to_market_open_positions() {
    let mut store = UserSyncStore::default();
    store.positions.insert(
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
    store.fills.insert(
        42,
        BTreeMap::from([(
            11,
            json!({
                "id": 11,
                "accountId": 42,
                "contractId": 3570918,
                "buySell": "Buy",
                "price": 5000.0,
                "qty": 1,
                "timestamp": 1
            }),
        )]),
    );
    let accounts = vec![AccountInfo {
        id: 42,
        name: "REPLAY".to_string(),
        raw: json!({
            "id": 42,
            "name": "REPLAY",
            "source": "replay",
            "startingBalance": 100000.0
        }),
    }];
    let market = MarketSnapshot {
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        bars: vec![Bar {
            ts_ns: 0,
            open: 5000.0,
            high: 5001.5,
            low: 4999.5,
            close: 5001.0,
        }],
        trade_markers: Vec::new(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 1,
        live_bars: 0,
        status: String::new(),
    };

    let snapshots = store.build_snapshots(&accounts, Some(&market), &BTreeMap::new());
    let snapshot = snapshots.first().expect("snapshot should exist");

    assert_eq!(snapshot.realized_pnl, Some(0.0));
    assert_eq!(snapshot.unrealized_pnl, Some(50.0));
    assert_eq!(snapshot.balance, Some(100000.0));
    assert_eq!(snapshot.cash_balance, Some(100000.0));
    assert_eq!(snapshot.net_liq, Some(100050.0));
}

#[test]
fn replay_snapshots_roll_realized_pnl_into_balance() {
    let mut store = UserSyncStore::default();
    store.fills.insert(
        42,
        BTreeMap::from([
            (
                11,
                json!({
                    "id": 11,
                    "accountId": 42,
                    "contractId": 3570918,
                    "buySell": "Buy",
                    "price": 5000.0,
                    "qty": 1,
                    "timestamp": 1
                }),
            ),
            (
                12,
                json!({
                    "id": 12,
                    "accountId": 42,
                    "contractId": 3570918,
                    "buySell": "Sell",
                    "price": 5002.0,
                    "qty": 1,
                    "timestamp": 2
                }),
            ),
        ]),
    );
    let accounts = vec![AccountInfo {
        id: 42,
        name: "REPLAY".to_string(),
        raw: json!({
            "id": 42,
            "name": "REPLAY",
            "source": "replay",
            "startingBalance": 100000.0
        }),
    }];
    let market = MarketSnapshot {
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        bars: vec![Bar {
            ts_ns: 0,
            open: 5002.0,
            high: 5002.0,
            low: 5002.0,
            close: 5002.0,
        }],
        trade_markers: Vec::new(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 1,
        live_bars: 0,
        status: String::new(),
    };

    let snapshots = store.build_snapshots(&accounts, Some(&market), &BTreeMap::new());
    let snapshot = snapshots.first().expect("snapshot should exist");

    assert_eq!(snapshot.realized_pnl, Some(100.0));
    assert_eq!(snapshot.unrealized_pnl, Some(0.0));
    assert_eq!(snapshot.balance, Some(100100.0));
    assert_eq!(snapshot.cash_balance, Some(100100.0));
    assert_eq!(snapshot.net_liq, Some(100100.0));
}
