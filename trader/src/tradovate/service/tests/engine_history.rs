use super::super::*;
use super::support::{test_session, test_state};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn broker_history_filters_manual_other_engine_and_other_contract_fills() {
    let mut session = test_session();
    session.market.value_per_point = Some(50.0);
    session.market.bars = vec![Bar {
        ts_ns: Utc::now().timestamp_nanos_opt().expect("timestamp"),
        open: 5_000.0,
        high: 5_002.0,
        low: 4_999.0,
        close: 5_001.0,
        volume: None,
    }];
    start_engine_run(&mut session).expect("engine run");
    let prefix = session
        .engine_run
        .as_ref()
        .expect("run")
        .order_prefix
        .clone();
    let now = Utc::now().to_rfc3339();
    session.user_store.orders.insert(
        42,
        BTreeMap::from([
            (100, json!({"id": 100, "accountId": 42, "contractId": 3570918, "symbol": "ESM6", "action": "Buy", "clOrdId": format!("{prefix}-1-entry")})),
            (101, json!({"id": 101, "accountId": 42, "contractId": 3570918, "symbol": "ESM6", "action": "Sell", "clOrdId": format!("{prefix}-2-entry")})),
            (102, json!({"id": 102, "accountId": 42, "contractId": 3570918, "symbol": "ESM6", "action": "Sell", "clOrdId": "manual-web-order"})),
            (103, json!({"id": 103, "accountId": 42, "contractId": 4095561, "symbol": "GCQ6", "action": "Buy", "clOrdId": format!("{prefix}-gc-entry")})),
            (104, json!({"id": 104, "accountId": 42, "contractId": 3570918, "symbol": "ESM6", "action": "Buy", "clOrdId": "midas-rother-engine-entry"})),
        ]),
    );
    for (fill_id, order_id, contract_id, symbol, action, price) in [
        (1, 100, 3_570_918, "ESM6", "Buy", 5_000.0),
        (2, 101, 3_570_918, "ESM6", "Sell", 5_001.0),
        (3, 102, 3_570_918, "ESM6", "Sell", 5_010.0),
        (4, 103, 4_095_561, "GCQ6", "Buy", 4_000.0),
        (5, 104, 3_570_918, "ESM6", "Buy", 4_999.0),
    ] {
        session.user_store.history_fills.insert(
            fill_id,
            json!({
                "id": fill_id,
                "orderId": order_id,
                "contractId": contract_id,
                "symbol": symbol,
                "action": action,
                "qty": 1,
                "price": price,
                "timestamp": now
            }),
        );
    }
    session
        .user_store
        .fill_fees
        .insert(900, json!({"id": 900, "fillId": 2, "amount": 2.0}));

    refresh_engine_history(&mut session);

    let history = &session.engine_run.as_ref().expect("run").history;
    assert_eq!(history.fills.len(), 2);
    assert_eq!(history.position_qty, 0);
    assert_eq!(history.fees, 2.0);
    assert_eq!(history.realized_pnl, 48.0);
    assert_eq!(history.wins, 1);
    assert_eq!(history.losses, 0);
    assert_eq!(
        history
            .fills
            .iter()
            .map(|fill| fill.fill_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn broker_history_fees_do_not_turn_a_gross_winner_into_a_loss() {
    let mut session = test_session();
    session.market.value_per_point = Some(50.0);
    start_engine_run(&mut session).expect("engine run");
    let prefix = session
        .engine_run
        .as_ref()
        .expect("run")
        .order_prefix
        .clone();
    let now = Utc::now().to_rfc3339();
    session.user_store.orders.insert(
        42,
        BTreeMap::from([
            (
                201,
                json!({
                    "id": 201,
                    "accountId": 42,
                    "contractId": 3570918,
                    "symbol": "ESM6",
                    "action": "Buy",
                    "clOrdId": format!("{prefix}-entry")
                }),
            ),
            (
                202,
                json!({
                    "id": 202,
                    "accountId": 42,
                    "contractId": 3570918,
                    "symbol": "ESM6",
                    "action": "Sell",
                    "clOrdId": format!("{prefix}-exit")
                }),
            ),
        ]),
    );
    session.user_store.history_fills.insert(
        203,
        json!({
            "id": 203,
            "orderId": 201,
            "contractId": 3570918,
            "symbol": "ESM6",
            "action": "Buy",
            "qty": 1,
            "price": 5000.0,
            "timestamp": now
        }),
    );
    session.user_store.history_fills.insert(
        204,
        json!({
            "id": 204,
            "orderId": 202,
            "contractId": 3570918,
            "symbol": "ESM6",
            "action": "Sell",
            "qty": 1,
            "price": 5000.02,
            "commission": -2.0,
            "timestamp": now
        }),
    );

    refresh_engine_history(&mut session);

    let history = &session.engine_run.as_ref().expect("run").history;
    assert!((history.realized_pnl + 1.0).abs() < 1e-9);
    assert_eq!(history.fees, 2.0);
    assert_eq!(history.wins, 1);
    assert_eq!(history.losses, 0);
}

#[test]
fn broker_history_attributes_broker_strategy_child_orders() {
    let mut session = test_session();
    session.market.value_per_point = Some(50.0);
    start_engine_run(&mut session).expect("engine run");
    let prefix = session
        .engine_run
        .as_ref()
        .expect("run")
        .order_prefix
        .clone();
    let now = Utc::now().to_rfc3339();
    session.user_store.order_strategies.insert(
        700,
        json!({
            "id": 700,
            "accountId": 42,
            "contractId": 3570918,
            "symbol": "ESM6",
            "uuid": format!("{prefix}-strategy")
        }),
    );
    session.user_store.order_strategy_links.insert(
        701,
        json!({"id": 701, "orderStrategyId": 700, "orderId": 702}),
    );
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            702,
            json!({
                "id": 702,
                "accountId": 42,
                "contractId": 3570918,
                "symbol": "ESM6",
                "action": "Buy"
            }),
        )]),
    );
    session.user_store.history_fills.insert(
        703,
        json!({
            "id": 703,
            "orderId": 702,
            "contractId": 3570918,
            "symbol": "ESM6",
            "qty": 1,
            "price": 5000.0,
            "timestamp": now
        }),
    );

    refresh_engine_history(&mut session);

    let history = &session.engine_run.as_ref().expect("run").history;
    assert_eq!(history.fills.len(), 1);
    assert_eq!(history.position_qty, 1);
    assert_eq!(history.average_entry_price, Some(5_000.0));
}

#[test]
fn engine_history_run_rejects_preexisting_selected_contract_position() {
    let mut session = test_session();
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "symbol": "ESM6",
                "netPos": 1,
                "netPrice": 5000.0
            }),
        )]),
    );

    let err = start_engine_run(&mut session).expect_err("position should block history start");

    assert!(err.to_string().contains("pre-existing broker position 1"));
    assert!(session.engine_run.is_none());
}

#[tokio::test]
async fn replay_state_reemits_existing_engine_history_for_tui_reattach() {
    let mut session = test_session();
    session.replay_enabled = true;
    start_engine_run(&mut session).expect("engine run");
    let expected_run_id = session.engine_run.as_ref().expect("run").run_id.clone();
    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    super::super::commands::replay_state(&mut state, &event_tx)
        .await
        .expect("replay state");

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::EngineHistoryUpdated(history) if history.run_id == expected_run_id
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::ReplayExecutionLedgerSnapshot(snapshot)
            if snapshot.schema_version == crate::broker::REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION
                && snapshot.fills.is_empty()
    )));
}

#[tokio::test]
async fn automated_liquidation_ack_registers_broker_order_with_engine_run() {
    let mut session = test_session();
    start_engine_run(&mut session).expect("engine run");
    let request_id = format!(
        "{}-liquidate",
        session.engine_run.as_ref().expect("run").order_prefix
    );
    let mut state = test_state(session);
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    super::super::internal::handle_broker_order_ack(
        BrokerOrderAck {
            endpoint: "order/liquidateposition",
            cl_ord_id: request_id,
            order_id: Some(808),
            submit_rtt_ms: 1,
            message: "close submitted".to_string(),
        },
        &mut state,
        &event_tx,
        internal_tx,
    );

    assert!(
        state
            .session
            .as_ref()
            .and_then(|session| session.engine_run.as_ref())
            .is_some_and(|run| run.owned_order_ids.contains(&808))
    );
}
