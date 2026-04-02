use super::*;
use serde_json::json;

#[test]
fn replay_bar_fills_take_profit_and_clears_sibling_strategy_orders() {
    let mut broker = ReplayBrokerState::default();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    broker.positions.insert(
        key,
        SimPosition {
            position_id: 30_001,
            qty: 1,
            avg_price: 6600.0,
            symbol: "ES 06-26".to_string(),
        },
    );

    let strategy_id = 40_001;
    broker.order_strategies.insert(
        strategy_id,
        SimOrderStrategyState {
            entity: json!({
                "id": strategy_id,
                "accountId": 42,
                "contractId": 3570918,
                "symbol": "ES 06-26",
                "status": "Active",
            }),
            order_ids: vec![1001, 1002],
            link_ids: vec![5001, 5002],
        },
    );
    broker.active_orders.insert(
        1001,
        SimActiveOrder {
            order: json!({
                "id": 1001,
                "orderId": 1001,
                "accountId": 42,
                "contractId": 3570918,
                "symbol": "ES 06-26",
                "action": "Sell",
                "orderQty": 1,
                "orderType": "Limit",
                "price": 6604.0,
                "ordStatus": "Working",
                "clOrdId": "midas-tp",
                "orderStrategyId": strategy_id,
            }),
            link_id: Some(5001),
            strategy_id: Some(strategy_id),
        },
    );
    broker.active_orders.insert(
        1002,
        SimActiveOrder {
            order: json!({
                "id": 1002,
                "orderId": 1002,
                "accountId": 42,
                "contractId": 3570918,
                "symbol": "ES 06-26",
                "action": "Sell",
                "orderQty": 1,
                "orderType": "Stop",
                "stopPrice": 6592.0,
                "ordStatus": "Working",
                "clOrdId": "midas-sl",
                "orderStrategyId": strategy_id,
            }),
            link_id: Some(5002),
            strategy_id: Some(strategy_id),
        },
    );

    let events = broker.simulate_replay_bar(&Bar {
        ts_ns: 123,
        open: 6601.0,
        high: 6604.5,
        low: 6599.5,
        close: 6604.0,
    });

    assert_eq!(events.len(), 1);
    let envelopes = match &events[0] {
        InternalEvent::UserEntities(envelopes) => envelopes,
        _ => panic!("expected replay bar to emit user entities"),
    };

    assert!(envelopes.iter().any(|envelope| {
        envelope.entity_type == "fill" && json_i64(&envelope.entity, "orderId") == Some(1001)
    }));
    assert!(envelopes.iter().any(|envelope| {
        envelope.entity_type == "order"
            && !envelope.deleted
            && json_i64(&envelope.entity, "orderId") == Some(1001)
            && envelope.entity.get("ordStatus").and_then(Value::as_str) == Some("Filled")
    }));
    assert!(envelopes.iter().any(|envelope| {
        envelope.entity_type == "order"
            && envelope.deleted
            && json_i64(&envelope.entity, "orderId") == Some(1002)
    }));
    assert!(envelopes.iter().any(|envelope| {
        envelope.entity_type == "orderStrategy"
            && envelope.deleted
            && json_i64(&envelope.entity, "id") == Some(strategy_id)
    }));
    assert!(
        envelopes
            .iter()
            .any(|envelope| envelope.entity_type == "position" && envelope.deleted)
    );
    assert!(broker.active_orders.is_empty());
    assert!(broker.order_strategies.is_empty());
    assert!(!broker.positions.contains_key(&key));
}
