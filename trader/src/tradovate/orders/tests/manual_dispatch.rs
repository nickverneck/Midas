use super::*;

#[test]
fn manual_close_uses_liquidate_position() {
    let mut session = test_session();
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
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome = dispatch_manual_order(&mut session, &broker_tx, ManualOrderAction::Close)
        .expect("close should queue liquidation");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(0)
        }
    ));
    assert!(session.order_submit_in_flight);
    assert!(session.order_latency_tracker.is_none());

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::LiquidatePosition { liquidation, .. } => {
            assert_eq!(liquidation.target_qty, Some(0));
            assert_eq!(liquidation.interrupt_order_strategy_id, None);
            assert!(liquidation.cancel_order_ids.is_empty());
            assert_eq!(
                liquidation.payload.get("accountId").and_then(Value::as_i64),
                Some(42)
            );
            assert_eq!(
                liquidation
                    .payload
                    .get("contractId")
                    .and_then(Value::as_i64),
                Some(3570918)
            );
            assert_eq!(
                liquidation.payload.get("admin").and_then(Value::as_bool),
                Some(false)
            );
        }
        _ => panic!("expected liquidation command"),
    }
}

#[test]
fn manual_buy_from_flat_ignores_stale_active_order_strategy() {
    let mut session = test_session();
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key: StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        order_strategy_id: 429017060108,
        target_qty: -1,
    });
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome = dispatch_manual_order(&mut session, &broker_tx, ManualOrderAction::Buy)
        .expect("manual buy should queue market order");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued { target_qty: None }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, None);
            assert_eq!(
                order.payload.get("action").and_then(Value::as_str),
                Some("Buy")
            );
        }
        _ => panic!("expected market order command"),
    }
}

#[test]
fn manual_buy_interrupts_live_strategy_and_cancels_protection() {
    let mut session = test_session();
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
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key: StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
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
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome = dispatch_manual_order(&mut session, &broker_tx, ManualOrderAction::Buy)
        .expect("manual buy should clear live strategy protection before re-entry");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued { target_qty: None }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, Some(77));
            assert_eq!(order.cancel_order_ids, vec![1001]);
            assert_eq!(
                order.payload.get("action").and_then(Value::as_str),
                Some("Buy")
            );
        }
        _ => panic!("expected market order command"),
    }
}
