use super::*;

#[test]
fn automated_market_reversal_ignores_stale_bare_order_strategy() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
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
    session.user_store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working",
            "uuid": "midas-stale-strategy"
        }),
    );
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_target_position_order(&mut session, &broker_tx, -1, true, "ema_cross signal")
            .expect("stale bare strategy should not block market reversal fallback");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(-1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, None);
            assert_eq!(order.target_qty, Some(-1));
            assert_eq!(order.order_action, "Sell");
            assert_eq!(order.order_qty, 2);
        }
        _ => panic!("expected market order command"),
    }
}

#[test]
fn automated_market_reversal_interrupts_live_strategy_and_cancels_protection() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Lua;
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

    let outcome =
        dispatch_target_position_order(&mut session, &broker_tx, -1, true, "ema_cross signal")
            .expect("market reversal fallback should clear live strategy protection");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(-1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, Some(77));
            assert_eq!(order.cancel_order_ids, vec![1001]);
            assert_eq!(order.target_qty, Some(-1));
            assert_eq!(order.order_action, "Sell");
            assert_eq!(order.order_qty, 2);
        }
        _ => panic!("expected market order command"),
    }
}
