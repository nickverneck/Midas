use super::*;

#[test]
fn native_strategy_entry_from_flat_ignores_stale_active_order_strategy() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key: StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        order_strategy_id: 429017060108,
        target_qty: -1,
    });
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_native_order_strategy_target(&mut session, &broker_tx, 1, "ema_cross signal")
            .expect("flat-to-long entry should queue order strategy");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.interrupt_order_strategy_id, None);
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.entry_order_qty, 1);
            assert_eq!(
                strategy.payload.get("accountId").and_then(Value::as_i64),
                Some(42)
            );
        }
        _ => panic!("expected order strategy command"),
    }
}

#[test]
fn native_strategy_entry_with_trailing_stop_uses_market_order_path() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_config.native_ema.stop_loss_ticks = 8.0;
    session.execution_config.native_ema.use_trailing_stop = true;
    session.market.tick_size = Some(0.25);
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_target_position_order(&mut session, &broker_tx, 1, true, "ema_cross signal")
            .expect("trailing configs should fall back to market entry + managed protection");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(
                order.payload.get("action").and_then(Value::as_str),
                Some("Buy")
            );
        }
        _ => panic!("expected market order command"),
    }
}

#[test]
fn automated_reversal_uses_market_order_path_for_direct_mode() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
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
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_native_order_strategy_target(&mut session, &broker_tx, -1, "ema_cross signal")
            .expect("non-flat direct reversal should queue a market order");

    match outcome {
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(-1),
        } => match broker_rx.try_recv().expect("broker command queued") {
            BrokerCommand::MarketOrder { order, .. } => {
                assert_eq!(order.interrupt_order_strategy_id, None);
                assert!(order.cancel_order_ids.is_empty());
                assert_eq!(order.target_qty, Some(-1));
                assert_eq!(order.order_qty, 2);
                assert_eq!(order.order_action, "Sell");
            }
            _ => panic!("expected market order command"),
        },
        _ => panic!("expected queued direct reversal"),
    }
}

#[test]
fn automated_reversal_interrupts_live_order_strategy_path() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
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
        dispatch_native_order_strategy_target(&mut session, &broker_tx, -1, "ema_cross signal")
            .expect("live reversal should interrupt the previous strategy path");

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
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Sell");
        }
        _ => panic!("expected market order command"),
    }
}

#[test]
fn automated_reversal_cancels_orphan_native_protection_when_strategy_path_missing() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
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
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            1001,
            json!({
                "id": 1001,
                "accountId": 42,
                "contractId": 3570918,
                "ordStatus": "Working",
                "clOrdId": "midas-orphan-tp"
            }),
        )]),
    );
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_native_order_strategy_target(&mut session, &broker_tx, -1, "ema_cross signal")
            .expect("reversal should clear orphan protection before starting a new strategy");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(-1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, None);
            assert_eq!(order.cancel_order_ids, vec![1001]);
            assert_eq!(order.target_qty, Some(-1));
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Sell");
        }
        _ => panic!("expected market order command"),
    }
}

#[test]
fn automated_reversal_flatten_mode_queues_flatten_before_entry() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
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
        dispatch_native_order_strategy_target(&mut session, &broker_tx, -1, "ema_cross signal")
            .expect("staged reversal should queue a flatten first");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(0)
        }
    ));
    assert_eq!(
        session
            .execution_runtime
            .pending_reversal_entry
            .as_ref()
            .map(|entry| entry.target_qty),
        Some(-1)
    );

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.interrupt_order_strategy_id, Some(77));
            assert_eq!(order.cancel_order_ids, vec![1001]);
            assert_eq!(order.target_qty, Some(0));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Sell");
        }
        _ => panic!("expected staged flatten market order"),
    }
}

#[test]
fn automated_reversal_closeall_mode_queues_liquidation_then_entry_strategy() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
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
        dispatch_native_order_strategy_target(&mut session, &broker_tx, -1, "ema_cross signal")
            .expect("immediate close-all reversal should queue both submits");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(-1)
        }
    ));
    assert!(session.execution_runtime.pending_reversal_entry.is_none());
    assert!(session.order_latency_tracker.is_some());

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::LiquidateThenOrderStrategy {
            liquidation,
            strategy,
            ..
        } => {
            assert_eq!(liquidation.target_qty, Some(-1));
            assert_eq!(liquidation.interrupt_order_strategy_id, None);
            assert!(liquidation.cancel_order_ids.is_empty());
            assert_eq!(
                liquidation
                    .payload
                    .get("contractId")
                    .and_then(Value::as_i64),
                Some(3570918)
            );
            assert_eq!(strategy.interrupt_order_strategy_id, None);
            assert!(strategy.cancel_order_ids.is_empty());
            assert_eq!(strategy.target_qty, -1);
            assert_eq!(strategy.entry_order_qty, 1);
            assert_eq!(
                strategy.payload.get("action").and_then(Value::as_str),
                Some("Sell")
            );
        }
        _ => panic!("expected close-all plus order strategy command"),
    }
}

#[test]
fn automated_native_target_uses_order_strategy_with_target_qty() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
    let (broker_tx, mut broker_rx) = unbounded_channel();

    let outcome =
        dispatch_target_position_order(&mut session, &broker_tx, 1, true, "ema_cross signal")
            .expect("automated target should queue order strategy");

    assert!(matches!(
        outcome,
        MarketOrderDispatchOutcome::Queued {
            target_qty: Some(1)
        }
    ));

    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.interrupt_order_strategy_id, None);
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.entry_order_qty, 1);
            assert_eq!(
                strategy.payload.get("action").and_then(Value::as_str),
                Some("Buy")
            );
        }
        _ => panic!("expected order strategy command"),
    }
}
