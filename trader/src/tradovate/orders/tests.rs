use super::super::*;
use super::*;
use serde_json::json;
use std::collections::BTreeMap;
use tokio::sync::mpsc::unbounded_channel;

fn test_session() -> SessionState {
    let (request_tx, _request_rx) = unbounded_channel();
    SessionState {
        cfg: AppConfig::default(),
        session_kind: SessionKind::Live,
        replay_enabled: false,
        tokens: TokenBundle {
            access_token: "access".to_string(),
            md_access_token: "md".to_string(),
            expiration_time: None,
            user_id: None,
            user_name: None,
        },
        accounts: vec![AccountInfo {
            id: 42,
            name: "SIM".to_string(),
            raw: json!({}),
        }],
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store: UserSyncStore::default(),
        selected_account_id: Some(42),
        selected_contract: Some(ContractSuggestion {
            id: 3570918,
            name: "ESM6".to_string(),
            description: "E-mini S&P".to_string(),
            raw: json!({}),
        }),
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    }
}

#[test]
fn price_offset_from_ticks_uses_tick_size() {
    assert_eq!(price_offset_from_ticks(8.0, Some(0.25)), Some(2.0));
    assert_eq!(price_offset_from_ticks(0.0, Some(0.25)), None);
    assert_eq!(price_offset_from_ticks(8.0, Some(0.0)), None);
}

#[test]
fn signed_bracket_offsets_match_order_side() {
    assert_eq!(signed_profit_target_offset("Buy", 2.0), 2.0);
    assert_eq!(signed_profit_target_offset("Sell", 2.0), -2.0);
    assert_eq!(signed_stop_loss_offset("Buy", 2.0), -2.0);
    assert_eq!(signed_stop_loss_offset("Sell", 2.0), 2.0);
}

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
