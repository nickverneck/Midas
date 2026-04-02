use super::*;
use serde_json::{Value, json};

fn test_session() -> SessionState {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
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
            name: "ESH6".to_string(),
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
fn reconcile_keeps_known_strategy_id_while_position_is_open() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
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

    reconcile_selected_active_order_strategy(&mut session);

    assert_eq!(
        session
            .active_order_strategy
            .as_ref()
            .map(|tracked| tracked.order_strategy_id),
        Some(77)
    );
}

#[test]
fn waits_for_strategy_owned_protection_during_hydration_grace_without_linked_orders() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-hydrating-strategy".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    assert!(should_wait_for_strategy_owned_protection(&session));
}

#[test]
fn stale_pending_target_clears_when_broker_has_no_live_order_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(100);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(4),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-stale-entry".to_string(),
        strategy_owned_protection: false,
        order_id: Some(77),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 100,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("stale pending target should reconcile");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(99));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target 1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("Pending target 1 cleared")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-stale-entry")
                && message.contains("pending target 1")
    )));
}

#[test]
fn strategy_loop_clears_stale_pending_target_when_broker_path_missing() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(11),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 0 -> -1)".to_string()),
        cl_ord_id: "midas-stale-strategy-loop".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("strategy loop should clear stale pending target");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(199));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("Pending target -1 cleared")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-stale-strategy-loop")
                && message.contains("pending target -1")
    )));
}

#[test]
fn strategy_loop_keeps_pending_target_during_broker_path_grace_after_ack() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(300);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-fresh-strategy-loop".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.market.bars = vec![Bar {
        ts_ns: 300,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("fresh order-strategy ack should keep pending target briefly");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.order_latency_tracker.is_some());
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for prior automated order to settle")
    );
}

#[test]
fn strategy_loop_keeps_market_order_pending_target_during_position_sync_grace() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(320);
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-market-sync".to_string(),
        strategy_owned_protection: false,
        order_id: Some(88),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 320,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("market-order reversal should wait for position sync");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(-1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(320));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for position sync after automated order settle")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution market position sync wait")
                || message.contains("Waiting for position sync after automated order settle")
    )));
}

#[test]
fn strategy_loop_keeps_order_strategy_pending_target_during_position_sync_grace() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(325);
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(3),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-strategy-position-sync".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 325,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("order-strategy entry should wait for position sync");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(325));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for position sync after automated order settle")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution market position sync wait")
                || message.contains("Waiting for position sync after automated order settle")
    )));
}

#[test]
fn strategy_loop_clears_market_order_pending_target_after_position_sync_grace_expires() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(330);
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - time::Duration::from_secs(4),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-market-sync-stale".to_string(),
        strategy_owned_protection: false,
        order_id: Some(99),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.market.bars = vec![Bar {
        ts_ns: 330,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    }];
    session.market.history_loaded = 1;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("stale market-order pending target should eventually clear");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(329));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared")
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("pending target cleared")
                && message.contains("request midas-market-sync-stale")
    )));
}

#[test]
fn waits_for_strategy_owned_protection_when_linked_orders_exist() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
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
    session.user_store.orders.insert(
        42,
        BTreeMap::from([
            (
                1001,
                json!({
                    "id": 1001,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working"
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working",
                    "orderType": "Limit",
                    "price": 4998.0,
                    "clOrdId": "midas-tp"
                }),
            ),
            (
                1003,
                json!({
                    "id": 1003,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working",
                    "orderType": "Stop",
                    "stopPrice": 5002.0,
                    "clOrdId": "midas-sl"
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

    assert!(should_wait_for_strategy_owned_protection(&session));
}

#[test]
fn sync_execution_protection_waits_for_hydrating_order_strategy_before_ack() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_config.native_ema.stop_loss_ticks = 8.0;
    session.execution_runtime.armed = true;
    session.market.tick_size = Some(0.25);
    session.user_store.positions.insert(
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty -1 -> 1)".to_string()),
        cl_ord_id: "midas-hydrating-strategy".to_string(),
        strategy_owned_protection: true,
        order_id: Some(88),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    sync_execution_protection(&mut session, &broker_tx, None)
        .expect("hydrating strategy-owned protection should block manual sync");

    assert!(
        broker_rx.try_recv().is_err(),
        "manual protection sync should not queue while broker-owned bracket is still hydrating"
    );
}

#[test]
fn direct_reversal_syncs_native_protection_at_net_position_size() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.execution_config.order_qty = 1;
    session.market.tick_size = Some(0.25);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -1,
                "netPrice": 6545.0
            }),
        )]),
    );
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-direct-reversal".to_string(),
        strategy_owned_protection: false,
        order_id: Some(77),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    sync_execution_protection(&mut session, &broker_tx, None)
        .expect("direct reversal should queue native TP/SL at net position size");

    match broker_rx
        .try_recv()
        .expect("expected native protection command")
    {
        BrokerCommand::NativeProtection { sync, .. } => match sync.operation {
            ProtectionSyncOperation::Replace {
                request: ProtectionPlaceRequest::Oco { payload },
                ..
            } => {
                assert_eq!(payload.get("orderQty").and_then(Value::as_i64), Some(1));
                assert_eq!(
                    payload
                        .get("other")
                        .and_then(|other| other.get("orderQty"))
                        .and_then(Value::as_i64),
                    Some(1)
                );
            }
            _ => panic!("expected OCO protection sync"),
        },
        _ => panic!("expected native protection command"),
    }
}

#[test]
fn stale_strategy_record_without_live_child_orders_does_not_block_direct_reversal_sync() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.execution_config.order_qty = 1;
    session.market.tick_size = Some(0.25);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.user_store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "ActiveStrategy",
            "customTag50": "midas-stale-strategy",
        }),
    );
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -1,
                "netPrice": 6545.0
            }),
        )]),
    );
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-direct-reversal".to_string(),
        strategy_owned_protection: false,
        order_id: Some(77),
        order_strategy_id: None,
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    sync_execution_protection(&mut session, &broker_tx, None)
        .expect("stale strategy records should not block direct reversal protection sync");

    assert!(matches!(
        broker_rx.try_recv(),
        Ok(BrokerCommand::NativeProtection { .. })
    ));
}

#[test]
fn account_sync_waits_when_position_is_temporarily_flat_but_strategy_orders_are_live() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaAngle;
    session.execution_config.native_hma.take_profit_ticks = 30.0;
    session.execution_config.native_hma.stop_loss_ticks = 30.0;
    session.execution_runtime.armed = true;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.user_store.order_strategies.insert(
        77,
        json!({
            "id": 77,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working",
        }),
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
                    "orderStrategyId": 77,
                    "ordStatus": "Working",
                    "orderType": "Limit",
                    "price": 5030.0,
                    "action": "Sell"
                }),
            ),
            (
                1002,
                json!({
                    "id": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderStrategyId": 77,
                    "ordStatus": "Working",
                    "orderType": "Stop",
                    "stopPrice": 4970.0,
                    "action": "Sell"
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

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("flat hydration gap should wait for broker sync");

    assert!(
        broker_rx.try_recv().is_err(),
        "account sync should not clear broker-owned protection during the hydration gap"
    );
    assert_eq!(
        session.execution_runtime.last_summary,
        "Waiting for broker sync: flat position reported while an order path is still active."
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::ExecutionState(state)
            if state.runtime.last_summary
                == "Waiting for broker sync: flat position reported while an order path is still active."
    )));
}

#[test]
fn account_sync_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_runtime.armed = true;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -2,
                "netPrice": 5000.0
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

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("live strategy path should suppress transient drift disarm");

    assert!(session.execution_runtime.armed);
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync")
    );
}

#[test]
fn broker_sync_wait_logs_once_with_observability_context() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_runtime.armed = true;
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: -2,
            take_profit_price: Some(4998.0),
            stop_price: Some(5002.0),
            last_requested_take_profit_price: Some(4998.0),
            last_requested_stop_price: Some(5002.0),
            take_profit_cl_ord_id: Some("midas-tp".to_string()),
            stop_cl_ord_id: Some("midas-sl".to_string()),
            take_profit_order_id: Some(1002),
            stop_order_id: Some(1003),
        },
    );
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -2,
                "netPrice": 5000.0
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

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("first wait transition should emit debug context");
    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("repeated wait transition should not duplicate debug context");

    let wait_logs = std::iter::from_fn(|| event_rx.try_recv().ok())
        .filter_map(|event| match event {
            ServiceEvent::DebugLog(message) if message.contains("execution broker sync wait") => {
                Some(message)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let unique_wait_logs = wait_logs
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(unique_wait_logs.len(), 1);
    let wait_log = unique_wait_logs
        .first()
        .expect("expected broker sync wait log");
    assert!(wait_log.contains("tracked strategy 77 (1 active linked)"));
    assert!(wait_log.contains("broker strategy none (0 active linked)"));
    assert!(wait_log.contains("tp 4998.00 [order 1002 clOrdId midas-tp]"));
    assert!(wait_log.contains("sl 5002.00 [order 1003 clOrdId midas-sl]"));
}

#[test]
fn strategy_loop_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(100);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -2,
                "netPrice": 5000.0
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("active strategy path should suppress transient drift disarm");

    assert!(session.execution_runtime.armed);
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync")
    );
}

#[test]
fn live_bar_signal_timing_can_trade_on_forming_range_bar() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.market.history_loaded = 5;
    session.market.bars = vec![
        Bar {
            ts_ns: 1,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 2,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 3,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 4,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 5,
            open: 8.0,
            high: 8.5,
            low: 7.5,
            close: 8.0,
        },
        Bar {
            ts_ns: 6,
            open: 12.0,
            high: 12.5,
            low: 11.5,
            close: 12.0,
        },
    ];
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("live timing should evaluate the forming bar");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from live-bar buy signal"),
    }
}

#[test]
fn strategy_loop_waits_when_flat_qty_conflicts_with_live_broker_path() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 10;
    session.execution_config.native_ema.slow_length = 30;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.contract_id = Some(3570918);
    session.market.history_loaded = 31;
    session.market.bars = (0..31)
        .map(|idx| Bar {
            ts_ns: idx + 1,
            open: 6410.0,
            high: 6411.0,
            low: 6409.0,
            close: if idx < 30 { 6410.0 } else { 6420.0 },
        })
        .collect();
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(30);
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("flat/local qty with live broker path should wait instead of opening");

    assert!(broker_rx.try_recv().is_err(), "no order should be queued");
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("flat position reported while an order path is still active")
    );
}

#[test]
fn strategy_loop_waits_for_position_sync_after_native_protection_activity() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_config.native_ema.take_profit_ticks = 10.0;
    session.execution_config.native_ema.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.history_loaded = 5;
    session.market.bars = vec![
        Bar {
            ts_ns: 1,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 2,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 3,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 4,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 5,
            open: 8.0,
            high: 8.5,
            low: 7.5,
            close: 8.0,
        },
        Bar {
            ts_ns: 6,
            open: 12.0,
            high: 12.5,
            low: 11.5,
            close: 12.0,
        },
    ];
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
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
    session.managed_protection.insert(
        StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        ManagedProtectionOrders {
            signed_qty: -1,
            take_profit_price: Some(4997.5),
            stop_price: Some(5002.5),
            last_requested_take_profit_price: Some(4997.5),
            last_requested_stop_price: Some(5002.5),
            take_profit_cl_ord_id: Some("midas-short-tp".to_string()),
            stop_cl_ord_id: Some("midas-short-sl".to_string()),
            take_profit_order_id: Some(1001),
            stop_order_id: Some(1002),
        },
    );

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("missing native protection legs should pause signal evaluation");

    assert!(broker_rx.try_recv().is_err(), "no order should be queued");
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync after native protection activity")
    );
}

#[test]
fn execution_state_snapshot_includes_selected_protection_prices() {
    let mut session = test_session();
    session.market.contract_id = Some(3570918);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "avgPrice": 6659.75
            }),
        )]),
    );
    session.managed_protection.insert(
        StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6662.5),
            stop_price: Some(6659.0),
            last_requested_take_profit_price: Some(6662.5),
            last_requested_stop_price: Some(6659.0),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: None,
            stop_order_id: None,
        },
    );

    let snapshot = execution_state_snapshot(&session);

    assert_eq!(snapshot.market_entry_price, Some(6659.75));
    assert_eq!(snapshot.selected_contract_take_profit_price, Some(6662.5));
    assert_eq!(snapshot.selected_contract_stop_price, Some(6659.0));
}

#[test]
fn duplicate_symbol_position_record_does_not_trigger_false_drift_disarm() {
    let mut session = test_session();
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.order_qty = 1;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_summary = "Armed".to_string();
    session.user_store.positions.insert(
        42,
        BTreeMap::from([
            (
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 6659.75
                }),
            ),
            (
                2,
                json!({
                    "id": 2,
                    "accountId": 42,
                    "symbol": "ESH6",
                    "netPos": 1,
                    "avgPrice": 6659.75
                }),
            ),
        ]),
    );

    handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
        .expect("duplicate mirror records should not disarm");

    assert!(session.execution_runtime.armed);
    assert_eq!(selected_market_position_qty(&session), 1);
    assert_eq!(selected_market_entry_price(&session), Some(6659.75));
}

#[test]
fn replay_trailing_stop_sync_ratchets_forward_without_backsliding() {
    let mut session = test_session();
    session.replay_enabled = true;
    session.execution_runtime.armed = true;
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.stop_loss_ticks = 8.0;
    session.execution_config.native_ema.use_trailing_stop = true;
    session.execution_config.native_ema.trail_trigger_ticks = 4.0;
    session.execution_config.native_ema.trail_offset_ticks = 2.0;
    session.market.contract_id = Some(3570918);
    session.market.tick_size = Some(0.25);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "avgPrice": 100.0
            }),
        )]),
    );

    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: None,
            stop_price: Some(98.0),
            last_requested_take_profit_price: None,
            last_requested_stop_price: Some(98.0),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: Some("replay-stop".to_string()),
            take_profit_order_id: None,
            stop_order_id: Some(1002),
        },
    );

    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let first_bar = Bar {
        ts_ns: 1,
        open: 100.0,
        high: 101.5,
        low: 99.75,
        close: 101.0,
    };
    sync_execution_protection(&mut session, &broker_tx, Some(&first_bar))
        .expect("initial trailing sync should queue a stop update");

    let first_next_state = match broker_rx.try_recv().expect("expected stop modify command") {
        BrokerCommand::NativeProtection { sync, .. } => {
            assert!(
                sync.simulate,
                "replay mode should use simulated protection sync"
            );
            match sync.operation {
                ProtectionSyncOperation::ModifyStop { payload } => {
                    assert_eq!(payload.get("orderId").and_then(Value::as_i64), Some(1002));
                    assert_eq!(
                        payload.get("stopPrice").and_then(Value::as_f64),
                        Some(101.0)
                    );
                }
                _ => panic!("expected stop modification"),
            }
            sync.next_state
                .expect("planner should provide the updated protection snapshot")
        }
        _ => panic!("expected native protection command"),
    };
    assert_eq!(
        session
            .execution_runtime
            .ema_execution
            .position
            .as_ref()
            .and_then(|position| position.current_stop_price),
        Some(101.0)
    );
    session.protection_sync_in_flight = false;
    session.managed_protection.insert(key, first_next_state);

    let weaker_bar = Bar {
        ts_ns: 2,
        open: 101.0,
        high: 101.25,
        low: 100.0,
        close: 100.5,
    };
    sync_execution_protection(&mut session, &broker_tx, Some(&weaker_bar))
        .expect("weaker bar should be a no-op");
    assert!(
        broker_rx.try_recv().is_err(),
        "trailing stop should not backslide or resend an unchanged modify"
    );
    assert_eq!(
        session
            .execution_runtime
            .ema_execution
            .position
            .as_ref()
            .and_then(|position| position.current_stop_price),
        Some(101.0)
    );

    let stronger_bar = Bar {
        ts_ns: 3,
        open: 100.5,
        high: 102.0,
        low: 100.5,
        close: 101.75,
    };
    sync_execution_protection(&mut session, &broker_tx, Some(&stronger_bar))
        .expect("stronger bar should ratchet the trailing stop forward");

    let second_next_state = match broker_rx
        .try_recv()
        .expect("expected second stop modify command")
    {
        BrokerCommand::NativeProtection { sync, .. } => {
            assert!(sync.simulate);
            match sync.operation {
                ProtectionSyncOperation::ModifyStop { payload } => {
                    assert_eq!(payload.get("orderId").and_then(Value::as_i64), Some(1002));
                    assert_eq!(
                        payload.get("stopPrice").and_then(Value::as_f64),
                        Some(101.5)
                    );
                }
                _ => panic!("expected stop modification"),
            }
            sync.next_state
                .expect("planner should retain the replay protection snapshot")
        }
        _ => panic!("expected native protection command"),
    };
    assert_eq!(second_next_state.stop_order_id, Some(1002));
    assert_eq!(second_next_state.stop_price, Some(101.5));
    assert_eq!(
        session
            .execution_runtime
            .ema_execution
            .position
            .as_ref()
            .and_then(|position| position.current_stop_price),
        Some(101.5)
    );
}
