use super::*;

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
