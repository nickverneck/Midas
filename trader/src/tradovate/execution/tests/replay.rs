use super::*;

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
