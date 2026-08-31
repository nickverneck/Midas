use super::super::*;

#[test]
fn live_bar_auto_trail_signal_uses_broker_owned_order_strategy() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_config.native_ema.stop_loss_ticks = 8.0;
    session.execution_config.native_ema.use_trailing_stop = true;
    session.execution_config.native_ema.trail_trigger_ticks = 4.0;
    session.execution_config.native_ema.trail_offset_ticks = 2.0;
    session.market.tick_size = Some(0.25);
    session.market.history_loaded = 5;
    session.market.bars = vec![
        Bar {
            ts_ns: 1,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 2,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 3,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 4,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 5,
            open: 8.0,
            high: 8.5,
            low: 7.5,
            close: 8.0,
            volume: None,
        },
        Bar {
            ts_ns: 6,
            open: 12.0,
            high: 12.5,
            low: 11.5,
            close: 12.0,
            volume: None,
        },
    ];
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("live timing should submit broker-native protection");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    match broker_rx.try_recv().expect("broker command queued") {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.order_action, "Buy");
            let params: Value = serde_json::from_str(
                strategy
                    .payload
                    .get("params")
                    .and_then(Value::as_str)
                    .expect("params should be serialized JSON"),
            )
            .expect("params should parse");
            let bracket = params
                .get("brackets")
                .and_then(Value::as_array)
                .and_then(|brackets| brackets.first())
                .expect("strategy should include a bracket");
            assert_eq!(
                bracket
                    .get("autoTrail")
                    .and_then(|auto_trail| auto_trail.get("trigger"))
                    .and_then(Value::as_f64),
                Some(1.0)
            );
            assert_eq!(
                bracket
                    .get("autoTrail")
                    .and_then(|auto_trail| auto_trail.get("stopLoss"))
                    .and_then(Value::as_f64),
                Some(0.5)
            );
        }
        BrokerCommand::NativeProtection { .. } => {
            panic!("live auto-trail protection must remain broker-owned")
        }
        _ => panic!("expected broker order strategy command"),
    }
    assert!(
        broker_rx.try_recv().is_err(),
        "software protection should not be queued after broker-owned strategy entry"
    );
}

#[test]
fn strategy_loop_waits_for_position_sync_after_native_protection_activity() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
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
            volume: None,
        },
        Bar {
            ts_ns: 2,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 3,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 4,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
            volume: None,
        },
        Bar {
            ts_ns: 5,
            open: 8.0,
            high: 8.5,
            low: 7.5,
            close: 8.0,
            volume: None,
        },
        Bar {
            ts_ns: 6,
            open: 12.0,
            high: 12.5,
            low: 11.5,
            close: 12.0,
            volume: None,
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
