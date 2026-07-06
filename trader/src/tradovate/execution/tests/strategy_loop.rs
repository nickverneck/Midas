use super::*;

fn et_ts_ns(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
    chrono_tz::America::New_York
        .with_ymd_and_hms(year, month, day, hour, minute, 0)
        .single()
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap()
}

fn seed_long_position(session: &mut SessionState) {
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "avgPrice": 5000.0
            }),
        )]),
    );
}

#[test]
fn simple_strategy_path_queues_market_order_without_pending_or_inflight_gates() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::SimpleDiagnostic;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(5);
    session.order_submit_in_flight = true;
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 8.0, 12.0, 12.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
        })
        .collect();
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -1,
                "netPrice": 5000.0
            }),
        )]),
    );

    maybe_run_simple_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("simple path should bypass pending and in-flight gates");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    let command = broker_rx
        .try_recv()
        .expect("simple path should queue a market order");
    match command {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Buy");
            assert!(order.interrupt_order_strategy_id.is_none());
            assert!(order.cancel_order_ids.is_empty());
        }
        _ => panic!("simple path should only queue a market order"),
    }
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("simple strategy dispatch")
                && message.contains("submit_in_flight ignored")
    )));
}

#[test]
fn simple_strategy_rechecks_revised_closed_bar_with_same_timestamp() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::SimpleDiagnostic;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    let old_last_bar = Bar {
        ts_ns: 6,
        open: 12.0,
        high: 12.5,
        low: 11.5,
        close: 12.0,
    };
    session.execution_runtime.last_closed_bar_fingerprint = Some(bar_fingerprint(&old_last_bar));
    session.market.history_loaded = 6;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 12.0, 8.0]
        .into_iter()
        .enumerate()
        .map(|(idx, close)| Bar {
            ts_ns: idx as i64 + 1,
            open: close,
            high: close + 0.5,
            low: close - 0.5,
            close,
        })
        .collect();
    seed_long_position(&mut session);

    maybe_run_simple_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("same-timestamp bar revision should be evaluated");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(-1));
    match broker_rx
        .try_recv()
        .expect("revised closed bar should queue a sell")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(-1));
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Sell");
        }
        _ => panic!("expected market order from revised-bar sell signal"),
    }
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
fn strategy_blockout_flattens_inside_configured_preclose_window() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let bar_ts = et_ts_ns(2026, 3, 9, 16, 20);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.blockout_enabled = true;
    session.execution_config.blockout_minutes_before_close = 45.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(bar_ts - 60_000_000_000);
    session.market.session_profile = Some(InstrumentSessionProfile::FuturesGlobex);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: bar_ts,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.0,
    }];
    seed_long_position(&mut session);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("blockout should flatten open position");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(0));
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Session hold active")
    );
    match broker_rx.try_recv().expect("flatten command queued") {
        BrokerCommand::LiquidatePosition { liquidation, .. } => {
            assert_eq!(liquidation.target_qty, Some(0));
            assert_eq!(liquidation.contract_name, "ESH6");
        }
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(0));
            assert_eq!(order.contract_name, "ESH6");
            assert_eq!(order.order_action, "Sell");
            assert_eq!(order.order_qty, 1);
        }
        _ => panic!("expected liquidation command"),
    }
}

#[test]
fn disabled_strategy_blockout_does_not_flatten_preclose_position() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let bar_ts = et_ts_ns(2026, 3, 9, 16, 20);
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.blockout_enabled = false;
    session.execution_config.blockout_minutes_before_close = 45.0;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(bar_ts - 60_000_000_000);
    session.market.session_profile = Some(InstrumentSessionProfile::FuturesGlobex);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: bar_ts,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.0,
    }];
    seed_long_position(&mut session);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("disabled blockout should allow strategy evaluation");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(broker_rx.try_recv().is_err(), "no flatten should be queued");
    assert!(
        !session
            .execution_runtime
            .last_summary
            .contains("Session hold active")
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
