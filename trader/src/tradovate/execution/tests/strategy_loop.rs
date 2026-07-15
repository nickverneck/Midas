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

fn seed_short_position(session: &mut SessionState) {
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
}

fn seed_live_strategy_child(session: &mut SessionState, order_strategy_id: i64) {
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id,
        target_qty: 0,
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
            "orderStrategyId": order_strategy_id,
            "orderId": 1001
        }),
    );
}

#[test]
fn staged_reversal_waits_for_broker_path_clear_before_entry() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.market.tick_size = Some(0.25);
    session.execution_runtime.pending_reversal_entry = Some(PendingNativeReversalEntry {
        target_qty: 1,
        reason: "ema_cross staged reversal".to_string(),
        started_at: time::Instant::now(),
        flat_seen_at: None,
    });
    seed_live_strategy_child(&mut session, 77);

    assert!(
        continue_staged_reversal(&mut session, &broker_tx, &event_tx, 0)
            .expect("staged reversal should keep waiting")
    );

    assert!(
        broker_rx.try_recv().is_err(),
        "entry must not submit while broker path is still live"
    );
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        session
            .execution_runtime
            .pending_reversal_entry
            .as_ref()
            .is_some_and(|entry| entry.flat_seen_at.is_some())
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution staged reversal wait")
                && message.contains("flat_wait")
    )));
}

#[test]
fn staged_reversal_submits_broker_owned_entry_after_flat_path_clear() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    session.execution_config.native_ema.take_profit_ticks = 8.0;
    session.execution_config.native_ema.stop_loss_ticks = 6.0;
    session.execution_config.native_ema.use_trailing_stop = true;
    session.execution_config.native_ema.trail_trigger_ticks = 4.0;
    session.execution_config.native_ema.trail_offset_ticks = 2.0;
    session.market.tick_size = Some(0.25);
    session.execution_runtime.pending_reversal_entry = Some(PendingNativeReversalEntry {
        target_qty: 1,
        reason: "ema_cross staged reversal".to_string(),
        started_at: time::Instant::now(),
        flat_seen_at: Some(time::Instant::now()),
    });

    assert!(
        continue_staged_reversal(&mut session, &broker_tx, &event_tx, 0)
            .expect("staged reversal should submit entry")
    );

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.execution_runtime.pending_reversal_entry.is_none());
    match broker_rx
        .try_recv()
        .expect("entry strategy should be queued")
    {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.entry_order_qty, 1);
            assert_eq!(strategy.order_action, "Buy");
            let params: Value = serde_json::from_str(
                strategy
                    .payload
                    .get("params")
                    .and_then(Value::as_str)
                    .expect("strategy params should be serialized JSON"),
            )
            .expect("strategy params should parse");
            let bracket = params
                .get("brackets")
                .and_then(Value::as_array)
                .and_then(|brackets| brackets.first())
                .expect("strategy should include broker-owned bracket");
            assert!(bracket.get("profitTarget").is_some());
            assert!(bracket.get("stopLoss").is_some());
            assert!(bracket.get("autoTrail").is_some());
        }
        _ => panic!("expected broker-owned order strategy entry"),
    }
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("execution staged reversal entry submit")
                && message.contains("broker path clear")
                && message.contains("flat_wait")
    )));
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
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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

    assert_eq!(
        session.execution_runtime.pending_target_qty,
        Some(1),
        "{}",
        session.execution_runtime.last_summary
    );
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
fn simple_strategy_does_not_recheck_revised_closed_bar_with_same_timestamp() {
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
        .expect("same-timestamp bar revision should be ignored");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        broker_rx.try_recv().is_err(),
        "same-timestamp revised closed bar should not queue another order"
    );
}

#[test]
fn closed_bar_timing_uses_t_minus_one_for_live_minute_bar() {
    let mut session = test_session();
    session.session_kind = SessionKind::Live;
    session.bar_type = BarType::Minute1;
    session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;

    let mature_ts = 1_000;
    let live_ts = 2_000;
    session.market.history_loaded = 2;
    session.market.live_bars = 1;
    session.market.status = "Subscribed to Standard 1 Min bars for ESH6".to_string();
    session.market.bars = vec![
        Bar {
            ts_ns: mature_ts,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: live_ts,
            open: 10.0,
            high: 12.5,
            low: 9.75,
            close: 12.0,
        },
    ];

    assert_eq!(effective_closed_bar_len(&session), 1);
    assert_eq!(latest_strategy_bar_ts(&session), Some(mature_ts));
    assert_eq!(
        session
            .market
            .bars
            .get(effective_closed_bar_len(&session))
            .map(|bar| bar.ts_ns),
        Some(live_ts)
    );
}

#[test]
fn live_bar_timing_includes_latest_live_minute_bar() {
    let mut session = test_session();
    session.session_kind = SessionKind::Live;
    session.bar_type = BarType::Minute1;
    session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
    session.market.history_loaded = 2;
    session.market.live_bars = 1;
    session.market.status = "Subscribed to Standard 1 Min bars for ESH6".to_string();
    session.market.bars = vec![
        Bar {
            ts_ns: 1_000,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 2_000,
            open: 10.0,
            high: 12.5,
            low: 9.75,
            close: 12.0,
        },
    ];

    assert_eq!(strategy_bars(&session).len(), 2);
    assert_eq!(latest_strategy_bar_ts(&session), Some(2_000));
}

#[test]
fn closed_bar_timing_uses_reported_t_minus_one_when_forming_bar_is_present() {
    let mut session = test_session();
    session.session_kind = SessionKind::Live;
    session.bar_type = BarType::Minute1;
    session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;
    session.market.history_loaded = 2;
    session.market.live_bars = 1;
    session.market.status = "Subscribed to Standard 1 Min bars for ESH6".to_string();
    session.market.bars = vec![
        Bar {
            ts_ns: 1_000,
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.0,
        },
        Bar {
            ts_ns: 2_000,
            open: 10.0,
            high: 11.5,
            low: 9.75,
            close: 11.0,
        },
        Bar {
            ts_ns: 3_000,
            open: 11.0,
            high: 12.5,
            low: 10.75,
            close: 12.0,
        },
    ];

    assert_eq!(effective_closed_bar_len(&session), 2);
    assert_eq!(latest_strategy_bar_ts(&session), Some(2_000));
    assert_eq!(
        session
            .market
            .bars
            .get(effective_closed_bar_len(&session))
            .map(|bar| bar.ts_ns),
        Some(3_000)
    );
}

#[test]
fn guarded_closed_bar_delay_waits_extra_completed_bar_before_entry() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_config.native_signal_delay_bars = 1;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("delay should wait until the crossed bar becomes eligible");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(broker_rx.try_recv().is_err(), "delay should suppress entry");

    session.market.bars.push(Bar {
        ts_ns: 8,
        open: 13.0,
        high: 13.5,
        low: 12.5,
        close: 13.0,
    });
    session.market.history_loaded = 8;

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("delayed crossed bar should queue an entry");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    match broker_rx
        .try_recv()
        .expect("delayed closed-bar signal should queue a buy")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from delayed buy signal"),
    }
}

#[test]
fn guarded_closed_bar_signal_dispatches_once_even_if_position_returns_flat() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("first closed-bar signal should queue an entry");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    broker_rx
        .try_recv()
        .expect("first closed-bar signal should queue an order");
    let _ = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();

    session.execution_runtime.pending_target_qty = None;
    session.order_submit_in_flight = false;
    session.order_latency_tracker = None;
    session.execution_runtime.last_closed_bar_ts = Some(6);

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("same closed-bar signal should be blocked after a flat return");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        broker_rx.try_recv().is_err(),
        "same closed-bar signal should not queue a second order"
    );
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("already dispatched")
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | ema_cross | closed-bar already dispatched")
                && message.contains("signal Buy")
                && message.contains("bar_ts 7")
    )));
}

#[test]
fn guarded_closed_bar_blocks_repeat_flat_entry_side_until_opposite_dispatch() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
    session.execution_config.native_ema.fast_length = 2;
    session.execution_config.native_ema.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(5);
    session.execution_runtime.last_dispatched_entry_signal = Some(StrategySignal::EnterShort);
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("same-side flat signal should be consumed");

    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(
        broker_rx.try_recv().is_err(),
        "same-side flat signal should not queue another entry"
    );
    assert!(
        session
            .execution_runtime
            .last_summary
            .contains("Sell entry side already dispatched while flat")
    );

    session.execution_runtime.last_closed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("opposite flat signal should still be allowed");

    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(
        session.execution_runtime.last_dispatched_entry_signal,
        Some(StrategySignal::EnterLong)
    );
    match broker_rx
        .try_recv()
        .expect("opposite signal should queue a buy")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 1);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from opposite buy signal"),
    }
}

#[test]
fn hma_direct_path_executes_current_closed_bar_cross() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::HmaDirect;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(6);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(6);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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
    seed_short_position(&mut session);
    let precheck = session.execution_config.native_hma_cross.evaluate(
        &session.market.bars,
        Some(crate::strategies::PositionSide::Short),
    );
    assert!(
        precheck
            .fast_hma
            .zip(precheck.slow_hma)
            .is_some_and(|(fast, slow)| fast > slow),
        "{}",
        precheck.summary()
    );

    maybe_run_hma_direct_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("HMA direct should queue current-bar crossover");

    assert_eq!(
        session.execution_runtime.pending_target_qty,
        Some(1),
        "{}",
        session.execution_runtime.last_summary
    );
    match broker_rx
        .try_recv()
        .expect("HMA direct should queue a buy reversal")
    {
        BrokerCommand::MarketOrder { order, .. } => {
            assert_eq!(order.target_qty, Some(1));
            assert_eq!(order.order_qty, 2);
            assert_eq!(order.order_action, "Buy");
        }
        _ => panic!("expected market order from HMA direct buy signal"),
    }

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("hma direct dispatch")
                && message.contains("Prior HMA Side: fast<slow")
                && message.contains("HMA Side: fast>slow")
    )));
}

#[test]
fn hma_direct_path_does_not_enter_from_stale_cross_state() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::HmaDirect;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(7);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(5);
    session.market.history_loaded = 8;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0, 13.0]
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
    seed_short_position(&mut session);

    maybe_run_hma_direct_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("HMA direct should evaluate stale crossover state without ordering");

    assert_eq!(
        session.execution_runtime.pending_target_qty, None,
        "{}",
        session.execution_runtime.last_summary
    );
    assert!(
        broker_rx.try_recv().is_err(),
        "stale crossover state should not queue an order"
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("hma direct eval")
                && message.contains("signal Hold")
                && message.contains("Prior HMA Side: unset")
    )));
}

#[test]
fn guarded_hma_cross_blocks_same_bar_revised_side_change_after_dispatch() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_strategy = NativeStrategyKind::HmaCross;
    session.execution_config.native_execution_path = NativeExecutionPath::Guarded;
    session.execution_config.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    session.execution_config.native_hma_cross.fast_length = 2;
    session.execution_config.native_hma_cross.slow_length = 4;
    session.execution_config.native_hma_cross.take_profit_ticks = 5.0;
    session.execution_config.native_hma_cross.stop_loss_ticks = 10.0;
    session.market.tick_size = Some(0.25);
    session.market.contract_id = Some(3570918);
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(7);
    session.execution_runtime.last_closed_bar_fingerprint = Some(0);
    session.execution_runtime.last_dispatched_signal_bar_ts = Some(7);
    session.execution_runtime.last_dispatched_entry_signal = Some(StrategySignal::EnterShort);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_side = Some(crate::strategies::hma_cross::HmaCrossSide::Below);
    session
        .execution_runtime
        .hma_cross_execution
        .last_observed_bar_ts = Some(7);
    session.market.history_loaded = 7;
    session.market.bars = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 12.0]
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
    seed_short_position(&mut session);
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
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
        .expect("guarded HMA cross should block a revised side change on the same bar");

    assert_eq!(
        session.execution_runtime.pending_target_qty, None,
        "{}",
        session.execution_runtime.last_summary
    );
    assert!(session.execution_runtime.pending_reversal_entry.is_none());
    assert!(
        broker_rx.try_recv().is_err(),
        "same-bar revised side change should not queue a broker command"
    );
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy closed-bar revision | hma_cross")
                && message.contains("same timestamp fingerprint changed")
                && message.contains("previous_fingerprint Some(0)")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | hma_cross | closed-bar already dispatched")
                && message.contains("signal Buy")
                && message.contains("Prior HMA Side: fast<slow")
                && message.contains("HMA Side: fast>slow")
    )));
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
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
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
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("strategy eval | ema_cross | no target")
                && message.contains("signal Hold")
                && message.contains("target_qty none")
    )));
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
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 0,
            }),
        )]),
    );
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: None,
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-hydrating-flat-strategy".to_string(),
        order_id: Some(1001),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: false,
    });
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
fn strategy_loop_clears_stale_flat_broker_path_after_sync_grace() {
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
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 0,
            }),
        )]),
    );
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
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now()
            - Duration::from_millis((ORDER_STRATEGY_POSITION_SYNC_GRACE_MS + 1) as u64),
        signal_started_at: None,
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-stale-flat-strategy".to_string(),
        order_id: Some(1001),
        order_strategy_id: Some(77),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
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
            "uuid": "midas-stale-flat-strategy"
        }),
    );
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            1001,
            json!({
                "id": 1001,
                "accountId": 42,
                "contractId": 3570918,
                "orderStrategyId": 77,
                "ordStatus": "Working",
                "orderType": "Limit",
                "price": 6422.50,
                "action": "Sell",
                "clOrdId": "midas-stale-tp"
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
        .expect("stale flat broker path should be cleared by the next entry");

    match broker_rx
        .try_recv()
        .expect("entry strategy should be queued")
    {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.interrupt_order_strategy_id, Some(77));
            assert!(strategy.cancel_order_ids.is_empty());
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.order_action, "Buy");
        }
        _ => panic!("expected order strategy entry"),
    }
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
