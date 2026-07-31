use super::super::*;
use super::support::*;

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
    session.replay_enabled = true;
    session.cfg.replay_signal_diagnostics = true;
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
            volume: None,
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
                && message.starts_with("strategy decision |")
                && message.contains("strategy=ema_cross")
                && message.contains("broker=tradovate")
                && message.contains("path=simple diagnostic")
                && message.contains("decision=dispatching")
                && message.contains("signal=Buy")
    )));
    let diagnostic = session
        .execution_runtime
        .replay_signal_diagnostics
        .first()
        .expect("opt-in replay should capture the evaluated signal");
    assert_eq!(diagnostic.decision, "dispatching");
    assert_eq!(diagnostic.order_action.as_deref(), Some("Buy"));
    assert_eq!(diagnostic.order_qty, Some(2));
    assert_eq!(diagnostic.indicator_name, "EMA");
    assert_eq!(diagnostic.bar_timestamp_ns, 7);
}

#[test]
fn live_signal_diagnostics_remain_disabled_even_when_enabled() {
    let mut session = test_session();
    session.cfg.replay_signal_diagnostics = true;

    record_replay_signal_diagnostic(
        &mut session,
        1,
        StrategySignal::EnterLong,
        0,
        0,
        Some(1),
        "dispatching",
        "test",
        Some("Buy"),
        Some(1),
        "test",
    );

    assert!(
        session
            .execution_runtime
            .replay_signal_diagnostics
            .is_empty()
    );
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
        volume: None,
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
            volume: None,
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
fn closed_bar_timing_uses_t_minus_one_for_live_time_bars() {
    for bar_type in [BarType::minute(1), BarType::minute(5), BarType::second(15)] {
        let mut session = test_session();
        session.session_kind = SessionKind::Live;
        session.bar_type = bar_type;
        session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;

        let mature_ts = 1_000;
        let live_ts = 2_000;
        session.market.history_loaded = 2;
        session.market.live_bars = 1;
        session.market.status =
            format!("Subscribed to Standard {} bars for ESH6", bar_type.label());
        session.market.bars = vec![
            Bar {
                ts_ns: mature_ts,
                open: 10.0,
                high: 10.5,
                low: 9.5,
                close: 10.0,
                volume: None,
            },
            Bar {
                ts_ns: live_ts,
                open: 10.0,
                high: 12.5,
                low: 9.75,
                close: 12.0,
                volume: None,
            },
        ];

        assert_eq!(
            effective_closed_bar_len(&session),
            1,
            "bar_type={}",
            bar_type.label()
        );
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
}

#[test]
fn live_bar_timing_includes_latest_live_minute_bar() {
    let mut session = test_session();
    session.session_kind = SessionKind::Live;
    session.bar_type = BarType::minute(1);
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
            volume: None,
        },
        Bar {
            ts_ns: 2_000,
            open: 10.0,
            high: 12.5,
            low: 9.75,
            close: 12.0,
            volume: None,
        },
    ];

    assert_eq!(strategy_bars(&session).len(), 2);
    assert_eq!(latest_strategy_bar_ts(&session), Some(2_000));
}

#[test]
fn closed_bar_timing_uses_reported_t_minus_one_when_forming_bar_is_present() {
    let mut session = test_session();
    session.session_kind = SessionKind::Live;
    session.bar_type = BarType::minute(1);
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
            volume: None,
        },
        Bar {
            ts_ns: 2_000,
            open: 10.0,
            high: 11.5,
            low: 9.75,
            close: 11.0,
            volume: None,
        },
        Bar {
            ts_ns: 3_000,
            open: 11.0,
            high: 12.5,
            low: 10.75,
            close: 12.0,
            volume: None,
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
