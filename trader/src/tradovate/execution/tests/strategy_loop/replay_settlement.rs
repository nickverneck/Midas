use super::super::*;

fn replay_protected_exit_fill(
    account_id: i64,
    contract_id: i64,
    strategy_id: i64,
) -> EntityEnvelope {
    EntityEnvelope {
        entity_type: "fill".to_string(),
        deleted: false,
        entity: json!({
            "id": 9001,
            "accountId": account_id,
            "contractId": contract_id,
            "orderId": 7001,
            "orderStrategyId": strategy_id,
            "source": "replay",
            "price": 4995.0,
            "qty": 1,
            "buySell": "Sell",
            "timestamp": 1_000,
            "replayProtectionOrderId": 7001,
            "replayExitReason": "take_profit"
        }),
    }
}

fn seed_replay_exit_lifecycle(session: &mut SessionState, strategy_id: i64) {
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.pending_signal_context = Some(PendingSignalLatencyContext {
        started_at: time::Instant::now(),
        description: "replay entry".to_string(),
    });
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: None,
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-replay-entry".to_string(),
        order_id: Some(7000),
        order_strategy_id: Some(strategy_id),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: strategy_id,
        target_qty: 1,
    });
    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(5005.0),
            stop_price: Some(4990.0),
            take_profit_cl_ord_id: Some("midas-replay-tp".to_string()),
            stop_cl_ord_id: Some("midas-replay-sl".to_string()),
            take_profit_order_id: Some(7001),
            stop_order_id: Some(7002),
        },
    );
    session.user_store.positions.insert(
        key.account_id,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": key.account_id,
                "contractId": key.contract_id,
                "netPos": 0
            }),
        )]),
    );
}

#[test]
fn replay_protected_exit_releases_matching_guarded_lifecycle() {
    let mut session = test_session();
    seed_replay_exit_lifecycle(&mut session, 77);

    let settlement =
        settle_replay_protected_exit(&mut session, &[replay_protected_exit_fill(42, 3570918, 77)])
            .expect("matching terminal replay protection fill should settle");

    assert_eq!(settlement.account_id, 42);
    assert_eq!(settlement.contract_id, 3570918);
    assert_eq!(settlement.order_strategy_id, 77);
    assert_eq!(settlement.reason, "take_profit");
    assert!(session.order_latency_tracker.is_none());
    assert!(!session.order_submit_in_flight);
    assert!(session.pending_signal_context.is_none());
    assert!(session.execution_runtime.pending_target_qty.is_none());
    assert!(session.active_order_strategy.is_none());
    assert!(session.managed_protection.is_empty());
}

#[test]
fn replay_protected_exit_settlement_is_instrument_and_strategy_scoped() {
    let mut session = test_session();
    seed_replay_exit_lifecycle(&mut session, 77);

    assert!(
        settle_replay_protected_exit(
            &mut session,
            &[replay_protected_exit_fill(42, 4_095_561, 77)],
        )
        .is_none()
    );
    assert!(session.order_latency_tracker.is_some());
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));

    assert!(
        settle_replay_protected_exit(&mut session, &[replay_protected_exit_fill(42, 3570918, 78)],)
            .is_none()
    );
    assert!(session.order_latency_tracker.is_some());
    assert_eq!(
        session
            .active_order_strategy
            .as_ref()
            .map(|tracked| tracked.order_strategy_id),
        Some(77)
    );
}

#[test]
fn live_protected_exit_does_not_bypass_guarded_broker_grace() {
    let mut session = test_session();
    seed_replay_exit_lifecycle(&mut session, 77);
    session.session_kind = SessionKind::Live;

    assert!(
        settle_replay_protected_exit(&mut session, &[replay_protected_exit_fill(42, 3570918, 77)],)
            .is_none()
    );
    assert!(session.order_latency_tracker.is_some());
    assert!(session.order_submit_in_flight);
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.active_order_strategy.is_some());
}

#[test]
fn replay_protected_exit_allows_the_next_guarded_entry_without_wall_clock_grace() {
    let mut session = test_session();
    seed_replay_exit_lifecycle(&mut session, 77);
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
            volume: None,
        })
        .collect();
    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = Some(30);

    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    assert!(
        settle_replay_protected_exit(&mut session, &[replay_protected_exit_fill(42, 3570918, 77)],)
            .is_some()
    );

    maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
        .expect("guarded replay path should re-evaluate after terminal protection fill");

    match broker_rx.try_recv().expect("next entry should be queued") {
        BrokerCommand::OrderStrategy { strategy, .. } => {
            assert_eq!(strategy.target_qty, 1);
            assert_eq!(strategy.order_action, "Buy");
        }
        _ => panic!("expected guarded order-strategy entry"),
    }
}
