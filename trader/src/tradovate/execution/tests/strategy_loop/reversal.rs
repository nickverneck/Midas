use super::super::*;
use super::support::*;

#[test]
fn staged_reversal_waits_for_broker_path_clear_before_entry() {
    let mut session = test_session();
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
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
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
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
