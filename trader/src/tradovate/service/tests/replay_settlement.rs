use super::super::*;
use super::support::{test_session, test_state};
use serde_json::json;
use std::collections::BTreeMap;

#[tokio::test]
async fn replay_protected_exit_is_reported_and_releases_guarded_state() {
    let strategy_id = 77_i64;
    let mut session = test_session();
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_config.kind = StrategyKind::Native;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
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
        key: StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        order_strategy_id: strategy_id,
        target_qty: 1,
    });
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

    let entities = vec![
        EntityEnvelope {
            entity_type: "orderStrategy".to_string(),
            deleted: true,
            entity: json!({"id": strategy_id}),
        },
        EntityEnvelope {
            entity_type: "orderStrategyLink".to_string(),
            deleted: true,
            entity: json!({
                "id": 7002,
                "orderStrategyId": strategy_id,
                "orderId": 7001
            }),
        },
        EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({
                "id": 9001,
                "accountId": 42,
                "contractId": 3570918,
                "orderId": 7001,
                "orderStrategyId": strategy_id,
                "source": "replay",
                "price": 5005.0,
                "qty": 1,
                "buySell": "Sell",
                "timestamp": 1_000,
                "replayProtectionOrderId": 7001,
                "replayExitReason": "take_profit"
            }),
        },
        EntityEnvelope {
            entity_type: "position".to_string(),
            deleted: false,
            entity: json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 0
            }),
        },
    ];

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

    handle_internal(
        InternalEvent::UserEntities(entities),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("replay user entities should settle the guarded lifecycle");

    let session = state.session.expect("session should persist");
    assert!(session.order_latency_tracker.is_none());
    assert!(!session.order_submit_in_flight);
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.active_order_strategy.is_none());
    let mut reported_settlement = false;
    while let Ok(event) = event_rx.try_recv() {
        if matches!(
            event,
            ServiceEvent::Status(message)
                if message.contains("Replay protected exit settled")
                    && message.contains("take_profit")
        ) {
            reported_settlement = true;
        }
    }
    assert!(reported_settlement);
}
