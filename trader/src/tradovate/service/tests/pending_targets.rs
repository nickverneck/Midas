use super::super::*;
use super::support::{test_session, test_state};
use serde_json::json;
use std::collections::BTreeMap;

use crate::strategy::{NativeReversalMode, StrategyKind};

#[tokio::test]
async fn pending_target_watchdog_respects_order_strategy_position_sync_grace() {
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 6400.0,
        high: 6401.0,
        low: 6399.0,
        close: 6400.5,
        volume: None,
    }];
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - Duration::from_secs(3),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-strategy-position-sync".to_string(),
        order_id: Some(77),
        order_strategy_id: Some(88),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_internal(
        InternalEvent::PendingTargetWatchdog,
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("watchdog should preserve order-strategy pending target during sync grace");

    let session = state.session.expect("session should persist");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.order_latency_tracker.is_some());
    assert!(session.execution_runtime.last_summary.is_empty());

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(!events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message) if message.contains("pending target cleared")
    )));
}

#[tokio::test]
async fn set_target_position_records_pending_target_for_staged_reversal() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_hma.take_profit_ticks = 30.0;
    session.execution_config.native_hma.stop_loss_ticks = 30.0;
    session.execution_config.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    session.execution_runtime.armed = true;
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "netPrice": 6400.0
            }),
        )]),
    );
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
        signal_started_at: None,
        signal_context: None,
        cl_ord_id: "midas-live-strategy".to_string(),
        order_id: None,
        order_strategy_id: Some(77),
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });

    let (broker_tx, mut _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    let mut state = ServiceState {
        client: Client::builder().build().expect("client"),
        broker_tx,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        replay_lookup_job: None,
        replay_download_job: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    };
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_command(
        ServiceCommand::SetTargetPosition {
            target_qty: -1,
            automated: true,
            reason: "test staged reversal".to_string(),
        },
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("staged reversal target should queue");

    let session = state.session.expect("session should persist");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(0));
    assert!(session.execution_runtime.pending_reversal_entry.is_some());
    assert!(session.order_submit_in_flight);
}
