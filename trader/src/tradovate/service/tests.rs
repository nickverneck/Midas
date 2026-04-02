use super::*;
use crate::strategy::{NativeReversalMode, StrategyKind};
use serde_json::json;
use std::collections::BTreeMap;

fn test_session() -> SessionState {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    SessionState {
        cfg: AppConfig::default(),
        session_kind: SessionKind::Live,
        replay_enabled: false,
        tokens: TokenBundle {
            access_token: "access".to_string(),
            md_access_token: "md".to_string(),
            expiration_time: None,
            user_id: None,
            user_name: None,
        },
        accounts: vec![AccountInfo {
            id: 42,
            name: "SIM".to_string(),
            raw: json!({}),
        }],
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store: UserSyncStore::default(),
        selected_account_id: Some(42),
        selected_contract: Some(ContractSuggestion {
            id: 3570918,
            name: "ESM6".to_string(),
            description: "E-mini S&P".to_string(),
            raw: json!({}),
        }),
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    }
}

fn test_state(session: SessionState) -> ServiceState {
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    ServiceState {
        client: Client::builder().build().expect("client"),
        broker_tx,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    }
}

#[tokio::test]
async fn stale_market_order_interrupt_recovers_and_rearms_signal() {
    let stale_strategy_id = 453147950116_i64;
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 6400.0,
        high: 6401.0,
        low: 6399.0,
        close: 6400.5,
    }];
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: stale_strategy_id,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-stale-direct-reversal".to_string(),
        strategy_owned_protection: false,
        order_id: Some(77),
        order_strategy_id: Some(stale_strategy_id),
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.user_store.order_strategies.insert(
        stale_strategy_id,
        json!({
            "id": stale_strategy_id,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working"
        }),
    );

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_internal(
        InternalEvent::BrokerOrderFailed(BrokerOrderFailure {
            cl_ord_id: "midas-stale-direct-reversal".to_string(),
            message: format!(
                "strategy {stale_strategy_id} was already inactive; waiting for broker sync before retrying the reversal"
            ),
            target_qty: Some(-1),
            stale_interrupt: true,
        }),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("stale market interrupt should recover");

    let session = state.session.expect("session should persist");
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.active_order_strategy.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(199));
    assert_eq!(
        session.execution_runtime.last_summary,
        "Previous strategy was already inactive; retrying current signal after broker sync."
    );
    assert!(
        !session
            .user_store
            .order_strategies
            .contains_key(&stale_strategy_id)
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("submit stale")
                && message.contains("already inactive")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("already inactive")
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ServiceEvent::Error(_)))
    );
}

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
    }];
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - Duration::from_secs(3),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-strategy-position-sync".to_string(),
        strategy_owned_protection: true,
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
        strategy_owned_protection: true,
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
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
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
