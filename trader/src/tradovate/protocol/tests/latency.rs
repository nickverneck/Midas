use super::*;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn tracker_matches_entity_binds_order_id_from_cl_ord_id() {
    let mut tracker = OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: None,
        signal_context: None,
        cl_ord_id: "midas-1-entry".to_string(),
        strategy_owned_protection: false,
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    };
    let entity = json!({
        "orderId": 42,
        "clOrdId": "midas-1-entry"
    });

    assert!(tracker_matches_entity(&mut tracker, &entity));
    assert_eq!(tracker.order_id, Some(42));
}

#[test]
fn update_latency_from_order_strategy_link_binds_order_id() {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut session = SessionState {
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
        accounts: Vec::new(),
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: Some(OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: None,
            signal_context: None,
            cl_ord_id: "midas-1-strategy".to_string(),
            strategy_owned_protection: true,
            order_id: None,
            order_strategy_id: Some(77),
            seen_recorded: false,
            exec_report_recorded: false,
            fill_recorded: false,
        }),
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store: UserSyncStore::default(),
        selected_account_id: None,
        selected_contract: None,
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    };
    let mut latency = LatencySnapshot::default();
    let link = EntityEnvelope {
        entity_type: "orderStrategyLink".to_string(),
        deleted: false,
        entity: json!({
            "id": 700,
            "orderStrategyId": 77,
            "orderId": 42,
            "label": "entry"
        }),
    };

    assert!(!update_latency_from_envelope(
        &mut session,
        &mut latency,
        &link
    ));
    assert_eq!(
        session
            .order_latency_tracker
            .as_ref()
            .and_then(|tracker| tracker.order_id),
        Some(42)
    );
}

#[test]
fn update_latency_from_envelope_records_seen_ack_and_fill() {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut session = SessionState {
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
        accounts: Vec::new(),
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: Some(OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
            cl_ord_id: "midas-1-entry".to_string(),
            strategy_owned_protection: false,
            order_id: Some(42),
            order_strategy_id: None,
            seen_recorded: false,
            exec_report_recorded: false,
            fill_recorded: false,
        }),
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store: UserSyncStore::default(),
        selected_account_id: None,
        selected_contract: None,
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    };
    let mut latency = LatencySnapshot::default();

    let order = EntityEnvelope {
        entity_type: "order".to_string(),
        deleted: false,
        entity: json!({ "orderId": 42 }),
    };
    let exec_report = EntityEnvelope {
        entity_type: "executionReport".to_string(),
        deleted: false,
        entity: json!({ "orderId": 42, "execType": "New" }),
    };
    let fill = EntityEnvelope {
        entity_type: "fill".to_string(),
        deleted: false,
        entity: json!({ "orderId": 42, "price": 5000.25, "qty": 1 }),
    };

    assert!(update_latency_from_envelope(
        &mut session,
        &mut latency,
        &order
    ));
    assert!(update_latency_from_envelope(
        &mut session,
        &mut latency,
        &exec_report
    ));
    assert!(update_latency_from_envelope(
        &mut session,
        &mut latency,
        &fill
    ));
    assert!(latency.last_order_seen_ms.is_some());
    assert!(latency.last_exec_report_ms.is_some());
    assert!(latency.last_fill_ms.is_some());
    assert!(latency.last_signal_seen_ms.is_some());
    assert!(latency.last_signal_ack_ms.is_some());
    assert!(latency.last_signal_fill_ms.is_some());
}

#[test]
fn trade_marker_from_fill_uses_order_action_for_side_and_contract() {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut user_store = UserSyncStore::default();
    user_store.orders.insert(
        42,
        BTreeMap::from([(
            77,
            json!({
                "id": 77,
                "accountId": 42,
                "action": "Sell",
                "contractId": 3570918,
                "symbol": "ESH6"
            }),
        )]),
    );
    let session = SessionState {
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
        accounts: Vec::new(),
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store,
        selected_account_id: Some(42),
        selected_contract: None,
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    };
    let fill = json!({
        "id": 501,
        "accountId": 42,
        "orderId": 77,
        "price": 5000.25,
        "qty": 1,
        "timestamp": "2026-03-15T13:45:00Z"
    });

    let marker = trade_marker_from_fill(&session, &fill).expect("fill marker should resolve");

    assert_eq!(marker.fill_id, Some(501));
    assert_eq!(marker.account_id, Some(42));
    assert_eq!(marker.contract_id, Some(3570918));
    assert_eq!(marker.contract_name.as_deref(), Some("ESH6"));
    assert_eq!(marker.side, TradeMarkerSide::Sell);
    assert_eq!(marker.price, 5000.25);
    assert_eq!(marker.qty, 1);
}

#[test]
fn record_trade_marker_deduplicates_fill_ids() {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut session = SessionState {
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
        accounts: Vec::new(),
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
        selected_contract: None,
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    };
    let marker = TradeMarker {
        fill_id: Some(501),
        account_id: Some(42),
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        ts_ns: 1,
        price: 5000.25,
        qty: 1,
        side: TradeMarkerSide::Buy,
    };

    assert!(record_trade_marker(&mut session, marker.clone()));
    assert!(!record_trade_marker(&mut session, marker));
    assert_eq!(session.market.trade_markers.len(), 1);
}
