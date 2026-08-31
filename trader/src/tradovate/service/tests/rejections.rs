use super::super::*;
use super::support::{test_session, test_state};
use serde_json::json;
use std::collections::BTreeMap;

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
        volume: None,
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
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

    handle_internal(
        InternalEvent::BrokerOrderFailed(BrokerOrderFailure {
            endpoint: "orderStrategy/interruptorderstrategy",
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
                && message.contains("endpoint orderStrategy/interruptorderstrategy")
                && message.contains("clOrdId midas-stale-direct-reversal")
                && message.contains("target Some(-1)")
                && message.contains("already inactive")
                && message.contains("pending target none")
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
async fn order_strategy_submit_failure_debug_logs_request_and_target() {
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("hma_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-start-fail".to_string(),
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

    handle_internal(
        InternalEvent::OrderStrategyFailed(BrokerOrderStrategyFailure {
            endpoint: "orderStrategy/startorderstrategy",
            uuid: "midas-start-fail".to_string(),
            message: "broker rejected entry".to_string(),
            target_qty: -1,
            stale_interrupt: false,
        }),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("order strategy failure should be handled");

    let session = state.session.expect("session should persist");
    assert!(!session.order_submit_in_flight);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert_eq!(
        session.execution_runtime.last_summary,
        "broker rejected entry"
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("submit failed")
                && message.contains("endpoint orderStrategy/startorderstrategy")
                && message.contains("uuid midas-start-fail")
                && message.contains("target -1")
                && message.contains("broker rejected entry")
                && message.contains("pending target none")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Error(message)
            if message == "broker rejected entry"
    )));
}

#[tokio::test]
async fn asynchronous_command_rejection_reports_reason_and_clears_pending_strategy() {
    let strategy_id = 592_891_516_837_i64;
    let order_id = 592_891_516_838_i64;
    let report_id = 592_891_516_839_i64;
    let contract_id = 4_095_561_i64;
    let mut session = test_session();
    session.selected_contract = Some(ContractSuggestion {
        id: contract_id,
        name: "GCQ6".to_string(),
        description: "Gold August 2026".to_string(),
        raw: json!({}),
    });
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_summary = "Strategy submitted".to_string();
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key: StrategyProtectionKey {
            account_id: 42,
            contract_id,
        },
        order_strategy_id: strategy_id,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-gcq6-strategy".to_string(),
        order_id: None,
        order_strategy_id: Some(strategy_id),
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });

    let entities = vec![
        EntityEnvelope {
            entity_type: "orderStrategy".to_string(),
            deleted: false,
            entity: json!({
                "id": strategy_id,
                "accountId": 42,
                "contractId": contract_id,
                "status": "ExecutionFailed",
                "uuid": "midas-gcq6-strategy"
            }),
        },
        EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: json!({
                "id": order_id,
                "accountId": 42,
                "contractId": contract_id,
                "action": "Buy",
                "ordStatus": "Rejected"
            }),
        },
        EntityEnvelope {
            entity_type: "command".to_string(),
            deleted: false,
            entity: json!({
                "id": order_id,
                "orderId": order_id,
                "commandType": "New",
                "commandStatus": "RiskRejected"
            }),
        },
        EntityEnvelope {
            entity_type: "commandReport".to_string(),
            deleted: false,
            entity: json!({
                "id": report_id,
                "commandId": order_id,
                "commandStatus": "RiskRejected",
                "rejectReason": "LiquidationOnlyBeforeExpiration",
                "text": "Liquidation only, contract is about to be expired. Please contact the Trade Desk.",
                "ordStatus": "Rejected"
            }),
        },
    ];

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

    handle_internal(
        InternalEvent::UserEntities(entities.clone()),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx.clone(),
    )
    .await
    .expect("command rejection should be handled");
    handle_internal(
        InternalEvent::UserEntities(entities),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("duplicate command rejection should be ignored");

    let session = state.session.expect("session should persist");
    assert!(session.execution_runtime.armed);
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.order_latency_tracker.is_none());
    assert!(session.active_order_strategy.is_none());
    assert_eq!(
        session.execution_runtime.last_summary,
        "Broker rejected Buy GCQ6 on SIM: LiquidationOnlyBeforeExpiration — Liquidation only, contract is about to be expired. Please contact the Trade Desk."
    );
    assert!(
        session
            .user_store
            .reported_command_rejections
            .contains(&report_id)
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    let rejection_notices = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ServiceEvent::BrokerRejection(message)
                    if message.contains("Broker rejected Buy GCQ6 on SIM")
                        && message.contains("LiquidationOnlyBeforeExpiration")
                        && message.contains("contact the Trade Desk")
            )
        })
        .count();
    assert_eq!(rejection_notices, 1);
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("broker rejection")
                && message.contains("command report 592891516839")
                && message.contains("order 592891516838")
                && message.contains("account 42")
                && message.contains("contract 4095561")
                && message.contains("status RiskRejected")
    )));
}

#[test]
fn gc_rejection_does_not_affect_armed_es_strategy() {
    let es_contract_id = 3_570_918_i64;
    let gc_contract_id = 4_095_561_i64;
    let es_strategy_id = 592_891_516_900_i64;
    let es_order_id = 592_891_516_901_i64;
    let gc_order_id = 592_891_516_902_i64;
    let gc_command_id = 592_891_516_903_i64;
    let gc_report_id = 592_891_516_904_i64;
    let es_key = StrategyProtectionKey {
        account_id: 42,
        contract_id: es_contract_id,
    };
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_summary = "ES strategy active".to_string();
    session.order_submit_in_flight = true;
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key: es_key,
        order_strategy_id: es_strategy_id,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ES buy".to_string()),
        cl_ord_id: "midas-es-strategy".to_string(),
        order_id: Some(es_order_id),
        order_strategy_id: Some(es_strategy_id),
        seen_recorded: true,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.managed_protection.insert(
        es_key,
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6_400.0),
            stop_price: Some(6_350.0),
            take_profit_cl_ord_id: Some("midas-es-tp".to_string()),
            stop_cl_ord_id: Some("midas-es-sl".to_string()),
            take_profit_order_id: Some(es_order_id + 1),
            stop_order_id: Some(es_order_id + 2),
        },
    );
    session.user_store.orders.insert(
        42,
        BTreeMap::from([(
            gc_order_id,
            json!({
                "id": gc_order_id,
                "accountId": 42,
                "contractId": gc_contract_id,
                "symbol": "GCQ6",
                "action": "Buy",
                "ordStatus": "Rejected",
                "clOrdId": "unrelated-gc-order"
            }),
        )]),
    );
    session.user_store.commands.insert(
        gc_command_id,
        json!({
            "id": gc_command_id,
            "orderId": gc_order_id,
            "commandStatus": "RiskRejected"
        }),
    );
    let report = json!({
        "id": gc_report_id,
        "commandId": gc_command_id,
        "commandStatus": "RiskRejected",
        "rejectReason": "LiquidationOnlyBeforeExpiration",
        "text": "GC contract is liquidation only",
        "ordStatus": "Rejected"
    });
    let rejection =
        super::super::internal::broker_rejection_notice(&session, gc_report_id, &report);
    assert!(!rejection.affects_active_submission);
    assert!(!rejection.affects_active_strategy);
    assert!(!rejection.matches_selected_instrument);
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);

    super::super::internal::apply_broker_rejection(&mut session, &event_tx, rejection);

    assert!(session.execution_runtime.armed);
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(session.execution_runtime.last_summary, "ES strategy active");
    assert!(session.order_submit_in_flight);
    let tracker = session
        .order_latency_tracker
        .as_ref()
        .expect("ES tracker should remain");
    assert_eq!(tracker.cl_ord_id, "midas-es-strategy");
    let active = session
        .active_order_strategy
        .as_ref()
        .expect("ES strategy should remain tracked");
    assert_eq!(active.key, es_key);
    assert_eq!(active.order_strategy_id, es_strategy_id);
    assert!(session.managed_protection.contains_key(&es_key));

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("GCQ6") || message.contains("contract 4095561")
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ServiceEvent::BrokerRejection(_)))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ServiceEvent::ExecutionState(_)))
    );
}
