use super::debug::{
    debug_signal_latency_suffix, emit_debug_logs_from_latency_delta, format_debug_latency_ms,
};
use super::*;

pub(super) async fn handle_internal(
    internal: InternalEvent,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::UserEntities(entities) => {
            handle_user_entities(entities, state, event_tx, internal_tx)?
        }
        InternalEvent::SnapshotsBuilt {
            revision,
            snapshots,
        } => {
            if revision == state.snapshot_revision && state.session.is_some() {
                let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
            }
        }
        InternalEvent::RestLatencyMeasured(rest_rtt_ms) => {
            state.latency.rest_rtt_ms = Some(rest_rtt_ms);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::UserSocketStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::Market(update) => {
            handle_market_update(update, state, event_tx, market_tx, internal_tx)?
        }
        InternalEvent::BrokerOrderAck(ack) => {
            handle_broker_order_ack(ack, state, event_tx, internal_tx)
        }
        InternalEvent::BrokerOrderFailed(failure) => {
            handle_broker_order_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::OrderStrategyAck(ack) => {
            handle_order_strategy_ack(ack, state, event_tx, internal_tx)
        }
        InternalEvent::OrderStrategyFailed(failure) => {
            handle_order_strategy_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::ProtectionSyncApplied(ack) => {
            handle_protection_sync_applied(ack, state, event_tx, internal_tx)?
        }
        InternalEvent::ProtectionSyncFailed(failure) => {
            handle_protection_sync_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::PendingTargetWatchdog => handle_pending_target_watchdog(state, event_tx)?,
        InternalEvent::Error(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
    }
    Ok(())
}

fn handle_user_entities(
    entities: Vec<EntityEnvelope>,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let history_entities_changed = entities.iter().any(|envelope| {
        matches!(
            envelope.entity_type.to_ascii_lowercase().as_str(),
            "order" | "command" | "orderstrategy" | "orderstrategylink" | "fill" | "fillfee"
        )
    });
    let mut latency_changed = false;
    let mut trade_markers_changed = false;
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        for envelope in &entities {
            let previous_latency = state.latency;
            latency_changed |= update_latency_from_envelope(session, &mut state.latency, envelope);
            emit_debug_logs_from_latency_delta(event_tx, session, previous_latency, state.latency);
            session.user_store.apply(envelope.clone());
        }
        let broker_rejections = collect_new_broker_rejections(session);
        for envelope in &entities {
            if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                continue;
            }
            if let Some(marker) = trade_marker_from_fill(session, &envelope.entity) {
                let emit_fill_detail =
                    fill_matches_active_latency_tracker(session, &envelope.entity);
                let fill_detail =
                    emit_fill_detail.then(|| fill_debug_detail(session, &marker, &envelope.entity));
                if record_trade_marker(session, marker) {
                    trade_markers_changed = true;
                    if let Some(detail) = fill_detail {
                        let _ = event_tx
                            .send(ServiceEvent::DebugLog(format!("fill detail | {detail}")));
                    }
                }
            }
        }
        if trade_markers_changed {
            let _ = event_tx.send(ServiceEvent::TradeMarkersUpdated(
                session.market.trade_markers.clone(),
            ));
        }
        match session.execution_config.native_execution_path {
            NativeExecutionPath::SimpleDiagnostic | NativeExecutionPath::HmaDirect => {
                handle_simple_execution_account_sync(session, event_tx);
            }
            NativeExecutionPath::Guarded => {
                handle_execution_account_sync(session, &broker_tx, event_tx)?;
            }
        }
        for rejection in broker_rejections {
            apply_broker_rejection(session, event_tx, rejection);
        }
        if history_entities_changed {
            refresh_engine_history(session);
            emit_engine_history(event_tx, session);
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    if latency_changed {
        let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct BrokerRejectionNotice {
    pub(super) message: String,
    pub(super) debug: String,
    pub(super) affects_active_submission: bool,
    pub(super) affects_active_strategy: bool,
    pub(super) matches_selected_instrument: bool,
}

fn collect_new_broker_rejections(session: &mut SessionState) -> Vec<BrokerRejectionNotice> {
    let pending = session
        .user_store
        .command_reports
        .iter()
        .filter(|(report_id, report)| {
            !session
                .user_store
                .reported_command_rejections
                .contains(report_id)
                && command_report_is_rejection(report)
        })
        .map(|(report_id, report)| (*report_id, report.clone()))
        .collect::<Vec<_>>();

    pending
        .into_iter()
        .map(|(report_id, report)| {
            let notice = broker_rejection_notice(session, report_id, &report);
            session
                .user_store
                .reported_command_rejections
                .insert(report_id);
            notice
        })
        .collect()
}

fn command_report_is_rejection(report: &Value) -> bool {
    report
        .get("commandStatus")
        .and_then(Value::as_str)
        .is_some_and(|status| {
            matches!(
                status.trim().to_ascii_lowercase().as_str(),
                "riskrejected" | "executionrejected"
            )
        })
        || report
            .get("ordStatus")
            .and_then(Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("Rejected"))
}

pub(super) fn broker_rejection_notice(
    session: &SessionState,
    report_id: i64,
    report: &Value,
) -> BrokerRejectionNotice {
    let command_id = json_i64(report, "commandId");
    let command = command_id.and_then(|id| session.user_store.commands.get(&id));
    let order_id = json_i64(report, "orderId")
        .or_else(|| command.and_then(|command| json_i64(command, "orderId")))
        .or(command_id);
    let order = order_id.and_then(|id| session.user_store.find_order_by_id(id));
    let account_id = order.and_then(|order| extract_account_id("order", order));
    let contract_id = order.and_then(|order| json_i64(order, "contractId"));
    let action = order
        .and_then(|order| order.get("action"))
        .and_then(Value::as_str);
    let order_qty = order
        .and_then(|order| pick_number(order, &["orderQty", "qty", "quantity"]))
        .map(|qty| qty.abs().round() as i32)
        .filter(|qty| *qty > 0);
    let account_name = account_id.and_then(|account_id| {
        session
            .accounts
            .iter()
            .find(|account| account.id == account_id)
            .map(|account| account.name.as_str())
    });
    let contract_name = contract_id
        .and_then(|contract_id| {
            session
                .selected_contract
                .as_ref()
                .filter(|contract| contract.id == contract_id)
                .map(|contract| contract.name.as_str())
        })
        .or_else(|| {
            order.and_then(|order| {
                order
                    .get("symbol")
                    .or_else(|| order.get("contractName"))
                    .and_then(Value::as_str)
            })
        });

    let mut subject = "Broker rejected".to_string();
    if let Some(action) = action {
        subject.push(' ');
        subject.push_str(action);
    }
    if let Some(order_qty) = order_qty {
        subject.push(' ');
        subject.push_str(&order_qty.to_string());
    }
    if let Some(contract_name) = contract_name {
        subject.push(' ');
        subject.push_str(contract_name);
    } else if let Some(contract_id) = contract_id {
        subject.push_str(&format!(" contract {contract_id}"));
    } else if let Some(order_id) = order_id {
        subject.push_str(&format!(" order {order_id}"));
    }
    if let Some(account_name) = account_name {
        subject.push_str(" on ");
        subject.push_str(account_name);
    } else if let Some(account_id) = account_id {
        subject.push_str(&format!(" on account {account_id}"));
    }

    let command_status = report
        .get("commandStatus")
        .and_then(Value::as_str)
        .unwrap_or("Rejected");
    let reject_reason = report
        .get("rejectReason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.trim().is_empty());
    let broker_text = report
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty());
    let reason = reject_reason.unwrap_or(command_status);
    let mut message = format!("{subject}: {reason}");
    if let Some(broker_text) = broker_text.filter(|text| !text.eq_ignore_ascii_case(reason)) {
        message.push_str(" — ");
        message.push_str(broker_text);
    }

    let key = account_id
        .zip(contract_id)
        .map(|(account_id, contract_id)| StrategyProtectionKey {
            account_id,
            contract_id,
        });
    let strategy_match = key.is_some_and(|key| {
        session
            .active_order_strategy
            .as_ref()
            .is_some_and(|tracked| tracked.key == key)
    });
    let matches_selected_instrument = key.is_some_and(|key| {
        session.selected_account_id == Some(key.account_id)
            && session
                .selected_contract
                .as_ref()
                .is_some_and(|contract| contract.id == key.contract_id)
    });
    let tracked_order_match = session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker
                .order_id
                .is_some_and(|tracked_id| Some(tracked_id) == order_id)
                || command
                    .and_then(|command| command.get("clOrdId"))
                    .and_then(Value::as_str)
                    .is_some_and(|cl_ord_id| cl_ord_id == tracker.cl_ord_id)
                || order
                    .and_then(|order| order.get("clOrdId"))
                    .and_then(Value::as_str)
                    .is_some_and(|cl_ord_id| cl_ord_id == tracker.cl_ord_id)
        });
    let debug = format!(
        "broker rejection | command report {report_id} | command {} | order {} | account {} | contract {} | status {command_status} | reason {} | {}",
        command_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        order_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        account_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        contract_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        reject_reason.unwrap_or("none"),
        broker_text.unwrap_or("no broker text"),
    );

    BrokerRejectionNotice {
        message,
        debug,
        affects_active_submission: strategy_match || tracked_order_match,
        affects_active_strategy: strategy_match,
        matches_selected_instrument,
    }
}

pub(super) fn apply_broker_rejection(
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
    rejection: BrokerRejectionNotice,
) {
    if rejection.affects_active_submission {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        session.execution_runtime.pending_target_qty = None;
        if rejection.affects_active_strategy {
            clear_selected_order_strategy_state(session);
        }
        session.execution_runtime.last_summary = rejection.message.clone();
        emit_execution_state(event_tx, session);
    }
    let _ = event_tx.send(ServiceEvent::DebugLog(rejection.debug));
    if rejection.matches_selected_instrument || rejection.affects_active_submission {
        let _ = event_tx.send(ServiceEvent::BrokerRejection(rejection.message));
    }
}

fn handle_market_update(
    update: MarketUpdate,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    if state.session.is_none() {
        return Ok(());
    }

    let broker_tx = state.broker_tx.clone();
    let (display_snapshot, closed_bar_advanced, engine_history) = {
        let session = state.session.as_mut().expect("checked session above");
        let closed_bar_advanced = apply_market_update(&mut session.market, update);
        match session.execution_config.native_execution_path {
            NativeExecutionPath::Guarded => {
                maybe_run_execution_strategy(session, &broker_tx, event_tx)?;
            }
            NativeExecutionPath::SimpleDiagnostic => {
                maybe_run_simple_execution_strategy(session, &broker_tx, event_tx)?;
            }
            NativeExecutionPath::HmaDirect => {
                maybe_run_hma_direct_execution_strategy(session, &broker_tx, event_tx)?;
            }
        }
        refresh_engine_history_mark(session);
        (
            display_market_snapshot(&session.market),
            closed_bar_advanced,
            session.engine_run.as_ref().map(|run| run.history.clone()),
        )
    };
    if closed_bar_advanced {
        request_snapshot_refresh(state, &internal_tx);
    }
    let _ = market_tx.send(display_snapshot);
    if let Some(history) = engine_history {
        let _ = event_tx.send(ServiceEvent::EngineHistoryUpdated(history));
    }
    Ok(())
}

pub(super) fn handle_broker_order_ack(
    ack: BrokerOrderAck,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    let mut signal_submit_ms = None;
    let mut signal_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if let Some(run) = session.engine_run.as_mut()
            && ack.cl_ord_id.starts_with(&run.order_prefix)
            && let Some(order_id) = ack.order_id
        {
            run.owned_order_ids.insert(order_id);
        }
        if let Some(tracker) = session.order_latency_tracker.as_mut() {
            if tracker.cl_ord_id == ack.cl_ord_id {
                tracker.order_id = ack.order_id;
                signal_submit_ms = tracker
                    .signal_started_at
                    .map(|started_at| started_at.elapsed().as_millis() as u64);
                signal_context = tracker.signal_context.clone();
            }
        }
    }

    apply_submit_latency(&mut state.latency, ack.submit_rtt_ms, signal_submit_ms);
    let debug_message = format!(
        "submit {}{} | endpoint {} | {}",
        format_debug_latency_ms(ack.submit_rtt_ms),
        debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
        ack.endpoint,
        ack.message
    );
    let _ = event_tx.send(ServiceEvent::Status(ack.message));
    let _ = event_tx.send(ServiceEvent::DebugLog(debug_message));
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    schedule_pending_target_watchdog(internal_tx);
}

fn handle_broker_order_failed(
    failure: BrokerOrderFailure,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let mut stale_interrupt_recovered = false;
    let mut observability_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if session
            .order_latency_tracker
            .as_ref()
            .is_some_and(|tracker| tracker.cl_ord_id == failure.cl_ord_id)
        {
            session.order_latency_tracker = None;
        }
        if let Some(target_qty) = failure.target_qty {
            if session.execution_runtime.pending_target_qty == Some(target_qty) {
                session.execution_runtime.pending_target_qty = None;
                if failure.stale_interrupt {
                    clear_selected_order_strategy_state(session);
                    session.execution_runtime.last_summary =
                        "Previous strategy was already inactive; retrying current signal after broker sync."
                            .to_string();
                    if let Some(last_closed_ts) = latest_strategy_bar_ts(session) {
                        session.execution_runtime.last_closed_bar_ts =
                            Some(last_closed_ts.saturating_sub(1));
                    }
                    stale_interrupt_recovered = true;
                    emit_execution_state(event_tx, session);
                } else {
                    session.execution_runtime.last_summary = failure.message.clone();
                    emit_execution_state(event_tx, session);
                }
            }
        }
        observability_context = Some(execution_observability_context(session));
    }

    let debug_message =
        format_broker_order_failure_debug(&failure, observability_context.as_deref());
    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit stale | {debug_message}"
        )));
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit failed | {debug_message}"
        )));
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
}

fn format_broker_order_failure_debug(
    failure: &BrokerOrderFailure,
    observability_context: Option<&str>,
) -> String {
    let mut message = format!(
        "endpoint {} | clOrdId {} | target {:?} | {}",
        failure.endpoint, failure.cl_ord_id, failure.target_qty, failure.message
    );
    if let Some(context) = observability_context {
        message.push_str(" | ");
        message.push_str(context);
    }
    message
}

fn handle_order_strategy_ack(
    ack: BrokerOrderStrategyAck,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    let mut signal_submit_ms = None;
    let mut signal_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if let Some(tracker) = session.order_latency_tracker.as_mut() {
            if tracker.cl_ord_id == ack.uuid {
                tracker.order_strategy_id = ack.order_strategy_id;
                signal_submit_ms = tracker
                    .signal_started_at
                    .map(|started_at| started_at.elapsed().as_millis() as u64);
                signal_context = tracker.signal_context.clone();
            }
        }
        if let Some(order_strategy_id) = ack.order_strategy_id {
            session.active_order_strategy = Some(TrackedOrderStrategy {
                key: ack.key,
                order_strategy_id,
                target_qty: ack.target_qty,
            });
        }
    }

    apply_submit_latency(&mut state.latency, ack.submit_rtt_ms, signal_submit_ms);
    let debug_message = format!(
        "submit {}{} | endpoint {} | {}",
        format_debug_latency_ms(ack.submit_rtt_ms),
        debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
        ack.endpoint,
        ack.message
    );
    let _ = event_tx.send(ServiceEvent::Status(ack.message));
    let _ = event_tx.send(ServiceEvent::DebugLog(debug_message));
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    schedule_pending_target_watchdog(internal_tx);
}

fn handle_order_strategy_failed(
    failure: BrokerOrderStrategyFailure,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let mut stale_interrupt_recovered = false;
    let mut observability_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if session
            .order_latency_tracker
            .as_ref()
            .is_some_and(|tracker| tracker.cl_ord_id == failure.uuid)
        {
            session.order_latency_tracker = None;
        }
        if session.execution_runtime.pending_target_qty == Some(failure.target_qty) {
            session.execution_runtime.pending_target_qty = None;
        }
        if failure.stale_interrupt {
            clear_selected_order_strategy_state(session);
            session.execution_runtime.last_summary =
                "Previous strategy was already inactive; retrying current signal after broker sync."
                    .to_string();
            if let Some(last_closed_ts) = latest_strategy_bar_ts(session) {
                session.execution_runtime.last_closed_bar_ts =
                    Some(last_closed_ts.saturating_sub(1));
            }
            stale_interrupt_recovered = true;
            emit_execution_state(event_tx, session);
        } else if session.execution_runtime.pending_target_qty.is_none() {
            session.execution_runtime.last_summary = failure.message.clone();
            emit_execution_state(event_tx, session);
        }
        observability_context = Some(execution_observability_context(session));
    }

    let debug_message =
        format_order_strategy_failure_debug(&failure, observability_context.as_deref());
    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit stale | {debug_message}"
        )));
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit failed | {debug_message}"
        )));
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
}

fn format_order_strategy_failure_debug(
    failure: &BrokerOrderStrategyFailure,
    observability_context: Option<&str>,
) -> String {
    let mut message = format!(
        "endpoint {} | uuid {} | target {} | {}",
        failure.endpoint, failure.uuid, failure.target_qty, failure.message
    );
    if let Some(context) = observability_context {
        message.push_str(" | ");
        message.push_str(context);
    }
    message
}

fn handle_protection_sync_applied(
    ack: ProtectionSyncAck,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        session.protection_sync_in_flight = false;
        match ack.next_state {
            Some(next_state) => {
                session.managed_protection.insert(ack.key, next_state);
            }
            None => {
                session.managed_protection.remove(&ack.key);
            }
        }

        if let Some(desired) = session.pending_protection_sync.take() {
            sync_native_protection_target(session, &broker_tx, desired)?;
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    if let Some(message) = ack.message {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "protection sync applied | endpoint {}",
        ack.endpoint
    )));
    Ok(())
}

fn handle_protection_sync_failed(
    failure: ProtectionSyncFailure,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        session.protection_sync_in_flight = false;
        if let Some(desired) = session.pending_protection_sync.take() {
            sync_native_protection_target(session, &broker_tx, desired)?;
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "protection sync failed | endpoint {} | {}",
        failure.endpoint, failure.message
    )));
    let _ = event_tx.send(ServiceEvent::Error(failure.message));
    Ok(())
}

fn handle_pending_target_watchdog(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        return Ok(());
    };
    let Some(pending) = session.execution_runtime.pending_target_qty else {
        return Ok(());
    };
    if pending == 0 || selected_contract_has_live_broker_path(session) {
        return Ok(());
    }

    let actual_qty = selected_market_position_qty(session);
    if should_wait_for_automated_position_sync(session, pending, actual_qty) {
        return Ok(());
    }
    clear_stale_pending_target(session, pending, actual_qty, event_tx);
    emit_execution_state(event_tx, session);
    Ok(())
}

fn apply_submit_latency(
    latency: &mut LatencySnapshot,
    submit_rtt_ms: u64,
    signal_submit_ms: Option<u64>,
) {
    latency.last_order_ack_ms = Some(submit_rtt_ms);
    latency.last_order_seen_ms = None;
    latency.last_exec_report_ms = None;
    latency.last_fill_ms = None;
    latency.last_signal_submit_ms = signal_submit_ms;
    latency.last_signal_seen_ms = None;
    latency.last_signal_ack_ms = None;
    latency.last_signal_fill_ms = None;
}

fn schedule_pending_target_watchdog(internal_tx: UnboundedSender<InternalEvent>) {
    tokio::spawn(async move {
        time::sleep(Duration::from_secs(PENDING_TARGET_WATCHDOG_DELAY_SECS)).await;
        let _ = internal_tx.send(InternalEvent::PendingTargetWatchdog);
    });
}
