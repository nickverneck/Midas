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
        for envelope in &entities {
            if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                continue;
            }
            if let Some(marker) = trade_marker_from_fill(session, &envelope.entity) {
                trade_markers_changed |= record_trade_marker(session, marker);
            }
        }
        if trade_markers_changed {
            let _ = event_tx.send(ServiceEvent::TradeMarkersUpdated(
                session.market.trade_markers.clone(),
            ));
        }
        handle_execution_account_sync(session, &broker_tx, event_tx)?;
    }
    request_snapshot_refresh(state, &internal_tx);
    if latency_changed {
        let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    }
    Ok(())
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
    let (display_snapshot, closed_bar_advanced) = {
        let session = state.session.as_mut().expect("checked session above");
        let closed_bar_advanced = apply_market_update(&mut session.market, update);
        maybe_run_execution_strategy(session, &broker_tx, event_tx)?;
        (
            display_market_snapshot(&session.market),
            closed_bar_advanced,
        )
    };
    if closed_bar_advanced {
        request_snapshot_refresh(state, &internal_tx);
    }
    let _ = market_tx.send(display_snapshot);
    Ok(())
}

fn handle_broker_order_ack(
    ack: BrokerOrderAck,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    let mut signal_submit_ms = None;
    let mut signal_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
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
        "submit {}{} | {}",
        format_debug_latency_ms(ack.submit_rtt_ms),
        debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
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
    }

    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit stale | {}",
            failure.message
        )));
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit failed | {}",
            failure.message
        )));
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
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
        "submit {}{} | {}",
        format_debug_latency_ms(ack.submit_rtt_ms),
        debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
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
    }

    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit stale | {}",
            failure.message
        )));
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "submit failed | {}",
            failure.message
        )));
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
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
