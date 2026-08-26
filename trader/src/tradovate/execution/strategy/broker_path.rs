use super::*;

pub(crate) fn effective_market_position_qty(session: &SessionState) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

pub(crate) fn strategy_has_live_broker_path(
    session: &SessionState,
    key: StrategyProtectionKey,
    order_strategy_id: i64,
) -> bool {
    session
        .user_store
        .linked_strategy_orders(key.account_id, order_strategy_id)
        .into_iter()
        .any(|order| order_is_active(order) && order_contract_id(order) == Some(key.contract_id))
}

pub(crate) const ORDER_STRATEGY_HYDRATION_GRACE_MS: u128 = 1_500;
pub(crate) const MARKET_ORDER_POSITION_SYNC_GRACE_MS: u128 = 3_000;
pub(crate) const ORDER_STRATEGY_POSITION_SYNC_GRACE_MS: u128 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReplayProtectedExitSettlement {
    pub(crate) account_id: i64,
    pub(crate) contract_id: i64,
    pub(crate) order_strategy_id: i64,
    pub(crate) reason: String,
}

/// Releases the guarded execution lifecycle after a replay-simulated,
/// broker-owned protection order closes the position.
///
/// Replay advances virtual bars much faster than wall-clock time.  The live
/// broker path therefore cannot use its normal position-sync grace to infer
/// that a terminal TP/SL fill has completed: the entry's latency tracker can
/// otherwise block every subsequent signal for ten virtual seconds.  Replay
/// emits an explicit terminal fill marker, so settle only the matching
/// account/contract/order-strategy lifecycle and leave the live path untouched.
pub(crate) fn settle_replay_protected_exit(
    session: &mut SessionState,
    entities: &[EntityEnvelope],
) -> Option<ReplayProtectedExitSettlement> {
    if !session.replay_enabled || session.session_kind != SessionKind::Replay {
        return None;
    }

    let key = selected_strategy_key(session).ok()?;
    let (order_strategy_id, reason) = entities.iter().rev().find_map(|envelope| {
        if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
            return None;
        }

        let fill = &envelope.entity;
        if !fill
            .get("source")
            .and_then(Value::as_str)
            .is_some_and(|source| source.eq_ignore_ascii_case("replay"))
        {
            return None;
        }

        let reason = fill
            .get("replayExitReason")
            .and_then(Value::as_str)
            .filter(|reason| {
                matches!(
                    reason.to_ascii_lowercase().as_str(),
                    "take_profit" | "stop_loss" | "trailing_stop"
                )
            })?;
        let account_id = extract_account_id("fill", fill)?;
        let contract_id = json_i64(fill, "contractId")?;
        let order_strategy_id = json_i64(fill, "orderStrategyId")?;
        (account_id == key.account_id && contract_id == key.contract_id)
            .then(|| (order_strategy_id, reason.to_string()))
    })?;

    // A partial protection fill is not terminal and must continue through the
    // ordinary guarded reconciliation path.
    if selected_market_position_qty(session) != 0 {
        return None;
    }

    let tracked_strategy_id = session
        .order_latency_tracker
        .as_ref()
        .and_then(|tracker| tracker.order_strategy_id);
    let active_strategy_id = session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .map(|tracked| tracked.order_strategy_id);
    let visible_strategy_id = session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(extract_entity_id);

    // Never let a terminal fill for an older strategy release a newer
    // submission.  The replay simulator normally deletes the strategy and
    // linked children in the same entity batch, but this guard also covers
    // delayed or reordered entity delivery.
    if tracked_strategy_id.is_some_and(|id| id != order_strategy_id)
        || active_strategy_id.is_some_and(|id| id != order_strategy_id)
        || visible_strategy_id.is_some_and(|id| id != order_strategy_id)
    {
        return None;
    }
    if tracked_strategy_id != Some(order_strategy_id)
        && active_strategy_id != Some(order_strategy_id)
    {
        return None;
    }

    // A remaining live child means the protection fill was only partial (or a
    // new bracket was already created), so leave normal reconciliation in
    // charge of that lifecycle.
    if strategy_has_live_broker_path(session, key, order_strategy_id) {
        return None;
    }

    if tracked_strategy_id == Some(order_strategy_id) {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        session.pending_signal_context = None;
    }
    session.execution_runtime.pending_target_qty = None;
    session.managed_protection.remove(&key);
    if active_strategy_id == Some(order_strategy_id) {
        session.active_order_strategy = None;
    }

    Some(ReplayProtectedExitSettlement {
        account_id: key.account_id,
        contract_id: key.contract_id,
        order_strategy_id,
        reason,
    })
}

pub(crate) fn tracker_within_broker_path_grace(
    session: &SessionState,
    order_strategy_id: i64,
) -> bool {
    session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.order_strategy_id == Some(order_strategy_id)
                && tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_HYDRATION_GRACE_MS
        })
}

pub(crate) fn selected_contract_has_live_broker_path(session: &SessionState) -> bool {
    if session.order_submit_in_flight {
        return true;
    }

    let Some(account_id) = session.selected_account_id else {
        return false;
    };
    let Some(key) = selected_strategy_key(session).ok() else {
        return false;
    };

    if let Some(tracker) = session.order_latency_tracker.as_ref() {
        if let Some(order_id) = tracker.order_id.or_else(|| {
            session
                .user_store
                .order_id_by_client_id(account_id, &tracker.cl_ord_id)
        }) {
            if session
                .user_store
                .find_order(account_id, order_id)
                .is_some_and(order_is_active)
            {
                return true;
            }
        }

        if let Some(order_strategy_id) = tracker.order_strategy_id {
            if strategy_has_live_broker_path(session, key, order_strategy_id) {
                return true;
            }
            if tracker_within_broker_path_grace(session, order_strategy_id) {
                return true;
            }
        }
    }

    active_order_strategy_matches_selected(session).is_some_and(|tracked| {
        strategy_has_live_broker_path(session, key, tracked.order_strategy_id)
    })
}

pub(crate) fn pending_target_has_live_broker_path(session: &SessionState) -> bool {
    selected_contract_has_live_broker_path(session)
}

fn selected_contract_position_qty(session: &SessionState) -> Option<i32> {
    let account_id = session.selected_account_id?;
    let contract = session.selected_contract.as_ref()?;
    session
        .user_store
        .contract_position_qty(account_id, contract)
        .map(|qty| qty.round() as i32)
}

pub(super) fn flat_broker_path_should_wait(session: &SessionState) -> bool {
    if !selected_contract_has_live_broker_path(session) {
        return false;
    }
    if selected_contract_position_qty(session) != Some(0) {
        return true;
    }
    if session.order_submit_in_flight || session.protection_sync_in_flight {
        return true;
    }
    session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_POSITION_SYNC_GRACE_MS
        })
}

pub(super) fn emit_pending_target_gate_debug(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    source: &str,
    pending_target_qty: i32,
    actual_qty: i32,
    reached: bool,
    overshot: bool,
    has_live_broker_path: bool,
    waiting_for_position_sync: bool,
) {
    let effective_qty = effective_market_position_qty(session);
    emit_debug_log(event_tx, session, || {
        format!(
            "strategy pending target gate | {source} | pending target {pending_target_qty} | actual {actual_qty} | effective {effective_qty} | reached {reached} | overshot {overshot} | live broker path {has_live_broker_path} | waiting position sync {waiting_for_position_sync} | {}",
            execution_observability_context(session)
        )
    });
}

pub(crate) fn should_wait_for_automated_position_sync(
    session: &SessionState,
    pending_target_qty: i32,
    actual_qty: i32,
) -> bool {
    if pending_target_qty == 0 || actual_qty == pending_target_qty {
        return false;
    }

    let Some(tracker) = session.order_latency_tracker.as_ref() else {
        return false;
    };
    let has_submission_progress = tracker.order_strategy_id.is_some()
        || tracker.order_id.is_some()
        || tracker.seen_recorded
        || tracker.exec_report_recorded
        || tracker.fill_recorded;
    if !has_submission_progress {
        return false;
    }

    let grace_ms = if tracker.order_strategy_id.is_some() {
        ORDER_STRATEGY_POSITION_SYNC_GRACE_MS
    } else {
        MARKET_ORDER_POSITION_SYNC_GRACE_MS
    };

    tracker.started_at.elapsed().as_millis() <= grace_ms
}

pub(crate) fn clear_stale_pending_target(
    session: &mut SessionState,
    pending: i32,
    actual_qty: i32,
    event_tx: &UnboundedSender<ServiceEvent>,
) {
    // Capture the request context before clearing the tracker.  Otherwise
    // the diagnostic loses the clOrdId/order id that explains which stale
    // submission was recovered.
    let pending_context = execution_observability_context(session);
    session.execution_runtime.pending_target_qty = None;
    session.pending_signal_context = None;
    session.order_latency_tracker = None;
    session.execution_runtime.last_summary = format!(
        "Pending target {pending} cleared: broker has no active order path and position is still {actual_qty}; re-evaluating."
    );
    force_reevaluate_pending_window(session);
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Pending target {pending} cleared: broker has no active order path; re-evaluating."
    )));
    emit_debug_log(event_tx, session, || {
        format!(
            "pending target cleared | target {pending} | actual {actual_qty} | broker has no active order path | {}",
            pending_context
        )
    });
}

pub(super) fn force_reevaluate_pending_window(session: &mut SessionState) {
    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        return;
    }

    let Some(latest_strategy_ts) = latest_strategy_bar_ts(session) else {
        return;
    };
    if session
        .execution_runtime
        .last_closed_bar_ts
        .is_some_and(|last_seen| last_seen < latest_strategy_ts)
    {
        return;
    }
    session.execution_runtime.last_closed_bar_ts = Some(latest_strategy_ts.saturating_sub(1));
    session.execution_runtime.last_closed_bar_fingerprint = None;
}
