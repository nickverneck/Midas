use super::*;

pub(crate) fn arm_execution_strategy(session: &mut SessionState) {
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_closed_bar_fingerprint = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
    session.execution_runtime.last_closed_bar_fingerprint =
        latest_strategy_bar_fingerprint(session);
    seed_hma_cross_observed_side(session);
    session.execution_runtime.last_summary =
        if session.execution_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        } else {
            format!(
                "Native {} armed; waiting for first {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        };
}

pub(crate) fn disarm_execution_strategy(session: &mut SessionState, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.last_closed_bar_fingerprint = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

pub(crate) fn handle_execution_account_sync(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);
    reconcile_selected_active_order_strategy(session);
    hydrate_selected_order_strategy_protection(session);

    let mut runtime_changed = false;
    let max_automated_qty = session.execution_config.order_qty.max(1);
    if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && actual_qty.abs() > max_automated_qty
    {
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Waiting for broker sync: temporary position {actual_qty} exceeds max {max_automated_qty} while an order path is still active."
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution broker sync wait",
            );
            session.execution_runtime.last_summary = next_summary;
        } else {
            let reason = format!(
                "Automation disarmed: position drifted to {actual_qty}, above configured max {max_automated_qty}."
            );
            emit_execution_transition_debug(event_tx, session, &reason, "execution drift disarm");
            disarm_execution_strategy(session, reason);
        }
        runtime_changed = true;
    }
    if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && actual_qty == 0
        && session.execution_runtime.pending_target_qty.is_none()
        && flat_broker_path_should_wait(session)
    {
        let next_summary =
            "Waiting for broker sync: flat position reported while an order path is still active."
                .to_string();
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution flat broker sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }
    if let Some(pending) = session.execution_runtime.pending_target_qty {
        let reached = pending == actual_qty;
        // Position overshot past target — entry filled, then something else
        // (e.g. orphaned orders) pushed position further. Release pending so
        // the strategy can correct on the next bar.
        let overshot =
            (pending > 0 && actual_qty > pending) || (pending < 0 && actual_qty < pending);
        let has_live_broker_path = pending_target_has_live_broker_path(session);
        let waiting_for_position_sync = !has_live_broker_path
            && should_wait_for_automated_position_sync(session, pending, actual_qty);
        emit_pending_target_gate_debug(
            event_tx,
            session,
            "account sync",
            pending,
            actual_qty,
            reached,
            overshot,
            has_live_broker_path,
            waiting_for_position_sync,
        );
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
            force_reevaluate_pending_window(session);
            if continue_staged_reversal(session, broker_tx, event_tx, actual_qty)? {
                runtime_changed = true;
            } else {
                let next_summary = if reached {
                    format!("Position confirmed at target {actual_qty}")
                } else {
                    format!(
                        "Position at {actual_qty} (target was {pending}); re-evaluating on next bar"
                    )
                };
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution pending target settled",
                );
                session.execution_runtime.last_summary = next_summary;
                runtime_changed = true;
            }
        } else if pending != 0 && !has_live_broker_path {
            if waiting_for_position_sync {
                let next_summary = format!(
                    "Waiting for position sync after automated order settle (actual {actual_qty}, pending target {pending})."
                );
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution market position sync wait",
                );
                session.execution_runtime.last_summary = next_summary;
                runtime_changed = true;
            } else {
                clear_stale_pending_target(session, pending, actual_qty, event_tx);
                runtime_changed = true;
            }
        }
    }

    if session.execution_runtime.pending_target_qty.is_none()
        && continue_staged_reversal(session, broker_tx, event_tx, actual_qty)?
    {
        runtime_changed = true;
    }

    let waiting_for_native_protection_sync = session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && session.execution_runtime.pending_target_qty.is_none()
        && selected_managed_protection_waiting_for_position_sync(session);
    if waiting_for_native_protection_sync {
        let next_summary = format!(
            "Waiting for broker sync after native protection activity (position still {actual_qty})."
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution native protection sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        runtime_changed = true;
    } else if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
    {
        sync_execution_protection(session, broker_tx, None)?;
    }

    if runtime_changed {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}
