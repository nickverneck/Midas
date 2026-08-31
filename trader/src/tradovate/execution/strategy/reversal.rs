use super::*;

fn staged_reversal_timing(entry: &PendingNativeReversalEntry) -> String {
    let elapsed_ms = entry.started_at.elapsed().as_millis();
    let flat_wait = entry
        .flat_seen_at
        .map(|flat_seen_at| format!(" | flat_wait {}ms", flat_seen_at.elapsed().as_millis()))
        .unwrap_or_default();
    format!("elapsed {elapsed_ms}ms{flat_wait}")
}

pub(crate) fn continue_staged_reversal(
    session: &mut SessionState,
    broker_tx: &dyn BrokerCommandSink,
    event_tx: &ServiceEventSender,
    actual_qty: i32,
) -> Result<bool> {
    let Some(staged) = session.execution_runtime.pending_reversal_entry.clone() else {
        return Ok(false);
    };

    if actual_qty.signum() == staged.target_qty.signum() && actual_qty != 0 {
        session.execution_runtime.pending_reversal_entry = None;
        let next_summary = format!(
            "Staged reversal resolved at target {actual_qty} ({}).",
            staged_reversal_timing(&staged)
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution staged reversal resolved",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(true);
    }

    if actual_qty == 0 {
        if let Some(entry) = session.execution_runtime.pending_reversal_entry.as_mut() {
            if entry.flat_seen_at.is_none() {
                entry.flat_seen_at = Some(time::Instant::now());
            }
        }
        let staged = session
            .execution_runtime
            .pending_reversal_entry
            .clone()
            .unwrap_or(staged);
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Staged reversal flat; waiting for broker path to clear before entering {} ({}).",
                staged.target_qty,
                staged_reversal_timing(&staged)
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution staged reversal wait",
            );
            session.execution_runtime.last_summary = next_summary;
            emit_execution_state(event_tx, session);
            return Ok(true);
        }

        let entry_summary = format!(
            "Flat confirmed and broker path clear; submitting staged reversal entry to {} ({}).",
            staged.target_qty,
            staged_reversal_timing(&staged)
        );
        let dispatch_outcome = dispatch_target_position_order(
            session,
            broker_tx,
            staged.target_qty,
            true,
            &staged.reason,
        )?;
        match dispatch_outcome {
            MarketOrderDispatchOutcome::NoOp { message } => {
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = message.clone();
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
            MarketOrderDispatchOutcome::Queued { target_qty } => {
                session.execution_runtime.pending_target_qty = target_qty;
                session.execution_runtime.pending_reversal_entry = None;
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &entry_summary,
                    "execution staged reversal entry submit",
                );
                session.execution_runtime.last_summary = entry_summary;
            }
        }
        emit_execution_state(event_tx, session);
        return Ok(true);
    }

    let flatten_reason = format!(
        "{} | staged reversal flatten {} -> 0 before {}",
        staged.reason, actual_qty, staged.target_qty
    );
    let dispatch_outcome =
        dispatch_target_position_order(session, broker_tx, 0, true, &flatten_reason)?;
    match dispatch_outcome {
        MarketOrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message.clone();
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            session.execution_runtime.last_summary = format!(
                "Flattening {} before staged reversal to {}.",
                actual_qty, staged.target_qty
            );
        }
    }
    emit_execution_state(event_tx, session);
    Ok(true)
}
