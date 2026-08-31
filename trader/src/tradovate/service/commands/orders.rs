use super::*;

pub(super) fn manual_order(
    action: ManualOrderAction,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    #[cfg(not(feature = "manual-orders"))]
    {
        let _ = action;
        let _ = state;
        let _ = event_tx.send(ServiceEvent::Error(
            "Manual order commands are disabled; rebuild with --features manual-orders."
                .to_string(),
        ));
        return Ok(());
    }

    #[cfg(feature = "manual-orders")]
    {
        let broker_tx = state.broker_tx.clone();
        let Some(session) = state.session.as_mut() else {
            bail!("connect first");
        };
        match dispatch_manual_order(session, &broker_tx, action)? {
            MarketOrderDispatchOutcome::NoOp { message } => {
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
            MarketOrderDispatchOutcome::Queued { target_qty } => {
                if let Some(target_qty) = target_qty {
                    session.execution_runtime.pending_target_qty = Some(target_qty);
                    session.execution_runtime.last_summary =
                        "Manual close requested; waiting for flat position.".to_string();
                    emit_execution_state(event_tx, session);
                }
            }
        }
        Ok(())
    }
}

pub(super) fn set_target_position(
    target_qty: i32,
    automated: bool,
    reason: String,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    match dispatch_target_position_order(session, &broker_tx, target_qty, automated, &reason)? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

pub(super) fn profile_legacy_order_strategy_target(
    target_qty: i32,
    reason: String,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    match dispatch_profile_legacy_order_strategy_target(session, &broker_tx, target_qty, &reason)? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

pub(super) fn sync_native_protection_command(
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: String,
    state: &mut ServiceState,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    sync_native_protection(
        session,
        &broker_tx,
        signed_qty,
        take_profit_price,
        stop_price,
        &reason,
    )?;
    request_snapshot_refresh(state, &internal_tx);
    Ok(())
}
