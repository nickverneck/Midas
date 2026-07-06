use super::*;

pub(crate) fn handle_simple_execution_account_sync(
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
) {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);
    emit_execution_state(event_tx, session);
}

pub(crate) fn maybe_run_simple_execution_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Simple diagnostic {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        let latest_fingerprint = latest_strategy_bar_fingerprint(session);
        if session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts)
            && session.execution_runtime.last_closed_bar_fingerprint == latest_fingerprint
        {
            let _ = event_tx.send(ServiceEvent::DebugLog(format!(
                "simple strategy gate | {} | closed-bar timing waiting for next bar | last_bar_ts {}",
                active_native_slug(session),
                last_strategy_ts
            )));
            return Ok(());
        }
        session.execution_runtime.last_closed_bar_fingerprint = latest_fingerprint;
    }
    let previous_strategy_ts = session.execution_runtime.last_closed_bar_ts;
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);

    let (signal_bar, signal, summary) = {
        let bars = strategy_bars(session);
        if bars.is_empty() {
            bail!("latest strategy bar disappeared during simple strategy evaluation");
        }
        evaluate_active_execution_strategy_since(session, bars, actual_qty, previous_strategy_ts)
    };
    session.execution_runtime.last_summary = format!("Simple diagnostic: {summary}");

    let Some(target_qty) =
        target_qty_for_signal(signal, actual_qty, session.execution_config.order_qty)
    else {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "simple strategy eval | {} | signal {} | no target | bar_ts {} | actual_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            signal_bar.ts_ns,
            actual_qty,
            summary
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    let delta = target_qty.saturating_sub(actual_qty);
    if delta == 0 {
        let _ = event_tx.send(ServiceEvent::DebugLog(format!(
            "simple strategy eval | {} | signal {} | target already actual | target_qty {} | bar_ts {} | actual_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            target_qty,
            signal_bar.ts_ns,
            actual_qty,
            summary
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let account_id = session
        .selected_account_id
        .context("select an account before sending simple diagnostic orders")?;
    let account = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .cloned()
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .clone()
        .context("select a contract before sending simple diagnostic orders")?;

    let order_action = if delta > 0 { "Buy" } else { "Sell" };
    let order_qty = delta.unsigned_abs() as i32;
    let reason = format!(
        "simple diagnostic {} {} on {} | actual {} -> target {} | {}",
        active_native_slug(session),
        signal.label(),
        active_signal_timing_label(session),
        actual_qty,
        target_qty,
        summary
    );
    session.pending_signal_context = Some(PendingSignalLatencyContext {
        started_at: time::Instant::now(),
        description: format!(
            "simple {} {} (qty {} -> {})",
            active_native_slug(session),
            signal.label(),
            actual_qty,
            target_qty
        ),
    });
    let order = build_market_order_request(
        session,
        &account,
        &contract,
        order_action,
        order_qty,
        "Simple Strategy",
        true,
        Some(&reason),
        Some(target_qty),
        None,
        Vec::new(),
    );
    enqueue_market_order(session, broker_tx, order)?;
    session.execution_runtime.pending_target_qty = Some(target_qty);
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "simple strategy dispatch | {} | endpoint order/placeorder | signal {} | bar_ts {} | actual_qty {} | target_qty {} | order_action {} | order_qty {} | submit_in_flight ignored | pending_target overwritten | {}",
        active_native_slug(session),
        signal.label(),
        signal_bar.ts_ns,
        actual_qty,
        target_qty,
        order_action,
        order_qty,
        summary
    )));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Simple strategy {} signal: {} {} (qty {} -> {})",
        active_native_slug(session),
        order_action,
        order_qty,
        actual_qty,
        target_qty
    )));
    emit_execution_state(event_tx, session);
    Ok(())
}
