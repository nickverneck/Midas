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
    let observability = execution_observability_context(session);
    session.execution_runtime.pending_target_qty = None;
    session.pending_signal_context = None;
    session.order_latency_tracker = None;
    session.execution_runtime.last_summary = format!(
        "Pending target {pending} cleared: broker has no active order path and position is still {actual_qty}; re-evaluating."
    );
    session.execution_runtime.last_closed_bar_ts =
        latest_strategy_bar_ts(session).map(|last_strategy_ts| last_strategy_ts.saturating_sub(1));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Pending target {pending} cleared: broker has no active order path; re-evaluating."
    )));
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "pending target cleared | target {pending} | actual {actual_qty} | broker has no active order path | {observability}"
    )));
}

pub(crate) fn evaluate_active_execution_strategy(
    session: &SessionState,
    bars: &[Bar],
    current_qty: i32,
) -> (StrategySignal, String) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            let evaluation = session
                .execution_config
                .native_hma
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
        NativeStrategyKind::EmaCross => {
            let evaluation = session
                .execution_config
                .native_ema
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
    }
}

pub(crate) fn target_qty_for_signal(
    signal: StrategySignal,
    current_qty: i32,
    base_qty: i32,
) -> Option<i32> {
    let base_qty = base_qty.max(1);
    match signal {
        StrategySignal::Hold => None,
        StrategySignal::EnterLong => Some(base_qty),
        StrategySignal::EnterShort => Some(-base_qty),
        StrategySignal::ExitLongOnShortSignal => {
            if current_qty > 0 {
                Some(0)
            } else {
                None
            }
        }
    }
}

pub(crate) fn continue_staged_reversal(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
    actual_qty: i32,
) -> Result<bool> {
    let Some(staged) = session.execution_runtime.pending_reversal_entry.clone() else {
        return Ok(false);
    };

    if actual_qty.signum() == staged.target_qty.signum() && actual_qty != 0 {
        session.execution_runtime.pending_reversal_entry = None;
        let next_summary = format!("Staged reversal resolved at target {actual_qty}");
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
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Staged reversal flat; waiting for broker path to clear before entering {}.",
                staged.target_qty
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
                session.execution_runtime.last_summary = format!(
                    "Flat confirmed; submitting staged reversal entry to {}.",
                    staged.target_qty
                );
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

pub(crate) fn arm_execution_strategy(session: &mut SessionState) {
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
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
        && selected_contract_has_live_broker_path(session)
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
        // (e.g. orphaned orders) pushed position further.  Release pending
        // so the strategy can correct on the next bar.
        let overshot =
            (pending > 0 && actual_qty > pending) || (pending < 0 && actual_qty < pending);
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
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
        } else if pending != 0 && !pending_target_has_live_broker_path(session) {
            if should_wait_for_automated_position_sync(session, pending, actual_qty) {
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

pub(crate) fn maybe_run_execution_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let actual_market_qty = selected_market_position_qty(session);
    let actual_market_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_market_qty, actual_market_entry);
    if session.execution_runtime.pending_target_qty.is_none()
        && continue_staged_reversal(session, broker_tx, event_tx, actual_market_qty)?
    {
        return Ok(());
    }
    let max_automated_qty = session.execution_config.order_qty.max(1);
    if actual_market_qty.abs() > max_automated_qty {
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Waiting for broker sync: temporary position {actual_market_qty} exceeds max {max_automated_qty} while an order path is still active."
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
                "Automation disarmed: position drifted to {actual_market_qty}, above configured max {max_automated_qty}."
            );
            emit_execution_transition_debug(event_tx, session, &reason, "execution drift disarm");
            disarm_execution_strategy(session, reason);
        }
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if actual_market_qty == 0
        && session.execution_runtime.pending_target_qty.is_none()
        && selected_contract_has_live_broker_path(session)
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

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        if pending_target_qty != 0 && !pending_target_has_live_broker_path(session) {
            if should_wait_for_automated_position_sync(
                session,
                pending_target_qty,
                actual_market_qty,
            ) {
                let next_summary = format!(
                    "Waiting for position sync after automated order settle (actual {actual_market_qty}, pending target {pending_target_qty})."
                );
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution market position sync wait",
                );
                session.execution_runtime.last_summary = next_summary;
                emit_execution_state(event_tx, session);
                return Ok(());
            }

            clear_stale_pending_target(session, pending_target_qty, actual_market_qty, event_tx);
            emit_execution_state(event_tx, session);
            return Ok(());
        }
        let next_summary = format!(
            "Waiting for prior automated order to settle (actual {actual_market_qty}, pending target {pending_target_qty})."
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution pending target wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if selected_managed_protection_waiting_for_position_sync(session) {
        let next_summary = format!(
            "Waiting for broker sync after native protection activity (position still {actual_market_qty})."
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution native protection sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current {}; waiting for next update.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts)
    {
        return Ok(());
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let current_qty = effective_market_position_qty(session);
    let (signal_bar, signal, summary) = {
        let bars = strategy_bars(session);
        let signal_bar = bars
            .last()
            .cloned()
            .context("latest strategy bar disappeared during strategy evaluation")?;
        let (signal, summary) = evaluate_active_execution_strategy(session, bars, current_qty);
        (signal_bar, signal, summary)
    };

    if let Some(window) = session_window_at(session, signal_bar.ts_ns) {
        if window.hold_entries {
            if actual_market_qty != 0 {
                if !native_order_strategy_enabled(session) {
                    sync_native_protection(
                        session,
                        broker_tx,
                        0,
                        None,
                        None,
                        &format!("{} session auto-close", active_native_slug(session)),
                    )?;
                }
                let reason = if window.session_open {
                    format!(
                        "{} session auto-close {:.0}m before {} close",
                        active_native_slug(session),
                        window.minutes_to_close.unwrap_or_default(),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                } else {
                    format!(
                        "{} session hold until {} reopen",
                        active_native_slug(session),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                };
                match dispatch_target_position_order(session, broker_tx, 0, true, &reason)? {
                    MarketOrderDispatchOutcome::NoOp { message } => {
                        let _ = event_tx.send(ServiceEvent::Status(message));
                    }
                    MarketOrderDispatchOutcome::Queued { target_qty } => {
                        session.execution_runtime.pending_target_qty = target_qty;
                    }
                }
                session.execution_runtime.last_summary = if window.session_open {
                    format!(
                        "Session hold active; flattening {} {:.0}m before close.",
                        actual_market_qty,
                        window.minutes_to_close.unwrap_or_default()
                    )
                } else {
                    format!(
                        "Session closed; flattening {} and holding until reopen.",
                        actual_market_qty
                    )
                };
                emit_execution_state(event_tx, session);
                return Ok(());
            }

            sync_execution_protection(session, broker_tx, Some(&signal_bar))?;
            session.execution_runtime.last_summary = if window.session_open {
                format!(
                    "Session hold active; no new entries with {:.0}m to close.",
                    window.minutes_to_close.unwrap_or_default()
                )
            } else {
                "Session closed; holding flat until reopen.".to_string()
            };
            emit_execution_state(event_tx, session);
            return Ok(());
        }
    }

    session.execution_runtime.last_summary = summary.clone();

    let Some(target_qty) =
        target_qty_for_signal(signal, current_qty, session.execution_config.order_qty)
    else {
        sync_execution_protection(session, broker_tx, Some(&signal_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if target_qty == current_qty {
        sync_execution_protection(session, broker_tx, Some(&signal_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        current_qty,
        target_qty
    )));

    if current_qty != 0 && !native_order_strategy_enabled(session) {
        sync_native_protection(
            session,
            broker_tx,
            0,
            None,
            None,
            &format!(
                "{} target transition {} -> {}",
                active_native_slug(session),
                current_qty,
                target_qty
            ),
        )?;
    }

    let reason = format!(
        "{} {} | {}",
        active_native_slug(session),
        signal.label(),
        summary
    );
    let signal_context = PendingSignalLatencyContext {
        started_at: time::Instant::now(),
        description: format!(
            "{} {} (qty {} -> {})",
            active_native_slug(session),
            signal.label(),
            current_qty,
            target_qty
        ),
    };
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "signal | {} | {}",
        signal_context.description, summary
    )));
    session.pending_signal_context = Some(signal_context);
    let dispatch_outcome =
        match dispatch_target_position_order(session, broker_tx, target_qty, true, &reason) {
            Ok(outcome) => outcome,
            Err(err) => {
                session.pending_signal_context = None;
                return Err(err);
            }
        };
    match dispatch_outcome {
        MarketOrderDispatchOutcome::NoOp { message } => {
            session.pending_signal_context = None;
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
        }
    }
    emit_execution_state(event_tx, session);
    Ok(())
}
