use super::*;

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

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        let reached = pending_target_qty == actual_market_qty;
        let overshot = (pending_target_qty > 0 && actual_market_qty > pending_target_qty)
            || (pending_target_qty < 0 && actual_market_qty < pending_target_qty);
        let has_live_broker_path = pending_target_has_live_broker_path(session);
        let waiting_for_position_sync = !has_live_broker_path
            && should_wait_for_automated_position_sync(
                session,
                pending_target_qty,
                actual_market_qty,
            );
        emit_pending_target_gate_debug(
            event_tx,
            session,
            "strategy loop",
            pending_target_qty,
            actual_market_qty,
            reached,
            overshot,
            has_live_broker_path,
            waiting_for_position_sync,
        );
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
            force_reevaluate_pending_window(session);
            let next_summary = if reached {
                format!("Position confirmed at target {actual_market_qty}; re-evaluating.")
            } else {
                format!(
                    "Position at {actual_market_qty} (target was {pending_target_qty}); re-evaluating."
                )
            };
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution pending target settled",
            );
            session.execution_runtime.last_summary = next_summary;
            emit_execution_state(event_tx, session);
        } else if pending_target_qty != 0 && !has_live_broker_path {
            if waiting_for_position_sync {
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
        } else {
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
        session.execution_runtime.last_closed_bar_fingerprint =
            latest_strategy_bar_fingerprint(session);
        seed_hma_cross_observed_side(session);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current {}; waiting for next update.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        let latest_fingerprint = latest_strategy_bar_fingerprint(session);
        if session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts) {
            // A correction is actionable only when both sides have a known
            // fingerprint and the value changed.  Older sessions/tests may
            // have an anchored timestamp without the new fingerprint; treat
            // that as the normal same-bar gate so signal-delay timing remains
            // unchanged instead of evaluating the bar a second time.
            let corrected = matches!(
                (
                    session.execution_runtime.last_closed_bar_fingerprint,
                    latest_fingerprint,
                ),
                (Some(previous), Some(latest)) if previous != latest
            );
            if !corrected {
                let gate_detail = format!(
                    "strategy gate | {} | closed-bar fingerprint unchanged | last_bar_ts {} | fingerprint {:?} | actual_qty {} | {}",
                    active_native_slug(session),
                    last_strategy_ts,
                    latest_fingerprint,
                    actual_market_qty,
                    guarded_strategy_eval_context(session, actual_market_qty)
                );
                let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
                    session,
                    TradovateStrategyDecisionDebug {
                        path: "guarded",
                        decision: "blocked",
                        signal: None,
                        bar_ts: Some(last_strategy_ts),
                        actual_qty: actual_market_qty,
                        effective_qty: effective_market_position_qty(session),
                        target_qty: None,
                        strategy_detail: "n/a",
                        gate_detail,
                        fingerprint: latest_fingerprint,
                    },
                )));
                return Ok(());
            }

            let gate_detail = format!(
                "strategy closed-bar revision | {} | same timestamp fingerprint changed | bar_ts {} | previous_fingerprint {:?} | latest_fingerprint {:?} | actual_qty {} | {}",
                active_native_slug(session),
                last_strategy_ts,
                session.execution_runtime.last_closed_bar_fingerprint,
                latest_fingerprint,
                actual_market_qty,
                guarded_strategy_eval_context(session, actual_market_qty)
            );
            let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
                session,
                TradovateStrategyDecisionDebug {
                    path: "guarded",
                    decision: "closed-bar revision",
                    signal: None,
                    bar_ts: Some(last_strategy_ts),
                    actual_qty: actual_market_qty,
                    effective_qty: effective_market_position_qty(session),
                    target_qty: None,
                    strategy_detail: "n/a",
                    gate_detail,
                    fingerprint: latest_fingerprint,
                },
            )));
        }
        session.execution_runtime.last_closed_bar_fingerprint = latest_fingerprint;
    }
    let previous_strategy_ts = session.execution_runtime.last_closed_bar_ts;
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let current_qty = effective_market_position_qty(session);
    let (signal_bar, signal, summary, debug_summary) = if session.replay_enabled
        && session.cfg.replay_evaluator_mode == crate::broker::ReplayEvaluatorMode::Streaming
        && session.execution_config.native_strategy == NativeStrategyKind::EmaCross
    {
        // The streaming EMA evaluator only needs the immutable bar slice and
        // its recursive runtime. Temporarily moving that runtime out lets us
        // borrow the market bars directly instead of cloning the 4,096-bar
        // signal window on every replay update. The runtime is restored before
        // any order/protection work observes it.
        let config = session.execution_config.native_ema.clone();
        let source_update_sequence = session.execution_runtime.market_update_sequence;
        let market_update = session.execution_runtime.market_update_kind;
        let mut ema_runtime = std::mem::take(&mut session.execution_runtime.ema_execution);
        let result = {
            let bars = signal_evaluation_bars(session);
            if bars.is_empty() {
                bail!("latest strategy bar disappeared during strategy evaluation");
            }
            let current_side = side_from_signed_qty(current_qty);
            if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
                let signal_bar = bars
                    .last()
                    .cloned()
                    .context("latest strategy bar disappeared during strategy evaluation")?;
                let evaluation = config.evaluate_streaming_with_market_update(
                    &mut ema_runtime,
                    bars,
                    current_side,
                    source_update_sequence,
                    market_update,
                );
                (
                    signal_bar,
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            } else {
                // Preserve the legacy closed-bar behavior when several bars
                // arrived while an order/protection/session gate was active:
                // evaluate each newly eligible bar and retain the latest
                // actionable signal. The common one-bar case remains O(1).
                let start_idx = previous_strategy_ts
                    .and_then(|ts| bars.iter().position(|bar| bar.ts_ns > ts))
                    .unwrap_or_else(|| bars.len().saturating_sub(1));
                let mut latest = None;
                for idx in start_idx..bars.len() {
                    let signal_bar = bars[idx].clone();
                    let evaluation = config.evaluate_streaming_with_market_update(
                        &mut ema_runtime,
                        &bars[..=idx],
                        current_side,
                        (idx + 1 == bars.len())
                            .then_some(source_update_sequence)
                            .flatten(),
                        market_update,
                    );
                    let candidate = (
                        signal_bar,
                        evaluation.signal,
                        evaluation.summary(),
                        evaluation.debug_summary(),
                    );
                    if candidate.1 != StrategySignal::Hold || latest.is_none() {
                        latest = Some(candidate);
                    }
                }
                latest.context("latest strategy bar disappeared during strategy evaluation")?
            }
        };
        session.execution_runtime.ema_execution = ema_runtime;
        result
    } else {
        let bars = signal_evaluation_bars(session).to_vec();
        if bars.is_empty() {
            bail!("latest strategy bar disappeared during strategy evaluation");
        }
        evaluate_active_execution_strategy_since_mut(
            session,
            &bars,
            current_qty,
            previous_strategy_ts,
        )
    };
    let protection_bar = strategy_bars(session)
        .last()
        .cloned()
        .unwrap_or_else(|| signal_bar.clone());

    if let Some(window) = session_window_at(session, protection_bar.ts_ns) {
        if window.hold_entries {
            if actual_market_qty != 0 {
                record_replay_signal_diagnostic(
                    session,
                    signal_bar.ts_ns,
                    signal,
                    actual_market_qty,
                    current_qty,
                    Some(0),
                    "session_hold_flattening",
                    "session hold requires flattening before the close/reopen window",
                    Some(if actual_market_qty > 0 { "Sell" } else { "Buy" }),
                    Some(actual_market_qty.abs()),
                    &debug_summary,
                );
                emit_guarded_strategy_eval_debug(
                    event_tx,
                    session,
                    "session hold flattening position",
                    signal,
                    signal_bar.ts_ns,
                    actual_market_qty,
                    current_qty,
                    Some(0),
                    &debug_summary,
                );
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

            record_replay_signal_diagnostic(
                session,
                signal_bar.ts_ns,
                signal,
                actual_market_qty,
                current_qty,
                target_qty_for_signal(signal, current_qty, session.execution_config.order_qty),
                "session_hold_blocked_entries",
                "session hold blocks new entries while flat",
                None,
                None,
                &debug_summary,
            );
            emit_guarded_strategy_eval_debug(
                event_tx,
                session,
                "session hold blocked entries",
                signal,
                signal_bar.ts_ns,
                actual_market_qty,
                current_qty,
                target_qty_for_signal(signal, current_qty, session.execution_config.order_qty),
                &debug_summary,
            );
            sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
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
        record_replay_signal_diagnostic(
            session,
            signal_bar.ts_ns,
            signal,
            actual_market_qty,
            current_qty,
            None,
            "no_target",
            "signal did not produce a target position",
            None,
            None,
            &debug_summary,
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "no target",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            None,
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if target_qty == current_qty {
        record_replay_signal_diagnostic(
            session,
            signal_bar.ts_ns,
            signal,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            "target_already_current",
            "target position already matches effective position",
            None,
            None,
            &debug_summary,
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "target already current",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if closed_bar_signal_already_dispatched(session, signal_bar.ts_ns) {
        session.execution_runtime.last_summary = format!(
            "Signal on closed bar {} already dispatched; waiting for a new signal bar.",
            signal_bar.ts_ns
        );
        record_replay_signal_diagnostic(
            session,
            signal_bar.ts_ns,
            signal,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            "closed_bar_already_dispatched",
            "same closed-bar signal was already dispatched",
            None,
            None,
            &debug_summary,
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "closed-bar already dispatched",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if entry_signal_consumed_while_flat(session, signal, current_qty) {
        session.execution_runtime.last_summary = format!(
            "{} entry side already dispatched while flat; waiting for the opposite entry signal before another {}.",
            signal.label(),
            signal.label()
        );
        record_replay_signal_diagnostic(
            session,
            signal_bar.ts_ns,
            signal,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            "flat_entry_already_consumed",
            "entry side was already consumed while flat",
            None,
            None,
            &debug_summary,
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "flat entry side already consumed",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    record_replay_signal_diagnostic(
        session,
        signal_bar.ts_ns,
        signal,
        actual_market_qty,
        current_qty,
        Some(target_qty),
        "dispatching",
        "target delta passed all guarded execution gates",
        Some(if target_qty > current_qty {
            "Buy"
        } else {
            "Sell"
        }),
        Some(target_qty.saturating_sub(current_qty).abs()),
        &debug_summary,
    );
    emit_guarded_strategy_eval_debug(
        event_tx,
        session,
        "dispatching",
        signal,
        signal_bar.ts_ns,
        actual_market_qty,
        current_qty,
        Some(target_qty),
        &debug_summary,
    );

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} on {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        active_signal_timing_label(session),
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
        "{} {} on {} | {}",
        active_native_slug(session),
        signal.label(),
        active_signal_timing_label(session),
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
            mark_closed_bar_signal_dispatched(session, signal_bar.ts_ns, signal);
        }
    }
    emit_execution_state(event_tx, session);
    Ok(())
}
