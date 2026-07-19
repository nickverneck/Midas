use super::*;

pub(crate) fn maybe_run_hma_direct_execution_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    if session.execution_config.native_strategy != NativeStrategyKind::HmaCross {
        session.execution_runtime.last_summary =
            "HMA Direct path only supports HMA Crossover; falling back to no-op.".to_string();
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let actual_qty = selected_market_position_qty(session);
    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "HMA Direct {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        if session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts) {
            let gate_detail = format!(
                "hma direct gate | closed-bar timing waiting for new signal bar | last_bar_ts {} | {}",
                last_strategy_ts,
                hma_cross_market_debug(session, actual_qty)
            );
            let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
                session,
                TradovateStrategyDecisionDebug {
                    path: "hma direct",
                    decision: "blocked",
                    signal: None,
                    bar_ts: Some(last_strategy_ts),
                    actual_qty,
                    effective_qty: actual_qty,
                    target_qty: None,
                    strategy_detail: "n/a",
                    gate_detail,
                    fingerprint: latest_strategy_bar_fingerprint(session),
                },
            )));
            return Ok(());
        }
        session.execution_runtime.last_closed_bar_fingerprint =
            latest_strategy_bar_fingerprint(session);
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);

    let bars = signal_evaluation_bars(session).to_vec();
    if bars.is_empty() {
        bail!("latest strategy bar disappeared during HMA direct evaluation");
    }
    let signal_bar = bars
        .last()
        .expect("checked non-empty strategy bars")
        .clone();
    let config = session.execution_config.native_hma_cross.clone();
    let evaluation = config.evaluate_current_cross(
        &mut session.execution_runtime.hma_cross_execution,
        &bars,
        side_from_signed_qty(actual_qty),
    );
    let signal = evaluation.signal;
    let debug_summary = evaluation.debug_summary();
    let summary = format!(
        "{} | {}",
        evaluation.summary(),
        hma_cross_market_debug(session, actual_qty)
    );
    session.execution_runtime.last_summary = format!("HMA Direct: {summary}");

    let Some(target_qty) =
        target_qty_for_signal(signal, actual_qty, session.execution_config.order_qty)
    else {
        let gate_detail = format!(
            "hma direct eval | {} | signal {} | no target | bar_ts {} | actual_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            signal_bar.ts_ns,
            actual_qty,
            hma_cross_market_debug(session, actual_qty)
        );
        let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
            session,
            TradovateStrategyDecisionDebug {
                path: "hma direct",
                decision: "no target",
                signal: Some(signal),
                bar_ts: Some(signal_bar.ts_ns),
                actual_qty,
                effective_qty: actual_qty,
                target_qty: None,
                strategy_detail: &debug_summary,
                gate_detail,
                fingerprint: latest_strategy_bar_fingerprint(session),
            },
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    let delta = target_qty.saturating_sub(actual_qty);
    if delta == 0 {
        let gate_detail = format!(
            "hma direct eval | {} | signal {} | target already actual | target_qty {} | bar_ts {} | actual_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            target_qty,
            signal_bar.ts_ns,
            actual_qty,
            hma_cross_market_debug(session, actual_qty)
        );
        let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
            session,
            TradovateStrategyDecisionDebug {
                path: "hma direct",
                decision: "target already actual",
                signal: Some(signal),
                bar_ts: Some(signal_bar.ts_ns),
                actual_qty,
                effective_qty: actual_qty,
                target_qty: Some(target_qty),
                strategy_detail: &debug_summary,
                gate_detail,
                fingerprint: latest_strategy_bar_fingerprint(session),
            },
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_dispatched_signal_bar_ts == Some(signal_bar.ts_ns)
    {
        session.execution_runtime.last_summary = format!(
            "HMA direct signal on closed bar {} already dispatched; waiting for a new signal bar.",
            signal_bar.ts_ns
        );
        let gate_detail = format!(
            "hma direct eval | {} | closed-bar already dispatched | signal {} | bar_ts {} | actual_qty {} | target_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            signal_bar.ts_ns,
            actual_qty,
            target_qty,
            hma_cross_market_debug(session, actual_qty)
        );
        let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
            session,
            TradovateStrategyDecisionDebug {
                path: "hma direct",
                decision: "closed-bar already dispatched",
                signal: Some(signal),
                bar_ts: Some(signal_bar.ts_ns),
                actual_qty,
                effective_qty: actual_qty,
                target_qty: Some(target_qty),
                strategy_detail: &debug_summary,
                gate_detail,
                fingerprint: latest_strategy_bar_fingerprint(session),
            },
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if entry_signal_consumed_while_flat(session, signal, actual_qty) {
        session.execution_runtime.last_summary = format!(
            "HMA direct {} entry side already dispatched while flat; waiting for the opposite entry signal before another {}.",
            signal.label(),
            signal.label()
        );
        let gate_detail = format!(
            "hma direct eval | {} | flat entry side already consumed | signal {} | bar_ts {} | actual_qty {} | target_qty {} | {}",
            active_native_slug(session),
            signal.label(),
            signal_bar.ts_ns,
            actual_qty,
            target_qty,
            hma_cross_market_debug(session, actual_qty)
        );
        let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
            session,
            TradovateStrategyDecisionDebug {
                path: "hma direct",
                decision: "flat entry side already consumed",
                signal: Some(signal),
                bar_ts: Some(signal_bar.ts_ns),
                actual_qty,
                effective_qty: actual_qty,
                target_qty: Some(target_qty),
                strategy_detail: &debug_summary,
                gate_detail,
                fingerprint: latest_strategy_bar_fingerprint(session),
            },
        )));
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let account_id = session
        .selected_account_id
        .context("select an account before sending HMA direct orders")?;
    let account = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .cloned()
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .clone()
        .context("select a contract before sending HMA direct orders")?;

    let order_action = if delta > 0 { "Buy" } else { "Sell" };
    let order_qty = delta.unsigned_abs() as i32;
    let reason = format!(
        "hma direct {} {} on {} | actual {} -> target {} | {}",
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
            "hma direct {} {} (qty {} -> {})",
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
        "HMA Direct Strategy",
        true,
        Some(&reason),
        Some(target_qty),
        None,
        Vec::new(),
    );
    enqueue_market_order(session, broker_tx, order)?;
    session.execution_runtime.pending_target_qty = Some(target_qty);
    mark_closed_bar_signal_dispatched(session, signal_bar.ts_ns, signal);
    let gate_detail = format!(
        "hma direct dispatch | {} | endpoint order/placeorder | signal {} | bar_ts {} | actual_qty {} | target_qty {} | order_action {} | order_qty {} | submit_in_flight ignored | pending_target overwritten | {}",
        active_native_slug(session),
        signal.label(),
        signal_bar.ts_ns,
        actual_qty,
        target_qty,
        order_action,
        order_qty,
        hma_cross_market_debug(session, actual_qty)
    );
    let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
        session,
        TradovateStrategyDecisionDebug {
            path: "hma direct",
            decision: "dispatching",
            signal: Some(signal),
            bar_ts: Some(signal_bar.ts_ns),
            actual_qty,
            effective_qty: actual_qty,
            target_qty: Some(target_qty),
            strategy_detail: &debug_summary,
            gate_detail,
            fingerprint: latest_strategy_bar_fingerprint(session),
        },
    )));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "HMA direct {} signal: {} {} (qty {} -> {})",
        active_native_slug(session),
        order_action,
        order_qty,
        actual_qty,
        target_qty
    )));
    emit_execution_state(event_tx, session);
    Ok(())
}

pub(crate) fn hma_cross_market_debug(session: &SessionState, actual_qty: i32) -> String {
    let closed_len = effective_closed_bar_len(session);
    let market_len = session.market.bars.len();
    let closed_ts = closed_bars(session).last().map(|bar| bar.ts_ns);
    let forming_ts = session.market.bars.get(closed_len).map(|bar| bar.ts_ns);
    let current_side = side_from_signed_qty(actual_qty);
    let closed_summary = session
        .execution_config
        .native_hma_cross
        .evaluate(closed_bars(session), current_side)
        .summary();
    let forming_summary = if forming_ts.is_some() {
        session
            .execution_config
            .native_hma_cross
            .evaluate(&session.market.bars, current_side)
            .summary()
    } else {
        "no forming bar".to_string()
    };

    format!(
        "closed_len {closed_len} | market_bars {market_len} | closed_ts {:?} | forming_ts {:?} | stored_side {} | stored_delta {} | closed_eval [{}] | forming_preview [{}]",
        closed_ts,
        forming_ts,
        session
            .execution_runtime
            .hma_cross_execution
            .last_observed_side
            .map(|side| side.label())
            .unwrap_or("unset"),
        session
            .execution_runtime
            .hma_cross_execution
            .last_observed_delta
            .map(|delta| format!("{delta:.4}"))
            .unwrap_or_else(|| "unset".to_string()),
        closed_summary,
        forming_summary
    )
}
