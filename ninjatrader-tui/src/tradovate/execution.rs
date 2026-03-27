fn execution_state_snapshot(session: &SessionState) -> ExecutionStateSnapshot {
    let (take_profit_price, stop_price) = selected_protection_prices(session);
    ExecutionStateSnapshot {
        config: session.execution_config.clone(),
        runtime: session.execution_runtime.snapshot(),
        selected_account_id: session.selected_account_id,
        selected_contract_name: session.selected_contract.as_ref().map(|contract| contract.name.clone()),
        market_position_qty: selected_market_position_qty(session),
        market_entry_price: selected_market_entry_price(session),
        selected_contract_take_profit_price: take_profit_price,
        selected_contract_stop_price: stop_price,
    }
}

fn emit_execution_state(event_tx: &UnboundedSender<ServiceEvent>, session: &SessionState) {
    let _ = event_tx.send(ServiceEvent::ExecutionState(execution_state_snapshot(
        session,
    )));
}

fn closed_bars(session: &SessionState) -> &[Bar] {
    let closed_len = session.market.history_loaded.min(session.market.bars.len());
    &session.market.bars[..closed_len]
}

fn latest_closed_bar_ts(session: &SessionState) -> Option<i64> {
    closed_bars(session).last().map(|bar| bar.ts_ns)
}

fn session_window_at(session: &SessionState, ts_ns: i64) -> Option<InstrumentSessionWindow> {
    session
        .market
        .session_profile
        .map(|profile| profile.evaluate(ts_ns))
}

fn selected_contract_positions<'a>(session: &'a SessionState) -> Vec<&'a Value> {
    let Some(account_id) = session.selected_account_id else {
        return Vec::new();
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return Vec::new();
    };
    session
        .user_store
        .positions
        .get(&account_id)
        .into_iter()
        .flat_map(|positions| positions.values())
        .filter(|position| position_matches_contract(position, contract))
        .collect()
}

fn selected_market_position_qty(session: &SessionState) -> i32 {
    let qty = selected_contract_positions(session)
        .into_iter()
        .filter_map(position_qty)
        .sum::<f64>();
    qty.round() as i32
}

fn selected_market_entry_price(session: &SessionState) -> Option<f64> {
    let positions = selected_contract_positions(session);
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in positions {
        let qty = position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }

    if total_qty <= f64::EPSILON {
        None
    } else {
        Some(weighted_sum / total_qty)
    }
}

fn selected_protection_prices(session: &SessionState) -> (Option<f64>, Option<f64>) {
    let Some(account_id) = session.selected_account_id else {
        return (None, None);
    };
    let Some(contract_id) = session.market.contract_id else {
        return (None, None);
    };

    session
        .managed_protection
        .get(&StrategyProtectionKey {
            account_id,
            contract_id,
        })
        .map(|orders| (orders.take_profit_price, orders.stop_price))
        .unwrap_or((None, None))
}

fn active_native_slug(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.slug()
}

fn active_native_label(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.label()
}

fn active_native_uses_protection(session: &SessionState) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            session.execution_config.native_hma.uses_native_protection()
        }
        NativeStrategyKind::EmaCross => {
            session.execution_config.native_ema.uses_native_protection()
        }
    }
}

fn active_order_strategy_matches_selected(session: &SessionState) -> Option<TrackedOrderStrategy> {
    let key = selected_strategy_key(session).ok()?;
    session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .cloned()
}

fn reconcile_selected_active_order_strategy(session: &mut SessionState) {
    let Some(key) = selected_strategy_key(session).ok() else {
        session.active_order_strategy = None;
        return;
    };
    let selected = session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(|strategy| {
            Some(TrackedOrderStrategy {
                key,
                order_strategy_id: extract_entity_id(strategy)?,
                target_qty: active_order_strategy_matches_selected(session)
                    .map(|tracked| tracked.target_qty)
                    .unwrap_or_default(),
            })
        });

    if let Some(selected) = selected {
        session.active_order_strategy = Some(selected);
    } else if let Some(tracked) = session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .cloned()
    {
        let has_open_position = selected_market_position_qty(session) != 0;
        let has_linked_orders = !session
            .user_store
            .linked_strategy_orders(key.account_id, tracked.order_strategy_id)
            .is_empty();
        if !has_open_position && !has_linked_orders {
            session.active_order_strategy = None;
        }
    }
}

fn hydrate_selected_order_strategy_protection(session: &mut SessionState) {
    let Some(tracked) = active_order_strategy_matches_selected(session) else {
        return;
    };
    let signed_qty = selected_market_position_qty(session);
    if signed_qty == 0 {
        session.managed_protection.remove(&tracked.key);
        return;
    }

    let linked_orders = session
        .user_store
        .linked_strategy_orders(tracked.key.account_id, tracked.order_strategy_id);
    if linked_orders.is_empty() {
        return;
    }

    let mut take_profit_order_id = None;
    let mut stop_order_id = None;
    let mut take_profit_price = None;
    let mut stop_price = None;
    let mut take_profit_cl_ord_id = None;
    let mut stop_cl_ord_id = None;

    for order in linked_orders {
        if !order_is_active(order) {
            continue;
        }
        if order_contract_id(order) != Some(tracked.key.contract_id) {
            continue;
        }
        match order_type(order).as_deref().unwrap_or_default() {
            "limit" | "mit" => {
                take_profit_order_id = extract_entity_id(order);
                take_profit_price = pick_number(order, &["price"]);
                take_profit_cl_ord_id = order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
            }
            "stop" | "stoplimit" | "trailingstop" | "trailingstoplimit" => {
                stop_order_id = extract_entity_id(order);
                stop_price = pick_number(order, &["stopPrice", "price"]);
                stop_cl_ord_id = order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
            }
            _ => {}
        }
    }

    if take_profit_order_id.is_none() && stop_order_id.is_none() {
        return;
    }

    session.managed_protection.insert(
        tracked.key,
        ManagedProtectionOrders {
            signed_qty,
            take_profit_price,
            stop_price,
            take_profit_cl_ord_id,
            stop_cl_ord_id,
            take_profit_order_id,
            stop_order_id,
        },
    );
}

fn clear_selected_order_strategy_state(session: &mut SessionState) {
    let Some(key) = selected_strategy_key(session).ok() else {
        session.active_order_strategy = None;
        return;
    };

    let mut stale_ids = Vec::new();
    if let Some(strategy_id) = session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(extract_entity_id)
    {
        stale_ids.push(strategy_id);
    }
    if let Some(strategy_id) = session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .map(|tracked| tracked.order_strategy_id)
    {
        if !stale_ids.contains(&strategy_id) {
            stale_ids.push(strategy_id);
        }
    }

    for strategy_id in stale_ids {
        session.user_store.order_strategies.remove(&strategy_id);
        session
            .user_store
            .order_strategy_links
            .retain(|_, link| json_i64(link, "orderStrategyId") != Some(strategy_id));
    }

    if session
        .active_order_strategy
        .as_ref()
        .is_some_and(|tracked| tracked.key == key)
    {
        session.active_order_strategy = None;
    }
    session.managed_protection.remove(&key);
}

fn should_wait_for_strategy_owned_protection(session: &SessionState) -> bool {
    if !native_order_strategy_enabled(session) || selected_market_position_qty(session) == 0 {
        return false;
    }
    let Some(tracked) = active_order_strategy_matches_selected(session) else {
        return false;
    };
    let has_linked_orders = session
        .user_store
        .linked_strategy_orders(tracked.key.account_id, tracked.order_strategy_id)
        .into_iter()
        .any(|order| {
            order_is_active(order) && order_contract_id(order) == Some(tracked.key.contract_id)
        });
    has_linked_orders && !session.managed_protection.contains_key(&tracked.key)
}

fn sync_active_execution_position(
    session: &mut SessionState,
    signed_qty: i32,
    entry_price: Option<f64>,
) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.sync_position(
            &mut session.execution_runtime.hma_execution,
            signed_qty,
            entry_price,
        ),
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.sync_position(
            &mut session.execution_runtime.ema_execution,
            signed_qty,
            entry_price,
        ),
    }
}

fn take_profit_price(session: &SessionState, entry_price: f64, signed_qty: i32) -> Option<f64> {
    let side = side_from_signed_qty(signed_qty)?;
    let offset = match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session
            .execution_config
            .native_hma
            .take_profit_offset(session.market.tick_size)?,
        NativeStrategyKind::EmaCross => session
            .execution_config
            .native_ema
            .take_profit_offset(session.market.tick_size)?,
    };

    Some(match side {
        crate::strategies::PositionSide::Long => entry_price + offset,
        crate::strategies::PositionSide::Short => entry_price - offset,
    })
}

fn combined_stop_price(session: &mut SessionState, trailing_bar: Option<&Bar>) -> Option<f64> {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_hma
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.hma_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_hma
                .current_effective_stop_price(
                    &session.execution_runtime.hma_execution,
                    session.market.tick_size,
                )
        }
        NativeStrategyKind::EmaCross => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_ema
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.ema_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_ema
                .current_effective_stop_price(
                    &session.execution_runtime.ema_execution,
                    session.market.tick_size,
                )
        }
    }
}

fn sync_execution_protection(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    trailing_bar: Option<&Bar>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }
    if !active_native_uses_protection(session) {
        return Ok(());
    }
    if session.execution_runtime.pending_target_qty.is_some() {
        return Ok(());
    }
    if should_wait_for_strategy_owned_protection(session) {
        return Ok(());
    }

    let signed_qty = selected_market_position_qty(session);
    let entry_price = selected_market_entry_price(session);
    sync_active_execution_position(session, signed_qty, entry_price);
    let reason = if signed_qty == 0 {
        format!("{} flat", active_native_slug(session))
    } else if trailing_bar.is_some() {
        format!("{} bar sync", active_native_slug(session))
    } else {
        format!("{} position sync", active_native_slug(session))
    };

    let (take_profit_price, stop_price) = if signed_qty == 0 {
        (None, None)
    } else if let Some(entry_price) = entry_price {
        (
            take_profit_price(session, entry_price, signed_qty),
            combined_stop_price(session, trailing_bar),
        )
    } else {
        return Ok(());
    };

    sync_native_protection(
        session,
        broker_tx,
        signed_qty,
        take_profit_price,
        stop_price,
        &reason,
    )
}

fn effective_market_position_qty(session: &SessionState) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

fn pending_target_has_live_broker_path(session: &SessionState) -> bool {
    if session.order_submit_in_flight {
        return true;
    }

    let Some(tracker) = session.order_latency_tracker.as_ref() else {
        return false;
    };
    let Some(account_id) = session.selected_account_id else {
        return false;
    };

    if let Some(order_id) = tracker
        .order_id
        .or_else(|| session.user_store.order_id_by_client_id(account_id, &tracker.cl_ord_id))
    {
        if session
            .user_store
            .find_order(account_id, order_id)
            .is_some_and(order_is_active)
        {
            return true;
        }
    }

    let Some(key) = selected_strategy_key(session).ok() else {
        return false;
    };
    let Some(order_strategy_id) = tracker.order_strategy_id else {
        return false;
    };

    if session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(extract_entity_id)
        == Some(order_strategy_id)
    {
        return true;
    }

    session
        .user_store
        .linked_strategy_orders(key.account_id, order_strategy_id)
        .into_iter()
        .any(|order| order_is_active(order) && order_contract_id(order) == Some(key.contract_id))
}

fn clear_stale_pending_target(
    session: &mut SessionState,
    pending: i32,
    actual_qty: i32,
    event_tx: &UnboundedSender<ServiceEvent>,
) {
    session.execution_runtime.pending_target_qty = None;
    session.pending_signal_context = None;
    session.order_latency_tracker = None;
    session.execution_runtime.last_summary = format!(
        "Pending target {pending} cleared: broker has no active order path and position is still {actual_qty}; re-evaluating."
    );
    session.execution_runtime.last_closed_bar_ts = latest_closed_bar_ts(session)
        .map(|last_closed_ts| last_closed_ts.saturating_sub(1));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Pending target {pending} cleared: broker has no active order path; re-evaluating."
    )));
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "pending target cleared | target {pending} | actual {actual_qty} | broker has no active order path"
    )));
}

fn evaluate_active_execution_strategy(
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

fn target_qty_for_signal(signal: StrategySignal, current_qty: i32, base_qty: i32) -> Option<i32> {
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

fn arm_execution_strategy(session: &mut SessionState) {
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
    session.execution_runtime.last_closed_bar_ts = latest_closed_bar_ts(session);
    session.execution_runtime.last_summary =
        if session.execution_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current closed bar.",
                active_native_label(session)
            )
        } else {
            format!(
                "Native {} armed; waiting for first closed bar.",
                active_native_label(session)
            )
        };
}

fn disarm_execution_strategy(session: &mut SessionState, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

fn handle_execution_account_sync(
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
        disarm_execution_strategy(
            session,
            format!(
                "Automation disarmed: position drifted to {actual_qty}, above configured max {max_automated_qty}."
            ),
        );
        runtime_changed = true;
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
            session.execution_runtime.last_summary = if reached {
                format!("Position confirmed at target {actual_qty}")
            } else {
                format!(
                    "Position at {actual_qty} (target was {pending}); re-evaluating on next bar"
                )
            };
            runtime_changed = true;
        } else if pending != 0 && !pending_target_has_live_broker_path(session) {
            clear_stale_pending_target(session, pending, actual_qty, event_tx);
            runtime_changed = true;
        }
    }

    if session.execution_runtime.armed && session.execution_config.kind == StrategyKind::Native {
        sync_execution_protection(session, broker_tx, None)?;
    }

    if runtime_changed {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

fn maybe_run_execution_strategy(
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
    let max_automated_qty = session.execution_config.order_qty.max(1);
    if actual_market_qty.abs() > max_automated_qty {
        disarm_execution_strategy(
            session,
            format!(
                "Automation disarmed: position drifted to {actual_market_qty}, above configured max {max_automated_qty}."
            ),
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        session.execution_runtime.last_summary = format!(
            "Waiting for prior automated order to settle (actual {actual_market_qty}, pending target {pending_target_qty})."
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let Some(last_closed_ts) = latest_closed_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for market data.",
            active_native_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_closed_ts);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current bar; waiting for next close.",
            active_native_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_runtime.last_closed_bar_ts == Some(last_closed_ts) {
        return Ok(());
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_closed_ts);

    let current_qty = effective_market_position_qty(session);
    let (last_closed, signal, summary) = {
        let closed = closed_bars(session);
        let last_closed = closed
            .last()
            .cloned()
            .context("latest closed bar disappeared during strategy evaluation")?;
        let (signal, summary) = evaluate_active_execution_strategy(session, closed, current_qty);
        (last_closed, signal, summary)
    };

    if let Some(window) = session_window_at(session, last_closed.ts_ns) {
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

            sync_execution_protection(session, broker_tx, Some(&last_closed))?;
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
        sync_execution_protection(session, broker_tx, Some(&last_closed))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if target_qty == current_qty {
        sync_execution_protection(session, broker_tx, Some(&last_closed))?;
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

#[cfg(test)]
mod execution_tests {
    use super::*;
    use serde_json::json;

    fn test_session() -> SessionState {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        SessionState {
            cfg: AppConfig::default(),
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: vec![AccountInfo {
                id: 42,
                name: "SIM".to_string(),
                raw: json!({}),
            }],
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: None,
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store: UserSyncStore::default(),
            selected_account_id: Some(42),
            selected_contract: Some(ContractSuggestion {
                id: 3570918,
                name: "ESH6".to_string(),
                description: "E-mini S&P".to_string(),
                raw: json!({}),
            }),
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
        }
    }

    #[test]
    fn reconcile_keeps_known_strategy_id_while_position_is_open() {
        let mut session = test_session();
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: -1,
        });
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": -1
                }),
            )]),
        );

        reconcile_selected_active_order_strategy(&mut session);

        assert_eq!(
            session
                .active_order_strategy
                .as_ref()
                .map(|tracked| tracked.order_strategy_id),
            Some(77)
        );
    }

    #[test]
    fn does_not_wait_for_strategy_owned_protection_without_linked_orders() {
        let mut session = test_session();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_ema.take_profit_ticks = 8.0;
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: -1,
        });
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": -1
                }),
            )]),
        );

        assert!(!should_wait_for_strategy_owned_protection(&session));
    }

    #[test]
    fn stale_pending_target_clears_when_broker_has_no_live_order_path() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        session.execution_runtime.pending_target_qty = Some(1);
        session.execution_runtime.last_closed_bar_ts = Some(100);
        session.order_latency_tracker = Some(OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
            cl_ord_id: "midas-stale-entry".to_string(),
            order_id: Some(77),
            order_strategy_id: None,
            seen_recorded: true,
            exec_report_recorded: false,
            fill_recorded: false,
        });
        session.market.bars = vec![Bar {
            ts_ns: 100,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        }];
        session.market.history_loaded = 1;

        handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
            .expect("stale pending target should reconcile");

        assert_eq!(session.execution_runtime.pending_target_qty, None);
        assert!(session.order_latency_tracker.is_none());
        assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(99));
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Pending target 1 cleared"));

        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.into_iter().any(|event| matches!(
            event,
            ServiceEvent::Status(message)
                if message.contains("Pending target 1 cleared")
        )));
    }

    #[test]
    fn waits_for_strategy_owned_protection_when_linked_orders_exist() {
        let mut session = test_session();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_ema.take_profit_ticks = 8.0;
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: -1,
        });
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": -1
                }),
            )]),
        );
        session.user_store.orders.insert(
            42,
            BTreeMap::from([(
                1001,
                json!({
                    "id": 1001,
                    "accountId": 42,
                    "contractId": 3570918,
                    "ordStatus": "Working"
                }),
            )]),
        );
        session.user_store.order_strategy_links.insert(
            1,
            json!({
                "id": 1,
                "orderStrategyId": 77,
                "orderId": 1001
            }),
        );

        assert!(should_wait_for_strategy_owned_protection(&session));
    }

    #[test]
    fn execution_state_snapshot_includes_selected_protection_prices() {
        let mut session = test_session();
        session.market.contract_id = Some(3570918);
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 6659.75
                }),
            )]),
        );
        session.managed_protection.insert(
            StrategyProtectionKey {
                account_id: 42,
                contract_id: 3570918,
            },
            ManagedProtectionOrders {
                signed_qty: 1,
                take_profit_price: Some(6662.5),
                stop_price: Some(6659.0),
                take_profit_cl_ord_id: None,
                stop_cl_ord_id: None,
                take_profit_order_id: None,
                stop_order_id: None,
            },
        );

        let snapshot = execution_state_snapshot(&session);

        assert_eq!(snapshot.market_entry_price, Some(6659.75));
        assert_eq!(snapshot.selected_contract_take_profit_price, Some(6662.5));
        assert_eq!(snapshot.selected_contract_stop_price, Some(6659.0));
    }
}
