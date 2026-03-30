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

fn strategy_bars(session: &SessionState) -> &[Bar] {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        &session.market.bars
    } else {
        closed_bars(session)
    }
}

fn latest_strategy_bar_ts(session: &SessionState) -> Option<i64> {
    strategy_bars(session).last().map(|bar| bar.ts_ns)
}

fn active_signal_timing_label(session: &SessionState) -> &'static str {
    match session.execution_config.native_signal_timing {
        NativeSignalTiming::ClosedBar => "closed bar",
        NativeSignalTiming::LiveBar => "live bar",
    }
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

fn format_selected_managed_protection(session: &SessionState) -> String {
    let Some(key) = selected_strategy_key(session).ok() else {
        return "none".to_string();
    };
    let Some(protection) = session.managed_protection.get(&key) else {
        return "none".to_string();
    };

    fn format_leg(
        label: &str,
        price: Option<f64>,
        order_id: Option<i64>,
        cl_ord_id: Option<&str>,
    ) -> String {
        match price {
            Some(price) => format!(
                "{label} {price:.2} [order {} clOrdId {}]",
                order_id
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                cl_ord_id.unwrap_or("none"),
            ),
            None => format!("{label} n/a"),
        }
    }

    format!(
        "qty {} | {} | {}",
        protection.signed_qty,
        format_leg(
            "tp",
            protection.take_profit_price,
            protection.take_profit_order_id,
            protection.take_profit_cl_ord_id.as_deref(),
        ),
        format_leg(
            "sl",
            protection.stop_price,
            protection.stop_order_id,
            protection.stop_cl_ord_id.as_deref(),
        ),
    )
}

fn format_selected_tracker_state(session: &SessionState) -> String {
    let Some(tracker) = session.order_latency_tracker.as_ref() else {
        return "none".to_string();
    };

    format!(
        "request {} | order {} | strategy {} | signal {}",
        tracker.cl_ord_id,
        tracker
            .order_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        tracker
            .order_strategy_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        tracker.signal_context.as_deref().unwrap_or("none"),
    )
}

fn active_linked_order_count(
    session: &SessionState,
    account_id: i64,
    contract_id: i64,
    order_strategy_id: i64,
) -> usize {
    session
        .user_store
        .linked_strategy_orders(account_id, order_strategy_id)
        .into_iter()
        .filter(|order| order_is_active(order) && order_contract_id(order) == Some(contract_id))
        .count()
}

fn execution_observability_context(session: &SessionState) -> String {
    let pending_target = session
        .execution_runtime
        .pending_target_qty
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let tracked_strategy = active_order_strategy_matches_selected(session);
    let tracked_strategy_id = tracked_strategy.as_ref().map(|tracked| tracked.order_strategy_id);
    let broker_strategy_id = selected_strategy_key(session)
        .ok()
        .and_then(|key| {
            session
                .user_store
                .find_active_order_strategy(key.account_id, key.contract_id)
                .and_then(extract_entity_id)
                .map(|strategy_id| (key, strategy_id))
        });

    let tracked_linked_orders = tracked_strategy
        .as_ref()
        .map(|tracked| {
            active_linked_order_count(
                session,
                tracked.key.account_id,
                tracked.key.contract_id,
                tracked.order_strategy_id,
            )
        })
        .unwrap_or_default();
    let broker_linked_orders = broker_strategy_id
        .map(|(key, strategy_id)| {
            active_linked_order_count(session, key.account_id, key.contract_id, strategy_id)
        })
        .unwrap_or_default();

    format!(
        "pending target {} | tracker {} | tracked strategy {} ({} active linked) | broker strategy {} ({} active linked) | managed {}",
        pending_target,
        format_selected_tracker_state(session),
        tracked_strategy_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        tracked_linked_orders,
        broker_strategy_id
            .map(|(_, value)| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        broker_linked_orders,
        format_selected_managed_protection(session),
    )
}

fn emit_execution_transition_debug(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    next_summary: &str,
    label: &str,
) {
    if session.execution_runtime.last_summary == next_summary {
        return;
    }

    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "{label} | {next_summary} | {}",
        execution_observability_context(session)
    )));
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
            last_requested_take_profit_price: take_profit_price,
            last_requested_stop_price: stop_price,
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

fn strategy_has_live_broker_path(
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

const ORDER_STRATEGY_HYDRATION_GRACE_MS: u128 = 1_500;
const MARKET_ORDER_POSITION_SYNC_GRACE_MS: u128 = 3_000;

fn tracker_within_broker_path_grace(
    session: &SessionState,
    order_strategy_id: i64,
) -> bool {
    session.order_latency_tracker.as_ref().is_some_and(|tracker| {
        tracker.order_strategy_id == Some(order_strategy_id)
            && tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_HYDRATION_GRACE_MS
    })
}

fn selected_contract_has_live_broker_path(session: &SessionState) -> bool {
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

fn pending_target_has_live_broker_path(session: &SessionState) -> bool {
    selected_contract_has_live_broker_path(session)
}

fn should_wait_for_market_position_sync(
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
    if tracker.order_strategy_id.is_some() {
        return false;
    }
    if !(tracker.order_id.is_some()
        || tracker.seen_recorded
        || tracker.exec_report_recorded
        || tracker.fill_recorded)
    {
        return false;
    }

    tracker.started_at.elapsed().as_millis() <= MARKET_ORDER_POSITION_SYNC_GRACE_MS
}

fn clear_stale_pending_target(
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
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session)
        .map(|last_strategy_ts| last_strategy_ts.saturating_sub(1));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Pending target {pending} cleared: broker has no active order path; re-evaluating."
    )));
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "pending target cleared | target {pending} | actual {actual_qty} | broker has no active order path | {observability}"
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

fn continue_staged_reversal(
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

        let dispatch_outcome =
            dispatch_target_position_order(session, broker_tx, staged.target_qty, true, &staged.reason)?;
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
    let dispatch_outcome = dispatch_target_position_order(session, broker_tx, 0, true, &flatten_reason)?;
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
            if should_wait_for_market_position_sync(session, pending, actual_qty) {
                let next_summary = format!(
                    "Waiting for position sync after market order settle (actual {actual_qty}, pending target {pending})."
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
            if should_wait_for_market_position_sync(session, pending_target_qty, actual_market_qty)
            {
                let next_summary = format!(
                    "Waiting for position sync after market order settle (actual {actual_market_qty}, pending target {pending_target_qty})."
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

#[cfg(test)]
mod execution_tests {
    use super::*;
    use serde_json::{Value, json};

    fn test_session() -> SessionState {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
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
            started_at: time::Instant::now() - time::Duration::from_secs(4),
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
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::Status(message)
                if message.contains("Pending target 1 cleared")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::DebugLog(message)
                if message.contains("pending target cleared")
                    && message.contains("request midas-stale-entry")
                    && message.contains("pending target 1")
        )));
    }

    #[test]
    fn strategy_loop_clears_stale_pending_target_when_broker_path_missing() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        session.execution_runtime.pending_target_qty = Some(-1);
        session.execution_runtime.last_closed_bar_ts = Some(200);
        session.order_latency_tracker = Some(OrderLatencyTracker {
            started_at: time::Instant::now() - time::Duration::from_secs(3),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Sell (qty 0 -> -1)".to_string()),
            cl_ord_id: "midas-stale-strategy-loop".to_string(),
            order_id: Some(88),
            order_strategy_id: Some(77),
            seen_recorded: true,
            exec_report_recorded: false,
            fill_recorded: false,
        });
        session.market.bars = vec![Bar {
            ts_ns: 200,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        }];
        session.market.history_loaded = 1;

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("strategy loop should clear stale pending target");

        assert_eq!(session.execution_runtime.pending_target_qty, None);
        assert!(session.order_latency_tracker.is_none());
        assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(199));
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared"));

        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::Status(message)
                if message.contains("Pending target -1 cleared")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::DebugLog(message)
                if message.contains("pending target cleared")
                    && message.contains("request midas-stale-strategy-loop")
                    && message.contains("pending target -1")
        )));
    }

    #[test]
    fn strategy_loop_keeps_pending_target_during_broker_path_grace_after_ack() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        session.execution_runtime.pending_target_qty = Some(1);
        session.execution_runtime.last_closed_bar_ts = Some(300);
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: 1,
        });
        session.order_latency_tracker = Some(OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
            cl_ord_id: "midas-fresh-strategy-loop".to_string(),
            order_id: Some(88),
            order_strategy_id: Some(77),
            seen_recorded: true,
            exec_report_recorded: false,
            fill_recorded: false,
        });
        session.market.bars = vec![Bar {
            ts_ns: 300,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        }];
        session.market.history_loaded = 1;

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("fresh order-strategy ack should keep pending target briefly");

        assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
        assert!(session.order_latency_tracker.is_some());
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Waiting for prior automated order to settle"));
    }

    #[test]
    fn strategy_loop_keeps_market_order_pending_target_during_position_sync_grace() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        session.execution_runtime.pending_target_qty = Some(-1);
        session.execution_runtime.last_closed_bar_ts = Some(320);
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1
                }),
            )]),
        );
        session.order_latency_tracker = Some(OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
            cl_ord_id: "midas-market-sync".to_string(),
            order_id: Some(88),
            order_strategy_id: None,
            seen_recorded: true,
            exec_report_recorded: true,
            fill_recorded: true,
        });
        session.market.bars = vec![Bar {
            ts_ns: 320,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        }];
        session.market.history_loaded = 1;

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("market-order reversal should wait for position sync");

        assert_eq!(session.execution_runtime.pending_target_qty, Some(-1));
        assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(320));
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Waiting for position sync after market order settle"));

        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::DebugLog(message)
                if message.contains("execution market position sync wait")
                    || message.contains("Waiting for position sync after market order settle")
        )));
    }

    #[test]
    fn strategy_loop_clears_market_order_pending_target_after_position_sync_grace_expires() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        session.execution_runtime.pending_target_qty = Some(-1);
        session.execution_runtime.last_closed_bar_ts = Some(330);
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1
                }),
            )]),
        );
        session.order_latency_tracker = Some(OrderLatencyTracker {
            started_at: time::Instant::now() - time::Duration::from_secs(4),
            signal_started_at: Some(time::Instant::now()),
            signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
            cl_ord_id: "midas-market-sync-stale".to_string(),
            order_id: Some(99),
            order_strategy_id: None,
            seen_recorded: true,
            exec_report_recorded: true,
            fill_recorded: true,
        });
        session.market.bars = vec![Bar {
            ts_ns: 330,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        }];
        session.market.history_loaded = 1;

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("stale market-order pending target should eventually clear");

        assert_eq!(session.execution_runtime.pending_target_qty, None);
        assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(329));
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Pending target -1 cleared"));

        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            ServiceEvent::DebugLog(message)
                if message.contains("pending target cleared")
                    && message.contains("request midas-market-sync-stale")
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
    fn account_sync_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_ema.take_profit_ticks = 8.0;
        session.execution_runtime.armed = true;
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
                    "netPos": -2,
                    "netPrice": 5000.0
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

        handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
            .expect("live strategy path should suppress transient drift disarm");

        assert!(session.execution_runtime.armed);
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync"));
    }

    #[test]
    fn broker_sync_wait_logs_once_with_observability_context() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_runtime.armed = true;
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: -1,
        });
        session.managed_protection.insert(
            key,
            ManagedProtectionOrders {
                signed_qty: -2,
                take_profit_price: Some(4998.0),
                stop_price: Some(5002.0),
                last_requested_take_profit_price: Some(4998.0),
                last_requested_stop_price: Some(5002.0),
                take_profit_cl_ord_id: Some("midas-tp".to_string()),
                stop_cl_ord_id: Some("midas-sl".to_string()),
                take_profit_order_id: Some(1002),
                stop_order_id: Some(1003),
            },
        );
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": -2,
                    "netPrice": 5000.0
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

        handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
            .expect("first wait transition should emit debug context");
        handle_execution_account_sync(&mut session, &broker_tx, &event_tx)
            .expect("repeated wait transition should not duplicate debug context");

        let wait_logs = std::iter::from_fn(|| event_rx.try_recv().ok())
            .filter_map(|event| match event {
                ServiceEvent::DebugLog(message)
                    if message.contains("execution broker sync wait") =>
                {
                    Some(message)
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(wait_logs.len(), 1);
        let wait_log = wait_logs.first().expect("expected broker sync wait log");
        assert!(wait_log.contains("tracked strategy 77 (1 active linked)"));
        assert!(wait_log.contains("broker strategy none (0 active linked)"));
        assert!(wait_log.contains("tp 4998.00 [order 1002 clOrdId midas-tp]"));
        assert!(wait_log.contains("sl 5002.00 [order 1003 clOrdId midas-sl]"));
    }

    #[test]
    fn strategy_loop_does_not_disarm_on_transient_oversize_with_live_strategy_path() {
        let mut session = test_session();
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_ema.take_profit_ticks = 8.0;
        session.execution_runtime.armed = true;
        session.execution_runtime.last_closed_bar_ts = Some(100);
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
                    "netPos": -2,
                    "netPrice": 5000.0
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

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("active strategy path should suppress transient drift disarm");

        assert!(session.execution_runtime.armed);
        assert!(session
            .execution_runtime
            .last_summary
            .contains("Waiting for broker sync"));
    }

    #[test]
    fn live_bar_signal_timing_can_trade_on_forming_range_bar() {
        let mut session = test_session();
        let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
        session.execution_config.native_ema.fast_length = 2;
        session.execution_config.native_ema.slow_length = 4;
        session.market.history_loaded = 5;
        session.market.bars = vec![
            Bar { ts_ns: 1, open: 10.0, high: 10.5, low: 9.5, close: 10.0 },
            Bar { ts_ns: 2, open: 10.0, high: 10.5, low: 9.5, close: 10.0 },
            Bar { ts_ns: 3, open: 10.0, high: 10.5, low: 9.5, close: 10.0 },
            Bar { ts_ns: 4, open: 10.0, high: 10.5, low: 9.5, close: 10.0 },
            Bar { ts_ns: 5, open: 8.0, high: 8.5, low: 7.5, close: 8.0 },
            Bar { ts_ns: 6, open: 12.0, high: 12.5, low: 11.5, close: 12.0 },
        ];
        session.execution_runtime.armed = true;
        session.execution_runtime.last_closed_bar_ts = Some(6);

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("live timing should evaluate the forming bar");

        assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
        match broker_rx.try_recv().expect("broker command queued") {
            BrokerCommand::MarketOrder { order, .. } => {
                assert_eq!(order.target_qty, Some(1));
                assert_eq!(order.order_qty, 1);
                assert_eq!(order.order_action, "Buy");
            }
            _ => panic!("expected market order from live-bar buy signal"),
        }
    }

    #[test]
    fn strategy_loop_waits_when_flat_qty_conflicts_with_live_broker_path() {
        let mut session = test_session();
        let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_signal_timing = NativeSignalTiming::LiveBar;
        session.execution_config.native_ema.fast_length = 10;
        session.execution_config.native_ema.slow_length = 30;
        session.execution_config.native_ema.take_profit_ticks = 10.0;
        session.execution_config.native_ema.stop_loss_ticks = 10.0;
        session.market.tick_size = Some(0.25);
        session.market.contract_id = Some(3570918);
        session.market.history_loaded = 31;
        session.market.bars = (0..31)
            .map(|idx| Bar {
                ts_ns: idx + 1,
                open: 6410.0,
                high: 6411.0,
                low: 6409.0,
                close: if idx < 30 { 6410.0 } else { 6420.0 },
            })
            .collect();
        session.execution_runtime.armed = true;
        session.execution_runtime.last_closed_bar_ts = Some(30);
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.active_order_strategy = Some(TrackedOrderStrategy {
            key,
            order_strategy_id: 77,
            target_qty: 1,
        });
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

        maybe_run_execution_strategy(&mut session, &broker_tx, &event_tx)
            .expect("flat/local qty with live broker path should wait instead of opening");

        assert!(broker_rx.try_recv().is_err(), "no order should be queued");
        assert!(session
            .execution_runtime
            .last_summary
            .contains("flat position reported while an order path is still active"));
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
                last_requested_take_profit_price: Some(6662.5),
                last_requested_stop_price: Some(6659.0),
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

    #[test]
    fn replay_trailing_stop_sync_ratchets_forward_without_backsliding() {
        let mut session = test_session();
        session.replay_enabled = true;
        session.execution_runtime.armed = true;
        session.execution_config.kind = StrategyKind::Native;
        session.execution_config.native_strategy = NativeStrategyKind::EmaCross;
        session.execution_config.native_ema.stop_loss_ticks = 8.0;
        session.execution_config.native_ema.use_trailing_stop = true;
        session.execution_config.native_ema.trail_trigger_ticks = 4.0;
        session.execution_config.native_ema.trail_offset_ticks = 2.0;
        session.market.contract_id = Some(3570918);
        session.market.tick_size = Some(0.25);
        session.user_store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 100.0
                }),
            )]),
        );

        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        session.managed_protection.insert(
            key,
            ManagedProtectionOrders {
                signed_qty: 1,
                take_profit_price: None,
                stop_price: Some(98.0),
                last_requested_take_profit_price: None,
                last_requested_stop_price: Some(98.0),
                take_profit_cl_ord_id: None,
                stop_cl_ord_id: Some("replay-stop".to_string()),
                take_profit_order_id: None,
                stop_order_id: Some(1002),
            },
        );

        let (broker_tx, mut broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let first_bar = Bar {
            ts_ns: 1,
            open: 100.0,
            high: 101.5,
            low: 99.75,
            close: 101.0,
        };
        sync_execution_protection(&mut session, &broker_tx, Some(&first_bar))
            .expect("initial trailing sync should queue a stop update");

        let first_next_state = match broker_rx.try_recv().expect("expected stop modify command") {
            BrokerCommand::NativeProtection { sync, .. } => {
                assert!(sync.simulate, "replay mode should use simulated protection sync");
                match sync.operation {
                    ProtectionSyncOperation::ModifyStop { payload } => {
                        assert_eq!(payload.get("orderId").and_then(Value::as_i64), Some(1002));
                        assert_eq!(payload.get("stopPrice").and_then(Value::as_f64), Some(101.0));
                    }
                    _ => panic!("expected stop modification"),
                }
                sync.next_state.expect("planner should provide the updated protection snapshot")
            }
            _ => panic!("expected native protection command"),
        };
        assert_eq!(
            session
                .execution_runtime
                .ema_execution
                .position
                .as_ref()
                .and_then(|position| position.current_stop_price),
            Some(101.0)
        );
        session.protection_sync_in_flight = false;
        session.managed_protection.insert(key, first_next_state);

        let weaker_bar = Bar {
            ts_ns: 2,
            open: 101.0,
            high: 101.25,
            low: 100.0,
            close: 100.5,
        };
        sync_execution_protection(&mut session, &broker_tx, Some(&weaker_bar))
            .expect("weaker bar should be a no-op");
        assert!(
            broker_rx.try_recv().is_err(),
            "trailing stop should not backslide or resend an unchanged modify"
        );
        assert_eq!(
            session
                .execution_runtime
                .ema_execution
                .position
                .as_ref()
                .and_then(|position| position.current_stop_price),
            Some(101.0)
        );

        let stronger_bar = Bar {
            ts_ns: 3,
            open: 100.5,
            high: 102.0,
            low: 100.5,
            close: 101.75,
        };
        sync_execution_protection(&mut session, &broker_tx, Some(&stronger_bar))
            .expect("stronger bar should ratchet the trailing stop forward");

        let second_next_state = match broker_rx
            .try_recv()
            .expect("expected second stop modify command")
        {
            BrokerCommand::NativeProtection { sync, .. } => {
                assert!(sync.simulate);
                match sync.operation {
                    ProtectionSyncOperation::ModifyStop { payload } => {
                        assert_eq!(payload.get("orderId").and_then(Value::as_i64), Some(1002));
                        assert_eq!(payload.get("stopPrice").and_then(Value::as_f64), Some(101.5));
                    }
                    _ => panic!("expected stop modification"),
                }
                sync.next_state.expect("planner should retain the replay protection snapshot")
            }
            _ => panic!("expected native protection command"),
        };
        assert_eq!(second_next_state.stop_order_id, Some(1002));
        assert_eq!(second_next_state.stop_price, Some(101.5));
        assert_eq!(
            session
                .execution_runtime
                .ema_execution
                .position
                .as_ref()
                .and_then(|position| position.current_stop_price),
            Some(101.5)
        );
    }
}
