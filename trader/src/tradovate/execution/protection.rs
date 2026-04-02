use super::*;

pub(crate) fn active_order_strategy_matches_selected(
    session: &SessionState,
) -> Option<TrackedOrderStrategy> {
    let key = selected_strategy_key(session).ok()?;
    session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .cloned()
}

pub(crate) fn reconcile_selected_active_order_strategy(session: &mut SessionState) {
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

pub(crate) fn hydrate_selected_order_strategy_protection(session: &mut SessionState) {
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

pub(crate) fn clear_selected_order_strategy_state(session: &mut SessionState) {
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

pub(crate) fn should_wait_for_strategy_owned_protection(session: &SessionState) -> bool {
    if !native_order_strategy_enabled(session) || selected_market_position_qty(session) == 0 {
        return false;
    }

    if session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.strategy_owned_protection
                && tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_POSITION_SYNC_GRACE_MS
        })
    {
        return true;
    }

    // Tradovate demo/live can leave orderStrategyLink/list empty even while the
    // broker-native TP/SL bracket is active. In that case, keep waiting only if
    // the selected contract still shows working protective orders. A stale
    // orderStrategy record without any live child orders should not block native
    // protection sync after a market-delta reversal.
    selected_contract_has_active_protective_orders(session)
}

pub(crate) fn selected_contract_has_active_protective_orders(session: &SessionState) -> bool {
    let Some(key) = selected_strategy_key(session).ok() else {
        return false;
    };
    session
        .user_store
        .orders
        .get(&key.account_id)
        .into_iter()
        .flat_map(|orders| orders.values())
        .any(|order| {
            order_is_active(order)
                && order_contract_id(order) == Some(key.contract_id)
                && matches!(
                    order_type(order)
                        .as_deref()
                        .map(|order_type| order_type.to_ascii_lowercase()),
                    Some(ref order_type)
                        if matches!(
                            order_type.as_str(),
                            "limit"
                                | "mit"
                                | "stop"
                                | "stoplimit"
                                | "trailingstop"
                                | "trailingstoplimit"
                        )
                )
        })
}

pub(crate) fn selected_managed_protection_waiting_for_position_sync(
    session: &mut SessionState,
) -> bool {
    if session.protection_sync_in_flight || session.pending_protection_sync.is_some() {
        return false;
    }

    let Some(key) = selected_strategy_key(session).ok() else {
        return false;
    };
    let actual_qty = selected_market_position_qty(session);
    if actual_qty == 0 {
        return false;
    }

    refresh_managed_protection_order_ids(session, key);
    let Some(protection) = session.managed_protection.get(&key).cloned() else {
        return false;
    };
    if protection.signed_qty != actual_qty {
        return false;
    }

    let (take_profit_candidates, stop_candidates) = collect_live_protection_orders(session, key);
    let take_profit_live = protection.take_profit_price.is_none()
        || recover_live_protection_order(
            protection.take_profit_cl_ord_id.as_deref(),
            &take_profit_candidates,
        )
        .is_some();
    let stop_live = protection.stop_price.is_none()
        || recover_live_protection_order(protection.stop_cl_ord_id.as_deref(), &stop_candidates)
            .is_some();

    !take_profit_live || !stop_live
}

pub(crate) fn sync_active_execution_position(
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

pub(crate) fn take_profit_price(
    session: &SessionState,
    entry_price: f64,
    signed_qty: i32,
) -> Option<f64> {
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

pub(crate) fn combined_stop_price(
    session: &mut SessionState,
    trailing_bar: Option<&Bar>,
) -> Option<f64> {
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

pub(crate) fn sync_execution_protection(
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
