use super::*;

#[derive(Clone)]
pub(crate) struct LiveProtectionOrder {
    order_id: i64,
    cl_ord_id: Option<String>,
    price: Option<f64>,
}

pub(crate) fn sync_native_protection(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: &str,
) -> Result<()> {
    let desired = build_desired_native_protection(
        session,
        signed_qty,
        take_profit_price,
        stop_price,
        reason,
    )?;
    sync_native_protection_target(session, broker_tx, desired)
}

pub(crate) fn sync_native_protection_target(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    desired: DesiredNativeProtection,
) -> Result<()> {
    if session.protection_sync_in_flight {
        session.pending_protection_sync = Some(desired);
        return Ok(());
    }

    let Some(sync) = plan_native_protection_sync(session, desired)? else {
        return Ok(());
    };

    session.protection_sync_in_flight = true;
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::NativeProtection { request_tx, sync })
        .is_err()
    {
        session.protection_sync_in_flight = false;
        bail!("broker gateway is closed");
    }
    Ok(())
}

fn build_desired_native_protection(
    session: &SessionState,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: &str,
) -> Result<DesiredNativeProtection> {
    let order_ctx = resolve_order_context(session)?;
    Ok(DesiredNativeProtection {
        key: StrategyProtectionKey {
            account_id: order_ctx.account.id,
            contract_id: order_ctx.contract.id,
        },
        account_name: order_ctx.account.name.clone(),
        contract_name: order_ctx.contract.name.clone(),
        signed_qty,
        take_profit_price: sanitize_price(take_profit_price),
        stop_price: sanitize_price(stop_price),
        reason: reason.to_string(),
    })
}

pub(super) fn plan_native_protection_sync(
    session: &mut SessionState,
    desired: DesiredNativeProtection,
) -> Result<Option<PendingProtectionSync>> {
    let DesiredNativeProtection {
        key,
        account_name,
        contract_name,
        signed_qty,
        take_profit_price,
        stop_price,
        reason,
    } = desired;

    if signed_qty == 0 || (take_profit_price.is_none() && stop_price.is_none()) {
        let detached = detach_strategy_protection_by_key(session, key);
        if detached.cancel_order_ids.is_empty() {
            return Ok(None);
        }
        return Ok(Some(PendingProtectionSync {
            simulate: session.replay_enabled,
            key,
            account_name: account_name.clone(),
            contract_name: contract_name.clone(),
            operation: ProtectionSyncOperation::Clear {
                cancel_order_ids: detached.cancel_order_ids,
            },
            message: Some(format!(
                "Native protection cleared for {} on {} ({reason})",
                contract_name, account_name
            )),
            next_state: None,
        }));
    }

    let exit_action = if signed_qty > 0 { "Sell" } else { "Buy" };
    let order_qty = signed_qty.abs().max(1);

    refresh_managed_protection_order_ids(session, key);
    if let Some(existing) = session.managed_protection.get(&key).cloned() {
        let missing_live_leg = (take_profit_price.is_some()
            && existing.take_profit_order_id.is_none())
            || (stop_price.is_some() && existing.stop_order_id.is_none());
        let same_position = existing.signed_qty == signed_qty;
        let same_take_profit = prices_match(existing.take_profit_price, take_profit_price);
        let same_stop = prices_match(existing.stop_price, stop_price);
        let same_requested_take_profit =
            prices_match(existing.last_requested_take_profit_price, take_profit_price);
        let same_requested_stop = prices_match(existing.last_requested_stop_price, stop_price);
        if same_position
            && same_take_profit
            && same_stop
            && same_requested_take_profit
            && same_requested_stop
        {
            return Ok(None);
        }

        if same_position && same_take_profit && stop_price.is_some() && !missing_live_leg {
            let Some(stop_order_id) = existing.stop_order_id else {
                // Do not cancel and recreate a healthy bracket just because the live stop ID has
                // not been observed yet. That path can exceed broker working-order limits on
                // constrained accounts and leave the position unprotected.
                return Ok(None);
            };
            let next_stop_price = stop_price.expect("checked is_some");
            let mut next_state = existing;
            next_state.stop_price = Some(next_stop_price);
            next_state.last_requested_take_profit_price = take_profit_price;
            next_state.last_requested_stop_price = Some(next_stop_price);
            return Ok(Some(PendingProtectionSync {
                simulate: session.replay_enabled,
                key,
                account_name: account_name.clone(),
                contract_name: contract_name.clone(),
                operation: ProtectionSyncOperation::ModifyStop {
                    payload: build_modify_native_stop_order_payload(
                        &session.cfg.time_in_force,
                        stop_order_id,
                        order_qty,
                        next_stop_price,
                    ),
                },
                message: Some(format!(
                    "Native stop updated to {:.2} on {} ({reason})",
                    next_stop_price, contract_name
                )),
                next_state: Some(next_state),
            }));
        }
    }

    let detached = detach_strategy_protection_by_key(session, key);

    let tp_cl_ord_id = take_profit_price.map(|_| next_strategy_cl_ord_id(session, "tp"));
    let stop_cl_ord_id = stop_price.map(|_| next_strategy_cl_ord_id(session, "sl"));

    let (request, action_label) = match (take_profit_price, stop_price) {
        (Some(tp), Some(stop)) => (
            ProtectionPlaceRequest::Oco {
                payload: build_native_oco_order_payload(
                    session,
                    &account_name,
                    key.account_id,
                    &contract_name,
                    exit_action,
                    order_qty,
                    tp,
                    tp_cl_ord_id.as_deref(),
                    stop,
                    stop_cl_ord_id.as_deref(),
                ),
            },
            "TP/SL",
        ),
        (Some(tp), None) => (
            ProtectionPlaceRequest::TakeProfit {
                payload: build_native_limit_order_payload(
                    session,
                    &account_name,
                    key.account_id,
                    &contract_name,
                    exit_action,
                    order_qty,
                    tp,
                    tp_cl_ord_id.as_deref(),
                ),
            },
            "TP",
        ),
        (None, Some(stop)) => (
            ProtectionPlaceRequest::StopLoss {
                payload: build_native_stop_order_payload(
                    session,
                    &account_name,
                    key.account_id,
                    &contract_name,
                    exit_action,
                    order_qty,
                    stop,
                    stop_cl_ord_id.as_deref(),
                ),
            },
            "SL",
        ),
        (None, None) => unreachable!("checked above"),
    };

    let next_state = ManagedProtectionOrders {
        signed_qty,
        take_profit_price,
        stop_price,
        last_requested_take_profit_price: take_profit_price,
        last_requested_stop_price: stop_price,
        take_profit_cl_ord_id: tp_cl_ord_id,
        stop_cl_ord_id,
        take_profit_order_id: None,
        stop_order_id: None,
    };

    Ok(Some(PendingProtectionSync {
        simulate: session.replay_enabled,
        key,
        account_name: account_name.clone(),
        contract_name: contract_name.clone(),
        operation: ProtectionSyncOperation::Replace {
            cancel_order_ids: detached.cancel_order_ids,
            request,
        },
        message: Some(format!(
            "Native {action_label} protection live for {} on {}: {} ({reason})",
            contract_name,
            account_name,
            format_protection_prices(take_profit_price, stop_price)
        )),
        next_state: Some(next_state),
    }))
}

pub(crate) fn selected_strategy_key(session: &SessionState) -> Result<StrategyProtectionKey> {
    let order_ctx = resolve_order_context(session)?;
    Ok(StrategyProtectionKey {
        account_id: order_ctx.account.id,
        contract_id: order_ctx.contract.id,
    })
}

const ORDER_STRATEGY_INTERRUPT_GRACE_MS: u128 = 1_500;

pub(super) fn strategy_has_live_linked_orders(
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

fn strategy_within_interrupt_grace(session: &SessionState, order_strategy_id: i64) -> bool {
    session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.order_strategy_id == Some(order_strategy_id)
                && tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_INTERRUPT_GRACE_MS
        })
}

pub(super) fn selected_active_order_strategy_id(session: &SessionState) -> Option<i64> {
    let key = selected_strategy_key(session).ok()?;
    if let Some(order_strategy_id) = session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(extract_entity_id)
    {
        if strategy_has_live_linked_orders(session, key, order_strategy_id)
            || strategy_within_interrupt_grace(session, order_strategy_id)
        {
            return Some(order_strategy_id);
        }
    }
    session
        .active_order_strategy
        .as_ref()
        .filter(|tracked| tracked.key == key)
        .filter(|tracked| {
            strategy_has_live_linked_orders(session, key, tracked.order_strategy_id)
                || strategy_within_interrupt_grace(session, tracked.order_strategy_id)
        })
        .map(|tracked| tracked.order_strategy_id)
}

pub(super) fn detach_strategy_protection_for_selected(
    session: &mut SessionState,
) -> Result<DetachedStrategyProtection> {
    let order_ctx = resolve_order_context(session)?;
    let key = StrategyProtectionKey {
        account_id: order_ctx.account.id,
        contract_id: order_ctx.contract.id,
    };
    Ok(detach_strategy_protection_by_key(session, key))
}

fn detach_strategy_protection_by_key(
    session: &mut SessionState,
    key: StrategyProtectionKey,
) -> DetachedStrategyProtection {
    refresh_managed_protection_order_ids(session, key);
    let existing = session.managed_protection.remove(&key);

    let session = &*session;

    let linked_ids: Vec<i64> = if selected_strategy_key(session).ok() == Some(key) {
        selected_active_order_strategy_id(session)
            .into_iter()
            .flat_map(|order_strategy_id| {
                session
                    .user_store
                    .linked_strategy_orders(key.account_id, order_strategy_id)
                    .into_iter()
            })
            .filter(|order| {
                order_is_active(order) && order_contract_id(order) == Some(key.contract_id)
            })
            .filter_map(extract_entity_id)
            .collect()
    } else {
        Vec::new()
    };

    // Collect ALL active strategy orders for this contract from the user store.
    // This catches orphaned orders whose IDs we never captured from the OCO
    // placement response (the stop leg ID often arrives later via WebSocket).
    let orphan_ids: Vec<i64> = session
        .user_store
        .orders
        .get(&key.account_id)
        .into_iter()
        .flat_map(|orders| orders.iter())
        .filter(|(_, order)| {
            order_is_active(order)
                && order_contract_id(order) == Some(key.contract_id)
                && order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id.starts_with("midas-"))
        })
        .filter_map(|(id, _)| Some(*id))
        .collect();

    let mut cancel_order_ids = linked_ids;
    for order_id in orphan_ids {
        if !cancel_order_ids.contains(&order_id) {
            cancel_order_ids.push(order_id);
        }
    }
    if let Some(state) = existing.as_ref() {
        for order_id in [state.stop_order_id, state.take_profit_order_id]
            .into_iter()
            .flatten()
        {
            if !cancel_order_ids.contains(&order_id) {
                cancel_order_ids.push(order_id);
            }
        }
    }

    DetachedStrategyProtection { cancel_order_ids }
}

pub(crate) fn collect_live_protection_orders(
    session: &SessionState,
    key: StrategyProtectionKey,
) -> (Vec<LiveProtectionOrder>, Vec<LiveProtectionOrder>) {
    let mut take_profit_orders: Vec<LiveProtectionOrder> = Vec::new();
    let mut stop_orders: Vec<LiveProtectionOrder> = Vec::new();

    let mut classify = |order: &Value| {
        if !order_is_active(order) || order_contract_id(order) != Some(key.contract_id) {
            return;
        }
        let Some(order_id) = extract_entity_id(order) else {
            return;
        };
        let candidate = LiveProtectionOrder {
            order_id,
            cl_ord_id: order
                .get("clOrdId")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            price: match order_type(order).as_deref().unwrap_or_default() {
                "limit" | "mit" => pick_number(order, &["price"]),
                "stop" | "stoplimit" | "trailingstop" | "trailingstoplimit" => {
                    pick_number(order, &["stopPrice", "price"])
                }
                _ => None,
            },
        };
        match order_type(order).as_deref().unwrap_or_default() {
            "limit" | "mit" => {
                if !take_profit_orders
                    .iter()
                    .any(|existing| existing.order_id == order_id)
                {
                    take_profit_orders.push(candidate);
                }
            }
            "stop" | "stoplimit" | "trailingstop" | "trailingstoplimit" => {
                if !stop_orders
                    .iter()
                    .any(|existing| existing.order_id == order_id)
                {
                    stop_orders.push(candidate);
                }
            }
            _ => {}
        }
    };

    if selected_strategy_key(session).ok() == Some(key) {
        if let Some(order_strategy_id) = selected_active_order_strategy_id(session) {
            for order in session
                .user_store
                .linked_strategy_orders(key.account_id, order_strategy_id)
            {
                classify(order);
            }
        }
    }

    if let Some(orders) = session.user_store.orders.get(&key.account_id) {
        for order in orders.values() {
            if !order
                .get("clOrdId")
                .and_then(Value::as_str)
                .is_some_and(|cl_ord_id| cl_ord_id.starts_with("midas-"))
            {
                continue;
            }
            classify(order);
        }
    }

    (take_profit_orders, stop_orders)
}

pub(crate) fn recover_live_protection_order(
    expected_cl_ord_id: Option<&str>,
    candidates: &[LiveProtectionOrder],
) -> Option<LiveProtectionOrder> {
    if let Some(cl_ord_id) = expected_cl_ord_id {
        return candidates
            .iter()
            .find(|candidate| candidate.cl_ord_id.as_deref() == Some(cl_ord_id))
            .cloned();
    }
    (candidates.len() == 1).then(|| candidates[0].clone())
}

pub(crate) fn refresh_managed_protection_order_ids(
    session: &mut SessionState,
    key: StrategyProtectionKey,
) {
    let (take_profit_candidates, stop_candidates) = collect_live_protection_orders(session, key);
    let Some(state) = session.managed_protection.get_mut(&key) else {
        return;
    };
    if state.take_profit_order_id.is_none() {
        if let Some(cl_ord_id) = state.take_profit_cl_ord_id.as_deref() {
            state.take_profit_order_id = session
                .user_store
                .order_id_by_client_id(key.account_id, cl_ord_id);
        }
    }
    if state.take_profit_order_id.is_none() {
        if let Some(candidate) = recover_live_protection_order(
            state.take_profit_cl_ord_id.as_deref(),
            &take_profit_candidates,
        ) {
            state.take_profit_order_id = Some(candidate.order_id);
            if state.take_profit_cl_ord_id.is_none() {
                state.take_profit_cl_ord_id = candidate.cl_ord_id;
            }
            if state.take_profit_price.is_none() {
                state.take_profit_price = candidate.price;
            }
        }
    }
    if state.stop_order_id.is_none() {
        if let Some(cl_ord_id) = state.stop_cl_ord_id.as_deref() {
            state.stop_order_id = session
                .user_store
                .order_id_by_client_id(key.account_id, cl_ord_id);
        }
    }
    if state.stop_order_id.is_none() {
        if let Some(candidate) =
            recover_live_protection_order(state.stop_cl_ord_id.as_deref(), &stop_candidates)
        {
            state.stop_order_id = Some(candidate.order_id);
            if state.stop_cl_ord_id.is_none() {
                state.stop_cl_ord_id = candidate.cl_ord_id;
            }
            if state.stop_price.is_none() {
                state.stop_price = candidate.price;
            }
        }
    }
}

fn build_native_limit_order_payload(
    session: &SessionState,
    account_name: &str,
    account_id: i64,
    contract_name: &str,
    action: &str,
    order_qty: i32,
    price: f64,
    cl_ord_id: Option<&str>,
) -> Value {
    with_cl_ord_id(
        json!({
            "accountSpec": account_name,
            "accountId": account_id,
            "action": action,
            "symbol": contract_name,
            "orderQty": order_qty,
            "orderType": "Limit",
            "price": price,
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": true,
        }),
        cl_ord_id,
    )
}

fn build_native_stop_order_payload(
    session: &SessionState,
    account_name: &str,
    account_id: i64,
    contract_name: &str,
    action: &str,
    order_qty: i32,
    stop_price: f64,
    cl_ord_id: Option<&str>,
) -> Value {
    with_cl_ord_id(
        json!({
            "accountSpec": account_name,
            "accountId": account_id,
            "action": action,
            "symbol": contract_name,
            "orderQty": order_qty,
            "orderType": "Stop",
            "stopPrice": stop_price,
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": true,
        }),
        cl_ord_id,
    )
}

fn build_native_oco_order_payload(
    session: &SessionState,
    account_name: &str,
    account_id: i64,
    contract_name: &str,
    action: &str,
    order_qty: i32,
    take_profit_price: f64,
    take_profit_cl_ord_id: Option<&str>,
    stop_price: f64,
    stop_cl_ord_id: Option<&str>,
) -> Value {
    let mut payload = with_cl_ord_id(
        json!({
            "accountSpec": account_name,
            "accountId": account_id,
            "action": action,
            "symbol": contract_name,
            "orderQty": order_qty,
            "orderType": "Limit",
            "price": take_profit_price,
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": true,
            "other": {
                "accountSpec": account_name,
                "accountId": account_id,
                "action": action,
                "symbol": contract_name,
                "orderQty": order_qty,
                "orderType": "Stop",
                "stopPrice": stop_price,
                "timeInForce": session.cfg.time_in_force,
                "isAutomated": true,
            }
        }),
        take_profit_cl_ord_id,
    );
    if let Some(other) = payload.get_mut("other").and_then(Value::as_object_mut) {
        if let Some(cl_ord_id) = stop_cl_ord_id {
            other.insert("clOrdId".to_string(), Value::String(cl_ord_id.to_string()));
        }
    }
    payload
}

fn build_modify_native_stop_order_payload(
    time_in_force: &str,
    order_id: i64,
    order_qty: i32,
    stop_price: f64,
) -> Value {
    json!({
        "orderId": order_id,
        "orderQty": order_qty,
        "orderType": "Stop",
        "stopPrice": stop_price,
        "timeInForce": time_in_force,
        "isAutomated": true,
    })
}

fn format_protection_prices(take_profit_price: Option<f64>, stop_price: Option<f64>) -> String {
    let mut parts = Vec::new();
    if let Some(price) = take_profit_price {
        parts.push(format!("tp {:.2}", price));
    }
    if let Some(price) = stop_price {
        parts.push(format!("sl {:.2}", price));
    }
    parts.join(", ")
}
