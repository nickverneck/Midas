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

    // Tradovate native TP, SL, and trailing stops must be broker-owned via
    // orderStrategy/startorderstrategy. Positive protection intents are
    // intentionally ignored here so no caller can synthesize app-managed
    // placeOrder/placeOCO protection after an entry has filled.
    Ok(None)
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

    // Only app-created protection is cancelable here. Broker-owned
    // orderStrategy child orders must be left for Tradovate to manage.
    let mut cancel_order_ids: Vec<i64> = session
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
        .filter_map(|(id, order)| (!order_belongs_to_strategy(session, *id, order)).then_some(*id))
        .collect();

    if let Some(state) = existing.as_ref() {
        if state
            .stop_cl_ord_id
            .as_deref()
            .is_some_and(|id| id.starts_with("midas-"))
            && !state
                .stop_order_id
                .is_some_and(|order_id| order_id_belongs_to_strategy(session, order_id))
        {
            push_unique_order_id(&mut cancel_order_ids, state.stop_order_id);
        }
        if state
            .take_profit_cl_ord_id
            .as_deref()
            .is_some_and(|id| id.starts_with("midas-"))
            && !state
                .take_profit_order_id
                .is_some_and(|order_id| order_id_belongs_to_strategy(session, order_id))
        {
            push_unique_order_id(&mut cancel_order_ids, state.take_profit_order_id);
        }
    }

    DetachedStrategyProtection { cancel_order_ids }
}

fn order_belongs_to_strategy(session: &SessionState, order_id: i64, order: &Value) -> bool {
    json_i64(order, "orderStrategyId").is_some() || order_id_belongs_to_strategy(session, order_id)
}

fn order_id_belongs_to_strategy(session: &SessionState, order_id: i64) -> bool {
    session
        .user_store
        .order_strategy_links
        .values()
        .any(|link| json_i64(link, "orderId") == Some(order_id))
}

fn push_unique_order_id(cancel_order_ids: &mut Vec<i64>, order_id: Option<i64>) {
    let Some(order_id) = order_id else {
        return;
    };
    if !cancel_order_ids.contains(&order_id) {
        cancel_order_ids.push(order_id);
    }
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
