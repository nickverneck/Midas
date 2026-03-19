enum MarketOrderDispatchOutcome {
    NoOp { message: String },
    Queued { target_qty: Option<i32> },
}

fn dispatch_manual_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    action: ManualOrderAction,
) -> Result<MarketOrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    ensure_no_market_order_submit_in_flight(session)?;

    let (order_action, order_qty, action_label, automated, reason_suffix) = match action {
        ManualOrderAction::Buy => ("Buy", session.cfg.order_qty, "Buy", false, None),
        ManualOrderAction::Sell => ("Sell", session.cfg.order_qty, "Sell", false, None),
        ManualOrderAction::Close => {
            let Some(net_qty) = session
                .user_store
                .contract_position_qty(account.id, &contract)
            else {
                return Ok(MarketOrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract.name, account.name
                    ),
                });
            };
            let close_qty = net_qty.abs().round() as i32;
            if close_qty <= 0 {
                return Ok(MarketOrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract.name, account.name
                    ),
                });
            }
            let close_action = if net_qty > 0.0 { "Sell" } else { "Buy" };
            (close_action, close_qty, "Close", false, None)
        }
    };

    let interrupt_order_strategy_id = selected_active_order_strategy_id(session);
    let detached = if interrupt_order_strategy_id.is_some() {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
    } else {
        detach_strategy_protection_for_selected(session)?
    };

    let order = build_market_order_request(
        session,
        &account,
        &contract,
        order_action,
        order_qty,
        action_label,
        automated,
        reason_suffix,
        None,
        interrupt_order_strategy_id,
        detached.cancel_order_ids,
    );
    enqueue_market_order(session, broker_tx, order)?;
    Ok(MarketOrderDispatchOutcome::Queued { target_qty: None })
}

fn dispatch_native_order_strategy_target(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    target_qty: i32,
    reason: &str,
) -> Result<MarketOrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    ensure_no_market_order_submit_in_flight(session)?;
    let current_qty = session
        .user_store
        .contract_position_qty(account.id, &contract)
        .unwrap_or(0.0)
        .round() as i32;
    if current_qty.abs() > session.execution_config.order_qty.max(1) {
        bail!(
            "automated position drift detected ({current_qty}); refusing new strategy transition"
        );
    }

    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Strategy target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract.name, account.name
            ),
        });
    }

    let interrupt_order_strategy_id = selected_active_order_strategy_id(session);
    if current_qty != 0 && interrupt_order_strategy_id.is_none() {
        bail!("active order strategy id is unknown; refusing automated reversal/flatten");
    }

    if target_qty == 0 {
        let order_action = if delta > 0 { "Buy" } else { "Sell" };
        let order_qty = delta.unsigned_abs() as i32;
        let order = build_market_order_request(
            session,
            &account,
            &contract,
            order_action,
            order_qty,
            "Strategy",
            true,
            Some(reason),
            Some(target_qty),
            interrupt_order_strategy_id,
            Vec::new(),
        );
        enqueue_market_order(session, broker_tx, order)?;
        return Ok(MarketOrderDispatchOutcome::Queued {
            target_qty: Some(target_qty),
        });
    }

    let order_action = if delta > 0 { "Buy" } else { "Sell" };
    let entry_order_qty = delta.unsigned_abs() as i32;
    let strategy = build_order_strategy_request(
        session,
        &account,
        &contract,
        order_action,
        entry_order_qty,
        target_qty,
        Some(reason),
        interrupt_order_strategy_id,
    )?;
    enqueue_order_strategy(session, broker_tx, strategy)?;
    Ok(MarketOrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

fn dispatch_target_position_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    target_qty: i32,
    automated: bool,
    reason: &str,
) -> Result<MarketOrderDispatchOutcome> {
    if automated && native_order_strategy_enabled(session) {
        return dispatch_native_order_strategy_target(session, broker_tx, target_qty, reason);
    }

    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    ensure_no_market_order_submit_in_flight(session)?;
    let current_qty = session
        .user_store
        .contract_position_qty(account.id, &contract)
        .unwrap_or(0.0)
        .round() as i32;
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Strategy target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract.name, account.name
            ),
        });
    }

    let interrupt_order_strategy_id = selected_active_order_strategy_id(session);
    let detached = if interrupt_order_strategy_id.is_some() {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
    } else {
        detach_strategy_protection_for_selected(session)?
    };

    let order_action = if delta > 0 { "Buy" } else { "Sell" };
    let order_qty = delta.unsigned_abs() as i32;
    let action_label = "Strategy";
    let reason_suffix = Some(format!(
        "target {} -> {} ({reason})",
        current_qty, target_qty
    ));

    let order = build_market_order_request(
        session,
        &account,
        &contract,
        order_action,
        order_qty,
        action_label,
        automated,
        reason_suffix.as_deref(),
        Some(target_qty),
        interrupt_order_strategy_id,
        detached.cancel_order_ids,
    );
    enqueue_market_order(session, broker_tx, order)?;
    Ok(MarketOrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

struct OrderContext<'a> {
    account: &'a AccountInfo,
    contract: &'a ContractSuggestion,
}

fn resolve_order_context<'a>(session: &'a SessionState) -> Result<OrderContext<'a>> {
    let account_id = session
        .selected_account_id
        .context("select an account before sending orders")?;
    let account = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .as_ref()
        .context("select a contract before sending orders")?;
    Ok(OrderContext { account, contract })
}

fn sync_native_protection(
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

fn sync_native_protection_target(
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

fn plan_native_protection_sync(
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
        let same_position = existing.signed_qty == signed_qty;
        let same_take_profit = prices_match(existing.take_profit_price, take_profit_price);
        let same_stop = prices_match(existing.stop_price, stop_price);
        if same_position && same_take_profit && same_stop {
            return Ok(None);
        }

        if same_position
            && same_take_profit
            && stop_price.is_some()
            && existing.stop_order_id.is_some()
            && existing.take_profit_price.is_some() == take_profit_price.is_some()
        {
            let stop_order_id = existing.stop_order_id.expect("checked is_some");
            let next_stop_price = stop_price.expect("checked is_some");
            let mut next_state = existing;
            next_state.stop_price = Some(next_stop_price);
            return Ok(Some(PendingProtectionSync {
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
        take_profit_cl_ord_id: tp_cl_ord_id,
        stop_cl_ord_id,
        take_profit_order_id: None,
        stop_order_id: None,
    };

    Ok(Some(PendingProtectionSync {
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

fn ensure_no_market_order_submit_in_flight(session: &SessionState) -> Result<()> {
    if session.order_submit_in_flight {
        bail!("order submission already in flight");
    }
    Ok(())
}

fn selected_strategy_key(session: &SessionState) -> Result<StrategyProtectionKey> {
    let order_ctx = resolve_order_context(session)?;
    Ok(StrategyProtectionKey {
        account_id: order_ctx.account.id,
        contract_id: order_ctx.contract.id,
    })
}

fn selected_active_order_strategy_id(session: &SessionState) -> Option<i64> {
    let key = selected_strategy_key(session).ok()?;
    if let Some(tracked) = session.active_order_strategy.as_ref() {
        if tracked.key == key {
            return Some(tracked.order_strategy_id);
        }
    }
    session
        .user_store
        .find_active_order_strategy(key.account_id, key.contract_id)
        .and_then(extract_entity_id)
}

fn current_native_fixed_take_profit_ticks(session: &SessionState) -> f64 {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.take_profit_ticks,
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.take_profit_ticks,
    }
}

fn current_native_fixed_stop_ticks(session: &SessionState) -> f64 {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.stop_loss_ticks,
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.stop_loss_ticks,
    }
}

fn native_order_strategy_enabled(session: &SessionState) -> bool {
    if session.execution_config.kind != StrategyKind::Native {
        return false;
    }
    current_native_fixed_take_profit_ticks(session) > 0.0 || current_native_fixed_stop_ticks(session) > 0.0
}

fn build_order_strategy_request(
    session: &mut SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    order_action: &str,
    entry_order_qty: i32,
    target_qty: i32,
    reason_suffix: Option<&str>,
    interrupt_order_strategy_id: Option<i64>,
) -> Result<PendingOrderStrategyTransition> {
    let take_profit_ticks = current_native_fixed_take_profit_ticks(session);
    let stop_loss_ticks = current_native_fixed_stop_ticks(session);
    if take_profit_ticks <= 0.0 && stop_loss_ticks <= 0.0 {
        bail!("order-strategy entry requires a fixed take-profit or stop-loss");
    }
    let bracket_qty = target_qty.abs().max(1);
    let mut bracket = json!({
        "qty": bracket_qty,
        "trailingStop": false,
    });
    if take_profit_ticks > 0.0 {
        let signed_target = if order_action.eq_ignore_ascii_case("Buy") {
            take_profit_ticks
        } else {
            -take_profit_ticks
        };
        bracket["profitTarget"] = json!(signed_target);
    }
    if stop_loss_ticks > 0.0 {
        bracket["stopLoss"] = json!(stop_loss_ticks);
    }

    let params = json!({
        "entryVersion": {
            "orderQty": entry_order_qty,
            "orderType": "Market",
        },
        "brackets": [bracket],
    });
    let uuid = next_strategy_cl_ord_id(session, "strategy");
    let key = StrategyProtectionKey {
        account_id: account.id,
        contract_id: contract.id,
    };
    let mut payload = json!({
        "accountSpec": account.name,
        "accountId": account.id,
        "symbol": contract.name,
        "action": order_action,
        "orderStrategyTypeId": 2,
        "params": params.to_string(),
        "uuid": uuid,
    });
    if let Some(custom_tag50) = empty_as_none(&session.cfg.custom_tag50) {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert(
                "customTag50".to_string(),
                Value::String(custom_tag50.to_string()),
            );
        }
    }

    Ok(PendingOrderStrategyTransition {
        uuid,
        payload,
        interrupt_order_strategy_id,
        order_action: order_action.to_string(),
        entry_order_qty,
        target_qty,
        contract_name: contract.name.clone(),
        account_name: account.name.clone(),
        reason_suffix: reason_suffix.map(ToString::to_string),
        key,
    })
}

fn build_market_order_request(
    session: &mut SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    order_action: &str,
    order_qty: i32,
    action_label: &str,
    automated: bool,
    reason_suffix: Option<&str>,
    target_qty: Option<i32>,
    interrupt_order_strategy_id: Option<i64>,
    cancel_order_ids: Vec<i64>,
) -> PendingMarketOrder {
    let cl_ord_id = next_strategy_cl_ord_id(session, "entry");
    let payload = with_cl_ord_id(
        json!({
            "accountSpec": account.name,
            "accountId": account.id,
            "action": order_action,
            "symbol": contract.name,
            "orderQty": order_qty,
            "orderType": "Market",
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": automated
        }),
        Some(cl_ord_id.as_str()),
    );

    PendingMarketOrder {
        cl_ord_id,
        payload,
        interrupt_order_strategy_id,
        cancel_order_ids,
        action_label: action_label.to_string(),
        order_action: order_action.to_string(),
        order_qty,
        contract_name: contract.name.clone(),
        account_name: account.name.clone(),
        reason_suffix: reason_suffix.map(ToString::to_string),
        target_qty,
    }
}

fn enqueue_market_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    order: PendingMarketOrder,
) -> Result<()> {
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        cl_ord_id: order.cl_ord_id.clone(),
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::MarketOrder { request_tx, order })
        .is_err()
    {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}

fn enqueue_order_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    strategy: PendingOrderStrategyTransition,
) -> Result<()> {
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        cl_ord_id: strategy.uuid.clone(),
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::OrderStrategy {
            request_tx,
            strategy,
        })
        .is_err()
    {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}

fn detach_strategy_protection_for_selected(
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

    let mut cancel_order_ids = orphan_ids.clone();
    if let Some(state) = existing.as_ref() {
        cancel_order_ids.extend(
            [state.stop_order_id, state.take_profit_order_id]
                .into_iter()
                .flatten()
                .filter(|id| !orphan_ids.contains(id)),
        );
    }

    DetachedStrategyProtection { cancel_order_ids }
}

fn refresh_managed_protection_order_ids(session: &mut SessionState, key: StrategyProtectionKey) {
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
    if state.stop_order_id.is_none() {
        if let Some(cl_ord_id) = state.stop_cl_ord_id.as_deref() {
            state.stop_order_id = session
                .user_store
                .order_id_by_client_id(key.account_id, cl_ord_id);
        }
    }
}

fn next_strategy_cl_ord_id(session: &mut SessionState, suffix: &str) -> String {
    let nonce = session.next_strategy_order_nonce;
    session.next_strategy_order_nonce = session.next_strategy_order_nonce.saturating_add(1);
    let ts = Utc::now().timestamp_millis();
    format!("midas-{ts}-{nonce}-{suffix}")
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

async fn cancel_orders_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_ids: &[i64],
) -> Result<bool> {
    let mut cancelled = false;
    for order_id in order_ids {
        if cancel_order_by_id(request_tx, *order_id).await? {
            cancelled = true;
        }
    }
    Ok(cancelled)
}

async fn interrupt_order_strategy_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_strategy_id: i64,
) -> Result<()> {
    let payload = json!({
        "orderStrategyId": order_strategy_id,
    });
    let _ = request_order_json(request_tx, "orderStrategy/interruptorderstrategy", &payload).await?;
    Ok(())
}

async fn cancel_order_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_id: i64,
) -> Result<bool> {
    let payload = json!({
        "orderId": order_id,
        "isAutomated": true,
    });
    match request_order_json(request_tx, "order/cancelorder", &payload).await {
        Ok(_) => Ok(true),
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("TooLate") || msg.contains("Already cancelled") {
                Ok(false)
            } else {
                Err(err)
            }
        }
    }
}

async fn request_order_json(
    request_tx: &UnboundedSender<UserSocketCommand>,
    endpoint: &str,
    payload: &Value,
) -> Result<Value> {
    let (response_tx, response_rx) = oneshot::channel();
    request_tx
        .send(UserSocketCommand {
            endpoint: endpoint.to_string(),
            query: None,
            body: Some(payload.clone()),
            response_tx,
        })
        .map_err(|_| anyhow::anyhow!("user websocket request channel is closed"))?;

    let parsed = response_rx
        .await
        .map_err(|_| anyhow::anyhow!("user websocket response channel was dropped"))?
        .map_err(anyhow::Error::msg)?;
    if let Some(failure) = parsed.get("failureReason").and_then(Value::as_str) {
        if !failure.trim().is_empty() {
            bail!("{endpoint} rejected: {failure}");
        }
    }
    if let Some(err_text) = parsed.get("errorText").and_then(Value::as_str) {
        if !err_text.trim().is_empty() {
            bail!("{endpoint} errorText: {err_text}");
        }
    }
    Ok(parsed)
}
