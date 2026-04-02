use super::*;

fn current_native_fixed_take_profit_ticks(session: &SessionState) -> f64 {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.take_profit_ticks,
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.take_profit_ticks,
    }
}

fn current_native_fixed_take_profit_offset(session: &SessionState) -> Option<f64> {
    price_offset_from_ticks(
        current_native_fixed_take_profit_ticks(session),
        session.market.tick_size,
    )
}

fn current_native_fixed_stop_ticks(session: &SessionState) -> f64 {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.stop_loss_ticks,
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.stop_loss_ticks,
    }
}

fn active_native_uses_trailing_stop(session: &SessionState) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.use_trailing_stop,
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.use_trailing_stop,
    }
}

fn current_native_fixed_stop_offset(session: &SessionState) -> Option<f64> {
    price_offset_from_ticks(
        current_native_fixed_stop_ticks(session),
        session.market.tick_size,
    )
}

pub(super) fn price_offset_from_ticks(ticks: f64, tick_size: Option<f64>) -> Option<f64> {
    let tick_size = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0)?;
    if ticks <= 0.0 {
        return None;
    }
    Some(ticks * tick_size)
}

pub(super) fn signed_profit_target_offset(order_action: &str, offset: f64) -> f64 {
    if order_action.eq_ignore_ascii_case("Buy") {
        offset
    } else {
        -offset
    }
}

pub(super) fn signed_stop_loss_offset(order_action: &str, offset: f64) -> f64 {
    if order_action.eq_ignore_ascii_case("Buy") {
        -offset
    } else {
        offset
    }
}

pub(crate) fn native_order_strategy_enabled(session: &SessionState) -> bool {
    if session.execution_config.kind != StrategyKind::Native {
        return false;
    }
    if active_native_uses_trailing_stop(session) {
        return false;
    }
    current_native_fixed_take_profit_ticks(session) > 0.0
        || current_native_fixed_stop_ticks(session) > 0.0
}

pub(super) fn build_order_strategy_request(
    session: &mut SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    order_action: &str,
    entry_order_qty: i32,
    target_qty: i32,
    reason_suffix: Option<&str>,
    interrupt_order_strategy_id: Option<i64>,
    cancel_order_ids: Vec<i64>,
) -> Result<PendingOrderStrategyTransition> {
    let take_profit_ticks = current_native_fixed_take_profit_ticks(session);
    let stop_loss_ticks = current_native_fixed_stop_ticks(session);
    if take_profit_ticks <= 0.0 && stop_loss_ticks <= 0.0 {
        bail!("order-strategy entry requires a fixed take-profit or stop-loss");
    }
    let take_profit_offset = current_native_fixed_take_profit_offset(session);
    let stop_loss_offset = current_native_fixed_stop_offset(session);
    if (take_profit_ticks > 0.0 && take_profit_offset.is_none())
        || (stop_loss_ticks > 0.0 && stop_loss_offset.is_none())
    {
        bail!("order-strategy entry requires a valid market tick size for TP/SL offsets");
    }
    let bracket_qty = target_qty.abs().max(1);
    let mut bracket = json!({
        "qty": bracket_qty,
        "trailingStop": false,
    });
    if let Some(offset) = take_profit_offset {
        bracket["profitTarget"] = json!(signed_profit_target_offset(order_action, offset));
    }
    if let Some(offset) = stop_loss_offset {
        bracket["stopLoss"] = json!(signed_stop_loss_offset(order_action, offset));
    }

    let params = json!({
        "entryVersion": {
            "orderQty": entry_order_qty,
            "orderType": "Market",
            "timeInForce": session.cfg.time_in_force,
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
    let reference_price = session.market.bars.last().map(|bar| bar.close);
    let take_profit_price = reference_price.and_then(|price| {
        take_profit_offset.map(|offset| price + signed_profit_target_offset(order_action, offset))
    });
    let stop_price = reference_price.and_then(|price| {
        stop_loss_offset.map(|offset| price + signed_stop_loss_offset(order_action, offset))
    });

    Ok(PendingOrderStrategyTransition {
        simulate: session.replay_enabled,
        uuid,
        payload,
        interrupt_order_strategy_id,
        cancel_order_ids,
        order_action: order_action.to_string(),
        entry_order_qty,
        target_qty,
        contract_name: contract.name.clone(),
        account_name: account.name.clone(),
        reference_ts_ns: session.market.bars.last().map(|bar| bar.ts_ns),
        reference_price,
        take_profit_price,
        stop_price,
        reason_suffix: reason_suffix.map(ToString::to_string),
        key,
    })
}

pub(super) fn build_market_order_request(
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
    let current_qty = session
        .user_store
        .contract_position_qty(account.id, contract)
        .unwrap_or(0.0)
        .round() as i32;
    let simulated_next_qty = target_qty.unwrap_or_else(|| {
        current_qty
            + if order_action.eq_ignore_ascii_case("Buy") {
                order_qty
            } else {
                -order_qty
            }
    });
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
        simulate: session.replay_enabled,
        cl_ord_id,
        payload,
        account_id: account.id,
        contract_id: contract.id,
        interrupt_order_strategy_id,
        cancel_order_ids,
        action_label: action_label.to_string(),
        order_action: order_action.to_string(),
        order_qty,
        contract_name: contract.name.clone(),
        account_name: account.name.clone(),
        reference_ts_ns: session.market.bars.last().map(|bar| bar.ts_ns),
        reference_price: session.market.bars.last().map(|bar| bar.close),
        simulated_next_qty,
        reason_suffix: reason_suffix.map(ToString::to_string),
        target_qty,
    }
}

pub(super) fn build_liquidation_request(
    session: &mut SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    automated: bool,
    target_qty: Option<i32>,
    interrupt_order_strategy_id: Option<i64>,
    cancel_order_ids: Vec<i64>,
) -> PendingLiquidation {
    PendingLiquidation {
        simulate: session.replay_enabled,
        request_id: next_strategy_cl_ord_id(session, "liquidate"),
        payload: json!({
            "accountId": account.id,
            "contractId": contract.id,
            "admin": false,
            "isAutomated": automated,
        }),
        account_id: account.id,
        contract_id: contract.id,
        account_name: account.name.clone(),
        contract_name: contract.name.clone(),
        reference_ts_ns: session.market.bars.last().map(|bar| bar.ts_ns),
        reference_price: session.market.bars.last().map(|bar| bar.close),
        target_qty,
        interrupt_order_strategy_id,
        cancel_order_ids,
    }
}
