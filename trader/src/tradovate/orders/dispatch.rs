use super::*;

pub(crate) enum MarketOrderDispatchOutcome {
    NoOp { message: String },
    Queued { target_qty: Option<i32> },
}

pub(crate) fn dispatch_manual_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    action: ManualOrderAction,
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
            if net_qty.abs().round() as i32 <= 0 {
                return Ok(MarketOrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract.name, account.name
                    ),
                });
            }
            let liquidation = build_liquidation_request(
                session,
                &account,
                &contract,
                false,
                Some(0),
                None,
                Vec::new(),
            );
            enqueue_liquidation(session, broker_tx, liquidation)?;
            return Ok(MarketOrderDispatchOutcome::Queued {
                target_qty: Some(0),
            });
        }
    };

    let interrupt_order_strategy_id = if current_qty != 0 {
        selected_active_order_strategy_id(session)
    } else {
        None
    };
    let detached = if current_qty != 0 {
        detach_strategy_protection_for_selected(session)?
    } else {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
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

pub(super) fn dispatch_native_order_strategy_target(
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

    let strategy_key = StrategyProtectionKey {
        account_id: account.id,
        contract_id: contract.id,
    };
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Strategy target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract.name, account.name
            ),
        });
    }

    let is_reversal =
        current_qty != 0 && target_qty != 0 && current_qty.signum() != target_qty.signum();
    if is_reversal
        && session.execution_config.native_reversal_mode == NativeReversalMode::FlattenConfirmEnter
    {
        let interrupt_order_strategy_id = selected_active_order_strategy_id(session);
        if interrupt_order_strategy_id.is_none() {
            return Ok(MarketOrderDispatchOutcome::NoOp {
                message: format!(
                    "Waiting for broker order-strategy state to reconcile before staged reversal {} -> {} on {} ({reason})",
                    current_qty, target_qty, contract.name
                ),
            });
        }

        let order_action = if current_qty > 0 { "Sell" } else { "Buy" };
        let order_qty = current_qty.abs();
        let flatten_reason = format!(
            "{reason} | staged reversal flatten {} -> 0 before {}",
            current_qty, target_qty
        );
        let detached = detach_strategy_protection_for_selected(session)?;
        let order = build_market_order_request(
            session,
            &account,
            &contract,
            order_action,
            order_qty,
            "Strategy",
            true,
            Some(&flatten_reason),
            Some(0),
            interrupt_order_strategy_id,
            detached.cancel_order_ids,
        );
        enqueue_market_order(session, broker_tx, order)?;
        session.execution_runtime.pending_reversal_entry = Some(PendingNativeReversalEntry {
            target_qty,
            reason: reason.to_string(),
        });
        return Ok(MarketOrderDispatchOutcome::Queued {
            target_qty: Some(0),
        });
    }

    if is_reversal
        && session.execution_config.native_reversal_mode == NativeReversalMode::CloseAllEnter
    {
        let _detached = detach_strategy_protection_for_selected(session)?;
        let liquidation = build_liquidation_request(
            session,
            &account,
            &contract,
            true,
            Some(target_qty),
            None,
            Vec::new(),
        );
        let order_action = if target_qty > 0 { "Buy" } else { "Sell" };
        let strategy = build_order_strategy_request(
            session,
            &account,
            &contract,
            order_action,
            target_qty.abs(),
            target_qty,
            Some(reason),
            None,
            Vec::new(),
        )?;
        enqueue_liquidation_then_order_strategy(session, broker_tx, liquidation, strategy)?;
        return Ok(MarketOrderDispatchOutcome::Queued {
            target_qty: Some(target_qty),
        });
    }

    if is_reversal && session.execution_config.native_reversal_mode == NativeReversalMode::Direct {
        let interrupt_order_strategy_id = selected_active_order_strategy_id(session);
        let detached = detach_strategy_protection_for_selected(session)?;
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
            detached.cancel_order_ids,
        );
        enqueue_market_order(session, broker_tx, order)?;
        return Ok(MarketOrderDispatchOutcome::Queued {
            target_qty: Some(target_qty),
        });
    }

    let interrupt_order_strategy_id = if current_qty != 0 {
        selected_active_order_strategy_id(session).filter(|order_strategy_id| {
            strategy_has_live_broker_path(session, strategy_key, *order_strategy_id)
        })
    } else {
        None
    };
    if current_qty != 0 && !is_reversal && interrupt_order_strategy_id.is_none() {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Waiting for broker order-strategy state to reconcile before automated transition {} -> {} on {} ({reason})",
                current_qty, target_qty, contract.name
            ),
        });
    }
    let detached = if current_qty != 0 {
        detach_strategy_protection_for_selected(session)?
    } else {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
    };

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
            detached.cancel_order_ids,
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
        detached.cancel_order_ids,
    )?;
    enqueue_order_strategy(session, broker_tx, strategy)?;
    Ok(MarketOrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

pub(crate) fn dispatch_profile_legacy_order_strategy_target(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    target_qty: i32,
    reason: &str,
) -> Result<MarketOrderDispatchOutcome> {
    if target_qty == 0 {
        return dispatch_target_position_order(session, broker_tx, target_qty, true, reason);
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
    if current_qty.abs() > session.execution_config.order_qty.max(1) {
        bail!(
            "automated position drift detected ({current_qty}); refusing legacy strategy transition"
        );
    }

    let strategy_key = StrategyProtectionKey {
        account_id: account.id,
        contract_id: contract.id,
    };
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Strategy target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract.name, account.name
            ),
        });
    }

    let interrupt_order_strategy_id = if current_qty != 0 {
        selected_active_order_strategy_id(session).filter(|order_strategy_id| {
            strategy_has_live_broker_path(session, strategy_key, *order_strategy_id)
        })
    } else {
        None
    };
    if current_qty != 0 && interrupt_order_strategy_id.is_none() {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Waiting for broker order-strategy state to reconcile before legacy strategy transition {} -> {} on {} ({reason})",
                current_qty, target_qty, contract.name
            ),
        });
    }
    let detached = if current_qty != 0 {
        detach_strategy_protection_for_selected(session)?
    } else {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
    };

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
        detached.cancel_order_ids,
    )?;
    enqueue_order_strategy(session, broker_tx, strategy)?;
    Ok(MarketOrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

pub(crate) fn dispatch_target_position_order(
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

    let interrupt_order_strategy_id = if current_qty != 0 {
        selected_active_order_strategy_id(session)
    } else {
        None
    };
    let detached = if current_qty != 0 {
        detach_strategy_protection_for_selected(session)?
    } else {
        DetachedStrategyProtection {
            cancel_order_ids: Vec::new(),
        }
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
