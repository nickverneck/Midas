use super::*;

pub(crate) fn execution_state_snapshot(session: &SessionState) -> ExecutionStateSnapshot {
    let (take_profit_price, stop_price) = selected_protection_prices(session);
    ExecutionStateSnapshot {
        config: session.execution_config.clone(),
        runtime: session.execution_runtime.snapshot(),
        selected_account_id: session.selected_account_id,
        selected_contract_name: session
            .selected_contract
            .as_ref()
            .map(|contract| contract.name.clone()),
        market_position_qty: selected_market_position_qty(session),
        market_entry_price: selected_market_entry_price(session),
        selected_contract_take_profit_price: take_profit_price,
        selected_contract_stop_price: stop_price,
    }
}

pub(crate) fn execution_probe_order_snapshot(order: &Value) -> ExecutionProbeOrder {
    ExecutionProbeOrder {
        order_id: extract_entity_id(order),
        order_strategy_id: json_i64(order, "orderStrategyId"),
        cl_ord_id: order
            .get("clOrdId")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        order_type: order
            .get("orderType")
            .and_then(Value::as_str)
            .or_else(|| order.get("ordType").and_then(Value::as_str))
            .map(ToString::to_string),
        action: order
            .get("action")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        order_qty: pick_number(order, &["orderQty", "qty", "quantity"])
            .map(|value| value.abs().round() as i32),
        price: pick_number(order, &["price"]),
        stop_price: pick_number(order, &["stopPrice"]),
        status: order
            .get("ordStatus")
            .and_then(Value::as_str)
            .or_else(|| order.get("status").and_then(Value::as_str))
            .map(ToString::to_string),
    }
}

pub(crate) fn parse_order_strategy_params_qtys(strategy: &Value) -> (Option<i32>, Vec<i32>) {
    let params = strategy
        .get("params")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok());
    let entry_qty = params
        .as_ref()
        .and_then(|params| params.get("entryVersion"))
        .and_then(|entry| pick_number(entry, &["orderQty", "qty", "quantity"]))
        .map(|value| value.abs().round() as i32);
    let bracket_qtys = params
        .as_ref()
        .and_then(|params| params.get("brackets"))
        .and_then(Value::as_array)
        .map(|brackets| {
            brackets
                .iter()
                .filter_map(|bracket| {
                    pick_number(bracket, &["qty", "orderQty", "quantity"])
                        .map(|value| value.abs().round() as i32)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    (entry_qty, bracket_qtys)
}

pub(crate) fn execution_probe_snapshot(
    session: &SessionState,
    latency: LatencySnapshot,
    tag: String,
) -> ExecutionProbeSnapshot {
    let execution_state = execution_state_snapshot(session);
    let selected_key = selected_strategy_key(session).ok();
    let account_id = session.selected_account_id;
    let tracker_order_id = session.order_latency_tracker.as_ref().and_then(|tracker| {
        tracker.order_id.or_else(|| {
            account_id.and_then(|account_id| {
                session
                    .user_store
                    .order_id_by_client_id(account_id, &tracker.cl_ord_id)
            })
        })
    });
    let tracker_order_is_active =
        tracker_order_id
            .zip(account_id)
            .is_some_and(|(order_id, account_id)| {
                session
                    .user_store
                    .find_order(account_id, order_id)
                    .is_some_and(order_is_active)
            });
    let tracker_order_strategy_id = session
        .order_latency_tracker
        .as_ref()
        .and_then(|tracker| tracker.order_strategy_id);
    let tracker_strategy_has_live_orders =
        selected_key
            .zip(tracker_order_strategy_id)
            .is_some_and(|(key, order_strategy_id)| {
                strategy_has_live_broker_path(session, key, order_strategy_id)
            });
    let tracker_within_strategy_grace =
        tracker_order_strategy_id.is_some_and(|order_strategy_id| {
            tracker_within_broker_path_grace(session, order_strategy_id)
        });
    let tracked_order_strategy_id =
        active_order_strategy_matches_selected(session).map(|tracked| tracked.order_strategy_id);
    let broker_strategy = selected_key.and_then(|key| {
        session
            .user_store
            .find_active_order_strategy(key.account_id, key.contract_id)
    });
    let broker_order_strategy_id = broker_strategy.and_then(extract_entity_id);
    let broker_order_strategy_status = broker_strategy.and_then(|strategy| {
        strategy
            .get("status")
            .and_then(Value::as_str)
            .or_else(|| strategy.get("strategyStatus").and_then(Value::as_str))
            .map(ToString::to_string)
    });
    let (broker_strategy_entry_order_qty, broker_strategy_bracket_qtys) = broker_strategy
        .map(parse_order_strategy_params_qtys)
        .unwrap_or((None, Vec::new()));

    let selected_working_orders = selected_key
        .and_then(|key| {
            session
                .user_store
                .orders
                .get(&key.account_id)
                .map(|orders| (key, orders))
        })
        .map(|(key, orders)| {
            orders
                .values()
                .filter(|order| {
                    order_is_active(order) && order_contract_id(order) == Some(key.contract_id)
                })
                .map(execution_probe_order_snapshot)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let linked_active_orders = match (selected_key, broker_order_strategy_id) {
        (Some(key), Some(order_strategy_id)) => session
            .user_store
            .linked_strategy_orders(key.account_id, order_strategy_id)
            .into_iter()
            .filter(|order| {
                order_is_active(order) && order_contract_id(order) == Some(key.contract_id)
            })
            .map(execution_probe_order_snapshot)
            .collect(),
        _ => Vec::new(),
    };

    let managed_protection = selected_key
        .and_then(|key| session.managed_protection.get(&key))
        .map(|protection| ExecutionProbeManagedProtection {
            signed_qty: protection.signed_qty,
            take_profit_price: protection.take_profit_price,
            stop_price: protection.stop_price,
            take_profit_order_id: protection.take_profit_order_id,
            stop_order_id: protection.stop_order_id,
            take_profit_cl_ord_id: protection.take_profit_cl_ord_id.clone(),
            stop_cl_ord_id: protection.stop_cl_ord_id.clone(),
        });

    ExecutionProbeSnapshot {
        tag,
        captured_at_utc: Utc::now(),
        execution_state,
        latency,
        order_submit_in_flight: session.order_submit_in_flight,
        protection_sync_in_flight: session.protection_sync_in_flight,
        tracker_order_id,
        tracker_order_is_active,
        tracker_order_strategy_id,
        tracker_strategy_has_live_orders,
        tracker_within_strategy_grace,
        tracked_order_strategy_id,
        broker_order_strategy_id,
        broker_order_strategy_status,
        broker_strategy_entry_order_qty,
        broker_strategy_bracket_qtys,
        selected_working_orders,
        linked_active_orders,
        managed_protection,
    }
}

pub(crate) fn emit_execution_state(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
) {
    let _ = event_tx.send(ServiceEvent::ExecutionState(execution_state_snapshot(
        session,
    )));
}

pub(crate) fn closed_bars(session: &SessionState) -> &[Bar] {
    let closed_len = session.market.history_loaded.min(session.market.bars.len());
    &session.market.bars[..closed_len]
}

pub(crate) fn strategy_bars(session: &SessionState) -> &[Bar] {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        &session.market.bars
    } else {
        closed_bars(session)
    }
}

pub(crate) fn latest_strategy_bar_ts(session: &SessionState) -> Option<i64> {
    strategy_bars(session).last().map(|bar| bar.ts_ns)
}

pub(crate) fn active_signal_timing_label(session: &SessionState) -> &'static str {
    match session.execution_config.native_signal_timing {
        NativeSignalTiming::ClosedBar => "closed bar",
        NativeSignalTiming::LiveBar => "live bar",
    }
}

pub(crate) fn session_window_at(
    session: &SessionState,
    ts_ns: i64,
) -> Option<InstrumentSessionWindow> {
    session
        .market
        .session_profile
        .map(|profile| profile.evaluate(ts_ns))
}

pub(crate) fn selected_contract_position<'a>(session: &'a SessionState) -> Option<&'a Value> {
    let account_id = session.selected_account_id?;
    let contract = session.selected_contract.as_ref()?;
    best_contract_position(
        session
            .user_store
            .positions
            .get(&account_id)
            .into_iter()
            .flat_map(|positions| positions.values()),
        contract,
    )
}

pub(crate) fn selected_market_position_qty(session: &SessionState) -> i32 {
    selected_contract_position(session)
        .and_then(position_qty)
        .unwrap_or_default()
        .round() as i32
}

pub(crate) fn selected_market_entry_price(session: &SessionState) -> Option<f64> {
    selected_contract_position(session).and_then(position_entry_price)
}

pub(crate) fn selected_protection_prices(session: &SessionState) -> (Option<f64>, Option<f64>) {
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

pub(crate) fn format_selected_managed_protection(session: &SessionState) -> String {
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

pub(crate) fn format_selected_tracker_state(session: &SessionState) -> String {
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

pub(crate) fn active_linked_order_count(
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

pub(crate) fn execution_observability_context(session: &SessionState) -> String {
    let pending_target = session
        .execution_runtime
        .pending_target_qty
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let tracked_strategy = active_order_strategy_matches_selected(session);
    let tracked_strategy_id = tracked_strategy
        .as_ref()
        .map(|tracked| tracked.order_strategy_id);
    let broker_strategy_id = selected_strategy_key(session).ok().and_then(|key| {
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

pub(crate) fn emit_execution_transition_debug(
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

pub(crate) fn active_native_slug(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.slug()
}

pub(crate) fn active_native_label(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.label()
}

pub(crate) fn active_native_uses_protection(session: &SessionState) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            session.execution_config.native_hma.uses_native_protection()
        }
        NativeStrategyKind::EmaCross => {
            session.execution_config.native_ema.uses_native_protection()
        }
    }
}
