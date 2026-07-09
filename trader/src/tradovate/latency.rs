fn update_latency_from_envelope(
    session: &mut SessionState,
    latency: &mut LatencySnapshot,
    envelope: &EntityEnvelope,
) -> bool {
    if envelope.deleted {
        return false;
    }
    let Some(tracker) = session.order_latency_tracker.as_mut() else {
        return false;
    };

    match envelope.entity_type.to_ascii_lowercase().as_str() {
        "orderstrategylink" => {
            if let Some(expected_strategy_id) = tracker.order_strategy_id {
                if json_i64(&envelope.entity, "orderStrategyId") == Some(expected_strategy_id) {
                    if tracker.order_id.is_none() {
                        tracker.order_id = json_i64(&envelope.entity, "orderId");
                    }
                }
            }
            false
        }
        "order" => {
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.seen_recorded {
                return false;
            }
            tracker.seen_recorded = true;
            latency.last_order_seen_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            if let Some(signal_started_at) = tracker.signal_started_at {
                latency.last_signal_seen_ms =
                    Some(signal_started_at.elapsed().as_millis() as u64);
            }
            true
        }
        "executionreport" => {
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.exec_report_recorded {
                return false;
            }
            tracker.exec_report_recorded = true;
            latency.last_exec_report_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            if let Some(signal_started_at) = tracker.signal_started_at {
                latency.last_signal_ack_ms =
                    Some(signal_started_at.elapsed().as_millis() as u64);
            }
            true
        }
        "fill" => {
            if !tracker_matches_fill_entity(tracker, &envelope.entity) || tracker.fill_recorded {
                return false;
            }
            tracker.fill_recorded = true;
            latency.last_fill_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            if let Some(signal_started_at) = tracker.signal_started_at {
                latency.last_signal_fill_ms =
                    Some(signal_started_at.elapsed().as_millis() as u64);
            }
            true
        }
        _ => false,
    }
}

fn tracker_matches_entity(tracker: &mut OrderLatencyTracker, entity: &Value) -> bool {
    let entity_order_id = json_i64(entity, "orderId").or_else(|| json_i64(entity, "id"));
    tracker_matches_entity_order_id(tracker, entity, entity_order_id)
}

fn tracker_matches_fill_entity(tracker: &mut OrderLatencyTracker, entity: &Value) -> bool {
    tracker_matches_entity_order_id(tracker, entity, json_i64(entity, "orderId"))
}

fn tracker_matches_entity_order_id(
    tracker: &mut OrderLatencyTracker,
    entity: &Value,
    entity_order_id: Option<i64>,
) -> bool {
    let entity_cl_ord_id = entity.get("clOrdId").and_then(Value::as_str);
    let entity_order_strategy_id = json_i64(entity, "orderStrategyId");

    if let Some(expected_strategy_id) = tracker.order_strategy_id {
        if entity_order_strategy_id == Some(expected_strategy_id) {
            if tracker.order_id.is_none() {
                tracker.order_id = entity_order_id;
            }
            return true;
        }
    }

    if let Some(expected_order_id) = tracker.order_id {
        if entity_order_id == Some(expected_order_id) {
            return true;
        }
    }

    if entity_cl_ord_id.is_some_and(|value| value == tracker.cl_ord_id) {
        if tracker.order_id.is_none() {
            tracker.order_id = entity_order_id;
        }
        return true;
    }

    false
}

fn fill_matches_active_latency_tracker(session: &SessionState, fill: &Value) -> bool {
    let Some(tracker) = session.order_latency_tracker.as_ref() else {
        return false;
    };
    let entity_cl_ord_id = fill.get("clOrdId").and_then(Value::as_str);
    let entity_order_id = json_i64(fill, "orderId");
    let entity_order_strategy_id = json_i64(fill, "orderStrategyId");

    if let Some(expected_strategy_id) = tracker.order_strategy_id {
        if entity_order_strategy_id == Some(expected_strategy_id) {
            return true;
        }
    }
    if let Some(expected_order_id) = tracker.order_id {
        if entity_order_id == Some(expected_order_id) {
            return true;
        }
    }
    entity_cl_ord_id.is_some_and(|value| value == tracker.cl_ord_id)
}

fn record_trade_marker(session: &mut SessionState, marker: TradeMarker) -> bool {
    if let Some(fill_id) = marker.fill_id {
        if session
            .market
            .trade_markers
            .iter()
            .any(|existing| existing.fill_id == Some(fill_id))
        {
            return false;
        }
    }

    session.market.trade_markers.push(marker);
    if session.market.trade_markers.len() > 200 {
        let overflow = session.market.trade_markers.len() - 200;
        session.market.trade_markers.drain(0..overflow);
    }
    true
}

fn fill_debug_detail(session: &SessionState, marker: &TradeMarker, fill: &Value) -> String {
    let side = match marker.side {
        TradeMarkerSide::Buy => "Buy",
        TradeMarkerSide::Sell => "Sell",
    };
    let contract = marker
        .contract_name
        .clone()
        .or_else(|| {
            marker
                .contract_id
                .map(|contract_id| format!("contract {contract_id}"))
        })
        .unwrap_or_else(|| "selected contract".to_string());
    let mut parts = vec![format!(
        "{side} {} {contract} @ {:.2}",
        marker.qty, marker.price
    )];

    if let Some(fill_id) = marker.fill_id {
        parts.push(format!("fill {fill_id}"));
    }
    if let Some(order_id) = json_i64(fill, "orderId") {
        parts.push(format!("order {order_id}"));
    }
    if let Some(order_strategy_id) = json_i64(fill, "orderStrategyId") {
        parts.push(format!("strategy {order_strategy_id}"));
    }
    if let Some(cl_ord_id) = fill.get("clOrdId").and_then(Value::as_str) {
        parts.push(format!("clOrdId {cl_ord_id}"));
    }

    if let Some(account_id) = marker.account_id {
        let account_label = session
            .accounts
            .iter()
            .find(|account| account.id == account_id)
            .map(|account| format!("{} ({account_id})", account.name))
            .unwrap_or_else(|| account_id.to_string());
        parts.push(format!("account {account_label}"));
    }
    if let Some(contract_id) = marker.contract_id {
        parts.push(format!("contractId {contract_id}"));
    }

    if let Some(fee) = pick_number(
        fill,
        &[
            "commission",
            "commissions",
            "fee",
            "fees",
            "totalFees",
            "clearingFee",
            "exchangeFee",
        ],
    ) {
        parts.push(format!("fee {}", format_signed_money(fee)));
    }
    if let Some(pnl) = pick_number(
        fill,
        &[
            "pnl",
            "Pnl",
            "profitLoss",
            "profitAndLoss",
            "realizedPnl",
            "realizedPnL",
            "netPnl",
        ],
    ) {
        parts.push(format!("pnl {}", format_signed_money(pnl)));
    }
    if let Some(source) = fill.get("source").and_then(Value::as_str) {
        parts.push(format!("source {source}"));
    }
    parts.push(format!("ts_ns {}", marker.ts_ns));

    parts.join(" | ")
}

fn format_signed_money(value: f64) -> String {
    if value.is_sign_negative() {
        format!("{value:.2}")
    } else {
        format!("+{value:.2}")
    }
}

fn trade_marker_from_fill(session: &SessionState, fill: &Value) -> Option<TradeMarker> {
    let fill_id = extract_entity_id(fill)?;
    let order_id = json_i64(fill, "orderId");
    let fill_account_id = extract_account_id("fill", fill);
    let order_match = find_order_for_fill(session, fill_account_id, order_id);
    let account_id = fill_account_id
        .or_else(|| order_match.map(|(account_id, _)| account_id))
        .or_else(|| {
            fill_matches_active_latency_tracker(session, fill).then_some(session.selected_account_id)?
        })?;
    let order = order_match.map(|(_, order)| order);
    let side = order
        .and_then(trade_side_from_order)
        .or_else(|| trade_side_from_fill(fill))?;
    let price = pick_number(fill, &["price", "fillPrice", "lastPrice", "avgPrice"])?;
    let qty = pick_number(fill, &["qty", "fillQty", "lastQty", "quantity"])?
        .abs()
        .round() as i32;
    if qty <= 0 {
        return None;
    }

    let ts_ns = json_timestamp_ns(fill, &["timestamp", "fillTime", "createdTime"])
        .or_else(|| session.market.bars.last().map(|bar| bar.ts_ns))?;
    let contract_id = json_i64(fill, "contractId")
        .or_else(|| {
            fill.get("contract")
                .and_then(|contract| json_i64(contract, "id"))
        })
        .or_else(|| order.and_then(order_contract_id));
    let contract_name = fill
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| fill.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| fill.get("name").and_then(Value::as_str))
        .or_else(|| order.and_then(order_symbol))
        .map(ToString::to_string);

    Some(TradeMarker {
        fill_id: Some(fill_id),
        account_id: Some(account_id),
        contract_id,
        contract_name,
        ts_ns,
        price,
        qty,
        side,
    })
}

fn find_order_for_fill<'a>(
    session: &'a SessionState,
    account_id: Option<i64>,
    order_id: Option<i64>,
) -> Option<(i64, &'a Value)> {
    let order_id = order_id?;
    if let Some(account_id) = account_id {
        if let Some(order) = session.user_store.find_order(account_id, order_id) {
            return Some((account_id, order));
        }
    }
    session
        .user_store
        .orders
        .iter()
        .find_map(|(account_id, orders)| orders.get(&order_id).map(|order| (*account_id, order)))
}

fn trade_side_from_order(order: &Value) -> Option<TradeMarkerSide> {
    order
        .get("action")
        .and_then(Value::as_str)
        .and_then(trade_side_from_text)
}

fn trade_side_from_fill(fill: &Value) -> Option<TradeMarkerSide> {
    ["buySell", "side", "action"]
        .iter()
        .find_map(|key| fill.get(*key).and_then(Value::as_str))
        .and_then(trade_side_from_text)
}

fn trade_side_from_text(value: &str) -> Option<TradeMarkerSide> {
    match value.trim().to_ascii_lowercase().as_str() {
        "buy" | "bot" | "b" | "long" => Some(TradeMarkerSide::Buy),
        "sell" | "sld" | "s" | "short" => Some(TradeMarkerSide::Sell),
        _ => None,
    }
}

fn order_contract_id(order: &Value) -> Option<i64> {
    json_i64(order, "contractId").or_else(|| {
        order
            .get("contract")
            .and_then(|contract| json_i64(contract, "id"))
    })
}

fn order_symbol(order: &Value) -> Option<&str> {
    order
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| order.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| order.get("name").and_then(Value::as_str))
        .or_else(|| {
            order
                .get("contract")
                .and_then(|contract| contract.get("name"))
                .and_then(Value::as_str)
        })
}

fn order_type(order: &Value) -> Option<String> {
    order
        .get("orderType")
        .and_then(Value::as_str)
        .or_else(|| order.get("ordType").and_then(Value::as_str))
        .map(str::trim)
        .map(|value| value.to_ascii_lowercase())
}

fn strategy_contract_id(strategy: &Value) -> Option<i64> {
    json_i64(strategy, "contractId").or_else(|| {
        strategy
            .get("contract")
            .and_then(|contract| json_i64(contract, "id"))
    })
}

fn json_timestamp_ns(value: &Value, keys: &[&str]) -> Option<i64> {
    for key in keys {
        let Some(raw) = value.get(*key) else {
            continue;
        };
        if let Some(timestamp) = raw.as_i64().and_then(normalize_unix_timestamp_ns) {
            return Some(timestamp);
        }
        if let Some(text) = raw.as_str() {
            if let Some(timestamp) = parse_bar_timestamp_ns(text) {
                return Some(timestamp);
            }
            if let Ok(parsed) = text.parse::<i64>() {
                if let Some(timestamp) = normalize_unix_timestamp_ns(parsed) {
                    return Some(timestamp);
                }
            }
        }
    }
    None
}

fn normalize_unix_timestamp_ns(raw: i64) -> Option<i64> {
    let magnitude = raw.unsigned_abs();
    if magnitude >= 1_000_000_000_000_000_000 {
        Some(raw)
    } else if magnitude >= 1_000_000_000_000_000 {
        raw.checked_mul(1_000)
    } else if magnitude >= 1_000_000_000_000 {
        raw.checked_mul(1_000_000)
    } else if magnitude >= 1_000_000_000 {
        raw.checked_mul(1_000_000_000)
    } else {
        None
    }
}
