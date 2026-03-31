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
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.fill_recorded {
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
    let entity_cl_ord_id = entity.get("clOrdId").and_then(Value::as_str);
    let entity_order_id = json_i64(entity, "orderId").or_else(|| json_i64(entity, "id"));
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

fn trade_marker_from_fill(session: &SessionState, fill: &Value) -> Option<TradeMarker> {
    let fill_id = extract_entity_id(fill)?;
    let account_id = extract_account_id("fill", fill)?;
    let order_id = json_i64(fill, "orderId");
    let order = order_id.and_then(|order_id| session.user_store.find_order(account_id, order_id));
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
