use super::state::{InternalEvent, IronbeamSession};
use crate::broker::{ContractSuggestion, InstrumentSessionProfile};
use crate::config::TradingEnvironment;
use anyhow::{Context, Result, bail};
use reqwest::RequestBuilder;
use serde_json::Value;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;
use tokio::time;

pub(super) const FOLLOWUP_REFRESH_DELAY_MS: u64 = 600;

pub(super) async fn request_json(request: RequestBuilder, context: &str) -> Result<(Value, u64)> {
    let started = Instant::now();
    let response = request
        .send()
        .await
        .with_context(|| format!("{context}: send request"))?;
    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{context} failed ({status}): {body}");
    }
    if body.trim().is_empty() {
        return Ok((Value::Null, elapsed_ms));
    }
    let parsed: Value =
        serde_json::from_str(&body).with_context(|| format!("{context}: parse JSON response"))?;
    ensure_api_ok(&parsed, context)?;
    Ok((parsed, elapsed_ms))
}

pub(super) fn ensure_api_ok(parsed: &Value, context: &str) -> Result<()> {
    if let Some(status) = parsed.get("status").and_then(Value::as_str)
        && !status.eq_ignore_ascii_case("OK")
    {
        let message = parsed
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default();
        bail!("{context} rejected ({status}): {message}");
    }
    Ok(())
}

pub(super) fn contract_symbol(contract: &ContractSuggestion) -> &str {
    pick_str(&contract.raw, &["symbol", "exchSym"]).unwrap_or(contract.name.as_str())
}

pub(super) fn contract_session_profile(contract: &ContractSuggestion) -> InstrumentSessionProfile {
    let symbol_type = pick_str(&contract.raw, &["symbolType", "productType"])
        .unwrap_or_default()
        .to_ascii_lowercase();
    if symbol_type.contains("equity") || symbol_type.contains("stock") {
        InstrumentSessionProfile::EquityRth
    } else {
        InstrumentSessionProfile::FuturesGlobex
    }
}

pub(super) fn position_symbol(position: &Value) -> Option<&str> {
    pick_str(position, &["exchSym", "symbol", "s"])
}

pub(super) fn account_id_string(value: &Value) -> Option<String> {
    pick_str(value, &["accountId", "a"]).map(ToString::to_string)
}

pub(super) fn position_key(position: &Value) -> Option<String> {
    if let Some(id) = pick_str(position, &["positionId", "id"]) {
        return Some(id.to_string());
    }
    let symbol = position_symbol(position)?;
    let side = pick_str(position, &["side", "sd", "s"]).unwrap_or("LONG");
    Some(format!("{symbol}:{side}"))
}

pub(super) fn order_id_string(order: &Value) -> Option<String> {
    pick_str(order, &["orderId", "oid", "id"]).map(ToString::to_string)
}

pub(super) fn fill_key(fill: &Value) -> Option<String> {
    if let Some(update_id) = pick_str(fill, &["orderUpdateId", "fillId", "id"]) {
        return Some(update_id.to_string());
    }
    let order_id = order_id_string(fill)?;
    let fill_date = pick_str(fill, &["fillDate"]).unwrap_or("unknown");
    let fill_qty = pick_number(fill, &["fillQuantity", "quantity"]).unwrap_or_default();
    Some(format!("{order_id}:{fill_date}:{fill_qty:.4}"))
}

pub(super) fn order_symbol(order: &Value) -> Option<&str> {
    pick_str(order, &["exchSym", "symbol", "s"])
}

pub(super) fn order_side(order: &Value) -> Option<&str> {
    pick_str(order, &["side", "sd"])
}

pub(super) fn order_quantity(order: &Value) -> Option<i32> {
    pick_number(order, &["quantity", "q"]).map(|qty| qty.round() as i32)
}

pub(super) fn order_price(order: &Value) -> Option<f64> {
    pick_number(order, &["limitPrice", "stopPrice", "price", "lp", "sp"])
}

pub(super) fn normalized_order_type(order: &Value) -> Option<&'static str> {
    match pick_str(order, &["orderType", "ot"])?
        .trim()
        .to_ascii_uppercase()
        .as_str()
    {
        "1" | "MARKET" => Some("MARKET"),
        "2" | "LIMIT" => Some("LIMIT"),
        "3" | "STOP" => Some("STOP"),
        "4" | "STOP_LIMIT" | "STOPLIMIT" => Some("STOP_LIMIT"),
        _ => None,
    }
}

pub(super) fn is_protection_order_type(order: &Value) -> bool {
    matches!(
        normalized_order_type(order),
        Some("LIMIT" | "STOP" | "STOP_LIMIT")
    )
}

pub(super) fn order_is_active(order: &Value) -> bool {
    let status = pick_str(order, &["status", "st"])
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase();
    if matches!(
        status.as_str(),
        "FILLED"
            | "CANCELLED"
            | "CANCELED"
            | "REJECTED"
            | "EXPIRED"
            | "DONE"
            | "CLOSED"
            | "LIQUIDATED"
            | "INVALID"
    ) {
        return false;
    }

    let quantity = order_quantity(order).unwrap_or_default();
    let filled = pick_number(order, &["fillQuantity", "fillTotalQuantity"])
        .unwrap_or_default()
        .round() as i32;
    quantity <= 0 || filled < quantity
}

pub(super) fn signed_position_qty(position: &Value) -> Option<f64> {
    let qty = pick_number(position, &["quantity", "q", "netPos"])?;
    let side = pick_str(position, &["side", "sd"]).unwrap_or("LONG");
    if side.eq_ignore_ascii_case("SHORT") || qty < 0.0 {
        Some(-qty.abs())
    } else {
        Some(qty)
    }
}

pub(super) fn position_entry_price(position: &Value) -> Option<f64> {
    pick_number(position, &["price", "avgPrice", "averagePrice", "netPrice"])
}

pub(super) fn value_items(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![value],
        _ => Vec::new(),
    }
}

pub(super) fn position_update_groups(value: &Value) -> Vec<(String, Vec<Value>, bool)> {
    match value {
        Value::Array(items) => items
            .iter()
            .flat_map(position_update_groups)
            .collect::<Vec<_>>(),
        Value::Object(object) => {
            if let Some(items) = object.get("positions").and_then(Value::as_array) {
                let Some(account_id) = account_id_string(value) else {
                    return Vec::new();
                };
                return vec![(account_id, items.to_vec(), true)];
            }
            let Some(account_id) = account_id_string(value) else {
                return Vec::new();
            };
            vec![(account_id, vec![Value::Object(object.clone())], false)]
        }
        _ => Vec::new(),
    }
}

pub(super) fn pick_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| value.get(key)?.as_str())
}

pub(super) fn pick_number(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| value.get(key))
        .and_then(value_f64)
}

pub(super) fn pick_i64(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .find_map(|key| value.get(key))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().map(|value| value as i64))
                .or_else(|| value.as_str()?.parse::<i64>().ok())
        })
}

pub(super) fn value_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|number| number as f64))
        .or_else(|| value.as_u64().map(|number| number as f64))
        .or_else(|| value.as_str()?.parse::<f64>().ok())
}

pub(super) fn parse_timestamp_ns(value: &Value) -> Option<i64> {
    if let Some(raw) = value
        .as_i64()
        .and_then(normalize_unix_timestamp_ns)
        .or_else(|| value.as_str()?.parse::<i64>().ok())
        .and_then(normalize_unix_timestamp_ns)
    {
        return Some(raw);
    }

    let text = value.as_str()?;
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|dt| dt.timestamp_nanos_opt().unwrap_or_default())
        .filter(|ts| *ts > 0)
}

pub(super) fn normalize_unix_timestamp_ns(raw: i64) -> Option<i64> {
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

pub(super) fn account_name_by_id(session: &IronbeamSession, account_id: i64) -> Option<&str> {
    session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .map(|account| account.name.as_str())
}

pub(super) fn sanitize_price(price: Option<f64>) -> Option<f64> {
    price.filter(|price| price.is_finite() && *price > 0.0)
}

pub(super) fn prices_match(left: Option<f64>, right: Option<f64>) -> bool {
    match (sanitize_price(left), sanitize_price(right)) {
        (None, None) => true,
        (Some(left), Some(right)) => (left - right).abs() <= 1e-6,
        _ => false,
    }
}

pub(super) fn stable_id(raw: &str) -> i64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash & i64::MAX as u64) as i64
}

pub(super) fn empty_as_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub(super) fn ironbeam_duration_code(time_in_force: &str) -> &'static str {
    match time_in_force.trim().to_ascii_lowercase().as_str() {
        "gtc" | "gtt" | "good_till_cancel" | "good till cancel" => "1",
        _ => "0",
    }
}

pub(super) fn ironbeam_rest_url(env: TradingEnvironment) -> &'static str {
    match env {
        TradingEnvironment::Sim => "https://demo.ironbeamapi.com/v2",
        TradingEnvironment::Live => "https://live.ironbeamapi.com/v2",
    }
}

pub(super) fn ironbeam_ws_base_url(env: TradingEnvironment) -> &'static str {
    match env {
        TradingEnvironment::Sim => "wss://demo.ironbeamapi.com/v2",
        TradingEnvironment::Live => "wss://live.ironbeamapi.com/v2",
    }
}

pub(super) fn schedule_followup_refresh(internal_tx: UnboundedSender<InternalEvent>) {
    schedule_refresh(
        internal_tx,
        Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
        Some("Refreshing Ironbeam account state after order activity.".to_string()),
    );
}

pub(super) fn schedule_refresh(
    internal_tx: UnboundedSender<InternalEvent>,
    delay: Duration,
    reason: Option<String>,
) {
    tokio::spawn(async move {
        time::sleep(delay).await;
        let _ = internal_tx.send(InternalEvent::RefreshAccountState { reason });
    });
}
