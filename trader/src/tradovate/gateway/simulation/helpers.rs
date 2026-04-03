use super::*;

pub(super) fn exit_action_for_target(target_qty: i32) -> &'static str {
    if target_qty > 0 { "Sell" } else { "Buy" }
}

pub(super) fn synthetic_ts_ns(reference_ts_ns: Option<i64>) -> i64 {
    reference_ts_ns
        .or_else(|| Utc::now().timestamp_nanos_opt())
        .unwrap_or_default()
}

pub(super) fn working_order_entity(
    order_id: i64,
    contract_id: i64,
    payload: &Value,
    order_strategy_id: Option<i64>,
) -> Value {
    let mut entity = json!({
        "id": order_id,
        "orderId": order_id,
        "accountId": json_i64(payload, "accountId"),
        "contractId": contract_id,
        "symbol": payload.get("symbol").and_then(Value::as_str),
        "action": payload.get("action").and_then(Value::as_str),
        "orderQty": json_i64(payload, "orderQty"),
        "orderType": payload.get("orderType").and_then(Value::as_str).unwrap_or("Limit"),
        "ordStatus": "Working",
    });
    if let Some(cl_ord_id) = payload.get("clOrdId").and_then(Value::as_str) {
        entity["clOrdId"] = Value::String(cl_ord_id.to_string());
    }
    if let Some(price) = payload.get("price").and_then(Value::as_f64) {
        entity["price"] = json!(price);
    }
    if let Some(stop_price) = payload.get("stopPrice").and_then(Value::as_f64) {
        entity["stopPrice"] = json!(stop_price);
    }
    if let Some(order_strategy_id) = order_strategy_id {
        entity["orderStrategyId"] = json!(order_strategy_id);
    }
    entity
}
