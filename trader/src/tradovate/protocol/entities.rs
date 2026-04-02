use super::*;

pub(crate) fn extract_entity_envelopes(item: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if item.get("e").and_then(Value::as_str) == Some("props") {
        if let Some(d) = item.get("d") {
            let deleted = matches!(
                d.get("eventType").and_then(Value::as_str),
                Some("Deleted") | Some("deleted")
            );
            if let Some(entity_type) = d.get("entityType").and_then(Value::as_str) {
                if let Some(entity) = d.get("entity") {
                    out.push(EntityEnvelope {
                        entity_type: entity_type.to_string(),
                        deleted,
                        entity: entity.clone(),
                    });
                }
                if let Some(entities) = d.get("entities").and_then(Value::as_array) {
                    for entity in entities {
                        out.push(EntityEnvelope {
                            entity_type: entity_type.to_string(),
                            deleted,
                            entity: entity.clone(),
                        });
                    }
                }
            }
        }
    }

    if let Some(d) = item.get("d") {
        out.extend(extract_response_entities(d));
    }

    out
}

fn extract_response_entities(payload: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if let Some(items) = payload.as_array() {
        for item in items {
            out.extend(extract_response_entities(item));
        }
        return out;
    }

    let Some(obj) = payload.as_object() else {
        return out;
    };

    if let Some(entity_type) = obj.get("entityType").and_then(Value::as_str) {
        if let Some(entity) = obj.get("entity") {
            out.push(EntityEnvelope {
                entity_type: entity_type.to_string(),
                deleted: false,
                entity: entity.clone(),
            });
        }
        if let Some(entities) = obj.get("entities").and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: entity_type.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    for (key, plural) in [
        ("account", "accounts"),
        ("accountRiskStatus", "accountRiskStatuses"),
        ("cashBalance", "cashBalances"),
        ("position", "positions"),
        ("order", "orders"),
        ("orderStrategy", "orderStrategies"),
        ("orderStrategyLink", "orderStrategyLinks"),
        ("executionReport", "executionReports"),
        ("fill", "fills"),
    ] {
        if let Some(entity) = obj.get(key) {
            if entity.is_object() {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
        if let Some(entities) = obj.get(plural).and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    out
}

pub(crate) fn known_order_id(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| json_i64(value, key))
}

pub(crate) fn first_known_order_id(value: &Value) -> Option<i64> {
    known_order_id(value, &["orderId", "id", "otherId", "stopOrderId"])
}

pub(crate) fn order_is_active(order: &Value) -> bool {
    let Some(status) = order
        .get("ordStatus")
        .and_then(Value::as_str)
        .or_else(|| order.get("status").and_then(Value::as_str))
    else {
        return true;
    };

    !matches!(
        status.to_ascii_lowercase().as_str(),
        "filled" | "cancelled" | "canceled" | "rejected" | "expired" | "stopped" | "finished"
    )
}

pub(crate) fn extract_entity_id(value: &Value) -> Option<i64> {
    json_i64(value, "id")
}

pub(crate) fn extract_account_id(entity_type: &str, value: &Value) -> Option<i64> {
    if entity_type.eq_ignore_ascii_case("account") {
        return json_i64(value, "id");
    }
    json_i64(value, "accountId")
        .or_else(|| {
            value
                .get("account")
                .and_then(|account| account.get("id"))
                .and_then(Value::as_i64)
        })
        .or_else(|| value.get("account").and_then(Value::as_i64))
        .or_else(|| json_i64(value, "id"))
}
