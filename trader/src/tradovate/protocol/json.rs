use super::*;

pub(crate) fn json_number(value: &Value, key: &str) -> Option<f64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_f64() {
        return Some(v);
    }
    if let Some(v) = raw.as_i64() {
        return Some(v as f64);
    }
    if let Some(v) = raw.as_u64() {
        return Some(v as f64);
    }
    raw.as_str().and_then(|text| text.parse::<f64>().ok())
}

pub(crate) fn json_i64(value: &Value, key: &str) -> Option<i64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_i64() {
        return Some(v);
    }
    if let Some(v) = raw.as_u64() {
        return i64::try_from(v).ok();
    }
    raw.as_str().and_then(|text| text.parse::<i64>().ok())
}

pub(crate) fn sanitize_price(price: Option<f64>) -> Option<f64> {
    price.filter(|value| value.is_finite() && *value > 0.0)
}

pub(crate) fn with_cl_ord_id(mut payload: Value, cl_ord_id: Option<&str>) -> Value {
    if let Some(cl_ord_id) = cl_ord_id {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("clOrdId".to_string(), Value::String(cl_ord_id.to_string()));
        }
    }
    payload
}

pub(crate) fn prices_match(lhs: Option<f64>, rhs: Option<f64>) -> bool {
    match (lhs, rhs) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

pub(crate) fn empty_as_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
