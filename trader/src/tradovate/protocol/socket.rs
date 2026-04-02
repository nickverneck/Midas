use super::*;

pub(crate) fn parse_status_code(msg: &Value) -> Option<i64> {
    if let Some(code) = msg.get("s").and_then(Value::as_i64) {
        return Some(code);
    }
    msg.get("s")
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<i64>().ok())
}

pub(crate) fn parse_frame(raw: &str) -> (char, Option<Value>) {
    let mut chars = raw.chars();
    let frame_type = chars.next().unwrap_or('\0');
    let offset = frame_type.len_utf8();
    let payload = raw.get(offset..).unwrap_or("");
    let value = if payload.is_empty() {
        None
    } else {
        serde_json::from_str(payload).ok()
    };
    (frame_type, value)
}

pub(crate) fn create_message(
    endpoint: &str,
    id: u64,
    query: Option<&str>,
    body: Option<&Value>,
) -> String {
    match (query, body) {
        (Some(query), Some(body)) => format!("{endpoint}\n{id}\n{query}\n{body}"),
        (Some(query), None) => format!("{endpoint}\n{id}\n{query}"),
        (None, Some(body)) => format!("{endpoint}\n{id}\n\n{body}"),
        (None, None) => format!("{endpoint}\n{id}\n\n"),
    }
}

pub(crate) fn parse_socket_response(message: &Value) -> Result<Value, String> {
    let Some(status) = parse_status_code(message) else {
        return Err("websocket response missing status code".to_string());
    };
    let payload = message.get("d").cloned().unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        Ok(payload)
    } else if let Some(text) = payload.as_str() {
        Err(format!("websocket request failed ({status}): {text}"))
    } else {
        Err(format!("websocket request failed ({status}): {payload}"))
    }
}
