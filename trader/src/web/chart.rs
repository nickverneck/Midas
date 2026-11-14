//! Local chart endpoints used by the web UI.
//!
//! The chart-cache endpoint delegates to the existing CLI command so its
//! catalog and paging semantics remain identical. Live market data is read
//! from the engine's read-only Unix socket and stripped to chart-safe fields.

use axum::{
    Router,
    extract::{Path, RawQuery},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    env,
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    net::UnixStream,
    process::Command,
    time::timeout,
};

const MAX_BUFFER: usize = 8 * 1024 * 1024;
const MAX_OFFSET: u64 = 10_000_000;
const MAX_LIVE_LINE: usize = 8 * 1024 * 1024;
static CHART_CACHE_RUNNING: AtomicBool = AtomicBool::new(false);

pub(super) fn router() -> Router {
    Router::new()
        .route("/api/chart-cache", get(chart_catalog))
        .route("/api/chart-cache/{dataset}", get(chart_dataset))
        .route("/api/live-market", get(live_market))
}

async fn chart_catalog() -> Response {
    chart_cache(Vec::new()).await
}

async fn chart_dataset(Path(dataset): Path<String>, RawQuery(query): RawQuery) -> Response {
    if dataset.len() != 64
        || !dataset
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return error(
            StatusCode::BAD_REQUEST,
            "dataset must be a 64 character chart-cache dataset id.",
        );
    }
    let args = match parse_page_query(query.as_deref().unwrap_or("")) {
        Ok(args) => args,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let mut args = args;
    args.extend(["--dataset".into(), dataset]);
    chart_cache(args).await
}

fn parse_page_query(query: &str) -> Result<Vec<String>, String> {
    let mut values = std::collections::HashMap::<String, String>::new();
    let allowed: HashSet<&str> = [
        "input",
        "bar_kind",
        "bar_value",
        "chart_mode",
        "offset",
        "limit",
        "start_ns",
        "end_ns",
    ]
    .into_iter()
    .collect();
    for pair in query.split('&').filter(|part| !part.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let key = decode_query_component(key)?;
        let value = decode_query_component(value)?;
        if !allowed.contains(key.as_str()) {
            return Err(format!("Unsupported query parameter: {key}."));
        }
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("Query parameter {key} may only be specified once."));
        }
    }
    let input = values
        .get("input")
        .map(String::as_str)
        .unwrap_or("server-bars");
    if !["server-bars", "raw-ticks"].contains(&input) {
        return Err("input must be server-bars or raw-ticks.".into());
    }
    let kind = values
        .get("bar_kind")
        .map(String::as_str)
        .unwrap_or("minute");
    if !["minute", "second", "tick", "volume", "range"].contains(&kind) {
        return Err("bar_kind must be minute, second, tick, volume, or range.".into());
    }
    let bar_value = query_uint(&values, "bar_value", 1)?;
    if !(1..=1_000_000).contains(&bar_value) {
        return Err("bar_value must be between 1 and 1000000.".into());
    }
    let mode = values
        .get("chart_mode")
        .map(String::as_str)
        .unwrap_or("ohlc");
    if !["ohlc", "heikin-ashi"].contains(&mode) {
        return Err("chart_mode must be ohlc or heikin-ashi.".into());
    }
    let offset = query_uint(&values, "offset", 0)?;
    if offset > MAX_OFFSET {
        return Err(format!("offset may not exceed {MAX_OFFSET}."));
    }
    let limit = query_uint(&values, "limit", 1000)?;
    if !(1..=5000).contains(&limit) {
        return Err("limit must be between 1 and 5000.".into());
    }
    let start = values
        .get("start_ns")
        .map(|value| query_i64(value, "start_ns"))
        .transpose()?;
    let end = values
        .get("end_ns")
        .map(|value| query_i64(value, "end_ns"))
        .transpose()?;
    if let (Some(start), Some(end)) = (start, end) {
        if start >= end {
            return Err("start_ns must be earlier than end_ns.".into());
        }
    }
    let mut args = vec![
        "--input".into(),
        input.into(),
        "--bar-kind".into(),
        kind.into(),
        "--bar-value".into(),
        bar_value.to_string(),
        "--chart-mode".into(),
        mode.into(),
        "--offset".into(),
        offset.to_string(),
        "--limit".into(),
        limit.to_string(),
    ];
    if let Some(start) = start {
        args.extend(["--start-ns".into(), start.to_string()]);
    }
    if let Some(end) = end {
        args.extend(["--end-ns".into(), end.to_string()]);
    }
    Ok(args)
}

fn decode_query_component(raw: &str) -> Result<String, String> {
    let bytes = raw.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hi = (bytes[index + 1] as char).to_digit(16);
                let lo = (bytes[index + 2] as char).to_digit(16);
                match (hi, lo) {
                    (Some(hi), Some(lo)) => decoded.push((hi * 16 + lo) as u8),
                    _ => return Err("Invalid URL query encoding.".into()),
                }
                index += 3;
            }
            b'%' => return Err("Invalid URL query encoding.".into()),
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(decoded).map_err(|_| "Invalid URL query encoding.".into())
}

fn query_uint(
    values: &std::collections::HashMap<String, String>,
    name: &str,
    default: u64,
) -> Result<u64, String> {
    let Some(raw) = values.get(name).filter(|value| !value.is_empty()) else {
        return Ok(default);
    };
    if raw.len() > 10 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{name} must be a non-negative integer."));
    }
    raw.parse()
        .map_err(|_| format!("{name} is outside the supported range."))
}

fn query_i64(raw: &str, name: &str) -> Result<i64, String> {
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    if digits.is_empty() || digits.len() > 19 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "{name} must be a signed UTC epoch nanosecond integer."
        ));
    }
    raw.parse::<i64>()
        .map_err(|_| format!("{name} is outside the supported signed 64-bit range."))
}

async fn chart_cache(args: Vec<String>) -> Response {
    if CHART_CACHE_RUNNING
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "The chart cache reader is busy. Retry shortly.",
        );
    }
    struct RunningGuard;
    impl Drop for RunningGuard {
        fn drop(&mut self) {
            CHART_CACHE_RUNNING.store(false, Ordering::Release);
        }
    }
    let _guard = RunningGuard;
    let result = timeout(Duration::from_secs(15), run_chart_cache(args)).await;
    match result {
        Err(_) => error(
            StatusCode::GATEWAY_TIMEOUT,
            "The trader chart-cache command exceeded its 15 second limit.",
        ),
        Ok(Ok(body)) => json_response(StatusCode::OK, body),
        Ok(Err((status, message))) => error(status, &message),
    }
}

async fn run_chart_cache(args: Vec<String>) -> Result<Value, (StatusCode, String)> {
    let executable = match env::var_os("MIDAS_TRADER_BIN") {
        Some(path) => super::standalone::absolute(PathBuf::from(path)),
        None => {
            env::current_exe().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        }
    };
    let mut command = Command::new(&executable);
    command.current_dir(super::standalone::trader_directory());
    if let Some(config) = env::var_os("MIDAS_CONFIG") {
        command
            .arg("--config")
            .arg(super::standalone::absolute(PathBuf::from(config)));
    }
    command
        .arg("chart-cache")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Trader CLI not found.".into(),
            )
        } else {
            (
                StatusCode::BAD_GATEWAY,
                "Unable to start chart-cache command.".into(),
            )
        }
    })?;
    let stdout = child.stdout.take().ok_or((
        StatusCode::BAD_GATEWAY,
        "Missing chart-cache output.".into(),
    ))?;
    let mut output = Vec::new();
    stdout
        .take(MAX_BUFFER as u64 + 1)
        .read_to_end(&mut output)
        .await
        .map_err(|_| {
            (
                StatusCode::BAD_GATEWAY,
                "Unable to read chart-cache output.".into(),
            )
        })?;
    if output.len() > MAX_BUFFER {
        return Err((
            StatusCode::BAD_GATEWAY,
            "The trader chart-cache command returned too much data.".into(),
        ));
    }
    let status = child.wait().await.map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "Trader chart-cache command failed.".into(),
        )
    })?;
    if !status.success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            "Trader chart-cache command failed.".into(),
        ));
    }
    let mut value: Value = serde_json::from_slice(&output).map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "The trader chart-cache command returned invalid JSON.".into(),
        )
    })?;
    stringify_timestamps(&mut value);
    Ok(value)
}

async fn live_market() -> Response {
    match timeout(Duration::from_millis(2500), read_live_market()).await {
        Err(_) => json_response(
            StatusCode::GATEWAY_TIMEOUT,
            json!({"connection_status":"disconnected","error":"Timed out waiting for engine market data."}),
        ),
        Ok(Ok(value)) => json_response(StatusCode::OK, value),
        Ok(Err((status, message))) => json_response(
            status,
            json!({"connection_status":"disconnected","error":message}),
        ),
    }
}

async fn read_live_market() -> Result<Value, (StatusCode, String)> {
    let socket = socket_path()?;
    let stream = timeout(Duration::from_millis(500), UnixStream::connect(&socket))
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Engine socket is unavailable.".into(),
            )
        })?
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Engine socket is disconnected.".into(),
            )
        })?;
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    timeout(
        Duration::from_secs(2),
        (&mut reader)
            .take(MAX_LIVE_LINE as u64 + 1)
            .read_until(b'\n', &mut line),
    )
    .await
    .map_err(|_| {
        (
            StatusCode::GATEWAY_TIMEOUT,
            "Timed out waiting for engine market data.".into(),
        )
    })?
    .map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Engine socket is disconnected.".into(),
        )
    })?;
    if line.len() > MAX_LIVE_LINE {
        return Err((
            StatusCode::BAD_GATEWAY,
            "Engine market message exceeded the size limit.".into(),
        ));
    }
    if line.is_empty() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Engine socket closed before market data arrived.".into(),
        ));
    }
    let wire: Value = serde_json::from_slice(&line).map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "Could not decode engine market data.".into(),
        )
    })?;
    let snapshot = wire
        .pointer("/Event/event/MarketSnapshot")
        .or_else(|| wire.pointer("/event/event/MarketSnapshot"))
        .ok_or((
            StatusCode::BAD_GATEWAY,
            "Engine did not provide a market snapshot.".into(),
        ))?;
    chart_payload(snapshot)
}

fn socket_path() -> Result<PathBuf, (StatusCode, String)> {
    if let Some(configured) = env::var_os("MIDAS_ENGINE_SOCKET") {
        let path = PathBuf::from(configured);
        if !path.is_absolute() {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "MIDAS_ENGINE_SOCKET must be an absolute path.".into(),
            ));
        }
        Ok(path)
    } else {
        let cwd = env::current_dir()
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
        let local = cwd.join(".run/trader-engine.sock");
        if local.exists() {
            Ok(local)
        } else {
            Ok(cwd.join("../.run/trader-engine.sock"))
        }
    }
}

fn chart_payload(snapshot: &Value) -> Result<Value, (StatusCode, String)> {
    let bars = snapshot.get("bars").and_then(Value::as_array).ok_or((
        StatusCode::BAD_GATEWAY,
        "Engine returned an invalid market snapshot.".into(),
    ))?;
    let selected = bars.iter().rev().take(5000).collect::<Vec<_>>();
    let mut output_bars = Vec::with_capacity(selected.len());
    for bar in selected.into_iter().rev() {
        let ts = bar.get("ts_ns").and_then(|value| match value {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        });
        let valid_ts = ts.as_deref().is_some_and(|raw| {
            let digits = raw.strip_prefix('-').unwrap_or(raw);
            !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit())
                && raw.parse::<i64>().is_ok()
        });
        let (Some(ts_ns), Some(open), Some(high), Some(low), Some(close)) = (
            ts.filter(|_| valid_ts),
            number(bar, "open"),
            number(bar, "high"),
            number(bar, "low"),
            number(bar, "close"),
        ) else {
            return Err((
                StatusCode::BAD_GATEWAY,
                "Engine returned an invalid market bar.".into(),
            ));
        };
        if high < low {
            return Err((
                StatusCode::BAD_GATEWAY,
                "Engine returned an invalid market bar.".into(),
            ));
        }
        let mut item = json!({"ts_ns":ts_ns,"open":open,"high":high,"low":low,"close":close});
        if let Some(volume) = number(bar, "volume") {
            item["volume"] = json!(volume);
        }
        output_bars.push(item);
    }
    let latest = output_bars
        .last()
        .and_then(|bar| bar.get("ts_ns"))
        .cloned()
        .unwrap_or(Value::Null);
    let contract = snapshot
        .get("contract_name")
        .filter(|v| v.is_string())
        .cloned()
        .unwrap_or(Value::Null);
    let status = snapshot.get("status").and_then(Value::as_str).unwrap_or("");
    Ok(json!({
        "connection_status":"connected", "chart_freshness":"unknown", "source":"engine_market_snapshot",
        "broker_source":"not reported by engine IPC", "contract":contract, "bars":output_bars,
        "latest_bar_ts_ns":latest, "chart_as_of_ts_ns":latest, "engine_status":status.chars().take(256).collect::<String>()
    }))
}

fn stringify_timestamps(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, item) in object.iter_mut() {
                if matches!(key.as_str(), "timestamp_ns" | "ts_ns") {
                    if let Value::Number(number) = item {
                        *item = Value::String(number.to_string());
                    }
                } else {
                    stringify_timestamps(item);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(stringify_timestamps),
        _ => {}
    }
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
}

fn error(status: StatusCode, message: &str) -> Response {
    json_response(status, json!({"error":message}))
}

fn json_response(status: StatusCode, value: Value) -> Response {
    let mut response = (status, axum::Json(value)).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_query_preserves_large_nanoseconds_and_rejects_invalid_ranges() {
        let args = parse_page_query("input=raw-ticks&bar_kind=volume&bar_value=50&start_ns=1790000000000000001&end_ns=1790000000000000002").unwrap();
        assert!(args.iter().any(|s| s == "1790000000000000001"));
        assert!(parse_page_query("limit=5001").is_err());
        assert!(parse_page_query("start_ns=5&end_ns=5").is_err());
        assert!(parse_page_query("input=raw-ticks&unexpected=1").is_err());
    }
    #[test]
    fn market_snapshot_removes_unrelated_fields_and_preserves_nanoseconds() {
        let snapshot = json!({"contract_name":"ES", "account_id":"secret", "bars":[{"ts_ns":1790000000000000001i64,"open":1,"high":3,"low":1,"close":2,"extra":"secret"}]});
        let result = chart_payload(&snapshot).unwrap();
        assert_eq!(result["bars"][0]["ts_ns"], "1790000000000000001");
        assert!(result.get("account_id").is_none());
        assert!(result["bars"][0].get("extra").is_none());
        let invalid = json!({"bars":[{"ts_ns":"-","open":1,"high":3,"low":1,"close":2}]});
        assert!(chart_payload(&invalid).is_err());
    }
}
