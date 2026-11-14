//! Read-only standalone market endpoints used by the local web UI.
//!
//! The router is mounted by the web server module. The stream endpoint starts
//! the existing `standalone-market --stream` worker and forwards only validated
//! snapshots as SSE; provider diagnostics and child stderr are never returned.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::Request,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    process::Command,
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    time::timeout,
};

const MAX_REQUEST_BYTES: usize = 2_048;
const MAX_STREAM_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_BARS: usize = 5_000;
const SYMBOL: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static RUNNING: OnceLock<Arc<Semaphore>> = OnceLock::new();

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarketRequest {
    contract: String,
    bar_kind: String,
    bar_value: u32,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: &'static str,
}

pub(super) fn router() -> Router {
    Router::new()
        .route("/api/standalone-market", post(snapshot))
        .route("/api/standalone-market/stream", post(stream))
}

fn error(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "no-store")],
        Json(ApiError { error: message }),
    )
        .into_response()
}

fn validate(request: &MarketRequest) -> Result<(), Response> {
    if request.contract.is_empty()
        || request.contract.len() > 16
        || !request
            .contract
            .bytes()
            .all(|b| SYMBOL.as_bytes().contains(&b))
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Enter an exact alphanumeric contract symbol from 1 to 16 characters.",
        ));
    }
    if !["minute", "second", "tick", "volume", "range"].contains(&request.bar_kind.as_str())
        || !(1..=100_000).contains(&request.bar_value)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Choose minute, second, tick, volume, or range bars with a whole number from 1 to 100000.",
        ));
    }
    Ok(())
}

async fn request_body(request: Request) -> Result<MarketRequest, Response> {
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|n| n > MAX_REQUEST_BYTES)
    {
        return Err(error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body is too large.",
        ));
    }
    let bytes = axum::body::to_bytes(request.into_body(), MAX_REQUEST_BYTES)
        .await
        .map_err(|_| error(StatusCode::PAYLOAD_TOO_LARGE, "Request body is too large."))?;
    let request: MarketRequest = serde_json::from_slice(&bytes).map_err(|_| {
        error(
            StatusCode::BAD_REQUEST,
            "Request body must be valid JSON with only contract, bar_kind, and bar_value fields.",
        )
    })?;
    validate(&request)?;
    Ok(request)
}

fn acquire() -> Result<OwnedSemaphorePermit, Response> {
    RUNNING
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "A standalone market request is already running.",
            )
        })
}

async fn snapshot(request: Request) -> Response {
    let request = match request_body(request).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let _guard = match acquire() {
        Ok(v) => v,
        Err(r) => return r,
    };
    let config = config_args();
    let mut command = Command::new(trader_binary());
    command.current_dir(trader_directory());
    command
        .args(config)
        .args([
            "standalone-market",
            "--contract",
            &request.contract,
            "--bar-kind",
            &request.bar_kind,
            "--bar-value",
            &request.bar_value.to_string(),
            "--timeout-seconds",
            "15",
        ])
        .env("RUST_LOG", "off")
        .kill_on_drop(true)
        .stderr(std::process::Stdio::null());
    let read = async {
        let mut child = command.stdout(std::process::Stdio::piped()).spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("missing stdout"))?;
        let mut data = Vec::new();
        stdout
            .take(MAX_STREAM_LINE_BYTES as u64 + 1)
            .read_to_end(&mut data)
            .await?;
        if data.len() > MAX_STREAM_LINE_BYTES {
            return Err(std::io::Error::other("output too large"));
        }
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((status, data))
    };
    let output = match timeout(Duration::from_secs(20), read).await {
        Err(_) => {
            return error(
                StatusCode::GATEWAY_TIMEOUT,
                "Standalone market request exceeded its 20 second limit.",
            );
        }
        Ok(Err(_)) => {
            return error(
                StatusCode::BAD_GATEWAY,
                "Standalone market request failed. Check local provider configuration and connectivity.",
            );
        }
        Ok(Ok((status, _))) if !status.success() => {
            return error(
                StatusCode::BAD_GATEWAY,
                "Standalone market request failed. Check local provider configuration and connectivity.",
            );
        }
        Ok(Ok((_, output))) => output,
    };
    let value: Value = match serde_json::from_slice(&output) {
        Ok(value) => value,
        Err(_) => {
            return error(
                StatusCode::BAD_GATEWAY,
                "Trader returned invalid standalone market JSON.",
            );
        }
    };
    match safe_payload(value, &request) {
        Ok(value) => (
            StatusCode::OK,
            [(header::CACHE_CONTROL, "no-store")],
            Json(value),
        )
            .into_response(),
        Err(message) => error(StatusCode::BAD_GATEWAY, message),
    }
}

fn trader_binary() -> PathBuf {
    std::env::var_os("MIDAS_TRADER_BIN")
        .map(|p| absolute(PathBuf::from(p)))
        .unwrap_or_else(|| std::env::current_exe().unwrap_or_else(|_| PathBuf::from("trader")))
}
pub(super) fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    }
}
pub(super) fn trader_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("MIDAS_TRADER_DIR") {
        return PathBuf::from(path);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("trader/Cargo.toml").is_file() {
        cwd.join("trader")
    } else if cwd.join("../trader/Cargo.toml").is_file() {
        cwd.join("../trader")
    } else {
        cwd
    }
}
fn config_args() -> Vec<String> {
    std::env::var_os("MIDAS_CONFIG")
        .or_else(|| std::env::var_os("TRADER_CONFIG"))
        .or_else(|| std::env::var_os("MIDAS_TUI_CONFIG"))
        .map(|path| {
            vec![
                "--config".to_owned(),
                absolute(PathBuf::from(path)).to_string_lossy().into_owned(),
            ]
        })
        .unwrap_or_default()
}

fn safe_payload(source: Value, request: &MarketRequest) -> Result<Value, &'static str> {
    let bars = source
        .get("bars")
        .and_then(Value::as_array)
        .ok_or("Trader returned an invalid standalone market snapshot.")?;
    if bars.len() > MAX_BARS {
        return Err("Trader returned an invalid standalone market snapshot.");
    }
    let mut clean = Vec::with_capacity(bars.len());
    for bar in bars {
        let ts = match bar.get("ts_ns") {
            Some(Value::String(ts)) if valid_i64_string(ts) => ts.clone(),
            Some(Value::Number(n)) if n.as_i64().is_some() => n.to_string(),
            _ => return Err("Trader returned an invalid standalone market bar."),
        };
        let (open, high, low, close) = (
            number(bar, "open"),
            number(bar, "high"),
            number(bar, "low"),
            number(bar, "close"),
        );
        let (Some(open), Some(high), Some(low), Some(close)) = (open, high, low, close) else {
            return Err("Trader returned an invalid standalone market bar.");
        };
        if high < low {
            return Err("Trader returned an invalid standalone market bar.");
        }
        let mut item = json!({"ts_ns":ts,"open":open,"high":high,"low":low,"close":close});
        if let Some(volume) = number(bar, "volume") {
            item["volume"] = json!(volume);
        }
        clean.push(item);
    }
    let latest = clean
        .last()
        .and_then(|b| b.get("ts_ns"))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(json!({
        "connection_status": if source.get("connection_status").and_then(Value::as_str) == Some("connected") { "connected" } else { "unknown" },
        "chart_freshness":"unknown",
        "source": clipped(source.get("source"), 100, "provider_market_data"),
        "broker_source": clipped(source.get("broker_source"), 100, "Provider not reported"),
        "fidelity": clipped(source.get("fidelity"), 200, "Provider chart bars; not a guaranteed complete trade tape."),
        "contract":request.contract,"bar_kind":request.bar_kind,"bar_value":request.bar_value,
        "bars":clean,"latest_bar_ts_ns":latest,
        "observed_at":source.get("observed_at").and_then(Value::as_str).map(|s| s.chars().take(64).collect::<String>())
    }))
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
}
fn clipped(value: Option<&Value>, max: usize, fallback: &str) -> String {
    value
        .and_then(Value::as_str)
        .map(|s| s.chars().take(max).collect())
        .unwrap_or_else(|| fallback.to_owned())
}
fn valid_i64_string(value: &str) -> bool {
    let digits = value.strip_prefix('-').unwrap_or(value);
    !digits.is_empty()
        && (digits == "0" || !digits.starts_with('0'))
        && digits.bytes().all(|b| b.is_ascii_digit())
        && value.parse::<i64>().is_ok()
}

async fn read_limited_line(
    reader: &mut BufReader<tokio::process::ChildStdout>,
) -> std::io::Result<Option<String>> {
    let mut line = Vec::new();
    let count = (&mut *reader)
        .take(MAX_STREAM_LINE_BYTES as u64 + 1)
        .read_until(b'\n', &mut line)
        .await?;
    if count == 0 {
        return Ok(None);
    }
    if line.len() > MAX_STREAM_LINE_BYTES {
        return Err(std::io::Error::other("line too long"));
    }
    String::from_utf8(line)
        .map(Some)
        .map_err(|_| std::io::Error::other("invalid UTF-8"))
}

async fn stream(request: Request) -> Response {
    let request = match request_body(request).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let permit = match acquire() {
        Ok(v) => v,
        Err(r) => return r,
    };
    let mut cmd = Command::new(trader_binary());
    cmd.current_dir(trader_directory());
    cmd.args(config_args())
        .args([
            "standalone-market",
            "--contract",
            &request.contract,
            "--bar-kind",
            &request.bar_kind,
            "--bar-value",
            &request.bar_value.to_string(),
            "--timeout-seconds",
            "900",
            "--stream",
        ])
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let (tx, rx) = mpsc::channel::<Result<String, &'static str>>(8);
    let task = tokio::spawn(async move {
        let _permit: OwnedSemaphorePermit = permit;
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(_) => {
                let _ = tx.send(Err("Trader CLI is unavailable.")).await;
                return;
            }
        };
        let Some(stdout) = child.stdout.take() else {
            let _ = tx.send(Err("Standalone market stream failed.")).await;
            return;
        };
        let mut reader = BufReader::new(stdout);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(900);
        let mut first = true;
        loop {
            let next = if first {
                match tokio::select! {
                    _ = tx.closed() => break,
                    result = timeout(Duration::from_secs(95), read_limited_line(&mut reader)) => result,
                } {
                    Err(_) => {
                        let _ = tx.send(Err("Standalone market stream produced no first snapshot within its startup deadline.")).await;
                        break;
                    }
                    Ok(value) => value,
                }
            } else {
                tokio::select! {
                    _ = tx.closed() => break,
                    result = timeout(deadline.saturating_duration_since(tokio::time::Instant::now()), read_limited_line(&mut reader)) => result.unwrap_or(Ok(None)),
                }
            };
            first = false;
            match next {
                Ok(Some(line)) if line.len() <= MAX_STREAM_LINE_BYTES => {
                    let value = serde_json::from_str(&line)
                        .ok()
                        .and_then(|v| safe_payload(v, &request).ok());
                    match value {
                        Some(snapshot) => {
                            if tx
                                .send(Ok(format!("event: snapshot\ndata: {snapshot}\n\n")))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        None => {
                            let _ = tx
                                .send(Err(
                                    "Trader returned an invalid standalone market stream snapshot.",
                                ))
                                .await;
                            break;
                        }
                    }
                }
                Ok(Some(_)) => {
                    let _ = tx
                        .send(Err("Trader market stream line exceeded its size limit."))
                        .await;
                    break;
                }
                Ok(None) => {
                    if tokio::time::Instant::now() < deadline {
                        let _ = tx
                            .send(Err("Standalone market stream stopped unexpectedly."))
                            .await;
                    }
                    break;
                }
                Err(_) => {
                    let _ = tx.send(Err("Standalone market stream failed. Check local provider configuration and connectivity.")).await;
                    break;
                }
            }
        }
        let _ = child.start_kill();
        let _ = timeout(Duration::from_secs(2), child.wait()).await;
    });
    let stream = futures_util::stream::unfold(
        (rx, task, false),
        |(mut rx, task, sent_status)| async move {
            if sent_status {
                return None;
            }
            match rx.recv().await {
                Some(Ok(event)) => Some((
                    Ok::<Bytes, std::convert::Infallible>(Bytes::from(event)),
                    (rx, task, false),
                )),
                Some(Err(message)) => {
                    let data = format!(
                        "event: status\ndata: {}\n\n",
                        json!({"connection_status":"disconnected","error":message})
                    );
                    Some((Ok(Bytes::from(data)), (rx, task, true)))
                }
                None => {
                    task.abort();
                    let data = format!(
                        "event: status\ndata: {}\n\n",
                        json!({"connection_status":"stopped","reason":"Stream time limit reached."})
                    );
                    Some((Ok(Bytes::from(data)), (rx, task, true)))
                }
            }
        },
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store, no-transform")
        .header(header::CONNECTION, "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Standalone market stream failed.",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn payload_preserves_exact_timestamps_and_removes_secrets() {
        let request = MarketRequest {
            contract: "ESZ6".into(),
            bar_kind: "minute".into(),
            bar_value: 1,
        };
        let snapshot = json!({"connection_status":"connected", "secret":"do-not-forward", "bars":[{"ts_ns":1790000000000000001i64,"open":1,"high":3,"low":1,"close":2,"account":"hidden"}]});
        let output = safe_payload(snapshot, &request).unwrap();
        assert_eq!(output["bars"][0]["ts_ns"], "1790000000000000001");
        assert!(output.get("secret").is_none());
        assert!(output["bars"][0].get("account").is_none());
    }
    #[test]
    fn market_request_rejects_bad_contracts_and_bar_values() {
        assert!(
            validate(&MarketRequest {
                contract: "ES;rm".into(),
                bar_kind: "minute".into(),
                bar_value: 1
            })
            .is_err()
        );
        assert!(
            validate(&MarketRequest {
                contract: "ESZ6".into(),
                bar_kind: "minute".into(),
                bar_value: 0
            })
            .is_err()
        );
    }
}
