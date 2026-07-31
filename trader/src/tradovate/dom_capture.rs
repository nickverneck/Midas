#[derive(Debug, Clone)]
pub struct DomCaptureOptions {
    pub contract: String,
    pub start: String,
    pub end: String,
    pub output: PathBuf,
    pub speed: u16,
    pub initial_balance: f64,
    pub overwrite: bool,
}

#[derive(Debug, Clone)]
pub struct LiveDomCaptureOptions {
    pub contract: String,
    pub duration_seconds: u64,
    pub output: PathBuf,
    pub overwrite: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DomCaptureSummary {
    snapshots: usize,
    first_ts_ns: Option<i64>,
    last_ts_ns: Option<i64>,
}

type DomSocket = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;
type DomRead = futures_util::stream::SplitStream<DomSocket>;
type DomWrite = futures_util::stream::SplitSink<DomSocket, Message>;

const DOM_CAPTURE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DOM_CAPTURE_MAX_SPEED: u16 = 400;

/// Capture historical full-book snapshots through Tradovate Market Replay.
///
/// The replay socket is deliberately separate from the normal engine market
/// socket. Initializing it creates/rewinds a disposable broker replay session,
/// then md/subscribeDOM emits the historical DOM stream as the replay clock
/// advances. No live orders or engine market subscriptions are started.
pub async fn capture_replay_dom(cfg: &AppConfig, options: DomCaptureOptions) -> Result<()> {
    let start = parse_dom_capture_timestamp(&options.start, "--start")?;
    let end = parse_dom_capture_timestamp(&options.end, "--end")?;
    if start >= end {
        bail!("--end must be after --start");
    }
    if options.speed == 0 || options.speed > DOM_CAPTURE_MAX_SPEED {
        bail!(
            "--speed must be between 1 and {}",
            DOM_CAPTURE_MAX_SPEED
        );
    }
    if !options.initial_balance.is_finite() || options.initial_balance < 0.0 {
        bail!("--initial-balance must be finite and non-negative");
    }

    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract = resolve_dom_capture_contract(
        &client,
        &cfg.env,
        &tokens.access_token,
        &options.contract,
    )
    .await?;

    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.replay_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.replay_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    write
        .send(Message::Text(format!(
            "authorize\n1\n\n{}",
            tokens.access_token
        )))
        .await
        .context("authorize replay websocket")?;
    wait_dom_response(&mut read, 1, "replay websocket authorization").await?;

    let start_timestamp = start.to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    send_dom_request(
        &mut write,
        "replay/checkreplaysession",
        2,
        &json!({ "startTimestamp": start_timestamp }),
    )
    .await?;
    let check_response = wait_dom_response(
        &mut read,
        2,
        "check historical replay session entitlement",
    )
    .await?;
    let check_status = check_response
        .pointer("/d/checkStatus")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(check_status.to_ascii_lowercase().as_str(), "ok" | "eligible") {
        bail!(
            "Tradovate historical replay is not available for {}: {}",
            start.to_rfc3339(),
            if check_status.is_empty() {
                "provider omitted checkStatus"
            } else {
                check_status
            }
        );
    }

    send_dom_request(
        &mut write,
        "replay/initializeclock",
        3,
        &json!({
            "startTimestamp": start_timestamp,
            "speed": options.speed,
            "initialBalance": options.initial_balance,
        }),
    )
    .await?;
    let initialize_response =
        wait_dom_response(&mut read, 3, "initialize historical replay clock").await?;
    if !initialize_response
        .pointer("/d/ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let error_text = initialize_response
            .pointer("/d/errorText")
            .and_then(Value::as_str)
            .unwrap_or("provider did not confirm replay initialization");
        bail!("historical replay initialization failed: {error_text}");
    }

    send_dom_request(
        &mut write,
        "md/subscribeDOM",
        4,
        &json!({ "symbol": contract.id }),
    )
    .await?;
    wait_dom_response(&mut read, 4, "subscribe to historical DOM").await?;

    let timeout = historical_dom_capture_timeout(start, end, options.speed);
    let deadline = time::Instant::now() + timeout;
    let mut snapshots = Vec::new();
    let mut replay_ts_ns: Option<i64> = None;

    loop {
        let remaining = deadline.saturating_duration_since(time::Instant::now());
        if remaining.is_zero() {
            bail!(
                "timed out capturing historical DOM for {} after {}",
                contract.name,
                humantime_like_duration(timeout)
            );
        }
        let items = read_dom_items(&mut read, remaining, "historical DOM capture").await?;
        let Some(items) = items else {
            break;
        };
        for item in items {
            if let Some(clock_ts_ns) = parse_replay_clock_timestamp(&item) {
                replay_ts_ns =
                    Some(replay_ts_ns.map_or(clock_ts_ns, |current| current.max(clock_ts_ns)));
            }
            collect_dom_snapshots(&item, contract.id, Some((start, end)), &mut snapshots)?;
        }
        if replay_ts_ns.is_some_and(|ts_ns| {
            ts_ns >= end.timestamp_nanos_opt().unwrap_or(i64::MAX)
        }) {
            break;
        }
    }

    let _ = send_dom_request(
        &mut write,
        "md/unsubscribeDOM",
        5,
        &json!({ "symbol": contract.id }),
    )
    .await;
    let summary = write_dom_jsonl(&options.output, &snapshots, options.overwrite)?;
    println!("Historical Level 2 capture complete.");
    println!("Contract: {}", contract.name);
    println!("Date range: {} to {}", start.to_rfc3339(), end.to_rfc3339());
    println!("Snapshots: {}", summary.snapshots);
    println!("Output: {}", options.output.display());
    Ok(())
}

/// Capture live DOM in a standalone opt-in process. The normal live engine
/// never calls this function, so it adds no subscription or task to trading.
pub async fn capture_live_dom(cfg: &AppConfig, options: LiveDomCaptureOptions) -> Result<()> {
    if options.duration_seconds == 0 {
        bail!("--duration-seconds must be > 0");
    }
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract = resolve_dom_capture_contract(
        &client,
        &cfg.env,
        &tokens.access_token,
        &options.contract,
    )
    .await?;

    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.market_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();
    write
        .send(Message::Text(format!(
            "authorize\n1\n\n{}",
            tokens.md_access_token
        )))
        .await
        .context("authorize live market websocket")?;
    wait_dom_response(&mut read, 1, "live market websocket authorization").await?;

    send_dom_request(
        &mut write,
        "md/subscribeDOM",
        2,
        &json!({ "symbol": contract.id }),
    )
    .await?;
    wait_dom_response(&mut read, 2, "subscribe to live DOM").await?;

    let deadline = time::Instant::now() + Duration::from_secs(options.duration_seconds);
    let mut snapshots = Vec::new();
    loop {
        tokio::select! {
            _ = time::sleep_until(deadline) => break,
            next = read.next() => {
                let Some(next) = next else { break; };
                let message = next.context("read live market websocket")?;
                let Some(items) = dom_items_from_message(message) else { continue; };
                for item in items {
                    collect_dom_snapshots(&item, contract.id, None, &mut snapshots)?;
                }
            }
        }
    }

    let _ = send_dom_request(
        &mut write,
        "md/unsubscribeDOM",
        3,
        &json!({ "symbol": contract.id }),
    )
    .await;
    let summary = write_dom_jsonl(&options.output, &snapshots, options.overwrite)?;
    println!("Live Level 2 capture complete.");
    println!("Contract: {}", contract.name);
    println!("Duration: {} seconds", options.duration_seconds);
    println!("Snapshots: {}", summary.snapshots);
    println!("Output: {}", options.output.display());
    Ok(())
}

async fn resolve_dom_capture_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    requested: &str,
) -> Result<ContractSuggestion> {
    let requested = requested.trim();
    if requested.is_empty() {
        bail!("contract cannot be empty");
    }
    let suggestions = search_contracts(client, env, token, requested, 25).await?;
    suggestions
        .into_iter()
        .find(|contract| contract.name.eq_ignore_ascii_case(requested))
        .with_context(|| {
            format!(
                "Tradovate did not return an exact contract match for {}; use the exact active/entitled contract symbol",
                requested
            )
        })
}

async fn send_dom_request(
    write: &mut DomWrite,
    endpoint: &str,
    request_id: u64,
    body: &Value,
) -> Result<()> {
    write
        .send(Message::Text(create_message(
            endpoint,
            request_id,
            None,
            Some(body),
        )))
        .await
        .with_context(|| format!("send {endpoint}"))?;
    Ok(())
}

async fn wait_dom_response(read: &mut DomRead, request_id: u64, operation: &str) -> Result<Value> {
    loop {
        let items = read_dom_items(read, DOM_CAPTURE_REQUEST_TIMEOUT, operation)
            .await?
            .with_context(|| format!("{operation}: websocket closed before response"))?;
        for item in items {
            if item.get("i").and_then(Value::as_u64) != Some(request_id) {
                continue;
            }
            let status = parse_status_code(&item);
            if !status.is_some_and(|code| (200..300).contains(&code)) {
                bail!(
                    "{operation} failed: {}",
                    dom_websocket_failure(operation, status)
                );
            }
            return Ok(item);
        }
    }
}

fn dom_websocket_failure(endpoint: &str, status: Option<i64>) -> String {
    match status {
        Some(status) => format!(
            "{endpoint} failed with websocket status {status}; provider response body omitted"
        ),
        None => format!(
            "{endpoint} response did not include a status code; provider response body omitted"
        ),
    }
}

async fn read_dom_items(
    read: &mut DomRead,
    timeout: Duration,
    operation: &str,
) -> Result<Option<Vec<Value>>> {
    let next = time::timeout(timeout, read.next())
        .await
        .with_context(|| format!("timed out during {operation}"))?;
    let Some(next) = next else {
        return Ok(None);
    };
    let message = next.with_context(|| format!("read websocket during {operation}"))?;
    Ok(dom_items_from_message(message))
}

fn dom_items_from_message(message: Message) -> Option<Vec<Value>> {
    let raw = match message {
        Message::Text(text) => text,
        Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        Message::Close(_) => return None,
        _ => return Some(Vec::new()),
    };
    let (frame_type, payload) = parse_frame(&raw);
    if frame_type != 'a' {
        return Some(Vec::new());
    }
    payload.and_then(|payload| payload.as_array().cloned())
}

fn collect_dom_snapshots(
    item: &Value,
    contract_id: i64,
    timestamp_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    snapshots: &mut Vec<ReplayMarketDom>,
) -> Result<()> {
    let Some(doms) = item
        .get("d")
        .and_then(|value| value.get("doms"))
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    for dom in doms {
        if dom.get("contractId").and_then(Value::as_i64) != Some(contract_id) {
            continue;
        }
        let Some(ts_ns) = dom_timestamp_ns(dom) else {
            continue;
        };
        if let Some((start, end)) = timestamp_range {
            let start_ns = start
                .timestamp_nanos_opt()
                .context("DOM capture start is outside supported timestamp range")?;
            let end_ns = end
                .timestamp_nanos_opt()
                .context("DOM capture end is outside supported timestamp range")?;
            if ts_ns < start_ns || ts_ns >= end_ns {
                continue;
            }
        }
        let mut snapshot = ReplayMarketDom {
            ts_ns,
            bids: parse_dom_levels(dom.get("bids"))?,
            asks: parse_dom_levels(dom.get("asks").or_else(|| dom.get("offers")))?,
        };
        if snapshot.bids.is_empty() && snapshot.asks.is_empty() {
            continue;
        }
        normalize_captured_dom(&mut snapshot)?;
        snapshots.push(snapshot);
    }
    Ok(())
}

fn parse_dom_levels(value: Option<&Value>) -> Result<Vec<ReplayDomLevel>> {
    let Some(levels) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::with_capacity(levels.len());
    for level in levels {
        let Some(price) = level.get("price").and_then(Value::as_f64) else {
            continue;
        };
        let Some(size) = level.get("size").and_then(Value::as_f64) else {
            continue;
        };
        if price.is_finite() && price > 0.0 && size.is_finite() && size > 0.0 {
            parsed.push(ReplayDomLevel { price, size });
        }
    }
    Ok(parsed)
}

fn normalize_captured_dom(snapshot: &mut ReplayMarketDom) -> Result<()> {
    if snapshot.ts_ns <= 0 {
        bail!("DOM snapshot timestamp must be positive");
    }
    snapshot.bids.retain(|level| {
        level.price.is_finite()
            && level.price > 0.0
            && level.size.is_finite()
            && level.size > 0.0
    });
    snapshot.asks.retain(|level| {
        level.price.is_finite()
            && level.price > 0.0
            && level.size.is_finite()
            && level.size > 0.0
    });
    snapshot
        .bids
        .sort_by(|left, right| right.price.total_cmp(&left.price));
    snapshot
        .asks
        .sort_by(|left, right| left.price.total_cmp(&right.price));
    if snapshot.bids.is_empty() && snapshot.asks.is_empty() {
        bail!("DOM snapshot contained no positive-size levels");
    }
    Ok(())
}

fn dom_timestamp_ns(dom: &Value) -> Option<i64> {
    let raw = dom.get("timestamp")?;
    if let Some(text) = raw.as_str() {
        return parse_bar_timestamp_ns(text);
    }
    let value = raw.as_i64()?;
    if value.abs() < 100_000_000_000 {
        value.checked_mul(1_000_000_000)
    } else if value.abs() < 100_000_000_000_000 {
        value.checked_mul(1_000_000)
    } else if value.abs() < 100_000_000_000_000_000 {
        value.checked_mul(1_000)
    } else {
        Some(value)
    }
}

fn parse_replay_clock_timestamp(item: &Value) -> Option<i64> {
    if item.get("e").and_then(Value::as_str) != Some("clock") {
        return None;
    }
    let raw = item.get("d")?;
    let clock = raw
        .as_str()
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .or_else(|| raw.as_object().cloned().map(Value::Object))?;
    let timestamp = clock.get("t")?;
    if let Some(text) = timestamp.as_str() {
        parse_bar_timestamp_ns(text)
    } else {
        timestamp.as_i64().and_then(|value| {
            if value.abs() < 100_000_000_000 {
                value.checked_mul(1_000_000_000)
            } else if value.abs() < 100_000_000_000_000 {
                value.checked_mul(1_000_000)
            } else {
                Some(value)
            }
        })
    }
}

fn parse_dom_capture_timestamp(raw: &str, label: &str) -> Result<DateTime<Utc>> {
    let raw = raw.trim();
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(raw) {
        return Ok(timestamp.with_timezone(&Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .map(|value| value.and_utc())
            .with_context(|| format!("parse {label} timestamp {raw}"));
    }
    bail!("parse {label} timestamp {raw} as RFC3339 UTC")
}

fn historical_dom_capture_timeout(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    speed: u16,
) -> Duration {
    let market_seconds = (end - start).num_seconds().max(1) as u64;
    let replay_seconds = market_seconds
        .saturating_mul(100)
        .saturating_div(u64::from(speed).max(1));
    Duration::from_secs(replay_seconds.saturating_add(90).max(120))
}

fn humantime_like_duration(duration: Duration) -> String {
    if duration.as_secs() >= 3600 {
        format!(
            "{}h{}m",
            duration.as_secs() / 3600,
            (duration.as_secs() / 60) % 60
        )
    } else if duration.as_secs() >= 60 {
        format!("{}m{}s", duration.as_secs() / 60, duration.as_secs() % 60)
    } else {
        format!("{}s", duration.as_secs())
    }
}

fn write_dom_jsonl(
    path: &Path,
    snapshots: &[ReplayMarketDom],
    overwrite: bool,
) -> Result<DomCaptureSummary> {
    if snapshots.is_empty() {
        bail!("DOM capture produced no usable snapshots; no output file was written");
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut file = if overwrite {
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
    } else {
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
    }
    .with_context(|| {
        if overwrite {
            format!("open DOM capture output {}", path.display())
        } else {
            format!(
                "create DOM capture output {} (use --overwrite to replace it)",
                path.display()
            )
        }
    })?;
    for snapshot in snapshots {
        serde_json::to_writer(&mut file, snapshot).context("serialize DOM snapshot")?;
        use std::io::Write;
        file.write_all(b"\n")
            .context("write DOM snapshot newline")?;
    }
    file.sync_all()
        .with_context(|| format!("flush DOM capture output {}", path.display()))?;
    Ok(DomCaptureSummary {
        snapshots: snapshots.len(),
        first_ts_ns: snapshots.first().map(|snapshot| snapshot.ts_ns),
        last_ts_ns: snapshots.last().map(|snapshot| snapshot.ts_ns),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dom_parser_maps_offers_to_asks_and_normalizes_levels() {
        let item = serde_json::json!({
            "e": "md",
            "d": {
                "doms": [{
                    "contractId": 42,
                    "timestamp": "2026-07-31T15:00:00.123Z",
                    "bids": [
                        {"price": 100.0, "size": 1.0},
                        {"price": 100.25, "size": 2.0},
                        {"price": 99.75, "size": 0.0}
                    ],
                    "offers": [
                        {"price": 100.75, "size": 3.0},
                        {"price": 100.5, "size": 4.0}
                    ]
                }]
            }
        });
        let mut snapshots = Vec::new();
        collect_dom_snapshots(&item, 42, None, &mut snapshots).expect("parse DOM");
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].bids[0].price, 100.25);
        assert_eq!(snapshots[0].asks[0].price, 100.5);
        assert_eq!(snapshots[0].bids.len(), 2);
    }

    #[test]
    fn dom_parser_filters_contract_and_time_range() {
        let item = serde_json::json!({
            "d": {
                "doms": [
                    {
                        "contractId": 7,
                        "timestamp": "2026-07-31T15:00:00Z",
                        "bids": [{"price": 1.0, "size": 1.0}]
                    },
                    {
                        "contractId": 8,
                        "timestamp": "2026-07-31T15:00:00Z",
                        "bids": [{"price": 1.0, "size": 1.0}]
                    }
                ]
            }
        });
        let start = parse_dom_capture_timestamp("2026-07-31T15:00:00Z", "start").unwrap();
        let end = parse_dom_capture_timestamp("2026-07-31T15:01:00Z", "end").unwrap();
        let mut snapshots = Vec::new();
        collect_dom_snapshots(&item, 8, Some((start, end)), &mut snapshots).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].ts_ns, start.timestamp_nanos_opt().unwrap());
    }

    #[test]
    fn replay_clock_timestamp_accepts_provider_string_payload() {
        let item = serde_json::json!({
            "e": "clock",
            "d": "{\"t\":\"2026-07-31T15:00:00.000Z\",\"s\":400}"
        });
        assert_eq!(
            parse_replay_clock_timestamp(&item),
            Some(
                parse_dom_capture_timestamp("2026-07-31T15:00:00Z", "clock")
                    .unwrap()
                    .timestamp_nanos_opt()
                    .unwrap()
            )
        );
    }
}
