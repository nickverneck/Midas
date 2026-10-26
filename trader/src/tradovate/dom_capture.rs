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

type DomSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type DomRead = futures_util::stream::SplitStream<DomSocket>;
type DomWrite = futures_util::stream::SplitSink<DomSocket, Message>;

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
pub struct HostedReplayQuoteCapture {
    pub contract: ContractSuggestion,
    pub ticks: Vec<crate::replay_cache::ReplayCacheRawTickRow>,
    pub tick_specs: crate::replay_cache::ReplayCacheTickSpecs,
    pub contract_metadata: crate::replay_cache::ReplayCacheContractMetadata,
    pub session_template: Option<String>,
    pub warnings: Vec<String>,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, Default)]
struct HostedQuoteState {
    bid_price: Option<f64>,
    bid_size: Option<f64>,
    ask_price: Option<f64>,
    ask_size: Option<f64>,
}

/// Safe counters for diagnosing a hosted Replay feed without retaining any
/// provider payloads, credentials, or account data.
#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HostedReplayQuoteDiagnostics {
    websocket_array_frames: u64,
    replay_clock_events: u64,
    md_items: u64,
    md_items_without_quote_array: u64,
    quote_rows: u64,
    quote_rows_without_contract_id: u64,
    quote_rows_other_contract: u64,
    matching_contract_rows: u64,
    trade_rows: u64,
    trade_rows_without_valid_values: u64,
    trade_rows_without_timestamp: u64,
    trade_rows_outside_window: u64,
    valid_trade_ticks: u64,
    market_updates: u64,
    closed_bars: u64,
}

#[cfg(feature = "replay")]
impl HostedReplayQuoteDiagnostics {
    fn summary(self) -> String {
        format!(
            "feed diagnostics: array frames={}, clock events={}, md items={}, md items without quote array={}, quote rows={}, rows without contract id={}, rows for other contracts={}, matching rows={}, trade rows={}, invalid trade values={}, missing trade timestamps={}, out-of-window trades={}, valid trade ticks={}, market updates={}, closed bars={}",
            self.websocket_array_frames,
            self.replay_clock_events,
            self.md_items,
            self.md_items_without_quote_array,
            self.quote_rows,
            self.quote_rows_without_contract_id,
            self.quote_rows_other_contract,
            self.matching_contract_rows,
            self.trade_rows,
            self.trade_rows_without_valid_values,
            self.trade_rows_without_timestamp,
            self.trade_rows_outside_window,
            self.valid_trade_ticks,
            self.market_updates,
            self.closed_bars,
        )
    }
}

const DOM_CAPTURE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DOM_CAPTURE_MAX_SPEED: u16 = 400;

/// Prefer the replay service host returned with a user's token when that host
/// is preserved in the token file. Older NinjaTrader browser-token captures
/// do not include apiHosts, so retain the documented shared-host fallback.
fn replay_ws_url(cfg: &AppConfig) -> String {
    let dynamic_host = fs::read_to_string(&cfg.token_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| {
            value
                .pointer("/apiHosts/replay")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|host| {
                    !host.is_empty()
                        && host.chars().all(|character| {
                            character.is_ascii_alphanumeric() || ".-".contains(character)
                        })
                })
                .map(ToString::to_string)
        });
    dynamic_host
        .map(|host| format!("wss://{host}/v1/websocket"))
        .unwrap_or_else(|| cfg.env.replay_ws_url().to_string())
}

/// Validate an interactive NinjaTrader token without changing hosted replay
/// state. In particular, this function never calls replay/initializeclock.
pub async fn validate_ninjatrader_token(cfg: &AppConfig, start: Option<&str>) -> Result<()> {
    if cfg.simulation_proxy.enabled {
        bail!("NinjaTrader token validation is unavailable in simulation_proxy mode");
    }
    let start = start
        .map(|value| parse_dom_capture_timestamp(value, "--start"))
        .transpose()?
        .unwrap_or_else(|| Utc::now() - chrono::Duration::days(1));
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let rest_url = cfg.broker_rest_url();
    let response = client
        .get(format!("{rest_url}/auth/me"))
        .bearer_auth(&tokens.access_token)
        .send()
        .await
        .context("request auth/me for NinjaTrader token validation")?;
    if !response.status().is_success() {
        bail!(
            "NinjaTrader token failed auth/me validation (HTTP {}); response body omitted",
            response.status()
        );
    }

    let replay_url = ninja_replay_ws_url(&tokens)
        .context("NinjaTrader token did not provide its authenticated apiHosts.replay host")?;
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(&replay_url, ws_config)
        .await
        .with_context(|| format!("connect {replay_url}"))?;
    let (mut write, mut read) = ws_stream.split();
    write
        .send(Message::Text(format!(
            "authorize\n1\n\n{}",
            tokens.access_token
        )))
        .await
        .context("authorize NinjaTrader token on replay websocket")?;
    wait_dom_response(&mut read, 1, "replay websocket authorization").await?;

    send_dom_request(
        &mut write,
        "replay/checkreplaysession",
        2,
        &json!({
            "startTimestamp": start.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        }),
    )
    .await?;
    let response = wait_dom_response(&mut read, 2, "check historical replay entitlement").await?;
    let status = response
        .pointer("/d/checkStatus")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(status.to_ascii_lowercase().as_str(), "ok" | "eligible") {
        bail!(
            "token passed auth/me, but Market Replay is unavailable for {} ({})",
            start.to_rfc3339(),
            if status.is_empty() {
                "provider omitted checkStatus"
            } else {
                status
            }
        );
    }

    println!("NinjaTrader token: valid (auth/me)");
    println!(
        "Market Replay entitlement: eligible for {}",
        start.to_rfc3339()
    );
    println!("Hosted replay state: unchanged");
    Ok(())
}

#[cfg(feature = "replay")]
pub async fn capture_hosted_replay_quotes(
    cfg: &AppConfig,
    contract: ContractSuggestion,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    speed: u16,
    on_authenticated: impl FnOnce(),
    mut on_progress: impl FnMut(DateTime<Utc>, usize) + Send,
) -> Result<HostedReplayQuoteCapture> {
    if cfg.simulation_proxy.enabled {
        bail!("hosted Market Replay is unavailable in simulation_proxy mode");
    }
    if start >= end {
        bail!("Market Replay end must be after its start");
    }
    if !(25..=DOM_CAPTURE_MAX_SPEED).contains(&speed) {
        bail!("Market Replay stream speed must be between 25 and {DOM_CAPTURE_MAX_SPEED}");
    }

    let session = prepare_replay_download_session_after_auth(
        cfg,
        &contract.name,
        Some(&contract),
        on_authenticated,
    )
    .await?;
    let replay_url = replay_ws_url(cfg);
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(&replay_url, ws_config)
        .await
        .with_context(|| format!("connect {replay_url}"))?;
    let (mut write, mut read) = ws_stream.split();
    write
        .send(Message::Text(format!(
            "authorize\n1\n\n{}",
            session.replay_access_token()
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
    let check_response =
        wait_dom_response(&mut read, 2, "check historical replay entitlement").await?;
    let check_status = check_response
        .pointer("/d/checkStatus")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(
        check_status.to_ascii_lowercase().as_str(),
        "ok" | "eligible"
    ) {
        bail!(
            "Market Replay is unavailable for {} ({})",
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
            "speed": speed,
            "initialBalance": cfg.replay_initial_capital,
        }),
    )
    .await?;
    let initialized = wait_dom_response(&mut read, 3, "initialize hosted Market Replay").await?;
    if !initialized
        .pointer("/d/ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let error = initialized
            .pointer("/d/errorText")
            .and_then(Value::as_str)
            .unwrap_or("provider did not confirm replay initialization");
        bail!("hosted Market Replay initialization failed: {error}");
    }

    send_dom_request(
        &mut write,
        "md/subscribeQuote",
        4,
        &json!({ "symbol": session.contract().id }),
    )
    .await?;
    wait_dom_response(&mut read, 4, "subscribe to hosted replay quotes").await?;

    let end_ns = end
        .timestamp_nanos_opt()
        .context("Market Replay end is outside supported timestamp range")?;
    let market_seconds = (end - start).num_seconds().max(1) as u64;
    let speed_timeout = market_seconds
        .saturating_mul(100)
        .saturating_div(u64::from(speed).max(1))
        .saturating_mul(2)
        .saturating_add(120);
    let mut deadline = time::Instant::now() + Duration::from_secs(speed_timeout.max(120));
    let mut heartbeat = time::interval(Duration::from_secs(2));
    heartbeat.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    let mut quote_states = std::collections::HashMap::<i64, HostedQuoteState>::new();
    let mut ticks = Vec::new();
    let mut diagnostics = HostedReplayQuoteDiagnostics::default();
    let mut replay_ts_ns = None::<i64>;
    let mut last_progress_ts_ns = None::<i64>;
    let mut replay_speed = Some(speed);
    let mut observed_replay_speed = None;

    loop {
        let remaining = deadline.saturating_duration_since(time::Instant::now());
        if remaining.is_zero() {
            bail!(
                "timed out receiving hosted Market Replay quotes for {}; {}",
                session.contract().name,
                diagnostics.summary(),
            );
        }
        tokio::select! {
            _ = heartbeat.tick() => {
                write.send(Message::Text("[]".to_string())).await.context("send replay websocket heartbeat")?;
            }
            next = time::timeout(remaining, read.next()) => {
                let next = next.context("timed out receiving hosted Market Replay data")?;
                let Some(next) = next else {
                    ensure_hosted_replay_window_complete(
                        replay_ts_ns,
                        end_ns,
                        "provider reached EOF",
                    )?;
                    break;
                };
                let message = next.context("read hosted Market Replay websocket")?;
                let Some(items) = hosted_replay_items_from_message(message)? else {
                    ensure_hosted_replay_window_complete(
                        replay_ts_ns,
                        end_ns,
                        "provider closed the websocket",
                    )?;
                    break;
                };
                diagnostics.websocket_array_frames = diagnostics.websocket_array_frames.saturating_add(1);
                for item in &items {
                    if let Some(clock_ts_ns) = parse_replay_clock_timestamp(item) {
                        diagnostics.replay_clock_events = diagnostics.replay_clock_events.saturating_add(1);
                        replay_ts_ns = Some(replay_ts_ns.map_or(clock_ts_ns, |current| current.max(clock_ts_ns)));
                        if last_progress_ts_ns.is_none_or(|last| clock_ts_ns.saturating_sub(last) >= 30_000_000_000) {
                            on_progress(
                                DateTime::<Utc>::from_timestamp_nanos(clock_ts_ns),
                                ticks.len(),
                            );
                            last_progress_ts_ns = Some(clock_ts_ns);
                        }
                        let clock_speed = parse_replay_clock_speed(item);
                        if let Some(clock_speed) = clock_speed {
                            observed_replay_speed = Some(clock_speed);
                        }
                        replay_speed = clock_speed.or(replay_speed);
                        if let Some(clock_ts_ns) = replay_ts_ns {
                            let left = end_ns.saturating_sub(clock_ts_ns).max(0) as u64 / 1_000_000_000;
                            if let Some(actual_speed) = replay_speed.filter(|value| *value > 0) {
                                let estimate = left.saturating_mul(100).saturating_div(u64::from(actual_speed))
                                    .saturating_add(120);
                                deadline = time::Instant::now() + Duration::from_secs(estimate.max(120));
                            } else {
                                deadline = time::Instant::now() + Duration::from_secs(120);
                            }
                        }
                    }
                    collect_hosted_quote_ticks(
                        item,
                        session.contract().id,
                        start,
                        end,
                        &mut quote_states,
                        &mut ticks,
                        &mut diagnostics,
                    )?;
                }
                if hosted_replay_window_complete(replay_ts_ns, end_ns) {
                    break;
                }
            }
        }
    }

    ensure_hosted_replay_window_complete(replay_ts_ns, end_ns, "capture stopped")?;

    let _ = send_dom_request(
        &mut write,
        "md/unsubscribeQuote",
        5,
        &json!({ "symbol": session.contract().id }),
    )
    .await;
    if ticks.is_empty() {
        bail!(
            "hosted Market Replay produced no sampled last-trade quote updates for {}; {}",
            session.contract().name,
            diagnostics.summary(),
        );
    }
    ticks.sort_by(|left, right| {
        left.ts_ns
            .cmp(&right.ts_ns)
            .then(left.tick_id.cmp(&right.tick_id))
    });
    for (index, tick) in ticks.iter_mut().enumerate() {
        tick.tick_id = Some(i64::try_from(index + 1).unwrap_or(i64::MAX));
    }
    let mut warnings = session.warnings().to_vec();
    warnings.push("Hosted Market Replay md/subscribeQuote last/ls updates are sampled provider quote updates, not a guaranteed complete exchange trade tape.".to_string());
    if let Some(actual_speed) = observed_replay_speed.filter(|actual| *actual != speed) {
        warnings.push(format!(
            "The hosted replay clock reported {actual_speed}% while {speed}% was requested; the provider may limit replay speed."
        ));
    }
    Ok(HostedReplayQuoteCapture {
        contract: session.contract().clone(),
        ticks,
        tick_specs: session.tick_specs().clone(),
        contract_metadata: session.contract_metadata().clone(),
        session_template: session.session_template().map(ToString::to_string),
        warnings,
    })
}

/// Capture historical full-book snapshots through Tradovate Market Replay.
///
/// The replay socket is deliberately separate from the normal engine market
/// socket. Initializing it creates/rewinds a disposable broker replay session,
/// then md/subscribeDOM emits the historical DOM stream as the replay clock
/// advances. No live orders or engine market subscriptions are started.
pub async fn capture_replay_dom(cfg: &AppConfig, options: DomCaptureOptions) -> Result<()> {
    if cfg.simulation_proxy.enabled {
        bail!(
            "DOM capture is unavailable in simulation_proxy mode; use a proxy market fixture instead"
        );
    }
    let start = parse_dom_capture_timestamp(&options.start, "--start")?;
    let end = parse_dom_capture_timestamp(&options.end, "--end")?;
    if start >= end {
        bail!("--end must be after --start");
    }
    if options.speed == 0 || options.speed > DOM_CAPTURE_MAX_SPEED {
        bail!("--speed must be between 1 and {}", DOM_CAPTURE_MAX_SPEED);
    }
    if !options.initial_balance.is_finite() || options.initial_balance < 0.0 {
        bail!("--initial-balance must be finite and non-negative");
    }

    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract =
        resolve_dom_capture_contract(&client, &cfg.env, &tokens.access_token, &options.contract)
            .await?;

    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let replay_url = replay_ws_url(cfg);
    let (ws_stream, _) = connect_low_latency_ws(&replay_url, ws_config)
        .await
        .with_context(|| format!("connect {replay_url}"))?;
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
    let check_response =
        wait_dom_response(&mut read, 2, "check historical replay session entitlement").await?;
    let check_status = check_response
        .pointer("/d/checkStatus")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(
        check_status.to_ascii_lowercase().as_str(),
        "ok" | "eligible"
    ) {
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
        if replay_ts_ns.is_some_and(|ts_ns| ts_ns >= end.timestamp_nanos_opt().unwrap_or(i64::MAX))
        {
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
    if cfg.simulation_proxy.enabled {
        bail!(
            "live DOM capture is unavailable in simulation_proxy mode; disable the proxy before connecting to a broker"
        );
    }
    if options.duration_seconds == 0 {
        bail!("--duration-seconds must be > 0");
    }
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract =
        resolve_dom_capture_contract(&client, &cfg.env, &tokens.access_token, &options.contract)
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
    let rest_url = env.rest_url();
    let suggestions = search_contracts(client, rest_url, token, requested, 25).await?;
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
        level.price.is_finite() && level.price > 0.0 && level.size.is_finite() && level.size > 0.0
    });
    snapshot.asks.retain(|level| {
        level.price.is_finite() && level.price > 0.0 && level.size.is_finite() && level.size > 0.0
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

fn parse_replay_clock_speed(item: &Value) -> Option<u16> {
    if item.get("e").and_then(Value::as_str) != Some("clock") {
        return None;
    }
    let raw = item.get("d")?;
    let clock = raw
        .as_str()
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .or_else(|| raw.as_object().cloned().map(Value::Object))?;
    u16::try_from(clock.get("s")?.as_u64()?).ok()
}

#[cfg(feature = "replay")]
fn hosted_replay_items_from_message(message: Message) -> Result<Option<Vec<Value>>> {
    let raw = match message {
        Message::Text(text) => text.to_string(),
        Message::Binary(bytes) => {
            String::from_utf8(bytes.to_vec()).context("decode hosted replay websocket frame")?
        }
        Message::Close(_) => return Ok(None),
        _ => return Ok(Some(Vec::new())),
    };
    let (frame_type, payload) = parse_frame(&raw);
    match frame_type {
        'a' => {
            let payload = payload.context("malformed hosted Market Replay array frame")?;
            let items = payload
                .as_array()
                .context("malformed hosted Market Replay array frame payload")?;
            Ok(Some(items.clone()))
        }
        'h' | 'o' => Ok(Some(Vec::new())),
        'c' => Ok(None),
        _ => bail!("malformed hosted Market Replay websocket frame"),
    }
}

#[cfg(feature = "replay")]
fn hosted_replay_window_complete(replay_clock_ns: Option<i64>, end_ns: i64) -> bool {
    replay_clock_ns.is_some_and(|timestamp| timestamp >= end_ns)
}

#[cfg(feature = "replay")]
fn ensure_hosted_replay_window_complete(
    replay_clock_ns: Option<i64>,
    end_ns: i64,
    stop_reason: &str,
) -> Result<()> {
    if hosted_replay_window_complete(replay_clock_ns, end_ns) {
        return Ok(());
    }
    let requested_end = DateTime::<Utc>::from_timestamp_nanos(end_ns).to_rfc3339();
    let last_clock = replay_clock_ns
        .map(|timestamp| DateTime::<Utc>::from_timestamp_nanos(timestamp).to_rfc3339())
        .unwrap_or_else(|| "not reported".to_string());
    bail!(
        "hosted Market Replay capture incomplete: {stop_reason} before the requested end-exclusive time {requested_end}; last replay clock was {last_clock}. Partial data was discarded"
    );
}

#[cfg(feature = "replay")]
fn collect_hosted_quote_ticks(
    item: &Value,
    contract_id: i64,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    states: &mut std::collections::HashMap<i64, HostedQuoteState>,
    ticks: &mut Vec<crate::replay_cache::ReplayCacheRawTickRow>,
    diagnostics: &mut HostedReplayQuoteDiagnostics,
) -> Result<()> {
    if item.get("e").and_then(Value::as_str) != Some("md") {
        return Ok(());
    }
    diagnostics.md_items = diagnostics.md_items.saturating_add(1);
    let Some(data) = item.get("d") else {
        diagnostics.md_items_without_quote_array =
            diagnostics.md_items_without_quote_array.saturating_add(1);
        return Ok(());
    };
    let Some(quotes) = data
        .get("quotes")
        .and_then(Value::as_array)
        .or_else(|| data.get("q").and_then(Value::as_array))
    else {
        diagnostics.md_items_without_quote_array =
            diagnostics.md_items_without_quote_array.saturating_add(1);
        return Ok(());
    };
    for quote in quotes {
        diagnostics.quote_rows = diagnostics.quote_rows.saturating_add(1);
        let Some(id) = quote
            .get("contractId")
            .or_else(|| quote.get("id"))
            .and_then(Value::as_i64)
        else {
            diagnostics.quote_rows_without_contract_id =
                diagnostics.quote_rows_without_contract_id.saturating_add(1);
            continue;
        };
        if id != contract_id {
            diagnostics.quote_rows_other_contract =
                diagnostics.quote_rows_other_contract.saturating_add(1);
            continue;
        }
        diagnostics.matching_contract_rows = diagnostics.matching_contract_rows.saturating_add(1);
        let state = states.entry(id).or_default();
        let entries = quote.get("entries");
        let bid = entries.and_then(|entries| entries.get("Bid"));
        let offer = entries.and_then(|entries| entries.get("Offer"));
        state.bid_price = quote
            .get("bid")
            .and_then(Value::as_f64)
            .or_else(|| {
                bid.and_then(|entry| entry.get("price").or_else(|| entry.get("Price")))
                    .and_then(Value::as_f64)
            })
            .filter(|value| value.is_finite() && *value > 0.0)
            .or(state.bid_price);
        state.bid_size = quote
            .get("bs")
            .and_then(Value::as_f64)
            .or_else(|| {
                bid.and_then(|entry| entry.get("size").or_else(|| entry.get("Size")))
                    .and_then(Value::as_f64)
            })
            .filter(|value| value.is_finite() && *value >= 0.0)
            .or(state.bid_size);
        state.ask_price = quote
            .get("ask")
            .and_then(Value::as_f64)
            .or_else(|| {
                offer
                    .and_then(|entry| entry.get("price").or_else(|| entry.get("Price")))
                    .and_then(Value::as_f64)
            })
            .filter(|value| value.is_finite() && *value > 0.0)
            .or(state.ask_price);
        state.ask_size = quote
            .get("as")
            .and_then(Value::as_f64)
            .or_else(|| {
                offer
                    .and_then(|entry| entry.get("size").or_else(|| entry.get("Size")))
                    .and_then(Value::as_f64)
            })
            .filter(|value| value.is_finite() && *value >= 0.0)
            .or(state.ask_size);

        let trade = entries.and_then(|entries| entries.get("Trade"));
        let has_trade_values = if entries.is_some() {
            trade.is_some()
        } else {
            quote.get("last").is_some() || quote.get("ls").is_some()
        };
        if has_trade_values {
            diagnostics.trade_rows = diagnostics.trade_rows.saturating_add(1);
        }
        let (price, size) = if entries.is_some() {
            let Some(trade) = trade else {
                continue;
            };
            (
                trade
                    .get("price")
                    .or_else(|| trade.get("Price"))
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0),
                trade
                    .get("size")
                    .or_else(|| trade.get("Size"))
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0),
            )
        } else {
            (
                quote
                    .get("last")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0),
                quote
                    .get("ls")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value > 0.0),
            )
        };
        let (Some(price), Some(size)) = (price, size) else {
            if has_trade_values {
                diagnostics.trade_rows_without_valid_values =
                    diagnostics.trade_rows_without_valid_values.saturating_add(1);
            }
            continue;
        };
        let raw_timestamp = quote.get("timestamp").or_else(|| quote.get("t"));
        let timestamp = raw_timestamp.and_then(|raw| {
            if let Some(value) = raw.as_str() {
                DateTime::parse_from_rfc3339(value)
                    .ok()
                    .map(|value| value.with_timezone(&Utc))
            } else {
                raw.as_i64().and_then(|value| {
                    let millis = if value.unsigned_abs() < 100_000_000_000 {
                        value.checked_mul(1_000)
                    } else {
                        Some(value)
                    }?;
                    DateTime::<Utc>::from_timestamp_millis(millis)
                })
            }
        });
        let Some(timestamp) = timestamp else {
            diagnostics.trade_rows_without_timestamp =
                diagnostics.trade_rows_without_timestamp.saturating_add(1);
            continue;
        };
        if timestamp < start || timestamp >= end {
            diagnostics.trade_rows_outside_window =
                diagnostics.trade_rows_outside_window.saturating_add(1);
            continue;
        }
        let ts_ns = timestamp
            .timestamp_nanos_opt()
            .context("hosted replay quote timestamp is outside nanosecond range")?;
        let next_id = i64::try_from(ticks.len().saturating_add(1)).unwrap_or(i64::MAX);
        ticks.push(crate::replay_cache::ReplayCacheRawTickRow {
            timestamp,
            ts_ns,
            tick_id: Some(next_id),
            price,
            size,
            bid_price: state.bid_price,
            bid_size: state.bid_size,
            ask_price: state.ask_price,
            ask_size: state.ask_size,
            chart_id: None,
            trade_date: timestamp.format("%Y%m%d").to_string().parse::<i32>().ok(),
            packet_source: Some("hosted_market_replay_quote".to_string()),
            packet_base_ts_ms: None,
            packet_base_price_ticks: None,
        });
        diagnostics.valid_trade_ticks = diagnostics.valid_trade_ticks.saturating_add(1);
    }
    Ok(())
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
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
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

    #[cfg(feature = "replay")]
    #[test]
    fn hosted_replay_quote_parser_builds_sampled_raw_tick_with_quote_context() {
        let item = serde_json::json!({
            "e": "md",
            "d": {
                "q": [
                    {"id": 99, "t": 1793466000123_i64, "last": 5400.25, "ls": 3.0,
                     "bid": 5400.0, "bs": 12.0, "ask": 5400.25, "as": 7.0},
                    {"id": 100, "t": 1793466000123_i64, "last": 1.0, "ls": 1.0}
                ]
            }
        });
        let start = DateTime::<Utc>::from_timestamp_millis(1793466000000).unwrap();
        let end = start + chrono::Duration::minutes(1);
        let mut states = std::collections::HashMap::new();
        let mut ticks = Vec::new();
        let mut diagnostics = HostedReplayQuoteDiagnostics::default();
        collect_hosted_quote_ticks(
            &item,
            99,
            start,
            end,
            &mut states,
            &mut ticks,
            &mut diagnostics,
        )
            .expect("collect quote tick");
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].price, 5400.25);
        assert_eq!(ticks[0].size, 3.0);
        assert_eq!(ticks[0].bid_price, Some(5400.0));
        assert_eq!(ticks[0].ask_price, Some(5400.25));
        assert_eq!(ticks[0].tick_id, Some(1));
        assert_eq!(
            ticks[0].packet_source.as_deref(),
            Some("hosted_market_replay_quote")
        );
        assert_eq!(diagnostics.md_items, 1);
        assert_eq!(diagnostics.quote_rows, 2);
        assert_eq!(diagnostics.matching_contract_rows, 1);
        assert_eq!(diagnostics.quote_rows_other_contract, 1);
        assert_eq!(diagnostics.valid_trade_ticks, 1);
    }

    #[cfg(feature = "replay")]
    #[test]
    fn hosted_replay_quote_parser_accepts_official_quotes_entries_and_rfc3339_timestamps() {
        let start = parse_dom_capture_timestamp("2026-07-31T15:00:00Z", "start").unwrap();
        let end = start + chrono::Duration::minutes(1);
        let mut states = std::collections::HashMap::new();
        let mut ticks = Vec::new();
        let mut diagnostics = HostedReplayQuoteDiagnostics::default();
        let bid_offer_update = serde_json::json!({
            "e": "md",
            "d": {"quotes": [{
                "contractId": 99,
                "timestamp": "2026-07-31T15:00:00.100000000Z",
                "entries": {
                    "Bid": {"price": 5400.0, "size": 12.0},
                    "Offer": {"price": 5400.25, "size": 7.0}
                }
            }]}
        });
        collect_hosted_quote_ticks(
            &bid_offer_update,
            99,
            start,
            end,
            &mut states,
            &mut ticks,
            &mut diagnostics,
        )
        .expect("collect bid/offer context");
        assert!(ticks.is_empty(), "bid/offer updates are not trade ticks");

        let trade_update = serde_json::json!({
            "e": "md",
            "d": {"quotes": [
                {
                    "contractId": 99,
                    "timestamp": "2026-07-31T11:00:00.123456789-04:00",
                    "entries": {"Trade": {"price": 5400.25, "size": 3.0}}
                },
                {
                    "contractId": 100,
                    "timestamp": "2026-07-31T15:00:00.123456789Z",
                    "entries": {"Trade": {"Price": 1.0, "Size": 1.0}}
                },
                {
                    "contractId": 99,
                    "timestamp": "2026-07-31T15:00:00.223456789Z",
                    "entries": {
                        "Bid": {"Price": 5400.25, "Size": 10.0},
                        "Offer": {"Price": 5400.5, "Size": 8.0},
                        "Trade": {"Price": 5400.5, "Size": 2.0}
                    }
                }
            ]}
        });
        collect_hosted_quote_ticks(
            &trade_update,
            99,
            start,
            end,
            &mut states,
            &mut ticks,
            &mut diagnostics,
        )
        .expect("collect official trade tick");

        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].price, 5400.25);
        assert_eq!(ticks[0].size, 3.0);
        assert_eq!(
            ticks[0].ts_ns,
            parse_dom_capture_timestamp("2026-07-31T15:00:00.123456789Z", "quote")
                .unwrap()
                .timestamp_nanos_opt()
                .unwrap()
        );
        assert_eq!(ticks[0].bid_price, Some(5400.0));
        assert_eq!(ticks[0].bid_size, Some(12.0));
        assert_eq!(diagnostics.quote_rows, 4);
        assert_eq!(diagnostics.matching_contract_rows, 3);
        assert_eq!(diagnostics.quote_rows_other_contract, 1);
        assert_eq!(diagnostics.trade_rows, 2);
        assert_eq!(diagnostics.trade_rows_outside_window, 0);
        assert_eq!(diagnostics.valid_trade_ticks, 2);
        assert_eq!(ticks[0].ask_price, Some(5400.25));
        assert_eq!(ticks[0].ask_size, Some(7.0));
        assert_eq!(ticks[1].price, 5400.5);
        assert_eq!(ticks[1].size, 2.0);
        assert_eq!(ticks[1].bid_price, Some(5400.25));
        assert_eq!(ticks[1].bid_size, Some(10.0));
        assert_eq!(ticks[1].ask_price, Some(5400.5));
        assert_eq!(ticks[1].ask_size, Some(8.0));
    }

    #[cfg(feature = "replay")]
    #[test]
    fn hosted_replay_window_requires_clock_at_or_past_exclusive_end() {
        let end = parse_dom_capture_timestamp("2026-07-31T15:01:00Z", "end")
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        assert!(!hosted_replay_window_complete(None, end));
        assert!(!hosted_replay_window_complete(Some(end - 1), end));
        assert!(hosted_replay_window_complete(Some(end), end));
        assert!(hosted_replay_window_complete(Some(end + 1), end));

        let error = ensure_hosted_replay_window_complete(
            Some(end - 1),
            end,
            "provider closed the websocket",
        )
        .expect_err("an early close cannot complete a hosted capture")
        .to_string();
        assert!(error.contains("capture incomplete"));
        assert!(error.contains("end-exclusive time"));
        assert!(error.contains("provider closed the websocket"));
    }

    #[cfg(feature = "replay")]
    #[test]
    fn hosted_replay_stream_parser_rejects_malformed_array_frames() {
        let error = hosted_replay_items_from_message(Message::Text("a{".into()))
            .expect_err("malformed stream must fail before caching")
            .to_string();
        assert!(error.contains("malformed hosted Market Replay array frame"));
    }

    #[test]
    fn replay_clock_speed_accepts_provider_string_payload() {
        let item = serde_json::json!({
            "e": "clock",
            "d": "{\"t\":\"2026-07-31T15:00:00.000Z\",\"s\":200}"
        });
        assert_eq!(parse_replay_clock_speed(&item), Some(200));
    }

    #[test]
    fn replay_socket_uses_dynamic_host_from_token_file_when_available() {
        let path =
            std::env::temp_dir().join(format!("trader-replay-host-{}.json", std::process::id()));
        fs::write(
            &path,
            r#"{"token":"redacted-test-token","apiHosts":{"replay":"replay.org.example"}}"#,
        )
        .expect("write token host fixture");
        let mut config = AppConfig::default();
        config.token_path = path.clone();
        assert_eq!(
            replay_ws_url(&config),
            "wss://replay.org.example/v1/websocket"
        );
        fs::remove_file(path).expect("remove token host fixture");
    }
}
