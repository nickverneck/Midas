use super::*;

use std::collections::{BTreeMap, HashMap};

type ReplaySocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type ReplayRead = futures_util::stream::SplitStream<ReplaySocket>;
type ReplayWrite = futures_util::stream::SplitSink<ReplaySocket, Message>;

const HOSTED_REPLAY_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
// Startup consists of several individually bounded provider requests. Keep
// the overall guard longer than those per-request deadlines so it cannot
// mask which handshake stage actually timed out.
const HOSTED_REPLAY_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const HOSTED_REPLAY_MAX_SPEED: u16 = NinjaTraderReplaySpeed::MAX;

pub(crate) struct HostedReplayOptions {
    pub(crate) config: AppConfig,
    pub(crate) generation: u64,
    pub(crate) contract: ContractSuggestion,
    pub(crate) start: DateTime<Utc>,
    pub(crate) end: Option<DateTime<Utc>>,
    pub(crate) speed: NinjaTraderReplaySpeed,
    pub(crate) initial_balance: f64,
    pub(crate) bar_type: BarType,
    pub(crate) candle_mode: CandleMode,
}

pub(crate) struct HostedReplayStarted {
    pub(crate) accounts: Vec<AccountInfo>,
    pub(crate) market_specs: MarketSpecs,
    pub(crate) request_tx: UserSocketCommandSender,
    pub(crate) task: JoinHandle<()>,
    pub(crate) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(crate) tokens: TokenBundle,
}

/// Initialize and join a hosted replay session. The worker performs all
/// provider operations on the authenticated replay WebSocket and waits for
/// the replay-session user sync before returning the ephemeral account.
pub(crate) async fn start_hosted_replay(
    client: &Client,
    options: HostedReplayOptions,
    internal_tx: InternalEventSender,
) -> Result<HostedReplayStarted> {
    validate_options(&options)?;
    options.config.validate()?;
    if options.config.broker != BrokerKind::Tradovate {
        bail!("NinjaTrader hosted Replay requires the Tradovate broker configuration");
    }
    if options.config.simulation_proxy.enabled {
        bail!("NinjaTrader hosted Replay is unavailable in simulation_proxy mode");
    }

    let tokens = authenticate(client, &options.config).await?;
    let market_specs = time::timeout(
        HOSTED_REPLAY_REQUEST_TIMEOUT,
        fetch_contract_specs(
            client,
            &options.config.broker_rest_url(),
            &tokens.access_token,
            &options.contract,
        ),
    )
    .await
    .context("timed out loading contract specifications for NinjaTrader hosted Replay")??;
    let market_specs = validate_replay_market_specs(market_specs)?;
    let replay_ws_url = ninja_replay_ws_url(&tokens)?;
    let user_id = tokens
        .user_id
        .context("NinjaTrader authentication did not provide a userId for Replay user sync")?;
    if user_id <= 0 {
        bail!("NinjaTrader authentication returned an invalid userId");
    }
    save_token_cache(&options.config.session_cache_path, &tokens)?;

    let (request_tx, request_rx) = user_socket_command_channel();
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let (startup_tx, startup_rx) = oneshot::channel::<Result<Vec<AccountInfo>, String>>();
    let task_tokens = tokens.clone();
    let task_internal_tx = internal_tx.clone();
    let task_market_specs = market_specs;
    let generation = options.generation;
    let task = tokio::spawn(async move {
        hosted_replay_worker(
            replay_ws_url,
            task_tokens,
            user_id,
            options,
            request_rx,
            stop_rx,
            startup_tx,
            task_internal_tx,
            generation,
            task_market_specs,
        )
        .await;
    });

    let accounts = match time::timeout(HOSTED_REPLAY_STARTUP_TIMEOUT, startup_rx).await {
        Ok(Ok(Ok(accounts))) => accounts,
        Ok(Ok(Err(message))) => {
            let _ = stop_tx.send(true);
            let _ = task.await;
            bail!(message);
        }
        Ok(Err(_)) => {
            let _ = stop_tx.send(true);
            let _ = task.await;
            bail!("NinjaTrader Replay closed before its session account sync completed");
        }
        Err(_) => {
            let _ = stop_tx.send(true);
            task.abort();
            let _ = task.await;
            bail!("timed out waiting for the NinjaTrader Replay session account sync");
        }
    };

    Ok(HostedReplayStarted {
        accounts,
        market_specs,
        request_tx,
        task,
        stop_tx,
        tokens,
    })
}

fn validate_options(options: &HostedReplayOptions) -> Result<()> {
    if options.start.timestamp_nanos_opt().is_none() {
        bail!("NinjaTrader Replay start is outside the supported timestamp range");
    }
    if options
        .end
        .is_some_and(|end| end <= options.start || end.timestamp_nanos_opt().is_none())
    {
        bail!("NinjaTrader Replay end must be after its start");
    }
    if options.speed.percent() > HOSTED_REPLAY_MAX_SPEED {
        bail!("NinjaTrader Replay speed must be between 0 and {HOSTED_REPLAY_MAX_SPEED}");
    }
    if !options.initial_balance.is_finite() || options.initial_balance < 0.0 {
        bail!("NinjaTrader Replay initial balance must be finite and non-negative");
    }
    if options.contract.id <= 0 || options.contract.name.trim().is_empty() {
        bail!("NinjaTrader Replay requires a valid selected contract");
    }
    Ok(())
}

async fn hosted_replay_worker(
    replay_ws_url: String,
    tokens: TokenBundle,
    user_id: i64,
    options: HostedReplayOptions,
    mut request_rx: UserSocketCommandReceiver,
    mut stop_rx: tokio::sync::watch::Receiver<bool>,
    startup_tx: oneshot::Sender<Result<Vec<AccountInfo>, String>>,
    internal_tx: InternalEventSender,
    generation: u64,
    market_specs: MarketSpecs,
) {
    let mut startup_tx = Some(startup_tx);
    let result = hosted_replay_worker_inner(
        &replay_ws_url,
        &tokens,
        user_id,
        &options,
        &mut request_rx,
        &mut stop_rx,
        &mut startup_tx,
        &internal_tx,
        generation,
        market_specs,
    )
    .await;

    match result {
        Ok(HostedReplayExit::Stopped) => {}
        Ok(HostedReplayExit::Ended(reason)) => {
            let event = HostedReplayExit::Ended(reason)
                .into_internal_event(generation)
                .expect("an ended Replay exit produces an end event");
            let _ = internal_tx.send(event);
        }
        Ok(HostedReplayExit::Failed(reason)) => {
            if let Some(startup_tx) = startup_tx.take() {
                let _ = startup_tx.send(Err(reason));
            } else {
                let event = HostedReplayExit::Failed(reason)
                    .into_internal_event(generation)
                    .expect("a failed Replay exit produces an end event");
                let _ = internal_tx.send(event);
            }
        }
        Err(err) => {
            let message = format!("NinjaTrader hosted Replay: {err}");
            if let Some(startup_tx) = startup_tx.take() {
                let _ = startup_tx.send(Err(message));
            } else {
                let _ = internal_tx.send(InternalEvent::HostedReplayEnded {
                    generation,
                    reason: message,
                    failed: true,
                });
            }
        }
    }
}

enum HostedReplayExit {
    Stopped,
    Ended(String),
    Failed(String),
}

impl HostedReplayExit {
    fn into_internal_event(self, generation: u64) -> Option<InternalEvent> {
        let (reason, failed) = match self {
            Self::Stopped => return None,
            Self::Ended(reason) => (reason, false),
            Self::Failed(reason) => (reason, true),
        };
        Some(InternalEvent::HostedReplayEnded {
            generation,
            reason,
            failed,
        })
    }
}

async fn hosted_replay_worker_inner(
    replay_ws_url: &str,
    tokens: &TokenBundle,
    user_id: i64,
    options: &HostedReplayOptions,
    request_rx: &mut UserSocketCommandReceiver,
    stop_rx: &mut tokio::sync::watch::Receiver<bool>,
    startup_tx: &mut Option<oneshot::Sender<Result<Vec<AccountInfo>, String>>>,
    internal_tx: &InternalEventSender,
    generation: u64,
    market_specs: MarketSpecs,
) -> Result<HostedReplayExit> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: MAX_WEBSOCKET_WRITE_BUFFER_SIZE,
        max_message_size: Some(MAX_WEBSOCKET_MESSAGE_SIZE),
        max_frame_size: Some(MAX_WEBSOCKET_MESSAGE_SIZE),
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(replay_ws_url, ws_config)
        .await
        .with_context(|| {
            format!("connect authenticated NinjaTrader Replay host {replay_ws_url}")
        })?;
    let (mut write, mut read) = ws_stream.split();

    send_authorize(&mut write, 1, &tokens.access_token).await?;
    wait_response(&mut read, 1, "authorize NinjaTrader Replay WebSocket").await?;

    let start_timestamp = options
        .start
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    send_replay_request(
        &mut write,
        "replay/checkreplaysession",
        2,
        None,
        Some(&json!({ "startTimestamp": start_timestamp })),
    )
    .await?;
    let entitlement = wait_response(&mut read, 2, "check NinjaTrader Replay entitlement").await?;
    let check_status = entitlement
        .pointer("/d/checkStatus")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(
        check_status.to_ascii_lowercase().as_str(),
        "ok" | "eligible"
    ) {
        bail!(
            "NinjaTrader Market Replay entitlement unavailable for {}",
            options.start
        );
    }

    send_replay_request(
        &mut write,
        "replay/initializeclock",
        3,
        None,
        Some(&json!({
            "startTimestamp": start_timestamp,
            "speed": options.speed.percent(),
            "initialBalance": options.initial_balance,
        })),
    )
    .await?;
    let initialized = wait_response(&mut read, 3, "initialize NinjaTrader Replay clock").await?;
    if !initialized
        .pointer("/d/ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let message = initialized
            .pointer("/d/errorText")
            .and_then(Value::as_str)
            .unwrap_or("provider did not confirm replay initialization");
        bail!("NinjaTrader Replay clock initialization failed: {message}");
    }

    send_replay_request(
        &mut write,
        "md/subscribeQuote",
        4,
        None,
        Some(&json!({ "symbol": options.contract.id })),
    )
    .await?;
    wait_response(&mut read, 4, "subscribe to NinjaTrader Replay quotes").await?;
    let entity_types = [
        "account",
        "accountRiskStatus",
        "cashBalance",
        "position",
        "order",
        "command",
        "commandReport",
        "orderStrategy",
        "orderStrategyLink",
        "executionReport",
        "fill",
        "fillFee",
    ];
    send_replay_request(
        &mut write,
        "user/syncrequest",
        5,
        None,
        Some(&json!({
            "users": [user_id],
            "entityTypes": entity_types,
            "splitResponses": false,
        })),
    )
    .await?;

    internal_tx
        .send(InternalEvent::UserSocketStatus(format!(
            "Waiting for Replay quotes for {}; stream subscribed from {}; clock paused at 0%. Choose and arm an execution strategy to advance it (configured speed: {}%).",
            options.contract.name,
            options.start.to_rfc3339(),
            options.speed.percent()
        )))
        .map_err(|error| anyhow::anyhow!("replay status delivery failed: {error:?}"))?;

    let mut quote_state = HostedQuoteState::default();
    let mut diagnostics = HostedReplayQuoteDiagnostics::default();
    let mut bars = HostedBarAggregator::new(options.bar_type);
    let mut series = LiveSeries::new();
    let mut live_bars = 0usize;
    let mut latest_clock_ns = None;
    let end_ns = options.end.and_then(|end| end.timestamp_nanos_opt());
    let mut sync_id = Some(5_u64);
    let mut startup_completed = false;
    let mut message_id = 5_u64;
    let mut pending_requests = HashMap::<u64, oneshot::Sender<Result<Value, String>>>::new();
    let mut heartbeat = time::interval(Duration::from_millis(options.config.heartbeat_ms.max(250)));
    heartbeat.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    let mut quote_status = time::interval(Duration::from_secs(20));
    quote_status.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    quote_status.tick().await;
    let request_deadline = time::Instant::now() + HOSTED_REPLAY_REQUEST_TIMEOUT;

    loop {
        tokio::select! {
            biased;
            changed = stop_rx.changed() => {
                if changed.is_err() || *stop_rx.borrow() {
                    unsubscribe_quote(&mut write, options.contract.id).await;
                    return Ok(HostedReplayExit::Stopped);
                }
            }
            outbound = request_rx.recv() => {
                let Some(outbound) = outbound else {
                    unsubscribe_quote(&mut write, options.contract.id).await;
                    return Ok(HostedReplayExit::Stopped);
                };
                if !startup_completed {
                    let _ = outbound.response_tx.send(Err("NinjaTrader Replay account sync is not ready".to_string()));
                    continue;
                }
                if pending_requests.len() >= MAX_PENDING_USER_REQUESTS {
                    let _ = outbound.response_tx.send(Err(format!(
                        "NinjaTrader Replay WebSocket backpressure: {} requests pending",
                        MAX_PENDING_USER_REQUESTS
                    )));
                    continue;
                }
                message_id = message_id.saturating_add(1);
                let request_id = message_id;
                pending_requests.insert(request_id, outbound.response_tx);
                if let Err(error) = write.send(Message::Text(create_message(
                    &outbound.endpoint,
                    request_id,
                    outbound.query.as_deref(),
                    outbound.body.as_ref(),
                ))).await {
                    if let Some(response_tx) = pending_requests.remove(&request_id) {
                        let _ = response_tx.send(Err(format!("NinjaTrader Replay WebSocket send failed: {error}")));
                    }
                    bail!("send provider request on NinjaTrader Replay WebSocket failed: {error}");
                }
            }
            _ = heartbeat.tick() => {
                write.send(Message::Text("[]".to_string())).await
                    .context("send NinjaTrader Replay WebSocket heartbeat")?;
            }
            _ = quote_status.tick(), if diagnostics.market_updates == 0 => {
                let _ = internal_tx.send(InternalEvent::UserSocketStatus(format!(
                    "Waiting for Replay quotes; {}",
                    diagnostics.summary()
                )));
            }
            _ = time::sleep_until(request_deadline), if !startup_completed => {
                bail!("timed out waiting for the NinjaTrader Replay session user/syncRequest account response");
            }
            next = read.next() => {
                let raw = match next {
                    Some(Ok(Message::Text(text))) if text.len() <= MAX_WEBSOCKET_MESSAGE_SIZE => text.to_string(),
                    Some(Ok(Message::Binary(bytes))) if bytes.len() <= MAX_WEBSOCKET_MESSAGE_SIZE => {
                        String::from_utf8(bytes.to_vec()).context("decode NinjaTrader Replay WebSocket frame")?
                    }
                    Some(Ok(Message::Text(_))) | Some(Ok(Message::Binary(_))) => {
                        bail!("NinjaTrader Replay WebSocket message exceeded the configured size limit")
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        return Ok(HostedReplayExit::Failed(
                            "provider closed the Replay WebSocket unexpectedly".to_string(),
                        ));
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(error)) => bail!("read NinjaTrader Replay WebSocket failed: {error}"),
                };
                let (frame_type, payload) = parse_frame(&raw);
                if matches!(frame_type, 'h' | 'o') {
                    continue;
                }
                if frame_type == 'c' {
                    return Ok(HostedReplayExit::Failed(
                        "provider sent a Replay WebSocket close frame unexpectedly".to_string(),
                    ));
                }
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    bail!("malformed NinjaTrader Replay WebSocket array frame");
                };
                diagnostics.websocket_array_frames = diagnostics.websocket_array_frames.saturating_add(1);

                let mut sync_entities = Vec::<EntityEnvelope>::new();

                for item in items {
                    diagnostics.record_event_kind(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);
                    let status = parse_status_code(&item);
                    if let Some(clock) = parse_hosted_clock(&item) {
                        diagnostics.replay_clock_events = diagnostics.replay_clock_events.saturating_add(1);
                        latest_clock_ns = Some(latest_clock_ns.map_or(clock.0.timestamp_nanos_opt().unwrap_or(0), |current: i64| current.max(clock.0.timestamp_nanos_opt().unwrap_or(0))));
                        let _ = internal_tx.send(InternalEvent::HostedReplayClock {
                            generation,
                            timestamp: clock.0,
                            speed: clock.1,
                        });
                        if end_ns.is_some_and(|end_ns| latest_clock_ns.is_some_and(|current| current >= end_ns)) {
                            unsubscribe_quote(&mut write, options.contract.id).await;
                            return Ok(HostedReplayExit::Ended(format!(
                                "reached optional end time {}",
                                options.end.expect("end_ns implies end").to_rfc3339()
                            )));
                        }
                    }

                    if matches_pending_sync_response(response_id, sync_id) {
                        if status.is_some_and(|code| !(200..300).contains(&code)) {
                            let detail = parse_socket_response(&item).expect_err("non-success sync response");
                            bail!("NinjaTrader Replay user/syncRequest failed: {detail}");
                        }
                        let chunks = extract_admitted_entity_chunks(&item)?;
                        for chunk in chunks {
                            for entity in chunk {
                                sync_entities.push(entity);
                            }
                        }
                        if !startup_completed && status.is_some_and(|code| (200..300).contains(&code)) {
                            // This initial response comes from the dedicated Replay host after
                            // initializeclock. Only bind its unique account, and verify it is
                            // owned by the authenticated user. Do not infer identity from its
                            // name; the API does not document an account-name prefix.
                            let accounts = hosted_replay_session_accounts(&sync_entities, user_id)?;
                            let Some(startup_tx) = startup_tx.take() else {
                                bail!("NinjaTrader Replay startup handshake was already completed");
                            };
                            startup_tx.send(Ok(accounts)).map_err(|_| anyhow::anyhow!("NinjaTrader Replay startup receiver closed"))?;
                            startup_completed = true;
                        }
                        if !sync_entities.is_empty() {
                            send_user_entity_chunks(internal_tx, vec![std::mem::take(&mut sync_entities)]).await?;
                        }
                        sync_id = None;
                        continue;
                    }

                    if let Some(request_id) = response_id {
                        if let Some(response_tx) = pending_requests.remove(&request_id) {
                            let _ = response_tx.send(parse_socket_response(&item));
                            continue;
                        }
                    }

                    let chunks = extract_admitted_entity_chunks(&item)?;
                    if !chunks.is_empty() {
                        send_user_entity_chunks(internal_tx, chunks).await?;
                    }

                    if let Some(mut quote_ticks) = parse_hosted_quote_ticks(
                        &item,
                        options.contract.id,
                        options.start,
                        options.end,
                        &mut quote_state,
                        &mut diagnostics,
                    )? {
                        for tick in quote_ticks.drain(..) {
                            let before_closed_len = series.closed_bars.len();
                            let before_last_closed = series.closed_bars.last().cloned();
                            let before_forming = series.forming_bar.clone();
                            if let Some((closed, forming)) = bars.push_tick(
                                tick.ts_ns,
                                tick.last,
                                tick.size.unwrap_or(0.0),
                            ) {
                                if let Some(closed) = closed {
                                    series.push_closed_bar_capped(&closed, ENGINE_MARKET_BAR_LIMIT);
                                    live_bars = live_bars.saturating_add(1);
                                    diagnostics.closed_bars = diagnostics.closed_bars.saturating_add(1);
                                }
                                series.forming_bar = forming;
                                if let Some(update) = build_market_update(
                                    &options.contract,
                                    Some(market_specs),
                                    options.candle_mode,
                                    series.closed_bars.len(),
                                    live_bars,
                                    format!("NinjaTrader Replay streaming {}", options.contract.name),
                                    before_closed_len,
                                    before_last_closed.clone(),
                                    before_forming.clone(),
                                    &series,
                                ) {
                                    emit_hosted_replay_market(update, internal_tx).await?;
                                    diagnostics.market_updates = diagnostics.market_updates.saturating_add(1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn hosted_replay_session_accounts(
    entities: &[EntityEnvelope],
    user_id: i64,
) -> Result<Vec<AccountInfo>> {
    let mut accounts = BTreeMap::<i64, AccountInfo>::new();
    for entity in entities
        .iter()
        .filter(|entity| entity.entity_type.eq_ignore_ascii_case("account") && !entity.deleted)
    {
        let Some(id) = json_i64(&entity.entity, "id") else {
            bail!(
                "NinjaTrader Replay session sync returned an account without an id; refusing to bind it"
            );
        };
        if json_i64(&entity.entity, "userId") != Some(user_id) {
            bail!(
                "NinjaTrader Replay session sync returned an account not owned by the authenticated user; refusing to bind it"
            );
        }
        let Some(name) = entity.entity.get("name").and_then(Value::as_str) else {
            bail!(
                "NinjaTrader Replay session sync returned an account without a name; refusing to bind it"
            );
        };
        accounts.insert(
            id,
            AccountInfo {
                id,
                name: name.to_string(),
                raw: entity.entity.clone(),
            },
        );
    }
    if accounts.len() != 1 {
        bail!(
            "NinjaTrader Replay session sync returned {} account candidates; expected exactly one ephemeral Replay account",
            accounts.len()
        );
    }
    Ok(accounts.into_values().collect())
}

async fn send_authorize(write: &mut ReplayWrite, request_id: u64, token: &str) -> Result<()> {
    write
        .send(Message::Text(format!("authorize\n{request_id}\n\n{token}")))
        .await
        .context("authorize NinjaTrader Replay WebSocket")
}

async fn send_replay_request(
    write: &mut ReplayWrite,
    endpoint: &str,
    request_id: u64,
    query: Option<&str>,
    body: Option<&Value>,
) -> Result<()> {
    let frame = create_message(endpoint, request_id, query, body);
    write
        .send(Message::Text(frame))
        .await
        .with_context(|| format!("send {endpoint} on NinjaTrader Replay WebSocket"))
}

async fn wait_response(read: &mut ReplayRead, request_id: u64, operation: &str) -> Result<Value> {
    let deadline = time::Instant::now() + HOSTED_REPLAY_REQUEST_TIMEOUT;
    loop {
        let next = time::timeout_at(deadline, read.next())
            .await
            .with_context(|| format!("timed out during {operation}"))?;
        let Some(next) = next else {
            bail!("NinjaTrader Replay WebSocket closed during {operation}");
        };
        let message =
            next.with_context(|| format!("read NinjaTrader Replay WebSocket during {operation}"))?;
        let Some(items) = replay_items_from_message(message)? else {
            bail!("NinjaTrader Replay WebSocket closed during {operation}");
        };
        for item in items {
            if item.get("i").and_then(Value::as_u64) != Some(request_id) {
                continue;
            }
            let status = parse_status_code(&item);
            if !status.is_some_and(|code| (200..300).contains(&code)) {
                let detail = parse_socket_response(&item).expect_err("non-success response");
                bail!("{operation} failed: {detail}");
            }
            return Ok(item);
        }
    }
}

fn replay_items_from_message(message: Message) -> Result<Option<Vec<Value>>> {
    let raw = match message {
        Message::Text(text) => text.to_string(),
        Message::Binary(bytes) => {
            String::from_utf8(bytes.to_vec()).context("decode Replay WebSocket frame")?
        }
        Message::Close(_) => return Ok(None),
        _ => return Ok(Some(Vec::new())),
    };
    let (frame_type, payload) = parse_frame(&raw);
    match frame_type {
        'a' => Ok(payload.and_then(|payload| payload.as_array().cloned())),
        'h' | 'o' => Ok(Some(Vec::new())),
        'c' => Ok(None),
        _ => bail!("malformed NinjaTrader Replay WebSocket frame"),
    }
}

async fn unsubscribe_quote(write: &mut ReplayWrite, contract_id: i64) {
    let body = quote_unsubscribe_body(contract_id);
    let _ = write
        .send(Message::Text(create_message(
            "md/unsubscribeQuote",
            6,
            None,
            Some(&body),
        )))
        .await;
}

fn quote_unsubscribe_body(contract_id: i64) -> Value {
    json!({ "symbol": contract_id })
}

async fn emit_hosted_replay_market(
    update: MarketUpdate,
    internal_tx: &InternalEventSender,
) -> Result<()> {
    let (response_tx, response_rx) = oneshot::channel();
    internal_tx
        .send(InternalEvent::ReplayMarket {
            update,
            response_tx,
        })
        .map_err(|error| {
            anyhow::anyhow!("NinjaTrader Replay market event delivery failed: {error:?}")
        })?;
    response_rx
        .await
        .context("NinjaTrader Replay market event handler stopped")?
        .map_err(anyhow::Error::msg)
}

#[derive(Default)]
struct HostedQuoteState {
    bid_price: Option<f64>,
    bid_size: Option<f64>,
    ask_price: Option<f64>,
    ask_size: Option<f64>,
}

fn parse_hosted_quote_ticks(
    item: &Value,
    contract_id: i64,
    start: DateTime<Utc>,
    end: Option<DateTime<Utc>>,
    state: &mut HostedQuoteState,
    diagnostics: &mut HostedReplayQuoteDiagnostics,
) -> Result<Option<Vec<ReplayMarketTick>>> {
    if item.get("e").and_then(Value::as_str) != Some("md") {
        return Ok(None);
    }
    diagnostics.md_items = diagnostics.md_items.saturating_add(1);
    let Some(quotes) = item
        .pointer("/d/quotes")
        .and_then(Value::as_array)
        .or_else(|| item.pointer("/d/q").and_then(Value::as_array))
    else {
        diagnostics.md_items_without_quote_array =
            diagnostics.md_items_without_quote_array.saturating_add(1);
        return Ok(Some(Vec::new()));
    };
    let mut ticks = Vec::new();
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
        let price = if entries.is_some() {
            trade
                .and_then(|entry| entry.get("price").or_else(|| entry.get("Price")))
                .and_then(Value::as_f64)
        } else {
            quote.get("last").and_then(Value::as_f64)
        }
        .filter(|value| value.is_finite() && *value > 0.0);
        let size = if entries.is_some() {
            trade
                .and_then(|entry| entry.get("size").or_else(|| entry.get("Size")))
                .and_then(Value::as_f64)
        } else {
            quote.get("ls").and_then(Value::as_f64)
        }
        .filter(|value| value.is_finite() && *value > 0.0);
        let (Some(price), Some(size)) = (price, size) else {
            if has_trade_values {
                diagnostics.trade_rows_without_valid_values = diagnostics
                    .trade_rows_without_valid_values
                    .saturating_add(1);
            }
            continue;
        };
        let timestamp = quote
            .get("timestamp")
            .or_else(|| quote.get("t"))
            .and_then(parse_quote_timestamp);
        let Some(timestamp) = timestamp else {
            diagnostics.trade_rows_without_timestamp =
                diagnostics.trade_rows_without_timestamp.saturating_add(1);
            continue;
        };
        if timestamp < start || end.is_some_and(|end| timestamp >= end) {
            diagnostics.trade_rows_outside_window =
                diagnostics.trade_rows_outside_window.saturating_add(1);
            continue;
        }
        let ts_ns = timestamp
            .timestamp_nanos_opt()
            .context("Replay quote timestamp is out of range")?;
        ticks.push(ReplayMarketTick {
            ts_ns,
            last: price,
            size: Some(size),
            bid_price: state.bid_price,
            bid_size: state.bid_size,
            ask_price: state.ask_price,
            ask_size: state.ask_size,
        });
        diagnostics.valid_trade_ticks = diagnostics.valid_trade_ticks.saturating_add(1);
    }
    Ok(Some(ticks))
}

fn parse_quote_timestamp(raw: &Value) -> Option<DateTime<Utc>> {
    if let Some(text) = raw.as_str() {
        return DateTime::parse_from_rfc3339(text)
            .ok()
            .map(|value| value.with_timezone(&Utc));
    }
    let value = raw.as_i64()?;
    let millis = if value.unsigned_abs() < 100_000_000_000 {
        value.checked_mul(1_000)?
    } else {
        value
    };
    DateTime::<Utc>::from_timestamp_millis(millis)
}

fn parse_hosted_clock(item: &Value) -> Option<(DateTime<Utc>, NinjaTraderReplaySpeed)> {
    if item.get("e").and_then(Value::as_str) != Some("clock") {
        return None;
    }
    let clock = item
        .get("d")?
        .as_str()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .or_else(|| {
            item.get("d")
                .and_then(Value::as_object)
                .cloned()
                .map(Value::Object)
        })?;
    let timestamp = clock.get("t")?.as_str()?;
    let timestamp = DateTime::parse_from_rfc3339(timestamp)
        .ok()?
        .with_timezone(&Utc);
    let speed = u16::try_from(clock.get("s")?.as_u64()?).ok()?;
    Some((timestamp, NinjaTraderReplaySpeed::new(speed).ok()?))
}

fn matches_pending_sync_response(response_id: Option<u64>, sync_id: Option<u64>) -> bool {
    // Once the one-shot sync request completes, unsolicited stream events
    // generally have no response id. `None == None` must not classify them as
    // another sync response or they'll be discarded before quote parsing.
    sync_id.is_some() && response_id == sync_id
}

fn validate_replay_market_specs(specs: MarketSpecs) -> Result<MarketSpecs> {
    if !specs
        .tick_size
        .is_some_and(|tick_size| tick_size.is_finite() && tick_size > 0.0)
    {
        bail!("NinjaTrader hosted Replay contract metadata has no valid positive tick size");
    }
    Ok(specs)
}

struct HostedBarAggregator {
    bar_type: BarType,
    current_period_ns: Option<i64>,
    current: Option<Bar>,
    current_count: u32,
    current_volume: f64,
}

impl HostedBarAggregator {
    fn new(bar_type: BarType) -> Self {
        Self {
            bar_type,
            current_period_ns: None,
            current: None,
            current_count: 0,
            current_volume: 0.0,
        }
    }

    fn push_tick(
        &mut self,
        ts_ns: i64,
        price: f64,
        size: f64,
    ) -> Option<(Option<Bar>, Option<Bar>)> {
        if !price.is_finite() || price <= 0.0 {
            return None;
        }
        match self.bar_type.kind() {
            BarKind::Minute | BarKind::Second => self.push_time_tick(ts_ns, price, size),
            BarKind::Tick => self.push_count_tick(ts_ns, price),
            BarKind::Volume => self.push_volume_tick(ts_ns, price, size),
            BarKind::Range => self.push_range_tick(ts_ns, price),
        }
    }

    fn push_time_tick(
        &mut self,
        ts_ns: i64,
        price: f64,
        size: f64,
    ) -> Option<(Option<Bar>, Option<Bar>)> {
        let interval = match self.bar_type.kind() {
            BarKind::Minute => i64::from(self.bar_type.value()).saturating_mul(60_000_000_000),
            _ => i64::from(self.bar_type.value()).saturating_mul(1_000_000_000),
        }
        .max(1);
        let period = ts_ns - ts_ns.rem_euclid(interval);
        if self.current_period_ns == Some(period) {
            let bar = self.current.as_mut()?;
            bar.high = bar.high.max(price);
            bar.low = bar.low.min(price);
            bar.close = price;
            bar.volume = Some(bar.volume.unwrap_or_default() + size);
            return Some((None, Some(bar.clone())));
        }
        let closed = self.current.take();
        self.current_period_ns = Some(period);
        let next = new_hosted_bar(period, price, size);
        self.current = Some(next.clone());
        Some((closed, Some(next)))
    }

    fn push_count_tick(&mut self, ts_ns: i64, price: f64) -> Option<(Option<Bar>, Option<Bar>)> {
        let target = self.bar_type.value().max(1);
        if self.current.is_none() || self.current_count >= target {
            let closed = self.current.take();
            self.current = Some(new_hosted_bar(ts_ns, price, 1.0));
            self.current_count = 1;
            return Some((closed, self.current.clone()));
        }
        self.current_count = self.current_count.saturating_add(1);
        let bar = self.current.as_mut()?;
        bar.ts_ns = ts_ns;
        bar.high = bar.high.max(price);
        bar.low = bar.low.min(price);
        bar.close = price;
        bar.volume = Some(bar.volume.unwrap_or_default() + 1.0);
        Some((None, Some(bar.clone())))
    }

    fn push_volume_tick(
        &mut self,
        ts_ns: i64,
        price: f64,
        size: f64,
    ) -> Option<(Option<Bar>, Option<Bar>)> {
        let target = f64::from(self.bar_type.value().max(1));
        if self.current.is_none() {
            self.current = Some(new_hosted_bar(ts_ns, price, 0.0));
        }
        let bar = self.current.as_mut()?;
        bar.ts_ns = ts_ns;
        bar.high = bar.high.max(price);
        bar.low = bar.low.min(price);
        bar.close = price;
        self.current_volume += size;
        bar.volume = Some(self.current_volume);
        if self.current_volume >= target {
            let closed = self.current.take();
            self.current_volume = 0.0;
            return Some((closed, None));
        }
        Some((None, Some(bar.clone())))
    }

    fn push_range_tick(&mut self, ts_ns: i64, price: f64) -> Option<(Option<Bar>, Option<Bar>)> {
        let threshold = f64::from(self.bar_type.value().max(1));
        if self.current.is_none() {
            self.current = Some(new_hosted_bar(ts_ns, price, 1.0));
            return Some((None, self.current.clone()));
        }
        let bar = self.current.as_mut()?;
        bar.ts_ns = ts_ns;
        bar.high = bar.high.max(price);
        bar.low = bar.low.min(price);
        bar.close = price;
        if bar.high - bar.low >= threshold {
            let closed = self.current.take();
            return Some((closed, None));
        }
        Some((None, Some(bar.clone())))
    }
}

fn new_hosted_bar(ts_ns: i64, price: f64, size: f64) -> Bar {
    Bar {
        ts_ns,
        open: price,
        high: price,
        low: price,
        close: price,
        volume: Some(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsolicited_frames_without_ids_are_not_treated_as_sync_responses() {
        assert!(!matches_pending_sync_response(None, None));
        assert!(!matches_pending_sync_response(Some(7), None));
        assert!(!matches_pending_sync_response(None, Some(7)));
        assert!(matches_pending_sync_response(Some(7), Some(7)));
        assert!(!matches_pending_sync_response(Some(8), Some(7)));
    }

    #[test]
    fn hosted_replay_requires_a_valid_contract_tick_size() {
        let valid = MarketSpecs {
            session_profile: None,
            value_per_point: Some(20.0),
            tick_size: Some(0.25),
        };
        assert!(validate_replay_market_specs(valid).is_ok());

        for tick_size in [
            None,
            Some(0.0),
            Some(-0.25),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            let invalid = MarketSpecs { tick_size, ..valid };
            assert!(validate_replay_market_specs(invalid).is_err());
        }
    }

    #[test]
    fn quote_unsubscribe_uses_contract_symbol() {
        let body = quote_unsubscribe_body(42);

        assert_eq!(body, json!({ "symbol": 42 }));
        assert!(body.get("subscriptionId").is_none());
    }

    #[test]
    fn replay_host_must_come_from_authenticated_dynamic_hosts() {
        let mut tokens = TokenBundle {
            access_token: "token".to_string(),
            md_access_token: "md".to_string(),
            expiration_time: None,
            user_id: Some(1),
            user_name: None,
            api_hosts: Some(ApiHosts {
                replay: Some("tenant-replay.nt.example".to_string()),
            }),
        };
        assert_eq!(
            ninja_replay_ws_url(&tokens).unwrap(),
            "wss://tenant-replay.nt.example/v1/websocket"
        );
        tokens.api_hosts = Some(ApiHosts { replay: None });
        assert!(ninja_replay_ws_url(&tokens).is_err());
        tokens.api_hosts = Some(ApiHosts {
            replay: Some("https://live.example".to_string()),
        });
        assert!(ninja_replay_ws_url(&tokens).is_err());
    }

    #[test]
    fn hosted_replay_speed_is_provider_percent_not_local_speed_steps() {
        assert_eq!(NinjaTraderReplaySpeed::new(0).unwrap().percent(), 0);
        assert_eq!(NinjaTraderReplaySpeed::new(400).unwrap().percent(), 400);
        assert!(NinjaTraderReplaySpeed::new(401).is_err());
    }

    #[test]
    fn unexpected_provider_close_is_reported_as_hosted_replay_failure() {
        let event = HostedReplayExit::Failed(
            "provider closed the Replay WebSocket unexpectedly".to_string(),
        )
        .into_internal_event(17)
        .expect("provider close should create an end event");

        assert!(matches!(
            event,
            InternalEvent::HostedReplayEnded {
                generation: 17,
                failed: true,
                reason,
            } if reason.contains("closed the Replay WebSocket")
        ));
        assert!(HostedReplayExit::Stopped.into_internal_event(17).is_none());
    }

    #[test]
    fn session_sync_account_binding_requires_one_account() {
        let response = json!({
            "d": { "accounts": [{ "id": 90001, "userId": 7, "name": "Replay session account" }] }
        });
        let entities = extract_entity_envelopes(&response);
        let accounts = hosted_replay_session_accounts(&entities, 7).unwrap();
        assert_eq!(accounts.len(), 1);
        assert_eq!(accounts[0].id, 90001);
        assert_eq!(accounts[0].name, "Replay session account");

        let zero = json!({ "d": { "accounts": [] } });
        assert!(hosted_replay_session_accounts(&extract_entity_envelopes(&zero), 7).is_err());

        let multiple = json!({ "d": { "accounts": [
            { "id": 1, "userId": 7, "name": "Replay one" },
            { "id": 2, "userId": 7, "name": "Replay two" }
        ] } });
        assert!(hosted_replay_session_accounts(&extract_entity_envelopes(&multiple), 7).is_err());

        let wrong_user = json!({ "d": { "accounts": [
            { "id": 1, "userId": 8, "name": "Replay owned by someone else" }
        ] } });
        assert!(hosted_replay_session_accounts(&extract_entity_envelopes(&wrong_user), 7).is_err());
    }

    #[test]
    fn quote_stream_parser_emits_trade_ticks_and_keeps_quote_state() {
        let start = DateTime::parse_from_rfc3339("2026-10-01T14:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let item = json!({
            "e": "md",
            "d": { "quotes": [{
                "contractId": 42,
                "timestamp": "2026-10-01T14:30:01.123Z",
                "entries": {
                    "Bid": { "price": 100.0, "size": 5.0 },
                    "Offer": { "price": 100.25, "size": 4.0 },
                    "Trade": { "price": 100.25, "size": 2.0 }
                }
            }] }
        });
        let mut state = HostedQuoteState::default();
        let mut diagnostics = HostedReplayQuoteDiagnostics::default();
        let ticks = parse_hosted_quote_ticks(&item, 42, start, None, &mut state, &mut diagnostics)
            .unwrap()
            .unwrap();
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].last, 100.25);
        assert_eq!(ticks[0].size, Some(2.0));
        assert_eq!(ticks[0].bid_price, Some(100.0));
        assert_eq!(ticks[0].ask_price, Some(100.25));
        assert_eq!(
            ticks[0].ts_ns,
            start.timestamp_nanos_opt().unwrap() + 1_123_000_000
        );
        assert_eq!(diagnostics.md_items, 1);
        assert_eq!(diagnostics.quote_rows, 1);
        assert_eq!(diagnostics.matching_contract_rows, 1);
        assert_eq!(diagnostics.trade_rows, 1);
        assert_eq!(diagnostics.valid_trade_ticks, 1);
    }

    #[test]
    fn quote_stream_diagnostics_separate_nonmatching_and_out_of_window_rows() {
        let start = DateTime::parse_from_rfc3339("2026-10-01T14:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let end = start + chrono::Duration::minutes(1);
        let item = json!({
            "e": "md",
            "d": { "quotes": [
                {
                    "contractId": 77,
                    "timestamp": "2026-10-01T14:30:01Z",
                    "entries": { "Trade": { "price": 100.0, "size": 1.0 } }
                },
                {
                    "contractId": 42,
                    "timestamp": "2026-10-01T14:29:59Z",
                    "entries": { "Trade": { "price": 100.0, "size": 1.0 } }
                },
                {
                    "contractId": 42,
                    "timestamp": "2026-10-01T14:30:01Z",
                    "entries": { "Bid": { "price": 99.75, "size": 1.0 } }
                }
            ] }
        });
        let mut state = HostedQuoteState::default();
        let mut diagnostics = HostedReplayQuoteDiagnostics::default();
        let ticks =
            parse_hosted_quote_ticks(&item, 42, start, Some(end), &mut state, &mut diagnostics)
                .unwrap()
                .unwrap();

        assert!(ticks.is_empty());
        assert_eq!(diagnostics.quote_rows, 3);
        assert_eq!(diagnostics.quote_rows_other_contract, 1);
        assert_eq!(diagnostics.matching_contract_rows, 2);
        assert_eq!(diagnostics.trade_rows, 1);
        assert_eq!(diagnostics.trade_rows_outside_window, 1);
        assert_eq!(diagnostics.valid_trade_ticks, 0);
    }

    #[test]
    fn hosted_clock_parser_reads_provider_clock_and_speed() {
        let item = json!({
            "e": "clock",
            "d": "{\"t\":\"2026-10-01T14:30:00.000Z\",\"s\":100}"
        });
        let (timestamp, speed) = parse_hosted_clock(&item).unwrap();
        assert_eq!(timestamp.to_rfc3339(), "2026-10-01T14:30:00+00:00");
        assert_eq!(speed.percent(), 100);
    }

    #[test]
    fn hosted_time_bars_form_and_close_incrementally() {
        let mut bars = HostedBarAggregator::new(BarType::minute(1));
        let (closed, forming) = bars.push_tick(60_000_000_000, 100.0, 1.0).unwrap();
        assert!(closed.is_none());
        assert_eq!(forming.unwrap().open, 100.0);
        let (closed, forming) = bars.push_tick(61_000_000_000, 102.0, 2.0).unwrap();
        assert!(closed.is_none());
        assert_eq!(forming.unwrap().high, 102.0);
        let (closed, forming) = bars.push_tick(120_000_000_000, 103.0, 1.0).unwrap();
        assert_eq!(closed.unwrap().close, 102.0);
        assert_eq!(forming.unwrap().open, 103.0);
    }
}
