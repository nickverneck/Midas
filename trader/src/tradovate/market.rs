#[derive(Debug, Clone, Copy)]
struct MarketSpecs {
    session_profile: Option<InstrumentSessionProfile>,
    value_per_point: Option<f64>,
    tick_size: Option<f64>,
}

async fn fetch_contract_specs(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract: &ContractSuggestion,
) -> Result<MarketSpecs> {
    let contract_maturity_id = json_i64(&contract.raw, "contractMaturityId")
        .context("selected contract is missing contractMaturityId")?;
    let maturity_url = format!("{}/contractMaturity/item", env.rest_url());
    let maturity_response = client
        .get(&maturity_url)
        .bearer_auth(token)
        .query(&[("id", contract_maturity_id.to_string())])
        .send()
        .await?;
    let maturity_status = maturity_response.status();
    let maturity_body = maturity_response.text().await.unwrap_or_default();
    if !maturity_status.is_success() {
        bail!("contractMaturity/item failed ({maturity_status}): {maturity_body}");
    }
    let maturity: Value = serde_json::from_str(&maturity_body)?;
    let product_id =
        json_i64(&maturity, "productId").context("contractMaturity/item missing productId")?;

    let product_url = format!("{}/product/item", env.rest_url());
    let product_response = client
        .get(&product_url)
        .bearer_auth(token)
        .query(&[("id", product_id.to_string())])
        .send()
        .await?;
    let product_status = product_response.status();
    let product_body = product_response.text().await.unwrap_or_default();
    if !product_status.is_success() {
        bail!("product/item failed ({product_status}): {product_body}");
    }
    let product: Value = serde_json::from_str(&product_body)?;
    Ok(MarketSpecs {
        session_profile: Some(infer_session_profile(&product)),
        value_per_point: json_number(&product, "valuePerPoint"),
        tick_size: json_number(&product, "tickSize").or_else(|| json_number(&product, "minTick")),
    })
}

async fn fetch_entity_list(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    entity: &str,
) -> Result<Vec<Value>> {
    let url = format!("{}/{entity}/list", env.rest_url());
    let response = client.get(url).bearer_auth(token).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{entity}/list failed ({status}): {body}");
    }
    let parsed: Value = serde_json::from_str(&body)?;
    Ok(match parsed {
        Value::Array(items) => items,
        Value::Object(_) => vec![parsed],
        _ => Vec::new(),
    })
}

async fn seed_user_store(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    store: &mut UserSyncStore,
) {
    for entity in [
        "account",
        "accountRiskStatus",
        "cashBalance",
        "position",
        "order",
        "orderStrategy",
        "orderStrategyLink",
    ] {
        let Ok(items) = fetch_entity_list(client, env, token, entity).await else {
            continue;
        };
        for item in items {
            store.apply(EntityEnvelope {
                entity_type: entity.to_string(),
                deleted: false,
                entity: item,
            });
        }
    }
}

/// Create a low-latency WebSocket connection with TCP_NODELAY and TCP_QUICKACK (Linux).
async fn connect_low_latency_ws(
    url: &str,
    ws_config: WebSocketConfig,
) -> Result<(
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    tokio_tungstenite::tungstenite::http::Response<Option<Vec<u8>>>,
)> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let request = url.into_client_request()?;
    let host = request
        .uri()
        .host()
        .ok_or_else(|| anyhow::anyhow!("no host in URL"))?;
    let port = request.uri().port_u16().unwrap_or(443);

    let addr = format!("{host}:{port}")
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| anyhow::anyhow!("DNS resolution failed for {host}"))?;

    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    socket.set_nodelay(true)?;
    socket.set_nonblocking(true)?;

    // Minimize socket buffer sizes to reduce kernel-side latency
    let _ = socket.set_recv_buffer_size(64 * 1024);
    let _ = socket.set_send_buffer_size(64 * 1024);

    // Start non-blocking connect (returns EINPROGRESS)
    let _ = socket.connect(&addr.into());

    let std_stream: std::net::TcpStream = socket.into();
    let tcp_stream = tokio::net::TcpStream::from_std(std_stream)?;
    // Wait for the async connect to complete
    tcp_stream.writable().await?;

    // Re-apply TCP_NODELAY after connect (some OS reset it)
    tcp_stream.set_nodelay(true)?;

    // Linux: disable delayed ACKs
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let fd = tcp_stream.as_raw_fd();
        unsafe {
            let val: i32 = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &val as *const _ as *const libc::c_void,
                std::mem::size_of::<i32>() as libc::socklen_t,
            );
        }
    }

    let (ws, resp) =
        tokio_tungstenite::client_async_tls_with_config(request, tcp_stream, Some(ws_config), None)
            .await?;
    Ok((ws, resp))
}

async fn user_sync_worker(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    request_rx: UnboundedReceiver<UserSocketCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) =
        user_sync_worker_inner(cfg, tokens, account_ids, request_rx, internal_tx.clone()).await
    {
        let _ = internal_tx.send(InternalEvent::Error(format!("user sync: {err}")));
    }
}

async fn user_sync_worker_inner(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    mut request_rx: UnboundedReceiver<UserSocketCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.user_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.user_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let mut message_id = 1_u64;
    let authorize_id = message_id;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, tokens.access_token
        )))
        .await?;

    let mut sync_id = None;
    let mut authorized = false;
    let mut pending_requests = HashMap::<u64, oneshot::Sender<Result<Value, String>>>::new();
    let mut heartbeat = time::interval(Duration::from_millis(cfg.heartbeat_ms.max(250)));
    heartbeat.tick().await;

    loop {
        tokio::select! {
            biased;

            outbound = request_rx.recv() => {
                let Some(outbound) = outbound else {
                    break;
                };
                if !authorized {
                    let _ = outbound
                        .response_tx
                        .send(Err("user websocket is not authorized".to_string()));
                    continue;
                }

                message_id += 1;
                let request_id = message_id;
                pending_requests.insert(request_id, outbound.response_tx);
                let frame = create_message(
                    &outbound.endpoint,
                    request_id,
                    outbound.query.as_deref(),
                    outbound.body.as_ref(),
                );
                if let Err(err) = write.send(Message::Text(frame)).await {
                    if let Some(response_tx) = pending_requests.remove(&request_id) {
                        let _ = response_tx.send(Err(format!(
                            "user websocket send error: {err}"
                        )));
                    }
                    bail!("user websocket send error: {err}");
                }
            }
            _ = heartbeat.tick(), if authorized => {
                let _ = write.send(Message::Text("[]".to_string())).await;
            }
            next = read.next() => {
                let raw = match next {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => String::from_utf8_lossy(&bytes).to_string(),
                    Some(Ok(Message::Close(_))) => {
                        let _ = internal_tx.send(InternalEvent::UserSocketStatus("User-data websocket closed".to_string()));
                        break;
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(err)) => bail!("user websocket read error: {err}"),
                    None => break,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);

                    if !authorized && response_id == Some(authorize_id) {
                        let Some(status) = status else {
                            continue;
                        };
                        if status != 200 {
                            bail!("user websocket authorize failed ({status})");
                        }

                        authorized = true;
                        message_id += 1;
                        sync_id = Some(message_id);
                        let body = json!({
                            "splitResponses": true,
                            "accounts": account_ids,
                            "entityTypes": [
                                "account",
                                "accountRiskStatus",
                                "cashBalance",
                                "position",
                                "order",
                                "orderStrategy",
                                "orderStrategyLink",
                                "executionReport",
                                "fill"
                            ]
                        });
                        write
                            .send(Message::Text(create_message(
                                "user/syncrequest",
                                message_id,
                                None,
                                Some(&body),
                            )))
                            .await?;
                        let _ = internal_tx.send(InternalEvent::UserSocketStatus("User sync authorized".to_string()));
                        continue;
                    }

                    if status == Some(200) && response_id == sync_id {
                        let envelopes = extract_entity_envelopes(&item);
                        if !envelopes.is_empty() {
                            let _ = internal_tx.send(InternalEvent::UserEntities(envelopes));
                        }
                        continue;
                    }

                    if let Some(request_id) = response_id {
                        if let Some(response_tx) = pending_requests.remove(&request_id) {
                            let _ = response_tx.send(parse_socket_response(&item));
                            continue;
                        }
                    }

                    let envelopes = extract_entity_envelopes(&item);
                    if !envelopes.is_empty() {
                        let _ = internal_tx.send(InternalEvent::UserEntities(envelopes));
                    }
                }
            }
        }
    }

    Ok(())
}

async fn market_data_worker(
    cfg: AppConfig,
    access_token: String,
    contract: ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    bar_type: BarType,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) = market_data_worker_inner(
        cfg,
        access_token,
        contract,
        market_specs,
        bar_type,
        internal_tx.clone(),
    )
    .await
    {
        let _ = internal_tx.send(InternalEvent::Error(format!("market data: {err}")));
    }
}

async fn market_data_worker_inner(
    cfg: AppConfig,
    access_token: String,
    contract: ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    bar_type: BarType,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.market_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let mut message_id = 1_u64;
    let authorize_id = message_id;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, access_token
        )))
        .await?;

    let mut chart_req_id = None;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut authorized = false;
    let mut series = LiveSeries::new();
    let mut live_bars = 0_usize;

    let mut heartbeat = time::interval(Duration::from_millis(cfg.heartbeat_ms.max(250)));
    heartbeat.tick().await;

    loop {
        tokio::select! {
            _ = heartbeat.tick(), if authorized => {
                let _ = write.send(Message::Text("[]".to_string())).await;
            }
            next = read.next() => {
                let raw = match next {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => String::from_utf8_lossy(&bytes).to_string(),
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => continue,
                    Some(Err(err)) => bail!("market websocket read error: {err}"),
                    None => break,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };
                let before_closed_len = series.closed_bars.len();
                let before_last_closed = series.closed_bars.last().cloned();
                let before_forming = series.forming_bar.clone();

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);

                    if !authorized && status == Some(200) && response_id == Some(authorize_id) {
                        authorized = true;
                        message_id += 1;
                        chart_req_id = Some(message_id);
                        let body = json!({
                            "symbol": contract.id,
                            "chartDescription": bar_type.chart_description(),
                            "timeRange": {
                                "asMuchAsElements": cfg.history_bars
                            }
                        });
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                message_id,
                                None,
                                Some(&body),
                            )))
                            .await?;
                        continue;
                    }

                    if status == Some(200) && response_id == chart_req_id {
                        if let Some(d) = item.get("d") {
                            historical_id = d.get("historicalId").and_then(Value::as_i64).or(historical_id);
                            realtime_id = d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                    }

                    let Some(charts) = item
                        .get("d")
                        .and_then(|d| d.get("charts"))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };

                    for chart in charts {
                        let chart_id = chart.get("id").and_then(Value::as_i64);
                        let Some(bars) = chart.get("bars").and_then(Value::as_array) else {
                            continue;
                        };

                        for bar_json in bars {
                            let Some(bar) = parse_bar(bar_json) else {
                                continue;
                            };
                            let is_historical = chart_id.is_some()
                                && historical_id.is_some()
                                && chart_id == historical_id;
                            let is_realtime =
                                chart_id.is_some() && realtime_id.is_some() && chart_id == realtime_id;

                            if is_historical || (historical_id.is_none() && realtime_id.is_none()) {
                                series.push_closed_bar_capped(&bar, ENGINE_MARKET_BAR_LIMIT);
                                continue;
                            }

                            if is_realtime {
                                if let Some(current_ts) =
                                    series.forming_bar.as_ref().map(|current| current.ts_ns)
                                {
                                    if bar.ts_ns == current_ts {
                                        if let Some(current) = series.forming_bar.as_mut() {
                                            *current = bar;
                                        }
                                    } else if bar.ts_ns > current_ts {
                                        let closed =
                                            series.forming_bar.take().expect("forming bar exists");
                                        series.push_closed_bar_capped(
                                            &closed,
                                            ENGINE_MARKET_BAR_LIMIT,
                                        );
                                        series.forming_bar = Some(bar);
                                        live_bars = live_bars.saturating_add(1);
                                    }
                                } else {
                                    series.forming_bar = Some(bar);
                                }
                            }
                        }
                    }
                }

                if let Some(update) = build_market_update(
                    &contract,
                    market_specs,
                    series.closed_bars.len(),
                    live_bars,
                    format!("Subscribed to {} bars for {}", bar_type.label(), contract.name),
                    before_closed_len,
                    before_last_closed,
                    before_forming,
                    &series,
                ) {
                    let _ = internal_tx.send(InternalEvent::Market(update));
                }
            }
        }
    }

    Ok(())
}
