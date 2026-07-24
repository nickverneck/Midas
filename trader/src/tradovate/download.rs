use super::*;
use crate::replay_cache::{ReplayCacheRawTickRow, ReplayCacheTickSpecs};

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownloadRequest {
    pub contract: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub bar_type: BarType,
}

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownload {
    pub contract: ContractSuggestion,
    pub bars: Vec<Bar>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownloadRequest {
    pub contract: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownload {
    pub contract: ContractSuggestion,
    pub ticks: Vec<ReplayCacheRawTickRow>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub warnings: Vec<String>,
}

pub async fn download_replay_server_bars(
    cfg: &AppConfig,
    request: TradovateServerBarDownloadRequest,
) -> Result<TradovateServerBarDownload> {
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract = resolve_download_contract(
        &client,
        &cfg.env,
        &tokens.access_token,
        &request.contract,
        cfg.contract_suggest_limit.max(12),
    )
    .await?;
    let specs = fetch_contract_specs(&client, &cfg.env, &tokens.access_token, &contract)
        .await
        .ok();
    let request_body = server_bar_chart_request_body(&contract, &request);
    let bars = fetch_server_bars_over_market_ws(
        cfg,
        &tokens.md_access_token,
        request_body.clone(),
        request.start,
        request.end,
    )
    .await?;
    let mut warnings = Vec::new();
    if bars.is_empty() {
        warnings.push("Tradovate returned zero historical server bars.".to_string());
    }
    let session_template = specs.as_ref().and_then(|market_specs| {
        market_specs
            .session_profile
            .map(|profile| profile.label().to_string())
    });
    Ok(TradovateServerBarDownload {
        contract,
        bars,
        request_body,
        tick_specs: ReplayCacheTickSpecs {
            tick_size: specs
                .as_ref()
                .and_then(|market_specs| market_specs.tick_size)
                .unwrap_or(0.0),
            value_per_point: specs
                .as_ref()
                .and_then(|market_specs| market_specs.value_per_point)
                .unwrap_or(0.0),
        },
        session_template,
        warnings,
    })
}

pub async fn download_replay_raw_ticks(
    cfg: &AppConfig,
    request: TradovateRawTickDownloadRequest,
) -> Result<TradovateRawTickDownload> {
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let contract = resolve_download_contract(
        &client,
        &cfg.env,
        &tokens.access_token,
        &request.contract,
        cfg.contract_suggest_limit.max(12),
    )
    .await?;
    let specs = fetch_contract_specs(&client, &cfg.env, &tokens.access_token, &contract)
        .await
        .ok();
    let request_body = raw_tick_chart_request_body(&contract, &request);
    let tick_size = specs
        .as_ref()
        .and_then(|market_specs| market_specs.tick_size)
        .filter(|tick_size| tick_size.is_finite() && *tick_size > 0.0);
    let ticks = fetch_raw_ticks_over_market_ws(
        cfg,
        &tokens.md_access_token,
        request_body.clone(),
        request.start,
        request.end,
        tick_size,
    )
    .await?;
    let mut warnings = Vec::new();
    if ticks.is_empty() {
        warnings.push("Tradovate returned zero historical raw ticks.".to_string());
    }
    let session_template = specs.as_ref().and_then(|market_specs| {
        market_specs
            .session_profile
            .map(|profile| profile.label().to_string())
    });
    Ok(TradovateRawTickDownload {
        contract,
        ticks,
        request_body,
        tick_specs: ReplayCacheTickSpecs {
            tick_size: specs
                .as_ref()
                .and_then(|market_specs| market_specs.tick_size)
                .unwrap_or(0.0),
            value_per_point: specs
                .as_ref()
                .and_then(|market_specs| market_specs.value_per_point)
                .unwrap_or(0.0),
        },
        session_template,
        warnings,
    })
}

pub fn server_bar_chart_request_body(
    contract: &ContractSuggestion,
    request: &TradovateServerBarDownloadRequest,
) -> Value {
    json!({
        "symbol": contract.id,
        "chartDescription": request.bar_type.chart_description(),
        "timeRange": {
            "closestTimestamp": request.end.to_rfc3339(),
            "asFarAsTimestamp": request.start.to_rfc3339()
        }
    })
}

pub fn raw_tick_chart_request_body(
    contract: &ContractSuggestion,
    request: &TradovateRawTickDownloadRequest,
) -> Value {
    json!({
        "symbol": contract.id,
        "chartDescription": {
            "underlyingType": "Tick",
            "elementSize": 1,
            "elementSizeUnit": "UnderlyingUnits",
            "withHistogram": false
        },
        "timeRange": {
            "closestTimestamp": request.end.to_rfc3339(),
            "asFarAsTimestamp": request.start.to_rfc3339()
        }
    })
}

pub fn extract_historical_server_bars_from_chart_message(
    item: &Value,
    historical_id: Option<i64>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> (Vec<Bar>, bool) {
    let mut bars = Vec::new();
    let mut end_of_history = false;
    let Some(charts) = item
        .get("d")
        .and_then(|d| d.get("charts"))
        .and_then(Value::as_array)
    else {
        return (bars, end_of_history);
    };
    let start_ns = start.timestamp_nanos_opt().unwrap_or(i64::MIN);
    let end_ns = end.timestamp_nanos_opt().unwrap_or(i64::MAX);
    for chart in charts {
        let chart_id = chart.get("id").and_then(Value::as_i64);
        let is_historical = historical_id.is_none_or(|id| chart_id == Some(id));
        if !is_historical {
            continue;
        }
        if chart.get("eoh").and_then(Value::as_bool).unwrap_or(false) {
            end_of_history = true;
        }
        let Some(raw_bars) = chart.get("bars").and_then(Value::as_array) else {
            continue;
        };
        for raw_bar in raw_bars {
            let Some(bar) = parse_bar(raw_bar) else {
                continue;
            };
            if bar.ts_ns >= start_ns && bar.ts_ns < end_ns {
                bars.push(bar);
            }
        }
    }
    (bars, end_of_history)
}

pub fn extract_historical_raw_ticks_from_chart_message(
    item: &Value,
    historical_id: Option<i64>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fallback_tick_size: Option<f64>,
) -> (Vec<ReplayCacheRawTickRow>, bool) {
    let mut ticks = Vec::new();
    let mut end_of_history = false;
    let Some(charts) = item
        .get("d")
        .and_then(|d| d.get("charts"))
        .and_then(Value::as_array)
    else {
        return (ticks, end_of_history);
    };
    let start_ns = start.timestamp_nanos_opt().unwrap_or(i64::MIN);
    let end_ns = end.timestamp_nanos_opt().unwrap_or(i64::MAX);
    for chart in charts {
        let chart_id = chart.get("id").and_then(Value::as_i64);
        let is_historical = historical_id.is_none_or(|id| chart_id == Some(id));
        if !is_historical {
            continue;
        }
        if chart.get("eoh").and_then(Value::as_bool).unwrap_or(false) {
            end_of_history = true;
        }
        let Some(packet_ticks) = parse_raw_tick_packet(chart, start_ns, end_ns, fallback_tick_size)
        else {
            continue;
        };
        ticks.extend(packet_ticks);
    }
    (ticks, end_of_history)
}

fn parse_raw_tick_packet(
    chart: &Value,
    start_ns: i64,
    end_ns: i64,
    fallback_tick_size: Option<f64>,
) -> Option<Vec<ReplayCacheRawTickRow>> {
    let raw_ticks = chart.get("tks").and_then(Value::as_array)?;
    let base_price_ticks = json_i64(chart, "bp")?;
    let base_ts_ms = json_i64(chart, "bt")?;
    let tick_size = json_number(chart, "ts")
        .filter(|tick_size| tick_size.is_finite() && *tick_size > 0.0)
        .or(fallback_tick_size)?;
    let chart_id = chart.get("id").and_then(Value::as_i64);
    let trade_date = json_i64(chart, "td").and_then(|value| i32::try_from(value).ok());
    let packet_source = chart
        .get("s")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let mut rows = Vec::with_capacity(raw_ticks.len());

    for raw_tick in raw_ticks {
        let Some(relative_ts_ms) = json_i64(raw_tick, "t") else {
            continue;
        };
        let Some(price_offset_ticks) = json_i64(raw_tick, "p") else {
            continue;
        };
        let Some(size) = json_number(raw_tick, "s") else {
            continue;
        };
        let Some(ts_ms) = base_ts_ms.checked_add(relative_ts_ms) else {
            continue;
        };
        let Some(ts_ns) = ts_ms.checked_mul(1_000_000) else {
            continue;
        };
        if ts_ns < start_ns || ts_ns >= end_ns {
            continue;
        }
        let price_ticks = base_price_ticks.saturating_add(price_offset_ticks);
        let price = price_ticks as f64 * tick_size;
        let bid_size = json_number(raw_tick, "bs");
        let ask_size = json_number(raw_tick, "as");
        let bid_price = bid_size
            .and_then(|_| json_i64(raw_tick, "b"))
            .map(|offset| base_price_ticks.saturating_add(offset) as f64 * tick_size);
        let ask_price = ask_size
            .and_then(|_| json_i64(raw_tick, "a"))
            .map(|offset| base_price_ticks.saturating_add(offset) as f64 * tick_size);

        rows.push(ReplayCacheRawTickRow {
            timestamp: DateTime::<Utc>::from_timestamp_nanos(ts_ns),
            ts_ns,
            tick_id: json_i64(raw_tick, "id"),
            price,
            size,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            chart_id,
            trade_date,
            packet_source: packet_source.clone(),
            packet_base_ts_ms: Some(base_ts_ms),
            packet_base_price_ticks: Some(base_price_ticks),
        });
    }

    Some(rows)
}

async fn resolve_download_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract_symbol: &str,
    limit: usize,
) -> Result<ContractSuggestion> {
    let contracts = search_contracts(client, env, token, contract_symbol, limit).await?;
    contracts
        .iter()
        .find(|contract| contract.name.eq_ignore_ascii_case(contract_symbol))
        .cloned()
        .with_context(|| {
            let available = contracts
                .iter()
                .map(|contract| contract.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if available.is_empty() {
                format!("contract search returned no results for {contract_symbol}")
            } else {
                format!(
                    "contract search did not return exact symbol {contract_symbol}; suggestions: {available}"
                )
            }
        })
}

async fn fetch_server_bars_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Vec<Bar>> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.market_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let authorize_id = 1_u64;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, md_access_token
        )))
        .await?;

    let chart_req_id = 2_u64;
    let mut authorized = false;
    let mut chart_requested = false;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut bars = Vec::new();
    let timeout = time::sleep(Duration::from_secs(35));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                bail!("timed out waiting for Tradovate server-bar history");
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

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);
                    if response_id == Some(authorize_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            let message = parse_socket_response(&item)
                                .expect_err("authorization failure should carry websocket error");
                            bail!("market websocket authorize failed: {message}");
                        }
                        authorized = true;
                    }

                    if authorized && !chart_requested {
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                chart_req_id,
                                None,
                                Some(&request_body),
                            )))
                            .await?;
                        chart_requested = true;
                    }

                    if response_id == Some(chart_req_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            let message = parse_socket_response(&item)
                                .expect_err("chart failure should carry websocket error");
                            bail!("md/getChart failed: {message}");
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                    }

                    let (mut packet_bars, end_of_history) =
                        extract_historical_server_bars_from_chart_message(
                            &item,
                            historical_id,
                            start,
                            end,
                        );
                    bars.append(&mut packet_bars);
                    if end_of_history {
                        cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Ok(bars);
                    }
                }
            }
        }
    }

    cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
    Ok(bars)
}

async fn fetch_raw_ticks_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fallback_tick_size: Option<f64>,
) -> Result<Vec<ReplayCacheRawTickRow>> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .with_context(|| format!("connect {}", cfg.env.market_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let authorize_id = 1_u64;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, md_access_token
        )))
        .await?;

    let chart_req_id = 2_u64;
    let mut authorized = false;
    let mut chart_requested = false;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut ticks = Vec::new();
    let timeout = time::sleep(Duration::from_secs(60));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                bail!("timed out waiting for Tradovate raw tick history");
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

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);
                    if response_id == Some(authorize_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            let message = parse_socket_response(&item)
                                .expect_err("authorization failure should carry websocket error");
                            bail!("market websocket authorize failed: {message}");
                        }
                        authorized = true;
                    }

                    if authorized && !chart_requested {
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                chart_req_id,
                                None,
                                Some(&request_body),
                            )))
                            .await?;
                        chart_requested = true;
                    }

                    if response_id == Some(chart_req_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            let message = parse_socket_response(&item)
                                .expect_err("chart failure should carry websocket error");
                            bail!("md/getChart failed: {message}");
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                    }

                    let (mut packet_ticks, end_of_history) =
                        extract_historical_raw_ticks_from_chart_message(
                            &item,
                            historical_id,
                            start,
                            end,
                            fallback_tick_size,
                        );
                    ticks.append(&mut packet_ticks);
                    if end_of_history {
                        cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Ok(ticks);
                    }
                }
            }
        }
    }

    cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
    Ok(ticks)
}

async fn cancel_chart_subscriptions<S>(
    write: &mut S,
    historical_id: Option<i64>,
    realtime_id: Option<i64>,
) where
    S: futures_util::Sink<Message> + Unpin,
{
    let mut cancel_id = 10_000_u64;
    for subscription_id in [historical_id, realtime_id].into_iter().flatten() {
        cancel_id += 1;
        let body = json!({ "subscriptionId": subscription_id });
        let _ = write
            .send(Message::Text(create_message(
                "md/cancelChart",
                cancel_id,
                None,
                Some(&body),
            )))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dt(raw: &str) -> DateTime<Utc> {
        raw.parse().expect("valid timestamp")
    }

    #[test]
    fn server_bar_request_uses_chart_timerange_without_account_streams() {
        let contract = ContractSuggestion {
            id: 123,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({ "id": 123, "name": "MESU6" }),
        };
        let request = TradovateServerBarDownloadRequest {
            contract: "MESU6".to_string(),
            start: dt("2026-07-23T00:00:00Z"),
            end: dt("2026-07-24T00:00:00Z"),
            bar_type: BarType::minute(1),
        };

        let body = server_bar_chart_request_body(&contract, &request);

        assert_eq!(body["symbol"], 123);
        assert_eq!(
            body["chartDescription"],
            BarType::minute(1).chart_description()
        );
        assert_eq!(
            body["timeRange"]["asFarAsTimestamp"],
            "2026-07-23T00:00:00+00:00"
        );
        assert_eq!(
            body["timeRange"]["closestTimestamp"],
            "2026-07-24T00:00:00+00:00"
        );
        assert!(body.get("replayCache").is_none());
        assert!(body.get("accounts").is_none());
        assert!(body.get("entityTypes").is_none());
    }

    #[test]
    fn raw_tick_request_uses_one_tick_chart_without_account_streams() {
        let contract = ContractSuggestion {
            id: 123,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({ "id": 123, "name": "MESU6" }),
        };
        let request = TradovateRawTickDownloadRequest {
            contract: "MESU6".to_string(),
            start: dt("2026-07-23T00:00:00Z"),
            end: dt("2026-07-24T00:00:00Z"),
        };

        let body = raw_tick_chart_request_body(&contract, &request);

        assert_eq!(body["symbol"], 123);
        assert_eq!(body["chartDescription"]["underlyingType"], "Tick");
        assert_eq!(body["chartDescription"]["elementSize"], 1);
        assert_eq!(
            body["chartDescription"]["elementSizeUnit"],
            "UnderlyingUnits"
        );
        assert_eq!(
            body["timeRange"]["asFarAsTimestamp"],
            "2026-07-23T00:00:00+00:00"
        );
        assert_eq!(
            body["timeRange"]["closestTimestamp"],
            "2026-07-24T00:00:00+00:00"
        );
        assert!(body.get("accounts").is_none());
        assert!(body.get("entityTypes").is_none());
    }

    #[test]
    fn extracts_only_historical_bars_inside_requested_range() {
        let item = json!({
            "d": {
                "charts": [
                    {
                        "id": 7,
                        "bars": [
                            {"timestamp":"2026-07-22T23:59:00Z","open":1,"high":1,"low":1,"close":1,"volume":1},
                            {"timestamp":"2026-07-23T00:00:00Z","open":2,"high":3,"low":1,"close":2.5,"upVolume":5,"downVolume":4}
                        ]
                    },
                    {
                        "id": 8,
                        "bars": [
                            {"timestamp":"2026-07-23T00:00:00Z","open":9,"high":9,"low":9,"close":9,"volume":1}
                        ]
                    },
                    {"id": 7, "eoh": true}
                ]
            }
        });

        let (bars, eoh) = extract_historical_server_bars_from_chart_message(
            &item,
            Some(7),
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-24T00:00:00Z"),
        );

        assert!(eoh);
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].open, 2.0);
        assert_eq!(bars[0].volume, Some(9.0));
    }

    #[test]
    fn extracts_raw_ticks_from_historical_tick_packets() {
        let item = json!({
            "d": {
                "charts": [
                    {
                        "id": 7,
                        "s": "db",
                        "td": 20260723,
                        "bp": 29700,
                        "bt": 1784764800000i64,
                        "ts": 0.25,
                        "tks": [
                            {"t": 0, "p": 0, "s": 1, "b": -1, "a": 0, "bs": 12, "as": 14, "id": 1001},
                            {"t": 1, "p": 1, "s": 2, "id": 1002},
                            {"t": -1, "p": 9, "s": 1, "id": 999}
                        ]
                    },
                    {
                        "id": 8,
                        "bp": 999,
                        "bt": 1784764800000i64,
                        "ts": 0.25,
                        "tks": [
                            {"t": 0, "p": 0, "s": 1, "id": 2001}
                        ]
                    },
                    {"id": 7, "eoh": true}
                ]
            }
        });

        let (ticks, eoh) = extract_historical_raw_ticks_from_chart_message(
            &item,
            Some(7),
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-23T00:00:01Z"),
            None,
        );

        assert!(eoh);
        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].tick_id, Some(1001));
        assert_eq!(ticks[0].price, 7425.0);
        assert_eq!(ticks[0].bid_price, Some(7424.75));
        assert_eq!(ticks[0].ask_price, Some(7425.0));
        assert_eq!(ticks[0].size, 1.0);
        assert_eq!(ticks[0].chart_id, Some(7));
        assert_eq!(ticks[0].trade_date, Some(20260723));
        assert_eq!(ticks[0].packet_source.as_deref(), Some("db"));
        assert_eq!(ticks[1].tick_id, Some(1002));
        assert_eq!(ticks[1].price, 7425.25);
    }
}
