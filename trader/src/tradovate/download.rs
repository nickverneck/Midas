use super::*;
use crate::replay_cache::ReplayCacheTickSpecs;

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
}
