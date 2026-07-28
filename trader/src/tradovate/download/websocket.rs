use super::protocol::*;
use super::*;

pub(super) async fn fetch_server_bars_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<HistoricalRows<Bar>> {
    let request_window = DownloadWindow::new(start, end)?;
    let mut telemetry = HistoricalDownloadTelemetry::new(request_window);
    let started = std::time::Instant::now();
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .map_err(|_| {
            historical_failure(
                HistoricalDownloadFailureKind::Transport,
                "connect market-data websocket failed; transport details omitted",
                telemetry.clone(),
                started,
            )
        })?;
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
                telemetry.timed_out = true;
                telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                return Err(historical_failure(
                    HistoricalDownloadFailureKind::Timeout,
                    "timed out waiting for explicit Tradovate server-bar end-of-history",
                    telemetry,
                    started,
                ));
            }
            next = read.next() => {
                let message = match next {
                    Some(Ok(message)) => message,
                    Some(Err(_)) => {
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::Transport,
                            "market websocket read failed; transport details omitted",
                            telemetry,
                            started,
                        ));
                    }
                    None => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket ended before explicit Tradovate server-bar end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                };
                record_wire_message(&mut telemetry, &message);
                let raw = match message {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Message::Close(_) => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket closed before explicit Tradovate server-bar end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                    _ => continue,
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
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("market websocket authorize", status),
                                telemetry,
                                started,
                            ));
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
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("md/getChart", status),
                                telemetry,
                                started,
                            ));
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                        telemetry.historical_id = historical_id;
                        telemetry.realtime_id = realtime_id;
                    }

                    telemetry.provider_rows = telemetry.provider_rows.saturating_add(
                        historical_chart_row_count(&item, historical_id, "bars"),
                    );
                    let (mut packet_bars, end_of_history) =
                        extract_historical_server_bars_from_chart_message(
                            &item,
                            historical_id,
                            start,
                            end,
                        );
                    bars.append(&mut packet_bars);
                    if end_of_history {
                        telemetry.end_of_history = true;
                        telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        finish_telemetry(&mut telemetry, &bars, |bar| DateTime::<Utc>::from_timestamp_nanos(bar.ts_ns), started);
                        return Ok(HistoricalRows { rows: bars, telemetry });
                    }
                }
            }
        }
    }
}

pub(super) async fn fetch_raw_ticks_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fallback_tick_size: Option<f64>,
) -> Result<HistoricalRows<ReplayCacheRawTickRow>> {
    let request_window = DownloadWindow::new(start, end)?;
    let requested_element_limit = request_body
        .pointer("/timeRange/asMuchAsElements")
        .and_then(Value::as_u64);
    let mut telemetry = HistoricalDownloadTelemetry::new(request_window);
    let started = std::time::Instant::now();
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .map_err(|_| {
            historical_failure(
                HistoricalDownloadFailureKind::Transport,
                "connect market-data websocket failed; transport details omitted",
                telemetry.clone(),
                started,
            )
        })?;
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
    let mut provider_first_timestamp = None;
    let mut provider_last_timestamp = None;
    let mut ticks = Vec::new();
    let timeout = time::sleep(Duration::from_secs(60));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                telemetry.timed_out = true;
                telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                return Err(historical_failure(
                    HistoricalDownloadFailureKind::Timeout,
                    "timed out waiting for explicit Tradovate raw-tick end-of-history",
                    telemetry,
                    started,
                ));
            }
            next = read.next() => {
                let message = match next {
                    Some(Ok(message)) => message,
                    Some(Err(_)) => {
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::Transport,
                            "market websocket read failed; transport details omitted",
                            telemetry,
                            started,
                        ));
                    }
                    None => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket ended before explicit Tradovate raw-tick end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                };
                record_wire_message(&mut telemetry, &message);
                let raw = match message {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Message::Close(_) => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket closed before explicit Tradovate raw-tick end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                    _ => continue,
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
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("market websocket authorize", status),
                                telemetry,
                                started,
                            ));
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
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("md/getChart", status),
                                telemetry,
                                started,
                            ));
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                        telemetry.historical_id = historical_id;
                        telemetry.realtime_id = realtime_id;
                    }

                    telemetry.provider_rows = telemetry.provider_rows.saturating_add(
                        historical_chart_row_count(&item, historical_id, "tks"),
                    );
                    if let Some((first, last)) =
                        historical_raw_tick_timestamp_bounds(&item, historical_id)
                    {
                        provider_first_timestamp = Some(
                            provider_first_timestamp.map_or(first, |current: DateTime<Utc>| {
                                current.min(first)
                            }),
                        );
                        provider_last_timestamp = Some(
                            provider_last_timestamp.map_or(last, |current: DateTime<Utc>| {
                                current.max(last)
                            }),
                        );
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
                        telemetry.end_of_history = true;
                        telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        telemetry.page_count = 1;
                        telemetry.provider_first_timestamp = provider_first_timestamp;
                        telemetry.provider_last_timestamp = provider_last_timestamp;
                        finish_telemetry(&mut telemetry, &ticks, |tick| tick.timestamp, started);
                        raw_tick_element_limit_reached(
                            &mut telemetry,
                            requested_element_limit,
                            provider_first_timestamp,
                            start,
                        );
                        return Ok(HistoricalRows { rows: ticks, telemetry });
                    }
                }
            }
        }
    }
}

pub(super) fn raw_tick_element_limit_reached(
    telemetry: &mut HistoricalDownloadTelemetry,
    requested_element_limit: Option<u64>,
    provider_first_timestamp: Option<DateTime<Utc>>,
    requested_start: DateTime<Utc>,
) -> bool {
    let Some(limit) = requested_element_limit.filter(|limit| *limit > 0) else {
        return false;
    };
    if telemetry.provider_rows >= limit
        && provider_first_timestamp.is_none_or(|first| first > requested_start)
    {
        telemetry.cap = DownloadCapClassification::Yes;
        telemetry.cap_reason = format!(
            "Provider returned {} raw ticks at the effective {limit}-element page limit without reaching requested start {requested_start}; another backward page is required.",
            telemetry.provider_rows,
        );
        true
    } else {
        telemetry.cap = DownloadCapClassification::No;
        telemetry.cap_reason = format!(
            "Provider returned {} raw ticks and reached the requested start or stayed below the effective {limit}-element page limit.",
            telemetry.provider_rows,
        );
        false
    }
}

async fn cancel_chart_subscriptions<S>(
    write: &mut S,
    historical_id: Option<i64>,
    realtime_id: Option<i64>,
) -> bool
where
    S: futures_util::Sink<Message> + Unpin,
{
    let mut cancel_id = 10_000_u64;
    let mut sent = false;
    for subscription_id in [historical_id, realtime_id].into_iter().flatten() {
        cancel_id += 1;
        let body = json!({ "subscriptionId": subscription_id });
        if write
            .send(Message::Text(create_message(
                "md/cancelChart",
                cancel_id,
                None,
                Some(&body),
            )))
            .await
            .is_ok()
        {
            sent = true;
        }
    }
    sent
}
