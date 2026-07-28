use super::*;

// Tradovate caps chart history at 4,096 bars/ticks per request even when a
// larger asMuchAsElements value is supplied. Raw-tick callers must paginate
// backward from the oldest returned timestamp until the requested start.
pub(super) const RAW_TICK_REQUEST_MAX_ELEMENTS: u64 = 4_096;

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
        "symbol": contract.name,
        "chartDescription": {
            "underlyingType": "Tick",
            "elementSize": 1,
            "elementSizeUnit": "UnderlyingUnits",
            "withHistogram": false
        },
        "timeRange": {
            "asFarAsTimestamp": request.start.to_rfc3339(),
            "closestTimestamp": request.end.to_rfc3339(),
            "asMuchAsElements": RAW_TICK_REQUEST_MAX_ELEMENTS
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

pub(super) fn historical_raw_tick_timestamp_bounds(
    item: &Value,
    historical_id: Option<i64>,
) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let charts = item
        .get("d")
        .and_then(|d| d.get("charts"))
        .and_then(Value::as_array)?;
    let mut first_ms = None;
    let mut last_ms = None;
    for chart in charts {
        let chart_id = chart.get("id").and_then(Value::as_i64);
        if !historical_id.is_none_or(|id| chart_id == Some(id)) {
            continue;
        }
        let Some(base_ts_ms) = json_i64(chart, "bt") else {
            continue;
        };
        let Some(raw_ticks) = chart.get("tks").and_then(Value::as_array) else {
            continue;
        };
        for tick in raw_ticks {
            let Some(relative_ts_ms) = json_i64(tick, "t") else {
                continue;
            };
            let Some(ts_ms) = base_ts_ms.checked_add(relative_ts_ms) else {
                continue;
            };
            first_ms = Some(first_ms.map_or(ts_ms, |current: i64| current.min(ts_ms)));
            last_ms = Some(last_ms.map_or(ts_ms, |current: i64| current.max(ts_ms)));
        }
    }
    Some((
        DateTime::<Utc>::from_timestamp_millis(first_ms?)?,
        DateTime::<Utc>::from_timestamp_millis(last_ms?)?,
    ))
}

pub(super) fn parse_raw_tick_packet(
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

pub(super) struct HistoricalRows<T> {
    pub(super) rows: Vec<T>,
    pub(super) telemetry: HistoricalDownloadTelemetry,
}

pub(super) fn historical_failure(
    kind: HistoricalDownloadFailureKind,
    message: impl Into<String>,
    mut telemetry: HistoricalDownloadTelemetry,
    started: std::time::Instant,
) -> anyhow::Error {
    telemetry.elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    anyhow::Error::new(HistoricalDownloadFailure {
        kind,
        message: message.into(),
        telemetry,
        retry_after_ms: None,
    })
}

pub(super) fn websocket_status_failure_kind(status: Option<i64>) -> HistoricalDownloadFailureKind {
    match status {
        Some(401) => HistoricalDownloadFailureKind::Authentication,
        Some(403) => HistoricalDownloadFailureKind::Authorization,
        Some(408) => HistoricalDownloadFailureKind::Timeout,
        Some(413 | 414 | 422) => HistoricalDownloadFailureKind::SizeRejected,
        Some(429) => HistoricalDownloadFailureKind::RateLimited,
        _ => HistoricalDownloadFailureKind::Provider,
    }
}

pub(super) fn sanitized_websocket_failure(endpoint: &str, status: Option<i64>) -> String {
    match status {
        Some(status) => format!(
            "{endpoint} failed with websocket status {status}; provider response body omitted"
        ),
        None => format!(
            "{endpoint} response did not include a status code; provider response body omitted"
        ),
    }
}

pub(super) fn record_wire_message(telemetry: &mut HistoricalDownloadTelemetry, message: &Message) {
    telemetry.packet_count = telemetry.packet_count.saturating_add(1);
    let bytes = match message {
        Message::Text(text) => text.len(),
        Message::Binary(bytes) => bytes.len(),
        Message::Ping(bytes) | Message::Pong(bytes) => bytes.len(),
        Message::Close(frame) => frame
            .as_ref()
            .map(|frame| frame.reason.len().saturating_add(2))
            .unwrap_or_default(),
        Message::Frame(frame) => frame.payload().len(),
    };
    telemetry.wire_bytes = telemetry.wire_bytes.saturating_add(bytes as u64);
}

pub(super) fn historical_chart_row_count(
    item: &Value,
    historical_id: Option<i64>,
    field: &str,
) -> u64 {
    item.get("d")
        .and_then(|value| value.get("charts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|chart| {
            historical_id.is_none_or(|id| chart.get("id").and_then(Value::as_i64) == Some(id))
        })
        .filter_map(|chart| chart.get(field).and_then(Value::as_array))
        .map(|rows| rows.len() as u64)
        .sum()
}

pub(super) fn finish_telemetry<T>(
    telemetry: &mut HistoricalDownloadTelemetry,
    rows: &[T],
    timestamp: impl Fn(&T) -> DateTime<Utc>,
    started: std::time::Instant,
) {
    telemetry.elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    telemetry.normalized_rows = rows.len() as u64;
    telemetry.dropped_rows = telemetry
        .provider_rows
        .saturating_sub(telemetry.normalized_rows);
    telemetry.first_timestamp = rows.iter().map(&timestamp).min();
    telemetry.last_timestamp = rows.iter().map(timestamp).max();
}
