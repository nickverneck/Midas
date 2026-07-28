use super::super::protocol::{sanitized_websocket_failure, websocket_status_failure_kind};
use super::super::session::raw_tick_next_page_cursor;
use super::super::websocket::raw_tick_element_limit_reached;
use crate::replay_cache::{ReplayCacheRawTickRow, normalize_raw_tick_rows};
use crate::replay_download::{
    DownloadCapClassification, DownloadWindow, HistoricalDownloadFailureKind,
    HistoricalDownloadTelemetry,
};
use chrono::{TimeZone, Utc};

#[test]
fn websocket_statuses_preserve_authentication_and_retry_classification() {
    assert_eq!(
        websocket_status_failure_kind(Some(401)),
        HistoricalDownloadFailureKind::Authentication
    );
    assert_eq!(
        websocket_status_failure_kind(Some(429)),
        HistoricalDownloadFailureKind::RateLimited
    );
    assert!(HistoricalDownloadFailureKind::ClosedWithoutCompletion.is_transient());
    assert!(HistoricalDownloadFailureKind::ClosedWithoutCompletion.is_splittable());
}

#[test]
fn websocket_failures_never_include_provider_response_content() {
    let message = sanitized_websocket_failure("md/getChart", Some(403));

    assert!(message.contains("md/getChart"));
    assert!(message.contains("403"));
    assert!(message.contains("provider response body omitted"));
}

#[test]
fn raw_tick_element_ceiling_is_treated_as_incomplete_coverage() {
    let window = DownloadWindow::new(
        Utc.with_ymd_and_hms(2026, 7, 23, 0, 0, 0)
            .single()
            .expect("start"),
        Utc.with_ymd_and_hms(2026, 7, 23, 1, 0, 0)
            .single()
            .expect("end"),
    )
    .expect("window");
    let mut telemetry = HistoricalDownloadTelemetry::new(window);
    telemetry.provider_rows = 4_096;

    assert!(raw_tick_element_limit_reached(
        &mut telemetry,
        Some(4_096),
        Some(
            Utc.with_ymd_and_hms(2026, 7, 23, 0, 30, 0)
                .single()
                .expect("provider first")
        ),
        window.start,
    ));
    assert_eq!(telemetry.cap, DownloadCapClassification::Yes);

    telemetry.provider_rows = 4_095;
    assert!(!raw_tick_element_limit_reached(
        &mut telemetry,
        Some(4_096),
        Some(window.start),
        window.start,
    ));
    assert_eq!(telemetry.cap, DownloadCapClassification::No);
}

#[test]
fn capped_page_moves_cursor_to_oldest_provider_tick() {
    let window = DownloadWindow::new(
        Utc.with_ymd_and_hms(2026, 7, 28, 13, 0, 0)
            .single()
            .expect("start"),
        Utc.with_ymd_and_hms(2026, 7, 28, 14, 0, 0)
            .single()
            .expect("end"),
    )
    .expect("window");
    let oldest = Utc
        .with_ymd_and_hms(2026, 7, 28, 13, 57, 58)
        .single()
        .expect("oldest");
    let mut telemetry = HistoricalDownloadTelemetry::new(window);
    telemetry.cap = DownloadCapClassification::Yes;
    telemetry.provider_first_timestamp = Some(oldest);

    assert_eq!(
        raw_tick_next_page_cursor(&telemetry, window.start, window.end)
            .expect("pagination decision"),
        Some(oldest)
    );
}

#[test]
fn short_page_or_page_reaching_start_completes_coverage() {
    let window = DownloadWindow::new(
        Utc.with_ymd_and_hms(2026, 7, 28, 0, 0, 0)
            .single()
            .expect("start"),
        Utc.with_ymd_and_hms(2026, 7, 28, 1, 0, 0)
            .single()
            .expect("end"),
    )
    .expect("window");
    let mut telemetry = HistoricalDownloadTelemetry::new(window);
    telemetry.cap = DownloadCapClassification::No;
    assert_eq!(
        raw_tick_next_page_cursor(&telemetry, window.start, window.end)
            .expect("short page completes"),
        None
    );

    telemetry.cap = DownloadCapClassification::Yes;
    telemetry.provider_first_timestamp = Some(window.start);
    assert_eq!(
        raw_tick_next_page_cursor(&telemetry, window.start, window.end).expect("start reached"),
        None
    );
}

#[test]
fn pagination_rejects_a_non_decreasing_oldest_timestamp() {
    let window = DownloadWindow::new(
        Utc.with_ymd_and_hms(2026, 7, 28, 0, 0, 0)
            .single()
            .expect("start"),
        Utc.with_ymd_and_hms(2026, 7, 28, 1, 0, 0)
            .single()
            .expect("end"),
    )
    .expect("window");
    let mut telemetry = HistoricalDownloadTelemetry::new(window);
    telemetry.cap = DownloadCapClassification::Yes;
    telemetry.provider_first_timestamp = Some(window.end);

    let error = raw_tick_next_page_cursor(&telemetry, window.start, window.end)
        .expect_err("non-progressing page must fail closed");
    assert!(error.to_string().contains("no backward progress"));
}

#[test]
fn overlapping_page_boundary_tick_ids_are_deduplicated() {
    let timestamp = Utc
        .with_ymd_and_hms(2026, 7, 28, 13, 57, 58)
        .single()
        .expect("timestamp");
    let row = ReplayCacheRawTickRow {
        timestamp,
        ts_ns: timestamp.timestamp_nanos_opt().expect("timestamp nanos"),
        tick_id: Some(77),
        price: 6_400.0,
        size: 1.0,
        bid_price: None,
        bid_size: None,
        ask_price: None,
        ask_size: None,
        chart_id: Some(1),
        trade_date: Some(20260728),
        packet_source: Some("db".to_string()),
        packet_base_ts_ms: Some(timestamp.timestamp_millis()),
        packet_base_price_ticks: Some(25_600),
    };

    let normalized = normalize_raw_tick_rows(vec![row.clone(), row]);
    assert_eq!(normalized.rows.len(), 1);
    assert_eq!(normalized.duplicate_tick_ids, 1);
}
