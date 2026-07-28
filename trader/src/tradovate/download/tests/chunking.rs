use super::super::chunked::{
    effective_raw_tick_request_end, raw_tick_failure_should_split, replay_download_cancelled,
};
use crate::replay_download::{
    DownloadWindow, HistoricalDownloadFailure, HistoricalDownloadFailureKind,
    HistoricalDownloadTelemetry,
};
use chrono::{TimeZone, Utc};

#[test]
fn cancellation_reflects_the_latest_watch_value() {
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    assert!(!replay_download_cancelled(Some(&cancel_rx)));
    cancel_tx
        .send(true)
        .expect("cancellation receiver remains open");
    assert!(replay_download_cancelled(Some(&cancel_rx)));
}

#[test]
fn absent_cancellation_channel_is_not_cancelled() {
    assert!(!replay_download_cancelled(None));
}

#[test]
fn active_day_request_is_clamped_to_snapshot_time() {
    let start = Utc
        .with_ymd_and_hms(2026, 7, 28, 0, 0, 0)
        .single()
        .expect("start");
    let requested_end = Utc
        .with_ymd_and_hms(2026, 7, 29, 0, 0, 0)
        .single()
        .expect("requested end");
    let snapshot = Utc
        .with_ymd_and_hms(2026, 7, 28, 20, 15, 0)
        .single()
        .expect("snapshot");

    assert_eq!(
        effective_raw_tick_request_end(start, requested_end, snapshot, chrono::Duration::hours(1))
            .expect("active day clamps"),
        Utc.with_ymd_and_hms(2026, 7, 28, 20, 0, 0)
            .single()
            .expect("completed hour")
    );
}

#[test]
fn completed_historical_request_keeps_its_requested_end() {
    let start = Utc
        .with_ymd_and_hms(2026, 7, 27, 0, 0, 0)
        .single()
        .expect("start");
    let requested_end = Utc
        .with_ymd_and_hms(2026, 7, 28, 0, 0, 0)
        .single()
        .expect("requested end");
    let snapshot = Utc
        .with_ymd_and_hms(2026, 7, 28, 20, 15, 0)
        .single()
        .expect("snapshot");

    assert_eq!(
        effective_raw_tick_request_end(start, requested_end, snapshot, chrono::Duration::hours(1))
            .expect("historical end remains exact"),
        requested_end
    );
}

#[test]
fn entirely_future_request_is_rejected() {
    let start = Utc
        .with_ymd_and_hms(2026, 7, 29, 0, 0, 0)
        .single()
        .expect("start");
    let end = Utc
        .with_ymd_and_hms(2026, 7, 30, 0, 0, 0)
        .single()
        .expect("end");
    let snapshot = Utc
        .with_ymd_and_hms(2026, 7, 28, 20, 15, 0)
        .single()
        .expect("snapshot");

    assert!(
        effective_raw_tick_request_end(start, end, snapshot, chrono::Duration::hours(1)).is_err()
    );
}

#[test]
fn heartbeat_only_socket_close_retries_but_does_not_split() {
    let window = DownloadWindow::new(
        Utc.with_ymd_and_hms(2026, 7, 28, 0, 0, 0)
            .single()
            .expect("start"),
        Utc.with_ymd_and_hms(2026, 7, 28, 1, 0, 0)
            .single()
            .expect("end"),
    )
    .expect("window");
    let failure = HistoricalDownloadFailure {
        kind: HistoricalDownloadFailureKind::ClosedWithoutCompletion,
        message: "socket closed".to_string(),
        telemetry: HistoricalDownloadTelemetry::new(window),
        retry_after_ms: None,
    };

    assert!(failure.kind.is_transient());
    assert!(!raw_tick_failure_should_split(&failure));
}

#[test]
fn provider_rows_or_explicit_size_rejection_can_split() {
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
    telemetry.provider_rows = 1;
    let incomplete = HistoricalDownloadFailure {
        kind: HistoricalDownloadFailureKind::ClosedWithoutCompletion,
        message: "incomplete".to_string(),
        telemetry,
        retry_after_ms: None,
    };
    assert!(raw_tick_failure_should_split(&incomplete));

    let size_rejected = HistoricalDownloadFailure {
        kind: HistoricalDownloadFailureKind::SizeRejected,
        message: "too large".to_string(),
        telemetry: HistoricalDownloadTelemetry::new(window),
        retry_after_ms: None,
    };
    assert!(raw_tick_failure_should_split(&size_rejected));
}
