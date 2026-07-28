use super::super::worker::{
    replay_gap_duration, replay_history_loaded, replay_window_progress, scale_duration,
};
use crate::broker::{Bar, ReplayWindowSnapshot};
use std::time::Duration;

#[test]
fn replay_gap_duration_uses_market_timestamps() {
    let gap = replay_gap_duration(Some(1_000_000_000), 61_000_000_000, 5);
    assert_eq!(gap, Duration::from_secs(60));

    let zero_gap = replay_gap_duration(Some(61_000_000_000), 61_000_000_000, 5);
    assert_eq!(zero_gap, Duration::ZERO);
}

#[test]
fn scale_duration_applies_replay_multiplier() {
    assert_eq!(
        scale_duration(Duration::from_secs(60), 0.1),
        Duration::from_secs(6)
    );
    assert_eq!(
        scale_duration(Duration::from_millis(250), 2.0),
        Duration::from_millis(500)
    );
}

#[test]
fn dataset_view_history_stops_exactly_before_evaluation() {
    let bars = (0..5)
        .map(|index| Bar {
            ts_ns: index * 10,
            open: index as f64,
            high: index as f64,
            low: index as f64,
            close: index as f64,
            volume: Some(1.0),
        })
        .collect::<Vec<_>>();

    assert_eq!(replay_history_loaded(&bars, 100, Some(20)), 2);
    assert_eq!(replay_history_loaded(&bars, 3, None), 3);
}

#[test]
fn replay_window_progress_counts_only_rows_after_warmup() {
    let template = ReplayWindowSnapshot {
        preset: "Futures RTH (New York)".to_string(),
        input_timezone: "America/New_York".to_string(),
        warmup_start: "2026-07-23T12:30:00Z".parse().unwrap(),
        evaluation_start: "2026-07-23T13:30:00Z".parse().unwrap(),
        evaluation_end: "2026-07-23T20:00:00Z".parse().unwrap(),
        warmup_rows: 0,
        evaluation_rows_total: 0,
        evaluation_rows_processed: 0,
    };

    let progress = replay_window_progress(Some(&template), 60, 390, 12).unwrap();
    assert_eq!(progress.warmup_rows, 60);
    assert_eq!(progress.evaluation_rows_total, 390);
    assert_eq!(progress.evaluation_rows_processed, 12);
    assert_eq!(progress.evaluation_rows_remaining(), 378);

    let capped = replay_window_progress(Some(&template), 60, 390, 500).unwrap();
    assert_eq!(capped.evaluation_rows_processed, 390);
}
