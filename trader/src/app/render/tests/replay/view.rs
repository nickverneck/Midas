use super::super::support::*;

#[cfg(feature = "replay")]
#[test]
fn replay_run_lines_label_disabled_start_states() {
    let mut missing_config = AppConfig::default();
    missing_config.replay_file_path =
        std::env::temp_dir().join("trader-replay-disabled-missing.Last.txt");
    let missing_app = App::new(missing_config);

    let missing_lines = rendered_text(missing_app.replay_run_control_lines());
    assert!(
        missing_lines
            .iter()
            .any(|line| line == "[Enter] Start Replay (missing dataset)")
    );

    let path = replay_test_file("disabled-volume");
    let mut config = AppConfig::default();
    config.replay_file_path = path;
    let mut volume_app = App::new(config);
    volume_app.bar_type = BarType::volume(6500);

    let volume_lines = rendered_text(volume_app.replay_run_control_lines());
    assert!(
        volume_lines
            .iter()
            .any(|line| line == "[Enter] Start Replay (volume unavailable)")
    );
    let market_lines = rendered_text(volume_app.replay_market_control_lines());
    assert!(
        market_lines
            .iter()
            .any(|line| line == "Volume needs per-trade size; this Last file only has price.")
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_run_lines_identify_the_selected_engine_mode() {
    let mut config = AppConfig::default();
    config.replay_engine_mode = crate::broker::ReplayEngineMode::Deterministic;
    config.replay_fixed_latency_ms = 75;
    let app = App::new(config);

    let lines = rendered_text(app.replay_run_control_lines());
    assert!(
        lines
            .iter()
            .any(|line| line == "Replay engine: Deterministic virtual time")
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "Fill model: raw next-bar open")
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "Latency model: fixed (75ms)")
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "Bar protection: conservative stop-first")
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_run_lines_mark_fixed_latency_as_ignored_in_legacy_mode() {
    let mut config = AppConfig::default();
    config.replay_fixed_latency_ms = 75;
    let app = App::new(config);

    let lines = rendered_text(app.replay_run_control_lines());
    assert!(
        lines
            .iter()
            .any(|line| line == "Latency model: ignored in Legacy")
    );
}

#[test]
fn dashboard_summary_hides_replay_speed_in_live_mode() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = SessionKind::Live;

    let lines = app
        .dashboard_summary_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert!(lines.iter().any(|line| line == "Mode: Live"));
    assert!(!lines.iter().any(|line| line.contains("Replay Speed")));
}

#[test]
fn dashboard_summary_shows_replay_speed_in_replay_mode() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = SessionKind::Replay;
    app.replay_speed = ReplaySpeed::X5;

    let lines = app
        .dashboard_summary_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert!(lines.iter().any(|line| line == "Mode: Replay"));
    assert!(lines.iter().any(|line| line == "Replay Speed: 5x"));
    assert!(lines.iter().any(|line| {
        line == "Replay Ledger: 0 fill(s) | gross +0.00 | ignored in Legacy | schema v2"
    }));
}

#[test]
fn dashboard_summary_labels_warmup_and_evaluation_progress() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = SessionKind::Replay;
    app.market.replay_window = Some(crate::broker::ReplayWindowSnapshot {
        preset: "Futures RTH (New York)".to_string(),
        input_timezone: "America/New_York".to_string(),
        warmup_start: "2026-07-23T12:30:00Z".parse().unwrap(),
        evaluation_start: "2026-07-23T13:30:00Z".parse().unwrap(),
        evaluation_end: "2026-07-23T20:00:00Z".parse().unwrap(),
        warmup_rows: 60,
        evaluation_rows_total: 390,
        evaluation_rows_processed: 12,
    });

    let lines = rendered_text(app.dashboard_summary_lines());
    assert!(
        lines
            .iter()
            .any(|line| line == "Replay Window: Futures RTH (New York) | America/New_York")
    );
    assert!(lines.iter().any(|line| {
        line == "Replay Range: 2026-07-23 09:30:00 EDT to 2026-07-23 16:00:00 EDT"
    }));
    assert!(
        lines
            .iter()
            .any(|line| line == "Replay Rows: warmup 60 | evaluation 12/390")
    );
}

#[test]
fn replay_speed_hotkeys_send_replay_only_commands() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.session_kind = SessionKind::Replay;

    app.handle_dashboard_key(key(KeyCode::Char(']')), &cmd_tx);

    match cmd_rx.try_recv().expect("expected replay-speed command") {
        ServiceCommand::SetReplaySpeed {
            speed: ReplaySpeed::X2,
        } => {}
        _ => panic!("expected replay-speed command"),
    }
    assert_eq!(app.replay_speed, ReplaySpeed::X2);

    app.handle_dashboard_key(key(KeyCode::Char('0')), &cmd_tx);

    match cmd_rx.try_recv().expect("expected realtime-speed command") {
        ServiceCommand::SetReplaySpeed {
            speed: ReplaySpeed::Realtime,
        } => {}
        _ => panic!("expected realtime-speed command"),
    }
    assert_eq!(app.replay_speed, ReplaySpeed::Realtime);
}
