use super::super::support::*;

#[cfg(feature = "replay")]
#[test]
fn login_replay_shortcut_opens_replay_screen_without_starting() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Login;
    app.focus = Focus::Env;
    app.bar_type = BarType::tick(25);

    app.handle_key(key(KeyCode::Char('r')), &cmd_tx);

    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.focus, Focus::BarTypeToggle);
    assert_eq!(app.bar_type, BarType::tick(25));
    assert!(cmd_rx.try_recv().is_err());
    assert!(app.header_tab_titles().contains(&"Replay"));
}

#[cfg(feature = "replay")]
#[test]
fn login_replay_focus_opens_replay_screen_without_starting() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Login;
    app.focus = Focus::ReplayMode;
    app.bar_type = BarType::second(5);

    app.handle_login_key(key(KeyCode::Enter), &cmd_tx);

    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.focus, Focus::BarTypeToggle);
    assert_eq!(app.bar_type, BarType::second(5));
    assert!(cmd_rx.try_recv().is_err());
}

#[cfg(feature = "replay")]
#[test]
fn replay_screen_start_uses_selected_market_controls() {
    let path = replay_test_file("start-selected");
    let mut config = AppConfig::default();
    config.replay_file_path = path.clone();
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayMode;
    app.bar_type = BarType::tick(100);
    app.candle_mode = CandleMode::HeikinAshi;

    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    match cmd_rx.try_recv().expect("expected replay start command") {
        ServiceCommand::EnterReplayMode {
            bar_type,
            candle_mode,
            config,
            replay_dataset_manifest,
            replay_dataset_view,
        } => {
            assert_eq!(bar_type, BarType::tick(100));
            assert_eq!(candle_mode, CandleMode::HeikinAshi);
            assert_eq!(config.replay_file_path, path);
            assert!(replay_dataset_manifest.is_none());
            assert!(replay_dataset_view.is_none());
        }
        _ => panic!("expected enter-replay command"),
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_lines_expose_ready_and_missing_metadata() {
    let path = replay_test_file("metadata");
    let mut config = AppConfig::default();
    config.replay_file_path = path;
    let app = App::new(config);

    let ready = rendered_text(app.replay_dataset_library_lines());
    assert!(
        ready
            .iter()
            .any(|line| line == "Configured Local Replay File")
    );
    assert!(
        ready
            .iter()
            .any(|line| line.starts_with("Configured file: trader-replay-metadata"))
    );
    assert!(ready.iter().any(|line| line == "Status: ready"));
    assert!(
        ready
            .iter()
            .any(|line| line.starts_with("Inferred contract: trader-replay-metadata"))
    );
    assert!(
        ready
            .iter()
            .any(|line| line == "Available bars: seconds, minutes, tick-count, range")
    );

    let mut missing_config = AppConfig::default();
    missing_config.replay_file_path =
        std::env::temp_dir().join("trader-replay-missing-file.Last.txt");
    let missing_app = App::new(missing_config);
    let missing = rendered_text(missing_app.replay_dataset_library_lines());
    assert!(
        missing
            .iter()
            .any(|line| line == "Status: missing local file")
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_lines_show_owned_cache_manifests_first() {
    let cache_root = replay_cache_test_root("ui");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let path = replay_test_file("cache-lines");
    let mut config = AppConfig::default();
    config.replay_file_path = path;
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::HeikinAshi;

    let lines = rendered_text(app.replay_dataset_library_lines());

    assert_eq!(
        lines.first().map(String::as_str),
        Some("Owned Cached Datasets")
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "Status: 1 manifest(s), selected dataset none")
    );
    assert!(lines.iter().any(|line| line
        == "Up/Down browses; Enter uses the selected cache; N adds data; D extends selected coverage; A uses automatic resolution."));
    assert!(
        lines
            .iter()
            .any(|line| line == "  Dataset [1]: MESU6 RTH 1m Heikin")
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("Shapes: 1 Min | Modes: OHLC, Heikin Ashi"))
    );
    assert!(
        lines
            .iter()
            .any(|line| line == &format!("  Manifest: {}", manifest_path.display()))
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "Configured Local Replay File")
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_compact_lines_fit_an_eighty_column_terminal() {
    let cache_root = replay_cache_test_root("compact-ui");
    write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    let mut app = App::new(config);
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::HeikinAshi;

    let dataset_lines = rendered_text(app.replay_dataset_library_compact_lines());
    let market_lines = rendered_text(app.replay_market_control_compact_lines());
    let run_lines = rendered_text(app.replay_run_control_compact_lines());

    assert!(dataset_lines.iter().all(|line| line.chars().count() <= 42));
    assert!(market_lines.iter().all(|line| line.chars().count() <= 33));
    assert!(run_lines.iter().all(|line| line.chars().count() <= 33));
    assert!(dataset_lines.iter().all(|line| !line.contains("Manifest:")));
}

#[cfg(feature = "replay")]
#[test]
fn replay_screen_start_accepts_matching_cached_jsonl_without_local_file() {
    let cache_root = replay_cache_test_root("start-cache");
    write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    config.replay_file_path =
        std::env::temp_dir().join("trader-replay-cache-only-missing.Last.txt");
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayMode;
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::HeikinAshi;

    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    match cmd_rx
        .try_recv()
        .expect("expected replay start command from cache")
    {
        ServiceCommand::EnterReplayMode {
            bar_type,
            candle_mode,
            replay_dataset_manifest,
            ..
        } => {
            assert_eq!(bar_type, BarType::minute(1));
            assert_eq!(candle_mode, CandleMode::HeikinAshi);
            assert!(replay_dataset_manifest.is_none());
        }
        _ => panic!("expected enter-replay command"),
    }
    assert!(app.logs.iter().any(|line| {
        line.message
            .contains("Replay mode requested: Heikin Ashi 1 Min (cache)")
    }));
}

#[cfg(feature = "replay")]
#[test]
fn replay_auto_selection_recognizes_downloaded_server_bar_parquet() {
    let cache_root = replay_cache_test_root("auto-parquet");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
            .expect("parse manifest");
    manifest["files"][0]["format"] = json!("parquet");
    manifest["files"][0]["relative_path"] =
        json!("server-bars/2026-07-23_to_2026-07-24_1minute.parquet");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write parquet manifest");
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    config.replay_file_path =
        std::env::temp_dir().join("trader-replay-auto-parquet-missing.Last.txt");
    let mut app = App::new(config);
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::Standard;

    assert_eq!(app.replay_dataset_index, None);
    assert!(app.replay_dataset_available());
    assert!(app.replay_cache_can_serve_selected_bar());
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_picker_selects_manifest_for_startup() {
    let cache_root = replay_cache_test_root("picker");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    config.replay_file_path = std::env::temp_dir().join("trader-replay-picker-missing.Last.txt");
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayDataset;
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::HeikinAshi;

    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.replay_dataset_index, Some(0));
    app.handle_replay_key(key(KeyCode::Char('a')), &cmd_tx);
    assert_eq!(app.replay_dataset_index, None);
    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.replay_dataset_index, Some(0));
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    assert_eq!(app.focus, Focus::ReplayMode);
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    match cmd_rx.try_recv().expect("expected replay start command") {
        ServiceCommand::EnterReplayMode {
            replay_dataset_manifest,
            ..
        } => assert_eq!(replay_dataset_manifest, Some(manifest_path)),
        _ => panic!("expected enter-replay command"),
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_view_tui_creates_edits_selects_and_starts_saved_view() {
    let cache_root = replay_cache_test_root("view-tui");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    config.replay_file_path = std::env::temp_dir().join("trader-replay-view-tui-missing.Last.txt");
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayDataset;
    app.bar_type = BarType::minute(1);
    app.candle_mode = CandleMode::HeikinAshi;

    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    app.handle_replay_key(key(KeyCode::Char('v')), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::DatasetViews);
    assert_eq!(app.focus, Focus::ReplayViewList);

    app.handle_replay_key(key(KeyCode::Char('n')), &cmd_tx);
    assert_eq!(app.focus, Focus::ReplayViewId);
    assert_eq!(
        app.replay_dataset_views
            .editor
            .as_ref()
            .expect("new editor")
            .preset,
        ReplayDatasetSessionPreset::FullSource
    );
    app.focus = Focus::ReplayViewSave;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    let view_path = cache_root.join(".views/mesu6_2026_07_23.json");
    assert_eq!(app.replay_dataset_view_path.as_ref(), Some(&view_path));
    assert!(view_path.is_file());
    assert_eq!(app.focus, Focus::ReplayViewList);

    app.handle_replay_key(key(KeyCode::Char('e')), &cmd_tx);
    app.focus = Focus::ReplayViewPreset;
    for _ in 0..5 {
        app.handle_replay_key(key(KeyCode::Right), &cmd_tx);
    }
    assert_eq!(
        app.replay_dataset_views
            .editor
            .as_ref()
            .expect("edit editor")
            .preset,
        ReplayDatasetSessionPreset::CustomUtc
    );
    app.focus = Focus::ReplayViewSave;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    let saved = crate::replay_cache::ReplayDatasetViewStore::new(&cache_root)
        .load("mesu6_2026_07_23")
        .expect("saved view");
    assert_eq!(saved.session_preset, ReplayDatasetSessionPreset::CustomUtc);

    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::Library);
    app.focus = Focus::ReplayMode;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    match cmd_rx.try_recv().expect("view-backed replay command") {
        ServiceCommand::EnterReplayMode {
            replay_dataset_manifest,
            replay_dataset_view,
            ..
        } => {
            assert_eq!(replay_dataset_manifest, Some(manifest_path));
            assert_eq!(replay_dataset_view, Some(view_path));
        }
        _ => panic!("expected enter-replay command"),
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_view_tui_keeps_editor_open_after_validation_failure() {
    let cache_root = replay_cache_test_root("view-tui-validation");
    write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayDataset;

    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    app.handle_replay_key(key(KeyCode::Char('v')), &cmd_tx);
    app.handle_replay_key(key(KeyCode::Char('n')), &cmd_tx);
    app.replay_dataset_views.editor.as_mut().expect("editor").id = "bad/id".to_string();
    app.focus = Focus::ReplayViewSave;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    assert!(app.replay_dataset_views.editor.is_some());
    assert!(app.replay_dataset_views.message.contains("dataset view id"));
    assert!(!cache_root.join(".views/bad/id.json").exists());
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_downloader_uses_selected_manifest_coverage() {
    let cache_root = replay_cache_test_root("download-command");
    write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayDataset;

    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    app.handle_key(key(KeyCode::Char('d')), &cmd_tx);

    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(
        app.replay_downloader.workflow,
        ReplayDownloadWorkflow::Extend
    );
    assert_eq!(app.replay_downloader.start_date, "2026-07-23");
    assert_eq!(app.replay_downloader.end_date, "2026-07-23");
    assert_eq!(
        app.replay_downloader.source_kind,
        Some(ReplayCacheSourceKind::ServerBars)
    );

    app.focus = Focus::ReplayDownloadSubmit;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);

    match cmd_rx.try_recv().expect("expected download command") {
        ServiceCommand::DownloadReplayData {
            instrument,
            contract,
            target,
            start_date,
            end_date,
            source_kind,
            display_name,
            ..
        } => {
            assert_eq!(instrument, "MES");
            assert_eq!(contract.name, "MESU6");
            assert_eq!(contract.id, 25_866_054);
            let target = target.expect("extend target");
            assert_eq!(
                target.manifest_path,
                cache_root.join("tradovate/sim/MES/MESU6/2026-07-23/manifest.json")
            );
            assert_eq!(start_date.to_string(), "2026-07-23");
            assert_eq!(end_date.to_string(), "2026-07-23");
            assert_eq!(source_kind, "server-bars");
            assert_eq!(display_name.as_deref(), Some("MESU6 RTH 1m Heikin"));
        }
        _ => panic!("expected replay download command"),
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_dataset_downloader_rejects_ambiguous_mixed_sources() {
    let cache_root = replay_cache_test_root("download-mixed");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
            .expect("parse manifest");
    manifest["source_kind"] = json!("mixed");
    manifest["files"]
        .as_array_mut()
        .expect("manifest files")
        .push(json!({
            "relative_path": "raw-ticks/2026-07-23_to_2026-07-24_ticks.parquet",
            "source_kind": "raw_ticks",
            "format": "parquet",
            "schema_version": 1,
            "compression": "snappy",
            "market_shape": {"session_template": "Globex"},
            "row_count": 100,
            "first_timestamp": "2026-07-23T13:30:00Z",
            "last_timestamp": "2026-07-23T20:00:00Z"
        }));
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write mixed manifest");

    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayDataset;

    app.handle_replay_key(key(KeyCode::Down), &cmd_tx);
    app.handle_key(key(KeyCode::Char('d')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(app.replay_downloader.source_kind, None);
    assert_eq!(app.focus, Focus::ReplayDownloadSource);
    assert!(
        app.replay_downloader
            .phase_message
            .contains("Mixed dataset")
    );

    app.handle_replay_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(
        app.replay_downloader.source_kind,
        Some(ReplayCacheSourceKind::RawTicks)
    );
    app.focus = Focus::ReplayDownloadSubmit;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    match cmd_rx.try_recv().expect("explicit raw tick download") {
        ServiceCommand::DownloadReplayData { source_kind, .. } => {
            assert_eq!(source_kind, "raw-ticks")
        }
        _ => panic!("expected replay download command"),
    }
}
