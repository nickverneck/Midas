use super::super::support::*;

#[cfg(feature = "replay")]
#[test]
fn replay_new_dataset_search_inspection_and_submit_stay_read_only() {
    let cache_root = replay_cache_test_root("new-download");
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Replay;
    app.focus = Focus::ReplayMode;

    app.handle_replay_key(key(KeyCode::Char('n')), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(app.focus, Focus::ReplayDownloadInstrument);
    assert_eq!(app.replay_downloader.workflow, ReplayDownloadWorkflow::New);

    app.replay_downloader.instrument_query = "MES".to_string();
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    let search_operation_id = match cmd_rx.try_recv().expect("replay contract search") {
        ServiceCommand::SearchReplayDownloadContracts {
            operation_id,
            query,
            config,
            ..
        } => {
            assert_eq!(query, "MES");
            assert_eq!(config.env, TradingEnvironment::Sim);
            assert_eq!(config.replay_cache_dir, cache_root);
            operation_id
        }
        _ => panic!("expected replay-only contract search command"),
    };
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Searching);

    let contract = ContractSuggestion {
        id: 4_399_631,
        name: "MESU6".to_string(),
        description: "Micro E-mini S&P Sep 2026".to_string(),
        raw: json!({"id": 4_399_631, "name": "MESU6"}),
    };
    app.handle_service_event(
        ServiceEvent::ReplayDownloadContractSearchResults {
            operation_id: search_operation_id,
            query: "MES".to_string(),
            results: vec![contract.clone()],
        },
        &cmd_tx,
    );
    app.focus = Focus::ReplayDownloadContract;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    let inspect_operation_id = match cmd_rx.try_recv().expect("contract metadata inspection") {
        ServiceCommand::InspectReplayDownloadContract {
            operation_id,
            contract: selected,
            ..
        } => {
            assert_eq!(selected.id, contract.id);
            operation_id
        }
        _ => panic!("expected replay contract inspection command"),
    };

    app.handle_service_event(
        ServiceEvent::ReplayDownloadContractInspected {
            operation_id: inspect_operation_id,
            contract,
            suggested_start_date: Some(
                chrono::NaiveDate::from_ymd_opt(2026, 6, 18).expect("start date"),
            ),
            suggested_end_date: Some(
                chrono::NaiveDate::from_ymd_opt(2026, 9, 18).expect("end date"),
            ),
            suggestion_basis: Some("adjacent maturity expirations".to_string()),
        },
        &cmd_tx,
    );
    assert_eq!(app.replay_downloader.start_date, "2026-06-18");
    assert_eq!(app.replay_downloader.end_date, "2026-09-18");
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Ready);

    app.replay_downloader.display_name = "MES U6 research".to_string();
    app.replay_downloader.tags = "baseline, hma".to_string();
    app.replay_downloader.bar_type = BarType::volume(6500);
    app.replay_downloader.candle_mode = CandleMode::HeikinAshi;
    app.focus = Focus::ReplayDownloadSubmit;
    app.handle_replay_key(key(KeyCode::Enter), &cmd_tx);
    match cmd_rx.try_recv().expect("replay download command") {
        ServiceCommand::DownloadReplayData {
            contract,
            source_kind,
            bar_type,
            candle_mode,
            display_name,
            tags,
            ..
        } => {
            assert_eq!(contract.name, "MESU6");
            assert_eq!(contract.id, 4_399_631);
            assert_eq!(source_kind, "server-bars");
            assert_eq!(bar_type, BarType::volume(6500));
            assert_eq!(candle_mode, CandleMode::HeikinAshi);
            assert_eq!(display_name.as_deref(), Some("MES U6 research"));
            assert_eq!(tags, vec!["baseline", "hma"]);
        }
        _ => panic!("expected replay download command"),
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_reports_progress_and_actual_cache_size() {
    let cache_root = replay_cache_test_root("download-progress");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.open_new_replay_downloader();
    let operation_id = app.replay_downloader.begin_operation();

    app.handle_service_event(
        ServiceEvent::ReplayDownloadProgress {
            operation_id,
            phase: ReplayDownloadPhase::Downloading,
            message: "Downloading raw ticks...".to_string(),
            estimated_rows: None,
            estimated_bytes: None,
        },
        &cmd_tx,
    );
    assert_eq!(
        app.replay_downloader.phase,
        ReplayDownloadPhase::Downloading
    );
    assert_eq!(app.replay_downloader.estimated_rows, None);

    app.handle_service_event(
        ServiceEvent::ReplayDownloadCompleted {
            operation_id,
            cache_root: cache_root.clone(),
            manifest_path: manifest_path.clone(),
            data_path: manifest_path.with_file_name("bars.parquet"),
            rows: 1_380,
            bytes: 40_411,
        },
        &cmd_tx,
    );
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Complete);
    assert_eq!(app.replay_downloader.actual_rows, Some(1_380));
    assert_eq!(app.replay_downloader.actual_bytes, Some(40_411));
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_ignores_every_stale_operation_event() {
    let cache_root = replay_cache_test_root("stale-events");
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.open_new_replay_downloader();
    let active = app.replay_downloader.begin_operation();
    let stale = ReplayDownloadOperationId::next();
    assert_ne!(active, stale);

    app.handle_service_event(
        ServiceEvent::ReplayDownloadContractSearchResults {
            operation_id: stale,
            query: "ES".to_string(),
            results: vec![contract(2, "ESU6")],
        },
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::ReplayDownloadContractInspected {
            operation_id: stale,
            contract: contract(2, "ESU6"),
            suggested_start_date: None,
            suggested_end_date: None,
            suggestion_basis: None,
        },
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::ReplayDownloadProgress {
            operation_id: stale,
            phase: ReplayDownloadPhase::WritingCache,
            message: "stale progress".to_string(),
            estimated_rows: Some(10),
            estimated_bytes: Some(20),
        },
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::ReplayDownloadCompleted {
            operation_id: stale,
            cache_root: cache_root.join("other"),
            manifest_path: cache_root.join("other/manifest.json"),
            data_path: cache_root.join("other/data.parquet"),
            rows: 10,
            bytes: 20,
        },
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::ReplayDownloadFailed {
            operation_id: stale,
            phase: ReplayDownloadPhase::Downloading,
            message: "stale failure".to_string(),
        },
        &cmd_tx,
    );

    assert_eq!(app.replay_downloader.active_operation_id, Some(active));
    assert!(app.replay_downloader.contract_results.is_empty());
    assert!(app.replay_downloader.exact_contract.is_none());
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Idle);
    assert_eq!(app.replay_downloader.actual_rows, None);
    assert_eq!(app.base_config.replay_cache_dir, cache_root);
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_waits_for_escape_cancellation_before_leaving() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    let escaped = app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::Downloading;
    app.handle_key(key(KeyCode::Esc), &cmd_tx);
    assert_eq!(app.replay_downloader.active_operation_id, Some(escaped));
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Cancelling);
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(ServiceCommand::CancelReplayDownloadOperation { operation_id })
            if operation_id == escaped
    ));

    app.handle_service_event(
        ServiceEvent::ReplayDownloadFailed {
            operation_id: escaped,
            phase: ReplayDownloadPhase::Cancelled,
            message: "cancelled before cache commit".to_string(),
        },
        &cmd_tx,
    );
    assert_eq!(app.replay_downloader.active_operation_id, None);
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Cancelled);
    app.handle_key(key(KeyCode::Esc), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::Library);
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_invalidates_changed_lookups_but_preserves_active_new_workflow() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    let environment = app.replay_downloader.begin_operation();
    app.focus = Focus::ReplayDownloadEnv;
    app.handle_replay_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.replay_downloader.active_operation_id, None);
    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(ServiceCommand::CancelReplayDownloadOperation { operation_id })
            if operation_id == environment
    ));

    let query = app.replay_downloader.begin_operation();
    app.focus = Focus::ReplayDownloadInstrument;
    app.handle_replay_key(key(KeyCode::Char('M')), &cmd_tx);
    assert_eq!(app.replay_downloader.active_operation_id, None);
    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(ServiceCommand::CancelReplayDownloadOperation { operation_id })
            if operation_id == query
    ));

    let active_download = app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::Downloading;
    app.open_new_replay_downloader();
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(active_download)
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_escape_cannot_hide_an_in_progress_cache_commit() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    let operation_id = app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::Downloading;

    app.handle_key(key(KeyCode::Esc), &cmd_tx);

    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Cancelling);
    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(ServiceCommand::CancelReplayDownloadOperation { operation_id: sent })
            if sent == operation_id
    ));

    app.handle_service_event(
        ServiceEvent::ReplayDownloadProgress {
            operation_id,
            phase: ReplayDownloadPhase::WritingCache,
            message: "Cancellation cannot interrupt an atomic cache commit that has already begun; the commit will finish."
                .to_string(),
            estimated_rows: None,
            estimated_bytes: None,
        },
        &cmd_tx,
    );

    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(
        app.replay_downloader.phase,
        ReplayDownloadPhase::WritingCache
    );
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert!(
        app.replay_downloader
            .phase_message
            .contains("cannot interrupt")
    );
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_downloading_f7_then_new_or_extend_preserves_active_operation() {
    let cache_root = replay_cache_test_root("active-navigation");
    write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root;
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.open_new_replay_downloader();
    let operation_id = app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::Downloading;
    app.replay_downloader.instrument_query = "ACTIVE".to_string();
    app.screen = Screen::Dashboard;
    app.focus = Focus::AccountList;

    app.handle_key(key(KeyCode::F(7)), &cmd_tx);
    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(app.focus, Focus::ReplayDownloadSubmit);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );

    app.handle_key(key(KeyCode::Char('n')), &cmd_tx);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert_eq!(app.replay_downloader.instrument_query, "ACTIVE");

    app.handle_key(key(KeyCode::Char('d')), &cmd_tx);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert_eq!(app.replay_downloader.instrument_query, "ACTIVE");

    app.replay_view = ReplayView::Library;
    app.focus = Focus::ReplayMode;
    app.handle_replay_key(key(KeyCode::Char('n')), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert_eq!(app.replay_downloader.instrument_query, "ACTIVE");

    app.replay_view = ReplayView::Library;
    app.focus = Focus::ReplayDataset;
    app.handle_replay_key(key(KeyCode::Char('d')), &cmd_tx);
    assert_eq!(app.replay_view, ReplayView::Downloader);
    assert_eq!(
        app.replay_downloader.active_operation_id,
        Some(operation_id)
    );
    assert_eq!(app.replay_downloader.instrument_query, "ACTIVE");
    assert!(app.status.contains("press Esc to cancel or wait"));
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_writing_cache_fkeys_reopen_and_retain_the_job() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.open_new_replay_downloader();
    let operation_id = app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::WritingCache;

    for code in [
        KeyCode::F(1),
        KeyCode::F(2),
        KeyCode::F(3),
        KeyCode::F(4),
        KeyCode::F(6),
        KeyCode::F(7),
    ] {
        app.screen = Screen::Dashboard;
        app.focus = Focus::AccountList;
        app.handle_key(key(code), &cmd_tx);
        assert_eq!(app.screen, Screen::Replay);
        assert_eq!(app.replay_view, ReplayView::Downloader);
        assert_eq!(
            app.replay_downloader.active_operation_id,
            Some(operation_id)
        );
        assert!(app.status.contains("wait for completion"));
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_terminal_events_restore_normal_navigation_and_workflows() {
    for (index, terminal_phase) in [
        ReplayDownloadPhase::Complete,
        ReplayDownloadPhase::Cancelled,
        ReplayDownloadPhase::Failed,
    ]
    .into_iter()
    .enumerate()
    {
        let cache_root = replay_cache_test_root(&format!("terminal-navigation-{index}"));
        let manifest_path = write_replay_cache_manifest(&cache_root);
        let mut config = AppConfig::default();
        config.replay_cache_dir = cache_root.clone();
        let mut app = App::new(config);
        app.session_mode = EngineCreateMode::Replay;
        let (cmd_tx, _cmd_rx) = unbounded_channel();
        app.screen = Screen::Replay;
        app.open_new_replay_downloader();
        let operation_id = app.replay_downloader.begin_operation();
        app.replay_downloader.phase = ReplayDownloadPhase::Downloading;

        if terminal_phase == ReplayDownloadPhase::Complete {
            app.handle_service_event(
                ServiceEvent::ReplayDownloadCompleted {
                    operation_id,
                    cache_root: cache_root.clone(),
                    manifest_path: manifest_path.clone(),
                    data_path: manifest_path.with_file_name("bars.parquet"),
                    rows: 1,
                    bytes: 1,
                },
                &cmd_tx,
            );
        } else {
            app.handle_service_event(
                ServiceEvent::ReplayDownloadFailed {
                    operation_id,
                    phase: terminal_phase,
                    message: terminal_phase.label().to_string(),
                },
                &cmd_tx,
            );
        }

        assert_eq!(app.replay_downloader.active_operation_id, None);
        app.handle_key(key(KeyCode::F(1)), &cmd_tx);
        assert_eq!(app.screen, Screen::Replay);
        app.handle_key(key(KeyCode::F(7)), &cmd_tx);
        assert_eq!(app.screen, Screen::Replay);
        assert_eq!(app.replay_view, ReplayView::Library);
        app.focus = Focus::ReplayMode;
        app.handle_key(key(KeyCode::Char('n')), &cmd_tx);
        assert_eq!(app.replay_view, ReplayView::Downloader);
        assert_eq!(app.replay_downloader.workflow, ReplayDownloadWorkflow::New);
        assert_eq!(app.replay_downloader.active_operation_id, None);
    }
}

#[cfg(feature = "replay")]
#[test]
fn replay_download_failure_preserves_active_cache_root_and_library() {
    let cache_root = replay_cache_test_root("failed-root");
    let manifest_path = write_replay_cache_manifest(&cache_root);
    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.open_new_replay_downloader();
    app.replay_downloader.cache_root = cache_root.join("replacement").display().to_string();
    let operation_id = app.replay_downloader.begin_operation();

    app.handle_service_event(
        ServiceEvent::ReplayDownloadFailed {
            operation_id,
            phase: ReplayDownloadPhase::WritingCache,
            message: "disk full".to_string(),
        },
        &cmd_tx,
    );

    assert_eq!(app.base_config.replay_cache_dir, cache_root);
    assert_eq!(app.replay_cache_library.datasets.len(), 1);
    assert_eq!(
        app.replay_cache_library.datasets[0].manifest_path,
        manifest_path
    );
    assert_eq!(app.replay_downloader.phase, ReplayDownloadPhase::Failed);
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_remains_operable_at_eighty_by_twenty_four() {
    let mut app = App::new(AppConfig::default());
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    app.focus = Focus::ReplayDownloadSubmit;
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");

    terminal
        .draw(|frame| app.draw(frame))
        .expect("draw downloader");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("Dataset Downloader"));
    assert!(rendered.contains("Start read-only download"));
    assert!(rendered.contains("Exact Contract"));
    assert!(rendered.contains("Download Job"));
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_shows_cancellable_guidance_at_eighty_by_twenty_four() {
    let mut app = App::new(AppConfig::default());
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::Downloading;
    app.replay_downloader.phase_message = "Downloading raw ticks...".to_string();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");

    terminal
        .draw(|frame| app.draw(frame))
        .expect("draw cancellable downloader");

    let rows = rendered_terminal_rows(&terminal, 80);
    assert!(
        rows.iter()
            .any(|row| row.contains("Esc requests cancellation."))
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("Wait for acknowledgement."))
    );
    assert!(!rows.iter().any(|row| row.contains("no cancellation")));
}

#[cfg(feature = "replay")]
#[test]
fn replay_downloader_shows_commit_guidance_at_eighty_by_twenty_four() {
    let mut app = App::new(AppConfig::default());
    app.screen = Screen::Replay;
    app.open_new_replay_downloader();
    app.replay_downloader.begin_operation();
    app.replay_downloader.phase = ReplayDownloadPhase::WritingCache;
    app.replay_downloader.phase_message = "Writing replay cache...".to_string();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");

    terminal
        .draw(|frame| app.draw(frame))
        .expect("draw committing downloader");

    let rows = rendered_terminal_rows(&terminal, 80);
    assert!(
        rows.iter()
            .any(|row| row.contains("Commit is non-interruptible."))
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("Esc will not hide this job."))
    );
    assert!(!rows.iter().any(|row| row.contains("no cancellation")));
}

#[cfg(feature = "replay")]
#[test]
fn replay_download_refresh_preserves_selected_manifest_after_resort() {
    let cache_root = replay_cache_test_root("selection-resort");
    let mes_manifest_path = write_replay_cache_manifest(&cache_root);
    let mut es_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mes_manifest_path).expect("read MES manifest"))
            .expect("parse MES manifest");
    es_manifest["instrument"]["symbol"] = json!("ES");
    es_manifest["contract"]["symbol"] = json!("ESU6");
    es_manifest["display_name"] = json!("ESU6 RTH 1m");
    es_manifest["coverage"]["end"] = json!("2026-07-23T21:00:00Z");
    let es_manifest_path = cache_root.join("tradovate/sim/ES/ESU6/2026-07-23/manifest.json");
    std::fs::create_dir_all(es_manifest_path.parent().expect("ES manifest parent"))
        .expect("create ES dataset");
    std::fs::write(
        &es_manifest_path,
        serde_json::to_vec_pretty(&es_manifest).expect("serialize ES manifest"),
    )
    .expect("write ES manifest");

    let mut config = AppConfig::default();
    config.replay_cache_dir = cache_root.clone();
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let mes_index = app
        .replay_cache_library
        .datasets
        .iter()
        .position(|dataset| dataset.manifest_path == mes_manifest_path)
        .expect("MES dataset index");
    assert_eq!(mes_index, 1);
    app.replay_dataset_index = Some(mes_index);
    let operation_id = app.replay_downloader.begin_operation();

    let mut mes_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mes_manifest_path).expect("read MES manifest"))
            .expect("parse MES manifest");
    mes_manifest["coverage"]["end"] = json!("2026-07-23T23:59:00Z");
    std::fs::write(
        &mes_manifest_path,
        serde_json::to_vec_pretty(&mes_manifest).expect("serialize updated MES manifest"),
    )
    .expect("write updated MES manifest");

    app.handle_service_event(
        ServiceEvent::ReplayDownloadCompleted {
            operation_id,
            cache_root: cache_root.clone(),
            manifest_path: mes_manifest_path.clone(),
            data_path: mes_manifest_path.with_file_name("bars.parquet"),
            rows: 1380,
            bytes: 40_411,
        },
        &cmd_tx,
    );

    let selected = app
        .replay_dataset_index
        .and_then(|index| app.replay_cache_library.datasets.get(index))
        .expect("selected dataset after refresh");
    assert_eq!(selected.manifest_path, mes_manifest_path);
    assert_eq!(app.replay_dataset_index, Some(0));
}
