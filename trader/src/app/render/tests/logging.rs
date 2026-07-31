use super::support::*;

#[test]
fn persisted_log_body_includes_session_stats_summary_and_events() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_000.0)]),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_015.0)]),
        &cmd_tx,
    );

    let body = app.build_persisted_log_body("20260403T120000Z");

    assert!(body.contains("[session_stats]"));
    assert!(body.contains("enabled: true"));
    assert!(body.contains("account_name: SIM"));
    assert!(body.contains("session_pnl: +15.00"));
    assert!(body.contains("delta=+15.00"));
}

#[test]
fn persisted_log_body_records_replay_warmup_and_evaluation_split() {
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

    let body = app.build_persisted_log_body("20260403T120000Z");

    assert!(body.contains("replay_window_preset: Futures RTH (New York)"));
    assert!(body.contains("replay_window_input_timezone: America/New_York"));
    assert!(body.contains(
        "replay_evaluation_local_range: 2026-07-23 09:30:00 EDT to 2026-07-23 16:00:00 EDT"
    ));
    assert!(body.contains("replay_warmup_rows: 60"));
    assert!(body.contains("replay_evaluation_rows_total: 390"));
    assert!(body.contains("replay_evaluation_rows_processed: 12"));
    assert!(body.contains("replay_evaluation_rows_remaining: 378"));
}

#[test]
fn persisted_log_body_includes_live_engine_review_metadata_without_secret_fields() {
    let mut config = AppConfig {
        password: "super-secret-password".to_string(),
        api_key: "super-secret-api-key".to_string(),
        token_override: "super-secret-token".to_string(),
        token_path: PathBuf::from(".auth/secret-token.json"),
        time_in_force: "GTC".to_string(),
        ..AppConfig::default()
    };
    config.order_qty = 3;
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);
    let active_key = engine_key(10);
    app.enter_engine_session_for_key(
        active_key.clone(),
        PathBuf::from("/tmp/trader-engine-10.sock"),
    );
    app.handle_engine_service_event(
        active_key,
        ServiceEvent::Connected {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_name: Some("tester".to_string()),
            auth_mode: AuthMode::TokenFile,
            session_kind: SessionKind::Replay,
            capabilities: BrokerCapabilities {
                replay: true,
                manual_orders: true,
                automated_orders: true,
                native_protection: true,
            },
        },
        true,
        &cmd_tx,
    );
    app.handle_service_event(ServiceEvent::ReplaySpeedUpdated(ReplaySpeed::X10), &cmd_tx);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );
    app.contract_results = vec![contract(99, "ESZ6")];
    app.selected_contract = 0;
    app.handle_service_event(
        ServiceEvent::Latency(LatencySnapshot {
            rest_rtt_ms: Some(42),
            last_order_ack_ms: Some(17),
            last_order_seen_ms: Some(13),
            last_exec_report_ms: Some(19),
            last_fill_ms: Some(23),
            last_signal_submit_ms: Some(3),
            last_signal_seen_ms: Some(5),
            last_signal_ack_ms: Some(7),
            last_signal_fill_ms: Some(11),
        }),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            runtime: ExecutionRuntimeSnapshot {
                armed: true,
                pending_target_qty: Some(2),
                last_summary: "strategy decision | buy".to_string(),
                ..ExecutionRuntimeSnapshot::default()
            },
            selected_account_id: Some(7),
            selected_contract_name: Some("ESZ6".to_string()),
            ..ExecutionStateSnapshot::default()
        }),
        &cmd_tx,
    );

    let body = app.build_persisted_log_body("20260403T120000Z");

    assert!(body.contains("engine_socket: /tmp/trader-engine-10.sock"));
    assert!(body.contains("engine_id: 10"));
    assert!(body.contains("engine_connection_state: connected"));
    assert!(body.contains("active_engine_key_display: /tmp/trader-engine-10.sock"));
    assert!(body.contains("active_engine_key: /tmp/trader-engine-10.sock"));
    assert!(body.contains("other_live_engine_count: 1"));
    assert!(body.contains("other_live_engines: 1"));
    assert!(body.contains("session_kind: Replay"));
    assert!(body.contains("replay_speed: 10x"));
    assert!(body.contains("replay_engine_mode: Legacy compatibility"));
    assert!(body.contains("replay_fill_model: legacy_reference_price"));
    assert!(body.contains("replay_fixed_latency_ms: 0"));
    assert!(body.contains("replay_ledger_schema_version: 3"));
    assert!(body.contains("replay_ledger_fill_count: 0"));
    assert!(body.contains("replay_ledger_gross_realized_pnl: 0.00000000"));
    assert!(body.contains("capability_replay: true"));
    assert!(body.contains("capability_manual_orders: true"));
    assert!(body.contains("capability_automated_orders: true"));
    assert!(body.contains("capability_native_protection: true"));
    assert!(body.contains("strategy_order_qty: 1"));
    assert!(body.contains("order_time_in_force: GTC"));
    assert!(body.contains("strategy_armed: true"));
    assert!(body.contains("pending_target: 2"));
    assert!(body.contains("last_strategy_summary: strategy decision | buy"));
    assert!(body.contains("latency_last_signal_fill_ms: 11"));
    assert!(body.contains("selected_account_id: 7"));
    assert!(body.contains("selected_account_name: SIM"));
    assert!(body.contains("selected_contract_id: 99"));
    assert!(body.contains("selected_contract_name: ESZ6"));
    assert!(!body.contains("super-secret"));
    assert!(!body.contains(".auth/secret-token.json"));
}

#[test]
fn persisted_log_body_redacts_raw_response_bodies_and_structured_payloads() {
    let config = AppConfig {
        log_mode: LogMode::Debug,
        ..AppConfig::default()
    };
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.logs.clear();
    app.persisted_logs.clear();

    app.handle_service_event(
        ServiceEvent::Error(
            "auth request failed (401 Unauthorized): {\"accessToken\":\"super-secret-token\",\"password\":\"hidden\"}"
                .to_string(),
        ),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::DebugLog(
            "request payload {\"apiKey\":\"super-secret-api-key\",\"qty\":1}".to_string(),
        ),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::DebugLog("bulk response [{\"token\":\"array-secret\"}]".to_string()),
        &cmd_tx,
    );
    app.push_log("operator note [manual check]".to_string());

    assert!(app.status.contains("super-secret-token"));
    assert!(
        app.persisted_logs
            .iter()
            .any(|entry| entry.message.contains("super-secret-api-key"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");

    assert!(body.contains(
        "status: Error: auth request failed (401 Unauthorized): [redacted broker response]"
    ));
    assert!(body.contains(
        "final_status: Error: auth request failed (401 Unauthorized): [redacted broker response]"
    ));
    assert!(
        body.contains("ERROR: auth request failed (401 Unauthorized): [redacted broker response]")
    );
    assert!(body.contains("DEBUG: request payload [redacted structured data]"));
    assert!(body.contains("DEBUG: bulk response [redacted structured data]"));
    assert!(body.contains("operator note [manual check]"));
    assert!(!body.contains("super-secret"));
    assert!(!body.contains("array-secret"));
    assert!(!body.contains("accessToken"));
    assert!(!body.contains("apiKey"));
    assert!(!body.contains("password"));
}

#[test]
fn persisted_log_review_summary_counts_logs_and_pnl_before_full_stats() {
    let config = AppConfig {
        log_mode: LogMode::Debug,
        ..AppConfig::default()
    };
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.persisted_logs.clear();
    app.logs.clear();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_000.0)]),
        &cmd_tx,
    );
    let mut latest = balance_snapshot(7, "SIM", 1_015.0);
    latest.realized_pnl = Some(12.25);
    latest.unrealized_pnl = Some(-1.50);
    app.handle_service_event(ServiceEvent::AccountSnapshotsLoaded(vec![latest]), &cmd_tx);
    app.handle_service_event(ServiceEvent::Error("bad fill".to_string()), &cmd_tx);
    app.handle_service_event(ServiceEvent::DebugLog("wire detail".to_string()), &cmd_tx);

    let body = app.build_persisted_log_body("20260403T120000Z");
    let review_index = body.find("[review_summary]").expect("review summary");
    let stats_index = body.find("[session_stats]").expect("session stats");

    assert!(review_index < stats_index);
    assert!(body.contains("selected_account_session_pnl: +15.00"));
    assert!(body.contains("selected_account_trade_pnl_ex_fees: +15.00"));
    assert!(body.contains("selected_account_realized_pnl: +12.25"));
    assert!(body.contains("selected_account_unrealized_pnl: -1.50"));
    assert!(body.contains("persisted_error_count: 1"));
    assert!(body.contains("persisted_debug_count: 1"));
}

#[test]
fn persisted_log_body_handles_disabled_stats_recent_events_clearly() {
    let config = AppConfig {
        session_stats_enabled: false,
        ..AppConfig::default()
    };
    let app = App::new(config);

    let body = app.build_persisted_log_body("20260403T120000Z");

    assert!(body.contains("[recent_session_events]"));
    assert!(body.contains("Tracking is disabled, so no balance-delta events were recorded."));
    assert!(body.contains("[session_stats]"));
    assert!(body.contains("enabled: false"));
    assert!(body.contains("Session stats tracking was disabled for this run."));
    assert!(!body.contains("[session_stats.account]"));
}

#[test]
fn persisted_recent_session_events_respect_hidden_fee_visibility() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.market.tick_size = Some(0.25);
    app.market.value_per_point = Some(5.0);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.00, 1.0),
        balance_snapshot_with_position(7, "SIM", 1_005.69, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_004.78, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_004.78, -1.0),
        balance_snapshot_with_position(7, "SIM", 996.37, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }
    app.handle_session_stats_key(key(KeyCode::Char('f')), &cmd_tx);

    let body = app.build_persisted_log_body("20260403T120000Z");
    let recent = section_between(&body, "[recent_session_events]", "[session_stats]");

    assert!(recent.contains("fees_visible: hidden"));
    assert!(recent.contains("balance short") && recent.contains("trade -7.50"));
    assert!(recent.contains("balance long") && recent.contains("trade +6.25"));
    assert!(!recent.contains("fee trade"));
    assert!(!recent.contains(" fees "));
    assert!(!recent.contains("mixed trade"));
}

#[test]
fn log_panel_lines_include_last_saved_path_stably() {
    let mut app = App::new(AppConfig::default());
    app.last_saved_log_path = Some(PathBuf::from(".run/trader-logs/session-test.txt"));
    for index in 0..10 {
        app.push_log(format!("status update {index}"));
    }

    let text = rendered_text(app.log_panel_lines());

    assert_eq!(
        text.first().map(String::as_str),
        Some("Last saved: .run/trader-logs/session-test.txt")
    );
    assert!(text.iter().any(|line| line.contains("status update 9")));
}

#[test]
fn push_log_records_wall_time_and_elapsed_delta() {
    let mut app = App::new(AppConfig::default());
    app.logs.clear();
    app.persisted_logs.clear();
    app.last_log_at = None;

    app.push_log("first event".to_string());
    std::thread::sleep(std::time::Duration::from_millis(5));
    app.push_log("second event".to_string());

    assert_eq!(app.logs.len(), 2);
    assert_eq!(app.persisted_logs.len(), 2);
    assert_eq!(app.logs[0].elapsed_since_previous, None);
    assert!(app.logs[1].elapsed_since_previous.is_some());
    assert!(app.logs[0].render_line().contains("first event"));
    assert!(app.logs[0].render_line().contains("+0ms"));
    assert!(app.logs[1].render_line().contains("second event"));
    assert!(app.logs[1].render_line().starts_with('['));
}

#[test]
fn persisted_logs_keep_more_entries_than_ui_logs() {
    let mut app = App::new(AppConfig::default());
    app.logs.clear();
    app.persisted_logs.clear();
    app.last_log_at = None;

    for idx in 0..(UI_LOG_ENTRY_LIMIT + 5) {
        app.push_log(format!("event {idx}"));
    }

    assert_eq!(app.logs.len(), UI_LOG_ENTRY_LIMIT);
    assert_eq!(
        app.logs.front().map(|entry| entry.message.as_str()),
        Some("event 5")
    );
    assert_eq!(app.persisted_logs.len(), UI_LOG_ENTRY_LIMIT + 5);
    assert_eq!(
        app.persisted_logs
            .front()
            .map(|entry| entry.message.as_str()),
        Some("event 0")
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("event 0"));
    assert!(body.contains("event 204"));
}

#[test]
fn debug_log_events_are_filtered_by_log_mode() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.logs.clear();
    app.persisted_logs.clear();

    app.handle_service_event(
        ServiceEvent::DebugLog("submit 42ms | order".to_string()),
        &cmd_tx,
    );
    assert!(app.logs.is_empty());
    assert!(app.persisted_logs.is_empty());

    app.form.log_mode = LogMode::Debug;
    app.handle_service_event(
        ServiceEvent::DebugLog("submit 42ms | order".to_string()),
        &cmd_tx,
    );

    assert_eq!(app.logs.len(), 1);
    assert_eq!(app.persisted_logs.len(), 1);
    assert_eq!(app.logs[0].message, "DEBUG: submit 42ms | order");
}
