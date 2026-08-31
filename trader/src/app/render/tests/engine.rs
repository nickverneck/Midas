use super::support::*;

#[test]
fn app_starts_on_engine_select() {
    let mut app = App::new(AppConfig::default());

    assert_eq!(app.screen, Screen::EngineSelect);
    assert_eq!(app.focus, Focus::EngineList);
    assert_eq!(app.selected_engine, 0);
    assert!(!app.awaiting_broker_selection());
    assert!(app.take_engine_selection_action().is_none());
}

#[cfg(feature = "replay")]
#[test]
fn engine_screen_f7_explains_replay_requires_engine_session() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();

    app.handle_key(key(KeyCode::F(7)), &cmd_tx);

    assert_eq!(app.screen, Screen::EngineSelect);
    assert!(app.status.contains("create-engine modal"));
    assert!(app.header_help_text().contains("Broker/Replay modal"));
    assert!(cmd_rx.try_recv().is_err());
}

#[test]
fn entering_engine_session_uses_old_broker_start_flow() {
    let mut app = App::new(AppConfig::default());

    app.enter_engine_session(PathBuf::from("/tmp/trader-engine.sock"));

    if compiled_brokers().len() > 1 {
        assert_eq!(app.screen, Screen::BrokerSelect);
        assert_eq!(app.focus, Focus::BrokerList);
        assert!(app.awaiting_broker_selection());
    } else {
        assert_eq!(app.screen, Screen::Login);
        assert_eq!(app.focus, Focus::Env);
        assert!(!app.awaiting_broker_selection());
    }
}

#[cfg(feature = "replay")]
#[test]
fn creating_replay_engine_uses_replay_only_navigation() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();

    app.enter_engine_session_for_key_with_mode(
        EngineKey::from_socket_path(PathBuf::from("/tmp/trader-replay-engine.sock").as_path()),
        PathBuf::from("/tmp/trader-replay-engine.sock"),
        EngineCreateMode::Replay,
    );

    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.focus, Focus::ReplayInstrumentQuery);
    assert_eq!(
        app.header_tab_titles(),
        vec!["Engine", "Replay", "Strategy", "Dashboard", "Analytics"]
    );
    assert!(!app.session_stats_affordance_visible());
    assert!(!app.header_help_text().contains("F1 login"));
    assert!(!app.header_help_text().contains("F2 selection"));

    app.handle_key(key(KeyCode::F(1)), &cmd_tx);
    app.handle_key(key(KeyCode::F(2)), &cmd_tx);
    app.handle_key(key(KeyCode::F(6)), &cmd_tx);
    assert_eq!(app.screen, Screen::Replay);
}

#[cfg(feature = "replay")]
#[test]
fn reentering_observed_replay_engine_restores_replay_navigation() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);

    // The engine overview observer learns the session kind from ReplayState
    // before the user selects the existing engine.
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::Connected {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_name: Some("Replay".to_string()),
            auth_mode: AuthMode::TokenFile,
            session_kind: SessionKind::Replay,
            capabilities: BrokerCapabilities::default(),
        },
        false,
        &cmd_tx,
    );

    app.enter_engine_session_for_key_with_mode(
        key.clone(),
        PathBuf::from("/tmp/trader-engine-10.sock"),
        EngineCreateMode::Replay,
    );
    app.leave_active_engine_session("Returned to engine overview.");

    app.enter_engine_session_for_key(key, PathBuf::from("/tmp/trader-engine-10.sock"));

    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.focus, Focus::ReplayInstrumentQuery);
    assert_eq!(app.session_mode, EngineCreateMode::Replay);
    assert_eq!(app.session_kind, SessionKind::Replay);
}

#[cfg(feature = "replay")]
#[test]
fn replay_disconnect_does_not_fall_back_to_broker_login() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let socket = PathBuf::from("/tmp/trader-replay-disconnect.sock");
    app.enter_engine_session_for_key_with_mode(
        EngineKey::from_socket_path(socket.as_path()),
        socket,
        EngineCreateMode::Replay,
    );

    app.handle_service_event(ServiceEvent::Disconnected, &cmd_tx);

    assert_eq!(app.screen, Screen::Replay);
    assert_eq!(app.focus, Focus::ReplayInstrumentQuery);
    assert!(!app.header_tab_titles().contains(&"Login"));
    assert!(!app.header_tab_titles().contains(&"Stats"));
}

#[test]
fn startup_disconnected_event_keeps_broker_picker_visible() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();

    app.enter_engine_session(PathBuf::from("/tmp/trader-engine.sock"));

    app.handle_service_event(ServiceEvent::Disconnected, &cmd_tx);

    if compiled_brokers().len() > 1 {
        assert_eq!(app.screen, Screen::BrokerSelect);
        assert_eq!(app.focus, Focus::BrokerList);
        assert!(app.awaiting_broker_selection());
    } else {
        assert_eq!(app.screen, Screen::Login);
        assert_eq!(app.focus, Focus::Env);
        assert!(!app.awaiting_broker_selection());
    }
}

#[test]
fn engine_picker_navigation_wraps_through_create_option() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);

    assert_eq!(app.selected_engine, 0);

    app.handle_engine_select_key(key(KeyCode::Down));
    assert_eq!(app.selected_engine, 1);

    app.handle_engine_select_key(key(KeyCode::Down));
    assert_eq!(app.selected_engine, 2);

    app.handle_engine_select_key(key(KeyCode::Down));
    assert_eq!(app.selected_engine, 0);

    app.handle_engine_select_key(key(KeyCode::Up));
    assert_eq!(app.selected_engine, 2);
}

#[cfg(not(feature = "replay"))]
#[test]
fn engine_picker_enter_on_create_emits_broker_create_action_without_replay() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);
    app.selected_engine = app.running_engines.len();

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::CreateNew {
            mode: EngineCreateMode::Broker,
        })
    );
}

#[cfg(feature = "replay")]
#[test]
fn engine_picker_create_opens_mode_modal_before_spawning() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);
    app.selected_engine = app.running_engines.len();

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert_eq!(
        app.pending_engine_create_mode,
        Some(EngineCreateMode::Broker)
    );
    assert!(app.take_engine_selection_action().is_none());

    app.handle_engine_select_key(key(KeyCode::Right));
    assert_eq!(
        app.pending_engine_create_mode,
        Some(EngineCreateMode::Replay)
    );
    app.handle_engine_select_key(key(KeyCode::Enter));

    assert!(app.pending_engine_create_mode.is_none());
    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::CreateNew {
            mode: EngineCreateMode::Replay,
        })
    );
}

#[cfg(feature = "replay")]
#[test]
fn engine_picker_mode_modal_escape_does_not_spawn() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);
    app.selected_engine = app.running_engines.len();

    app.handle_engine_select_key(key(KeyCode::Enter));
    app.handle_engine_select_key(key(KeyCode::Esc));

    assert!(app.pending_engine_create_mode.is_none());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("canceled"));
}

#[test]
fn engine_picker_create_disabled_by_no_spawn_removes_create_option() {
    let mut app = App::new(AppConfig::default());
    app.set_engine_creation_enabled(false);

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert!(!app.engine_create_affordance_visible());
    assert_eq!(app.engine_select_item_count(), app.engine_summaries.len());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("No live engines"));

    app.set_running_engines(vec![running_engine(10, true)]);
    app.handle_engine_select_key(key(KeyCode::Down));
    assert_eq!(app.selected_engine, 0);
}

#[test]
fn engine_picker_enter_on_live_engine_emits_attach_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::Attach {
            engine_key: EngineKey::from_socket_path(
                PathBuf::from("/tmp/trader-engine-10.sock").as_path()
            ),
            socket_path: PathBuf::from("/tmp/trader-engine-10.sock")
        })
    );
}

#[test]
fn engine_picker_enter_on_stale_engine_stays_on_picker() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, false)]);

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert_eq!(app.screen, Screen::EngineSelect);
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("stale"));
}

#[test]
fn engine_picker_ctrl_k_opens_kill_confirmation_without_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('k'));

    let confirmation = app
        .pending_engine_lifecycle_confirmation
        .as_ref()
        .expect("expected kill confirmation");
    assert_eq!(confirmation.action, EngineLifecycleAction::Kill);
    assert_eq!(confirmation.id, 10);
    assert_eq!(confirmation.state, EngineConnectionState::Observing);
    assert_eq!(
        confirmation.socket_path,
        PathBuf::from("/tmp/trader-engine-10.sock")
    );
    assert!(app.take_engine_selection_action().is_none());
}

#[test]
fn engine_picker_plain_k_and_x_do_not_start_destructive_actions() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(key(KeyCode::Char('k')));
    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());

    app.handle_engine_select_key(key(KeyCode::Char('x')));
    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());
}

#[test]
fn engine_picker_kill_cancel_emits_no_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('k'));
    app.handle_engine_select_key(key(KeyCode::Esc));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("Canceled kill"));
}

#[test]
fn engine_picker_kill_confirm_emits_kill_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('k'));
    app.handle_engine_select_key(key(KeyCode::Enter));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::Kill { id: 10 })
    );
}

#[test]
#[cfg(feature = "manual-orders")]
fn engine_picker_ctrl_x_opens_close_and_kill_confirmation_for_live_engine() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key,
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            runtime: ExecutionRuntimeSnapshot {
                armed: true,
                last_summary: "armed and tracking".to_string(),
                ..ExecutionRuntimeSnapshot::default()
            },
            selected_account_id: Some(7),
            selected_contract_name: Some("ESZ6".to_string()),
            market_position_qty: 3,
            ..ExecutionStateSnapshot::default()
        }),
        false,
        &cmd_tx,
    );

    app.handle_engine_select_key(ctrl_key('x'));

    let confirmation = app
        .pending_engine_lifecycle_confirmation
        .as_ref()
        .expect("expected close-and-kill confirmation");
    assert_eq!(confirmation.action, EngineLifecycleAction::CloseAndKill);
    assert_eq!(confirmation.id, 10);
    assert!(confirmation.broker_mode.contains("Tradovate"));
    assert_eq!(confirmation.instrument, "ESZ6");
    assert_eq!(confirmation.position, "3");
    assert_eq!(confirmation.latest_status, "armed and tracking");
    assert!(app.take_engine_selection_action().is_none());
}

#[test]
#[cfg(not(feature = "manual-orders"))]
fn engine_picker_ctrl_x_disabled_without_manual_orders() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('x'));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("Close-and-kill is disabled"));
}

#[test]
#[cfg(feature = "manual-orders")]
fn engine_picker_close_and_kill_cancel_emits_no_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('x'));
    app.handle_engine_select_key(key(KeyCode::Char('n')));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("Canceled close and kill"));
}

#[test]
#[cfg(feature = "manual-orders")]
fn engine_picker_close_and_kill_confirm_emits_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_select_key(ctrl_key('x'));
    app.handle_engine_select_key(key(KeyCode::Char('y')));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::CloseAndKill { id: 10 })
    );
}

#[test]
#[cfg(feature = "manual-orders")]
fn engine_picker_close_and_kill_refuses_stale_engine() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, false)]);

    app.handle_engine_select_key(ctrl_key('x'));

    assert!(app.pending_engine_lifecycle_confirmation.is_none());
    assert!(app.take_engine_selection_action().is_none());
    assert!(app.status.contains("Cannot close and kill engine 10"));
    assert!(app.status.contains("stale"));
}

#[test]
fn engine_summary_updates_from_replay_state_events() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::MarketSnapshot(MarketSnapshot {
            contract_name: Some("ESZ6".to_string()),
            status: "streaming".to_string(),
            ..MarketSnapshot::default()
        }),
        false,
        &cmd_tx,
    );
    let execution = ExecutionStateSnapshot {
        runtime: ExecutionRuntimeSnapshot {
            armed: true,
            last_summary: "armed and tracking".to_string(),
            ..ExecutionRuntimeSnapshot::default()
        },
        selected_account_id: Some(7),
        selected_contract_name: Some("ESZ6".to_string()),
        market_position_qty: 3,
        ..ExecutionStateSnapshot::default()
    };
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::ExecutionState(execution),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::Latency(LatencySnapshot {
            rest_rtt_ms: Some(42),
            ..LatencySnapshot::default()
        }),
        false,
        &cmd_tx,
    );

    let summary = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == key)
        .expect("expected engine summary");
    assert_eq!(summary.connection_state, EngineConnectionState::Connected);
    assert!(summary.broker_mode_label().contains("Tradovate"));
    assert_eq!(summary.account_label(), "SIM");
    assert_eq!(summary.instrument_label(), "ESZ6");
    assert_eq!(summary.position_label(), "3");
    assert_eq!(summary.latency_label(), "42ms");
    assert!(summary.strategy_label().contains("HMA Angle armed"));
    assert_eq!(summary.status_label(), "armed and tracking");
}

#[test]
fn engine_summary_tracks_disconnect_and_error_events() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_service_event(key.clone(), ServiceEvent::Disconnected, false, &cmd_tx);

    let summary = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == key)
        .expect("expected engine summary");
    assert_eq!(
        summary.connection_state,
        EngineConnectionState::Disconnected
    );
    assert_eq!(summary.status_label(), "Disconnected");

    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::Error("ipc failed".to_string()),
        false,
        &cmd_tx,
    );
    let summary = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == key)
        .expect("expected engine summary");
    assert_eq!(summary.connection_state, EngineConnectionState::Error);
    assert_eq!(summary.status_label(), "ipc failed");
}

#[test]
fn broker_rejection_is_prominent_without_marking_engine_connection_failed() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);

    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        true,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::BrokerRejection("GCQ6 liquidation only".to_string()),
        true,
        &cmd_tx,
    );

    let summary = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == key)
        .expect("expected engine summary");
    assert_eq!(summary.connection_state, EngineConnectionState::Connected);
    assert_eq!(summary.status_label(), "Rejected: GCQ6 liquidation only");
    assert_eq!(app.status, "Rejected: GCQ6 liquidation only");
    assert!(
        app.persisted_logs
            .iter()
            .any(|entry| entry.message == "REJECTED: GCQ6 liquidation only")
    );
}

#[test]
fn inactive_engine_events_do_not_mutate_detail_state() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let active_key = engine_key(10);
    let inactive_key = engine_key(11);
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);
    app.enter_engine_session_for_key(active_key, PathBuf::from("/tmp/trader-engine-10.sock"));
    app.screen = Screen::Dashboard;
    app.status = "detail stable".to_string();

    app.handle_engine_service_event(
        inactive_key.clone(),
        connected_event(BrokerKind::Ironbeam),
        false,
        &cmd_tx,
    );

    assert_eq!(app.screen, Screen::Dashboard);
    assert_eq!(app.status, "detail stable");
    let inactive = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == inactive_key)
        .expect("expected inactive summary");
    assert_eq!(inactive.connection_state, EngineConnectionState::Connected);
    assert!(inactive.broker_mode_label().contains("Ironbeam"));
}

#[test]
fn active_engine_events_preserve_detail_behavior() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));

    app.handle_engine_service_event(key, connected_event(BrokerKind::Tradovate), true, &cmd_tx);

    assert_eq!(app.screen, Screen::Selection);
    assert_eq!(app.focus, Focus::AccountList);
    assert!(app.status.contains("Connected to Tradovate"));
}

#[test]
fn attaching_to_armed_engine_restores_dashboard_and_market_selection() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    let socket = PathBuf::from("/tmp/trader-engine-10.sock");
    app.set_running_engines(vec![running_engine(10, true)]);

    // The engine overview observer normally receives this before the user
    // presses Enter, which lets attach skip the login/selection workflow.
    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            runtime: ExecutionRuntimeSnapshot {
                armed: true,
                last_summary: "armed and tracking".to_string(),
                ..ExecutionRuntimeSnapshot::default()
            },
            bar_type: Some(BarType::range(1)),
            candle_mode: Some(CandleMode::Standard),
            ..ExecutionStateSnapshot::default()
        }),
        false,
        &cmd_tx,
    );

    app.enter_engine_session_for_key(key, socket);

    assert_eq!(app.screen, Screen::Dashboard);
    assert_eq!(app.focus, Focus::AccountList);
    assert_eq!(app.bar_type, BarType::range(1));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn attach_hydration_restores_disarmed_engine_before_selection_is_chosen() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    let socket = PathBuf::from("/tmp/trader-engine-10.sock");
    app.set_running_engines(vec![running_engine(10, true)]);

    let mut execution_config = ExecutionStateSnapshot::default().config;
    execution_config.native_strategy = NativeStrategyKind::HmaCross;
    execution_config.native_hma_cross.fast_length = 3;
    execution_config.native_hma_cross.slow_length = 10;
    let execution = ExecutionStateSnapshot {
        config: execution_config,
        runtime: ExecutionRuntimeSnapshot {
            armed: false,
            last_summary: "strategy idle".to_string(),
            ..ExecutionRuntimeSnapshot::default()
        },
        bar_type: Some(BarType::range(10)),
        candle_mode: Some(CandleMode::Standard),
        selected_account_id: Some(7),
        selected_contract_name: Some("GCZ6".to_string()),
        ..ExecutionStateSnapshot::default()
    };

    // The TUI attach barrier first feeds the overview summary while the App
    // is still on its neutral engine screen. No default HMA Angle/empty
    // Selection screen is selected during this phase.
    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        false,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::ExecutionState(execution.clone()),
        false,
        &cmd_tx,
    );
    assert_eq!(app.screen, Screen::EngineSelect);

    // Once the summary is hydrated, activation restores the engine's actual
    // strategy and market selection, then the initial events are applied to
    // the active detail state exactly once.
    app.enter_engine_session_for_key(key.clone(), socket);
    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        true,
        &cmd_tx,
    );
    app.handle_engine_service_event(key, ServiceEvent::ExecutionState(execution), true, &cmd_tx);

    assert_eq!(app.screen, Screen::Selection);
    assert_eq!(app.focus, Focus::AccountList);
    assert_eq!(app.strategy.native_strategy, NativeStrategyKind::HmaCross);
    assert_eq!(app.strategy.native_hma_cross.fast_length, 3);
    assert_eq!(app.strategy.native_hma_cross.slow_length, 10);
    assert_eq!(app.bar_type, BarType::range(10));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn execution_state_restores_contract_without_market_snapshot() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));

    app.handle_engine_service_event(
        key,
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            selected_contract_name: Some("GCZ6".to_string()),
            ..ExecutionStateSnapshot::default()
        }),
        true,
        &cmd_tx,
    );

    assert_eq!(app.market.contract_name.as_deref(), Some("GCZ6"));
    assert_eq!(app.instrument_query, "GCZ6");
    assert!(
        rendered_text(app.selection_summary_lines())
            .iter()
            .any(|line| line == "Last subscribed contract: GCZ6")
    );
}

#[test]
fn execution_state_does_not_overwrite_newer_market_contract() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));

    app.handle_engine_service_event(
        key.clone(),
        ServiceEvent::MarketSnapshot(MarketSnapshot {
            contract_name: Some("ESZ6".to_string()),
            ..MarketSnapshot::default()
        }),
        true,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        key,
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            selected_contract_name: Some("GCZ6".to_string()),
            ..ExecutionStateSnapshot::default()
        }),
        true,
        &cmd_tx,
    );

    assert_eq!(app.market.contract_name.as_deref(), Some("ESZ6"));
}

#[test]
fn attach_falls_back_to_dashboard_when_armed_state_arrives_after_connect() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));

    app.handle_engine_service_event(
        key.clone(),
        connected_event(BrokerKind::Tradovate),
        true,
        &cmd_tx,
    );
    assert_eq!(app.screen, Screen::Selection);

    app.handle_engine_service_event(
        key,
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            runtime: ExecutionRuntimeSnapshot {
                armed: true,
                ..ExecutionRuntimeSnapshot::default()
            },
            bar_type: Some(BarType::range(10)),
            candle_mode: Some(CandleMode::Standard),
            ..ExecutionStateSnapshot::default()
        }),
        true,
        &cmd_tx,
    );

    assert_eq!(app.screen, Screen::Dashboard);
    assert_eq!(app.bar_type, BarType::range(10));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn attached_minute_engine_renders_observed_candle_mode() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));

    app.handle_engine_service_event(
        key,
        ServiceEvent::ExecutionState(ExecutionStateSnapshot {
            runtime: ExecutionRuntimeSnapshot {
                armed: true,
                ..ExecutionRuntimeSnapshot::default()
            },
            bar_type: Some(BarType::minute(1)),
            candle_mode: Some(CandleMode::HeikinAshi),
            ..ExecutionStateSnapshot::default()
        }),
        true,
        &cmd_tx,
    );

    let dashboard = rendered_text(app.dashboard_summary_lines());
    assert!(dashboard.iter().any(|line| line == "Bar Type: 1 Min"));
    assert!(dashboard.iter().any(|line| line == "Candles: Heikin Ashi"));
}

#[test]
fn active_engine_header_label_includes_identity_state_and_other_count() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    let active_key = engine_key(10);
    let other_key = engine_key(11);
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);
    app.enter_engine_session_for_key(
        active_key.clone(),
        PathBuf::from("/tmp/trader-engine-10.sock"),
    );

    app.handle_engine_service_event(
        active_key,
        connected_event(BrokerKind::Tradovate),
        true,
        &cmd_tx,
    );
    app.handle_engine_service_event(
        other_key,
        connected_event(BrokerKind::Ironbeam),
        false,
        &cmd_tx,
    );

    let label = app.active_engine_header_label();
    assert!(label.contains("#10"));
    assert!(label.contains("trader-engine-10.sock"));
    assert!(label.contains("connected"));
    assert!(label.contains("Live"));
    assert!(label.contains("+1 other"));
}

#[test]
fn active_engine_receiver_close_returns_to_engine_overview() {
    let mut app = App::new(AppConfig::default());
    let key = engine_key(10);
    app.set_running_engines(vec![running_engine(10, true)]);
    app.enter_engine_session_for_key(key.clone(), PathBuf::from("/tmp/trader-engine-10.sock"));
    app.screen = Screen::Dashboard;
    app.focus = Focus::AccountList;

    app.handle_engine_receiver_closed(&key, true);

    assert_eq!(app.screen, Screen::EngineSelect);
    assert_eq!(app.focus, Focus::EngineList);
    assert!(app.engine_socket_path.is_none());
    assert!(app.active_engine_key.is_none());
    assert_eq!(
        app.status,
        "Engine #10 trader-engine-10.sock connection closed; last state was live."
    );
    let summary = app
        .engine_summaries
        .iter()
        .find(|summary| summary.key == key)
        .expect("expected engine summary");
    assert_eq!(summary.connection_state, EngineConnectionState::Closed);
    assert_eq!(summary.status_label(), app.status);
}
