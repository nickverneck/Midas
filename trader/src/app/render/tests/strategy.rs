use super::support::*;

#[test]
fn strategy_reversal_mode_cycles_through_all_three_options() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Strategy;
    app.focus = Focus::NativeReversalMode;

    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::Direct
    );

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::FlattenConfirmEnter
    );

    app.handle_strategy_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::Direct
    );

    app.handle_strategy_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::CloseAllEnter
    );

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::Direct
    );

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::FlattenConfirmEnter
    );

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::CloseAllEnter
    );
}

#[test]
fn strategy_setup_scrolls_focused_option_into_short_panel() {
    let mut app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut app);
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    app.strategy.native_ema.use_trailing_stop = true;
    app.focus = Focus::EmaTrailOffsetTicks;

    let lines = app.strategy_setup_lines();
    let offset = focused_paragraph_scroll_offset(&lines, Rect::new(0, 0, 120, 7));

    assert!(
        offset > 0,
        "short setup panel should scroll down to the focused strategy option"
    );
    assert_focused_line_visible(&lines, Rect::new(0, 0, 120, 7));
}

#[test]
fn focused_list_state_selects_only_visible_focused_lists() {
    assert_eq!(focused_list_state(true, 7, 10).selected(), Some(7));
    assert_eq!(focused_list_state(false, 7, 10).selected(), None);
    assert_eq!(focused_list_state(true, 10, 10).selected(), None);
}

#[test]
fn strategy_setting_edit_updates_draft_without_service_command() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Strategy;
    app.focus = Focus::NativeReversalMode;

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);

    assert_eq!(
        app.strategy.native_reversal_mode,
        NativeReversalMode::FlattenConfirmEnter
    );
    assert!(
        cmd_rx.try_recv().is_err(),
        "strategy edits should stay draft-only until arm"
    );
}

#[test]
fn strategy_type_hides_lua_and_machine_learning_options() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Strategy;
    app.focus = Focus::StrategyKind;

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.strategy.kind, StrategyKind::Native);

    app.handle_strategy_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.strategy.kind, StrategyKind::Native);

    let notes = rendered_text(app.strategy_notes_lines());
    assert!(notes.iter().all(|line| !line.contains("Lua")));
    assert!(notes.iter().all(|line| !line.contains("Machine Learning")));
}

#[test]
fn monitor_only_strategy_setup_removes_arm_wording() {
    let app = App::new(AppConfig::default());

    let setup = strategy_setup_text(&app);
    let notes = rendered_text(app.strategy_notes_lines());
    let preview = rendered_text(app.strategy_preview_lines());

    assert!(
        setup
            .iter()
            .any(|line| line.contains("Continue / Monitor Only"))
    );
    assert!(setup.iter().all(|line| !line.contains("Arm Strategy")));
    assert!(
        notes
            .iter()
            .any(|line| line.contains("dashboard without arming"))
    );
    assert!(
        preview
            .iter()
            .any(|line| line.contains("monitor-only observation"))
    );
    assert!(
        preview
            .iter()
            .all(|line| !line.contains("automated market orders"))
    );
}

#[test]
fn strategy_readiness_reports_ready_to_arm_with_account_and_contract() {
    let mut app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);

    let readiness = app.strategy_readiness();
    let setup = strategy_setup_text(&app);
    let preview = rendered_text(app.strategy_preview_lines());

    assert_eq!(readiness.status, StrategyReadinessStatus::ReadyToArm);
    assert!(
        setup
            .iter()
            .any(|line| line.contains("Continue / Arm Native Strategy"))
    );
    assert!(preview.iter().any(|line| line == "Readiness: Ready to arm"));
}

#[test]
fn strategy_continue_without_selected_contract_opens_monitor_only_without_arming() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    app.focus = Focus::StrategyContinue;

    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    assert_eq!(app.screen, Screen::Dashboard);
    expect_select_account(&mut cmd_rx, 1);
    assert!(cmd_rx.try_recv().is_err());
}

#[test]
fn strategy_readiness_previews_pre_arm_adjustments() {
    let mut app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::HmaDirect;
    app.strategy.native_reversal_mode = NativeReversalMode::Direct;
    app.strategy.native_ema.take_profit_ticks = 8.0;

    let preview = rendered_text(app.strategy_preview_lines());

    assert!(preview.iter().any(|line| line == "Readiness: Ready to arm"));
    assert!(preview.iter().any(|line| line.contains("CloseAll > Enter")));
    assert!(preview.iter().any(|line| line.contains("Guarded")));
}

#[test]
fn invalid_crossover_lengths_need_attention_and_do_not_arm() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);
    app.screen = Screen::Strategy;
    app.focus = Focus::StrategyContinue;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_ema.fast_length = 20;
    app.strategy.native_ema.slow_length = 20;

    let setup = strategy_setup_text(&app);
    let preview = rendered_text(app.strategy_preview_lines());

    assert!(
        setup
            .iter()
            .any(|line| line.contains("Review Strategy Setup"))
    );
    assert!(
        preview
            .iter()
            .any(|line| line == "Readiness: Needs attention")
    );
    assert!(preview.iter().any(|line| line.contains("Fast EMA Length")));

    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    assert_eq!(app.screen, Screen::Strategy);
    assert!(cmd_rx.try_recv().is_err());
}

#[test]
fn strategy_continue_applies_draft_config_and_arms() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);
    app.screen = Screen::Strategy;
    app.focus = Focus::NativeReversalMode;

    app.handle_strategy_key(key(KeyCode::Right), &cmd_tx);

    assert!(
        cmd_rx.try_recv().is_err(),
        "draft edit should not touch the service"
    );

    app.focus = Focus::StrategyContinue;
    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected config sync command") {
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            assert_eq!(
                config.native_reversal_mode,
                NativeReversalMode::FlattenConfirmEnter
            );
        }
        _ => panic!("expected execution-config command"),
    }
    match cmd_rx.try_recv().expect("expected arm command") {
        ServiceCommand::ArmExecutionStrategy => {}
        _ => panic!("expected arm-execution command"),
    }
}

#[test]
fn manual_disarm_hotkey_sends_explicit_disarm_command() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Dashboard;

    app.handle_key(key(KeyCode::Char('d')), &cmd_tx);

    match cmd_rx.try_recv().expect("expected disarm command") {
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            assert_eq!(reason, "Manual strategy disarm requested.");
        }
        _ => panic!("expected disarm command"),
    }
}

#[test]
fn manual_disarm_hotkey_works_from_numeric_strategy_fields() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Strategy;
    app.focus = Focus::EmaFastLength;

    app.handle_key(key(KeyCode::Char('d')), &cmd_tx);

    match cmd_rx.try_recv().expect("expected disarm command") {
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            assert_eq!(reason, "Manual strategy disarm requested.");
        }
        _ => panic!("expected disarm command"),
    }
}

#[test]
fn strategy_protection_controls_hide_for_direct_reversal() {
    let mut app = App::new(AppConfig::default());
    app.selected_broker = BrokerKind::Tradovate;
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    app.strategy.native_reversal_mode = NativeReversalMode::Direct;
    app.strategy.native_ema.take_profit_ticks = 8.0;
    app.strategy.native_ema.stop_loss_ticks = 6.0;
    app.strategy.native_ema.use_trailing_stop = true;
    app.strategy.native_ema.trail_trigger_ticks = 10.0;
    app.strategy.native_ema.trail_offset_ticks = 3.0;

    let focus_order = app.strategy_focus_order();
    assert!(!focus_order.contains(&Focus::EmaTakeProfitTicks));
    assert!(!focus_order.contains(&Focus::EmaStopLossTicks));
    assert!(!focus_order.contains(&Focus::EmaTrailingStop));
    assert!(!focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(!focus_order.contains(&Focus::EmaTrailOffsetTicks));

    let setup_text = strategy_setup_text(&app);
    for hidden_label in [
        "Take Profit Ticks",
        "Stop Loss Ticks",
        "Trailing Stop",
        "Trail Trigger Ticks",
        "Trail Offset Ticks",
        "Auto Trail Preview",
    ] {
        assert!(
            setup_text.iter().all(|line| !line.contains(hidden_label)),
            "{hidden_label} should be hidden for Direct reversal"
        );
    }

    let detail_text = rendered_text(app.strategy_detail_lines());
    let preview_text = rendered_text(app.strategy_preview_lines());
    for hidden_label in [
        "Risk: tp_ticks",
        "trail_trigger",
        "trail_offset",
        "TP/SL",
        "Trailing stop",
        "tp=",
        "sl=",
        "trail=",
    ] {
        assert!(
            detail_text.iter().all(|line| !line.contains(hidden_label)),
            "{hidden_label} should be hidden from detail for Direct reversal"
        );
        assert!(
            preview_text.iter().all(|line| !line.contains(hidden_label)),
            "{hidden_label} should be hidden from preview for Direct reversal"
        );
    }
}

#[test]
fn trailing_stop_tick_fields_show_only_when_trailing_stop_is_enabled() {
    let mut hma_app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut hma_app);
    hma_app.strategy.kind = StrategyKind::Native;
    hma_app.strategy.native_strategy = NativeStrategyKind::HmaAngle;
    hma_app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    hma_app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    hma_app.strategy.native_hma.use_trailing_stop = false;

    let hma_focus_order = hma_app.strategy_focus_order();
    assert!(hma_focus_order.contains(&Focus::HmaTrailingStop));
    assert!(!hma_focus_order.contains(&Focus::HmaTrailTriggerTicks));
    assert!(!hma_focus_order.contains(&Focus::HmaTrailOffsetTicks));

    let hma_setup_text = strategy_setup_text(&hma_app);
    assert!(
        hma_setup_text
            .iter()
            .any(|line| line.contains("Trailing Stop"))
    );
    assert!(
        hma_setup_text
            .iter()
            .all(|line| !line.contains("Trail Trigger Ticks"))
    );
    assert!(
        hma_setup_text
            .iter()
            .all(|line| !line.contains("Trail Offset Ticks"))
    );

    let hma_detail_text = rendered_text(hma_app.strategy_detail_lines());
    assert!(
        hma_detail_text
            .iter()
            .all(|line| !line.contains("trail_trigger"))
    );
    assert!(
        hma_detail_text
            .iter()
            .all(|line| !line.contains("trail_offset"))
    );

    hma_app.strategy.native_hma.use_trailing_stop = true;
    let hma_focus_order = hma_app.strategy_focus_order();
    assert!(hma_focus_order.contains(&Focus::HmaTrailTriggerTicks));
    assert!(hma_focus_order.contains(&Focus::HmaTrailOffsetTicks));
    let hma_setup_text = strategy_setup_text(&hma_app);
    assert!(
        hma_setup_text
            .iter()
            .any(|line| line.contains("Trail Trigger Ticks"))
    );
    assert!(
        hma_setup_text
            .iter()
            .any(|line| line.contains("Trail Offset Ticks"))
    );

    let mut ema_app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut ema_app);
    ema_app.strategy.kind = StrategyKind::Native;
    ema_app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    ema_app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    ema_app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    ema_app.strategy.native_ema.use_trailing_stop = false;

    let ema_focus_order = ema_app.strategy_focus_order();
    assert!(ema_focus_order.contains(&Focus::EmaTrailingStop));
    assert!(!ema_focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(!ema_focus_order.contains(&Focus::EmaTrailOffsetTicks));

    let ema_setup_text = strategy_setup_text(&ema_app);
    assert!(
        ema_setup_text
            .iter()
            .any(|line| line.contains("Trailing Stop"))
    );
    assert!(
        ema_setup_text
            .iter()
            .all(|line| !line.contains("Trail Trigger Ticks"))
    );
    assert!(
        ema_setup_text
            .iter()
            .all(|line| !line.contains("Trail Offset Ticks"))
    );

    ema_app.strategy.native_ema.use_trailing_stop = true;
    let ema_focus_order = ema_app.strategy_focus_order();
    assert!(ema_focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(ema_focus_order.contains(&Focus::EmaTrailOffsetTicks));
    let ema_setup_text = strategy_setup_text(&ema_app);
    assert!(
        ema_setup_text
            .iter()
            .any(|line| line.contains("Trail Trigger Ticks"))
    );
    assert!(
        ema_setup_text
            .iter()
            .any(|line| line.contains("Trail Offset Ticks"))
    );

    let mut hma_cross_app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut hma_cross_app);
    hma_cross_app.strategy.kind = StrategyKind::Native;
    hma_cross_app.strategy.native_strategy = NativeStrategyKind::HmaCross;
    hma_cross_app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    hma_cross_app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    hma_cross_app.strategy.native_hma_cross.use_trailing_stop = false;

    let hma_cross_focus_order = hma_cross_app.strategy_focus_order();
    assert!(hma_cross_focus_order.contains(&Focus::EmaTrailingStop));
    assert!(!hma_cross_focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(!hma_cross_focus_order.contains(&Focus::EmaTrailOffsetTicks));

    hma_cross_app.strategy.native_hma_cross.use_trailing_stop = true;
    let hma_cross_focus_order = hma_cross_app.strategy_focus_order();
    assert!(hma_cross_focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(hma_cross_focus_order.contains(&Focus::EmaTrailOffsetTicks));
}

#[test]
fn strategy_protection_controls_show_for_broker_owned_reversal_modes() {
    for reversal_mode in [
        NativeReversalMode::FlattenConfirmEnter,
        NativeReversalMode::CloseAllEnter,
    ] {
        let mut app = App::new(AppConfig::default());
        enable_tradovate_controls(&mut app);
        app.strategy.kind = StrategyKind::Native;
        app.strategy.native_strategy = NativeStrategyKind::HmaAngle;
        app.strategy.native_execution_path = NativeExecutionPath::Guarded;
        app.strategy.native_reversal_mode = reversal_mode;
        app.strategy.native_hma.use_trailing_stop = true;

        let focus_order = app.strategy_focus_order();
        assert!(focus_order.contains(&Focus::HmaTakeProfitTicks));
        assert!(focus_order.contains(&Focus::HmaStopLossTicks));
        assert!(focus_order.contains(&Focus::HmaTrailingStop));
        assert!(focus_order.contains(&Focus::HmaTrailTriggerTicks));
        assert!(focus_order.contains(&Focus::HmaTrailOffsetTicks));

        let setup_text = strategy_setup_text(&app);
        for visible_label in [
            "Take Profit Ticks",
            "Stop Loss Ticks",
            "Trailing Stop",
            "Trail Trigger Ticks",
            "Trail Offset Ticks",
        ] {
            assert!(
                setup_text.iter().any(|line| line.contains(visible_label)),
                "{visible_label} should be visible for {}",
                reversal_mode.label()
            );
        }

        let detail_text = rendered_text(app.strategy_detail_lines());
        let preview_text = rendered_text(app.strategy_preview_lines());
        assert!(
            detail_text
                .iter()
                .any(|line| line.contains("Risk: tp_ticks"))
        );
        assert!(detail_text.iter().any(|line| line.contains("TP/SL")));
        assert!(preview_text.iter().any(|line| line.contains("tp=")));
        assert!(preview_text.iter().any(|line| line.contains("trail=")));
    }
}

#[test]
fn strategy_protection_controls_hide_for_non_guarded_paths() {
    for execution_path in [
        NativeExecutionPath::SimpleDiagnostic,
        NativeExecutionPath::HmaDirect,
    ] {
        let mut app = App::new(AppConfig::default());
        app.selected_broker = BrokerKind::Tradovate;
        app.strategy.kind = StrategyKind::Native;
        app.strategy.native_strategy = NativeStrategyKind::HmaCross;
        app.strategy.native_execution_path = execution_path;
        app.strategy.native_reversal_mode = NativeReversalMode::CloseAllEnter;
        app.strategy.native_hma_cross.take_profit_ticks = 8.0;
        app.strategy.native_hma_cross.stop_loss_ticks = 6.0;
        app.strategy.native_hma_cross.use_trailing_stop = true;
        app.strategy.native_hma_cross.trail_trigger_ticks = 10.0;
        app.strategy.native_hma_cross.trail_offset_ticks = 3.0;

        let focus_order = app.strategy_focus_order();
        assert!(!focus_order.contains(&Focus::EmaTakeProfitTicks));
        assert!(!focus_order.contains(&Focus::EmaStopLossTicks));
        assert!(!focus_order.contains(&Focus::EmaTrailingStop));
        assert!(!focus_order.contains(&Focus::EmaTrailTriggerTicks));
        assert!(!focus_order.contains(&Focus::EmaTrailOffsetTicks));

        let setup_text = strategy_setup_text(&app);
        for hidden_label in [
            "Take Profit Ticks",
            "Stop Loss Ticks",
            "Trailing Stop",
            "Trail Trigger Ticks",
            "Trail Offset Ticks",
            "Auto Trail Preview",
        ] {
            assert!(
                setup_text.iter().all(|line| !line.contains(hidden_label)),
                "{hidden_label} should be hidden for {}",
                execution_path.label()
            );
        }

        let detail_text = rendered_text(app.strategy_detail_lines());
        let preview_text = rendered_text(app.strategy_preview_lines());
        for hidden_label in [
            "Risk: tp_ticks",
            "trail_trigger",
            "trail_offset",
            "TP/SL",
            "Trailing stop",
            "tp=",
            "sl=",
            "trail=",
        ] {
            assert!(
                detail_text.iter().all(|line| !line.contains(hidden_label)),
                "{hidden_label} should be hidden from detail for {}",
                execution_path.label()
            );
            assert!(
                preview_text.iter().all(|line| !line.contains(hidden_label)),
                "{hidden_label} should be hidden from preview for {}",
                execution_path.label()
            );
        }
    }
}

#[test]
fn strategy_protection_controls_hide_when_capability_is_unavailable() {
    let mut app = App::new(AppConfig::default());
    app.selected_broker = BrokerKind::Tradovate;
    app.capabilities.native_protection = false;
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    app.strategy.native_ema.take_profit_ticks = 8.0;
    app.strategy.native_ema.stop_loss_ticks = 6.0;
    app.strategy.native_ema.use_trailing_stop = true;

    let focus_order = app.strategy_focus_order();
    assert!(!focus_order.contains(&Focus::EmaTakeProfitTicks));
    assert!(!focus_order.contains(&Focus::EmaStopLossTicks));
    assert!(!focus_order.contains(&Focus::EmaTrailingStop));

    let setup_text = strategy_setup_text(&app);
    assert!(
        setup_text
            .iter()
            .all(|line| !line.contains("Take Profit Ticks"))
    );
    assert!(
        setup_text
            .iter()
            .all(|line| !line.contains("Stop Loss Ticks"))
    );
}

#[test]
fn strategy_protection_controls_remain_visible_for_ironbeam_app_managed_protection() {
    let mut app = App::new(AppConfig::default());
    app.selected_broker = BrokerKind::Ironbeam;
    app.capabilities.native_protection = true;
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::HmaDirect;
    app.strategy.native_reversal_mode = NativeReversalMode::Direct;
    app.strategy.native_ema.take_profit_ticks = 8.0;
    app.strategy.native_ema.stop_loss_ticks = 6.0;
    app.strategy.native_ema.use_trailing_stop = true;
    app.strategy.native_ema.trail_trigger_ticks = 10.0;
    app.strategy.native_ema.trail_offset_ticks = 3.0;

    let focus_order = app.strategy_focus_order();
    assert!(focus_order.contains(&Focus::EmaTakeProfitTicks));
    assert!(focus_order.contains(&Focus::EmaStopLossTicks));
    assert!(focus_order.contains(&Focus::EmaTrailingStop));
    assert!(focus_order.contains(&Focus::EmaTrailTriggerTicks));
    assert!(focus_order.contains(&Focus::EmaTrailOffsetTicks));

    let setup_text = strategy_setup_text(&app);
    for visible_label in [
        "Take Profit Ticks",
        "Stop Loss Ticks",
        "Trailing Stop",
        "Trail Trigger Ticks",
        "Trail Offset Ticks",
    ] {
        assert!(
            setup_text.iter().any(|line| line.contains(visible_label)),
            "{visible_label} should remain visible for Ironbeam app-managed protection"
        );
    }

    let detail_text = rendered_text(app.strategy_detail_lines());
    let preview_text = rendered_text(app.strategy_preview_lines());
    assert!(
        detail_text
            .iter()
            .any(|line| line.contains("Risk: tp_ticks"))
    );
    assert!(detail_text.iter().any(|line| line.contains("TP/SL")));
    assert!(preview_text.iter().any(|line| line.contains("tp=")));
    assert!(preview_text.iter().any(|line| line.contains("trail=")));
}

#[test]
fn native_trade_levels_project_tp_sl_until_broker_sync() {
    let mut app = App::new(AppConfig::default());
    app.accounts = vec![account(42, "SIM")];
    app.account_snapshots = vec![account_snapshot(42, Some(1.0), Some(100.0), None, None)];
    app.selected_account = 0;
    app.market.tick_size = Some(0.25);
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_ema.take_profit_ticks = 8.0;
    app.strategy.native_ema.stop_loss_ticks = 6.0;

    let levels = app.displayed_trade_levels();

    assert_eq!(levels.entry_price, Some(100.0));
    assert_eq!(levels.take_profit_price, Some(102.0));
    assert_eq!(levels.stop_price, Some(98.5));
    assert!(levels.take_profit_projected);
    assert!(levels.stop_price_projected);
}

#[test]
fn synced_native_trade_levels_override_projected_values() {
    let mut app = App::new(AppConfig::default());
    app.accounts = vec![account(42, "SIM")];
    app.account_snapshots = vec![account_snapshot(
        42,
        Some(-1.0),
        Some(100.0),
        Some(98.0),
        Some(101.5),
    )];
    app.selected_account = 0;
    app.market.tick_size = Some(0.25);
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_ema.take_profit_ticks = 12.0;
    app.strategy.native_ema.stop_loss_ticks = 10.0;

    let levels = app.displayed_trade_levels();

    assert_eq!(levels.entry_price, Some(100.0));
    assert_eq!(levels.take_profit_price, Some(98.0));
    assert_eq!(levels.stop_price, Some(101.5));
    assert!(!levels.take_profit_projected);
    assert!(!levels.stop_price_projected);
}

#[test]
fn execution_state_syncs_selected_account_index_from_engine() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")]),
        &cmd_tx,
    );

    let snapshot = ExecutionStateSnapshot {
        selected_account_id: Some(2),
        ..ExecutionStateSnapshot::default()
    };
    app.handle_service_event(ServiceEvent::ExecutionState(snapshot), &cmd_tx);

    assert_eq!(app.selected_account, 1);
}

#[test]
fn strategy_continue_syncs_selected_account_before_arming() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;
    select_ready_contract(&mut app);
    app.focus = Focus::StrategyContinue;

    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 2);
    match cmd_rx.try_recv().expect("expected config command") {
        ServiceCommand::SetExecutionStrategyConfig(_) => {}
        _ => panic!("expected execution-config command"),
    }
    match cmd_rx.try_recv().expect("expected arm command") {
        ServiceCommand::ArmExecutionStrategy => {}
        _ => panic!("expected arm-execution command"),
    }
}

#[test]
fn strategy_continue_forces_guarded_when_settings_need_order_strategy_path() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);
    app.focus = Focus::StrategyContinue;
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::HmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::HmaDirect;
    app.strategy.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    app.strategy.native_hma_cross.use_trailing_stop = true;

    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected config command") {
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            assert_eq!(config.native_execution_path, NativeExecutionPath::Guarded);
            assert_eq!(
                config.native_reversal_mode,
                NativeReversalMode::CloseAllEnter
            );
            assert!(config.native_hma_cross.use_trailing_stop);
        }
        _ => panic!("expected execution-config command"),
    }
    match cmd_rx.try_recv().expect("expected arm command") {
        ServiceCommand::ArmExecutionStrategy => {}
        _ => panic!("expected arm-execution command"),
    }
}

#[test]
fn strategy_continue_forces_closeall_when_protection_needs_broker_owned_reversal() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136")];
    select_ready_contract(&mut app);
    app.focus = Focus::StrategyContinue;
    app.strategy.kind = StrategyKind::Native;
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_execution_path = NativeExecutionPath::Guarded;
    app.strategy.native_reversal_mode = NativeReversalMode::Direct;
    app.strategy.native_ema.take_profit_ticks = 8.0;

    app.handle_strategy_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected config command") {
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            assert_eq!(config.native_execution_path, NativeExecutionPath::Guarded);
            assert_eq!(
                config.native_reversal_mode,
                NativeReversalMode::CloseAllEnter
            );
            assert_eq!(config.native_ema.take_profit_ticks, 8.0);
        }
        _ => panic!("expected execution-config command"),
    }
    match cmd_rx.try_recv().expect("expected arm command") {
        ServiceCommand::ArmExecutionStrategy => {}
        _ => panic!("expected arm-execution command"),
    }
}
