use super::support::*;

#[test]
fn broker_picker_uses_arrow_keys_and_enter_to_open_login() {
    let mut app = App::new(AppConfig::default());
    app.enter_engine_session(PathBuf::from("/tmp/trader-engine.sock"));
    if app.available_brokers.len() < 2 {
        return;
    }

    let original = app.selected_broker;
    app.handle_broker_select_key(key(KeyCode::Down));
    assert_ne!(app.selected_broker, original);
    assert!(app.awaiting_broker_selection());

    app.handle_broker_select_key(key(KeyCode::Enter));
    assert_eq!(app.screen, Screen::Login);
    assert_eq!(app.focus, Focus::Env);
    assert!(!app.awaiting_broker_selection());
}

#[test]
fn broker_picker_normalizes_unsupported_market_controls() {
    let mut app = App::new(AppConfig::default());
    if !app.available_brokers.contains(&BrokerKind::Tradovate)
        || !app.available_brokers.contains(&BrokerKind::Ironbeam)
    {
        return;
    }

    app.selected_broker = BrokerKind::Tradovate;
    app.bar_type = BarType::volume(500);
    app.candle_mode = CandleMode::HeikinAshi;

    for _ in 0..app.available_brokers.len() {
        if app.selected_broker == BrokerKind::Ironbeam {
            break;
        }
        app.handle_broker_select_key(key(KeyCode::Down));
    }

    assert_eq!(app.selected_broker, BrokerKind::Ironbeam);
    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn app_new_normalizes_unsupported_config_market_controls() {
    let mut config = AppConfig::default();
    config.broker = BrokerKind::Ironbeam;
    config.candle_mode = CandleMode::HeikinAshi;

    let app = App::new(config);
    if app.selected_broker != BrokerKind::Ironbeam {
        return;
    }

    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn connected_event_normalizes_unsupported_market_controls() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.bar_type = BarType::range(3);
    app.candle_mode = CandleMode::HeikinAshi;

    app.handle_service_event(
        ServiceEvent::Connected {
            broker: BrokerKind::Ironbeam,
            env: TradingEnvironment::Sim,
            user_name: None,
            auth_mode: AuthMode::TokenFile,
            session_kind: SessionKind::Live,
            capabilities: BrokerCapabilities::default(),
        },
        &cmd_tx,
    );

    assert_eq!(app.selected_broker, BrokerKind::Ironbeam);
    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn f6_opens_session_stats_screen() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.enter_engine_session(PathBuf::from("/tmp/trader-engine.sock"));

    app.handle_key(key(KeyCode::F(6)), &cmd_tx);

    assert_eq!(app.screen, Screen::Stats);
    assert_eq!(app.focus, Focus::AccountList);
}

#[test]
fn disabled_session_stats_hides_navigation_affordances() {
    let mut config = AppConfig::default();
    config.session_stats_enabled = false;
    let mut app = App::new(config);
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.enter_engine_session(PathBuf::from("/tmp/trader-engine.sock"));
    app.screen = Screen::Dashboard;

    assert!(!app.header_tab_titles().contains(&"Stats"));
    assert!(!app.header_help_text().contains("F6 stats"));

    app.handle_key(key(KeyCode::F(6)), &cmd_tx);

    assert_eq!(app.screen, Screen::Dashboard);
}

#[test]
fn login_log_mode_toggle_uses_arrow_keys() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::LogMode;

    assert_eq!(app.form.log_mode, LogMode::Default);

    app.handle_login_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.form.log_mode, LogMode::Debug);

    app.handle_login_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.form.log_mode, LogMode::Default);

    app.handle_login_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.form.log_mode, LogMode::Quiet);

    app.handle_login_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.form.log_mode, LogMode::Default);
}

#[test]
fn login_scrolls_focused_option_into_short_panel() {
    let mut app = App::new(AppConfig::default());
    app.focus = Focus::Connect;

    let lines = app.connection_lines();
    let offset = focused_paragraph_scroll_offset(&lines, Rect::new(0, 0, 120, 8));

    assert!(offset > 0, "short panel should scroll down to Connect");
    assert_focused_line_visible(&lines, Rect::new(0, 0, 120, 8));
}

#[test]
fn unavailable_replay_is_hidden_from_login() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.selected_broker = BrokerKind::Ironbeam;
    app.focus = Focus::Env;

    assert!(!app.login_focus_order().contains(&Focus::ReplayMode));
    assert!(
        rendered_text(app.connection_lines())
            .iter()
            .all(|line| !line.contains("Replay Mode"))
    );
    assert!(
        rendered_text(app.login_notes_lines())
            .iter()
            .all(|line| !line.contains("Replay Mode"))
    );
    assert!(!app.header_help_text().contains("open replay"));

    app.handle_key(key(KeyCode::Char('r')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
}
