use super::support::*;

#[test]
fn selection_tab_order_reaches_bar_type_toggle() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::AccountList;

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::BarValue);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::CandleModeToggle);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::InstrumentQuery);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::ContractList);
}

#[test]
fn fixed_market_broker_hides_bar_and_candle_controls() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.selected_broker = BrokerKind::Ironbeam;
    app.normalize_market_controls_for_broker();
    app.focus = Focus::AccountList;

    let focus_order = app.selection_focus_order();
    assert!(!focus_order.contains(&Focus::BarTypeToggle));
    assert!(!focus_order.contains(&Focus::BarValue));
    assert!(!focus_order.contains(&Focus::CandleModeToggle));

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::InstrumentQuery);

    let search_text = rendered_text(app.selection_preview_lines());
    assert!(search_text.iter().any(|line| line == "Bar Type: 1 Min"));
    assert!(!search_text.iter().any(|line| line.starts_with("Candles:")));
    assert!(!app.header_help_text().contains("Left/Right bar type"));
}

#[test]
fn bar_type_toggle_uses_arrow_keys_not_enter() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.focus = Focus::BarTypeToggle;

    assert_eq!(app.bar_type, BarType::minute(1));

    app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.bar_type, BarType::second(1));
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);
    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.focus, Focus::BarValue);
}

#[test]
fn bar_value_accepts_arbitrary_numeric_input() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.focus = Focus::BarValue;

    app.handle_selection_key(key(KeyCode::Char('2')), &cmd_tx);
    app.handle_selection_key(key(KeyCode::Char('5')), &cmd_tx);
    assert_eq!(app.bar_type, BarType::minute(25));

    app.handle_selection_key(key(KeyCode::Backspace), &cmd_tx);
    assert_eq!(app.bar_type, BarType::minute(2));
}

#[test]
fn bar_type_cycles_through_supported_kinds() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.focus = Focus::BarTypeToggle;

    for expected in [
        BarKind::Second,
        BarKind::Tick,
        BarKind::Volume,
        BarKind::Range,
        BarKind::Minute,
    ] {
        app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
        assert_eq!(app.bar_type.kind(), expected);
    }
}

#[test]
fn candle_mode_toggle_uses_arrow_keys_not_enter() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.focus = Focus::CandleModeToggle;

    assert_eq!(app.candle_mode, CandleMode::Standard);

    app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.candle_mode, CandleMode::HeikinAshi);
    assert_eq!(app.focus, Focus::CandleModeToggle);

    app.handle_selection_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.candle_mode, CandleMode::Standard);
    assert_eq!(app.focus, Focus::CandleModeToggle);

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);
    assert_eq!(app.candle_mode, CandleMode::Standard);
    assert_eq!(app.focus, Focus::InstrumentQuery);
}

#[test]
fn live_selection_defaults_remain_one_minute_ohlc() {
    let app = App::new(AppConfig::default());

    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.candle_mode, CandleMode::Standard);
}

#[test]
fn contract_selection_syncs_selected_account_before_subscribe() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;
    app.contract_results = vec![contract(123, "ESM6")];
    app.focus = Focus::ContractList;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 2);
    match cmd_rx.try_recv().expect("expected subscribe-bars command") {
        ServiceCommand::SubscribeBars {
            contract,
            bar_type,
            candle_mode,
        } => {
            assert_eq!(contract.id, 123);
            assert_eq!(contract.name, "ESM6");
            assert_eq!(bar_type, BarType::minute(1));
            assert_eq!(candle_mode, CandleMode::Standard);
        }
        _ => panic!("expected subscribe-bars command"),
    }
}

#[test]
fn blocked_contract_requires_second_enter_to_override() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.accounts = vec![account(1, "DEMO4769136")];
    app.contract_results = vec![ContractSuggestion {
        id: 4_095_561,
        name: "GCQ6".to_string(),
        description: "Gold August 2026".to_string(),
        raw: json!({
            "contractMaturityId": 59107,
            "_midasContractMaturity": {
                "id": 59107,
                "expirationDate": "2026-08-27T17:30Z",
                "firstIntentDate": "2026-07-31T00:00Z",
                "archived": false
            }
        }),
    }];
    app.focus = Focus::ContractList;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
    assert!(app.status.contains("BLOCKED"));
    assert!(app.status.contains("Press Enter again to override"));
    assert_ne!(app.screen, Screen::Strategy);

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected override subscription") {
        ServiceCommand::SubscribeBars { contract, .. } => {
            assert_eq!(contract.name, "GCQ6");
        }
        _ => panic!("expected subscribe-bars command"),
    }
    assert_eq!(app.screen, Screen::Strategy);
    assert!(
        app.persisted_logs
            .iter()
            .any(|entry| entry.message.contains("safety override accepted for GCQ6"))
    );
}

#[test]
fn contract_selection_subscribes_with_selected_candle_mode() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.accounts = vec![account(1, "DEMO4769136")];
    app.contract_results = vec![contract(123, "ESM6")];
    app.candle_mode = CandleMode::HeikinAshi;
    app.focus = Focus::ContractList;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected subscribe-bars command") {
        ServiceCommand::SubscribeBars { candle_mode, .. } => {
            assert_eq!(candle_mode, CandleMode::HeikinAshi);
        }
        _ => panic!("expected subscribe-bars command"),
    }
}

#[test]
fn contract_selection_subscribes_range_with_standard_candle_mode() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.accounts = vec![account(1, "DEMO4769136")];
    app.contract_results = vec![contract(123, "ESM6")];
    app.bar_type = BarType::range(4);
    app.candle_mode = CandleMode::HeikinAshi;
    app.focus = Focus::ContractList;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    expect_select_account(&mut cmd_rx, 1);
    match cmd_rx.try_recv().expect("expected subscribe-bars command") {
        ServiceCommand::SubscribeBars {
            bar_type,
            candle_mode,
            ..
        } => {
            assert_eq!(bar_type, BarType::range(4));
            assert_eq!(candle_mode, CandleMode::Standard);
        }
        _ => panic!("expected subscribe-bars command"),
    }
}

#[test]
fn selection_flow_moves_from_account_to_bar_type_to_query_to_contract() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::AccountList;

    app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::BarValue);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::CandleModeToggle);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::InstrumentQuery);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::ContractList);
}

#[test]
fn bar_type_enter_advances_to_value_without_toggling() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::BarTypeToggle;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    assert_eq!(app.bar_type, BarType::minute(1));
    assert_eq!(app.focus, Focus::BarValue);
}

#[test]
fn range_bars_skip_candle_mode_focus() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.bar_type = BarType::range(1);
    app.focus = Focus::BarValue;

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);

    assert_eq!(app.focus, Focus::InstrumentQuery);
}

#[test]
fn range_selection_summaries_hide_candle_mode() {
    let mut app = App::new(AppConfig::default());
    app.bar_type = BarType::range(3);
    app.candle_mode = CandleMode::HeikinAshi;

    let selection = rendered_text(app.selection_summary_lines());
    let preview = rendered_text(app.selection_preview_lines());
    let dashboard = rendered_text(app.dashboard_summary_lines());

    assert!(selection.iter().any(|line| line == "Bar Type: 3 Range"));
    assert!(!selection.iter().any(|line| line.starts_with("Candles:")));
    assert!(!preview.iter().any(|line| line.starts_with("Candles:")));
    assert!(!dashboard.iter().any(|line| line.starts_with("Candles:")));
}
