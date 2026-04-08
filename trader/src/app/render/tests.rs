use super::*;
use crate::broker::{
    AccountInfo, AccountSnapshot, BrokerCapabilities, BrokerKind, ContractSuggestion,
    ManualOrderAction, ServiceEvent,
};
use crate::config::{AppConfig, LogMode};
use crate::strategy::{
    ExecutionStateSnapshot, NativeReversalMode, NativeStrategyKind, StrategyKind,
};
use serde_json::json;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn account(id: i64, name: &str) -> AccountInfo {
    AccountInfo {
        id,
        name: name.to_string(),
        raw: json!({}),
    }
}

fn contract(id: i64, name: &str) -> ContractSuggestion {
    ContractSuggestion {
        id,
        name: name.to_string(),
        description: "test contract".to_string(),
        raw: json!({}),
    }
}

fn account_snapshot(
    account_id: i64,
    market_position_qty: Option<f64>,
    market_entry_price: Option<f64>,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
) -> AccountSnapshot {
    AccountSnapshot {
        account_id,
        account_name: "SIM".to_string(),
        balance: None,
        cash_balance: None,
        net_liq: None,
        realized_pnl: None,
        unrealized_pnl: None,
        intraday_margin: None,
        open_position_qty: market_position_qty,
        market_position_qty,
        market_entry_price,
        selected_contract_take_profit_price: take_profit_price,
        selected_contract_stop_price: stop_price,
        raw_account: None,
        raw_risk: None,
        raw_cash: None,
        raw_positions: Vec::new(),
    }
}

fn balance_snapshot(account_id: i64, account_name: &str, balance: f64) -> AccountSnapshot {
    AccountSnapshot {
        account_id,
        account_name: account_name.to_string(),
        balance: Some(balance),
        cash_balance: Some(balance),
        net_liq: Some(balance),
        realized_pnl: None,
        unrealized_pnl: None,
        intraday_margin: None,
        open_position_qty: None,
        market_position_qty: None,
        market_entry_price: None,
        selected_contract_take_profit_price: None,
        selected_contract_stop_price: None,
        raw_account: None,
        raw_risk: None,
        raw_cash: None,
        raw_positions: Vec::new(),
    }
}

fn enable_tradovate_controls(app: &mut App) {
    app.selected_broker = BrokerKind::Tradovate;
    app.capabilities = BrokerCapabilities {
        replay: true,
        manual_orders: true,
        automated_orders: true,
        native_protection: true,
    };
}

fn expect_select_account(rx: &mut UnboundedReceiver<ServiceCommand>, account_id: i64) {
    match rx.try_recv().expect("expected select-account command") {
        ServiceCommand::SelectAccount { account_id: actual } => {
            assert_eq!(actual, account_id);
        }
        _ => panic!("expected select-account command"),
    }
}

#[test]
fn app_starts_on_broker_select_when_multiple_brokers_are_available() {
    let app = App::new(AppConfig::default());

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
fn startup_disconnected_event_keeps_broker_picker_visible() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();

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
fn broker_picker_uses_arrow_keys_and_enter_to_open_login() {
    let mut app = App::new(AppConfig::default());
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
fn f6_opens_session_stats_screen() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();

    app.handle_key(key(KeyCode::F(6)), &cmd_tx);

    assert_eq!(app.screen, Screen::Stats);
    assert_eq!(app.focus, Focus::AccountList);
}

#[test]
fn selection_tab_order_reaches_bar_type_toggle() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::AccountList;

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::InstrumentQuery);

    app.handle_selection_key(key(KeyCode::Tab), &cmd_tx);
    assert_eq!(app.focus, Focus::ContractList);
}

#[test]
fn bar_type_toggle_uses_arrow_keys_not_enter() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::BarTypeToggle;

    assert_eq!(app.bar_type, BarType::Minute1);

    app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.bar_type, BarType::Range1);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Left), &cmd_tx);
    assert_eq!(app.bar_type, BarType::Minute1);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);
    assert_eq!(app.bar_type, BarType::Minute1);
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
}

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
fn session_stats_track_wins_losses_and_flats_from_balance_deltas() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for balance in [1_000.0, 1_015.0, 1_005.0, 1_030.0, 1_030.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.account_id, 7);
    assert_eq!(stats.sample_count, 5);
    assert_eq!(stats.wins, 2);
    assert_eq!(stats.losses, 1);
    assert_eq!(stats.flat_moves, 1);
    assert_eq!(stats.event_count(), 3);
    assert_eq!(stats.avg_win(), Some(20.0));
    assert_eq!(stats.max_win, Some(25.0));
    assert_eq!(stats.avg_loss_signed(), Some(-10.0));
    assert_eq!(stats.max_loss_signed(), Some(-10.0));
    assert_eq!(stats.session_pnl(), 30.0);
    assert_eq!(stats.win_rate(), Some(2.0 / 3.0));
}

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
fn dashboard_manual_orders_sync_selected_account_first() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;

    app.handle_dashboard_key(key(KeyCode::Char('b')), &cmd_tx);

    expect_select_account(&mut cmd_rx, 2);
    match cmd_rx.try_recv().expect("expected manual-order command") {
        ServiceCommand::ManualOrder {
            action: ManualOrderAction::Buy,
        } => {}
        _ => panic!("expected buy manual-order command"),
    }
}

#[test]
fn dashboard_visual_toggle_updates_state_without_sending_commands() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();

    assert!(!app.dashboard_visuals_enabled);
    app.handle_dashboard_key(key(KeyCode::Char('v')), &cmd_tx);
    assert!(app.dashboard_visuals_enabled);
    assert!(cmd_rx.try_recv().is_err());

    app.handle_dashboard_key(key(KeyCode::Char('v')), &cmd_tx);
    assert!(!app.dashboard_visuals_enabled);
    assert!(cmd_rx.try_recv().is_err());
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
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            assert_eq!(contract.id, 123);
            assert_eq!(contract.name, "ESM6");
            assert_eq!(bar_type, BarType::Minute1);
        }
        _ => panic!("expected subscribe-bars command"),
    }
}

#[test]
fn strategy_continue_syncs_selected_account_before_arming() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;
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
fn selection_flow_moves_from_account_to_bar_type_to_query_to_contract() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::AccountList;

    app.handle_selection_key(key(KeyCode::Right), &cmd_tx);
    assert_eq!(app.focus, Focus::BarTypeToggle);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::InstrumentQuery);

    app.handle_selection_key(key(KeyCode::Down), &cmd_tx);
    assert_eq!(app.focus, Focus::ContractList);
}

#[test]
fn bar_type_enter_advances_to_query_without_toggling() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.focus = Focus::BarTypeToggle;

    app.handle_selection_key(key(KeyCode::Enter), &cmd_tx);

    assert_eq!(app.bar_type, BarType::Minute1);
    assert_eq!(app.focus, Focus::InstrumentQuery);
}

#[test]
fn push_log_records_wall_time_and_elapsed_delta() {
    let mut app = App::new(AppConfig::default());
    app.logs.clear();
    app.last_log_at = None;

    app.push_log("first event".to_string());
    std::thread::sleep(std::time::Duration::from_millis(5));
    app.push_log("second event".to_string());

    assert_eq!(app.logs.len(), 2);
    assert_eq!(app.logs[0].elapsed_since_previous, None);
    assert!(app.logs[1].elapsed_since_previous.is_some());
    assert!(app.logs[0].render_line().contains("first event"));
    assert!(app.logs[0].render_line().contains("+0ms"));
    assert!(app.logs[1].render_line().contains("second event"));
    assert!(app.logs[1].render_line().starts_with('['));
}

#[test]
fn debug_log_events_are_filtered_by_log_mode() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.logs.clear();

    app.handle_service_event(
        ServiceEvent::DebugLog("submit 42ms | order".to_string()),
        &cmd_tx,
    );
    assert!(app.logs.is_empty());

    app.form.log_mode = LogMode::Debug;
    app.handle_service_event(
        ServiceEvent::DebugLog("submit 42ms | order".to_string()),
        &cmd_tx,
    );

    assert_eq!(app.logs.len(), 1);
    assert_eq!(app.logs[0].message, "DEBUG: submit 42ms | order");
}

#[test]
fn disabled_session_stats_screen_shows_enable_hint() {
    let mut config = AppConfig::default();
    config.session_stats_enabled = false;
    let app = App::new(config);

    let lines = app.selected_session_stats_lines();

    assert!(lines[0].to_string().contains("disabled"));
    assert!(
        app.session_stats_overview_lines()
            .iter()
            .any(|line| line.to_string().contains("TRADER_SESSION_STATS_ENABLED=1"))
    );
}
