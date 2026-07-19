use super::*;
use crate::broker::{
    AccountInfo, AccountSnapshot, BarKind, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, LatencySnapshot, ManualOrderAction, MarketSnapshot, ServiceEvent,
};
use crate::config::{AppConfig, AuthMode, LogMode, TradingEnvironment};
use crate::engine_registry::RunningEngine;
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, NativeExecutionPath, NativeReversalMode,
    NativeStrategyKind, StrategyKind,
};
use serde_json::json;
use std::path::PathBuf;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn ctrl_key(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL)
}

fn line_span_with_fg(line: &Line<'_>, content: &str, color: Color) -> bool {
    line.spans
        .iter()
        .any(|span| span.content.as_ref() == content && span.style.fg == Some(color))
}

fn rendered_text(lines: Vec<Line<'static>>) -> Vec<String> {
    lines.into_iter().map(|line| line.to_string()).collect()
}

fn assert_money_eq(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 0.005,
        "expected {expected:.2}, got {actual:.2}"
    );
}

fn section_between<'a>(body: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = body.find(start).expect("expected section start");
    let after_start = &body[start_index..];
    let end_index = after_start[start.len()..]
        .find(end)
        .map(|index| index + start.len())
        .unwrap_or(after_start.len());
    &after_start[..end_index]
}

fn strategy_setup_text(app: &App) -> Vec<String> {
    rendered_text(app.strategy_setup_lines())
}

fn assert_focused_line_visible(lines: &[Line<'_>], area: Rect) {
    let offset = focused_paragraph_scroll_offset(lines, area) as usize;
    let visible_rows = area.height.saturating_sub(2) as usize;
    let focused = focused_line_index(lines).expect("expected one focused line");

    assert!(
        focused >= offset,
        "focused line should not be above the viewport"
    );
    assert!(
        focused < offset + visible_rows,
        "focused line should not be below the viewport"
    );
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

fn balance_snapshot_with_position(
    account_id: i64,
    account_name: &str,
    balance: f64,
    market_position_qty: f64,
) -> AccountSnapshot {
    let mut snapshot = balance_snapshot(account_id, account_name, balance);
    snapshot.open_position_qty = Some(market_position_qty);
    snapshot.market_position_qty = Some(market_position_qty);
    snapshot
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

fn select_ready_contract(app: &mut App) {
    app.contract_results = vec![contract(99, "ESZ6")];
    app.selected_contract = 0;
    app.market.contract_id = Some(99);
    app.market.contract_name = Some("ESZ6".to_string());
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
fn app_starts_on_engine_select() {
    let mut app = App::new(AppConfig::default());

    assert_eq!(app.screen, Screen::EngineSelect);
    assert_eq!(app.focus, Focus::EngineList);
    assert_eq!(app.selected_engine, 0);
    assert!(!app.awaiting_broker_selection());
    assert!(app.take_engine_selection_action().is_none());
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

fn running_engine(id: u32, live: bool) -> RunningEngine {
    RunningEngine {
        id,
        cwd: PathBuf::from("/tmp"),
        socket_path: PathBuf::from(format!("/tmp/trader-engine-{id}.sock")),
        socket_is_live: live,
    }
}

fn connected_event(broker: BrokerKind) -> ServiceEvent {
    ServiceEvent::Connected {
        broker,
        env: TradingEnvironment::Sim,
        user_name: Some("tester".to_string()),
        auth_mode: AuthMode::TokenFile,
        session_kind: SessionKind::Live,
        capabilities: BrokerCapabilities::default(),
    }
}

fn engine_key(id: u32) -> EngineKey {
    EngineKey::from_socket_path(PathBuf::from(format!("/tmp/trader-engine-{id}.sock")).as_path())
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

#[test]
fn engine_picker_enter_on_create_emits_create_action() {
    let mut app = App::new(AppConfig::default());
    app.set_running_engines(vec![running_engine(10, true)]);
    app.selected_engine = app.running_engines.len();

    app.handle_engine_select_key(key(KeyCode::Enter));

    assert_eq!(
        app.take_engine_selection_action(),
        Some(EngineSelectionAction::CreateNew)
    );
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
    assert!(!app.header_help_text().contains("connect/replay"));

    app.handle_key(key(KeyCode::Char('r')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
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
fn strategy_setup_scrolls_focused_option_into_short_panel() {
    let mut app = App::new(AppConfig::default());
    enable_tradovate_controls(&mut app);
    app.strategy.native_strategy = NativeStrategyKind::EmaCross;
    app.strategy.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
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
fn session_stats_reports_pnl_per_hour() {
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
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_020.0)]),
        &cmd_tx,
    );

    let end = chrono::Utc::now();
    let stats = app
        .session_stats
        .accounts
        .values_mut()
        .next()
        .expect("expected stats");
    stats.started_at_utc = end - chrono::Duration::hours(2);
    stats.last_updated_at_utc = end;

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("PnL/H: Net +10.00/h  Trade +10.00/h"))
    );

    let event_lines = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_lines
            .iter()
            .any(|line| line.contains("Hourly Trade PnL/H (local)"))
    );
    assert!(
        event_lines
            .iter()
            .any(|line| line.contains("+20.00/h net +20.00 fees 0.00"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("net_pnl_per_hour: +10.00/h"));
    assert!(body.contains("trade_pnl_per_hour: +10.00/h"));
    assert!(body.contains("hourly_local:"));
    assert!(body.contains("trade_per_hour=+20.00/h"));
}

#[test]
fn session_stats_attributes_balance_deltas_to_long_and_short_side() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.0, 1.0),
        balance_snapshot_with_position(7, "SIM", 1_015.0, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_015.0, -1.0),
        balance_snapshot_with_position(7, "SIM", 1_005.0, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.long_side.events, 1);
    assert_eq!(stats.long_side.wins, 1);
    assert_eq!(stats.long_side.losses, 0);
    assert_eq!(stats.long_side.pnl, 15.0);
    assert_eq!(stats.short_side.events, 1);
    assert_eq!(stats.short_side.wins, 0);
    assert_eq!(stats.short_side.losses, 1);
    assert_eq!(stats.short_side.pnl, -10.0);
    assert_eq!(stats.events[0].side, SessionTradeSide::Long);
    assert_eq!(
        stats.events[0].previous_position_side,
        SessionTradeSide::Long
    );
    assert_eq!(
        stats.events[0].current_position_side,
        SessionTradeSide::Flat
    );
    assert_eq!(stats.events[1].side, SessionTradeSide::Short);
    assert_eq!(
        stats.events[1].previous_position_side,
        SessionTradeSide::Short
    );
    assert_eq!(
        stats.events[1].current_position_side,
        SessionTradeSide::Flat
    );

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| { line.contains("Side PnL: Long +15.00 (1/0)  Short -10.00 (0/1)") })
    );

    let event_lines = app.session_stats_event_lines(8);
    let long_event_line = event_lines
        .iter()
        .find(|line| line.to_string().contains("balance long"))
        .expect("expected long balance event line");
    let short_event_line = event_lines
        .iter()
        .find(|line| line.to_string().contains("balance short"))
        .expect("expected short balance event line");
    assert!(line_span_with_fg(long_event_line, "long", Color::Cyan));
    assert!(line_span_with_fg(long_event_line, "1015.00", Color::Green));
    assert!(line_span_with_fg(long_event_line, "+15.00", Color::Green));
    assert!(line_span_with_fg(short_event_line, "short", Color::Magenta));
    assert!(line_span_with_fg(short_event_line, "1005.00", Color::Red));
    assert!(line_span_with_fg(short_event_line, "-10.00", Color::Red));

    let event_text = event_lines
        .iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance long pos long->flat"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance short pos short->flat"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("long_events: 1"));
    assert!(body.contains("long_pnl: +15.00"));
    assert!(body.contains("short_events: 1"));
    assert!(body.contains("short_pnl: -10.00"));
    assert!(body.contains("side=long"));
    assert!(body.contains("pos=long->flat"));
    assert!(body.contains("side=short"));
    assert!(body.contains("pos=short->flat"));
}

#[test]
fn session_stats_event_lines_include_position_transition_context() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.0, -1.0),
        balance_snapshot_with_position(7, "SIM", 1_012.5, 1.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.events[0].side, SessionTradeSide::Short);
    assert_eq!(
        stats.events[0].previous_position_side,
        SessionTradeSide::Short
    );
    assert_eq!(
        stats.events[0].current_position_side,
        SessionTradeSide::Long
    );

    let event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance short pos short->long"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("side=short"));
    assert!(body.contains("pos=short->long"));
}

#[test]
fn session_stats_filters_fee_only_and_mixed_fee_balance_deltas() {
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

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.wins, 1);
    assert_eq!(stats.losses, 1);
    assert_eq!(stats.flat_moves, 1);
    assert_eq!(stats.fee_events, 3);
    assert_eq!(stats.event_count(), 3);
    assert_money_eq(stats.session_pnl(), -3.63);
    assert_money_eq(stats.trade_pnl_ex_fees(), -1.25);
    assert_money_eq(stats.total_fees, -2.38);
    assert_money_eq(stats.long_side.pnl, 6.25);
    assert_money_eq(stats.short_side.pnl, -7.50);
    assert_eq!(stats.long_side.wins, 1);
    assert_eq!(stats.short_side.losses, 1);
    assert_eq!(stats.events[0].kind, SessionBalanceEventKind::Mixed);
    assert_eq!(stats.events[1].kind, SessionBalanceEventKind::Fee);
    assert_eq!(stats.events[2].kind, SessionBalanceEventKind::Mixed);
    assert_money_eq(stats.events[0].trade_delta, 6.25);
    assert_money_eq(stats.events[0].fee_delta, -0.56);
    assert_money_eq(stats.events[1].trade_delta, 0.0);
    assert_money_eq(stats.events[1].fee_delta, -0.91);
    assert_money_eq(stats.events[2].trade_delta, -7.50);
    assert_money_eq(stats.events[2].fee_delta, -0.91);

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: -1.25  Fees: -2.38 (3)"))
    );
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Wins: 1  Losses: 1  Flats: 1  Fee Events: 3"))
    );

    let event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade -7.50 fees -0.91"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("fee trade 0.00 fees -0.91"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade +6.25 fees -0.56"))
    );

    app.handle_session_stats_key(key(KeyCode::Char('f')), &cmd_tx);
    assert!(!app.session_stats_show_fees);
    assert_eq!(app.status, "Session stats fee rows hidden.");

    let overview_text = rendered_text(app.session_stats_overview_lines());
    assert!(overview_text.iter().any(|line| line == "Fees: hidden"));

    let hidden_account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        hidden_account_lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: -1.25  Fees hidden"))
    );
    assert!(
        hidden_account_lines
            .iter()
            .any(|line| line.contains("Wins: 1  Losses: 1  Flats: 1"))
    );
    assert!(
        !hidden_account_lines
            .iter()
            .any(|line| line.contains("Fee Events"))
    );

    let hidden_event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("(1/1, 2 trade events)"))
    );
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("balance short") && line.contains("trade -7.50"))
    );
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("balance long") && line.contains("trade +6.25"))
    );
    assert!(
        !hidden_event_text
            .iter()
            .any(|line| line.contains("fee trade") || line.contains("fees"))
    );
    assert!(
        !hidden_event_text
            .iter()
            .any(|line| line.contains("mixed trade"))
    );

    app.handle_session_stats_key(key(KeyCode::Char('F')), &cmd_tx);
    assert!(app.session_stats_show_fees);
    assert_eq!(app.status, "Session stats fee rows shown.");

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("fee_events: 3"));
    assert!(body.contains("total_fees: -2.38"));
    assert!(body.contains("trade_pnl_ex_fees: -1.25"));
    assert!(body.contains("kind=fee"));
    assert!(body.contains("trade_delta=+6.25 fee_delta=-0.56"));
    assert!(body.contains("trade_delta=-7.50 fee_delta=-0.91"));
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
fn session_stats_identity_keeps_same_account_separate_by_active_engine() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    let first_key = engine_key(10);
    app.enter_engine_session_for_key(
        first_key.clone(),
        PathBuf::from("/tmp/trader-engine-10.sock"),
    );
    for balance in [1_000.0, 1_010.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }
    assert_eq!(
        app.selected_session_stats()
            .expect("first engine stats")
            .session_pnl(),
        10.0
    );

    let second_key = engine_key(11);
    app.enter_engine_session_for_key(
        second_key.clone(),
        PathBuf::from("/tmp/trader-engine-11.sock"),
    );
    for balance in [2_000.0, 1_980.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }

    assert_eq!(app.session_stats.accounts.len(), 2);
    assert_eq!(
        app.selected_session_stats()
            .expect("second engine stats")
            .session_pnl(),
        -20.0
    );

    app.enter_engine_session_for_key(first_key, PathBuf::from("/tmp/trader-engine-10.sock"));
    assert_eq!(
        app.selected_session_stats()
            .expect("first engine stats after switch")
            .session_pnl(),
        10.0
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("engine_identity: engine:/tmp/trader-engine-10.sock"));
    assert!(body.contains("engine_identity: engine:/tmp/trader-engine-11.sock"));
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
fn dashboard_hides_manual_order_affordances_without_capability() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.capabilities.manual_orders = false;
    app.capabilities.automated_orders = false;
    app.accounts = vec![account(1, "SIM")];
    app.account_snapshots = vec![account_snapshot(1, None, None, None, None)];

    assert!(!app.header_help_text().contains("b/s/c manual"));
    assert!(!app.header_help_text().contains("timing/reversal mode"));
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .all(|line| !line.contains("b/s/c"))
    );
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .any(|line| line.contains("Keys v"))
    );

    app.handle_dashboard_key(key(KeyCode::Char('b')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
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
