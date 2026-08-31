pub(super) use super::super::*;
#[cfg(feature = "manual-orders")]
pub(super) use crate::broker::ManualOrderAction;
#[cfg(feature = "replay")]
pub(super) use crate::broker::ReplayDownloadPhase;
pub(super) use crate::broker::{
    AccountInfo, AccountSnapshot, BarKind, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, LatencySnapshot, MarketSnapshot, ServiceCommand, ServiceCommandReceiver,
    ServiceCommandSender, ServiceEvent, service_command_channel,
};
pub(super) use crate::config::{AppConfig, AuthMode, LogMode, TradingEnvironment};
pub(super) use crate::engine_registry::RunningEngine;
pub(super) use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, NativeExecutionPath, NativeReversalMode,
    NativeStrategyKind, StrategyKind,
};
#[cfg(feature = "replay")]
pub(super) use ratatui::{Terminal, backend::TestBackend};
pub(super) use serde_json::json;
pub(super) use std::path::PathBuf;
pub(super) fn unbounded_channel() -> (ServiceCommandSender, ServiceCommandReceiver) {
    service_command_channel(256)
}

pub(super) fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

pub(super) fn ctrl_key(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL)
}

pub(super) fn line_span_with_fg(line: &Line<'_>, content: &str, color: Color) -> bool {
    line.spans
        .iter()
        .any(|span| span.content.as_ref() == content && span.style.fg == Some(color))
}

pub(super) fn rendered_text(lines: Vec<Line<'static>>) -> Vec<String> {
    lines.into_iter().map(|line| line.to_string()).collect()
}

#[cfg(feature = "replay")]
pub(super) fn rendered_terminal_rows(
    terminal: &Terminal<TestBackend>,
    width: usize,
) -> Vec<String> {
    terminal
        .backend()
        .buffer()
        .content()
        .chunks(width)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
        .collect()
}

pub(super) fn assert_money_eq(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 0.005,
        "expected {expected:.2}, got {actual:.2}"
    );
}

pub(super) fn section_between<'a>(body: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = body.find(start).expect("expected section start");
    let after_start = &body[start_index..];
    let end_index = after_start[start.len()..]
        .find(end)
        .map(|index| index + start.len())
        .unwrap_or(after_start.len());
    &after_start[..end_index]
}

pub(super) fn strategy_setup_text(app: &App) -> Vec<String> {
    rendered_text(app.strategy_setup_lines())
}

pub(super) fn assert_focused_line_visible(lines: &[Line<'_>], area: Rect) {
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

pub(super) fn account(id: i64, name: &str) -> AccountInfo {
    AccountInfo {
        id,
        name: name.to_string(),
        raw: json!({}),
    }
}

pub(super) fn contract(id: i64, name: &str) -> ContractSuggestion {
    ContractSuggestion {
        id,
        name: name.to_string(),
        description: "test contract".to_string(),
        raw: json!({}),
    }
}

pub(super) fn account_snapshot(
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
        fees: None,
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

pub(super) fn balance_snapshot(
    account_id: i64,
    account_name: &str,
    balance: f64,
) -> AccountSnapshot {
    AccountSnapshot {
        account_id,
        account_name: account_name.to_string(),
        balance: Some(balance),
        cash_balance: Some(balance),
        net_liq: Some(balance),
        // These helpers model a broker whose reported realized PnL follows
        // the synthetic balance so legacy rendering tests remain balance
        // based. Tests that exercise mark-to-market behavior override this
        // field explicitly.
        realized_pnl: Some(balance),
        unrealized_pnl: None,
        fees: None,
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

pub(super) fn balance_snapshot_with_position(
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

pub(super) fn enable_tradovate_controls(app: &mut App) {
    app.selected_broker = BrokerKind::Tradovate;
    app.capabilities = BrokerCapabilities {
        replay: true,
        manual_orders: true,
        automated_orders: true,
        native_protection: true,
    };
}

pub(super) fn select_ready_contract(app: &mut App) {
    app.contract_results = vec![contract(99, "ESZ6")];
    app.selected_contract = 0;
    app.market.contract_id = Some(99);
    app.market.contract_name = Some("ESZ6".to_string());
}

pub(super) fn expect_select_account(rx: &mut ServiceCommandReceiver, account_id: i64) {
    match rx.try_recv().expect("expected select-account command") {
        ServiceCommand::SelectAccount { account_id: actual } => {
            assert_eq!(actual, account_id);
        }
        _ => panic!("expected select-account command"),
    }
}

pub(super) fn running_engine(id: u32, live: bool) -> RunningEngine {
    RunningEngine {
        id,
        cwd: PathBuf::from("/tmp"),
        socket_path: PathBuf::from(format!("/tmp/trader-engine-{id}.sock")),
        socket_is_live: live,
    }
}

pub(super) fn connected_event(broker: BrokerKind) -> ServiceEvent {
    ServiceEvent::Connected {
        broker,
        env: TradingEnvironment::Sim,
        user_name: Some("tester".to_string()),
        auth_mode: AuthMode::TokenFile,
        session_kind: SessionKind::Live,
        capabilities: BrokerCapabilities::default(),
    }
}

pub(super) fn engine_key(id: u32) -> EngineKey {
    EngineKey::from_socket_path(PathBuf::from(format!("/tmp/trader-engine-{id}.sock")).as_path())
}

#[cfg(feature = "replay")]
pub(super) fn replay_test_file(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "trader-replay-{name}-{}-{}.Last.txt",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::write(&path, "20260324 040000 1800000;6603;6603;6603.25;1\n")
        .expect("write replay test file");
    path
}

#[cfg(feature = "replay")]
pub(super) fn replay_cache_test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "trader-replay-cache-{name}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    path
}

#[cfg(feature = "replay")]
pub(super) fn write_replay_cache_manifest(root: &std::path::Path) -> PathBuf {
    let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
    std::fs::create_dir_all(dataset_dir.join("server-bars")).expect("create cache dirs");
    let ts_ns = chrono::DateTime::parse_from_rfc3339("2026-07-23T13:30:00Z")
        .expect("timestamp")
        .timestamp_nanos_opt()
        .expect("timestamp ns");
    std::fs::write(
        dataset_dir.join("server-bars/2026-07-23_to_2026-07-24_1minute.jsonl"),
        json!({
            "timestamp": "2026-07-23T13:30:00Z",
            "ts_ns": ts_ns,
            "open": 7430.0,
            "high": 7431.0,
            "low": 7429.0,
            "close": 7430.5,
            "volume": 100.0
        })
        .to_string(),
    )
    .expect("write cache data");
    let manifest = json!({
        "manifest_version": 1,
        "provider": "tradovate",
        "env": "sim",
        "instrument": {
            "symbol": "MES",
            "name": "Micro E-mini S&P 500",
            "exchange": "CME"
        },
        "contract": {
            "symbol": "MESU6",
            "id": 25866054,
            "expiration": "2026-09-18"
        },
        "display_name": "MESU6 RTH 1m Heikin",
        "coverage": {
            "start": "2026-07-23T13:30:00Z",
            "end": "2026-07-23T20:00:00Z",
            "trading_date": "2026-07-23"
        },
        "source_kind": "server_bars",
        "download_request": {
            "md": "getChart",
            "chartDescription": {
                "underlyingType": "MinuteBar",
                "elementSize": 1,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }
        },
        "tick_specs": {
            "tick_size": 0.25,
            "value_per_point": 5.0
        },
        "files": [{
            "relative_path": "server-bars/2026-07-23_to_2026-07-24_1minute.jsonl",
            "source_kind": "server_bars",
            "format": "jsonl",
            "schema_version": 1,
            "market_shape": {
                "bar_type": {
                    "kind": "minute",
                    "value": 1
                },
                "session_template": "Globex"
            },
            "row_count": 1,
            "first_timestamp": "2026-07-23T13:30:00Z",
            "last_timestamp": "2026-07-23T13:30:00Z",
            "data_hash": {
                "algorithm": "sha256",
                "value": "abc123"
            }
        }],
        "available_bar_shapes": [{
            "kind": "minute",
            "value": 1
        }],
        "available_chart_modes": ["standard", "heikin_ashi"],
        "tags": ["fixture"],
        "notes": "test manifest"
    });
    let manifest_path = dataset_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
    manifest_path
}
