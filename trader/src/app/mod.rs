use crate::broker::{
    AUTO_CLOSE_MINUTES_BEFORE_SESSION_END, AccountInfo, AccountSnapshot, BarType,
    BrokerCapabilities, BrokerKind, ContractSuggestion, InstrumentSessionWindow, LatencySnapshot,
    ManualOrderAction, MarketSnapshot, ReplaySpeed, ServiceCommand, ServiceEvent, SessionKind,
    TradeMarker, TradeMarkerSide, compiled_brokers, default_broker,
};
use crate::config::{AppConfig, AuthMode, LogMode, TradingEnvironment};
use crate::strategies::ema_cross::ema_series;
use crate::strategies::hma_angle::zero_lag_hma_series;
use crate::strategy::{
    LuaSourceMode, NativeSignalTiming, NativeStrategyKind, StrategyKind, StrategyState,
};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph, Tabs, Wrap,
    canvas::{Canvas, Line as CanvasLine},
};
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

pub struct App {
    base_config: AppConfig,
    available_brokers: Vec<BrokerKind>,
    selected_broker: BrokerKind,
    capabilities: BrokerCapabilities,
    form: FormState,
    strategy: StrategyState,
    screen: Screen,
    focus: Focus,
    pub should_quit: bool,
    status: String,
    accounts: Vec<AccountInfo>,
    account_snapshots: Vec<AccountSnapshot>,
    selected_account: usize,
    instrument_query: String,
    bar_type: BarType,
    contract_results: Vec<ContractSuggestion>,
    selected_contract: usize,
    market: MarketSnapshot,
    logs: VecDeque<LogEntry>,
    dashboard_visuals_enabled: bool,
    strategy_runtime: StrategyRuntimeState,
    strategy_numeric_input: Option<NumericInputState>,
    latency: LatencySnapshot,
    session_kind: SessionKind,
    replay_speed: ReplaySpeed,
    last_log_at: Option<Instant>,
    last_market_update_at: Option<Instant>,
}

#[derive(Debug, Clone)]
struct FormState {
    env: TradingEnvironment,
    auth_mode: AuthMode,
    log_mode: LogMode,
    token_override: String,
    username: String,
    password: String,
    api_key: String,
    app_id: String,
    app_version: String,
    cid: String,
    secret: String,
    token_path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    BrokerList,
    Env,
    AuthMode,
    LogMode,
    TokenOverride,
    Username,
    Password,
    ApiKey,
    AppId,
    AppVersion,
    Cid,
    Secret,
    TokenPath,
    Connect,
    ReplayMode,
    StrategyKind,
    OrderQty,
    NativeStrategy,
    NativeSignalTiming,
    NativeReversalMode,
    HmaLength,
    HmaMinAngle,
    HmaAngleLookback,
    HmaBarsRequired,
    HmaLongsOnly,
    HmaInverted,
    HmaTakeProfitTicks,
    HmaStopLossTicks,
    HmaTrailingStop,
    HmaTrailTriggerTicks,
    HmaTrailOffsetTicks,
    EmaFastLength,
    EmaSlowLength,
    EmaInverted,
    EmaTakeProfitTicks,
    EmaStopLossTicks,
    EmaTrailingStop,
    EmaTrailTriggerTicks,
    EmaTrailOffsetTicks,
    LuaSourceMode,
    LuaFilePath,
    LuaEditor,
    StrategyContinue,
    AccountList,
    InstrumentQuery,
    BarTypeToggle,
    ContractList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    BrokerSelect,
    Login,
    Strategy,
    Selection,
    Dashboard,
}

#[derive(Debug, Clone, Default)]
struct StrategyRuntimeState {
    armed: bool,
    last_closed_bar_ts: Option<i64>,
    pending_target_qty: Option<i32>,
    last_summary: String,
}

#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: chrono::DateTime<chrono::Local>,
    elapsed_since_previous: Option<std::time::Duration>,
    message: String,
}

#[derive(Debug, Clone)]
struct NumericInputState {
    focus: Focus,
    value: String,
}

include!("core.rs");
include!("input.rs");
include!("render.rs");
include!("views.rs");
include!("state.rs");
include!("form.rs");
include!("helpers.rs");
