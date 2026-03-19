use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategy::{LuaSourceMode, NativeStrategyKind, StrategyKind, StrategyState};
use crate::tradovate::{
    AUTO_CLOSE_MINUTES_BEFORE_SESSION_END, AccountInfo, AccountSnapshot, BarType,
    ContractSuggestion, InstrumentSessionWindow, LatencySnapshot, ManualOrderAction,
    MarketSnapshot, ServiceCommand, ServiceEvent, TradeMarker, TradeMarkerSide,
};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, Paragraph, Tabs, Wrap,
};
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

pub struct App {
    base_config: AppConfig,
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
    logs: VecDeque<String>,
    strategy_runtime: StrategyRuntimeState,
    strategy_numeric_input: Option<NumericInputState>,
    latency: LatencySnapshot,
    last_market_update_at: Option<Instant>,
}

#[derive(Debug, Clone)]
struct FormState {
    env: TradingEnvironment,
    auth_mode: AuthMode,
    token_override: String,
    username: String,
    password: String,
    app_id: String,
    app_version: String,
    cid: String,
    secret: String,
    token_path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Env,
    AuthMode,
    TokenOverride,
    Username,
    Password,
    AppId,
    AppVersion,
    Cid,
    Secret,
    TokenPath,
    Connect,
    StrategyKind,
    OrderQty,
    NativeStrategy,
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
