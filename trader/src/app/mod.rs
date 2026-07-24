#[cfg(feature = "manual-orders")]
use crate::broker::ManualOrderAction;
use crate::broker::{
    AccountInfo, AccountSnapshot, BarKind, BarType, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, InstrumentSessionWindow, LatencySnapshot, MarketSnapshot, ReplaySpeed,
    ServiceCommand, ServiceEvent, SessionKind, TradeMarker, TradeMarkerSide, compiled_brokers,
    default_broker,
};
use crate::config::{AppConfig, AuthMode, LogMode, TradingEnvironment};
use crate::engine_registry::RunningEngine;
#[cfg(feature = "replay")]
use crate::replay_cache::ReplayCacheLibrary;
use crate::strategies::ema_cross::ema_series;
use crate::strategies::hma_angle::zero_lag_hma_series;
use crate::strategies::hma_cross::hma_series;
use crate::strategy::{
    LuaSourceMode, NativeExecutionPath, NativeReversalMode, NativeSignalTiming, NativeStrategyKind,
    StrategyKind, StrategyState,
};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Clear, Dataset, GraphType, List, ListItem, ListState,
    Paragraph, Row, Table, Tabs, Wrap,
    canvas::{Canvas, Line as CanvasLine},
};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

const UI_LOG_ENTRY_LIMIT: usize = 200;
const PERSISTED_LOG_ENTRY_LIMIT: usize = 10_000;

pub struct App {
    base_config: AppConfig,
    #[cfg(feature = "replay")]
    replay_cache_library: ReplayCacheLibrary,
    available_brokers: Vec<BrokerKind>,
    selected_broker: BrokerKind,
    capabilities: BrokerCapabilities,
    form: FormState,
    strategy: StrategyState,
    screen: Screen,
    focus: Focus,
    running_engines: Vec<RunningEngine>,
    engine_summaries: Vec<EngineSummary>,
    selected_engine: usize,
    engine_creation_enabled: bool,
    pending_engine_lifecycle_confirmation: Option<EngineLifecycleConfirmation>,
    pending_engine_selection_action: Option<EngineSelectionAction>,
    engine_socket_path: Option<PathBuf>,
    active_engine_key: Option<EngineKey>,
    pub should_quit: bool,
    status: String,
    accounts: Vec<AccountInfo>,
    account_snapshots: Vec<AccountSnapshot>,
    selected_account: usize,
    instrument_query: String,
    bar_type: BarType,
    candle_mode: CandleMode,
    contract_results: Vec<ContractSuggestion>,
    selected_contract: usize,
    market: MarketSnapshot,
    logs: VecDeque<LogEntry>,
    persisted_logs: VecDeque<LogEntry>,
    last_saved_log_path: Option<PathBuf>,
    session_stats: SessionStatsState,
    session_stats_show_fees: bool,
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
    EngineList,
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
    NativeSignalDelayBars,
    NativeExecutionPath,
    NativeReversalMode,
    NativeBlockoutEnabled,
    NativeBlockoutMinutes,
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
    BarValue,
    CandleModeToggle,
    ContractList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    EngineSelect,
    BrokerSelect,
    Login,
    Replay,
    Strategy,
    Selection,
    Dashboard,
    Stats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EngineSelectionAction {
    Attach {
        engine_key: EngineKey,
        socket_path: PathBuf,
    },
    CreateNew,
    Refresh,
    Kill {
        id: u32,
    },
    CloseAndKill {
        id: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineLifecycleAction {
    Kill,
    CloseAndKill,
}

impl EngineLifecycleAction {
    fn title(self) -> &'static str {
        match self {
            Self::Kill => "Confirm Kill Engine",
            Self::CloseAndKill => "Confirm Close And Kill",
        }
    }

    fn status_verb(self) -> &'static str {
        match self {
            Self::Kill => "kill",
            Self::CloseAndKill => "close and kill",
        }
    }

    fn running_message(self, id: u32) -> String {
        match self {
            Self::Kill => format!("Killing engine {id}..."),
            Self::CloseAndKill => {
                format!("Closing the selected market and killing engine {id}...")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EngineLifecycleConfirmation {
    action: EngineLifecycleAction,
    engine_key: EngineKey,
    id: u32,
    socket_path: PathBuf,
    state: EngineConnectionState,
    broker_mode: String,
    account: String,
    instrument: String,
    position: String,
    strategy: String,
    latest_status: String,
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
include!("session_stats.rs");
mod engine_observation;
pub(crate) use engine_observation::{EngineConnectionState, EngineKey, EngineSummary};
mod render;
mod views;
include!("state.rs");
include!("form.rs");
include!("helpers.rs");
