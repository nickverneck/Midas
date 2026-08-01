#[cfg(feature = "manual-orders")]
use crate::broker::ManualOrderAction;
#[cfg(feature = "replay")]
use crate::broker::ReplaySignalDiagnostic;
use crate::broker::{
    AccountInfo, AccountSnapshot, BarKind, BarType, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, EngineHistorySnapshot, InstrumentSessionWindow, LatencySnapshot,
    MarketSnapshot, ReplayExecutionLedgerSummary, ReplayLatencyModel, ReplaySpeed, ServiceCommand,
    ServiceEvent, SessionKind, TradeMarker, TradeMarkerSide, compiled_brokers, default_broker,
};
#[cfg(feature = "replay")]
use crate::broker::{ReplayDownloadCacheTarget, ReplayDownloadOperationId, ReplayDownloadPhase};
use crate::config::{AppConfig, AuthMode, LogMode, TradingEnvironment};
use crate::engine_registry::RunningEngine;
#[cfg(feature = "replay")]
use crate::replay_cache::{
    ReplayCacheLibrary, ReplayCacheSourceKind, ReplayDatasetSessionPreset,
    ResolvedReplayDatasetView,
};
use crate::strategies::ema_cross::ema_series;
use crate::strategies::hma_angle::zero_lag_hma_series;
use crate::strategies::hma_cross::hma_series;
use crate::strategy::{
    LuaSourceMode, NativeExecutionPath, NativeReversalMode, NativeSignalTiming, NativeStrategyKind,
    StrategyKind, StrategyState,
};
#[cfg(feature = "replay")]
use crate::tradovate::replay::{
    ReplayEquityPoint, ReplayResultEntry, ReplayResultLibrarySnapshot, ReplaySweepRankingDocument,
    ReplaySweepRankingEntry, ReplaySweepRankingLibrarySnapshot, ReplaySweepRankingMetric,
    ReplaySweepRankingOptions, ReplayTradeExcursion, load_replay_equity,
    load_replay_result_entries, load_replay_signal_diagnostics, load_replay_sweep_ranking_entries,
    rank_replay_sweep,
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
    pending_contract_override: Option<i64>,
    market: MarketSnapshot,
    logs: VecDeque<LogEntry>,
    persisted_logs: VecDeque<LogEntry>,
    last_saved_log_path: Option<PathBuf>,
    session_stats: SessionStatsState,
    engine_history: Option<EngineHistorySnapshot>,
    session_stats_show_fees: bool,
    dashboard_visuals_enabled: bool,
    strategy_runtime: StrategyRuntimeState,
    strategy_numeric_input: Option<NumericInputState>,
    latency: LatencySnapshot,
    session_kind: SessionKind,
    replay_speed: ReplaySpeed,
    replay_execution_ledger: ReplayExecutionLedgerSummary,
    #[cfg(feature = "replay")]
    replay_dataset_index: Option<usize>,
    #[cfg(feature = "replay")]
    replay_dataset_view_path: Option<PathBuf>,
    #[cfg(feature = "replay")]
    replay_view: ReplayView,
    #[cfg(feature = "replay")]
    replay_downloader: ReplayDownloaderState,
    #[cfg(feature = "replay")]
    replay_dataset_views: ReplayDatasetViewsState,
    #[cfg(feature = "replay")]
    replay_analytics: ReplayAnalyticsState,
    analytics_return_screen: Screen,
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
    #[cfg(feature = "replay")]
    ReplayDataset,
    #[cfg(feature = "replay")]
    ReplayInitialCapital,
    #[cfg(feature = "replay")]
    ReplayMarginPerContract,
    #[cfg(feature = "replay")]
    ReplaySafetyBuffer,
    #[cfg(feature = "replay")]
    ReplaySafetyBufferPercent,
    #[cfg(feature = "replay")]
    ReplayViewList,
    #[cfg(feature = "replay")]
    ReplayViewId,
    #[cfg(feature = "replay")]
    ReplayViewPreset,
    #[cfg(feature = "replay")]
    ReplayViewTradingDate,
    #[cfg(feature = "replay")]
    ReplayViewStart,
    #[cfg(feature = "replay")]
    ReplayViewEnd,
    #[cfg(feature = "replay")]
    ReplayViewTimezone,
    #[cfg(feature = "replay")]
    ReplayViewWarmupMinutes,
    #[cfg(feature = "replay")]
    ReplayViewSave,
    #[cfg(feature = "replay")]
    ReplayDownloadProvider,
    #[cfg(feature = "replay")]
    ReplayDownloadEnv,
    #[cfg(feature = "replay")]
    ReplayDownloadInstrument,
    #[cfg(feature = "replay")]
    ReplayDownloadContract,
    #[cfg(feature = "replay")]
    ReplayDownloadStart,
    #[cfg(feature = "replay")]
    ReplayDownloadEnd,
    #[cfg(feature = "replay")]
    ReplayDownloadSource,
    #[cfg(feature = "replay")]
    ReplayDownloadBarType,
    #[cfg(feature = "replay")]
    ReplayDownloadBarValue,
    #[cfg(feature = "replay")]
    ReplayDownloadCandleMode,
    #[cfg(feature = "replay")]
    ReplayDownloadName,
    #[cfg(feature = "replay")]
    ReplayDownloadTags,
    #[cfg(feature = "replay")]
    ReplayDownloadCacheRoot,
    #[cfg(feature = "replay")]
    ReplayDownloadSubmit,
    ContractList,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayView {
    Library,
    Downloader,
    DatasetViews,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Default)]
struct ReplayDatasetViewsState {
    views: Vec<ResolvedReplayDatasetView>,
    warnings: Vec<String>,
    selected_index: Option<usize>,
    editor: Option<ReplayDatasetViewEditorState>,
    message: String,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayDatasetViewEditorState {
    id: String,
    preset: ReplayDatasetSessionPreset,
    trading_date: String,
    start: String,
    end: String,
    timezone: String,
    warmup_minutes: String,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayDownloadWorkflow {
    New,
    Extend,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayDownloaderState {
    workflow: ReplayDownloadWorkflow,
    active_operation_id: Option<ReplayDownloadOperationId>,
    target: Option<ReplayDownloadCacheTarget>,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument_query: String,
    contract_results: Vec<ContractSuggestion>,
    selected_contract: usize,
    exact_contract: Option<ContractSuggestion>,
    start_date: String,
    end_date: String,
    source_kind: Option<ReplayCacheSourceKind>,
    bar_type: BarType,
    candle_mode: CandleMode,
    display_name: String,
    tags: String,
    cache_root: String,
    phase: ReplayDownloadPhase,
    phase_message: String,
    estimated_rows: Option<u64>,
    estimated_bytes: Option<u64>,
    actual_rows: Option<u64>,
    actual_bytes: Option<u64>,
    suggestion_basis: Option<String>,
}

#[cfg(feature = "replay")]
impl ReplayDownloaderState {
    fn new(config: &AppConfig) -> Self {
        let today = chrono::Utc::now()
            .date_naive()
            .format("%Y-%m-%d")
            .to_string();
        Self {
            workflow: ReplayDownloadWorkflow::New,
            active_operation_id: None,
            target: None,
            provider: BrokerKind::Tradovate,
            env: config.env,
            instrument_query: String::new(),
            contract_results: Vec::new(),
            selected_contract: 0,
            exact_contract: None,
            start_date: today.clone(),
            end_date: today,
            source_kind: Some(ReplayCacheSourceKind::ServerBars),
            bar_type: BarType::default(),
            candle_mode: config.candle_mode,
            display_name: String::new(),
            tags: String::new(),
            cache_root: config.replay_cache_dir.display().to_string(),
            phase: ReplayDownloadPhase::Idle,
            phase_message: "Search for an exact contract to continue.".to_string(),
            estimated_rows: None,
            estimated_bytes: None,
            actual_rows: None,
            actual_bytes: None,
            suggestion_basis: None,
        }
    }

    fn begin_operation(&mut self) -> ReplayDownloadOperationId {
        let operation_id = ReplayDownloadOperationId::next();
        self.active_operation_id = Some(operation_id);
        operation_id
    }

    fn invalidate_operation(&mut self) {
        self.active_operation_id = None;
    }

    fn cancel_active_operation(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if let Some(operation_id) = self.active_operation_id.take() {
            let _ = cmd_tx.send(ServiceCommand::CancelReplayDownloadOperation { operation_id });
        }
    }

    fn accepts(&self, operation_id: ReplayDownloadOperationId) -> bool {
        self.active_operation_id == Some(operation_id)
    }
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
    Analytics,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalyticsFocus {
    Runs,
    Trades,
    Signals,
    Sweeps,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalyticsTradeSort {
    TradeId,
    LargestGiveback,
    LowestCapture,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalyticsSignalFilter {
    All,
    Orders,
    Blocked,
    SignalsOnly,
}

#[cfg(feature = "replay")]
impl AnalyticsSignalFilter {
    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Orders => "orders",
            Self::Blocked => "blocked/gated",
            Self::SignalsOnly => "non-hold signals",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::All => Self::Orders,
            Self::Orders => Self::Blocked,
            Self::Blocked => Self::SignalsOnly,
            Self::SignalsOnly => Self::All,
        }
    }
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayAnalyticsState {
    root: PathBuf,
    entries: Vec<ReplayResultEntry>,
    warnings: Vec<String>,
    selected_run: usize,
    selected_trade: usize,
    selected_signal: usize,
    selected_fee_scenario: usize,
    comparison_run: Option<usize>,
    focus: AnalyticsFocus,
    trade_sort: AnalyticsTradeSort,
    signal_filter: AnalyticsSignalFilter,
    signals: Vec<ReplaySignalDiagnostic>,
    signal_load_error: Option<String>,
    equity: Vec<ReplayEquityPoint>,
    equity_load_error: Option<String>,
    sweep_entries: Vec<ReplaySweepRankingEntry>,
    sweep_warnings: Vec<String>,
    selected_sweep: usize,
    selected_sweep_row: usize,
    sweep_metric: ReplaySweepRankingMetric,
    sweep_fee_scenario: Option<String>,
    sweep_min_closed_trades: usize,
    sweep_max_drawdown_pct: Option<f64>,
    sweep_ranking: Option<ReplaySweepRankingDocument>,
    sweep_ranking_error: Option<String>,
}

#[cfg(feature = "replay")]
impl ReplayAnalyticsState {
    fn new(root: PathBuf) -> Self {
        let mut state = Self {
            root,
            entries: Vec::new(),
            warnings: Vec::new(),
            selected_run: 0,
            selected_trade: 0,
            selected_signal: 0,
            selected_fee_scenario: 0,
            comparison_run: None,
            focus: AnalyticsFocus::Runs,
            trade_sort: AnalyticsTradeSort::TradeId,
            signal_filter: AnalyticsSignalFilter::All,
            signals: Vec::new(),
            signal_load_error: None,
            equity: Vec::new(),
            equity_load_error: None,
            sweep_entries: Vec::new(),
            sweep_warnings: Vec::new(),
            selected_sweep: 0,
            selected_sweep_row: 0,
            sweep_metric: ReplaySweepRankingMetric::Robustness,
            sweep_fee_scenario: None,
            sweep_min_closed_trades: 0,
            sweep_max_drawdown_pct: None,
            sweep_ranking: None,
            sweep_ranking_error: None,
        };
        state.refresh();
        state
    }

    fn refresh(&mut self) {
        let selected_path = self
            .entries
            .get(self.selected_run)
            .map(|entry| entry.path.clone());
        let comparison_path = self.comparison_entry().map(|entry| entry.path.clone());
        let selected_sweep_path = self
            .sweep_entries
            .get(self.selected_sweep)
            .map(|entry| entry.path.clone());
        let ReplayResultLibrarySnapshot { entries, warnings } =
            load_replay_result_entries(&self.root);
        self.entries = entries;
        self.warnings = warnings;
        self.selected_run = selected_path
            .and_then(|path| self.entries.iter().position(|entry| entry.path == path))
            .unwrap_or(0);
        self.comparison_run = comparison_path
            .and_then(|path| self.entries.iter().position(|entry| entry.path == path));
        self.refresh_sweeps(selected_sweep_path);
        self.clamp_selection();
        if self.focus == AnalyticsFocus::Signals {
            self.load_selected_signals();
        } else if self.focus != AnalyticsFocus::Sweeps {
            self.clear_selected_signals();
        }
        self.load_selected_equity();
    }

    fn refresh_sweeps(&mut self, selected_path: Option<PathBuf>) {
        let mut entries = Vec::new();
        let mut warnings = Vec::new();
        let mut roots = vec![self.root.clone(), PathBuf::from(".")];
        if let Some(parent) = self.root.parent() {
            roots.push(parent.to_path_buf());
        }
        for root in roots {
            let ReplaySweepRankingLibrarySnapshot {
                entries: discovered,
                warnings: discovered_warnings,
            } = load_replay_sweep_ranking_entries(&root);
            warnings.extend(discovered_warnings);
            for entry in discovered {
                let duplicate = entries.iter().any(|existing: &ReplaySweepRankingEntry| {
                    same_path(&existing.path, &entry.path)
                });
                if !duplicate {
                    entries.push(entry);
                }
            }
        }
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        self.sweep_entries = entries;
        self.sweep_warnings = warnings;
        self.selected_sweep = selected_path
            .and_then(|path| {
                self.sweep_entries
                    .iter()
                    .position(|entry| same_path(&entry.path, &path))
            })
            .unwrap_or(0);
        self.load_selected_sweep(false);
    }

    fn load_selected_sweep(&mut self, rerank: bool) {
        let Some(entry) = self.sweep_entries.get(self.selected_sweep).cloned() else {
            self.sweep_ranking = None;
            self.sweep_ranking_error = None;
            self.selected_sweep_row = 0;
            return;
        };
        if !rerank {
            self.sweep_metric = entry.document.options.metric;
            self.sweep_fee_scenario = entry
                .document
                .options
                .fee_scenario
                .clone()
                .filter(|scenario| !scenario.eq_ignore_ascii_case("active"));
            self.sweep_min_closed_trades = entry.document.options.min_closed_trades;
            self.sweep_max_drawdown_pct = entry.document.options.max_drawdown_pct;
            self.sweep_ranking = Some(entry.document);
            self.sweep_ranking_error = None;
            self.clamp_sweep_selection();
            return;
        }
        let summary_path = resolve_sweep_summary_path(&entry);
        let options = ReplaySweepRankingOptions {
            metric: self.sweep_metric,
            fee_scenario: self.sweep_fee_scenario.clone(),
            limit: 20,
            min_closed_trades: self.sweep_min_closed_trades,
            max_drawdown_pct: self.sweep_max_drawdown_pct,
        };
        match rank_replay_sweep(&summary_path, None, options, None, None) {
            Ok(document) => {
                self.sweep_ranking = Some(document);
                self.sweep_ranking_error = None;
            }
            Err(error) => {
                self.sweep_ranking = Some(entry.document);
                self.sweep_ranking_error = Some(error.to_string());
            }
        }
        self.clamp_sweep_selection();
    }

    fn clamp_selection(&mut self) {
        if self.entries.is_empty() {
            self.selected_run = 0;
            self.selected_trade = 0;
            self.selected_signal = 0;
            self.selected_fee_scenario = 0;
            self.clear_selected_signals();
            return;
        }
        self.selected_run = self.selected_run.min(self.entries.len() - 1);
        let trade_count = self.sorted_trades().len();
        self.selected_trade = self.selected_trade.min(trade_count.saturating_sub(1));
        let fee_count = self
            .selected_entry()
            .map(|entry| entry.document.fee_scenarios.len())
            .unwrap_or_default();
        self.selected_fee_scenario = self.selected_fee_scenario.min(fee_count.saturating_sub(1));
        let signal_count = self.filtered_signals().len();
        self.selected_signal = self.selected_signal.min(signal_count.saturating_sub(1));
        self.clamp_sweep_selection();
    }

    fn clamp_sweep_selection(&mut self) {
        if self.sweep_entries.is_empty() {
            self.selected_sweep = 0;
            self.selected_sweep_row = 0;
            return;
        }
        self.selected_sweep = self.selected_sweep.min(self.sweep_entries.len() - 1);
        let row_count = self
            .sweep_ranking
            .as_ref()
            .map(|document| document.rows.len())
            .unwrap_or_default();
        self.selected_sweep_row = self.selected_sweep_row.min(row_count.saturating_sub(1));
    }

    fn clear_selected_signals(&mut self) {
        self.signals.clear();
        self.signal_load_error = None;
        self.selected_signal = 0;
    }

    fn load_selected_equity(&mut self) {
        self.equity.clear();
        self.equity_load_error = None;
        if let Some(entry) = self.selected_entry() {
            match load_replay_equity(entry) {
                Ok(rows) => self.equity = rows,
                Err(error) => self.equity_load_error = Some(error.to_string()),
            }
        }
    }

    fn load_selected_signals(&mut self) {
        self.clear_selected_signals();
        let loaded = self.selected_entry().map(load_replay_signal_diagnostics);
        match loaded {
            Some(Ok(signals)) => self.signals = signals,
            Some(Err(error)) => self.signal_load_error = Some(error.to_string()),
            None => {}
        }
        self.selected_signal = self
            .selected_signal
            .min(self.filtered_signals().len().saturating_sub(1));
    }

    fn selected_entry(&self) -> Option<&ReplayResultEntry> {
        self.entries.get(self.selected_run)
    }

    fn selected_excursions(&self) -> &[ReplayTradeExcursion] {
        self.selected_entry()
            .and_then(|entry| entry.document.trade_excursions.as_deref())
            .unwrap_or(&[])
    }

    fn sorted_trades(&self) -> Vec<&ReplayTradeExcursion> {
        let mut trades = self.selected_excursions().iter().collect::<Vec<_>>();
        match self.trade_sort {
            AnalyticsTradeSort::TradeId => trades.sort_by_key(|trade| trade.trade_id),
            AnalyticsTradeSort::LargestGiveback => trades.sort_by(|left, right| {
                right
                    .giveback
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&left.giveback.unwrap_or(f64::NEG_INFINITY))
                    .then_with(|| left.trade_id.cmp(&right.trade_id))
            }),
            AnalyticsTradeSort::LowestCapture => trades.sort_by(|left, right| {
                left.mfe_capture_ratio
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right.mfe_capture_ratio.unwrap_or(f64::INFINITY))
                    .then_with(|| left.trade_id.cmp(&right.trade_id))
            }),
        }
        trades
    }

    fn filtered_signals(&self) -> Vec<&ReplaySignalDiagnostic> {
        self.signals
            .iter()
            .filter(|signal| match self.signal_filter {
                AnalyticsSignalFilter::All => true,
                AnalyticsSignalFilter::Orders => signal.order_action.is_some(),
                AnalyticsSignalFilter::Blocked => {
                    !is_non_blocking_signal_decision(&signal.decision)
                }
                AnalyticsSignalFilter::SignalsOnly => !signal.signal.eq_ignore_ascii_case("hold"),
            })
            .collect()
    }

    fn selected_signal(&self) -> Option<&ReplaySignalDiagnostic> {
        self.filtered_signals().get(self.selected_signal).copied()
    }

    fn selected_fee_scenario(&self) -> Option<&crate::tradovate::replay::ReplayFeeScenario> {
        self.selected_entry()
            .and_then(|entry| entry.document.fee_scenarios.get(self.selected_fee_scenario))
    }

    fn comparison_entry(&self) -> Option<&ReplayResultEntry> {
        self.comparison_run
            .and_then(|index| self.entries.get(index))
    }

    fn cycle_sort(&mut self) {
        self.trade_sort = match self.trade_sort {
            AnalyticsTradeSort::TradeId => AnalyticsTradeSort::LargestGiveback,
            AnalyticsTradeSort::LargestGiveback => AnalyticsTradeSort::LowestCapture,
            AnalyticsTradeSort::LowestCapture => AnalyticsTradeSort::TradeId,
        };
        self.selected_trade = 0;
    }

    fn cycle_signal_filter(&mut self) {
        self.signal_filter = self.signal_filter.next();
        self.selected_signal = 0;
    }

    fn cycle_fee_scenario(&mut self, direction: i32) {
        let Some(entry) = self.selected_entry() else {
            return;
        };
        let count = entry.document.fee_scenarios.len();
        if count == 0 {
            self.selected_fee_scenario = 0;
            return;
        }
        self.selected_fee_scenario = if direction < 0 {
            if self.selected_fee_scenario == 0 {
                count - 1
            } else {
                self.selected_fee_scenario - 1
            }
        } else {
            (self.selected_fee_scenario + 1) % count
        };
    }

    fn selected_sweep_entry(&self) -> Option<&ReplaySweepRankingEntry> {
        self.sweep_entries.get(self.selected_sweep)
    }

    fn selected_sweep_row(&self) -> Option<&crate::tradovate::replay::ReplaySweepRankingRow> {
        self.sweep_ranking
            .as_ref()
            .and_then(|document| document.rows.get(self.selected_sweep_row))
    }

    fn sweep_fee_scenarios(&self) -> Vec<String> {
        let Some(document) = self.sweep_ranking.as_ref() else {
            return vec!["active".to_string()];
        };
        let mut scenarios = vec!["active".to_string()];
        let mut add = |name: &str| {
            if !name.trim().is_empty()
                && !scenarios
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(name))
            {
                scenarios.push(name.to_string());
            }
        };
        add(&document.fee_scenario);
        for row in &document.rows {
            add(&row.fee_scenario);
        }
        scenarios
    }

    fn cycle_sweep_fee_scenario(&mut self, direction: i32) {
        let scenarios = self.sweep_fee_scenarios();
        if scenarios.is_empty() {
            self.sweep_fee_scenario = None;
            return;
        }
        let current = self.sweep_fee_scenario.as_deref().unwrap_or("active");
        let current_index = scenarios
            .iter()
            .position(|scenario| scenario.eq_ignore_ascii_case(current))
            .unwrap_or(0);
        let next = if direction < 0 {
            if current_index == 0 {
                scenarios.len() - 1
            } else {
                current_index - 1
            }
        } else {
            (current_index + 1) % scenarios.len()
        };
        self.sweep_fee_scenario = (next > 0).then(|| scenarios[next].clone());
        self.load_selected_sweep(true);
    }

    fn cycle_sweep_metric(&mut self, direction: i32) {
        let metrics = ReplaySweepRankingMetric::all();
        let current = metrics
            .iter()
            .position(|metric| *metric == self.sweep_metric)
            .unwrap_or(0);
        let next = if direction < 0 {
            if current == 0 {
                metrics.len() - 1
            } else {
                current - 1
            }
        } else {
            (current + 1) % metrics.len()
        };
        self.sweep_metric = metrics[next];
        self.load_selected_sweep(true);
    }

    fn cycle_sweep_filter(&mut self, drawdown: bool, direction: i32) {
        if drawdown {
            let values = [None, Some(5.0), Some(10.0), Some(20.0), Some(50.0)];
            let current = values
                .iter()
                .position(|value| *value == self.sweep_max_drawdown_pct)
                .unwrap_or(0);
            let next = if direction < 0 {
                if current == 0 {
                    values.len() - 1
                } else {
                    current - 1
                }
            } else {
                (current + 1) % values.len()
            };
            self.sweep_max_drawdown_pct = values[next];
        } else {
            let values = [0, 1, 5, 10, 20];
            let current = values
                .iter()
                .position(|value| *value == self.sweep_min_closed_trades)
                .unwrap_or(0);
            let next = if direction < 0 {
                if current == 0 {
                    values.len() - 1
                } else {
                    current - 1
                }
            } else {
                (current + 1) % values.len()
            };
            self.sweep_min_closed_trades = values[next];
        }
        self.load_selected_sweep(true);
    }

    fn cycle_sweep_source(&mut self, direction: i32) {
        if self.sweep_entries.is_empty() {
            return;
        }
        self.selected_sweep = if direction < 0 {
            if self.selected_sweep == 0 {
                self.sweep_entries.len() - 1
            } else {
                self.selected_sweep - 1
            }
        } else {
            (self.selected_sweep + 1) % self.sweep_entries.len()
        };
        self.selected_sweep_row = 0;
        self.load_selected_sweep(false);
    }
}

#[cfg(feature = "replay")]
fn same_path(left: &std::path::Path, right: &std::path::Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

#[cfg(feature = "replay")]
fn resolve_sweep_summary_path(entry: &ReplaySweepRankingEntry) -> PathBuf {
    let source = &entry.document.source_summary;
    if source.is_absolute() || source.is_file() {
        return source.clone();
    }
    let relative = entry
        .path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(source);
    if relative.is_file() {
        relative
    } else {
        source.clone()
    }
}

#[cfg(feature = "replay")]
fn is_non_blocking_signal_decision(decision: &str) -> bool {
    matches!(
        decision,
        "dispatching" | "target_already_actual" | "target_already_current" | "dispatch_noop"
    )
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
mod state;
mod views;
use state::{DisplayedAutoTrail, StrategyReadinessStatus};
include!("form.rs");
include!("helpers.rs");
