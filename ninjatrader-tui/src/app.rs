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

impl App {
    pub fn new(config: AppConfig) -> Self {
        let form = FormState::from_config(&config);
        let mut app = Self {
            base_config: config,
            form,
            strategy: StrategyState::new(),
            screen: Screen::Login,
            focus: Focus::Env,
            should_quit: false,
            status: "Idle".to_string(),
            accounts: Vec::new(),
            account_snapshots: Vec::new(),
            selected_account: 0,
            instrument_query: String::new(),
            bar_type: BarType::default(),
            contract_results: Vec::new(),
            selected_contract: 0,
            market: MarketSnapshot::default(),
            logs: VecDeque::new(),
            strategy_runtime: StrategyRuntimeState::default(),
            strategy_numeric_input: None,
            latency: LatencySnapshot::default(),
            last_market_update_at: None,
        };
        app.push_log(
            "Phase 1 enabled: auth, account selection, contract search, 1m history/live."
                .to_string(),
        );
        app.push_log("Dashboard hotkeys enabled: b buy, s sell, c close.".to_string());
        app.push_log(
            "Native HMA Angle and EMA Crossover strategies can auto-trade closed 1m bars once armed from Strategy."
                .to_string(),
        );
        app
    }

    pub fn current_config(&self) -> AppConfig {
        let mut cfg = self.base_config.clone();
        cfg.env = self.form.env;
        cfg.auth_mode = self.form.auth_mode;
        cfg.token_override = self.form.token_override.clone();
        cfg.username = self.form.username.clone();
        cfg.password = self.form.password.clone();
        cfg.app_id = self.form.app_id.clone();
        cfg.app_version = self.form.app_version.clone();
        cfg.cid = self.form.cid.clone();
        cfg.secret = self.form.secret.clone();
        cfg.token_path = self.form.token_path.clone().into();
        cfg
    }

    pub fn handle_service_event(
        &mut self,
        event: ServiceEvent,
        _cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match event {
            ServiceEvent::Status(message) => {
                self.status = message.clone();
                self.push_log(message);
            }
            ServiceEvent::Error(message) => {
                self.status = format!("Error: {message}");
                self.push_log(format!("ERROR: {message}"));
            }
            ServiceEvent::Connected {
                env,
                user_name,
                auth_mode,
            } => {
                self.form.env = env;
                self.form.auth_mode = auth_mode;
                self.screen = Screen::Selection;
                self.focus = Focus::AccountList;
                self.status = match user_name {
                    Some(name) => format!("Connected to {} as {}", env.label(), name),
                    None => format!("Connected to {}", env.label()),
                };
                self.push_log(self.status.clone());
            }
            ServiceEvent::Disconnected => {
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                self.accounts.clear();
                self.account_snapshots.clear();
                self.contract_results.clear();
                self.market = MarketSnapshot::default();
                self.strategy_runtime = StrategyRuntimeState::default();
                self.latency = LatencySnapshot::default();
                self.last_market_update_at = None;
                self.status = "Disconnected".to_string();
                self.push_log("Disconnected".to_string());
            }
            ServiceEvent::AccountsLoaded(accounts) => {
                self.accounts = accounts;
                if self.selected_account >= self.accounts.len() {
                    self.selected_account = 0;
                }
                self.push_log(format!("Loaded {} accounts", self.accounts.len()));
            }
            ServiceEvent::AccountSnapshotsLoaded(snapshots) => {
                self.account_snapshots = snapshots;
            }
            ServiceEvent::ContractSearchResults { query, results } => {
                self.contract_results = results;
                self.selected_contract = 0;
                self.push_log(format!(
                    "Contract search `{query}` returned {} result(s)",
                    self.contract_results.len()
                ));
            }
            ServiceEvent::MarketSnapshot(snapshot) => {
                self.market = snapshot;
                self.last_market_update_at = Some(Instant::now());
            }
            ServiceEvent::TradeMarkersUpdated(markers) => {
                self.market.trade_markers = markers;
            }
            ServiceEvent::Latency(snapshot) => {
                self.latency = snapshot;
            }
            ServiceEvent::ExecutionState(snapshot) => {
                self.strategy.apply_execution_config(&snapshot.config);
                self.strategy_runtime.armed = snapshot.runtime.armed;
                self.strategy_runtime.last_closed_bar_ts = snapshot.runtime.last_closed_bar_ts;
                self.strategy_runtime.pending_target_qty = snapshot.runtime.pending_target_qty;
                self.strategy_runtime.last_summary = snapshot.runtime.last_summary;
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }

        match key.code {
            KeyCode::F(1) => {
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                return;
            }
            KeyCode::F(2) => {
                self.screen = Screen::Selection;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::F(3) => {
                self.screen = Screen::Strategy;
                self.focus = Focus::StrategyKind;
                return;
            }
            KeyCode::F(4) => {
                self.screen = Screen::Dashboard;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::Esc => {
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                return;
            }
            _ => {}
        }

        if !self.is_text_focus() && key.code == KeyCode::Char('q') {
            self.should_quit = true;
            return;
        }

        match self.screen {
            Screen::Login => self.handle_login_key(key, cmd_tx),
            Screen::Strategy => self.handle_strategy_key(key, cmd_tx),
            Screen::Selection => self.handle_selection_key(key, cmd_tx),
            Screen::Dashboard => self.handle_dashboard_key(key, cmd_tx),
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(20),
                Constraint::Length(8),
            ])
            .split(frame.area());

        self.render_header(frame, layout[0]);
        match self.screen {
            Screen::Login => self.render_login_screen(frame, layout[1]),
            Screen::Strategy => self.render_strategy_screen(frame, layout[1]),
            Screen::Selection => self.render_selection_screen(frame, layout[1]),
            Screen::Dashboard => self.render_dashboard(frame, layout[1]),
        }
        self.render_logs(frame, layout[2]);
    }

    fn handle_login_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                self.focus = self.prev_login_focus();
                return;
            }
            KeyCode::Down | KeyCode::Tab => {
                self.focus = self.next_login_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::Env => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.form.env = self.form.env.toggle();
                }
                _ => {}
            },
            Focus::AuthMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.form.auth_mode = self.form.auth_mode.toggle();
                }
                _ => {}
            },
            Focus::Connect => {
                if key.code == KeyCode::Enter {
                    let cfg = self.current_config();
                    let _ = cmd_tx.send(ServiceCommand::Connect(cfg));
                    self.push_log("Connect requested".to_string());
                }
            }
            Focus::TokenOverride => edit_string(&mut self.form.token_override, key),
            Focus::Username => edit_string(&mut self.form.username, key),
            Focus::Password => edit_string(&mut self.form.password, key),
            Focus::AppId => edit_string(&mut self.form.app_id, key),
            Focus::AppVersion => edit_string(&mut self.form.app_version, key),
            Focus::Cid => edit_string(&mut self.form.cid, key),
            Focus::Secret => edit_string(&mut self.form.secret, key),
            Focus::TokenPath => edit_string(&mut self.form.token_path, key),
            Focus::StrategyKind
            | Focus::OrderQty
            | Focus::NativeStrategy
            | Focus::HmaLength
            | Focus::HmaMinAngle
            | Focus::HmaAngleLookback
            | Focus::HmaBarsRequired
            | Focus::HmaLongsOnly
            | Focus::HmaInverted
            | Focus::HmaTakeProfitTicks
            | Focus::HmaStopLossTicks
            | Focus::HmaTrailingStop
            | Focus::HmaTrailTriggerTicks
            | Focus::HmaTrailOffsetTicks
            | Focus::EmaFastLength
            | Focus::EmaSlowLength
            | Focus::EmaInverted
            | Focus::EmaTakeProfitTicks
            | Focus::EmaStopLossTicks
            | Focus::EmaTrailingStop
            | Focus::EmaTrailTriggerTicks
            | Focus::EmaTrailOffsetTicks
            | Focus::LuaSourceMode
            | Focus::LuaFilePath
            | Focus::LuaEditor
            | Focus::StrategyContinue
            | Focus::AccountList
            | Focus::InstrumentQuery
            | Focus::BarTypeToggle
            | Focus::ContractList => {}
        }
    }

    fn handle_strategy_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                self.clear_strategy_numeric_input();
                self.focus = self.prev_strategy_focus();
                return;
            }
            KeyCode::Down | KeyCode::Tab => {
                self.clear_strategy_numeric_input();
                self.focus = self.next_strategy_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::StrategyKind => match key.code {
                KeyCode::Left => {
                    self.strategy.kind = self.strategy.kind.prev();
                    self.disarm_native_strategy(cmd_tx);
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.kind = self.strategy.kind.next();
                    self.disarm_native_strategy(cmd_tx);
                }
                _ => {}
            },
            Focus::NativeStrategy => match key.code {
                KeyCode::Left => {
                    self.strategy.native_strategy = self.strategy.native_strategy.prev();
                    self.disarm_native_strategy(cmd_tx);
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_strategy = self.strategy.native_strategy.next();
                    self.disarm_native_strategy(cmd_tx);
                }
                _ => {}
            },
            Focus::OrderQty => {
                if edit_strategy_i32(
                    &mut self.strategy_numeric_input,
                    Focus::OrderQty,
                    &mut self.strategy.order_qty,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaLength => {
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaLength,
                    &mut self.strategy.native_hma.hma_length,
                    key,
                    2,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaMinAngle => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaMinAngle,
                    &mut self.strategy.native_hma.min_angle,
                    key,
                    0.0,
                    0.5,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaAngleLookback => {
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaAngleLookback,
                    &mut self.strategy.native_hma.angle_lookback,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaBarsRequired => {
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaBarsRequired,
                    &mut self.strategy.native_hma.bars_required_to_trade,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaLongsOnly => {
                if toggle_bool(&mut self.strategy.native_hma.longs_only, key) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaInverted => {
                if toggle_bool(&mut self.strategy.native_hma.inverted, key) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaTakeProfitTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTakeProfitTicks,
                    &mut self.strategy.native_hma.take_profit_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaStopLossTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaStopLossTicks,
                    &mut self.strategy.native_hma.stop_loss_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaTrailingStop => {
                if toggle_bool(&mut self.strategy.native_hma.use_trailing_stop, key) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaTrailTriggerTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTrailTriggerTicks,
                    &mut self.strategy.native_hma.trail_trigger_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::HmaTrailOffsetTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTrailOffsetTicks,
                    &mut self.strategy.native_hma.trail_offset_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaFastLength => {
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::EmaFastLength,
                    &mut self.strategy.native_ema.fast_length,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaSlowLength => {
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::EmaSlowLength,
                    &mut self.strategy.native_ema.slow_length,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaInverted => {
                if toggle_bool(&mut self.strategy.native_ema.inverted, key) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaTakeProfitTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::EmaTakeProfitTicks,
                    &mut self.strategy.native_ema.take_profit_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaStopLossTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::EmaStopLossTicks,
                    &mut self.strategy.native_ema.stop_loss_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaTrailingStop => {
                if toggle_bool(&mut self.strategy.native_ema.use_trailing_stop, key) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaTrailTriggerTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::EmaTrailTriggerTicks,
                    &mut self.strategy.native_ema.trail_trigger_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::EmaTrailOffsetTicks => {
                if edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::EmaTrailOffsetTicks,
                    &mut self.strategy.native_ema.trail_offset_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::LuaSourceMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.lua_source_mode = self.strategy.lua_source_mode.toggle();
                    self.disarm_native_strategy(cmd_tx);
                }
                _ => {}
            },
            Focus::LuaFilePath => {
                if key.code == KeyCode::Enter {
                    match self.strategy.load_lua_file() {
                        Ok(lines) => self.push_log(format!(
                            "Loaded Lua file `{}` ({} lines)",
                            self.strategy.lua_file_path, lines
                        )),
                        Err(err) => self.push_log(format!(
                            "ERROR: failed to load Lua file `{}`: {}",
                            self.strategy.lua_file_path, err
                        )),
                    }
                } else {
                    edit_string(&mut self.strategy.lua_file_path, key);
                    self.disarm_native_strategy(cmd_tx);
                }
            }
            Focus::LuaEditor => {
                let _ = self.strategy.lua_editor.handle_key(key);
                self.disarm_native_strategy(cmd_tx);
            }
            Focus::StrategyContinue => {
                if key.code == KeyCode::Enter {
                    self.screen = Screen::Dashboard;
                    self.focus = Focus::AccountList;
                    self.arm_native_strategy(cmd_tx);
                    self.push_log(format!(
                        "Strategy selected: {}",
                        self.strategy.summary_label()
                    ));
                }
            }
            Focus::Env
            | Focus::AuthMode
            | Focus::TokenOverride
            | Focus::Username
            | Focus::Password
            | Focus::AppId
            | Focus::AppVersion
            | Focus::Cid
            | Focus::Secret
            | Focus::TokenPath
            | Focus::Connect
            | Focus::AccountList
            | Focus::InstrumentQuery
            | Focus::BarTypeToggle
            | Focus::ContractList => {}
        }
    }

    fn handle_selection_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        match key.code {
            KeyCode::BackTab => {
                self.focus = self.prev_selection_focus();
                return;
            }
            KeyCode::Tab => {
                self.focus = self.next_selection_focus();
                return;
            }
            _ => {}
        }

        if self.focus == Focus::AccountList {
            match key.code {
                KeyCode::Up => {
                    self.selected_account = self.selected_account.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    if self.selected_account + 1 < self.accounts.len() {
                        self.selected_account += 1;
                    }
                    return;
                }
                KeyCode::Enter => {
                    if let Some(account) = self.accounts.get(self.selected_account) {
                        let _ = cmd_tx.send(ServiceCommand::SelectAccount {
                            account_id: account.id,
                        });
                    }
                    return;
                }
                KeyCode::Left => {
                    self.focus = Focus::ContractList;
                    return;
                }
                KeyCode::Right => {
                    self.focus = Focus::InstrumentQuery;
                    return;
                }
                _ => {}
            }
        }

        if self.focus == Focus::ContractList {
            match key.code {
                KeyCode::Up => {
                    self.selected_contract = self.selected_contract.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    if self.selected_contract + 1 < self.contract_results.len() {
                        self.selected_contract += 1;
                    }
                    return;
                }
                KeyCode::Enter => {
                    if let Some(contract) =
                        self.contract_results.get(self.selected_contract).cloned()
                    {
                        let _ = cmd_tx.send(ServiceCommand::SubscribeBars { contract, bar_type: self.bar_type });
                        self.screen = Screen::Strategy;
                        self.focus = Focus::StrategyKind;
                    }
                    return;
                }
                KeyCode::Left => {
                    self.focus = Focus::InstrumentQuery;
                    return;
                }
                KeyCode::Right => {
                    self.focus = Focus::AccountList;
                    return;
                }
                _ => {}
            }
        }

        match self.focus {
            Focus::InstrumentQuery => {
                match key.code {
                    KeyCode::Up => {
                        self.focus = Focus::AccountList;
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = Focus::BarTypeToggle;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = Focus::AccountList;
                        return;
                    }
                    KeyCode::Right => {
                        self.focus = Focus::BarTypeToggle;
                        return;
                    }
                    _ => {}
                }
                if key.code == KeyCode::Enter {
                    if !self.instrument_query.trim().is_empty() {
                        let _ = cmd_tx.send(ServiceCommand::SearchContracts {
                            query: self.instrument_query.trim().to_string(),
                            limit: self.base_config.contract_suggest_limit,
                        });
                    }
                } else {
                    edit_string(&mut self.instrument_query, key);
                }
            }
            Focus::BarTypeToggle => {
                match key.code {
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        self.bar_type = self.bar_type.toggle();
                        return;
                    }
                    KeyCode::Up => {
                        self.focus = Focus::InstrumentQuery;
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = Focus::ContractList;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = Focus::InstrumentQuery;
                        return;
                    }
                    KeyCode::Right => {
                        self.focus = Focus::ContractList;
                        return;
                    }
                    _ => {}
                }
            }
            Focus::AccountList
            | Focus::ContractList
            | Focus::Env
            | Focus::AuthMode
            | Focus::TokenOverride
            | Focus::Username
            | Focus::Password
            | Focus::AppId
            | Focus::AppVersion
            | Focus::Cid
            | Focus::Secret
            | Focus::TokenPath
            | Focus::StrategyKind
            | Focus::OrderQty
            | Focus::NativeStrategy
            | Focus::HmaLength
            | Focus::HmaMinAngle
            | Focus::HmaAngleLookback
            | Focus::HmaBarsRequired
            | Focus::HmaLongsOnly
            | Focus::HmaInverted
            | Focus::HmaTakeProfitTicks
            | Focus::HmaStopLossTicks
            | Focus::HmaTrailingStop
            | Focus::HmaTrailTriggerTicks
            | Focus::HmaTrailOffsetTicks
            | Focus::EmaFastLength
            | Focus::EmaSlowLength
            | Focus::EmaInverted
            | Focus::EmaTakeProfitTicks
            | Focus::EmaStopLossTicks
            | Focus::EmaTrailingStop
            | Focus::EmaTrailTriggerTicks
            | Focus::EmaTrailOffsetTicks
            | Focus::LuaSourceMode
            | Focus::LuaFilePath
            | Focus::LuaEditor
            | Focus::StrategyContinue
            | Focus::Connect => {}
        }
    }

    fn handle_dashboard_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let action = match key.code {
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match ch.to_ascii_lowercase() {
                    'b' => Some(ManualOrderAction::Buy),
                    's' => Some(ManualOrderAction::Sell),
                    'c' => Some(ManualOrderAction::Close),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(action) = action {
            let _ = cmd_tx.send(ServiceCommand::ManualOrder { action });
        }
    }

    fn render_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area);

        let screen_label = match self.screen {
            Screen::Login => "Login",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
        };
        let help = match self.screen {
            Screen::Login => {
                "F2 selection | Up/Down focus | Left/Right toggle | Enter connect | q quit"
            }
            Screen::Selection => {
                "F1 login | F3 strategy | F4 dashboard | Tab focus | Up/Down lists | Enter select/search"
            }
            Screen::Strategy => {
                "F1 login | F2 selection | F4 dashboard | Up/Down focus | Left/Right edit HMA | Lua editor supported"
            }
            Screen::Dashboard => {
                "F1 login | F2 selection | F3 strategy | native HMA auto-runs on closed bars | b/s/c manual | q quit"
            }
        };
        let titles = ["Login", "Selection", "Strategy", "Dashboard"]
            .into_iter()
            .map(Line::from)
            .collect::<Vec<_>>();
        let selected_tab = match self.screen {
            Screen::Login => 0,
            Screen::Selection => 1,
            Screen::Strategy => 2,
            Screen::Dashboard => 3,
        };
        let tabs = Tabs::new(titles)
            .select(selected_tab)
            .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan))
            .divider("|");
        frame.render_widget(tabs, rows[0]);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "NinjaTrader / Tradovate TUI",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(format!("Screen: {screen_label}")),
                Span::raw("  "),
                Span::raw(format!("Env: {}", self.form.env.label())),
                Span::raw("  "),
                Span::raw(format!("Auth: {}", self.form.auth_mode.label())),
            ]),
            Line::from(vec![
                Span::raw("Focus: "),
                Span::styled(
                    format!("{:?}", self.focus),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw("  "),
                Span::raw(&self.status),
                Span::raw("  "),
                Span::raw(self.latency_summary()),
                Span::raw("  "),
                Span::styled(help, Style::default().fg(Color::DarkGray)),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("Session"));
        frame.render_widget(header, rows[1]);
    }

    fn render_login_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(area);

        let login = Paragraph::new(self.connection_lines())
            .block(Block::default().borders(Borders::ALL).title("Login"))
            .wrap(Wrap { trim: false });
        frame.render_widget(login, columns[0]);

        let side = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(14), Constraint::Min(10)])
            .split(columns[1]);

        let notes = Paragraph::new(self.login_notes_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Auth Priority"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(notes, side[0]);

        let status = Paragraph::new(self.login_status_lines())
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        frame.render_widget(status, side[1]);
    }

    fn render_strategy_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(18), Constraint::Min(10)])
            .split(columns[0]);

        let setup = Paragraph::new(self.strategy_setup_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Strategy Setup"),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(setup, left[0]);

        let notes = Paragraph::new(self.strategy_notes_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Execution Priority"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(notes, left[1]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(16), Constraint::Length(10)])
            .split(columns[1]);

        let editor = Paragraph::new(self.strategy_detail_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(self.strategy_detail_title()),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(editor, right[0]);

        let preview = Paragraph::new(self.strategy_preview_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selected Strategy"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(preview, right[1]);
    }

    fn render_selection_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(8), Constraint::Min(10)])
            .split(columns[0]);

        let session = Paragraph::new(self.selection_summary_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selection Summary"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(session, left[0]);

        let account_items = if self.accounts.is_empty() {
            vec![ListItem::new(Line::from("No accounts loaded"))]
        } else {
            self.accounts
                .iter()
                .enumerate()
                .map(|(idx, account)| {
                    let mut label = account.name.clone();
                    if let Some(snapshot) = self.snapshot_for_account(account.id) {
                        if let Some(net_liq) = snapshot.net_liq.or(snapshot.balance) {
                            label.push_str(&format!("  |  {}", format_money(Some(net_liq))));
                        }
                    }
                    ListItem::new(styled_line(
                        label,
                        self.focus == Focus::AccountList && idx == self.selected_account,
                    ))
                })
                .collect()
        };
        let accounts = List::new(account_items)
            .block(Block::default().borders(Borders::ALL).title("Accounts"));
        frame.render_widget(accounts, left[1]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(10),
                Constraint::Length(8),
            ])
            .split(columns[1]);

        let search = Paragraph::new(vec![
            styled_line(
                format!("Query: {}", self.instrument_query),
                self.focus == Focus::InstrumentQuery,
            ),
            styled_line(
                format!("Bar Type: {}", self.bar_type.label()),
                self.focus == Focus::BarTypeToggle,
            ),
            Line::from(
                "Enter to search / toggle bar type. Enter on a result subscribes.",
            ),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Instrument Search"),
        )
        .wrap(Wrap { trim: true });
        frame.render_widget(search, right[0]);

        let results = if self.contract_results.is_empty() {
            vec![ListItem::new(Line::from("No contract results"))]
        } else {
            self.contract_results
                .iter()
                .enumerate()
                .map(|(idx, contract)| {
                    let text = format!("{}  |  {}", contract.name, contract.description);
                    ListItem::new(styled_line(
                        text,
                        self.focus == Focus::ContractList && idx == self.selected_contract,
                    ))
                })
                .collect()
        };
        let contract_list =
            List::new(results).block(Block::default().borders(Borders::ALL).title("Contracts"));
        frame.render_widget(contract_list, right[1]);

        let preview = Paragraph::new(self.selection_preview_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Current Selection"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(preview, right[2]);
    }

    fn render_dashboard(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(28),
                Constraint::Percentage(42),
                Constraint::Percentage(30),
            ])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(11),
                Constraint::Length(12),
            ])
            .split(columns[0]);

        let session = Paragraph::new(self.session_summary_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Session Summary"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(session, left[0]);

        let stats = Paragraph::new(self.stats_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Account Stats"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(stats, left[1]);

        self.render_chart(frame, columns[1]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Min(10)])
            .split(columns[2]);

        let preview = Paragraph::new(self.selection_preview_lines())
            .block(Block::default().borders(Borders::ALL).title("Selection"))
            .wrap(Wrap { trim: true });
        frame.render_widget(preview, right[0]);

        let debug = Paragraph::new(self.debug_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selected Account Raw View"),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(debug, right[1]);
    }

    fn render_chart(&self, frame: &mut Frame<'_>, area: Rect) {
        if self.market.bars.is_empty() {
            let empty = Paragraph::new(vec![
                Line::from(self.market.status.clone()),
                Line::from("Select a contract to load history + live bars."),
            ])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!("{} Market Data", self.bar_type.label())),
            );
            frame.render_widget(empty, area);
            return;
        }

        let bars = self
            .market
            .bars
            .iter()
            .rev()
            .take(180)
            .cloned()
            .collect::<Vec<_>>();
        let mut bars = bars.into_iter().rev().collect::<Vec<_>>();
        if bars.is_empty() {
            bars = self.market.bars.clone();
        }

        let points = bars
            .iter()
            .enumerate()
            .map(|(idx, bar)| (idx as f64, bar.close))
            .collect::<Vec<_>>();
        let selected_snapshot = self.selected_snapshot();
        let selected_account_id = selected_snapshot.map(|snapshot| snapshot.account_id);
        let entry_price = selected_snapshot
            .filter(|snapshot| snapshot.market_position_qty.unwrap_or_default().abs() > f64::EPSILON)
            .and_then(|snapshot| snapshot.market_entry_price)
            .filter(|price| price.is_finite());
        let take_profit_price = selected_snapshot
            .and_then(|snapshot| snapshot.selected_contract_take_profit_price)
            .filter(|price| price.is_finite());
        let stop_price = selected_snapshot
            .and_then(|snapshot| snapshot.selected_contract_stop_price)
            .filter(|price| price.is_finite());
        let mut chart_prices = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        let bar_label = self.bar_type.label();
        let mut title = match &self.market.contract_name {
            Some(name) => format!(
                "{bar_label} Market Data [{}] hist={} live={}",
                name, self.market.history_loaded, self.market.live_bars
            ),
            None => format!("{bar_label} Market Data"),
        };
        let mut overlay_labels = Vec::new();
        if let Some(price) = entry_price {
            overlay_labels.push(format!("EP {price:.2}"));
            chart_prices.push(price);
        }
        if let Some(price) = take_profit_price {
            overlay_labels.push(format!("TP {price:.2}"));
            chart_prices.push(price);
        }
        if let Some(price) = stop_price {
            overlay_labels.push(format!("SL {price:.2}"));
            chart_prices.push(price);
        }
        if !overlay_labels.is_empty() {
            title.push_str(" | ");
            title.push_str(&overlay_labels.join(" "));
        }
        let first_ts = bars.first().map(|bar| bar.ts_ns).unwrap_or_default();
        let last_ts = bars.last().map(|bar| bar.ts_ns).unwrap_or_default();
        let mut buy_marker_points = Vec::new();
        let mut sell_marker_points = Vec::new();
        for marker in &self.market.trade_markers {
            if marker.ts_ns < first_ts || marker.ts_ns > last_ts {
                continue;
            }
            if !trade_marker_matches_selection(marker, selected_account_id, &self.market) {
                continue;
            }
            let Some((idx, _)) = bars
                .iter()
                .enumerate()
                .min_by_key(|(_, bar)| bar.ts_ns.abs_diff(marker.ts_ns))
            else {
                continue;
            };
            chart_prices.push(marker.price);
            let point = (idx as f64, marker.price);
            match marker.side {
                TradeMarkerSide::Buy => buy_marker_points.push(point),
                TradeMarkerSide::Sell => sell_marker_points.push(point),
            }
        }
        let (min_close, max_close) = chart_prices.iter().copied().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_v, max_v), price| (min_v.min(price), max_v.max(price)),
        );
        let y_bounds = if min_close.is_finite() && max_close.is_finite() && min_close < max_close {
            let padding = ((max_close - min_close).abs() * 0.05)
                .max(self.market.tick_size.unwrap_or(0.25));
            [min_close - padding, max_close + padding]
        } else if min_close.is_finite() {
            [min_close - 1.0, min_close + 1.0]
        } else {
            [0.0, 1.0]
        };
        let mut segment_points = Vec::with_capacity(points.len().saturating_sub(1));
        let mut segment_colors = Vec::with_capacity(points.len().saturating_sub(1));
        for window in points.windows(2) {
            let start = window[0];
            let end = window[1];
            segment_points.push(vec![start, end]);
            segment_colors.push(if end.1 >= start.1 {
                Color::Green
            } else {
                Color::Red
            });
        }

        let mut datasets = segment_points
            .iter()
            .zip(segment_colors.iter())
            .map(|(segment, color)| {
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(*color))
                    .data(segment.as_slice())
            })
            .collect::<Vec<_>>();

        let last_point_data = points.last().copied().map(|point| vec![point]);
        let line_end = points.len().max(2) as f64 - 1.0;
        let entry_line = entry_price.map(|price| vec![(0.0, price), (line_end, price)]);
        let take_profit_line =
            take_profit_price.map(|price| vec![(0.0, price), (line_end, price)]);
        let stop_line = stop_price.map(|price| vec![(0.0, price), (line_end, price)]);
        if let Some(last_point) = points.last().copied() {
            let last_point_color = if points.len() >= 2 && last_point.1 < points[points.len() - 2].1
            {
                Color::Red
            } else {
                Color::Green
            };
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(last_point_color))
                    .data(last_point_data.as_deref().unwrap_or(&[])),
            );
        }
        if let Some(line) = entry_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(line),
            );
        }
        if let Some(line) = take_profit_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Green))
                    .data(line),
            );
        }
        if let Some(line) = stop_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Red))
                    .data(line),
            );
        }
        if !buy_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&buy_marker_points),
            );
        }
        if !sell_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Block)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Magenta))
                    .data(&sell_marker_points),
            );
        }
        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_axis(Axis::default().bounds([0.0, points.len().max(1) as f64]))
            .y_axis(Axis::default().bounds(y_bounds));
        frame.render_widget(chart, area);
    }

    fn render_logs(&self, frame: &mut Frame<'_>, area: Rect) {
        let lines = self
            .logs
            .iter()
            .rev()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(Line::from)
            .collect::<Vec<_>>();
        let logs = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Log"))
            .wrap(Wrap { trim: true });
        frame.render_widget(logs, area);
    }

    fn connection_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            styled_line(
                format!("Env: {}", self.form.env.label()),
                self.focus == Focus::Env,
            ),
            styled_line(
                format!("Auth Mode: {}", self.form.auth_mode.label()),
                self.focus == Focus::AuthMode,
            ),
            Line::from(""),
            Line::from("Token Sources"),
            styled_line(
                format!("Token Path: {}", self.form.token_path),
                self.focus == Focus::TokenPath,
            ),
            styled_line(
                format!(
                    "Token Override: {}",
                    display_token_override(
                        self.focus == Focus::TokenOverride,
                        &self.form.token_override,
                    )
                ),
                self.focus == Focus::TokenOverride,
            ),
            Line::from(""),
            Line::from("Credential Fallback"),
            styled_line(
                format!("Username: {}", self.form.username),
                self.focus == Focus::Username,
            ),
            styled_line(
                format!("Password: {}", mask(&self.form.password)),
                self.focus == Focus::Password,
            ),
            styled_line(
                format!("App ID: {}", self.form.app_id),
                self.focus == Focus::AppId,
            ),
            styled_line(
                format!("App Version: {}", self.form.app_version),
                self.focus == Focus::AppVersion,
            ),
            styled_line(format!("CID: {}", self.form.cid), self.focus == Focus::Cid),
            styled_line(
                format!("Secret: {}", mask(&self.form.secret)),
                self.focus == Focus::Secret,
            ),
            Line::from(""),
        ];
        lines.push(styled_line(
            "[Enter] Connect / Refresh Session".to_string(),
            self.focus == Focus::Connect,
        ));
        lines
    }

    fn login_notes_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from("1. Token Override is used first when non-empty."),
            Line::from("2. Token File mode reads token_path, then session cache."),
            Line::from("3. Credentials mode requests a fresh access token."),
            Line::from(""),
            Line::from("Use Up/Down to move between fields."),
            Line::from("Use Left/Right on Env or Auth Mode."),
            Line::from("Paste a token directly into Token Override when needed."),
        ]
    }

    fn login_status_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Current status: {}", self.status)),
            Line::from(format!("Environment REST: {}", self.form.env.rest_url())),
            Line::from(format!("User WebSocket: {}", self.form.env.user_ws_url())),
            Line::from(format!(
                "Market WebSocket: {}",
                self.form.env.market_ws_url()
            )),
            Line::from(""),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Last contract: {name}"),
                None => "Last contract: none".to_string(),
            }),
        ]
    }

    fn session_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Env: {}", self.form.env.label())),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!(
                "Strategy Status: {}",
                self.strategy_runtime_summary()
            )),
            Line::from(format!(
                "REST RTT: {}",
                format_latency_ms(self.latency.rest_rtt_ms)
            )),
            Line::from(format!(
                "Order Submit RTT: {}",
                format_latency_ms(self.latency.last_order_ack_ms)
            )),
            Line::from(format!(
                "Order Seen: {}",
                format_latency_ms(self.latency.last_order_seen_ms)
            )),
            Line::from(format!(
                "Exec Ack: {}",
                format_latency_ms(self.latency.last_exec_report_ms)
            )),
            Line::from(format!(
                "First Fill: {}",
                format_latency_ms(self.latency.last_fill_ms)
            )),
            Line::from(format!(
                "Market Update Age: {}",
                format_age_ms(self.market_update_age_ms())
            )),
            Line::from(format!("Auth Mode: {}", self.form.auth_mode.label())),
            Line::from(format!(
                "Token Override: {}",
                if self.form.token_override.trim().is_empty() {
                    "no"
                } else {
                    "yes"
                }
            )),
            Line::from(format!("Accounts: {}", self.accounts.len())),
            Line::from(format!(
                "Selected Contract: {}",
                self.market
                    .contract_name
                    .clone()
                    .unwrap_or_else(|| "none".to_string())
            )),
            Line::from(format!("Session Gate: {}", self.session_gate_summary())),
        ]
    }

    fn strategy_setup_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![styled_line(
            format!("Strategy Type: {}", self.strategy.kind.label()),
            self.focus == Focus::StrategyKind,
        )];

        lines.push(styled_line(
            format!(
                "Order Qty: {}",
                self.strategy_numeric_value(
                    Focus::OrderQty,
                    self.strategy.order_qty.to_string(),
                )
            ),
            self.focus == Focus::OrderQty,
        ));

        if self.strategy.kind == StrategyKind::Native {
            lines.push(styled_line(
                format!("Native Strategy: {}", self.strategy.native_strategy.label()),
                self.focus == Focus::NativeStrategy,
            ));
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    lines.push(styled_line(
                        format!(
                            "HMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::HmaLength,
                                self.strategy.native_hma.hma_length.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Min Angle: {}",
                            self.strategy_numeric_value(
                                Focus::HmaMinAngle,
                                format!("{:.1}", self.strategy.native_hma.min_angle),
                            )
                        ),
                        self.focus == Focus::HmaMinAngle,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Angle Lookback: {}",
                            self.strategy_numeric_value(
                                Focus::HmaAngleLookback,
                                self.strategy.native_hma.angle_lookback.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaAngleLookback,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Bars Required: {}",
                            self.strategy_numeric_value(
                                Focus::HmaBarsRequired,
                                self.strategy.native_hma.bars_required_to_trade.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaBarsRequired,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Longs Only: {}",
                            bool_label(self.strategy.native_hma.longs_only)
                        ),
                        self.focus == Focus::HmaLongsOnly,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Inverted: {}",
                            bool_label(self.strategy.native_hma.inverted)
                        ),
                        self.focus == Focus::HmaInverted,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Take Profit Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTakeProfitTicks,
                                format!("{:.0}", self.strategy.native_hma.take_profit_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaStopLossTicks,
                                format!("{:.0}", self.strategy.native_hma.stop_loss_ticks),
                            )
                        ),
                        self.focus == Focus::HmaStopLossTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trailing Stop: {}",
                            bool_label(self.strategy.native_hma.use_trailing_stop)
                        ),
                        self.focus == Focus::HmaTrailingStop,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Trigger Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTrailTriggerTicks,
                                format!("{:.0}", self.strategy.native_hma.trail_trigger_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTrailOffsetTicks,
                                format!("{:.0}", self.strategy.native_hma.trail_offset_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTrailOffsetTicks,
                    ));
                }
                NativeStrategyKind::EmaCross => {
                    lines.push(styled_line(
                        format!(
                            "Fast EMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaFastLength,
                                self.strategy.native_ema.fast_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaFastLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Slow EMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaSlowLength,
                                self.strategy.native_ema.slow_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaSlowLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Inverted: {}",
                            bool_label(self.strategy.native_ema.inverted)
                        ),
                        self.focus == Focus::EmaInverted,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Take Profit Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTakeProfitTicks,
                                format!("{:.0}", self.strategy.native_ema.take_profit_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaStopLossTicks,
                                format!("{:.0}", self.strategy.native_ema.stop_loss_ticks),
                            )
                        ),
                        self.focus == Focus::EmaStopLossTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trailing Stop: {}",
                            bool_label(self.strategy.native_ema.use_trailing_stop)
                        ),
                        self.focus == Focus::EmaTrailingStop,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Trigger Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTrailTriggerTicks,
                                format!("{:.0}", self.strategy.native_ema.trail_trigger_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTrailOffsetTicks,
                                format!("{:.0}", self.strategy.native_ema.trail_offset_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTrailOffsetTicks,
                    ));
                }
            }
            lines.push(Line::from(
                "Type numbers or use Left/Right for numeric fields. Backspace edits typed values. Zero TP/SL disables them.",
            ));
        } else if self.strategy.kind == StrategyKind::Lua {
            lines.push(styled_line(
                format!("Lua Input: {}", self.strategy.lua_source_mode.label()),
                self.focus == Focus::LuaSourceMode,
            ));
            if self.strategy.lua_source_mode == LuaSourceMode::File {
                lines.push(styled_line(
                    format!("Lua File: {}", self.strategy.lua_file_path),
                    self.focus == Focus::LuaFilePath,
                ));
                lines.push(Line::from(
                    "Press Enter on Lua File to load it into the preview/editor.",
                ));
            } else {
                lines.push(styled_line(
                    format!(
                        "Lua Editor: {} mode, {} lines",
                        self.strategy.lua_editor.mode().label(),
                        self.strategy.lua_editor.line_count()
                    ),
                    self.focus == Focus::LuaEditor,
                ));
                lines.push(Line::from(
                    "Normal mode: h j k l move, i insert, a append, o new line, x delete.",
                ));
            }
        }

        lines.push(Line::from(""));
        lines.push(styled_line(
            "[Enter] Continue To Dashboard / Arm Strategy".to_string(),
            self.focus == Focus::StrategyContinue,
        ));
        lines
    }

    fn strategy_notes_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from("Backend order: Native Rust > Lua > Machine Learning."),
            Line::from("Native strategies execute on newly closed 1m bars after you arm them."),
            Line::from("The native engine targets the selected contract position directly."),
            Line::from("TP, SL, and trailing stop are configured in ticks and synced natively."),
            Line::from(format!(
                "Native runtime auto-closes {}m before session close and holds until reopen.",
                AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
            )),
            Line::from("Lua can be loaded from file or typed directly in the TUI."),
            Line::from("ML remains selection-only for now."),
            Line::from(""),
            Line::from("Strategy screen controls:"),
            Line::from("Up/Down moves focus. Left/Right edits native params or toggles fields."),
            Line::from(
                "Enter on Continue arms the selected native strategy from the current closed bar.",
            ),
            Line::from(""),
            Line::from("Lua editor controls:"),
            Line::from("Normal: h/j/k/l move, i insert, a append, o open line, x delete."),
            Line::from("Insert: type text, Enter newline, Backspace delete, Esc back to normal."),
        ]
    }

    fn strategy_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Selected: {}", self.strategy.summary_label())),
            Line::from(match self.strategy.kind {
                StrategyKind::Native => format!(
                    "Native Rust {} is active and can submit automated market orders.",
                    self.strategy.native_strategy.label()
                ),
                StrategyKind::Lua => {
                    "Lua strategy source is ready for later execution wiring.".to_string()
                }
                StrategyKind::MachineLearning => {
                    "Machine Learning remains lowest execution priority.".to_string()
                }
            }),
            Line::from(format!("Runtime: {}", self.strategy_runtime_summary())),
        ];
        if self.strategy.kind == StrategyKind::Native {
            lines.push(Line::from(self.strategy.native_summary()));
        } else if self.strategy.kind == StrategyKind::Lua {
            lines.push(Line::from(format!(
                "Lua editor mode: {}",
                self.strategy.lua_editor.mode().label()
            )));
            lines.push(Line::from(format!(
                "Lua text size: {} chars",
                self.strategy.lua_editor.text().len()
            )));
        }
        lines
    }

    fn strategy_detail_title(&self) -> String {
        if self.strategy.kind == StrategyKind::Native {
            return "Native Strategy Detail".to_string();
        }
        if self.strategy.kind != StrategyKind::Lua {
            return "Strategy Detail".to_string();
        }
        match self.strategy.lua_source_mode {
            LuaSourceMode::File => "Lua File Preview".to_string(),
            LuaSourceMode::Editor => format!(
                "Lua Editor [{}] row={} col={}",
                self.strategy.lua_editor.mode().label(),
                self.strategy.lua_editor.cursor().0 + 1,
                self.strategy.lua_editor.cursor().1 + 1
            ),
        }
    }

    fn strategy_detail_lines(&self) -> Vec<Line<'static>> {
        if self.strategy.kind == StrategyKind::Native {
            let mut lines = match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => vec![
                    Line::from("HMA Angle Strategy"),
                    Line::from(format!("Type: {}", NativeStrategyKind::HmaAngle.label())),
                    Line::from(format!(
                        "Params: len={} min_angle={:.1} lookback={} bars_required={}",
                        self.strategy.native_hma.hma_length,
                        self.strategy.native_hma.min_angle,
                        self.strategy.native_hma.angle_lookback,
                        self.strategy.native_hma.bars_required_to_trade
                    )),
                    Line::from(format!(
                        "Flags: longs_only={} inverted={} trailing={}",
                        bool_label(self.strategy.native_hma.longs_only),
                        bool_label(self.strategy.native_hma.inverted),
                        bool_label(self.strategy.native_hma.use_trailing_stop)
                    )),
                    Line::from(format!(
                        "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                        self.strategy.native_hma.take_profit_ticks,
                        self.strategy.native_hma.stop_loss_ticks,
                        self.strategy.native_hma.trail_trigger_ticks,
                        self.strategy.native_hma.trail_offset_ticks,
                    )),
                    Line::from(""),
                    Line::from("Signal logic"),
                    Line::from(
                        "Buy: price crosses above zero-lag HMA with sufficient positive angle.",
                    ),
                    Line::from(
                        "Sell: price crosses below zero-lag HMA with sufficient negative angle.",
                    ),
                    Line::from("Inverted swaps buy/sell decisions before order routing."),
                    Line::from(format!(
                        "Auto-close holds flat {}m before the inferred session close.",
                        AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
                    )),
                ],
                NativeStrategyKind::EmaCross => vec![
                    Line::from("EMA Crossover Strategy"),
                    Line::from(format!("Type: {}", NativeStrategyKind::EmaCross.label())),
                    Line::from(format!(
                        "Params: fast={} slow={}",
                        self.strategy.native_ema.fast_length, self.strategy.native_ema.slow_length,
                    )),
                    Line::from(format!(
                        "Flags: inverted={} trailing={}",
                        bool_label(self.strategy.native_ema.inverted),
                        bool_label(self.strategy.native_ema.use_trailing_stop)
                    )),
                    Line::from(format!(
                        "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                        self.strategy.native_ema.take_profit_ticks,
                        self.strategy.native_ema.stop_loss_ticks,
                        self.strategy.native_ema.trail_trigger_ticks,
                        self.strategy.native_ema.trail_offset_ticks,
                    )),
                    Line::from(""),
                    Line::from("Signal logic"),
                    Line::from("Buy: fast EMA crosses above slow EMA."),
                    Line::from("Sell: fast EMA crosses below slow EMA."),
                    Line::from("Inverted swaps buy/sell decisions before order routing."),
                    Line::from(format!(
                        "Auto-close holds flat {}m before the inferred session close.",
                        AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
                    )),
                ],
            };
            lines.extend([
                Line::from(
                    "TP/SL are broker-native and keyed from the confirmed broker entry price.",
                ),
                Line::from("Trailing updates move the broker stop only on new closed bars."),
                Line::from(""),
                Line::from(format!("Live status: {}", self.strategy_runtime_summary())),
                Line::from(format!(
                    "Selected contract qty: {}",
                    self.effective_market_position_qty()
                )),
                Line::from(format!(
                    "Selected contract entry: {}",
                    format_money(
                        self.selected_snapshot()
                            .and_then(|snapshot| snapshot.market_entry_price)
                    )
                )),
                Line::from(format!(
                    "Tick Size: {}",
                    format_money(self.market.tick_size)
                )),
            ]);
            return lines;
        }
        if self.strategy.kind != StrategyKind::Lua {
            return vec![
                Line::from("Native and Lua strategy details appear here."),
                Line::from("Pick a strategy type on the left to configure it."),
            ];
        }

        let focused = self.focus == Focus::LuaEditor;
        let window_start = self.strategy.lua_editor.window_start(22);
        let cursor_row = self.strategy.lua_editor.cursor().0;
        let lines = self.strategy.lua_editor.visible_lines(22);
        lines
            .into_iter()
            .enumerate()
            .map(|(idx, line)| {
                let line_no = window_start + idx + 1;
                let style = if focused && window_start + idx == cursor_row {
                    Style::default().fg(Color::Black).bg(Color::Cyan)
                } else {
                    Style::default()
                };
                Line::from(Span::styled(format!("{:>3} {}", line_no, line), style))
            })
            .collect()
    }

    fn selection_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!(
                "Strategy Runtime: {}",
                self.strategy_runtime_summary()
            )),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(format!("Contract results: {}", self.contract_results.len())),
            Line::from(format!(
                "Last subscribed contract: {}",
                self.market
                    .contract_name
                    .clone()
                    .unwrap_or_else(|| "none".to_string())
            )),
            Line::from("F3 opens the monitoring dashboard."),
        ]
    }

    fn selection_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Status: {}", self.status)),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!("Query: {}", self.instrument_query)),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(match self.contract_results.get(self.selected_contract) {
                Some(contract) => format!("Selected contract: {}", contract.name),
                None => "Selected contract: none".to_string(),
            }),
        ];
        if let Some(snapshot) = self.selected_snapshot() {
            lines.push(Line::from(format!(
                "Account net liq: {}",
                format_money(snapshot.net_liq.or(snapshot.balance))
            )));
            lines.push(Line::from(format!(
                "Selected contract qty: {}",
                format_quantity(snapshot.market_position_qty)
            )));
        }
        lines
    }

    fn stats_lines(&self) -> Vec<Line<'static>> {
        let Some(snapshot) = self.selected_snapshot() else {
            return vec![
                Line::from("No selected account stats."),
                Line::from("Connect and wait for account sync."),
            ];
        };
        vec![
            Line::from(format!("Account: {}", snapshot.account_name)),
            Line::from(format!("Balance: {}", format_money(snapshot.balance))),
            Line::from(format!("Cash: {}", format_money(snapshot.cash_balance))),
            Line::from(format!("NetLiq: {}", format_money(snapshot.net_liq))),
            pnl_line("Session Realized PnL", snapshot.realized_pnl),
            pnl_line("Unrealized PnL", snapshot.unrealized_pnl),
            Line::from(format!(
                "Intraday Margin: {}",
                format_money(snapshot.intraday_margin)
            )),
            Line::from(format!(
                "Open Position Qty: {}",
                format_quantity(snapshot.open_position_qty)
            )),
            Line::from(format!(
                "Selected Contract Qty: {}",
                format_quantity(snapshot.market_position_qty)
            )),
            Line::from(format!(
                "Entry Price: {}",
                format_money(snapshot.market_entry_price)
            )),
            Line::from(format!(
                "Take Profit: {}",
                format_money(snapshot.selected_contract_take_profit_price)
            )),
            Line::from(format!(
                "Stop Loss: {}",
                format_money(snapshot.selected_contract_stop_price)
            )),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Active Contract: {name}"),
                None => "Active Contract: none".to_string(),
            }),
            Line::from(format!(
                "Order Qty: {}  TIF: {}",
                self.base_config.order_qty, self.base_config.time_in_force
            )),
            Line::from("Hotkeys: b buy | s sell | c close"),
        ]
    }

    fn debug_lines(&self) -> Vec<Line<'static>> {
        let Some(snapshot) = self.selected_snapshot() else {
            return vec![Line::from("No raw payload yet.")];
        };
        let raw = json_preview(snapshot);
        raw.lines()
            .take(20)
            .map(|line| Line::from(line.to_string()))
            .collect()
    }

    fn selected_snapshot(&self) -> Option<&AccountSnapshot> {
        let account = self.accounts.get(self.selected_account)?;
        self.snapshot_for_account(account.id)
    }

    fn snapshot_for_account(&self, account_id: i64) -> Option<&AccountSnapshot> {
        self.account_snapshots
            .iter()
            .find(|snapshot| snapshot.account_id == account_id)
    }

    fn login_focus_order(&self) -> Vec<Focus> {
        vec![
            Focus::Env,
            Focus::AuthMode,
            Focus::TokenPath,
            Focus::TokenOverride,
            Focus::Username,
            Focus::Password,
            Focus::AppId,
            Focus::AppVersion,
            Focus::Cid,
            Focus::Secret,
            Focus::Connect,
        ]
    }

    fn strategy_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::StrategyKind, Focus::OrderQty];
        if self.strategy.kind == StrategyKind::Native {
            order.push(Focus::NativeStrategy);
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => order.extend([
                    Focus::HmaLength,
                    Focus::HmaMinAngle,
                    Focus::HmaAngleLookback,
                    Focus::HmaBarsRequired,
                    Focus::HmaLongsOnly,
                    Focus::HmaInverted,
                    Focus::HmaTakeProfitTicks,
                    Focus::HmaStopLossTicks,
                    Focus::HmaTrailingStop,
                    Focus::HmaTrailTriggerTicks,
                    Focus::HmaTrailOffsetTicks,
                ]),
                NativeStrategyKind::EmaCross => order.extend([
                    Focus::EmaFastLength,
                    Focus::EmaSlowLength,
                    Focus::EmaInverted,
                    Focus::EmaTakeProfitTicks,
                    Focus::EmaStopLossTicks,
                    Focus::EmaTrailingStop,
                    Focus::EmaTrailTriggerTicks,
                    Focus::EmaTrailOffsetTicks,
                ]),
            }
        } else if self.strategy.kind == StrategyKind::Lua {
            order.push(Focus::LuaSourceMode);
            match self.strategy.lua_source_mode {
                LuaSourceMode::File => order.push(Focus::LuaFilePath),
                LuaSourceMode::Editor => order.push(Focus::LuaEditor),
            }
        }
        order.push(Focus::StrategyContinue);
        order
    }

    fn selection_focus_order(&self) -> Vec<Focus> {
        vec![
            Focus::AccountList,
            Focus::InstrumentQuery,
            Focus::ContractList,
        ]
    }

    fn next_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn next_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn next_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn is_text_focus(&self) -> bool {
        matches!(
            self.focus,
            Focus::TokenOverride
                | Focus::Username
                | Focus::Password
                | Focus::AppId
                | Focus::AppVersion
                | Focus::Cid
                | Focus::Secret
                | Focus::TokenPath
                | Focus::OrderQty
                | Focus::HmaLength
                | Focus::HmaMinAngle
                | Focus::HmaAngleLookback
                | Focus::HmaBarsRequired
                | Focus::HmaTakeProfitTicks
                | Focus::HmaStopLossTicks
                | Focus::HmaTrailTriggerTicks
                | Focus::HmaTrailOffsetTicks
                | Focus::EmaFastLength
                | Focus::EmaSlowLength
                | Focus::EmaTakeProfitTicks
                | Focus::EmaStopLossTicks
                | Focus::EmaTrailTriggerTicks
                | Focus::EmaTrailOffsetTicks
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::InstrumentQuery
        )
    }

    fn push_log(&mut self, message: String) {
        while self.logs.len() >= 200 {
            self.logs.pop_front();
        }
        self.logs.push_back(message);
    }

    fn clear_strategy_numeric_input(&mut self) {
        self.strategy_numeric_input = None;
    }

    fn strategy_numeric_value(&self, focus: Focus, fallback: String) -> String {
        self.strategy_numeric_input
            .as_ref()
            .filter(|draft| draft.focus == focus)
            .map(|draft| draft.value.clone())
            .unwrap_or(fallback)
    }

    fn market_update_age_ms(&self) -> Option<u64> {
        self.last_market_update_at
            .map(|instant| instant.elapsed().as_millis() as u64)
    }

    fn latency_summary(&self) -> String {
        format!(
            "REST {} | Submit {} | Seen {} | Ack {} | Fill {} | Market {}",
            format_latency_ms(self.latency.rest_rtt_ms),
            format_latency_ms(self.latency.last_order_ack_ms),
            format_latency_ms(self.latency.last_order_seen_ms),
            format_latency_ms(self.latency.last_exec_report_ms),
            format_latency_ms(self.latency.last_fill_ms),
            format_age_ms(self.market_update_age_ms()),
        )
    }

    fn sync_execution_strategy_config(&self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let _ = cmd_tx.send(ServiceCommand::SetExecutionStrategyConfig(
            self.strategy.execution_config(),
        ));
    }

    fn disarm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::DisarmExecutionStrategy {
            reason: "Native strategy config changed; press Continue to re-arm.".to_string(),
        });
    }

    fn arm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::ArmExecutionStrategy);
    }

    fn closed_bars(&self) -> &[crate::tradovate::Bar] {
        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        &self.market.bars[..closed_len]
    }

    fn latest_closed_bar_ts(&self) -> Option<i64> {
        self.closed_bars().last().map(|bar| bar.ts_ns)
    }

    fn session_window_at(&self, ts_ns: i64) -> Option<InstrumentSessionWindow> {
        self.market
            .session_profile
            .map(|profile| profile.evaluate(ts_ns))
    }

    fn latest_session_window(&self) -> Option<InstrumentSessionWindow> {
        self.latest_closed_bar_ts()
            .and_then(|ts_ns| self.session_window_at(ts_ns))
    }

    fn session_gate_summary(&self) -> String {
        let Some(profile) = self.market.session_profile else {
            return "n/a".to_string();
        };
        let Some(window) = self.latest_session_window() else {
            return format!("{} awaiting bars", profile.label());
        };

        if !window.session_open {
            return format!("{} closed; holding until reopen", profile.label());
        }

        let minutes_to_close = window
            .minutes_to_close
            .map(|minutes| format!("{minutes:.0}m"))
            .unwrap_or_else(|| "n/a".to_string());
        if window.hold_entries {
            format!(
                "{} hold active; flattening before close ({} left)",
                profile.label(),
                minutes_to_close
            )
        } else {
            format!("{} open; {} to close", profile.label(), minutes_to_close)
        }
    }

    fn strategy_runtime_summary(&self) -> String {
        if !self.strategy_runtime.last_summary.is_empty() {
            return self.strategy_runtime.last_summary.clone();
        }
        if self.strategy_runtime.armed {
            "Armed".to_string()
        } else {
            "Disarmed".to_string()
        }
    }

    fn effective_market_position_qty(&self) -> i32 {
        self.strategy_runtime.pending_target_qty.unwrap_or_else(|| {
            self.selected_snapshot()
                .and_then(|snapshot| snapshot.market_position_qty)
                .unwrap_or(0.0)
                .round() as i32
        })
    }

}

impl FormState {
    fn from_config(config: &AppConfig) -> Self {
        Self {
            env: config.env,
            auth_mode: config.auth_mode,
            token_override: config.token_override.clone(),
            username: config.username.clone(),
            password: config.password.clone(),
            app_id: config.app_id.clone(),
            app_version: config.app_version.clone(),
            cid: config.cid.clone(),
            secret: config.secret.clone(),
            token_path: config.token_path.display().to_string(),
        }
    }
}

fn edit_string(target: &mut String, key: KeyEvent) {
    match key.code {
        KeyCode::Backspace => {
            target.pop();
        }
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            target.push(ch);
        }
        _ => {}
    }
}

fn styled_line(text: String, focused: bool) -> Line<'static> {
    if focused {
        Line::from(Span::styled(
            text,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
    } else {
        Line::from(text)
    }
}

fn pnl_line(label: &str, value: Option<f64>) -> Line<'static> {
    Line::from(vec![
        Span::raw(format!("{label}: ")),
        Span::styled(format_signed_money(value), pnl_style(value)),
    ])
}

fn pnl_style(value: Option<f64>) -> Style {
    match value {
        Some(value) if value > 0.0 => Style::default().fg(Color::Green),
        Some(value) if value < 0.0 => Style::default().fg(Color::Red),
        _ => Style::default(),
    }
}

fn format_money(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_signed_money(value: Option<f64>) -> String {
    match value {
        Some(value) if value > 0.0 => format!("+{value:.2}"),
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_quantity(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_latency_ms(value: Option<u64>) -> String {
    match value {
        Some(value) => format!("{value}ms"),
        None => "n/a".to_string(),
    }
}

fn format_age_ms(value: Option<u64>) -> String {
    match value {
        Some(value) if value >= 1_000 => format!("{:.1}s", value as f64 / 1_000.0),
        Some(value) => format!("{value}ms"),
        None => "n/a".to_string(),
    }
}

fn trade_marker_matches_selection(
    marker: &TradeMarker,
    selected_account_id: Option<i64>,
    market: &MarketSnapshot,
) -> bool {
    let account_matches = selected_account_id
        .zip(marker.account_id)
        .map(|(selected, marker_account)| selected == marker_account)
        .unwrap_or(true);
    if !account_matches {
        return false;
    }

    if marker.contract_id.is_none() && marker.contract_name.is_none() {
        return true;
    }

    let contract_id_matches = marker
        .contract_id
        .zip(market.contract_id)
        .map(|(marker_contract_id, market_contract_id)| marker_contract_id == market_contract_id)
        .unwrap_or(false);
    let contract_name_matches = marker
        .contract_name
        .as_deref()
        .zip(market.contract_name.as_deref())
        .map(|(marker_contract_name, market_contract_name)| {
            marker_contract_name.eq_ignore_ascii_case(market_contract_name)
        })
        .unwrap_or(false);

    contract_id_matches || contract_name_matches
}

fn bool_label(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn mask(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        "*".repeat(value.len().min(16))
    }
}

fn display_token_override(focused: bool, value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    if focused || value.len() <= 18 {
        return value.to_string();
    }
    format!("{}...{}", &value[..8], &value[value.len() - 6..])
}

fn toggle_bool(target: &mut bool, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = !*target;
            true
        }
        _ => false,
    }
}

fn adjust_usize(target: &mut usize, key: KeyEvent, min: usize, step: usize) -> bool {
    match key.code {
        KeyCode::Left => {
            *target = target.saturating_sub(step).max(min);
            true
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = target.saturating_add(step).max(min);
            true
        }
        _ => false,
    }
}

fn adjust_float(target: &mut f64, key: KeyEvent, min: f64, step: f64) -> bool {
    match key.code {
        KeyCode::Left => {
            *target = (*target - step).max(min);
            true
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = (*target + step).max(min);
            true
        }
        _ => false,
    }
}

fn edit_strategy_usize(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut usize,
    key: KeyEvent,
    min: usize,
    step: usize,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        return adjust_usize(target, key, min, step);
    }

    match key.code {
        KeyCode::Backspace => {
            let next = numeric_backspace(draft, focus, &target.to_string());
            if let Some(value) = next.and_then(|value| value.parse::<usize>().ok()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if ch.is_ascii_digit()
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            let next = numeric_append(draft, focus, ch, false);
            if let Ok(value) = next.parse::<usize>() {
                *target = value.max(min);
                return true;
            }
        }
        _ => {}
    }
    false
}

fn edit_strategy_i32(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut i32,
    key: KeyEvent,
    min: i32,
    step: i32,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        match key.code {
            KeyCode::Left => {
                *target = target.saturating_sub(step).max(min);
                return true;
            }
            KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                *target = target.saturating_add(step).max(min);
                return true;
            }
            _ => return false,
        }
    }

    match key.code {
        KeyCode::Backspace => {
            let next = numeric_backspace(draft, focus, &target.to_string());
            if let Some(value) = next.and_then(|value| value.parse::<i32>().ok()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if ch.is_ascii_digit()
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            let next = numeric_append(draft, focus, ch, false);
            if let Ok(value) = next.parse::<i32>() {
                *target = value.max(min);
                return true;
            }
        }
        _ => {}
    }
    false
}

fn edit_strategy_float(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut f64,
    key: KeyEvent,
    min: f64,
    step: f64,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        return adjust_float(target, key, min, step);
    }

    match key.code {
        KeyCode::Backspace => {
            let current = format_float_input(*target);
            let next = numeric_backspace(draft, focus, &current);
            if let Some(value) = parse_float_input(next.as_deref()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            if ch.is_ascii_digit() || ch == '.' {
                let next = numeric_append(draft, focus, ch, true);
                if let Some(value) = parse_float_input(Some(next.as_str())) {
                    *target = value.max(min);
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}

fn numeric_append(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    ch: char,
    allow_decimal: bool,
) -> String {
    let entry = match draft {
        Some(entry) if entry.focus == focus => entry,
        _ => {
            *draft = Some(NumericInputState {
                focus,
                value: String::new(),
            });
            draft.as_mut().expect("draft just inserted")
        }
    };

    if ch == '.' {
        if !allow_decimal || entry.value.contains('.') {
            return entry.value.clone();
        }
        if entry.value.is_empty() {
            entry.value.push('0');
        }
    }
    entry.value.push(ch);
    entry.value.clone()
}

fn numeric_backspace(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    current: &str,
) -> Option<String> {
    let entry = match draft {
        Some(entry) if entry.focus == focus => entry,
        _ => {
            *draft = Some(NumericInputState {
                focus,
                value: current.to_string(),
            });
            draft.as_mut().expect("draft just inserted")
        }
    };
    entry.value.pop();
    Some(entry.value.clone())
}

fn format_float_input(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let mut text = value.to_string();
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.push('0');
        }
        text
    }
}

fn parse_float_input(value: Option<&str>) -> Option<f64> {
    let raw = value?;
    if raw.is_empty() {
        return None;
    }
    if raw == "." {
        return Some(0.0);
    }
    raw.parse::<f64>().ok()
}

fn json_preview(snapshot: &AccountSnapshot) -> String {
    let raw = serde_json::json!({
        "account": snapshot.raw_account,
        "risk": snapshot.raw_risk,
        "cash": snapshot.raw_cash,
        "positions": snapshot.raw_positions,
    });
    serde_json::to_string_pretty(&raw).unwrap_or_else(|_| "{}".to_string())
}
