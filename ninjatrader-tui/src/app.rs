use crate::automation::{StrategyDescriptor, default_strategy_catalog};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::{PositionSide, StrategySignal, side_from_signed_qty};
use crate::strategy::{LuaSourceMode, NativeStrategyKind, StrategyKind, StrategyState};
use crate::tradovate::{
    AccountInfo, AccountSnapshot, ContractSuggestion, ManualOrderAction, MarketSnapshot,
    ServiceCommand, ServiceEvent, TradeMarkerSide,
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
    contract_results: Vec<ContractSuggestion>,
    selected_contract: usize,
    market: MarketSnapshot,
    logs: VecDeque<String>,
    strategy_catalog: Vec<StrategyDescriptor>,
    strategy_runtime: StrategyRuntimeState,
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
    hma_execution: HmaAngleExecutionState,
    ema_execution: EmaCrossExecutionState,
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
            contract_results: Vec::new(),
            selected_contract: 0,
            market: MarketSnapshot::default(),
            logs: VecDeque::new(),
            strategy_catalog: default_strategy_catalog(),
            strategy_runtime: StrategyRuntimeState::default(),
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
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match event {
            ServiceEvent::Status(message) => {
                self.status = message.clone();
                self.push_log(message);
            }
            ServiceEvent::Error(message) => {
                self.status = format!("Error: {message}");
                if self.strategy_runtime.pending_target_qty.take().is_some() {
                    self.push_log(
                        "Cleared pending automated target after service error.".to_string(),
                    );
                }
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
                if let Some(pending) = self.strategy_runtime.pending_target_qty {
                    let actual = self.actual_market_position_qty();
                    if actual == pending {
                        self.strategy_runtime.pending_target_qty = None;
                        self.strategy_runtime.last_summary =
                            format!("Position confirmed at target {actual}");
                    }
                }
                if self.strategy_runtime.armed && self.strategy.kind == StrategyKind::Native {
                    self.maybe_sync_native_protection(cmd_tx, None);
                }
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
                let contract_changed = self.market.contract_id != snapshot.contract_id;
                self.market = snapshot;
                if contract_changed {
                    self.strategy_runtime.last_closed_bar_ts = self.latest_closed_bar_ts();
                    self.strategy_runtime.pending_target_qty = None;
                    self.reset_native_execution();
                    self.strategy_runtime.last_summary =
                        "Contract changed; strategy re-anchored to current bar.".to_string();
                }
                self.maybe_run_native_strategy(cmd_tx);
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
            Screen::Strategy => self.handle_strategy_key(key),
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
            | Focus::ContractList => {}
        }
    }

    fn handle_strategy_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                self.focus = self.prev_strategy_focus();
                return;
            }
            KeyCode::Down | KeyCode::Tab => {
                self.focus = self.next_strategy_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::StrategyKind => match key.code {
                KeyCode::Left => {
                    self.strategy.kind = self.strategy.kind.prev();
                    self.disarm_native_strategy();
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.kind = self.strategy.kind.next();
                    self.disarm_native_strategy();
                }
                _ => {}
            },
            Focus::NativeStrategy => match key.code {
                KeyCode::Left => {
                    self.strategy.native_strategy = self.strategy.native_strategy.prev();
                    self.disarm_native_strategy();
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_strategy = self.strategy.native_strategy.next();
                    self.disarm_native_strategy();
                }
                _ => {}
            },
            Focus::HmaLength => {
                if adjust_usize(&mut self.strategy.native_hma.hma_length, key, 2, 1) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaMinAngle => {
                if adjust_float(&mut self.strategy.native_hma.min_angle, key, 0.0, 0.5) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaAngleLookback => {
                if adjust_usize(&mut self.strategy.native_hma.angle_lookback, key, 1, 1) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaBarsRequired => {
                if adjust_usize(
                    &mut self.strategy.native_hma.bars_required_to_trade,
                    key,
                    1,
                    1,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaLongsOnly => {
                if toggle_bool(&mut self.strategy.native_hma.longs_only, key) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaInverted => {
                if toggle_bool(&mut self.strategy.native_hma.inverted, key) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaTakeProfitTicks => {
                if adjust_float(
                    &mut self.strategy.native_hma.take_profit_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaStopLossTicks => {
                if adjust_float(&mut self.strategy.native_hma.stop_loss_ticks, key, 0.0, 1.0) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaTrailingStop => {
                if toggle_bool(&mut self.strategy.native_hma.use_trailing_stop, key) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaTrailTriggerTicks => {
                if adjust_float(
                    &mut self.strategy.native_hma.trail_trigger_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::HmaTrailOffsetTicks => {
                if adjust_float(
                    &mut self.strategy.native_hma.trail_offset_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaFastLength => {
                if adjust_usize(&mut self.strategy.native_ema.fast_length, key, 1, 1) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaSlowLength => {
                if adjust_usize(&mut self.strategy.native_ema.slow_length, key, 1, 1) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaInverted => {
                if toggle_bool(&mut self.strategy.native_ema.inverted, key) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaTakeProfitTicks => {
                if adjust_float(
                    &mut self.strategy.native_ema.take_profit_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaStopLossTicks => {
                if adjust_float(&mut self.strategy.native_ema.stop_loss_ticks, key, 0.0, 1.0) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaTrailingStop => {
                if toggle_bool(&mut self.strategy.native_ema.use_trailing_stop, key) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaTrailTriggerTicks => {
                if adjust_float(
                    &mut self.strategy.native_ema.trail_trigger_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::EmaTrailOffsetTicks => {
                if adjust_float(
                    &mut self.strategy.native_ema.trail_offset_ticks,
                    key,
                    0.0,
                    1.0,
                ) {
                    self.disarm_native_strategy();
                }
            }
            Focus::LuaSourceMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.lua_source_mode = self.strategy.lua_source_mode.toggle();
                    self.disarm_native_strategy();
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
                    self.disarm_native_strategy();
                }
            }
            Focus::LuaEditor => {
                let _ = self.strategy.lua_editor.handle_key(key);
                self.disarm_native_strategy();
            }
            Focus::StrategyContinue => {
                if key.code == KeyCode::Enter {
                    self.screen = Screen::Dashboard;
                    self.focus = Focus::AccountList;
                    self.arm_native_strategy();
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
                        let _ = cmd_tx.send(ServiceCommand::SubscribeBars { contract });
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
                        self.focus = Focus::ContractList;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = Focus::AccountList;
                        return;
                    }
                    KeyCode::Right => {
                        self.focus = Focus::ContractList;
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
                Constraint::Length(4),
                Constraint::Min(10),
                Constraint::Length(8),
            ])
            .split(columns[1]);

        let search = Paragraph::new(vec![
            styled_line(
                format!("Query: {}", self.instrument_query),
                self.focus == Focus::InstrumentQuery,
            ),
            Line::from(
                "Enter to search contracts. Enter on a result subscribes and opens Strategy.",
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
                Constraint::Length(8),
                Constraint::Length(12),
                Constraint::Min(8),
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

        let strategy_lines = self
            .strategy_catalog
            .iter()
            .map(|item| {
                Line::from(format!(
                    "{} [{}] {} - {}",
                    item.name, item.priority, item.status, item.note
                ))
            })
            .collect::<Vec<_>>();
        let automation = Paragraph::new(strategy_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Automation Roadmap"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(automation, left[2]);

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
                Line::from("Select a contract to load 1-minute history + live bars."),
            ])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("1m Market Data"),
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
        let (min_close, max_close) = bars
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), bar| {
                (min_v.min(bar.close), max_v.max(bar.close))
            });
        let y_bounds = if min_close.is_finite() && max_close.is_finite() && min_close < max_close {
            [min_close, max_close]
        } else if min_close.is_finite() {
            [min_close - 1.0, min_close + 1.0]
        } else {
            [0.0, 1.0]
        };
        let title = match &self.market.contract_name {
            Some(name) => format!(
                "1m Market Data [{}] hist={} live={}",
                name, self.market.history_loaded, self.market.live_bars
            ),
            None => "1m Market Data".to_string(),
        };
        let marker_offset = ((y_bounds[1] - y_bounds[0]).abs() * 0.02).max(0.25);
        let first_ts = bars.first().map(|bar| bar.ts_ns).unwrap_or_default();
        let last_ts = bars.last().map(|bar| bar.ts_ns).unwrap_or_default();
        let mut buy_marker_points = Vec::new();
        let mut sell_marker_points = Vec::new();
        for marker in &self.market.trade_markers {
            if marker.ts_ns < first_ts || marker.ts_ns > last_ts {
                continue;
            }
            let Some((idx, _)) = bars
                .iter()
                .enumerate()
                .min_by_key(|(_, bar)| bar.ts_ns.abs_diff(marker.ts_ns))
            else {
                continue;
            };
            let point = match marker.side {
                TradeMarkerSide::Buy => {
                    (idx as f64, (marker.price - marker_offset).max(y_bounds[0]))
                }
                TradeMarkerSide::Sell => {
                    (idx as f64, (marker.price + marker_offset).min(y_bounds[1]))
                }
            };
            match marker.side {
                TradeMarkerSide::Buy => buy_marker_points.push(point),
                TradeMarkerSide::Sell => sell_marker_points.push(point),
            }
        }
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
        if !buy_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Block)
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
        ]
    }

    fn strategy_setup_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![styled_line(
            format!("Strategy Type: {}", self.strategy.kind.label()),
            self.focus == Focus::StrategyKind,
        )];

        if self.strategy.kind == StrategyKind::Native {
            lines.push(styled_line(
                format!("Native Strategy: {}", self.strategy.native_strategy.label()),
                self.focus == Focus::NativeStrategy,
            ));
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    lines.push(styled_line(
                        format!("HMA Length: {}", self.strategy.native_hma.hma_length),
                        self.focus == Focus::HmaLength,
                    ));
                    lines.push(styled_line(
                        format!("Min Angle: {:.1}", self.strategy.native_hma.min_angle),
                        self.focus == Focus::HmaMinAngle,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Angle Lookback: {}",
                            self.strategy.native_hma.angle_lookback
                        ),
                        self.focus == Focus::HmaAngleLookback,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Bars Required: {}",
                            self.strategy.native_hma.bars_required_to_trade
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
                            "Take Profit Ticks: {:.0}",
                            self.strategy.native_hma.take_profit_ticks
                        ),
                        self.focus == Focus::HmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {:.0}",
                            self.strategy.native_hma.stop_loss_ticks
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
                            "Trail Trigger Ticks: {:.0}",
                            self.strategy.native_hma.trail_trigger_ticks
                        ),
                        self.focus == Focus::HmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {:.0}",
                            self.strategy.native_hma.trail_offset_ticks
                        ),
                        self.focus == Focus::HmaTrailOffsetTicks,
                    ));
                }
                NativeStrategyKind::EmaCross => {
                    lines.push(styled_line(
                        format!("Fast EMA Length: {}", self.strategy.native_ema.fast_length),
                        self.focus == Focus::EmaFastLength,
                    ));
                    lines.push(styled_line(
                        format!("Slow EMA Length: {}", self.strategy.native_ema.slow_length),
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
                            "Take Profit Ticks: {:.0}",
                            self.strategy.native_ema.take_profit_ticks
                        ),
                        self.focus == Focus::EmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {:.0}",
                            self.strategy.native_ema.stop_loss_ticks
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
                            "Trail Trigger Ticks: {:.0}",
                            self.strategy.native_ema.trail_trigger_ticks
                        ),
                        self.focus == Focus::EmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {:.0}",
                            self.strategy.native_ema.trail_offset_ticks
                        ),
                        self.focus == Focus::EmaTrailOffsetTicks,
                    ));
                }
            }
            lines.push(Line::from(
                "Use Left/Right to change values and toggle booleans. Zero TP/SL disables them.",
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
            Line::from(format!(
                "Unrealized PnL: {}",
                format_money(snapshot.unrealized_pnl)
            )),
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
        let mut order = vec![Focus::StrategyKind];
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

    fn reset_native_execution(&mut self) {
        self.strategy_runtime.hma_execution = HmaAngleExecutionState::default();
        self.strategy_runtime.ema_execution = EmaCrossExecutionState::default();
    }

    fn disarm_native_strategy(&mut self) {
        if self.strategy_runtime.armed {
            self.strategy_runtime.armed = false;
            self.reset_native_execution();
            self.strategy_runtime.last_summary =
                "Native strategy config changed; press Continue to re-arm.".to_string();
        }
    }

    fn arm_native_strategy(&mut self) {
        self.strategy_runtime.pending_target_qty = None;
        if self.strategy.kind != StrategyKind::Native {
            self.strategy_runtime.armed = false;
            self.strategy_runtime.last_closed_bar_ts = None;
            self.strategy_runtime.last_summary =
                "Selected strategy is not an armed native runtime.".to_string();
            return;
        }

        self.strategy_runtime.armed = true;
        self.reset_native_execution();
        self.strategy_runtime.last_closed_bar_ts = self.latest_closed_bar_ts();
        self.strategy_runtime.last_summary = if self.strategy_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current closed bar.",
                self.strategy.native_strategy.label()
            )
        } else {
            format!(
                "Native {} armed; waiting for first closed bar.",
                self.strategy.native_strategy.label()
            )
        };
    }

    fn closed_bars(&self) -> &[crate::tradovate::Bar] {
        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        &self.market.bars[..closed_len]
    }

    fn latest_closed_bar_ts(&self) -> Option<i64> {
        self.closed_bars().last().map(|bar| bar.ts_ns)
    }

    fn actual_market_position_qty(&self) -> i32 {
        self.selected_snapshot()
            .and_then(|snapshot| snapshot.market_position_qty)
            .unwrap_or(0.0)
            .round() as i32
    }

    fn actual_market_entry_price(&self) -> Option<f64> {
        self.selected_snapshot()
            .and_then(|snapshot| snapshot.market_entry_price)
            .filter(|price| price.is_finite() && *price > 0.0)
    }

    fn active_native_slug(&self) -> &'static str {
        self.strategy.native_strategy.slug()
    }

    fn active_native_uses_protection(&self) -> bool {
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self.strategy.native_hma.uses_native_protection(),
            NativeStrategyKind::EmaCross => self.strategy.native_ema.uses_native_protection(),
        }
    }

    fn sync_active_native_position(&mut self, signed_qty: i32, entry_price: Option<f64>) {
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self.strategy.native_hma.sync_position(
                &mut self.strategy_runtime.hma_execution,
                signed_qty,
                entry_price,
            ),
            NativeStrategyKind::EmaCross => self.strategy.native_ema.sync_position(
                &mut self.strategy_runtime.ema_execution,
                signed_qty,
                entry_price,
            ),
        }
    }

    fn take_profit_price(&self, side: PositionSide, entry_price: f64) -> Option<f64> {
        let offset = match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self
                .strategy
                .native_hma
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::EmaCross => self
                .strategy
                .native_ema
                .take_profit_offset(self.market.tick_size)?,
        };
        Some(match side {
            PositionSide::Long => entry_price + offset,
            PositionSide::Short => entry_price - offset,
        })
    }

    fn combined_stop_price(&mut self, trailing_bar: Option<&crate::tradovate::Bar>) -> Option<f64> {
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                if let Some(bar) = trailing_bar {
                    let _ = self.strategy.native_hma.desired_trailing_stop_price(
                        &mut self.strategy_runtime.hma_execution,
                        bar,
                        self.market.tick_size,
                    );
                }
                self.strategy.native_hma.current_effective_stop_price(
                    &self.strategy_runtime.hma_execution,
                    self.market.tick_size,
                )
            }
            NativeStrategyKind::EmaCross => {
                if let Some(bar) = trailing_bar {
                    let _ = self.strategy.native_ema.desired_trailing_stop_price(
                        &mut self.strategy_runtime.ema_execution,
                        bar,
                        self.market.tick_size,
                    );
                }
                self.strategy.native_ema.current_effective_stop_price(
                    &self.strategy_runtime.ema_execution,
                    self.market.tick_size,
                )
            }
        }
    }

    fn maybe_sync_native_protection(
        &mut self,
        cmd_tx: &UnboundedSender<ServiceCommand>,
        trailing_bar: Option<&crate::tradovate::Bar>,
    ) {
        if !self.strategy_runtime.armed || self.strategy.kind != StrategyKind::Native {
            return;
        }
        if !self.active_native_uses_protection() {
            return;
        }

        let signed_qty = self.actual_market_position_qty();
        let actual_market_entry = self.actual_market_entry_price();
        self.sync_active_native_position(signed_qty, actual_market_entry);

        if signed_qty == 0 {
            let _ = cmd_tx.send(ServiceCommand::SyncNativeProtection {
                signed_qty: 0,
                take_profit_price: None,
                stop_price: None,
                reason: format!("{} flat", self.active_native_slug()),
            });
            return;
        }

        let Some(entry_price) = actual_market_entry else {
            return;
        };
        let Some(side) = side_from_signed_qty(signed_qty) else {
            return;
        };

        let take_profit_price = self.take_profit_price(side, entry_price);
        let stop_price = self.combined_stop_price(trailing_bar);
        let _ = cmd_tx.send(ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason: if trailing_bar.is_some() {
                format!("{} bar sync", self.active_native_slug())
            } else {
                format!("{} position sync", self.active_native_slug())
            },
        });
    }

    fn clear_native_protection(
        &self,
        cmd_tx: &UnboundedSender<ServiceCommand>,
        reason: impl Into<String>,
    ) {
        let _ = cmd_tx.send(ServiceCommand::SyncNativeProtection {
            signed_qty: 0,
            take_profit_price: None,
            stop_price: None,
            reason: reason.into(),
        });
    }

    fn effective_market_position_qty(&self) -> i32 {
        self.strategy_runtime
            .pending_target_qty
            .unwrap_or_else(|| self.actual_market_position_qty())
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

    fn evaluate_active_native_strategy(
        &self,
        bars: &[crate::tradovate::Bar],
        current_qty: i32,
    ) -> (StrategySignal, String) {
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let evaluation = self
                    .strategy
                    .native_hma
                    .evaluate(bars, side_from_signed_qty(current_qty));
                (evaluation.signal, evaluation.summary())
            }
            NativeStrategyKind::EmaCross => {
                let evaluation = self
                    .strategy
                    .native_ema
                    .evaluate(bars, side_from_signed_qty(current_qty));
                (evaluation.signal, evaluation.summary())
            }
        }
    }

    fn maybe_run_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if !self.strategy_runtime.armed || self.strategy.kind != StrategyKind::Native {
            return;
        }

        let actual_market_qty = self.actual_market_position_qty();
        let actual_market_entry = self.actual_market_entry_price();
        self.sync_active_native_position(actual_market_qty, actual_market_entry);

        if self.strategy_runtime.pending_target_qty.is_some() {
            self.strategy_runtime.last_summary =
                "Waiting for prior automated order to settle.".to_string();
            return;
        }

        let closed_bars = self.closed_bars().to_vec();
        let Some(last_closed) = closed_bars.last() else {
            self.strategy_runtime.last_summary = format!(
                "Native {} armed; waiting for market data.",
                self.strategy.native_strategy.label()
            );
            return;
        };

        if self.strategy_runtime.last_closed_bar_ts.is_none() {
            self.strategy_runtime.last_closed_bar_ts = Some(last_closed.ts_ns);
            self.strategy_runtime.last_summary = format!(
                "Native {} anchored to current bar; waiting for next close.",
                self.strategy.native_strategy.label()
            );
            return;
        }

        if self.strategy_runtime.last_closed_bar_ts == Some(last_closed.ts_ns) {
            return;
        }
        self.strategy_runtime.last_closed_bar_ts = Some(last_closed.ts_ns);

        let current_qty = self.effective_market_position_qty();
        let (signal, summary) = self.evaluate_active_native_strategy(&closed_bars, current_qty);
        self.strategy_runtime.last_summary = summary.clone();

        let Some(target_qty) =
            target_qty_for_signal(signal, current_qty, self.base_config.order_qty)
        else {
            self.maybe_sync_native_protection(cmd_tx, Some(last_closed));
            return;
        };

        if target_qty == current_qty {
            self.maybe_sync_native_protection(cmd_tx, Some(last_closed));
            return;
        }

        let reason = format!(
            "{} {} | {}",
            self.active_native_slug(),
            signal.label(),
            summary
        );
        if current_qty != 0 {
            self.clear_native_protection(
                cmd_tx,
                format!(
                    "{} target transition {} -> {}",
                    self.active_native_slug(),
                    current_qty,
                    target_qty
                ),
            );
        }
        let _ = cmd_tx.send(ServiceCommand::SetTargetPosition {
            target_qty,
            automated: true,
            reason: reason.clone(),
        });
        self.strategy_runtime.pending_target_qty = Some(target_qty);
        self.push_log(format!(
            "Native {} target {} -> {} ({})",
            self.strategy.native_strategy.label(),
            current_qty,
            target_qty,
            reason
        ));
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

fn format_money(value: Option<f64>) -> String {
    match value {
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

fn target_qty_for_signal(signal: StrategySignal, current_qty: i32, base_qty: i32) -> Option<i32> {
    let base_qty = base_qty.max(1);
    match signal {
        StrategySignal::Hold => None,
        StrategySignal::EnterLong => Some(base_qty),
        StrategySignal::EnterShort => Some(-base_qty),
        StrategySignal::ExitLongOnShortSignal => {
            if current_qty > 0 {
                Some(0)
            } else {
                None
            }
        }
    }
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
