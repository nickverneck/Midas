use crate::automation::{StrategyDescriptor, default_strategy_catalog};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategy::{LuaSourceMode, StrategyKind, StrategyState};
use crate::tradovate::{
    AccountInfo, AccountSnapshot, ContractSuggestion, MarketSnapshot, ServiceCommand, ServiceEvent,
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
        };
        app.push_log(
            "Phase 1 enabled: auth, account selection, contract search, 1m history/live."
                .to_string(),
        );
        app.push_log("Phase 2 hotkeys (buy/sell/close) remain intentionally disabled.".to_string());
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

    pub fn handle_service_event(&mut self, event: ServiceEvent) {
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
                KeyCode::Left => self.strategy.kind = self.strategy.kind.prev(),
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.kind = self.strategy.kind.next()
                }
                _ => {}
            },
            Focus::LuaSourceMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.lua_source_mode = self.strategy.lua_source_mode.toggle();
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
                }
            }
            Focus::LuaEditor => {
                let _ = self.strategy.lua_editor.handle_key(key);
            }
            Focus::StrategyContinue => {
                if key.code == KeyCode::Enter {
                    self.screen = Screen::Dashboard;
                    self.focus = Focus::AccountList;
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
            | Focus::LuaSourceMode
            | Focus::LuaFilePath
            | Focus::LuaEditor
            | Focus::StrategyContinue
            | Focus::Connect => {}
        }
    }

    fn handle_dashboard_key(&mut self, _key: KeyEvent, _cmd_tx: &UnboundedSender<ServiceCommand>) {}

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
                "F1 login | F2 selection | F4 dashboard | Up/Down focus | Left/Right change | Vim editor for Lua"
            }
            Screen::Dashboard => "F1 login | F2 selection | F3 strategy | monitoring view | q quit",
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
            .constraints([Constraint::Length(14), Constraint::Min(10)])
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

        let editor = Paragraph::new(self.lua_editor_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(self.lua_editor_title()),
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
        let datasets = vec![
            Dataset::default()
                .name("Close")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&points),
        ];
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

        if self.strategy.kind == StrategyKind::Lua {
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
            "[Enter] Continue To Dashboard".to_string(),
            self.focus == Focus::StrategyContinue,
        ));
        lines
    }

    fn strategy_notes_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from("Backend order: Native Rust > Lua > Machine Learning."),
            Line::from("Native and ML are selection-only for now."),
            Line::from("Lua can be loaded from file or typed directly in the TUI."),
            Line::from("Strategy execution will be added later; this screen is selection only."),
            Line::from(""),
            Line::from("Vim-style editor controls:"),
            Line::from("Normal: h/j/k/l move, i insert, a append, o open line, x delete."),
            Line::from("Insert: type text, Enter newline, Backspace delete, Esc back to normal."),
        ]
    }

    fn strategy_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Selected: {}", self.strategy.summary_label())),
            Line::from(match self.strategy.kind {
                StrategyKind::Native => {
                    "Native Rust will be the preferred runtime when execution is added.".to_string()
                }
                StrategyKind::Lua => {
                    "Lua strategy source is ready for later execution wiring.".to_string()
                }
                StrategyKind::MachineLearning => {
                    "Machine Learning remains lowest execution priority.".to_string()
                }
            }),
        ];
        if self.strategy.kind == StrategyKind::Lua {
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

    fn lua_editor_title(&self) -> String {
        if self.strategy.kind != StrategyKind::Lua {
            return "Lua Editor Preview".to_string();
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

    fn lua_editor_lines(&self) -> Vec<Line<'static>> {
        if self.strategy.kind != StrategyKind::Lua {
            return vec![
                Line::from("Lua editor is only active when Strategy Type is Lua."),
                Line::from("Pick Lua on the left to edit or load a script."),
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
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Active Contract: {name}"),
                None => "Active Contract: none".to_string(),
            }),
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
        if self.strategy.kind == StrategyKind::Lua {
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

fn json_preview(snapshot: &AccountSnapshot) -> String {
    let raw = serde_json::json!({
        "account": snapshot.raw_account,
        "risk": snapshot.raw_risk,
        "cash": snapshot.raw_cash,
        "positions": snapshot.raw_positions,
    });
    serde_json::to_string_pretty(&raw).unwrap_or_else(|_| "{}".to_string())
}
