impl App {
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
                    self.sync_selected_account(cmd_tx);
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
                    self.sync_selected_account(cmd_tx);
                    return;
                }
                KeyCode::Left => {
                    self.focus = Focus::ContractList;
                    return;
                }
                KeyCode::Right => {
                    self.focus = Focus::BarTypeToggle;
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
                        self.sync_selected_account(cmd_tx);
                        let _ = cmd_tx.send(ServiceCommand::SubscribeBars { contract, bar_type: self.bar_type });
                        self.screen = Screen::Strategy;
                        self.focus = Focus::StrategyKind;
                    }
                    return;
                }
                KeyCode::Left => {
                    self.focus = Focus::BarTypeToggle;
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
                        self.focus = Focus::BarTypeToggle;
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = Focus::ContractList;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = Focus::BarTypeToggle;
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
            Focus::BarTypeToggle => {
                match key.code {
                    KeyCode::Left | KeyCode::Right => {
                        self.bar_type = self.bar_type.toggle();
                        return;
                    }
                    KeyCode::Up => {
                        self.focus = Focus::AccountList;
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = Focus::InstrumentQuery;
                        return;
                    }
                    KeyCode::Enter => {
                        self.focus = Focus::InstrumentQuery;
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
                    'v' => {
                        self.dashboard_visuals_enabled = !self.dashboard_visuals_enabled;
                        self.push_log(format!(
                            "Dashboard visuals {}",
                            if self.dashboard_visuals_enabled {
                                "enabled"
                            } else {
                                "disabled"
                            }
                        ));
                        return;
                    }
                    'b' => Some(ManualOrderAction::Buy),
                    's' => Some(ManualOrderAction::Sell),
                    'c' => Some(ManualOrderAction::Close),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(action) = action {
            self.sync_selected_account(cmd_tx);
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
                "F2 selection | Up/Down focus | Left/Right toggle | Enter connect | F5/Ctrl+S save logs | q quit"
            }
            Screen::Selection => {
                "F1 login | F3 strategy | F4 dashboard | Tab focus | Left/Right bar type | Enter search/select | F5/Ctrl+S save logs"
            }
            Screen::Strategy => {
                "F1 login | F2 selection | F4 dashboard | Up/Down focus | Left/Right edit HMA | F5/Ctrl+S save logs"
            }
            Screen::Dashboard => {
                "F1 login | F2 selection | F3 strategy | native HMA auto-runs on closed bars | b/s/c manual | v visuals | F5/Ctrl+S save logs | q quit"
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
                format!("Bar Type: {}", self.bar_type.label()),
                self.focus == Focus::BarTypeToggle,
            ),
            styled_line(
                format!("Query: {}", self.instrument_query),
                self.focus == Focus::InstrumentQuery,
            ),
            Line::from(
                "Choose bar type first with Left/Right, then Enter or Down to move into Query. Enter on Query searches. Enter on a result subscribes.",
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
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(16), Constraint::Min(12)])
            .split(columns[0]);

        let session = Paragraph::new(self.dashboard_summary_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Session + Selection"),
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
        let overlay = self
            .dashboard_visuals_enabled
            .then(|| self.build_dashboard_visual_overlay(&bars, &buy_marker_points, &sell_marker_points));
        if let Some(overlay) = overlay.as_ref() {
            title.push_str(" | Visuals ");
            title.push_str(&overlay.label.to_uppercase());
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
        if !self.dashboard_visuals_enabled && !buy_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&buy_marker_points),
            );
        }
        if !self.dashboard_visuals_enabled && !sell_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Block)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Magenta))
                    .data(&sell_marker_points),
            );
        }
        let chart_block = Block::default().borders(Borders::ALL).title(title);
        let plot_area = chart_block.inner(area);
        let x_bounds = [0.0, points.len().max(1) as f64];
        let chart = Chart::new(datasets)
            .block(chart_block)
            .x_axis(Axis::default().bounds(x_bounds))
            .y_axis(Axis::default().bounds(y_bounds));
        frame.render_widget(chart, area);
        if let Some(overlay) = overlay.as_ref() {
            self.render_dashboard_canvas_overlay(frame, plot_area, x_bounds, y_bounds, overlay);
        }
    }

    fn render_dashboard_canvas_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        x_bounds: [f64; 2],
        y_bounds: [f64; 2],
        overlay: &DashboardVisualOverlay,
    ) {
        if area.is_empty() {
            return;
        }

        if !overlay.indicator_segments.is_empty() {
            let indicator_canvas = Canvas::default()
                .marker(symbols::Marker::Braille)
                .x_bounds(x_bounds)
                .y_bounds(y_bounds)
                .paint(|ctx| {
                    for segment in &overlay.indicator_segments {
                        ctx.draw(&CanvasLine {
                            x1: segment.start.0,
                            y1: segment.start.1,
                            x2: segment.end.0,
                            y2: segment.end.1,
                            color: segment.color,
                        });
                    }
                });
            frame.render_widget(indicator_canvas, area);
        }

        if overlay.glyphs.is_empty() {
            return;
        }

        let x_span = (x_bounds[1] - x_bounds[0]).abs().max(1.0);
        let y_span = (y_bounds[1] - y_bounds[0]).abs().max(0.0001);
        let dx = (x_span / 70.0).clamp(0.65, 2.5);
        let dy = (y_span / 35.0)
            .max(
                self.market
                    .tick_size
                    .filter(|tick| tick.is_finite() && *tick > 0.0)
                    .map(|tick| tick * 1.5)
                    .unwrap_or(y_span / 90.0),
            )
            .min(y_span / 8.0)
            .max(y_span / 150.0);
        let glyph_canvas = Canvas::default()
            .marker(symbols::Marker::HalfBlock)
            .x_bounds(x_bounds)
            .y_bounds(y_bounds)
            .paint(|ctx| {
                for glyph in overlay.glyphs.iter().copied() {
                    self.draw_overlay_glyph(ctx, glyph, dx, dy);
                }
            });
        frame.render_widget(glyph_canvas, area);
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
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Log [F5/Ctrl+S saves to .run/]"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(logs, area);
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::strategy::ExecutionStateSnapshot;
    use crate::tradovate::{AccountInfo, ContractSuggestion, ManualOrderAction, ServiceEvent};
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

    fn expect_select_account(rx: &mut UnboundedReceiver<ServiceCommand>, account_id: i64) {
        match rx.try_recv().expect("expected select-account command") {
            ServiceCommand::SelectAccount { account_id: actual } => {
                assert_eq!(actual, account_id);
            }
            _ => panic!("expected select-account command"),
        }
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
    fn execution_state_syncs_selected_account_index_from_engine() {
        let mut app = App::new(AppConfig::default());
        let (cmd_tx, _cmd_rx) = unbounded_channel();
        app.handle_service_event(
            ServiceEvent::AccountsLoaded(vec![
                account(1, "DEMO4769136"),
                account(2, "CHMMMLE422"),
            ]),
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
    fn dashboard_manual_orders_sync_selected_account_first() {
        let mut app = App::new(AppConfig::default());
        let (cmd_tx, mut cmd_rx) = unbounded_channel();
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
}
