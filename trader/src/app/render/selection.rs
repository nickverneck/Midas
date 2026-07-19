use super::*;

impl App {
    pub(in crate::app) fn handle_selection_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
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
                    self.focus = self.next_selection_focus();
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
                        let _ = cmd_tx.send(ServiceCommand::SubscribeBars {
                            contract,
                            bar_type: self.bar_type,
                            candle_mode: self.effective_candle_mode(),
                        });
                        self.screen = Screen::Strategy;
                        self.focus = Focus::StrategyKind;
                    }
                    return;
                }
                KeyCode::Left => {
                    self.focus = self.prev_selection_focus();
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
                        self.focus = self.prev_selection_focus();
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = Focus::ContractList;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = self.prev_selection_focus();
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
            Focus::BarTypeToggle => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    if self.bar_type_controls_visible() {
                        self.bar_type = match key.code {
                            KeyCode::Left => self.bar_type.previous_kind(),
                            _ => self.bar_type.next_kind(),
                        };
                    }
                    return;
                }
                KeyCode::Up => {
                    self.focus = self.prev_selection_focus();
                    return;
                }
                KeyCode::Down => {
                    self.focus = self.next_selection_focus();
                    return;
                }
                KeyCode::Enter => {
                    self.focus = self.next_selection_focus();
                    return;
                }
                _ => {}
            },
            Focus::BarValue => {
                match key.code {
                    KeyCode::Up => {
                        self.focus = self.prev_selection_focus();
                        return;
                    }
                    KeyCode::Down | KeyCode::Enter => {
                        self.focus = self.next_selection_focus();
                        return;
                    }
                    _ => {}
                }

                if !self.bar_type_controls_visible() {
                    self.bar_type = BarType::default();
                    return;
                }

                let mut value = self.bar_type.value() as usize;
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::BarValue,
                    &mut value,
                    key,
                    1,
                    1,
                ) {
                    self.bar_type = self
                        .bar_type
                        .with_value(value.min(u32::MAX as usize) as u32);
                    return;
                }
            }
            Focus::CandleModeToggle => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    if !self.candle_mode_controls_visible() {
                        self.candle_mode = CandleMode::Standard;
                        self.focus = Focus::InstrumentQuery;
                        return;
                    }
                    self.candle_mode = self.candle_mode.toggle();
                    return;
                }
                KeyCode::Up => {
                    self.focus = self.prev_selection_focus();
                    return;
                }
                KeyCode::Down => {
                    self.focus = self.next_selection_focus();
                    return;
                }
                KeyCode::Enter => {
                    self.focus = self.next_selection_focus();
                    return;
                }
                _ => {}
            },
            Focus::EngineList
            | Focus::AccountList
            | Focus::ContractList
            | Focus::Env
            | Focus::AuthMode
            | Focus::LogMode
            | Focus::TokenOverride
            | Focus::Username
            | Focus::Password
            | Focus::ApiKey
            | Focus::AppId
            | Focus::AppVersion
            | Focus::Cid
            | Focus::Secret
            | Focus::TokenPath
            | Focus::BrokerList
            | Focus::StrategyKind
            | Focus::OrderQty
            | Focus::NativeStrategy
            | Focus::HmaLength
            | Focus::HmaMinAngle
            | Focus::HmaAngleLookback
            | Focus::HmaBarsRequired
            | Focus::HmaLongsOnly
            | Focus::HmaInverted
            | Focus::NativeSignalTiming
            | Focus::NativeSignalDelayBars
            | Focus::NativeExecutionPath
            | Focus::NativeReversalMode
            | Focus::NativeBlockoutEnabled
            | Focus::NativeBlockoutMinutes
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
            | Focus::ReplayMode
            | Focus::Connect => {}
        }
    }

    pub(in crate::app) fn render_selection_screen(&self, frame: &mut Frame<'_>, area: Rect) {
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
            .block(Block::default().borders(Borders::ALL).title("Accounts"))
            .scroll_padding(1);
        let mut account_state = focused_list_state(
            self.focus == Focus::AccountList,
            self.selected_account,
            self.accounts.len(),
        );
        frame.render_stateful_widget(accounts, left[1], &mut account_state);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Min(10),
                Constraint::Length(8),
            ])
            .split(columns[1]);

        let mut search_lines = Vec::new();
        if self.bar_type_controls_visible() {
            search_lines.extend([
                styled_line(
                    format!("Bar Type: {}", self.bar_type.kind().label()),
                    self.focus == Focus::BarTypeToggle,
                ),
                styled_line(
                    format!("Value: {}", self.bar_value_text()),
                    self.focus == Focus::BarValue,
                ),
            ]);
        }
        if self.candle_mode_controls_visible() {
            search_lines.push(styled_line(
                format!("Candles: {}", self.candle_mode.label()),
                self.focus == Focus::CandleModeToggle,
            ));
        }
        let help = if self.bar_type_controls_visible() && self.candle_mode_controls_visible() {
            "Use Left/Right on bar type and candles, digits for value, Enter or Down advances. Enter on Query searches."
        } else if self.bar_type_controls_visible() {
            "Use Left/Right on bar type, digits for value, Enter or Down advances. Enter on Query searches."
        } else {
            "Enter on Query searches. Enter on a contract subscribes to bars."
        };
        search_lines.extend([
            styled_line(
                format!("Query: {}", self.instrument_query),
                self.focus == Focus::InstrumentQuery,
            ),
            Line::from(help),
        ]);

        let search_scroll = focused_paragraph_scroll_offset(&search_lines, right[0]);
        let search = Paragraph::new(search_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Instrument Search"),
            )
            .scroll((search_scroll, 0))
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
        let contract_list = contract_list.scroll_padding(1);
        let mut contract_state = focused_list_state(
            self.focus == Focus::ContractList,
            self.selected_contract,
            self.contract_results.len(),
        );
        frame.render_stateful_widget(contract_list, right[1], &mut contract_state);

        let preview = Paragraph::new(self.selection_preview_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Current Selection"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(preview, right[2]);
    }
}
