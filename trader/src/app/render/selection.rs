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
                        let _ = cmd_tx.send(ServiceCommand::SubscribeBars {
                            contract,
                            bar_type: self.bar_type,
                        });
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
            Focus::BarTypeToggle => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    if self.broker_supports_range_bars() {
                        self.bar_type = self.bar_type.toggle();
                    } else {
                        self.bar_type = BarType::Minute1;
                        self.status = format!(
                            "{} currently supports 1-minute bars only.",
                            self.selected_broker.label()
                        );
                    }
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
            },
            Focus::AccountList
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
            | Focus::NativeReversalMode
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
}
