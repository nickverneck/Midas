use super::*;

impl App {
    pub(in crate::app) fn handle_engine_select_key(&mut self, key: KeyEvent) {
        if self.pending_engine_create_mode.is_some() {
            self.handle_engine_create_mode_key(key);
            return;
        }
        if self.pending_engine_lifecycle_confirmation.is_some() {
            self.handle_engine_lifecycle_confirmation_key(key);
            return;
        }

        let item_count = self.engine_select_item_count();
        match key.code {
            KeyCode::Char('k') | KeyCode::Char('K')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.open_engine_lifecycle_confirmation(EngineLifecycleAction::Kill);
            }
            KeyCode::Char('x') | KeyCode::Char('X')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                if self.engine_close_and_kill_affordance_visible() {
                    self.open_engine_lifecycle_confirmation(EngineLifecycleAction::CloseAndKill);
                } else {
                    self.status =
                        "Close-and-kill is disabled; rebuild with --features manual-orders."
                            .to_string();
                    self.push_log(format!("ERROR: {}", self.status));
                }
            }
            KeyCode::Char('r') | KeyCode::Char('R')
                if !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.pending_engine_selection_action = Some(EngineSelectionAction::Refresh);
                self.status = "Refreshing engine list...".to_string();
                self.push_log(self.status.clone());
            }
            KeyCode::Char('q') | KeyCode::Char('Q')
                if !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Left | KeyCode::BackTab => {
                if item_count == 0 {
                    return;
                }
                self.selected_engine = (self.selected_engine + item_count - 1) % item_count;
                self.update_engine_select_status();
            }
            KeyCode::Down | KeyCode::Right | KeyCode::Tab => {
                if item_count == 0 {
                    return;
                }
                self.selected_engine = (self.selected_engine + 1) % item_count;
                self.update_engine_select_status();
            }
            KeyCode::Enter => {
                if self.engine_create_affordance_visible()
                    && self.selected_engine == self.engine_summaries.len()
                {
                    if self.replay_build_available() {
                        self.pending_engine_create_mode = Some(EngineCreateMode::Broker);
                        self.status =
                            "Choose whether the new engine is a broker or replay session."
                                .to_string();
                    } else {
                        self.start_engine_creation(EngineCreateMode::Broker);
                    }
                    return;
                }

                let Some(engine) = self.engine_summaries.get(self.selected_engine) else {
                    self.status =
                        "No live engines are available. Refresh the engine list.".to_string();
                    return;
                };
                if !engine.connection_state.attachable() {
                    self.status = format!(
                        "Engine {} is {}; choose a live engine or create a new one.",
                        engine
                            .id
                            .map(|id| id.to_string())
                            .unwrap_or_else(|| engine.socket_path.display().to_string()),
                        engine.connection_state.label()
                    );
                    self.push_log(format!("ERROR: {}", self.status));
                    return;
                }

                self.pending_engine_selection_action = Some(EngineSelectionAction::Attach {
                    engine_key: engine.key.clone(),
                    socket_path: engine.socket_path.clone(),
                });
                self.status = format!(
                    "Opening engine {}...",
                    engine
                        .id
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| engine.socket_path.display().to_string())
                );
                self.push_log(self.status.clone());
            }
            _ => {}
        }
    }

    fn handle_engine_create_mode_key(&mut self, key: KeyEvent) {
        let Some(selected_mode) = self.pending_engine_create_mode else {
            return;
        };
        let next_mode = match key.code {
            KeyCode::Up | KeyCode::Left => Some(match selected_mode {
                EngineCreateMode::Broker => EngineCreateMode::Replay,
                EngineCreateMode::Replay => EngineCreateMode::Broker,
            }),
            KeyCode::Down | KeyCode::Right | KeyCode::Tab => Some(match selected_mode {
                EngineCreateMode::Broker => EngineCreateMode::Replay,
                EngineCreateMode::Replay => EngineCreateMode::Broker,
            }),
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.pending_engine_create_mode = None;
                self.start_engine_creation(selected_mode);
                return;
            }
            KeyCode::Esc => {
                self.pending_engine_create_mode = None;
                self.status = "Engine creation canceled.".to_string();
                self.push_log(self.status.clone());
                return;
            }
            _ => None,
        };
        if let Some(mode) = next_mode {
            // The modal is only opened when replay is compiled in, so both
            // choices are valid here. Keeping the guard makes this handler
            // safe if state is restored from an older session.
            if mode == EngineCreateMode::Replay && !self.replay_build_available() {
                return;
            }
            self.pending_engine_create_mode = Some(mode);
            self.status = format!("{} engine selected.", mode.label());
        }
    }

    fn start_engine_creation(&mut self, mode: EngineCreateMode) {
        self.pending_engine_selection_action = Some(EngineSelectionAction::CreateNew { mode });
        self.status = format!("Creating a {} engine...", mode.label().to_ascii_lowercase());
        self.push_log(self.status.clone());
    }

    fn handle_engine_lifecycle_confirmation_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => self.confirm_engine_lifecycle_action(),
            KeyCode::Esc => self.cancel_engine_lifecycle_action(),
            KeyCode::Char('y') | KeyCode::Char('Y')
                if !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.confirm_engine_lifecycle_action();
            }
            KeyCode::Char('n') | KeyCode::Char('N')
                if !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.cancel_engine_lifecycle_action();
            }
            _ => {}
        }
    }

    fn open_engine_lifecycle_confirmation(&mut self, action: EngineLifecycleAction) {
        if action == EngineLifecycleAction::CloseAndKill
            && !self.engine_close_and_kill_affordance_visible()
        {
            self.status =
                "Close-and-kill is disabled; rebuild with --features manual-orders.".to_string();
            self.push_log(format!("ERROR: {}", self.status));
            return;
        }

        let Some(engine) = self.engine_summaries.get(self.selected_engine) else {
            self.status = "Select a running engine first.".to_string();
            return;
        };

        let Some(id) = engine.id else {
            self.status = format!(
                "Cannot {} engine at {} because no process ID is known.",
                action.status_verb(),
                engine.socket_path.display()
            );
            self.push_log(format!("ERROR: {}", self.status));
            return;
        };

        if action == EngineLifecycleAction::CloseAndKill && !engine.connection_state.attachable() {
            self.status = format!(
                "Cannot close and kill engine {id} because it is {}; choose a live engine.",
                engine.connection_state.label()
            );
            self.push_log(format!("ERROR: {}", self.status));
            return;
        }

        let confirmation = EngineLifecycleConfirmation {
            action,
            engine_key: engine.key.clone(),
            id,
            socket_path: engine.socket_path.clone(),
            state: engine.connection_state,
            broker_mode: engine.broker_mode_label(),
            account: engine.account_label(),
            instrument: engine.instrument_label(),
            position: engine.position_label(),
            strategy: engine.strategy_label(),
            latest_status: engine.status_label(),
        };
        self.status = format!("Confirm {} for engine {id}.", action.status_verb());
        self.pending_engine_lifecycle_confirmation = Some(confirmation);
    }

    fn confirm_engine_lifecycle_action(&mut self) {
        let Some(confirmation) = self.pending_engine_lifecycle_confirmation.take() else {
            return;
        };

        if confirmation.action == EngineLifecycleAction::CloseAndKill {
            if !self.engine_close_and_kill_affordance_visible() {
                self.status = "Close-and-kill is disabled; rebuild with --features manual-orders."
                    .to_string();
                self.push_log(format!("ERROR: {}", self.status));
                return;
            }
            if let Some(engine) = self
                .engine_summaries
                .iter()
                .find(|summary| summary.key == confirmation.engine_key)
            {
                if !engine.connection_state.attachable() {
                    self.status = format!(
                        "Cannot close and kill engine {} because it is {}; choose a live engine.",
                        confirmation.id,
                        engine.connection_state.label()
                    );
                    self.push_log(format!("ERROR: {}", self.status));
                    return;
                }
            }
        }

        self.pending_engine_selection_action = Some(match confirmation.action {
            EngineLifecycleAction::Kill => EngineSelectionAction::Kill {
                id: confirmation.id,
            },
            EngineLifecycleAction::CloseAndKill => EngineSelectionAction::CloseAndKill {
                id: confirmation.id,
            },
        });
        self.status = confirmation.action.running_message(confirmation.id);
        self.push_log(self.status.clone());
    }

    fn cancel_engine_lifecycle_action(&mut self) {
        let Some(confirmation) = self.pending_engine_lifecycle_confirmation.take() else {
            return;
        };
        self.status = format!(
            "Canceled {} for engine {}.",
            confirmation.action.status_verb(),
            confirmation.id
        );
        self.push_log(self.status.clone());
    }

    fn update_engine_select_status(&mut self) {
        if self.engine_create_affordance_visible()
            && self.selected_engine == self.engine_summaries.len()
        {
            self.status = "Create a new engine for this TUI session.".to_string();
            return;
        }
        if let Some(engine) = self.engine_summaries.get(self.selected_engine) {
            self.status = format!(
                "Selected engine {} ({}).",
                engine
                    .id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| engine.socket_path.display().to_string()),
                engine.connection_state.label()
            );
        } else {
            self.status = "No live engines are available. Refresh the engine list.".to_string();
        }
    }

    pub(in crate::app) fn render_engine_select_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(10),
                Constraint::Length(5),
            ])
            .split(area);

        let title = Paragraph::new(Line::from(Span::styled(
            "Engine Overview",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(title, layout[0]);

        let mut rows = self
            .engine_summaries
            .iter()
            .enumerate()
            .map(|(index, engine)| {
                let selected = index == self.selected_engine;
                let marker = if selected { ">" } else { " " };
                let id = engine
                    .id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "new".to_string());
                let socket = engine
                    .socket_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_else(|| engine.socket_path.to_str().unwrap_or("-"));
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else if engine.connection_state.attachable() {
                    Style::default().fg(Color::Gray)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                Row::new(vec![
                    Cell::from(marker.to_string()),
                    Cell::from(id),
                    Cell::from(engine.connection_state.label()),
                    Cell::from(socket.to_string()),
                    Cell::from(engine.broker_mode_label()),
                    Cell::from(engine.account_label()),
                    Cell::from(engine.instrument_label()),
                    Cell::from(engine.position_label()),
                    Cell::from(engine.strategy_label()),
                    Cell::from(engine.latency_label()),
                    Cell::from(engine.status_label()),
                ])
                .style(style)
            })
            .collect::<Vec<_>>();

        if self.engine_create_affordance_visible() {
            let create_selected = self.selected_engine == self.engine_summaries.len();
            let create_style = if create_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            let create_marker = if create_selected { ">" } else { " " };
            rows.push(
                Row::new(vec![
                    Cell::from(create_marker.to_string()),
                    Cell::from("+"),
                    Cell::from("create"),
                    Cell::from("new engine"),
                    Cell::from("-"),
                    Cell::from("-"),
                    Cell::from("-"),
                    Cell::from("-"),
                    Cell::from("-"),
                    Cell::from("-"),
                    Cell::from("Start a new engine"),
                ])
                .style(create_style),
            );
        }

        let table = Table::new(
            rows,
            [
                Constraint::Length(1),
                Constraint::Length(6),
                Constraint::Length(10),
                Constraint::Length(24),
                Constraint::Length(22),
                Constraint::Length(14),
                Constraint::Length(14),
                Constraint::Length(5),
                Constraint::Length(16),
                Constraint::Length(8),
                Constraint::Min(12),
            ],
        )
        .header(
            Row::new(vec![
                "",
                "ID",
                "State",
                "Socket",
                "Broker/Mode",
                "Account",
                "Instrument",
                "Pos",
                "Strategy",
                "RTT",
                "Status",
            ])
            .style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Active Engines"),
        );
        frame.render_widget(table, layout[1]);

        let open_help = if self.engine_create_affordance_visible() {
            "Enter opens a live engine or creates a new one. r refreshes the process list."
        } else {
            "Enter opens a selected live engine. r refreshes the process list."
        };
        let mut help_lines = vec![
            Line::from(open_help),
            Line::from(
                "Live/connected rows attach. Stale/closed rows remain for observation context.",
            ),
            if self.replay_build_available() {
                Line::from(
                    "The create row opens a Broker/Replay workflow modal; replay skips login and account stats.",
                )
            } else {
                Line::from("Replay workflow is unavailable in this build.")
            },
            self.engine_lifecycle_help_line(),
            Line::from("Destructive actions always open a confirmation prompt before running."),
        ];
        if self.engine_create_affordance_visible() {
            help_lines.push(Line::from(
                "The create row starts a separate engine process.",
            ));
        }
        let help = Paragraph::new(help_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Engine Controls"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(help, layout[2]);

        self.render_engine_lifecycle_confirmation(frame, area);
        self.render_engine_create_mode_modal(frame, area);
    }

    fn engine_lifecycle_help_line(&self) -> Line<'static> {
        let Some(engine) = self.engine_summaries.get(self.selected_engine) else {
            return Line::from("Select an engine row for lifecycle actions.");
        };
        if engine.id.is_none() {
            return Line::from("Lifecycle actions require a known engine process ID.");
        }
        if engine.connection_state.attachable() {
            if self.engine_close_and_kill_affordance_visible() {
                Line::from(
                    "Ctrl+K kill selected engine. Ctrl+X close selected market and kill engine.",
                )
            } else {
                Line::from("Ctrl+K kill selected engine.")
            }
        } else {
            Line::from("Ctrl+K kill selected stale/closed engine.")
        }
    }

    fn render_engine_lifecycle_confirmation(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(confirmation) = &self.pending_engine_lifecycle_confirmation else {
            return;
        };

        let popup = centered_engine_popup(area);
        let warning = match confirmation.action {
            EngineLifecycleAction::Kill => "This will terminate the selected engine process.",
            EngineLifecycleAction::CloseAndKill => {
                "This will close the selected market, then terminate the engine process."
            }
        };
        let lines = vec![
            Line::from(warning),
            Line::from(""),
            Line::from(format!("ID: {}", confirmation.id)),
            Line::from(format!("Socket: {}", confirmation.socket_path.display())),
            Line::from(format!("State: {}", confirmation.state.label())),
            Line::from(format!("Broker/Mode: {}", confirmation.broker_mode)),
            Line::from(format!("Account: {}", confirmation.account)),
            Line::from(format!("Instrument: {}", confirmation.instrument)),
            Line::from(format!("Position: {}", confirmation.position)),
            Line::from(format!("Strategy: {}", confirmation.strategy)),
            Line::from(format!("Latest Status: {}", confirmation.latest_status)),
            Line::from(""),
            Line::from("Enter or y confirms. Esc or n cancels."),
        ];

        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(confirmation.action.title())
                        .border_style(Style::default().fg(Color::Red)),
                )
                .wrap(Wrap { trim: true }),
            popup,
        );
    }

    fn render_engine_create_mode_modal(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(selected_mode) = self.pending_engine_create_mode else {
            return;
        };

        let popup = centered_engine_mode_popup(area);
        let broker_style = if selected_mode == EngineCreateMode::Broker {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let replay_style = if selected_mode == EngineCreateMode::Replay {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let lines = vec![
            Line::from("Choose the workflow for the new engine:"),
            Line::from(""),
            Line::from(vec![Span::styled("  Broker", broker_style)]),
            Line::from(format!("  {}", EngineCreateMode::Broker.summary())),
            Line::from(vec![Span::styled("  Replay", replay_style)]),
            Line::from(format!("  {}", EngineCreateMode::Replay.summary())),
            Line::from(""),
            Line::from(format!("Selected: {}", selected_mode.label())),
            Line::from("Left/Right or Up/Down switches. Enter continues. Esc cancels."),
        ];

        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Create Engine")
                        .border_style(Style::default().fg(Color::Cyan)),
                )
                .wrap(Wrap { trim: true }),
            popup,
        );
    }
}

fn centered_engine_popup(area: Rect) -> Rect {
    let width = area.width.saturating_mul(3).saturating_div(4).max(50);
    let height = 17;
    let width = width.min(area.width.saturating_sub(2).max(1));
    let height = height.min(area.height.saturating_sub(2).max(1));
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn centered_engine_mode_popup(area: Rect) -> Rect {
    let width = area.width.saturating_mul(4).saturating_div(5).max(64);
    let height = 12;
    let width = width.min(area.width.saturating_sub(2).max(1));
    let height = height.min(area.height.saturating_sub(2).max(1));
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}
