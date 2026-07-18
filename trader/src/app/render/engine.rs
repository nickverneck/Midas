use super::*;

impl App {
    pub(in crate::app) fn handle_engine_select_key(&mut self, key: KeyEvent) {
        let item_count = self.engine_summaries.len() + 1;
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Left | KeyCode::BackTab => {
                self.selected_engine = (self.selected_engine + item_count - 1) % item_count;
                self.update_engine_select_status();
            }
            KeyCode::Down | KeyCode::Right | KeyCode::Tab => {
                self.selected_engine = (self.selected_engine + 1) % item_count;
                self.update_engine_select_status();
            }
            KeyCode::Enter => {
                if self.selected_engine == self.engine_summaries.len() {
                    if !self.engine_creation_enabled {
                        self.status =
                            "Engine creation is disabled by --no-spawn-engine.".to_string();
                        self.push_log(format!("ERROR: {}", self.status));
                        return;
                    }
                    self.pending_engine_selection_action = Some(EngineSelectionAction::CreateNew);
                    self.status = "Creating a new engine...".to_string();
                    self.push_log(self.status.clone());
                    return;
                }

                let Some(engine) = self.engine_summaries.get(self.selected_engine) else {
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

    fn update_engine_select_status(&mut self) {
        if self.selected_engine == self.engine_summaries.len() {
            self.status = if self.engine_creation_enabled {
                "Create a new engine for this TUI session.".to_string()
            } else {
                "Engine creation is disabled by --no-spawn-engine.".to_string()
            };
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
        }
    }

    pub(in crate::app) fn render_engine_select_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(10),
                Constraint::Length(3),
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

        let create_selected = self.selected_engine == self.engine_summaries.len();
        let create_style = if create_selected && self.engine_creation_enabled {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else if self.engine_creation_enabled {
            Style::default().fg(Color::Gray)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let create_marker = if create_selected { ">" } else { " " };
        let create_state = if self.engine_creation_enabled {
            "create"
        } else {
            "disabled"
        };
        rows.push(
            Row::new(vec![
                Cell::from(create_marker.to_string()),
                Cell::from("+"),
                Cell::from(create_state),
                Cell::from("new engine"),
                Cell::from("-"),
                Cell::from("-"),
                Cell::from("-"),
                Cell::from("-"),
                Cell::from("-"),
                Cell::from("-"),
                Cell::from(if self.engine_creation_enabled {
                    "Start a new engine"
                } else {
                    "Disabled by --no-spawn-engine"
                }),
            ])
            .style(create_style),
        );

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

        let help = Paragraph::new(vec![
            Line::from("Enter opens a live engine in the existing detail flow. Inactive engines keep updating here."),
            Line::from("Stale or closed engines are shown for context but cannot be opened."),
        ])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Engine Controls"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(help, layout[2]);
    }
}
