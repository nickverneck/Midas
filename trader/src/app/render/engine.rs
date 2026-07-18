use super::*;

impl App {
    pub(in crate::app) fn handle_engine_select_key(&mut self, key: KeyEvent) {
        let item_count = self.running_engines.len() + 1;
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
                if self.selected_engine == self.running_engines.len() {
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

                let Some(engine) = self.running_engines.get(self.selected_engine) else {
                    return;
                };
                if !engine.socket_is_live {
                    self.status = format!(
                        "Engine {} is stale; choose a live engine or create a new one.",
                        engine.id
                    );
                    self.push_log(format!("ERROR: {}", self.status));
                    return;
                }

                self.pending_engine_selection_action = Some(EngineSelectionAction::Attach {
                    socket_path: engine.socket_path.clone(),
                });
                self.status = format!("Attaching to engine {}...", engine.id);
                self.push_log(self.status.clone());
            }
            _ => {}
        }
    }

    fn update_engine_select_status(&mut self) {
        if self.selected_engine == self.running_engines.len() {
            self.status = if self.engine_creation_enabled {
                "Create a new engine for this TUI session.".to_string()
            } else {
                "Engine creation is disabled by --no-spawn-engine.".to_string()
            };
            return;
        }
        if let Some(engine) = self.running_engines.get(self.selected_engine) {
            let status = if engine.socket_is_live {
                "live"
            } else {
                "stale"
            };
            self.status = format!("Selected engine {} ({status}).", engine.id);
        }
    }

    pub(in crate::app) fn render_engine_select_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let centered_row = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area);
        let centered_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(12),
                Constraint::Percentage(76),
                Constraint::Percentage(12),
            ])
            .split(centered_row[1])[1];

        let mut lines = vec![
            Line::from(Span::styled(
                "Select Engine",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Choose a live engine to attach to, or start a new engine."),
            Line::from(""),
        ];

        if self.running_engines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No running engines found.",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            lines.extend(
                self.running_engines
                    .iter()
                    .enumerate()
                    .map(|(index, engine)| {
                        let selected = index == self.selected_engine;
                        let status = if engine.socket_is_live {
                            "live"
                        } else {
                            "stale"
                        };
                        let marker = if selected { ">" } else { " " };
                        let label = format!(
                            "{marker} Engine {} [{status}]  {}",
                            engine.id,
                            engine.socket_path.display()
                        );
                        let style = if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(if engine.socket_is_live {
                                    Color::Cyan
                                } else {
                                    Color::DarkGray
                                })
                                .add_modifier(Modifier::BOLD)
                        } else if engine.socket_is_live {
                            Style::default().fg(Color::Gray)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        Line::from(Span::styled(label, style))
                    }),
            );
        }

        let create_selected = self.selected_engine == self.running_engines.len();
        let create_style = if !self.engine_creation_enabled {
            Style::default().fg(Color::DarkGray)
        } else if create_selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let create_marker = if create_selected { ">" } else { " " };
        let create_label = if self.engine_creation_enabled {
            "Create new engine"
        } else {
            "Create new engine (disabled by --no-spawn-engine)"
        };
        lines.extend([
            Line::from(""),
            Line::from(Span::styled(
                format!("{create_marker} {create_label}"),
                create_style,
            )),
            Line::from(""),
            Line::from("Stale engines are shown for context but cannot be attached."),
            Line::from("Use arrow keys to move. Press Enter to continue."),
        ]);

        let card = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Engine Selection"),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(card, centered_area);
    }
}
