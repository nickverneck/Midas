use super::*;

impl App {
    pub(in crate::app) fn header_help_text(&self) -> String {
        let stats_hint = self
            .session_stats_affordance_visible()
            .then_some("F6 stats");
        let mut items = match self.screen {
            Screen::EngineSelect => vec![
                if self.engine_create_affordance_visible() {
                    "Up/Down/Left/Right choose engine"
                } else {
                    "Up/Down/Left/Right choose live engine"
                }
                .to_string(),
                if self.engine_create_affordance_visible() {
                    "Enter attach/create".to_string()
                } else {
                    "Enter attach".to_string()
                },
                "r refresh".to_string(),
                "F5/Ctrl+S save logs".to_string(),
                "q quit".to_string(),
            ],
            Screen::BrokerSelect => {
                let mut items = vec![
                    "Up/Down/Left/Right choose broker".to_string(),
                    "Enter open login/token options".to_string(),
                ];
                if let Some(hint) = stats_hint {
                    items.push(hint.to_string());
                }
                items.extend(["F5/Ctrl+S save logs".to_string(), "q quit".to_string()]);
                items
            }
            Screen::Login => {
                let mut items = vec![
                    "F2 selection".to_string(),
                    "Up/Down focus".to_string(),
                    "Left/Right toggle env/auth/logs".to_string(),
                    if self.replay_affordance_visible() {
                        "Enter connect/replay".to_string()
                    } else {
                        "Enter connect".to_string()
                    },
                ];
                if let Some(hint) = stats_hint {
                    items.push(hint.to_string());
                }
                items.extend([
                    "Esc broker picker".to_string(),
                    "F5/Ctrl+S save logs".to_string(),
                    "q quit".to_string(),
                ]);
                items
            }
            Screen::Selection => {
                let mut items = vec![
                    "F1 login".to_string(),
                    "F3 strategy".to_string(),
                    "F4 dashboard".to_string(),
                ];
                if let Some(hint) = stats_hint {
                    items.push(hint.to_string());
                }
                items.push("Tab focus".to_string());
                if self.bar_type_controls_visible() {
                    items.push("Left/Right bar type".to_string());
                }
                items.extend([
                    "Enter search/select".to_string(),
                    "F5/Ctrl+S save logs".to_string(),
                ]);
                items
            }
            Screen::Strategy => {
                let mut items = vec![
                    "F1 login".to_string(),
                    "F2 selection".to_string(),
                    "F4 dashboard".to_string(),
                ];
                if let Some(hint) = stats_hint {
                    items.push(hint.to_string());
                }
                items.extend([
                    "Up/Down focus".to_string(),
                    "Left/Right edit native strategy settings".to_string(),
                    "F5/Ctrl+S save logs".to_string(),
                ]);
                items
            }
            Screen::Dashboard => {
                let mut items = vec![
                    "F1 login".to_string(),
                    "F2 selection".to_string(),
                    "F3 strategy".to_string(),
                ];
                if let Some(hint) = stats_hint {
                    items.push(hint.to_string());
                }
                if self.automated_strategy_affordance_visible() {
                    items
                        .push("native runtime follows configured timing/reversal mode".to_string());
                } else {
                    items.push("monitor-only runtime".to_string());
                }
                if self.manual_order_affordance_visible() {
                    items.push("b/s/c manual".to_string());
                }
                items.push("v visuals".to_string());
                if self.session_kind == SessionKind::Replay {
                    items.extend(["[/] replay speed".to_string(), "0 realtime".to_string()]);
                }
                items.extend(["F5/Ctrl+S save logs".to_string(), "q quit".to_string()]);
                items
            }
            Screen::Stats => vec![
                "F1 login".to_string(),
                "F2 selection".to_string(),
                "F3 strategy".to_string(),
                "F4 dashboard".to_string(),
                "Up/Down account".to_string(),
                "Enter re-sync".to_string(),
                "f fees".to_string(),
                "F5/Ctrl+S save logs".to_string(),
                "q quit".to_string(),
            ],
        };
        items.retain(|item| !item.is_empty());
        items.join(" | ")
    }

    pub(in crate::app) fn header_tab_titles(&self) -> Vec<&'static str> {
        let mut titles = vec![
            "Engine",
            "Broker",
            "Login",
            "Selection",
            "Strategy",
            "Dashboard",
        ];
        if self.session_stats_affordance_visible() {
            titles.push("Stats");
        }
        titles
    }

    fn header_selected_tab(&self) -> usize {
        match self.screen {
            Screen::EngineSelect => 0,
            Screen::BrokerSelect => 1,
            Screen::Login => 2,
            Screen::Selection => 3,
            Screen::Strategy => 4,
            Screen::Dashboard => 5,
            Screen::Stats if self.session_stats_affordance_visible() => 6,
            Screen::Stats => 5,
        }
    }

    pub(in crate::app) fn render_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area);

        let screen_label = match self.screen {
            Screen::EngineSelect => "Engine",
            Screen::BrokerSelect => "Broker",
            Screen::Login => "Login",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
            Screen::Stats => "Stats",
        };
        let help = self.header_help_text();
        let titles = self
            .header_tab_titles()
            .into_iter()
            .map(Line::from)
            .collect::<Vec<_>>();
        let selected_tab = self.header_selected_tab();
        let tabs = Tabs::new(titles)
            .select(selected_tab)
            .highlight_style(Style::default().fg(Color::Black).bg(Color::Cyan))
            .divider("|");
        frame.render_widget(tabs, rows[0]);

        let auth_label = if self.session_kind == SessionKind::Replay {
            "Replay"
        } else {
            self.form.auth_mode.label()
        };
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Trader",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(format!("Broker: {}", self.selected_broker.label())),
                Span::raw("  "),
                Span::raw(format!("Screen: {screen_label}")),
                Span::raw("  "),
                Span::raw(format!("Env: {}", self.form.env.label())),
                Span::raw("  "),
                Span::raw(format!("Auth: {auth_label}")),
                Span::raw("  "),
                Span::raw(format!("Logs: {}", self.form.log_mode.label())),
                Span::raw("  "),
                Span::raw(format!("Mode: {}", self.session_kind.label())),
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

    pub(in crate::app) fn log_panel_lines(&self) -> Vec<Line<'static>> {
        let log_take = if self.last_saved_log_path.is_some() {
            5
        } else {
            6
        };
        let mut lines = self
            .logs
            .iter()
            .rev()
            .take(log_take)
            .map(LogEntry::render_line)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(Line::from)
            .collect::<Vec<_>>();
        if let Some(path) = &self.last_saved_log_path {
            lines.insert(0, Line::from(format!("Last saved: {}", path.display())));
        }
        lines
    }

    pub(in crate::app) fn render_logs(&self, frame: &mut Frame<'_>, area: Rect) {
        let lines = self.log_panel_lines();
        let logs =
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(
                    if self.session_stats.enabled {
                        "Log [F5/Ctrl+S saves logs + stats to .run/]"
                    } else {
                        "Log [F5/Ctrl+S saves to .run/]"
                    },
                ))
                .wrap(Wrap { trim: true });
        frame.render_widget(logs, area);
    }
}
