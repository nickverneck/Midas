use super::*;

impl App {
    pub(in crate::app) fn render_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area);

        let screen_label = match self.screen {
            Screen::BrokerSelect => "Broker",
            Screen::Login => "Login",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
            Screen::Stats => "Stats",
        };
        let help = match self.screen {
            Screen::BrokerSelect => {
                "Up/Down/Left/Right choose broker | Enter open login/token options | F6 stats | F5/Ctrl+S save logs | q quit"
            }
            Screen::Login => {
                "F2 selection | Up/Down focus | Left/Right toggle env/auth/logs | Enter connect/replay | F6 stats | Esc broker picker | F5/Ctrl+S save logs | q quit"
            }
            Screen::Selection => {
                "F1 login | F3 strategy | F4 dashboard | F6 stats | Tab focus | Left/Right bar type | Enter search/select | F5/Ctrl+S save logs"
            }
            Screen::Strategy => {
                "F1 login | F2 selection | F4 dashboard | F6 stats | Up/Down focus | Left/Right edit native strategy settings | F5/Ctrl+S save logs"
            }
            Screen::Dashboard => {
                if self.session_kind == SessionKind::Replay {
                    "F1 login | F2 selection | F3 strategy | F6 stats | native runtime follows configured timing/reversal mode | b/s/c manual | v visuals | [/] replay speed | 0 realtime | F5/Ctrl+S save logs | q quit"
                } else {
                    "F1 login | F2 selection | F3 strategy | F6 stats | native runtime follows configured timing/reversal mode | b/s/c manual | v visuals | F5/Ctrl+S save logs | q quit"
                }
            }
            Screen::Stats => {
                "F1 login | F2 selection | F3 strategy | F4 dashboard | Up/Down choose account | Enter re-sync account | F5/Ctrl+S save logs | q quit"
            }
        };
        let titles = [
            "Broker",
            "Login",
            "Selection",
            "Strategy",
            "Dashboard",
            "Stats",
        ]
        .into_iter()
        .map(Line::from)
        .collect::<Vec<_>>();
        let selected_tab = match self.screen {
            Screen::BrokerSelect => 0,
            Screen::Login => 1,
            Screen::Selection => 2,
            Screen::Strategy => 3,
            Screen::Dashboard => 4,
            Screen::Stats => 5,
        };
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

    pub(in crate::app) fn render_logs(&self, frame: &mut Frame<'_>, area: Rect) {
        let lines = self
            .logs
            .iter()
            .rev()
            .take(6)
            .map(LogEntry::render_line)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(Line::from)
            .collect::<Vec<_>>();
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
