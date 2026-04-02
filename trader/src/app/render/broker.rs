use super::*;

impl App {
    pub(in crate::app) fn handle_broker_select_key(&mut self, key: KeyEvent) {
        let next_broker = match key.code {
            KeyCode::Up | KeyCode::Left => self
                .available_brokers
                .iter()
                .position(|broker| *broker == self.selected_broker)
                .map(|index| {
                    (index + self.available_brokers.len() - 1) % self.available_brokers.len()
                })
                .and_then(|index| self.available_brokers.get(index).copied()),
            KeyCode::Down | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => self
                .available_brokers
                .iter()
                .position(|broker| *broker == self.selected_broker)
                .map(|index| (index + 1) % self.available_brokers.len())
                .and_then(|index| self.available_brokers.get(index).copied()),
            _ => None,
        };

        if let Some(next_broker) = next_broker {
            self.selected_broker = next_broker;
            self.status = format!("Broker selected: {}", self.selected_broker.label());
            return;
        }

        if key.code == KeyCode::Enter {
            self.screen = Screen::Login;
            self.focus = Focus::Env;
            self.status = format!("Login for {}", self.selected_broker.label());
        }
    }

    pub(in crate::app) fn render_broker_select_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let centered_row = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(area);
        let centered_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(60),
                Constraint::Percentage(20),
            ])
            .split(centered_row[1])[1];

        let broker_details = match self.selected_broker {
            BrokerKind::Tradovate => {
                "Tradovate supports live and sim login, token-based auth, contract search, and replay in replay-enabled builds."
            }
            BrokerKind::Ironbeam => {
                "Ironbeam supports live and sim login, token and credential flows, account discovery, and live websocket bars."
            }
        };

        let mut lines = vec![
            Line::from(Span::styled(
                "Select Broker",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Choose the broker you want to use for this session."),
            Line::from(""),
        ];
        lines.extend(self.available_brokers.iter().map(|broker| {
            let selected = *broker == self.selected_broker;
            let label = if selected {
                format!("> {} <", broker.label())
            } else {
                broker.label().to_string()
            };
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Line::from(Span::styled(label, style))
        }));
        lines.extend([
            Line::from(""),
            Line::from(format!("Selected: {}", self.selected_broker.label())),
            Line::from(broker_details),
            Line::from(""),
            Line::from("Use the arrow keys to switch brokers."),
            Line::from("Press Enter to continue to the login screen."),
        ]);

        let card = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Broker Selection"),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(card, centered_area);
    }
}
