use super::*;

impl App {
    pub(in crate::app) fn handle_session_stats_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match key.code {
            KeyCode::Up => {
                self.selected_account = self.selected_account.saturating_sub(1);
            }
            KeyCode::Down => {
                if self.selected_account + 1 < self.accounts.len() {
                    self.selected_account += 1;
                }
            }
            KeyCode::Enter => {
                self.sync_selected_account(cmd_tx);
            }
            KeyCode::Char('f') | KeyCode::Char('F') => {
                self.session_stats_show_fees = !self.session_stats_show_fees;
                self.status = if self.session_stats_show_fees {
                    "Session stats fee rows shown.".to_string()
                } else {
                    "Session stats fee rows hidden.".to_string()
                };
            }
            _ => {}
        }
    }

    pub(in crate::app) fn render_session_stats_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Min(12)])
            .split(columns[0]);

        let overview = Paragraph::new(self.session_stats_overview_lines())
            .block(Block::default().borders(Borders::ALL).title("Tracker"))
            .wrap(Wrap { trim: true });
        frame.render_widget(overview, left[0]);

        let account = Paragraph::new(self.selected_session_stats_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selected Account Stats"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(account, left[1]);

        let events = Paragraph::new(self.session_stats_event_lines(area.height as usize))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Recent Balance Delta Events"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(events, columns[1]);
    }
}
