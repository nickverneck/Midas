use super::*;

impl App {
    pub(in crate::app) fn render_analytics_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        #[cfg(feature = "replay")]
        {
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(31), Constraint::Percentage(69)])
                .split(area);

            let run_items = self.analytics_run_items();
            let run_list = List::new(run_items)
                .block(Block::default().borders(Borders::ALL).title(format!(
                    "Saved Replay Runs ({})",
                    self.replay_analytics.entries.len()
                )))
                .highlight_symbol("> ")
                .scroll_padding(1);
            let mut run_state = focused_list_state(
                self.replay_analytics.focus == AnalyticsFocus::Runs,
                self.replay_analytics.selected_run,
                self.replay_analytics.entries.len(),
            );
            frame.render_stateful_widget(run_list, columns[0], &mut run_state);

            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(15),
                    Constraint::Min(8),
                    Constraint::Length(7),
                ])
                .split(columns[1]);

            let summary = Paragraph::new(self.analytics_summary_lines())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Research Summary (saved result; read-only)"),
                )
                .wrap(Wrap { trim: true });
            frame.render_widget(summary, right[0]);

            let rows = self.analytics_trade_table_rows();
            let table = Table::new(
                rows,
                [
                    Constraint::Length(5),
                    Constraint::Length(7),
                    Constraint::Length(11),
                    Constraint::Length(11),
                    Constraint::Length(11),
                    Constraint::Length(11),
                    Constraint::Length(10),
                    Constraint::Length(11),
                    Constraint::Length(14),
                ],
            )
            .header(
                Row::new([
                    "Trade",
                    "Side",
                    "Gross",
                    "MFE",
                    "MAE",
                    "Giveback",
                    "Capture",
                    "Post",
                    "Precision",
                ])
                .style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            )
            .column_spacing(1)
            .block(Block::default().borders(Borders::ALL).title(format!(
                "Trade Excursions | sort: {}",
                self.analytics_trade_sort_label()
            )));
            frame.render_widget(table, right[1]);

            let selected = Paragraph::new(self.analytics_selected_trade_lines())
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Selected Trade Detail"),
                )
                .wrap(Wrap { trim: true });
            frame.render_widget(selected, right[2]);
        }
        #[cfg(not(feature = "replay"))]
        {
            let disabled = Paragraph::new("Replay analytics requires the replay feature.")
                .block(Block::default().borders(Borders::ALL).title("Analytics"));
            frame.render_widget(disabled, area);
        }
    }
}
