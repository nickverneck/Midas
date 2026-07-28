use super::*;

impl App {
    pub(in crate::app) fn render_replay_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Downloader {
            self.render_replay_downloader_screen(frame, area);
            return;
        }
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::DatasetViews {
            self.render_replay_dataset_views_screen(frame, area);
            return;
        }

        let compact = area.width < 120 || area.height < 28;
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(area);

        let dataset_lines = if compact {
            self.replay_dataset_library_compact_lines()
        } else {
            self.replay_dataset_library_lines()
        };
        let dataset = Paragraph::new(dataset_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Dataset Library"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(dataset, columns[0]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Min(8)])
            .split(columns[1]);

        let market_lines = if compact {
            self.replay_market_control_compact_lines()
        } else {
            self.replay_market_control_lines()
        };
        let controls = Paragraph::new(market_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Market"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(controls, right[0]);

        let run_lines = if compact {
            self.replay_run_control_compact_lines()
        } else {
            self.replay_run_control_lines()
        };
        let run = Paragraph::new(run_lines)
            .block(Block::default().borders(Borders::ALL).title("Run"))
            .wrap(Wrap { trim: true });
        frame.render_widget(run, right[1]);
    }
}
