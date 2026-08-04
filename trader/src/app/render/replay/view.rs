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

        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Setup {
            self.render_replay_setup_screen(frame, area);
            return;
        }

        self.render_replay_library_screen(frame, area);
    }

    fn render_replay_library_screen(&self, frame: &mut Frame<'_>, area: Rect) {
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
        let dataset_scroll = focused_paragraph_scroll_offset(&dataset_lines, columns[0]);
        let dataset = Paragraph::new(dataset_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Cached Replay Contracts"),
            )
            .scroll((dataset_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(dataset, columns[0]);

        let metadata_lines = if compact {
            self.replay_dataset_metadata_compact_lines()
        } else {
            self.replay_dataset_metadata_lines()
        };
        let metadata_scroll = focused_paragraph_scroll_offset(&metadata_lines, columns[1]);
        let metadata = Paragraph::new(metadata_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Metadata"),
            )
            .scroll((metadata_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(metadata, columns[1]);
    }

    #[cfg(feature = "replay")]
    fn render_replay_setup_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let compact = area.width < 120 || area.height < 28;
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        let market_lines = if compact {
            self.replay_market_control_compact_lines()
        } else {
            self.replay_market_control_lines()
        };
        let market_scroll = focused_paragraph_scroll_offset(&market_lines, columns[0]);
        let market = Paragraph::new(market_lines)
            .block(Block::default().borders(Borders::ALL).title("Market Setup"))
            .scroll((market_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(market, columns[0]);

        let run_lines = if compact {
            self.replay_run_control_compact_lines()
        } else {
            self.replay_run_control_lines()
        };
        let run_scroll = focused_paragraph_scroll_offset(&run_lines, columns[1]);
        let run = Paragraph::new(run_lines)
            .block(Block::default().borders(Borders::ALL).title("Run Setup"))
            .scroll((run_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(run, columns[1]);
    }
}
