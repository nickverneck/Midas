use super::*;

impl App {
    pub(super) fn render_replay_dataset_views_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(area);
        let selected_path = self.replay_dataset_view_path.as_ref();
        let items = if self.replay_dataset_views.views.is_empty() {
            vec![ListItem::new(Line::from(
                "No saved views for this dataset.",
            ))]
        } else {
            self.replay_dataset_views
                .views
                .iter()
                .map(|resolved| {
                    let active = selected_path == Some(&resolved.view_path);
                    ListItem::new(vec![
                        Line::from(format!(
                            "{} {}",
                            if active { "*" } else { " " },
                            resolved.view.id
                        )),
                        Line::from(format!(
                            "  {} | warmup {}m",
                            resolved.view.session_preset.label(),
                            resolved.view.warmup.duration_seconds / 60
                        )),
                        Line::from(format!(
                            "  {} to {} UTC",
                            resolved.view.evaluation_start.format("%Y-%m-%d %H:%M"),
                            resolved.view.evaluation_end.format("%Y-%m-%d %H:%M")
                        )),
                    ])
                })
                .collect()
        };
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Saved Dataset Views"),
            )
            .highlight_symbol("> ")
            .scroll_padding(1);
        let mut state = focused_list_state(
            self.focus == Focus::ReplayViewList,
            self.replay_dataset_views.selected_index.unwrap_or(0),
            self.replay_dataset_views.views.len(),
        );
        frame.render_stateful_widget(list, columns[0], &mut state);

        let lines = if let Some(editor) = self.replay_dataset_views.editor.as_ref() {
            let mut lines = vec![
                Line::from("Esc cancels edit | Tab/Shift-Tab navigates"),
                styled_line(
                    format!("View ID: {}", editor.id),
                    self.focus == Focus::ReplayViewId,
                ),
                styled_line(
                    format!("Preset: {} (Left/Right)", editor.preset.label()),
                    self.focus == Focus::ReplayViewPreset,
                ),
            ];
            match editor.preset {
                ReplayDatasetSessionPreset::FullSource => {
                    lines.push(Line::from("Window: exact source coverage"));
                }
                ReplayDatasetSessionPreset::FuturesGlobex
                | ReplayDatasetSessionPreset::FuturesRthNewYork
                | ReplayDatasetSessionPreset::FuturesRthChicago => lines.push(styled_line(
                    format!("Trading date: {}", editor.trading_date),
                    self.focus == Focus::ReplayViewTradingDate,
                )),
                ReplayDatasetSessionPreset::CustomLocal => {
                    lines.extend([
                        styled_line(
                            format!("Start: {}", editor.start),
                            self.focus == Focus::ReplayViewStart,
                        ),
                        styled_line(
                            format!("End: {}", editor.end),
                            self.focus == Focus::ReplayViewEnd,
                        ),
                        styled_line(
                            format!("IANA timezone: {}", editor.timezone),
                            self.focus == Focus::ReplayViewTimezone,
                        ),
                    ]);
                }
                ReplayDatasetSessionPreset::CustomUtc => {
                    lines.extend([
                        styled_line(
                            format!("UTC start: {}", editor.start),
                            self.focus == Focus::ReplayViewStart,
                        ),
                        styled_line(
                            format!("UTC end: {}", editor.end),
                            self.focus == Focus::ReplayViewEnd,
                        ),
                    ]);
                }
            }
            lines.extend([
                styled_line(
                    format!("Warmup minutes: {}", editor.warmup_minutes),
                    self.focus == Focus::ReplayViewWarmupMinutes,
                ),
                Line::from("Warmup seeds indicators; trading stays flat."),
                styled_line(
                    "[Enter] Validate, save, and select".to_string(),
                    self.focus == Focus::ReplayViewSave,
                ),
                Line::from(""),
                Line::from(self.replay_dataset_views.message.clone()),
            ]);
            lines
        } else {
            let mut lines = vec![
                Line::from("Enter/Space: select highlighted view"),
                Line::from("N: create | E: edit | C: clear/use full source"),
                Line::from("Esc: return to Replay"),
                Line::from(""),
                Line::from("* marks the view currently selected for startup."),
                Line::from(""),
                Line::from(self.replay_dataset_views.message.clone()),
            ];
            for warning in self.replay_dataset_views.warnings.iter().take(3) {
                lines.push(Line::from(format!("Warning: {warning}")));
            }
            if let Some(resolved) = self
                .replay_dataset_views
                .selected_index
                .and_then(|index| self.replay_dataset_views.views.get(index))
            {
                lines.extend([
                    Line::from(""),
                    Line::from(format!("Source: {}", resolved.view.source.contract)),
                    Line::from(
                        resolved
                            .view
                            .evaluation_label()
                            .unwrap_or_else(|error| format!("Invalid label: {error}")),
                    ),
                    Line::from(format!("File: {}", resolved.view_path.display())),
                ]);
            }
            lines
        };
        let scroll = focused_paragraph_scroll_offset(&lines, columns[1]);
        let details = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(
                if self.replay_dataset_views.editor.is_some() {
                    "View Editor"
                } else {
                    "View Controls"
                },
            ))
            .scroll((scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(details, columns[1]);
    }
}
