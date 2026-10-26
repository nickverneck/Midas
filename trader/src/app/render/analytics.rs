use super::*;

impl App {
    pub(in crate::app) fn render_analytics_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        #[cfg(feature = "replay")]
        {
            if area.width < 110 {
                self.render_analytics_compact(frame, area);
                return;
            }

            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(31), Constraint::Percentage(69)])
                .split(area);

            let showing_sweeps = self.replay_analytics.focus == AnalyticsFocus::Sweeps;
            let run_items = if showing_sweeps {
                self.analytics_sweep_items()
            } else {
                self.analytics_run_items()
            };
            let run_list = List::new(run_items)
                .block(Block::default().borders(Borders::ALL).title(format!(
                    "{} ({})",
                    if showing_sweeps {
                        "Sweep Rankings"
                    } else {
                        "Saved Replay Runs"
                    },
                    if showing_sweeps {
                        self.replay_analytics.sweep_entries.len()
                    } else {
                        self.replay_analytics.entries.len()
                    }
                )))
                .highlight_symbol("> ")
                .scroll_padding(1);
            let mut run_state = focused_list_state(
                self.replay_analytics.focus == AnalyticsFocus::Runs
                    || self.replay_analytics.focus == AnalyticsFocus::Sweeps,
                if showing_sweeps {
                    self.replay_analytics.selected_sweep
                } else {
                    self.replay_analytics.selected_run
                },
                if showing_sweeps {
                    self.replay_analytics.sweep_entries.len()
                } else {
                    self.replay_analytics.entries.len()
                },
            );
            frame.render_stateful_widget(run_list, columns[0], &mut run_state);

            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(15),
                    Constraint::Length(8),
                    Constraint::Min(8),
                    Constraint::Length(7),
                ])
                .split(columns[1]);

            let summary_lines = if showing_sweeps {
                self.analytics_sweep_summary_lines()
            } else {
                self.analytics_summary_lines()
            };
            let summary = Paragraph::new(summary_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(if showing_sweeps {
                            "Sweep Summary (saved ranking; read-only)"
                        } else {
                            "Research Summary (saved result; read-only)"
                        }),
                )
                .wrap(Wrap { trim: true });
            frame.render_widget(summary, right[0]);

            if showing_sweeps {
                let chart = Paragraph::new("Equity chart is available for saved replay runs.")
                    .block(Block::default().borders(Borders::ALL).title("Equity Curve"))
                    .wrap(Wrap { trim: true });
                frame.render_widget(chart, right[1]);
            } else {
                self.render_analytics_equity_chart(frame, right[1]);
            }

            let (rows, widths, header, title) = if showing_sweeps {
                (
                    self.analytics_sweep_table_rows(),
                    vec![
                        Constraint::Length(4),
                        Constraint::Length(17),
                        Constraint::Length(11),
                        Constraint::Length(11),
                        Constraint::Length(10),
                        Constraint::Length(7),
                        Constraint::Length(8),
                        Constraint::Length(11),
                    ],
                    Row::new([
                        "Rank", "Run", "Metric", "Net", "DD%", "Trades", "Near", "Robust",
                    ]),
                    format!(
                        "Sweep Candidates | {} | j/k candidate",
                        self.replay_analytics.sweep_metric.label()
                    ),
                )
            } else if self.replay_analytics.focus == AnalyticsFocus::Signals {
                (
                    self.analytics_signal_table_rows(),
                    vec![
                        Constraint::Length(4),
                        Constraint::Length(14),
                        Constraint::Length(8),
                        Constraint::Length(6),
                        Constraint::Length(6),
                        Constraint::Length(7),
                        Constraint::Length(14),
                        Constraint::Length(14),
                    ],
                    Row::new([
                        "Bar", "Time", "Close", "Raw", "Eff", "Pos", "Decision", "Gate",
                    ]),
                    format!(
                        "Signal Diagnostics | filter: {} | g filter",
                        self.analytics_signal_filter_label()
                    ),
                )
            } else {
                (
                    self.analytics_trade_table_rows(),
                    vec![
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
                    ]),
                    format!(
                        "Trade Excursions | sort: {}",
                        self.analytics_trade_sort_label()
                    ),
                )
            };
            let table = Table::new(rows, widths)
                .header(
                    header.style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                )
                .column_spacing(1)
                .block(Block::default().borders(Borders::ALL).title(title));
            frame.render_widget(table, right[2]);

            let selected_lines = if showing_sweeps {
                self.analytics_selected_sweep_lines()
            } else if self.replay_analytics.focus == AnalyticsFocus::Signals {
                self.analytics_selected_signal_lines()
            } else if self.replay_analytics.focus == AnalyticsFocus::Runs {
                self.analytics_selected_run_lines()
            } else {
                self.analytics_selected_trade_lines()
            };
            let selected_title = if showing_sweeps {
                "Selected Sweep Candidate"
            } else if self.replay_analytics.focus == AnalyticsFocus::Signals {
                "Selected Signal Detail"
            } else if self.replay_analytics.focus == AnalyticsFocus::Runs {
                "Selected Run Detail"
            } else {
                "Selected Trade Detail"
            };
            let selected = Paragraph::new(selected_lines)
                .block(Block::default().borders(Borders::ALL).title(selected_title))
                .wrap(Wrap { trim: true });
            frame.render_widget(selected, right[3]);
        }
        #[cfg(not(feature = "replay"))]
        {
            let disabled = Paragraph::new("Replay analytics requires the replay feature.")
                .block(Block::default().borders(Borders::ALL).title("Analytics"));
            frame.render_widget(disabled, area);
        }
    }

    #[cfg(feature = "replay")]
    fn render_analytics_compact(&self, frame: &mut Frame<'_>, area: Rect) {
        let showing_sweeps = self.replay_analytics.focus == AnalyticsFocus::Sweeps;
        let list_focused = self.replay_analytics.focus == AnalyticsFocus::Runs || showing_sweeps;
        let (items, selected, count, list_title) = if showing_sweeps {
            let items = if self.replay_analytics.sweep_entries.is_empty() {
                vec![ListItem::new("No sweep rankings saved. Press r to rescan.")]
            } else {
                self.replay_analytics
                    .sweep_entries
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| {
                        let selected = self.replay_analytics.selected_sweep == index;
                        let marker = if selected {
                            if list_focused { "" } else { "> " }
                        } else {
                            "  "
                        };
                        ListItem::new(vec![
                            Line::from(format!(
                                "{marker}{} [{}]",
                                compact_analytics_label(&entry.document.sweep_id, 15),
                                entry.document.metric.label()
                            )),
                            Line::from(format!(
                                "  {}",
                                compact_analytics_label(&entry.document.name, 24)
                            )),
                            Line::from(format!(
                                "  {} | {} rows",
                                entry.document.fee_scenario,
                                entry.document.rows.len()
                            )),
                        ])
                    })
                    .collect()
            };
            (
                items,
                self.replay_analytics.selected_sweep,
                self.replay_analytics.sweep_entries.len(),
                format!(
                    "Sweep Rankings ({})",
                    self.replay_analytics.sweep_entries.len()
                ),
            )
        } else {
            let items = if self.replay_analytics.entries.is_empty() {
                vec![ListItem::new("No saved replay runs. Press r to rescan.")]
            } else {
                self.replay_analytics
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| {
                        let selected = self.replay_analytics.selected_run == index;
                        let marker = if selected {
                            if list_focused { "" } else { "> " }
                        } else {
                            "  "
                        };
                        let status = match entry.document.status {
                            crate::tradovate::replay::ReplayResultStatus::Completed => "ok",
                            crate::tradovate::replay::ReplayResultStatus::Failed => "failed",
                        };
                        ListItem::new(vec![
                            Line::from(format!(
                                "{marker}{} [{status}]",
                                compact_analytics_label(&entry.document.run_id, 16)
                            )),
                            Line::from(format!(
                                "  {} | {} trades",
                                compact_analytics_label(&entry.document.metadata.contract_name, 8),
                                entry.document.summary.trade_count
                            )),
                            Line::from(format!(
                                "  PnL {}",
                                format_signed_money(Some(entry.document.summary.net_pnl))
                            )),
                        ])
                    })
                    .collect()
            };
            (
                items,
                self.replay_analytics.selected_run,
                self.replay_analytics.entries.len(),
                format!(
                    "Saved Replay Runs ({})",
                    self.replay_analytics.entries.len()
                ),
            )
        };

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
            .split(area);
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(list_title))
            .highlight_symbol("> ")
            .scroll_padding(1);
        let mut list_state = focused_list_state(list_focused, selected, count);
        frame.render_stateful_widget(list, columns[0], &mut list_state);

        let (detail_lines, detail_title) = match self.replay_analytics.focus {
            AnalyticsFocus::Runs => {
                let mut lines = self
                    .replay_analytics
                    .selected_entry()
                    .map(|entry| {
                        let metadata = &entry.document.metadata;
                        let source = if metadata.source_label.trim().is_empty() {
                            metadata.source_kind.as_str()
                        } else {
                            metadata.source_label.as_str()
                        };
                        vec![
                            Line::from(format!("Source: {}", compact_analytics_label(source, 40))),
                            Line::from(format!(
                                "Warmup: {} | Evaluated: {}",
                                metadata.warmup_rows, metadata.evaluation_rows_processed
                            )),
                        ]
                    })
                    .unwrap_or_default();
                lines.extend(self.analytics_summary_lines());
                (lines, "Selected Run Summary".to_string())
            }
            AnalyticsFocus::Trades => {
                let count = self.replay_analytics.sorted_trades().len();
                (
                    self.analytics_selected_trade_lines(),
                    format!(
                        "Trade {}/{} | Up/Down select",
                        if count == 0 {
                            0
                        } else {
                            self.replay_analytics.selected_trade + 1
                        },
                        count
                    ),
                )
            }
            AnalyticsFocus::Signals => {
                let count = self.replay_analytics.filtered_signals().len();
                (
                    self.analytics_selected_signal_lines(),
                    format!(
                        "Signal {}/{} | Up/Down select",
                        if count == 0 {
                            0
                        } else {
                            self.replay_analytics.selected_signal + 1
                        },
                        count
                    ),
                )
            }
            AnalyticsFocus::Sweeps => {
                let count = self
                    .replay_analytics
                    .sweep_ranking
                    .as_ref()
                    .map(|ranking| ranking.rows.len())
                    .unwrap_or_default();
                (
                    self.analytics_selected_sweep_lines(),
                    format!(
                        "Candidate {}/{} | j/k select",
                        if count == 0 {
                            0
                        } else {
                            self.replay_analytics.selected_sweep_row + 1
                        },
                        count
                    ),
                )
            }
        };
        let inner_width = columns[1].width.saturating_sub(2).max(1) as usize;
        let visible_rows = columns[1].height.saturating_sub(2) as usize;
        let total_rows = detail_lines
            .iter()
            .map(|line| line.width().max(1).div_ceil(inner_width))
            .sum::<usize>();
        let max_scroll = total_rows
            .saturating_sub(visible_rows)
            .min(u16::MAX as usize) as u16;
        let scroll = self.replay_analytics.compact_scroll.min(max_scroll);
        let details = Paragraph::new(detail_lines)
            .block(Block::default().borders(Borders::ALL).title(detail_title))
            .wrap(Wrap { trim: true })
            .scroll((scroll, 0));
        frame.render_widget(details, columns[1]);
    }

    #[cfg(feature = "replay")]
    fn render_analytics_equity_chart(&self, frame: &mut Frame<'_>, area: Rect) {
        if self.replay_analytics.equity.is_empty() {
            let message = self
                .replay_analytics
                .equity_load_error
                .as_deref()
                .map(|error| format!("Equity unavailable: {error}"))
                .unwrap_or_else(|| "No equity sidecar saved for this result.".to_string());
            frame.render_widget(
                Paragraph::new(message)
                    .block(Block::default().borders(Borders::ALL).title("Equity Curve"))
                    .wrap(Wrap { trim: true }),
                area,
            );
            return;
        }
        let data = self
            .replay_analytics
            .equity
            .iter()
            .enumerate()
            .map(|(index, point)| (index as f64, point.equity))
            .collect::<Vec<_>>();
        let min = data
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::INFINITY, f64::min);
        let max = data
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let span = (max - min).abs().max(1.0);
        let dataset = Dataset::default()
            .name("equity")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&data);
        let chart = Chart::new(vec![dataset])
            .block(Block::default().borders(Borders::ALL).title(format!(
                "Equity Curve | {} points | {:.2} -> {:.2}",
                data.len(),
                data.first().map(|(_, value)| *value).unwrap_or_default(),
                data.last().map(|(_, value)| *value).unwrap_or_default()
            )))
            .x_axis(
                Axis::default()
                    .bounds([0.0, data.len().saturating_sub(1).max(1) as f64])
                    .labels(vec![Line::from("start"), Line::from("end")]),
            )
            .y_axis(
                Axis::default()
                    .bounds([min - span * 0.05, max + span * 0.05])
                    .labels(vec![
                        Line::from(format!("{min:.0}")),
                        Line::from(format!("{max:.0}")),
                    ]),
            );
        frame.render_widget(chart, area);
    }
}

#[cfg(feature = "replay")]
fn compact_analytics_label(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut compact = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    compact.push('…');
    compact
}
