use super::super::*;

impl App {
    pub(in crate::app) fn handle_analytics_key(&mut self, key: KeyEvent) {
        #[cfg(feature = "replay")]
        {
            match key.code {
                KeyCode::Tab | KeyCode::BackTab => {
                    self.replay_analytics.focus = match self.replay_analytics.focus {
                        AnalyticsFocus::Runs => AnalyticsFocus::Trades,
                        AnalyticsFocus::Trades => AnalyticsFocus::Signals,
                        AnalyticsFocus::Signals => AnalyticsFocus::Sweeps,
                        AnalyticsFocus::Sweeps => AnalyticsFocus::Runs,
                    };
                    if self.replay_analytics.focus == AnalyticsFocus::Signals {
                        self.replay_analytics.load_selected_signals();
                    } else if self.replay_analytics.focus == AnalyticsFocus::Runs {
                        self.replay_analytics.clear_selected_signals();
                    }
                    self.replay_analytics.clamp_selection();
                }
                KeyCode::Up => match self.replay_analytics.focus {
                    AnalyticsFocus::Runs => {
                        self.replay_analytics.selected_run =
                            self.replay_analytics.selected_run.saturating_sub(1);
                        self.replay_analytics.selected_trade = 0;
                        self.replay_analytics.selected_signal = 0;
                        self.replay_analytics.selected_fee_scenario = 0;
                        self.replay_analytics.clear_selected_signals();
                    }
                    AnalyticsFocus::Trades => {
                        self.replay_analytics.selected_trade =
                            self.replay_analytics.selected_trade.saturating_sub(1);
                    }
                    AnalyticsFocus::Signals => {
                        self.replay_analytics.selected_signal =
                            self.replay_analytics.selected_signal.saturating_sub(1);
                    }
                    AnalyticsFocus::Sweeps => {
                        self.replay_analytics.cycle_sweep_source(-1);
                    }
                },
                KeyCode::Down => match self.replay_analytics.focus {
                    AnalyticsFocus::Runs => {
                        if self.replay_analytics.selected_run + 1
                            < self.replay_analytics.entries.len()
                        {
                            self.replay_analytics.selected_run += 1;
                        }
                        self.replay_analytics.selected_trade = 0;
                        self.replay_analytics.selected_signal = 0;
                        self.replay_analytics.selected_fee_scenario = 0;
                        self.replay_analytics.clear_selected_signals();
                    }
                    AnalyticsFocus::Trades => {
                        let trade_count = self.replay_analytics.sorted_trades().len();
                        if self.replay_analytics.selected_trade + 1 < trade_count {
                            self.replay_analytics.selected_trade += 1;
                        }
                    }
                    AnalyticsFocus::Signals => {
                        let signal_count = self.replay_analytics.filtered_signals().len();
                        if self.replay_analytics.selected_signal + 1 < signal_count {
                            self.replay_analytics.selected_signal += 1;
                        }
                    }
                    AnalyticsFocus::Sweeps => {
                        self.replay_analytics.cycle_sweep_source(1);
                    }
                },
                KeyCode::Char('g') | KeyCode::Char('G') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Signals {
                        self.replay_analytics.cycle_signal_filter();
                        self.status = format!(
                            "Analytics signal filter: {}.",
                            self.replay_analytics.signal_filter.label()
                        );
                    }
                }
                KeyCode::Left => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_fee_scenario(-1);
                    } else {
                        self.replay_analytics.cycle_fee_scenario(-1);
                    }
                }
                KeyCode::Right => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_fee_scenario(1);
                    } else {
                        self.replay_analytics.cycle_fee_scenario(1);
                    }
                }
                KeyCode::Char('[') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_source(-1);
                    }
                }
                KeyCode::Char(']') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_source(1);
                    }
                }
                KeyCode::Char('m') | KeyCode::Char('M') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_metric(1);
                        self.status = format!(
                            "Sweep ranking metric: {}.",
                            self.replay_analytics.sweep_metric.label()
                        );
                    }
                }
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_fee_scenario(1);
                        self.status = format!(
                            "Sweep fee scenario: {}.",
                            self.replay_analytics
                                .sweep_fee_scenario
                                .as_deref()
                                .unwrap_or("active")
                        );
                    }
                }
                KeyCode::Char('d') | KeyCode::Char('D') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_filter(true, 1);
                        self.status = format!(
                            "Sweep drawdown filter: {}.",
                            self.replay_analytics
                                .sweep_max_drawdown_pct
                                .map(|value| format!("{value:.1}%"))
                                .unwrap_or_else(|| "none".to_string())
                        );
                    }
                }
                KeyCode::Char('t') | KeyCode::Char('T') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.cycle_sweep_filter(false, 1);
                        self.status = format!(
                            "Sweep minimum closed trades: {}.",
                            self.replay_analytics.sweep_min_closed_trades
                        );
                    }
                }
                KeyCode::Char('j') | KeyCode::Char('J') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        let count = self
                            .replay_analytics
                            .sweep_ranking
                            .as_ref()
                            .map(|document| document.rows.len())
                            .unwrap_or_default();
                        if self.replay_analytics.selected_sweep_row + 1 < count {
                            self.replay_analytics.selected_sweep_row += 1;
                        }
                    }
                }
                KeyCode::Char('k') | KeyCode::Char('K') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Sweeps {
                        self.replay_analytics.selected_sweep_row =
                            self.replay_analytics.selected_sweep_row.saturating_sub(1);
                    }
                }
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    if self.replay_analytics.focus == AnalyticsFocus::Trades {
                        self.replay_analytics.cycle_sort();
                        self.status = format!(
                            "Analytics trade sort: {}.",
                            self.analytics_trade_sort_label()
                        );
                    }
                }
                KeyCode::Char('c') | KeyCode::Char('C') => {
                    if self.replay_analytics.entries.is_empty() {
                        self.status = "No saved replay runs to compare.".to_string();
                    } else {
                        self.replay_analytics.comparison_run =
                            Some(self.replay_analytics.selected_run);
                        let run_id = self
                            .replay_analytics
                            .selected_entry()
                            .map(|entry| entry.document.run_id.as_str())
                            .unwrap_or("selected run");
                        self.status = format!("Analytics comparison baseline set to {run_id}.");
                    }
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    self.replay_analytics.refresh();
                    self.status = format!(
                        "Loaded {} saved replay result(s) from {}.",
                        self.replay_analytics.entries.len(),
                        self.replay_analytics.root.display()
                    );
                }
                _ => {}
            }
        }
        #[cfg(not(feature = "replay"))]
        let _ = key;
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_trade_sort_label(&self) -> &'static str {
        match self.replay_analytics.trade_sort {
            AnalyticsTradeSort::TradeId => "trade id",
            AnalyticsTradeSort::LargestGiveback => "largest giveback",
            AnalyticsTradeSort::LowestCapture => "lowest MFE capture",
        }
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_signal_filter_label(&self) -> &'static str {
        self.replay_analytics.signal_filter.label()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_sweep_items(&self) -> Vec<ListItem<'static>> {
        if self.replay_analytics.sweep_entries.is_empty() {
            return vec![ListItem::new(vec![
                Line::from("No sweep-ranking.json artifacts."),
                Line::from("Run rank-replay-sweep with --output, then press r."),
            ])];
        }
        self.replay_analytics
            .sweep_entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let marker = if self.replay_analytics.selected_sweep == index {
                    ">"
                } else {
                    " "
                };
                ListItem::new(vec![
                    Line::from(format!(
                        "{marker} {} [{}]",
                        entry.document.sweep_id,
                        entry.document.metric.label()
                    )),
                    Line::from(format!("  {}", entry.document.name)),
                    Line::from(format!(
                        "  {} | {} rows",
                        entry.document.fee_scenario,
                        entry.document.rows.len()
                    )),
                ])
            })
            .collect()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_sweep_summary_lines(&self) -> Vec<Line<'static>> {
        let Some(entry) = self.replay_analytics.selected_sweep_entry() else {
            return vec![
                Line::from("No saved sweep ranking selected."),
                Line::from("Create sweep-ranking.json with rank-replay-sweep."),
            ];
        };
        let Some(document) = self.replay_analytics.sweep_ranking.as_ref() else {
            return vec![Line::from("Sweep ranking is unavailable.")];
        };
        let mut lines = vec![
            Line::from(format!("Sweep: {} ({})", document.sweep_id, document.name)),
            Line::from(format!(
                "Source: {}",
                entry
                    .path
                    .parent()
                    .unwrap_or(entry.path.as_path())
                    .display()
            )),
            Line::from(format!(
                "Metric: {} | Fee scenario: {}",
                document.metric.label(),
                document.fee_scenario
            )),
            Line::from(format!(
                "Candidates: {} completed | {} after filters | {} shown",
                document.total_completed_candidates,
                document.filtered_candidates,
                document.rows.len()
            )),
            Line::from(format!(
                "Filters: min trades {} | max drawdown {}",
                document.options.min_closed_trades,
                document
                    .options
                    .max_drawdown_pct
                    .map(|value| format!("{value:.1}%"))
                    .unwrap_or_else(|| "none".to_string())
            )),
            Line::from("m metric | f fee | d drawdown | t trades | [/] sweep"),
        ];
        if let Some(error) = self.replay_analytics.sweep_ranking_error.as_deref() {
            lines.push(Line::from(format!("Re-rank error: {error}")));
        }
        if !self.replay_analytics.sweep_warnings.is_empty() {
            lines.push(Line::from(format!(
                "Warnings: {} ranking artifact issue(s)",
                self.replay_analytics.sweep_warnings.len()
            )));
        }
        lines
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_sweep_table_rows(&self) -> Vec<Row<'static>> {
        let Some(document) = self.replay_analytics.sweep_ranking.as_ref() else {
            return Vec::new();
        };
        document
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let selected = self.replay_analytics.selected_sweep_row == index
                    && self.replay_analytics.focus == AnalyticsFocus::Sweeps;
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else if row.robustness_score.unwrap_or_default() >= 0.0 {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Red)
                };
                Row::new(vec![
                    Cell::from(row.rank.to_string()),
                    Cell::from(row.run_id.clone()),
                    Cell::from(format_sweep_value(row.metric_value)),
                    Cell::from(format_sweep_value(row.net_pnl)),
                    Cell::from(format_sweep_value(row.max_drawdown_pct)),
                    Cell::from(
                        row.closed_trade_count
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "n/a".to_string()),
                    ),
                    Cell::from(row.neighborhood_count.to_string()),
                    Cell::from(format_sweep_value(row.robustness_score)),
                ])
                .style(style)
            })
            .collect()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_selected_sweep_lines(&self) -> Vec<Line<'static>> {
        let Some(row) = self.replay_analytics.selected_sweep_row() else {
            return vec![Line::from("No candidate selected.")];
        };
        let parameters = serde_json::to_string(&row.parameter_values).unwrap_or_default();
        let overrides = serde_json::to_string(&row.overrides).unwrap_or_default();
        vec![
            Line::from(format!(
                "{} | rank {} | {}",
                row.run_id, row.rank, row.fee_scenario
            )),
            Line::from(format!(
                "Metric {} | robustness {} | neighbors {} (median {})",
                format_sweep_value(row.metric_value),
                format_sweep_value(row.robustness_score),
                row.neighborhood_count,
                format_sweep_value(row.neighborhood_median_quality)
            )),
            Line::from(format!(
                "Net {} | Gross {} | Fees {} | Return/required {}",
                format_sweep_value(row.net_pnl),
                format_sweep_value(row.gross_pnl),
                format_sweep_value(row.fees),
                format_sweep_value(row.return_on_required_account_size_pct)
            )),
            Line::from(format!(
                "Drawdown {} ({}) | Required account {} | PF {}",
                format_sweep_value(row.max_drawdown),
                format_sweep_value(row.max_drawdown_pct),
                format_sweep_value(row.required_starting_capital),
                format_sweep_value(row.profit_factor)
            )),
            Line::from(format!(
                "Trades {} | win {} | avg giveback {} | MFE capture {}",
                row.closed_trade_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                format_sweep_value(row.win_rate_pct),
                format_sweep_value(row.average_giveback),
                format_sweep_value(row.average_mfe_capture_ratio)
            )),
            Line::from(format!("Parameters: {parameters}")),
            Line::from(format!("Overrides: {overrides}")),
        ]
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_run_items(&self) -> Vec<ListItem<'static>> {
        if self.replay_analytics.entries.is_empty() {
            return vec![ListItem::new(vec![
                Line::from("No saved replay results."),
                Line::from("Press r to rescan the result directory."),
            ])];
        }
        self.replay_analytics
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let document = &entry.document;
                let marker = if self.replay_analytics.selected_run == index {
                    ">"
                } else {
                    " "
                };
                let status = match document.status {
                    crate::tradovate::replay::ReplayResultStatus::Completed => "ok",
                    crate::tradovate::replay::ReplayResultStatus::Failed => "failed",
                };
                ListItem::new(vec![
                    Line::from(format!("{marker} {} [{status}]", document.run_id)),
                    Line::from(format!(
                        "  {} {} | {} trades | {} signals",
                        document.metadata.contract_name,
                        document.completed_at_utc.format("%Y-%m-%d %H:%M"),
                        document.summary.trade_count,
                        document.metadata.signal_diagnostic_count
                    )),
                    Line::from(format!(
                        "  PnL {} | {}",
                        format_signed_money(Some(document.summary.net_pnl)),
                        entry
                            .path
                            .parent()
                            .unwrap_or(entry.path.as_path())
                            .display()
                    )),
                ])
            })
            .collect()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_summary_lines(&self) -> Vec<Line<'static>> {
        let Some(entry) = self.replay_analytics.selected_entry() else {
            return vec![
                Line::from("No saved replay result selected."),
                Line::from(format!(
                    "Result root: {}",
                    self.replay_analytics.root.display()
                )),
                Line::from("Run a replay, then press r to reload saved results."),
            ];
        };
        let document = &entry.document;
        let scenario = self.replay_analytics.selected_fee_scenario();
        let (scenario_name, scenario_net, scenario_fees) = scenario
            .map(|scenario| {
                (
                    scenario.schedule.name.as_str(),
                    scenario.net_pnl,
                    scenario.fees,
                )
            })
            .unwrap_or((
                document.active_fee_scenario.as_str(),
                document.summary.net_pnl,
                document.summary.fees,
            ));
        let status = match document.status {
            crate::tradovate::replay::ReplayResultStatus::Completed => "completed",
            crate::tradovate::replay::ReplayResultStatus::Failed => "failed",
        };
        let mut lines = vec![
            Line::from(format!("Run: {} ({status})", document.run_id)),
            Line::from(format!(
                "Completed: {}",
                document.completed_at_utc.format("%Y-%m-%d %H:%M:%S UTC")
            )),
            Line::from(format!(
                "Instrument: {} ({})",
                document.metadata.contract_name, document.metadata.contract_id
            )),
            Line::from(format!(
                "Account: {} ({}) | Strategy: {}",
                document.metadata.account_name,
                document.metadata.account_id,
                format!(
                    "{} / {}",
                    document.metadata.strategy.kind.label(),
                    document.metadata.strategy.native_strategy.label()
                )
            )),
            Line::from(format!(
                "Path: {} | Precision: {}",
                document.metadata.signal_source,
                document
                    .summary
                    .precision
                    .first()
                    .map(String::as_str)
                    .unwrap_or("n/a")
            )),
            Line::from(format!(
                "Signals: {} | {} row(s){}",
                if document.metadata.signal_diagnostics_enabled {
                    "captured"
                } else {
                    "not captured"
                },
                document.metadata.signal_diagnostic_count,
                document
                    .artifacts
                    .signals_csv
                    .as_deref()
                    .map(|path| format!(" | {path}"))
                    .unwrap_or_default()
            )),
            Line::from(format!(
                "Scenario: {} | Net PnL: {} | Fees: {:.2}",
                scenario_name,
                format_signed_money(Some(scenario_net)),
                scenario_fees
            )),
            Line::from(format!(
                "Capital: initial {:.2} | Ending equity: {:.2} | Return: {}",
                document.summary.initial_capital,
                document.summary.ending_equity,
                format_percent(document.summary.return_on_initial_capital_pct)
            )),
            Line::from(format!(
                "Gross: {} | Net: {} | Fees: {:.2}",
                format_signed_money(Some(document.summary.gross_pnl)),
                format_signed_money(Some(scenario_net)),
                scenario_fees
            )),
            Line::from(format!(
                "Trades: {} ({} closed) | Win rate: {} | PF: {}",
                document.summary.trade_count,
                document.summary.closed_trade_count,
                format_percent(document.summary.win_rate_pct),
                format_ratio(document.summary.profit_factor)
            )),
            Line::from(format!(
                "Drawdown: {:.2} ({}) | Max open: {:.2}",
                document.summary.max_drawdown,
                format_percent(document.summary.max_drawdown_pct),
                document.summary.max_open_position
            )),
            Line::from(format!(
                "MFE/MAE avg: {} / {} | Giveback avg/median: {} / {}",
                format_money(document.summary.average_mfe_pnl),
                format_money(document.summary.average_mae_pnl),
                format_money(document.summary.average_giveback),
                format_money(document.summary.median_giveback)
            )),
            Line::from(format!(
                "Largest giveback: {} | MFE capture avg: {}",
                format_money(document.summary.largest_giveback),
                format_percent(
                    document
                        .summary
                        .average_mfe_capture_ratio
                        .map(|value| value * 100.0)
                )
            )),
        ];
        if let Some(analysis) = document.margin_analysis.as_ref() {
            lines.push(Line::from(format!(
                "Margin: peak {:.2} | Required {:.2} | Buffer {:.2} | Initial {}",
                analysis.peak_margin_requirement,
                analysis.required_starting_capital,
                analysis.minimum_equity_buffer_over_margin,
                if analysis.initial_capital_sufficient {
                    "sufficient"
                } else {
                    "insufficient"
                }
            )));
        } else {
            lines.push(Line::from("Margin: not configured for this result."));
        }
        if let Some(error) = document.error.as_deref() {
            lines.push(Line::from(format!("Run error: {error}")));
        }
        if document.metadata.post_exit_continuation_horizon_bars > 0 {
            lines.push(Line::from(format!(
                "Post-exit continuation: {} bars | Avg favorable PnL: {} | Largest: {}",
                document.metadata.post_exit_continuation_horizon_bars,
                format_money(document.summary.average_post_exit_favorable_pnl),
                format_money(document.summary.largest_post_exit_favorable_pnl)
            )));
        }
        if let Some(comparison) = self.replay_analytics.comparison_entry() {
            if comparison.path != entry.path {
                lines.push(Line::from(format!(
                    "Compare vs {}: net {} | trades {} | drawdown {}",
                    comparison.document.run_id,
                    format_signed_money(Some(
                        document.summary.net_pnl - comparison.document.summary.net_pnl,
                    )),
                    signed_usize_delta(
                        document.summary.trade_count,
                        comparison.document.summary.trade_count,
                    ),
                    format_signed_money(Some(
                        document.summary.max_drawdown - comparison.document.summary.max_drawdown,
                    ))
                )));
            }
        }
        if !self.replay_analytics.warnings.is_empty() {
            lines.push(Line::from(format!(
                "Warnings: {} invalid result file(s) skipped",
                self.replay_analytics.warnings.len()
            )));
        }
        lines
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_trade_table_rows(&self) -> Vec<Row<'static>> {
        self.replay_analytics
            .sorted_trades()
            .into_iter()
            .enumerate()
            .map(|(index, trade)| {
                let selected = self.replay_analytics.selected_trade == index
                    && self.replay_analytics.focus == AnalyticsFocus::Trades;
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    pnl_style(trade.giveback)
                };
                Row::new(vec![
                    Cell::from(trade.trade_id.to_string()),
                    Cell::from(trade.side.clone()),
                    Cell::from(format_signed_money(Some(trade.realized_gross_pnl))),
                    Cell::from(format_excursion_value(trade.mfe_pnl, trade.mfe_points)),
                    Cell::from(format_excursion_value(trade.mae_pnl, trade.mae_points)),
                    Cell::from(format_money(trade.giveback)),
                    Cell::from(
                        trade
                            .mfe_capture_ratio
                            .map(|value| format!("{:.1}%", value * 100.0))
                            .unwrap_or_else(|| "n/a".to_string()),
                    ),
                    Cell::from(format_money(trade.post_exit_favorable_pnl)),
                    Cell::from(trade.path_precision.clone()),
                ])
                .style(style)
            })
            .collect()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_selected_trade_lines(&self) -> Vec<Line<'static>> {
        let trades = self.replay_analytics.sorted_trades();
        let Some(trade) = trades.get(self.replay_analytics.selected_trade) else {
            return vec![Line::from(
                "No per-trade excursion rows are saved for this result.",
            )];
        };
        vec![
            Line::from(format!(
                "Trade {} {} {} @ {:.4}",
                trade.trade_id, trade.side, trade.quantity, trade.entry_price
            )),
            Line::from(format!(
                "MFE {} ({}pt) @ {:.4} | MAE {} ({}pt) @ {:.4}",
                format_money(trade.mfe_pnl),
                format_points(trade.mfe_points),
                trade.mfe_price,
                format_money(trade.mae_pnl),
                format_points(trade.mae_points),
                trade.mae_price
            )),
            Line::from(format!(
                "Giveback {} | Capture {} | To MFE {}",
                format_money(trade.giveback),
                trade
                    .mfe_capture_ratio
                    .map(|value| format!("{:.1}%", value * 100.0))
                    .unwrap_or_else(|| "n/a".to_string()),
                trade
                    .time_to_mfe_ns
                    .map(format_duration_ns)
                    .unwrap_or_else(|| "n/a".to_string())
            )),
            Line::from(format!(
                "Post-exit favorable: {} | {} bars observed",
                format_money(trade.post_exit_favorable_pnl),
                trade
                    .post_exit_bars_observed
                    .map(|bars| bars.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            )),
        ]
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_signal_table_rows(&self) -> Vec<Row<'static>> {
        self.replay_analytics
            .filtered_signals()
            .into_iter()
            .enumerate()
            .map(|(index, signal)| {
                let selected = self.replay_analytics.selected_signal == index
                    && self.replay_analytics.focus == AnalyticsFocus::Signals;
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else if signal.order_action.is_some() {
                    Style::default().fg(Color::Green)
                } else if !signal.signal.eq_ignore_ascii_case("hold") {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                };
                let position = signal
                    .target_qty
                    .map(|target| format!("{}>{target}", signal.effective_position_qty))
                    .unwrap_or_else(|| signal.effective_position_qty.to_string());
                Row::new(vec![
                    Cell::from(
                        signal
                            .bar_index
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "?".to_string()),
                    ),
                    Cell::from(format_signal_timestamp(signal.bar_timestamp_ns)),
                    Cell::from(format!("{:.4}", signal.bar_close)),
                    Cell::from(signal.raw_signal.clone()),
                    Cell::from(signal.effective_signal.clone()),
                    Cell::from(position),
                    Cell::from(signal.decision.clone()),
                    Cell::from(compact_signal_text(&signal.gate_reason, 24)),
                ])
                .style(style)
            })
            .collect()
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn analytics_selected_signal_lines(&self) -> Vec<Line<'static>> {
        let Some(signal) = self.replay_analytics.selected_signal() else {
            return if let Some(error) = self.replay_analytics.signal_load_error.as_deref() {
                vec![Line::from(format!(
                    "Signal diagnostics unavailable: {error}"
                ))]
            } else if self.replay_analytics.signals.is_empty() {
                vec![Line::from(
                    "No signal diagnostics are saved for this result. Enable replay_signal_diagnostics for future runs.",
                )]
            } else {
                vec![Line::from("No signal row matches the current filter.")]
            };
        };
        let indicator_values = format!(
            "{} prev {:.4}/{:.4} current {:.4}/{:.4}",
            signal.indicator_name,
            signal.previous_fast_indicator.unwrap_or(f64::NAN),
            signal.previous_slow_indicator.unwrap_or(f64::NAN),
            signal.fast_indicator.unwrap_or(f64::NAN),
            signal.slow_indicator.unwrap_or(f64::NAN)
        )
        .replace("NaN", "n/a");
        let position = signal
            .target_qty
            .map(|target| format!("{} -> {target}", signal.effective_position_qty))
            .unwrap_or_else(|| signal.effective_position_qty.to_string());
        vec![
            Line::from(format!(
                "Bar {} @ {} | OHLC {:.4}/{:.4}/{:.4}/{:.4}",
                signal
                    .bar_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                format_signal_timestamp(signal.bar_timestamp_ns),
                signal.bar_open,
                signal.bar_high,
                signal.bar_low,
                signal.bar_close
            )),
            Line::from(format!(
                "{} | path {} | {} | delay {}",
                signal.strategy,
                signal.execution_path,
                signal.signal_timing,
                signal.signal_delay_bars
            )),
            Line::from(format!(
                "Indicator: {indicator_values}{}",
                signal
                    .auxiliary_name
                    .as_ref()
                    .zip(signal.auxiliary_value)
                    .map(|(name, value)| format!(" | {name} {:.4}", value))
                    .unwrap_or_default()
            )),
            Line::from(format!(
                "Signal: {} | raw {} | effective {} | position {} (actual {})",
                signal.signal,
                signal.raw_signal,
                signal.effective_signal,
                position,
                signal.current_position_qty
            )),
            Line::from(format!(
                "Decision: {} | order {} {} | gate: {}",
                signal.decision,
                signal.order_action.as_deref().unwrap_or("n/a"),
                signal
                    .order_qty
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "".to_string()),
                signal.gate_reason
            )),
            Line::from(format!(
                "Hold reason: {} | fingerprint: {}",
                signal.hold_reason.as_deref().unwrap_or("n/a"),
                signal
                    .fingerprint
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            )),
            Line::from(format!("Detail: {}", signal.strategy_detail)),
        ]
    }
}

#[cfg(feature = "replay")]
fn signed_usize_delta(left: usize, right: usize) -> String {
    if left >= right {
        format!("+{}", left - right)
    } else {
        format!("-{}", right - left)
    }
}

#[cfg(feature = "replay")]
fn format_duration_ns(value: i64) -> String {
    let seconds = value.max(0) as f64 / 1_000_000_000.0;
    if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{seconds:.1}s")
    }
}

#[cfg(feature = "replay")]
fn format_excursion_value(pnl: Option<f64>, points: f64) -> String {
    pnl.map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| format_points(points))
}

#[cfg(feature = "replay")]
fn format_points(value: f64) -> String {
    format!("{value:.2}pt")
}

#[cfg(feature = "replay")]
fn format_signal_timestamp(timestamp_ns: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp(
        timestamp_ns.div_euclid(1_000_000_000),
        timestamp_ns.rem_euclid(1_000_000_000) as u32,
    )
    .map(|timestamp| timestamp.format("%m-%d %H:%M:%S").to_string())
    .unwrap_or_else(|| timestamp_ns.to_string())
}

#[cfg(feature = "replay")]
fn compact_signal_text(value: &str, max_chars: usize) -> String {
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

#[cfg(feature = "replay")]
fn format_sweep_value(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}
