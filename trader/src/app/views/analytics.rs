use super::super::*;

impl App {
    pub(in crate::app) fn handle_analytics_key(&mut self, key: KeyEvent) {
        #[cfg(feature = "replay")]
        {
            match key.code {
                KeyCode::Tab | KeyCode::BackTab => {
                    self.replay_analytics.focus = match self.replay_analytics.focus {
                        AnalyticsFocus::Runs => AnalyticsFocus::Trades,
                        AnalyticsFocus::Trades => AnalyticsFocus::Runs,
                    };
                    self.replay_analytics.clamp_selection();
                }
                KeyCode::Up => match self.replay_analytics.focus {
                    AnalyticsFocus::Runs => {
                        self.replay_analytics.selected_run =
                            self.replay_analytics.selected_run.saturating_sub(1);
                        self.replay_analytics.selected_trade = 0;
                        self.replay_analytics.selected_fee_scenario = 0;
                    }
                    AnalyticsFocus::Trades => {
                        self.replay_analytics.selected_trade =
                            self.replay_analytics.selected_trade.saturating_sub(1);
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
                        self.replay_analytics.selected_fee_scenario = 0;
                    }
                    AnalyticsFocus::Trades => {
                        let trade_count = self.replay_analytics.sorted_trades().len();
                        if self.replay_analytics.selected_trade + 1 < trade_count {
                            self.replay_analytics.selected_trade += 1;
                        }
                    }
                },
                KeyCode::Left => self.replay_analytics.cycle_fee_scenario(-1),
                KeyCode::Right => self.replay_analytics.cycle_fee_scenario(1),
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    self.replay_analytics.cycle_sort();
                    self.status = format!(
                        "Analytics trade sort: {}.",
                        self.analytics_trade_sort_label()
                    );
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
                        "  {} {} | {} trades",
                        document.metadata.contract_name,
                        document.completed_at_utc.format("%Y-%m-%d %H:%M"),
                        document.summary.trade_count
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
