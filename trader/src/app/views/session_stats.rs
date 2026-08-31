use super::super::*;

impl App {
    pub(in crate::app) fn session_stats_overview_lines(&self) -> Vec<Line<'static>> {
        if let Some(history) = self.engine_history.as_ref() {
            return vec![
                Line::from("Tracking: broker-attributed engine orders and fills"),
                Line::from(format!("Run: {}", history.run_id)),
                Line::from(format!(
                    "Started: {}",
                    format_session_stats_timestamp(history.started_at_utc, true)
                )),
                Line::from(format!(
                    "Account: {} ({})",
                    history.account_name, history.account_id
                )),
                Line::from(format!(
                    "Contract: {} ({})",
                    history.contract_name, history.contract_id
                )),
                Line::from(format!("Broker fills: {}", history.fills.len())),
                Line::from("Only this run's tagged strategy/direct orders are included."),
                Line::from("Manual Web UI and other-engine orders are excluded."),
            ];
        }
        let selected_account = self
            .accounts
            .get(self.selected_account)
            .map(|account| account.name.clone())
            .unwrap_or_else(|| "none".to_string());
        let selected_source = self
            .selected_session_stats()
            .map(|stats| stats.source.label().to_string())
            .unwrap_or_else(|| "n/a".to_string());

        let mut lines = vec![
            Line::from(format!("Tracking: {}", self.session_stats_enabled_label())),
            Line::from(format!(
                "Tracker Start: {}",
                format_session_stats_timestamp(self.session_stats.started_at_utc, true)
            )),
            Line::from(format!(
                "Tracked Accounts: {}",
                self.session_stats.tracked_accounts()
            )),
            Line::from(format!(
                "Recorded Events: {}",
                self.session_stats.total_event_count()
            )),
            Line::from(format!("Selected Account: {selected_account}")),
            Line::from(format!("Selected Source: {selected_source}")),
            Line::from(format!(
                "Fees: {}",
                if self.session_stats_show_fees {
                    "shown"
                } else {
                    "hidden"
                }
            )),
        ];

        if self.session_stats.enabled {
            lines.push(Line::from(
                "Up/Down switches accounts, Enter re-syncs, f toggles fee rows.",
            ));
            lines.push(Line::from(
                "F5/Ctrl+S saves both the normal log and the balance-delta session stats.",
            ));
        } else {
            lines.push(Line::from(
                "Enable with `session_stats_enabled = true` or TRADER_SESSION_STATS_ENABLED=1.",
            ));
        }

        lines
    }

    pub(in crate::app) fn selected_session_stats_lines(&self) -> Vec<Line<'static>> {
        if let Some(history) = self.engine_history.as_ref() {
            let net_pnl = history.realized_pnl + history.unrealized_pnl;
            let elapsed_hours = engine_history_elapsed_hours(history);
            let net_pnl_per_hour = elapsed_hours.map(|hours| net_pnl / hours);
            let trade_pnl_per_hour =
                elapsed_hours.map(|hours| (history.realized_pnl + history.fees) / hours);
            let fees_per_hour = elapsed_hours.map(|hours| -history.fees / hours);
            return vec![
                Line::from(format!(
                    "Engine run: {} on {}",
                    history.run_id, history.contract_name
                )),
                Line::from(format!(
                    "Position: {}  Entry: {}",
                    history.position_qty,
                    format_money(history.average_entry_price)
                )),
                Line::from(vec![
                    Span::raw("Realized: "),
                    Span::styled(
                        format_signed_money(Some(history.realized_pnl)),
                        pnl_style(Some(history.realized_pnl)),
                    ),
                    Span::raw("  Unrealized: "),
                    Span::styled(
                        format_signed_money(Some(history.unrealized_pnl)),
                        pnl_style(Some(history.unrealized_pnl)),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("Net: "),
                    Span::styled(
                        format_signed_money(Some(net_pnl)),
                        net_pnl_style(Some(net_pnl)),
                    ),
                    Span::raw("  Fees: "),
                    Span::styled(
                        format_signed_money(Some(-history.fees)),
                        pnl_style(Some(-history.fees)),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("Trade: "),
                    Span::styled(
                        format_signed_money(Some(history.realized_pnl + history.fees)),
                        pnl_style(Some(history.realized_pnl + history.fees)),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("PnL/H: Net "),
                    Span::styled(
                        format_money_per_hour(net_pnl_per_hour),
                        net_pnl_style(net_pnl_per_hour),
                    ),
                    Span::raw("  Trade "),
                    Span::styled(
                        format_money_per_hour(trade_pnl_per_hour),
                        pnl_style(trade_pnl_per_hour),
                    ),
                    Span::raw("  Fees "),
                    Span::styled(
                        format_money_per_hour(fees_per_hour),
                        pnl_style(fees_per_hour),
                    ),
                ]),
                Line::from(format!(
                    "Fills: {}  Wins: {}  Losses: {}",
                    history.fills.len(),
                    history.wins,
                    history.losses
                )),
                Line::from("Source: broker orders → strategy links → fills"),
            ];
        }
        if !self.session_stats.enabled {
            return vec![
                Line::from("Session stats tracking is disabled."),
                Line::from("Turn it on in config or via TRADER_SESSION_STATS_ENABLED=1."),
            ];
        }

        let Some(stats) = self.selected_session_stats() else {
            return vec![
                Line::from("No tracked stats for the selected account yet."),
                Line::from("Wait for account snapshots or switch to an account with activity."),
            ];
        };

        vec![
            Line::from(format!(
                "Account: {} ({})",
                stats.account_name, stats.account_id
            )),
            Line::from(format!(
                "Source: {}  Samples: {}  Moves: {}",
                stats.source.label(),
                stats.sample_count,
                stats.move_count()
            )),
            Line::from(format!(
                "Started: {}",
                format_session_stats_timestamp(stats.started_at_utc, true)
            )),
            Line::from(format!(
                "Updated: {}",
                format_session_stats_timestamp(stats.last_updated_at_utc, true)
            )),
            Line::from(format!(
                "Start: {:.2}  Current: {:.2}",
                stats.start_value, stats.current_value
            )),
            Line::from(vec![
                Span::raw("Session PnL: "),
                Span::styled(
                    format_signed_money(Some(stats.session_pnl())),
                    net_pnl_style(Some(stats.session_pnl())),
                ),
            ]),
            self.session_stats_trade_fee_summary_line(stats),
            Line::from(vec![
                Span::raw("PnL/H: Net "),
                Span::styled(
                    format_money_per_hour(stats.session_pnl_per_hour()),
                    net_pnl_style(stats.session_pnl_per_hour()),
                ),
                Span::raw("  Trade "),
                Span::styled(
                    format_money_per_hour(stats.trade_pnl_per_hour()),
                    pnl_style(stats.trade_pnl_per_hour()),
                ),
            ]),
            self.session_stats_move_summary_line(stats),
            Line::from(vec![
                Span::raw("Side PnL: Long "),
                Span::styled(
                    format_signed_money(Some(stats.long_side.pnl)),
                    pnl_style(Some(stats.long_side.pnl)),
                ),
                Span::raw(format!(
                    " ({}/{})  Short ",
                    stats.long_side.wins, stats.long_side.losses
                )),
                Span::styled(
                    format_signed_money(Some(stats.short_side.pnl)),
                    pnl_style(Some(stats.short_side.pnl)),
                ),
                Span::raw(format!(
                    " ({}/{})",
                    stats.short_side.wins, stats.short_side.losses
                )),
            ]),
            Line::from(format!(
                "Win Rate: {}  Profit Factor: {}",
                format_percent(stats.win_rate()),
                format_ratio(stats.profit_factor())
            )),
            Line::from(vec![
                Span::raw("Avg Win: "),
                Span::styled(
                    format_signed_money(stats.avg_win()),
                    pnl_style(stats.avg_win()),
                ),
                Span::raw("  Max Win: "),
                Span::styled(format_signed_money(stats.max_win), pnl_style(stats.max_win)),
            ]),
            Line::from(vec![
                Span::raw("Avg Loss: "),
                Span::styled(
                    format_signed_money(stats.avg_loss_signed()),
                    pnl_style(stats.avg_loss_signed()),
                ),
                Span::raw("  Max Loss: "),
                Span::styled(
                    format_signed_money(stats.max_loss_signed()),
                    pnl_style(stats.max_loss_signed()),
                ),
            ]),
            Line::from(vec![
                Span::raw("Last Raw Delta: "),
                Span::styled(
                    format_signed_money(stats.last_delta),
                    pnl_style(stats.last_delta),
                ),
                Span::raw("  Trade: "),
                Span::styled(
                    format_signed_money(stats.last_trade_delta),
                    pnl_style(stats.last_trade_delta),
                ),
            ]),
        ]
    }

    pub(in crate::app) fn session_stats_event_lines(&self, limit: usize) -> Vec<Line<'static>> {
        if let Some(history) = self.engine_history.as_ref() {
            if limit == 0 {
                return Vec::new();
            }
            if history.fills.is_empty() {
                return vec![Line::from(
                    "No broker-attributed fills for this engine run yet.",
                )];
            }

            let hourly_lines = engine_history_hourly_lines(history);
            // Keep at least one recent fill visible even when a small panel
            // cannot fit every active hour. The hourly header is retained and
            // the most recent hourly rows win when the block must be trimmed.
            let hourly_capacity = limit.saturating_sub(2);
            let mut lines = trim_engine_history_hourly_lines(hourly_lines, hourly_capacity);
            if !lines.is_empty() && lines.len() < limit {
                lines.push(Line::from(""));
            }
            let fill_limit = limit.saturating_sub(lines.len());
            let fill_lines = history.fills.iter().rev().take(fill_limit).map(|fill| {
                let side = match fill.side {
                    TradeMarkerSide::Buy => "BUY",
                    TradeMarkerSide::Sell => "SELL",
                };
                Line::from(vec![
                    Span::styled(side, engine_fill_side_style(fill.side)),
                    Span::raw(format!(" {} @ {:.2} | pnl ", fill.qty, fill.price)),
                    Span::styled(
                        format_signed_money(Some(fill.realized_pnl)),
                        pnl_style(Some(fill.realized_pnl)),
                    ),
                    Span::raw(format!(" | fill {} order {}", fill.fill_id, fill.order_id)),
                ])
            });
            lines.extend(fill_lines);
            lines.truncate(limit);
            return lines;
        }
        if !self.session_stats.enabled {
            return vec![Line::from(
                "Tracking is disabled, so no balance-delta events were recorded.",
            )];
        }

        let Some(stats) = self.selected_session_stats() else {
            return vec![Line::from(
                "No balance-delta events yet for the selected account.",
            )];
        };

        if stats.events.is_empty() {
            return vec![
                Line::from("No delta events yet."),
                Line::from("The first balance sample becomes the session baseline."),
            ];
        }

        let mut lines = hourly_session_stats_lines(stats, self.session_stats_show_fees);
        let event_limit = limit.saturating_sub(lines.len()).max(1);
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        stats
            .events
            .iter()
            .rev()
            .filter(|event| {
                self.session_stats_show_fees || event.kind != SessionBalanceEventKind::Fee
            })
            .take(event_limit)
            .map(|event| {
                if self.session_stats_show_fees {
                    self.session_stats_fee_detail_event_line(event)
                } else {
                    self.session_stats_trade_only_event_line(event)
                }
            })
            .for_each(|line| lines.push(line));
        lines.truncate(limit);
        lines
    }

    fn session_stats_trade_fee_summary_line(&self, stats: &AccountSessionStats) -> Line<'static> {
        if self.session_stats_show_fees {
            Line::from(vec![
                Span::raw("Trade PnL Ex Fees: "),
                Span::styled(
                    format_signed_money(Some(stats.trade_pnl_ex_fees())),
                    pnl_style(Some(stats.trade_pnl_ex_fees())),
                ),
                Span::raw("  Fees: "),
                Span::styled(
                    format_signed_money(Some(stats.total_fees)),
                    pnl_style(Some(stats.total_fees)),
                ),
                Span::raw(format!(" ({})", stats.fee_events)),
            ])
        } else {
            Line::from(vec![
                Span::raw("Trade PnL Ex Fees: "),
                Span::styled(
                    format_signed_money(Some(stats.trade_pnl_ex_fees())),
                    pnl_style(Some(stats.trade_pnl_ex_fees())),
                ),
                Span::raw("  Fees hidden"),
            ])
        }
    }

    fn session_stats_move_summary_line(&self, stats: &AccountSessionStats) -> Line<'static> {
        if self.session_stats_show_fees {
            Line::from(format!(
                "Wins: {}  Losses: {}  Flats: {}  Fee Events: {}",
                stats.wins, stats.losses, stats.flat_moves, stats.fee_events
            ))
        } else {
            Line::from(format!(
                "Wins: {}  Losses: {}  Flats: {}",
                stats.wins, stats.losses, stats.flat_moves
            ))
        }
    }

    fn session_stats_fee_detail_event_line(&self, event: &SessionBalanceEvent) -> Line<'static> {
        let delta_style = session_stats_pnl_style(Some(event.delta));
        let mut spans = vec![
            Span::raw(format!(
                "{} {} ",
                format_session_stats_timestamp(event.recorded_at_utc, false),
                event.source.label(),
            )),
            Span::styled(
                event.side.label().to_string(),
                session_trade_side_style(event.side),
            ),
            Span::raw(format!(
                " pos {} {:.2} -> ",
                format_session_position_transition(
                    event.previous_position_side,
                    event.current_position_side
                ),
                event.previous_value
            )),
            Span::styled(format!("{:.2}", event.current_value), delta_style),
            Span::raw(" ("),
            Span::styled(format_signed_money(Some(event.delta)), delta_style),
            Span::raw(")"),
            Span::raw(format!(" {}", event.kind.label())),
        ];
        if event.fee_delta.abs() >= SESSION_STATS_DELTA_EPSILON {
            spans.extend([
                Span::raw(" trade "),
                Span::styled(
                    format_signed_money(Some(event.trade_delta)),
                    session_stats_pnl_style(Some(event.trade_delta)),
                ),
                Span::raw(" fees "),
                Span::styled(
                    format_signed_money(Some(event.fee_delta)),
                    session_stats_pnl_style(Some(event.fee_delta)),
                ),
            ]);
        }
        Line::from(spans)
    }

    fn session_stats_trade_only_event_line(&self, event: &SessionBalanceEvent) -> Line<'static> {
        let trade_style = session_stats_pnl_style(Some(event.trade_delta));
        Line::from(vec![
            Span::raw(format!(
                "{} {} ",
                format_session_stats_timestamp(event.recorded_at_utc, false),
                event.source.label(),
            )),
            Span::styled(
                event.side.label().to_string(),
                session_trade_side_style(event.side),
            ),
            Span::raw(format!(
                " pos {} {:.2} -> {:.2} trade ",
                format_session_position_transition(
                    event.previous_position_side,
                    event.current_position_side
                ),
                event.previous_value,
                event.current_value
            )),
            Span::styled(format_signed_money(Some(event.trade_delta)), trade_style),
        ])
    }
}

fn session_trade_side_style(side: SessionTradeSide) -> Style {
    match side {
        SessionTradeSide::Long => Style::default().fg(Color::Blue),
        SessionTradeSide::Short => Style::default().fg(Color::Magenta),
        SessionTradeSide::Flat | SessionTradeSide::Unknown => Style::default().fg(Color::Gray),
    }
}

fn session_stats_pnl_style(value: Option<f64>) -> Style {
    match value {
        Some(value) if value > 0.0 => Style::default().fg(Color::Green),
        Some(value) if value < 0.0 => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::Gray),
    }
}

fn net_pnl_style(value: Option<f64>) -> Style {
    pnl_style(value).add_modifier(Modifier::BOLD)
}

fn engine_fill_side_style(side: TradeMarkerSide) -> Style {
    match side {
        TradeMarkerSide::Buy => Style::default().fg(Color::Cyan),
        TradeMarkerSide::Sell => Style::default().fg(Color::Magenta),
    }
}

fn engine_history_fill_hour(ts_ns: i64) -> usize {
    chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(ts_ns)
        .with_timezone(&chrono::Local)
        .format("%H")
        .to_string()
        .parse::<usize>()
        .unwrap_or_default()
        .min(23)
}

fn engine_history_elapsed_hours(history: &EngineHistorySnapshot) -> Option<f64> {
    let latest_fill_at = history
        .fills
        .iter()
        .map(|fill| chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(fill.ts_ns))
        .max();
    let end_at = history
        .updated_at_utc
        .or(latest_fill_at)
        .unwrap_or_else(chrono::Utc::now);
    let elapsed_ms = end_at
        .signed_duration_since(history.started_at_utc)
        .num_milliseconds();
    (elapsed_ms > 0).then_some(elapsed_ms as f64 / 3_600_000.0)
}

fn engine_history_hourly_lines(history: &EngineHistorySnapshot) -> Vec<Line<'static>> {
    let mut buckets = [(0.0_f64, 0_usize); 24];
    for fill in &history.fills {
        let bucket = &mut buckets[engine_history_fill_hour(fill.ts_ns)];
        bucket.0 += fill.realized_pnl;
        bucket.1 += 1;
    }

    let mut lines = vec![Line::from("Hourly Trade PnL/H (local, net of fees)")];
    lines.extend(
        buckets
            .into_iter()
            .enumerate()
            .filter(|(_, (_, fills))| *fills > 0)
            .map(|(hour, (pnl, fills))| {
                Line::from(vec![
                    Span::raw(format!("{hour:02}:00 ")),
                    Span::styled(format_money_per_hour(Some(pnl)), pnl_style(Some(pnl))),
                    Span::raw(format!(" ({fills} fills)")),
                ])
            }),
    );
    lines
}

fn trim_engine_history_hourly_lines(
    mut lines: Vec<Line<'static>>,
    capacity: usize,
) -> Vec<Line<'static>> {
    if capacity == 0 {
        return Vec::new();
    }
    if lines.len() <= capacity {
        return lines;
    }
    if capacity == 1 {
        lines.truncate(1);
        return lines;
    }

    let header = lines.remove(0);
    let rows_to_keep = capacity - 1;
    let row_start = lines.len().saturating_sub(rows_to_keep);
    let mut trimmed = vec![header];
    trimmed.extend(lines.into_iter().skip(row_start));
    trimmed
}

fn hourly_session_stats_lines(stats: &AccountSessionStats, show_fees: bool) -> Vec<Line<'static>> {
    let hourly_stats = stats.hourly_stats();
    if hourly_stats.is_empty() {
        return Vec::new();
    }

    let header = if show_fees {
        "Hourly PnL/H (local): Net | Trade | Fees"
    } else {
        "Hourly PnL/H (local): Net | Trade"
    };
    let mut lines = vec![Line::from(header)];
    lines.extend(hourly_stats.into_iter().map(|(hour, hourly)| {
        let mut spans = vec![
            Span::raw(format!("{hour:02}:00 Net ")),
            Span::styled(
                format_money_per_hour(Some(hourly.raw_pnl)),
                net_pnl_style(Some(hourly.raw_pnl)),
            ),
            Span::raw(" Trade "),
            Span::styled(
                format_money_per_hour(Some(hourly.trade_pnl)),
                pnl_style(Some(hourly.trade_pnl)),
            ),
        ];
        if show_fees {
            spans.extend([
                Span::raw(" Fees "),
                Span::styled(
                    format_money_per_hour(Some(hourly.fees)),
                    pnl_style(Some(hourly.fees)),
                ),
                Span::raw(format!(
                    " | W/L {}/{} | {} events",
                    hourly.wins, hourly.losses, hourly.events
                )),
            ]);
        } else {
            spans.push(Span::raw(format!(
                " | W/L {}/{} | {} trade events",
                hourly.wins,
                hourly.losses,
                hourly.wins + hourly.losses
            )));
        }
        Line::from(spans)
    }));
    lines
}
