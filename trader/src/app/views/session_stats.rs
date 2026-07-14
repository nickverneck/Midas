use super::super::*;

impl App {
    pub(in crate::app) fn session_stats_overview_lines(&self) -> Vec<Line<'static>> {
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
                    pnl_style(Some(stats.session_pnl())),
                ),
            ]),
            self.session_stats_trade_fee_summary_line(stats),
            Line::from(vec![
                Span::raw("PnL/H: Net "),
                Span::styled(
                    format_money_per_hour(stats.session_pnl_per_hour()),
                    pnl_style(stats.session_pnl_per_hour()),
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
        let delta_style = pnl_style(Some(event.delta));
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
            if event.fee_delta.abs() >= SESSION_STATS_DELTA_EPSILON {
                Span::raw(format!(
                    " trade {} fees {}",
                    format_signed_money(Some(event.trade_delta)),
                    format_signed_money(Some(event.fee_delta))
                ))
            } else {
                Span::raw(String::new())
            },
        ])
    }

    fn session_stats_trade_only_event_line(&self, event: &SessionBalanceEvent) -> Line<'static> {
        let trade_style = pnl_style(Some(event.trade_delta));
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
        SessionTradeSide::Long => Style::default().fg(Color::Cyan),
        SessionTradeSide::Short => Style::default().fg(Color::Magenta),
        SessionTradeSide::Flat | SessionTradeSide::Unknown => Style::default(),
    }
}

fn hourly_session_stats_lines(stats: &AccountSessionStats, show_fees: bool) -> Vec<Line<'static>> {
    let hourly_stats = stats.hourly_stats();
    if hourly_stats.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![Line::from("Hourly Trade PnL/H (local)")];
    lines.extend(hourly_stats.into_iter().map(|(hour, hourly)| {
        Line::from(vec![
            Span::raw(format!("{hour:02}:00 ")),
            Span::styled(
                format_money_per_hour(Some(hourly.trade_pnl)),
                pnl_style(Some(hourly.trade_pnl)),
            ),
            if show_fees {
                Span::raw(format!(
                    " net {} fees {} ({}/{}, {} events)",
                    format_signed_money(Some(hourly.raw_pnl)),
                    format_signed_money(Some(hourly.fees)),
                    hourly.wins,
                    hourly.losses,
                    hourly.events
                ))
            } else {
                Span::raw(format!(
                    " ({}/{}, {} trade events)",
                    hourly.wins,
                    hourly.losses,
                    hourly.wins + hourly.losses
                ))
            },
        ])
    }));
    lines
}
