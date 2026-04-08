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
        ];

        if self.session_stats.enabled {
            lines.push(Line::from(
                "Up/Down switches accounts, Enter re-syncs the selected account.",
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
            Line::from(format!(
                "Wins: {}  Losses: {}  Flats: {}",
                stats.wins, stats.losses, stats.flat_moves
            )),
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
                Span::raw("Last Delta: "),
                Span::styled(
                    format_signed_money(stats.last_delta),
                    pnl_style(stats.last_delta),
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

        stats
            .events
            .iter()
            .rev()
            .take(limit)
            .map(|event| {
                Line::from(format!(
                    "{} {} {:.2} -> {:.2} ({})",
                    format_session_stats_timestamp(event.recorded_at_utc, false),
                    event.source.label(),
                    event.previous_value,
                    event.current_value,
                    format_signed_money(Some(event.delta)),
                ))
            })
            .collect()
    }
}
