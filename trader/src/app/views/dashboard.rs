use super::super::*;

impl App {
    pub(in crate::app) fn dashboard_summary_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Status: {}", self.status)),
            Line::from(format!("Broker: {}", self.selected_broker.label())),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!("Mode: {}", self.session_kind.label())),
        ];
        if self.session_kind == SessionKind::Replay {
            lines.push(Line::from(format!(
                "Replay Speed: {}",
                self.replay_speed.label()
            )));
            lines.push(Line::from(format!(
                "Replay Ledger: {} fill(s) | gross {:+.2} | {} | schema v{}",
                self.replay_execution_ledger.fill_count,
                self.replay_execution_ledger.gross_realized_pnl,
                self.replay_execution_ledger.latency_model.label(),
                self.replay_execution_ledger.schema_version
            )));
            lines.push(Line::from(format!(
                "Replay Fill Model: {}",
                self.replay_execution_ledger.fill_model.label()
            )));
            lines.push(Line::from(format!(
                "Replay Protection: {}",
                self.replay_execution_ledger.bar_protection_policy.label()
            )));
            if let Some(window) = self.market.replay_window.as_ref() {
                lines.push(Line::from(format!(
                    "Replay Window: {} | {}",
                    window.preset, window.input_timezone
                )));
                lines.push(Line::from(format!(
                    "Replay Range: {}",
                    window.local_range_label()
                )));
                lines.push(Line::from(format!(
                    "Replay Rows: warmup {} | evaluation {}/{}",
                    window.warmup_rows,
                    window.evaluation_rows_processed,
                    window.evaluation_rows_total
                )));
            }
        }
        lines.extend([
            Line::from(format!(
                "Strategy Status: {}",
                self.strategy_runtime_summary()
            )),
            Line::from(format!("Auth Mode: {}", self.form.auth_mode.label())),
            Line::from(format!("Log Mode: {}", self.form.log_mode.label())),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Selected contract: {name}"),
                None => "Selected contract: none".to_string(),
            }),
            Line::from(format!("Bar Type: {}", self.bar_type.label())),
        ]);
        if self.candle_mode_controls_visible() {
            lines.push(Line::from(format!("Candles: {}", self.candle_mode.label())));
        }
        lines.extend([
            Line::from(format!("Session Gate: {}", self.session_gate_summary())),
            Line::from(format!(
                "Chart Overlay: {}",
                self.dashboard_visual_overlay_label()
            )),
            Line::from(format!(
                "REST RTT: {}",
                format_latency_ms(self.latency.rest_rtt_ms)
            )),
            Line::from(format!(
                "Order RTT: {}",
                format_latency_group(
                    self.latency.last_order_ack_ms,
                    self.latency.last_order_seen_ms,
                    self.latency.last_exec_report_ms,
                    self.latency.last_fill_ms,
                )
            )),
            Line::from(format!(
                "Signal RTT: {}",
                format_latency_group(
                    self.latency.last_signal_submit_ms,
                    self.latency.last_signal_seen_ms,
                    self.latency.last_signal_ack_ms,
                    self.latency.last_signal_fill_ms,
                )
            )),
            Line::from(format!(
                "Market Update Age: {}",
                format_age_ms(self.market_update_age_ms())
            )),
        ]);
        lines
    }

    pub(in crate::app) fn stats_lines(&self) -> Vec<Line<'static>> {
        let Some(snapshot) = self.selected_snapshot() else {
            return vec![
                Line::from("No selected account stats."),
                Line::from("Connect and wait for account sync."),
            ];
        };

        let trade_levels = self.displayed_trade_levels();
        let selected_unrealized = self.selected_contract_unrealized_pnl(snapshot);
        let tp_label = if trade_levels.take_profit_projected {
            "TP*"
        } else {
            "TP"
        };
        let sl_label = if trade_levels.stop_price_projected {
            "SL*"
        } else {
            "SL"
        };
        let mut keys = Vec::new();
        if self.manual_order_affordance_visible() {
            keys.push("b/s/c");
        }
        keys.push("v");
        if self.automated_strategy_affordance_visible() {
            keys.push("d");
        }
        if self.session_kind == SessionKind::Replay {
            keys.extend(["[/]", "0"]);
        }
        let hotkeys = if self.session_kind == SessionKind::Replay {
            format!(
                "Order {} {} | Keys {} ({})",
                self.base_config.order_qty,
                self.base_config.time_in_force,
                keys.join(" "),
                self.replay_speed.label(),
            )
        } else {
            format!(
                "Order {} {} | Keys {}",
                self.base_config.order_qty,
                self.base_config.time_in_force,
                keys.join(" ")
            )
        };

        let mut lines = vec![
            Line::from(format!("Acct: {}", snapshot.account_name)),
            Line::from(format!(
                "Bal: {}  Cash: {}",
                format_money(snapshot.balance),
                format_money(snapshot.cash_balance),
            )),
            Line::from(format!(
                "NetLiq: {}  Mgn: {}",
                format_money(snapshot.net_liq),
                format_money(snapshot.intraday_margin),
            )),
            Line::from(vec![
                Span::raw("Account realized: "),
                Span::styled(
                    format_signed_money(snapshot.realized_pnl),
                    pnl_style(snapshot.realized_pnl),
                ),
                Span::raw("  Selected unreal: "),
                Span::styled(
                    format_signed_money(selected_unrealized),
                    pnl_style(selected_unrealized),
                ),
            ]),
            Line::from(format!(
                "Selected position: {}",
                format_quantity(snapshot.market_position_qty),
            )),
            Line::from(format!(
                "Entry: {}  {tp_label}: {}  {sl_label}: {}",
                format_money(trade_levels.entry_price.or(snapshot.market_entry_price)),
                format_money(
                    trade_levels
                        .take_profit_price
                        .or(snapshot.selected_contract_take_profit_price)
                ),
                format_money(
                    trade_levels
                        .stop_price
                        .or(snapshot.selected_contract_stop_price)
                ),
            )),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Contract: {name}"),
                None => "Contract: none".to_string(),
            }),
            Line::from(hotkeys),
        ];
        if let Some(history) = self.engine_history.as_ref() {
            let net = history.realized_pnl + history.unrealized_pnl;
            lines.insert(
                5,
                Line::from(vec![
                    Span::raw("Engine run: "),
                    Span::styled(format_signed_money(Some(net)), pnl_style(Some(net))),
                    Span::raw(format!("  fills {}", history.fills.len())),
                ]),
            );
        }
        lines
    }
}
