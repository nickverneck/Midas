use super::super::*;

impl App {
    pub(in crate::app) fn dashboard_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Status: {}", self.status)),
            Line::from(format!("Broker: {}", self.selected_broker.label())),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!("Mode: {}", self.session_kind.label())),
            Line::from(if self.session_kind == SessionKind::Replay {
                format!("Replay Speed: {}", self.replay_speed.label())
            } else {
                "Replay Speed: inactive".to_string()
            }),
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
        ]
    }

    pub(in crate::app) fn stats_lines(&self) -> Vec<Line<'static>> {
        let Some(snapshot) = self.selected_snapshot() else {
            return vec![
                Line::from("No selected account stats."),
                Line::from("Connect and wait for account sync."),
            ];
        };

        let trade_levels = self.displayed_trade_levels();
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
        let hotkeys = if self.session_kind == SessionKind::Replay {
            format!(
                "Order {} {} | Keys b/s/c/v [/] 0 ({})",
                self.base_config.order_qty,
                self.base_config.time_in_force,
                self.replay_speed.label(),
            )
        } else {
            format!(
                "Order {} {} | Keys b/s/c/v",
                self.base_config.order_qty, self.base_config.time_in_force
            )
        };

        vec![
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
                Span::raw("Session: "),
                Span::styled(
                    format_signed_money(snapshot.realized_pnl),
                    pnl_style(snapshot.realized_pnl),
                ),
                Span::raw("  Unreal: "),
                Span::styled(
                    format_signed_money(snapshot.unrealized_pnl),
                    pnl_style(snapshot.unrealized_pnl),
                ),
            ]),
            Line::from(format!(
                "Open: {}  Sel: {}",
                format_quantity(snapshot.open_position_qty),
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
        ]
    }
}
