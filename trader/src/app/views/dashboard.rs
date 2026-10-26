use super::super::*;

impl App {
    fn dashboard_status_label(&self) -> String {
        if !self.session_kind.is_ninjatrader_replay()
            || !self.status.starts_with("Waiting for Replay quotes")
        {
            return self.status.clone();
        }

        let contract = self
            .market
            .contract_name
            .as_deref()
            .or_else(|| {
                self.status
                    .strip_prefix("Waiting for Replay quotes for ")
                    .and_then(|status| status.split_once(';').map(|(contract, _)| contract))
            })
            .unwrap_or("the selected contract");
        let clock_state = if self.hosted_replay_clock.is_none() {
            "provider clock has not advanced yet"
        } else if self.strategy_runtime.armed
            && self
                .hosted_replay_speed
                .is_some_and(|speed| speed.percent() > 0)
        {
            "provider clock is advancing"
        } else {
            "provider clock is paused"
        };

        format!("Waiting for Replay quotes for {contract}; {clock_state}.")
    }

    pub(in crate::app) fn dashboard_summary_lines(&self) -> Vec<Line<'static>> {
        let strategy_status = if self.session_kind.is_ninjatrader_replay() {
            if !self.hosted_replay_stream_connected {
                "Replay stream: disconnected".to_string()
            } else if self.ninjatrader_replay_awaiting_account() {
                "Replay: waiting for account".to_string()
            } else if !self.capabilities.automated_orders {
                "Replay orders: unavailable".to_string()
            } else if !self.strategy_runtime.last_summary.is_empty() {
                format!("Replay strategy: {}", self.strategy_runtime.last_summary)
            } else if self.strategy_runtime.armed {
                "Replay strategy: armed".to_string()
            } else {
                "Replay strategy: disarmed".to_string()
            }
        } else {
            format!("Strategy Status: {}", self.strategy_runtime_summary())
        };
        let provider_line = if self.session_kind.is_ninjatrader_replay() {
            "Provider: NinjaTrader Hosted Replay".to_string()
        } else {
            format!("Broker: {}", self.selected_broker.label())
        };
        let mut lines = vec![
            Line::from(format!("Status: {}", self.dashboard_status_label())),
            Line::from(provider_line),
        ];
        if self.session_kind.is_ninjatrader_replay() {
            // Keep order readiness visible in compact terminals, above the
            // longer clock and selection details that may wrap.
            lines.push(Line::from(strategy_status.clone()));
        }
        lines.extend([
            Line::from(format!(
                "{}: {}",
                if self.session_kind.is_ninjatrader_replay() {
                    "Strategy selection"
                } else {
                    "Strategy"
                },
                self.strategy.summary_label()
            )),
            Line::from(format!(
                "Engine strategy: {}",
                if self.session_kind.is_ninjatrader_replay() && !self.hosted_replay_stream_connected
                {
                    "inactive (Replay disconnected)".to_string()
                } else {
                    self.engine_strategy_label()
                }
            )),
            Line::from(format!(
                "Mode: {}",
                if self.session_kind.is_ninjatrader_replay() {
                    "Hosted Replay"
                } else {
                    self.session_kind.label()
                }
            )),
        ]);
        if self.session_kind.is_ninjatrader_replay() {
            lines.push(Line::from(format!(
                "Provider clock: {}",
                if !self.hosted_replay_stream_connected {
                    "disconnected".to_string()
                } else {
                    self.hosted_replay_clock
                        .map(|clock| clock.to_rfc3339())
                        .unwrap_or_else(|| "waiting for first clock update".to_string())
                }
            )));
            let speed = self
                .hosted_replay_speed
                .map(NinjaTraderReplaySpeed::percent)
                .unwrap_or(0);
            lines.push(Line::from(if !self.hosted_replay_stream_connected {
                "Provider speed: stopped".to_string()
            } else if speed == 0 {
                "Provider speed: paused (0%)".to_string()
            } else {
                format!("Provider speed: {speed}%")
            }));
            lines.push(Line::from(if !self.hosted_replay_stream_connected {
                "Replay is stopped; configure a stream in Replay Setup to continue.".to_string()
            } else if self.strategy_runtime.armed {
                "Clock advances while the execution strategy is armed.".to_string()
            } else {
                "Clock paused; choose on F3 and arm to advance.".to_string()
            }));
            if self.hosted_replay_stream_connected {
                lines.push(Line::from("x stops this app's Replay stream connection."));
            }
        } else if self.session_kind == SessionKind::Replay {
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
        if !self.session_kind.is_ninjatrader_replay() {
            lines.push(Line::from(strategy_status));
        }
        lines.extend([
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
        if self.session_kind.is_ninjatrader_replay() {
            keys.extend(["[/]", "0", "x stop"]);
        } else if self.session_kind == SessionKind::Replay {
            keys.extend(["[/]", "0"]);
        }
        let hotkeys = if self.session_kind.is_ninjatrader_replay() {
            format!(
                "Order {} {} | Keys {} (provider {}%)",
                self.base_config.order_qty,
                self.base_config.time_in_force,
                keys.join(" "),
                self.hosted_replay_speed
                    .map(NinjaTraderReplaySpeed::percent)
                    .unwrap_or(0),
            )
        } else if self.session_kind == SessionKind::Replay {
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
