impl App {
    fn connection_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            styled_line(
                format!("Env: {}", self.form.env.label()),
                self.focus == Focus::Env,
            ),
            styled_line(
                format!("Auth Mode: {}", self.form.auth_mode.label()),
                self.focus == Focus::AuthMode,
            ),
            styled_line(
                format!("Log Mode: {}", self.form.log_mode.label()),
                self.focus == Focus::LogMode,
            ),
            Line::from(""),
            Line::from("Token Sources"),
            styled_line(
                format!("Token Path: {}", self.form.token_path),
                self.focus == Focus::TokenPath,
            ),
            styled_line(
                format!(
                    "Token Override: {}",
                    display_token_override(
                        self.focus == Focus::TokenOverride,
                        &self.form.token_override,
                    )
                ),
                self.focus == Focus::TokenOverride,
            ),
            Line::from(""),
            Line::from("Credential Fallback"),
            styled_line(
                format!("Username: {}", self.form.username),
                self.focus == Focus::Username,
            ),
            styled_line(
                format!("Password: {}", mask(&self.form.password)),
                self.focus == Focus::Password,
            ),
            styled_line(
                format!("App ID: {}", self.form.app_id),
                self.focus == Focus::AppId,
            ),
            styled_line(
                format!("App Version: {}", self.form.app_version),
                self.focus == Focus::AppVersion,
            ),
            styled_line(format!("CID: {}", self.form.cid), self.focus == Focus::Cid),
            styled_line(
                format!("Secret: {}", mask(&self.form.secret)),
                self.focus == Focus::Secret,
            ),
            Line::from(""),
        ];
        lines.push(styled_line(
            "[Enter] Connect / Refresh Session".to_string(),
            self.focus == Focus::Connect,
        ));
        lines.push(styled_line(
            if cfg!(feature = "replay") {
                "[Enter] Replay Mode (local file, skips login)".to_string()
            } else {
                "[Enter] Replay Mode unavailable in this build".to_string()
            },
            self.focus == Focus::ReplayMode,
        ));
        lines
    }
    fn login_notes_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from("1. Token Override is used first when non-empty."),
            Line::from("2. Token File mode reads token_path, then session cache."),
            Line::from("3. Credentials mode requests a fresh access token."),
            Line::from("4. Debug log mode adds submit/seen/ack/fill lifecycle lines."),
            Line::from(""),
            Line::from("Use Up/Down to move between fields."),
            Line::from("Use Left/Right on Env, Auth Mode, or Log Mode."),
            Line::from("Paste a token directly into Token Override when needed."),
            Line::from(if cfg!(feature = "replay") {
                "Replay Mode loads the local tick file and starts on 1 Range bars by default."
            } else {
                "Replay Mode requires a build with `--features replay`."
            }),
        ]
    }

    fn login_status_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Current status: {}", self.status)),
            Line::from(format!("Session Mode: {}", self.session_kind.label())),
            Line::from(format!("Environment REST: {}", self.form.env.rest_url())),
            Line::from(format!("Log Mode: {}", self.form.log_mode.label())),
            Line::from(format!("User WebSocket: {}", self.form.env.user_ws_url())),
            Line::from(format!(
                "Market WebSocket: {}",
                self.form.env.market_ws_url()
            )),
            Line::from(""),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Last contract: {name}"),
                None => "Last contract: none".to_string(),
            }),
        ]
    }

    fn dashboard_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Status: {}", self.status)),
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

    fn strategy_setup_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![styled_line(
            format!("Strategy Type: {}", self.strategy.kind.label()),
            self.focus == Focus::StrategyKind,
        )];

        lines.push(styled_line(
            format!(
                "Order Qty: {}",
                self.strategy_numeric_value(
                    Focus::OrderQty,
                    self.strategy.order_qty.to_string(),
                )
            ),
            self.focus == Focus::OrderQty,
        ));

        if self.strategy.kind == StrategyKind::Native {
            lines.push(styled_line(
                format!("Native Strategy: {}", self.strategy.native_strategy.label()),
                self.focus == Focus::NativeStrategy,
            ));
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    lines.push(styled_line(
                        format!(
                            "HMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::HmaLength,
                                self.strategy.native_hma.hma_length.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Min Angle: {}",
                            self.strategy_numeric_value(
                                Focus::HmaMinAngle,
                                format!("{:.1}", self.strategy.native_hma.min_angle),
                            )
                        ),
                        self.focus == Focus::HmaMinAngle,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Angle Lookback: {}",
                            self.strategy_numeric_value(
                                Focus::HmaAngleLookback,
                                self.strategy.native_hma.angle_lookback.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaAngleLookback,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Bars Required: {}",
                            self.strategy_numeric_value(
                                Focus::HmaBarsRequired,
                                self.strategy.native_hma.bars_required_to_trade.to_string(),
                            )
                        ),
                        self.focus == Focus::HmaBarsRequired,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Longs Only: {}",
                            bool_label(self.strategy.native_hma.longs_only)
                        ),
                        self.focus == Focus::HmaLongsOnly,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Inverted: {}",
                            bool_label(self.strategy.native_hma.inverted)
                        ),
                        self.focus == Focus::HmaInverted,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Take Profit Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTakeProfitTicks,
                                format!("{:.0}", self.strategy.native_hma.take_profit_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaStopLossTicks,
                                format!("{:.0}", self.strategy.native_hma.stop_loss_ticks),
                            )
                        ),
                        self.focus == Focus::HmaStopLossTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trailing Stop: {}",
                            bool_label(self.strategy.native_hma.use_trailing_stop)
                        ),
                        self.focus == Focus::HmaTrailingStop,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Trigger Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTrailTriggerTicks,
                                format!("{:.0}", self.strategy.native_hma.trail_trigger_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::HmaTrailOffsetTicks,
                                format!("{:.0}", self.strategy.native_hma.trail_offset_ticks),
                            )
                        ),
                        self.focus == Focus::HmaTrailOffsetTicks,
                    ));
                }
                NativeStrategyKind::EmaCross => {
                    lines.push(styled_line(
                        format!(
                            "Fast EMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaFastLength,
                                self.strategy.native_ema.fast_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaFastLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Slow EMA Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaSlowLength,
                                self.strategy.native_ema.slow_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaSlowLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Inverted: {}",
                            bool_label(self.strategy.native_ema.inverted)
                        ),
                        self.focus == Focus::EmaInverted,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Take Profit Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTakeProfitTicks,
                                format!("{:.0}", self.strategy.native_ema.take_profit_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTakeProfitTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Stop Loss Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaStopLossTicks,
                                format!("{:.0}", self.strategy.native_ema.stop_loss_ticks),
                            )
                        ),
                        self.focus == Focus::EmaStopLossTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trailing Stop: {}",
                            bool_label(self.strategy.native_ema.use_trailing_stop)
                        ),
                        self.focus == Focus::EmaTrailingStop,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Trigger Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTrailTriggerTicks,
                                format!("{:.0}", self.strategy.native_ema.trail_trigger_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTrailTriggerTicks,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Trail Offset Ticks: {}",
                            self.strategy_numeric_value(
                                Focus::EmaTrailOffsetTicks,
                                format!("{:.0}", self.strategy.native_ema.trail_offset_ticks),
                            )
                        ),
                        self.focus == Focus::EmaTrailOffsetTicks,
                    ));
                }
            }
            lines.push(Line::from(
                "Type numbers or use Left/Right for numeric fields. Backspace edits typed values. Zero TP/SL disables them.",
            ));
        } else if self.strategy.kind == StrategyKind::Lua {
            lines.push(styled_line(
                format!("Lua Input: {}", self.strategy.lua_source_mode.label()),
                self.focus == Focus::LuaSourceMode,
            ));
            if self.strategy.lua_source_mode == LuaSourceMode::File {
                lines.push(styled_line(
                    format!("Lua File: {}", self.strategy.lua_file_path),
                    self.focus == Focus::LuaFilePath,
                ));
                lines.push(Line::from(
                    "Press Enter on Lua File to load it into the preview/editor.",
                ));
            } else {
                lines.push(styled_line(
                    format!(
                        "Lua Editor: {} mode, {} lines",
                        self.strategy.lua_editor.mode().label(),
                        self.strategy.lua_editor.line_count()
                    ),
                    self.focus == Focus::LuaEditor,
                ));
                lines.push(Line::from(
                    "Normal mode: h j k l move, i insert, a append, o new line, x delete.",
                ));
            }
        }

        lines.push(Line::from(""));
        lines.push(styled_line(
            "[Enter] Continue To Dashboard / Arm Strategy".to_string(),
            self.focus == Focus::StrategyContinue,
        ));
        lines
    }

    fn strategy_notes_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from("Backend order: Native Rust > Lua > Machine Learning."),
            Line::from("Native strategies execute on newly closed 1m bars after you arm them."),
            Line::from("The native engine targets the selected contract position directly."),
            Line::from("TP, SL, and trailing stop are configured in ticks and synced natively."),
            Line::from(format!(
                "Native runtime auto-closes {}m before session close and holds until reopen.",
                AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
            )),
            Line::from("Lua can be loaded from file or typed directly in the TUI."),
            Line::from("ML remains selection-only for now."),
            Line::from(""),
            Line::from("Strategy screen controls:"),
            Line::from("Up/Down moves focus. Left/Right edits native params or toggles fields."),
            Line::from(
                "Enter on Continue arms the selected native strategy from the current closed bar.",
            ),
            Line::from(""),
            Line::from("Lua editor controls:"),
            Line::from("Normal: h/j/k/l move, i insert, a append, o open line, x delete."),
            Line::from("Insert: type text, Enter newline, Backspace delete, Esc back to normal."),
        ]
    }

    fn strategy_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Selected: {}", self.strategy.summary_label())),
            Line::from(match self.strategy.kind {
                StrategyKind::Native => format!(
                    "Native Rust {} is active and can submit automated market orders.",
                    self.strategy.native_strategy.label()
                ),
                StrategyKind::Lua => {
                    "Lua strategy source is ready for later execution wiring.".to_string()
                }
                StrategyKind::MachineLearning => {
                    "Machine Learning remains lowest execution priority.".to_string()
                }
            }),
            Line::from(format!("Runtime: {}", self.strategy_runtime_summary())),
        ];
        if self.strategy.kind == StrategyKind::Native {
            lines.push(Line::from(self.strategy.native_summary()));
        } else if self.strategy.kind == StrategyKind::Lua {
            lines.push(Line::from(format!(
                "Lua editor mode: {}",
                self.strategy.lua_editor.mode().label()
            )));
            lines.push(Line::from(format!(
                "Lua text size: {} chars",
                self.strategy.lua_editor.text().len()
            )));
        }
        lines
    }

    fn strategy_detail_title(&self) -> String {
        if self.strategy.kind == StrategyKind::Native {
            return "Native Strategy Detail".to_string();
        }
        if self.strategy.kind != StrategyKind::Lua {
            return "Strategy Detail".to_string();
        }
        match self.strategy.lua_source_mode {
            LuaSourceMode::File => "Lua File Preview".to_string(),
            LuaSourceMode::Editor => format!(
                "Lua Editor [{}] row={} col={}",
                self.strategy.lua_editor.mode().label(),
                self.strategy.lua_editor.cursor().0 + 1,
                self.strategy.lua_editor.cursor().1 + 1
            ),
        }
    }

    fn strategy_detail_lines(&self) -> Vec<Line<'static>> {
        if self.strategy.kind == StrategyKind::Native {
            let mut lines = match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => vec![
                    Line::from("HMA Angle Strategy"),
                    Line::from(format!("Type: {}", NativeStrategyKind::HmaAngle.label())),
                    Line::from(format!(
                        "Params: len={} min_angle={:.1} lookback={} bars_required={}",
                        self.strategy.native_hma.hma_length,
                        self.strategy.native_hma.min_angle,
                        self.strategy.native_hma.angle_lookback,
                        self.strategy.native_hma.bars_required_to_trade
                    )),
                    Line::from(format!(
                        "Flags: longs_only={} inverted={} trailing={}",
                        bool_label(self.strategy.native_hma.longs_only),
                        bool_label(self.strategy.native_hma.inverted),
                        bool_label(self.strategy.native_hma.use_trailing_stop)
                    )),
                    Line::from(format!(
                        "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                        self.strategy.native_hma.take_profit_ticks,
                        self.strategy.native_hma.stop_loss_ticks,
                        self.strategy.native_hma.trail_trigger_ticks,
                        self.strategy.native_hma.trail_offset_ticks,
                    )),
                    Line::from(""),
                    Line::from("Signal logic"),
                    Line::from(
                        "Buy: price crosses above zero-lag HMA with sufficient positive angle.",
                    ),
                    Line::from(
                        "Sell: price crosses below zero-lag HMA with sufficient negative angle.",
                    ),
                    Line::from("Inverted swaps buy/sell decisions before order routing."),
                    Line::from(format!(
                        "Auto-close holds flat {}m before the inferred session close.",
                        AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
                    )),
                ],
                NativeStrategyKind::EmaCross => vec![
                    Line::from("EMA Crossover Strategy"),
                    Line::from(format!("Type: {}", NativeStrategyKind::EmaCross.label())),
                    Line::from(format!(
                        "Params: fast={} slow={}",
                        self.strategy.native_ema.fast_length, self.strategy.native_ema.slow_length,
                    )),
                    Line::from(format!(
                        "Flags: inverted={} trailing={}",
                        bool_label(self.strategy.native_ema.inverted),
                        bool_label(self.strategy.native_ema.use_trailing_stop)
                    )),
                    Line::from(format!(
                        "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                        self.strategy.native_ema.take_profit_ticks,
                        self.strategy.native_ema.stop_loss_ticks,
                        self.strategy.native_ema.trail_trigger_ticks,
                        self.strategy.native_ema.trail_offset_ticks,
                    )),
                    Line::from(""),
                    Line::from("Signal logic"),
                    Line::from("Buy: fast EMA crosses above slow EMA."),
                    Line::from("Sell: fast EMA crosses below slow EMA."),
                    Line::from("Inverted swaps buy/sell decisions before order routing."),
                    Line::from(format!(
                        "Auto-close holds flat {}m before the inferred session close.",
                        AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
                    )),
                ],
            };
            lines.extend([
                Line::from(
                    "TP/SL are broker-native and keyed from the confirmed broker entry price.",
                ),
                Line::from("Trailing updates move the broker stop only on new closed bars."),
                Line::from(""),
                Line::from(format!("Live status: {}", self.strategy_runtime_summary())),
                Line::from(format!(
                    "Selected contract qty: {}",
                    self.effective_market_position_qty()
                )),
                Line::from(format!(
                    "Selected contract entry: {}",
                    format_money(
                        self.selected_snapshot()
                            .and_then(|snapshot| snapshot.market_entry_price)
                    )
                )),
                Line::from(format!(
                    "Tick Size: {}",
                    format_money(self.market.tick_size)
                )),
            ]);
            return lines;
        }
        if self.strategy.kind != StrategyKind::Lua {
            return vec![
                Line::from("Native and Lua strategy details appear here."),
                Line::from("Pick a strategy type on the left to configure it."),
            ];
        }

        let focused = self.focus == Focus::LuaEditor;
        let window_start = self.strategy.lua_editor.window_start(22);
        let cursor_row = self.strategy.lua_editor.cursor().0;
        let lines = self.strategy.lua_editor.visible_lines(22);
        lines
            .into_iter()
            .enumerate()
            .map(|(idx, line)| {
                let line_no = window_start + idx + 1;
                let style = if focused && window_start + idx == cursor_row {
                    Style::default().fg(Color::Black).bg(Color::Cyan)
                } else {
                    Style::default()
                };
                Line::from(Span::styled(format!("{:>3} {}", line_no, line), style))
            })
            .collect()
    }

    fn selection_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!(
                "Strategy Runtime: {}",
                self.strategy_runtime_summary()
            )),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(format!("Bar Type: {}", self.bar_type.label())),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(format!("Contract results: {}", self.contract_results.len())),
            Line::from(format!(
                "Last subscribed contract: {}",
                self.market
                    .contract_name
                    .clone()
                    .unwrap_or_else(|| "none".to_string())
            )),
            Line::from("F3 opens the monitoring dashboard."),
        ]
    }

    fn selection_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Status: {}", self.status)),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!("Bar Type: {}", self.bar_type.label())),
            Line::from(format!("Query: {}", self.instrument_query)),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(match self.contract_results.get(self.selected_contract) {
                Some(contract) => format!("Selected contract: {}", contract.name),
                None => "Selected contract: none".to_string(),
            }),
        ];
        if let Some(snapshot) = self.selected_snapshot() {
            lines.push(Line::from(format!(
                "Account net liq: {}",
                format_money(snapshot.net_liq.or(snapshot.balance))
            )));
            lines.push(Line::from(format!(
                "Selected contract qty: {}",
                format_quantity(snapshot.market_position_qty)
            )));
        }
        lines
    }

    fn stats_lines(&self) -> Vec<Line<'static>> {
        let Some(snapshot) = self.selected_snapshot() else {
            return vec![
                Line::from("No selected account stats."),
                Line::from("Connect and wait for account sync."),
            ];
        };
        vec![
            Line::from(format!("Account: {}", snapshot.account_name)),
            Line::from(format!("Balance: {}", format_money(snapshot.balance))),
            Line::from(format!("Cash: {}", format_money(snapshot.cash_balance))),
            Line::from(format!("NetLiq: {}", format_money(snapshot.net_liq))),
            pnl_line("Session Realized PnL", snapshot.realized_pnl),
            pnl_line("Unrealized PnL", snapshot.unrealized_pnl),
            Line::from(format!(
                "Intraday Margin: {}",
                format_money(snapshot.intraday_margin)
            )),
            Line::from(format!(
                "Open Position Qty: {}",
                format_quantity(snapshot.open_position_qty)
            )),
            Line::from(format!(
                "Selected Contract Qty: {}",
                format_quantity(snapshot.market_position_qty)
            )),
            Line::from(format!(
                "Entry Price: {}",
                format_money(snapshot.market_entry_price)
            )),
            Line::from(format!(
                "Take Profit: {}",
                format_money(snapshot.selected_contract_take_profit_price)
            )),
            Line::from(format!(
                "Stop Loss: {}",
                format_money(snapshot.selected_contract_stop_price)
            )),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Active Contract: {name}"),
                None => "Active Contract: none".to_string(),
            }),
            Line::from(format!(
                "Order Qty: {}  TIF: {}",
                self.base_config.order_qty, self.base_config.time_in_force
            )),
            Line::from(if self.session_kind == SessionKind::Replay {
                format!(
                    "Hotkeys: b buy | s sell | c close | v visuals | [ slower | ] faster | 0 realtime ({})",
                    self.replay_speed.label()
                )
            } else {
                "Hotkeys: b buy | s sell | c close | v visuals".to_string()
            }),
        ]
    }

}
