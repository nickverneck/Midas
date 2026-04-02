impl App {
    fn connection_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Broker: {}", self.selected_broker.label())),
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
            Line::from(""),
        ];
        match self.selected_broker {
            BrokerKind::Ironbeam => {
                lines.push(styled_line(
                    format!("API Key: {}", mask(&self.form.api_key)),
                    self.focus == Focus::ApiKey,
                ));
                lines.push(Line::from(""));
            }
            BrokerKind::Tradovate => {
                lines.push(styled_line(
                    format!("App ID: {}", self.form.app_id),
                    self.focus == Focus::AppId,
                ));
                lines.push(styled_line(
                    format!("App Version: {}", self.form.app_version),
                    self.focus == Focus::AppVersion,
                ));
                lines.push(styled_line(
                    format!("CID: {}", self.form.cid),
                    self.focus == Focus::Cid,
                ));
                lines.push(styled_line(
                    format!("Secret: {}", mask(&self.form.secret)),
                    self.focus == Focus::Secret,
                ));
                lines.push(Line::from(""));
            }
        }
        lines.push(styled_line(
            "[Enter] Connect / Refresh Session".to_string(),
            self.focus == Focus::Connect,
        ));
        lines.push(styled_line(
            if self.broker_supports_replay() {
                "[Enter] Replay Mode (local file, skips login)".to_string()
            } else {
                "[Enter] Replay Mode unavailable for this broker/build".to_string()
            },
            self.focus == Focus::ReplayMode,
        ));
        lines
    }
    fn login_notes_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from("1. Token Override is used first when non-empty."),
            Line::from("2. Token File mode reads token_path, then session cache."),
            Line::from(match self.selected_broker {
                BrokerKind::Tradovate => "3. Credentials mode requests a fresh Tradovate access token.",
                BrokerKind::Ironbeam => "3. Credentials mode requests a fresh Ironbeam bearer token using username/password and optional api_key.",
            }),
            Line::from("4. Debug log mode adds submit/seen/ack/fill lifecycle lines."),
            Line::from(""),
            Line::from("Use Up/Down to move between fields."),
            Line::from("Use Left/Right on Env, Auth Mode, or Log Mode."),
            Line::from("Paste a token directly into Token Override when needed."),
            Line::from(if self.broker_supports_replay() {
                "Replay Mode loads the local tick file and starts on 1 Range bars by default."
            } else {
                "Replay Mode is only available on Tradovate builds with `--features replay`."
            }),
        ];
        if self.selected_broker == BrokerKind::Ironbeam {
            lines.push(Line::from("Ironbeam currently uses 1-minute bars on the selection screen."));
        }
        lines
    }

    fn login_status_lines(&self) -> Vec<Line<'static>> {
        let (rest_url, user_ws, market_ws) = match (self.selected_broker, self.form.env) {
            (BrokerKind::Tradovate, env) => (
                env.rest_url().to_string(),
                Some(env.user_ws_url().to_string()),
                Some(env.market_ws_url().to_string()),
            ),
            (BrokerKind::Ironbeam, TradingEnvironment::Sim) => (
                "https://demo.ironbeamapi.com/v2".to_string(),
                Some("wss://demo.ironbeamapi.com/v2/stream/{streamId}?token=...".to_string()),
                None,
            ),
            (BrokerKind::Ironbeam, TradingEnvironment::Live) => (
                "https://live.ironbeamapi.com/v2".to_string(),
                Some("wss://live.ironbeamapi.com/v2/stream/{streamId}?token=...".to_string()),
                None,
            ),
        };
        vec![
            Line::from(format!("Current status: {}", self.status)),
            Line::from(format!("Broker: {}", self.selected_broker.label())),
            Line::from(format!("Session Mode: {}", self.session_kind.label())),
            Line::from(format!("Environment REST: {rest_url}")),
            Line::from(format!("Log Mode: {}", self.form.log_mode.label())),
            Line::from(match user_ws {
                Some(url) => format!("User WebSocket: {url}"),
                None => "User WebSocket: n/a".to_string(),
            }),
            Line::from(match market_ws {
                Some(url) => format!("Market WebSocket: {url}"),
                None => "Market WebSocket: embedded stream".to_string(),
            }),
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
            lines.push(styled_line(
                format!(
                    "Signal Timing: {}",
                    self.strategy.native_signal_timing.label()
                ),
                self.focus == Focus::NativeSignalTiming,
            ));
            lines.push(styled_line(
                format!(
                    "Reversal Mode: {}",
                    self.strategy.native_reversal_mode.label()
                ),
                self.focus == Focus::NativeReversalMode,
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
                "Type numbers or use Left/Right for numeric fields. Left/Right/Enter toggles timing and reversal mode. Zero TP/SL disables them.",
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
        let mut lines = vec![Line::from(
            "Backend order: Native Rust > Lua > Machine Learning.",
        )];

        match self.strategy.kind {
            StrategyKind::Native => lines.extend([
                Line::from("Native can run on closed bars or live bars."),
                Line::from("Direct reversal is fastest."),
                Line::from("Flatten > Confirm > Enter is safer."),
                Line::from("CloseAll > Enter flattens the contract, then submits the reverse entry without waiting for position sync."),
                Line::from("TP/SL/trailing stop sync in ticks."),
                Line::from(format!(
                    "Auto-close flattens {}m before close.",
                    AUTO_CLOSE_MINUTES_BEFORE_SESSION_END
                )),
                Line::from("Controls: Up/Down move, Left/Right edit."),
                Line::from("Enter on Continue arms the strategy."),
            ]),
            StrategyKind::Lua => lines.extend([
                Line::from("Lua can load from file or editor."),
                Line::from("Native stays higher priority than Lua."),
                Line::from("ML remains selection-only for now."),
                Line::from("Controls: Up/Down move, Enter loads or arms."),
                Line::from("Lua normal: h/j/k/l move, i/a/o/x edit."),
                Line::from("Lua insert: type, Enter newline, Esc exit."),
            ]),
            StrategyKind::MachineLearning => lines.extend([
                Line::from("ML stays selection-only for now."),
                Line::from("Native and Lua remain higher priority."),
                Line::from("Controls: Up/Down move, Enter continues."),
            ]),
        }

        lines
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
                        "Flags: longs_only={} inverted={} trailing={} timing={} reversal={}",
                        bool_label(self.strategy.native_hma.longs_only),
                        bool_label(self.strategy.native_hma.inverted),
                        bool_label(self.strategy.native_hma.use_trailing_stop),
                        self.strategy.native_signal_timing.label(),
                        self.strategy.native_reversal_mode.label(),
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
                        "Flags: inverted={} trailing={} timing={} reversal={}",
                        bool_label(self.strategy.native_ema.inverted),
                        bool_label(self.strategy.native_ema.use_trailing_stop),
                        self.strategy.native_signal_timing.label(),
                        self.strategy.native_reversal_mode.label(),
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
                Line::from(
                    "Chart labels TP*/SL* are projected from the native config until broker protection sync lands.",
                ),
                Line::from(format!(
                    "Trailing updates follow the configured signal timing: {}.",
                    self.strategy.native_signal_timing.label()
                )),
                Line::from(""),
                Line::from(format!("Live status: {}", self.strategy_runtime_summary())),
                Line::from(format!(
                    "Selected contract qty: {}",
                    self.effective_market_position_qty()
                )),
                Line::from(format!(
                    "Selected contract entry: {}",
                    format_money(self.displayed_trade_levels().entry_price)
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
