use super::super::super::*;

impl App {
    pub(in crate::app) fn strategy_setup_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![styled_line(
            format!("Strategy Type: {}", self.strategy.kind.label()),
            self.focus == Focus::StrategyKind,
        )];

        lines.push(styled_line(
            format!(
                "Order Qty: {}",
                self.strategy_numeric_value(Focus::OrderQty, self.strategy.order_qty.to_string(),)
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
                    "Signal Delay Bars: {}",
                    self.strategy_numeric_value(
                        Focus::NativeSignalDelayBars,
                        self.strategy.native_signal_delay_bars.to_string(),
                    )
                ),
                self.focus == Focus::NativeSignalDelayBars,
            ));
            lines.push(styled_line(
                format!(
                    "Execution Path: {}",
                    self.strategy.native_execution_path.label()
                ),
                self.focus == Focus::NativeExecutionPath,
            ));
            lines.push(styled_line(
                format!(
                    "Reversal Mode: {}",
                    self.strategy.native_reversal_mode.label()
                ),
                self.focus == Focus::NativeReversalMode,
            ));
            lines.push(styled_line(
                format!("Blockout: {}", bool_label(self.strategy.blockout_enabled)),
                self.focus == Focus::NativeBlockoutEnabled,
            ));
            lines.push(styled_line(
                format!(
                    "Blockout Minutes Before Close: {}",
                    self.strategy_numeric_value(
                        Focus::NativeBlockoutMinutes,
                        format!("{:.0}", self.strategy.blockout_minutes_before_close),
                    )
                ),
                self.focus == Focus::NativeBlockoutMinutes,
            ));

            let show_protection_controls = self.native_protection_controls_visible();
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
                    if show_protection_controls {
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
                }
                NativeStrategyKind::EmaCross | NativeStrategyKind::HmaCross => {
                    let (
                        label,
                        fast_length,
                        slow_length,
                        inverted,
                        take_profit_ticks,
                        stop_loss_ticks,
                        use_trailing_stop,
                        trail_trigger_ticks,
                        trail_offset_ticks,
                    ) = match self.strategy.native_strategy {
                        NativeStrategyKind::EmaCross => (
                            "EMA",
                            self.strategy.native_ema.fast_length,
                            self.strategy.native_ema.slow_length,
                            self.strategy.native_ema.inverted,
                            self.strategy.native_ema.take_profit_ticks,
                            self.strategy.native_ema.stop_loss_ticks,
                            self.strategy.native_ema.use_trailing_stop,
                            self.strategy.native_ema.trail_trigger_ticks,
                            self.strategy.native_ema.trail_offset_ticks,
                        ),
                        NativeStrategyKind::HmaCross => (
                            "HMA",
                            self.strategy.native_hma_cross.fast_length,
                            self.strategy.native_hma_cross.slow_length,
                            self.strategy.native_hma_cross.inverted,
                            self.strategy.native_hma_cross.take_profit_ticks,
                            self.strategy.native_hma_cross.stop_loss_ticks,
                            self.strategy.native_hma_cross.use_trailing_stop,
                            self.strategy.native_hma_cross.trail_trigger_ticks,
                            self.strategy.native_hma_cross.trail_offset_ticks,
                        ),
                        NativeStrategyKind::HmaAngle => unreachable!(),
                    };
                    lines.push(styled_line(
                        format!(
                            "Fast {label} Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaFastLength,
                                fast_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaFastLength,
                    ));
                    lines.push(styled_line(
                        format!(
                            "Slow {label} Length: {}",
                            self.strategy_numeric_value(
                                Focus::EmaSlowLength,
                                slow_length.to_string(),
                            )
                        ),
                        self.focus == Focus::EmaSlowLength,
                    ));
                    lines.push(styled_line(
                        format!("Inverted: {}", bool_label(inverted)),
                        self.focus == Focus::EmaInverted,
                    ));
                    if show_protection_controls {
                        lines.push(styled_line(
                            format!(
                                "Take Profit Ticks: {}",
                                self.strategy_numeric_value(
                                    Focus::EmaTakeProfitTicks,
                                    format!("{:.0}", take_profit_ticks),
                                )
                            ),
                            self.focus == Focus::EmaTakeProfitTicks,
                        ));
                        lines.push(styled_line(
                            format!(
                                "Stop Loss Ticks: {}",
                                self.strategy_numeric_value(
                                    Focus::EmaStopLossTicks,
                                    format!("{:.0}", stop_loss_ticks),
                                )
                            ),
                            self.focus == Focus::EmaStopLossTicks,
                        ));
                        lines.push(styled_line(
                            format!("Trailing Stop: {}", bool_label(use_trailing_stop)),
                            self.focus == Focus::EmaTrailingStop,
                        ));
                        lines.push(styled_line(
                            format!(
                                "Trail Trigger Ticks: {}",
                                self.strategy_numeric_value(
                                    Focus::EmaTrailTriggerTicks,
                                    format!("{:.0}", trail_trigger_ticks),
                                )
                            ),
                            self.focus == Focus::EmaTrailTriggerTicks,
                        ));
                        lines.push(styled_line(
                            format!(
                                "Trail Offset Ticks: {}",
                                self.strategy_numeric_value(
                                    Focus::EmaTrailOffsetTicks,
                                    format!("{:.0}", trail_offset_ticks),
                                )
                            ),
                            self.focus == Focus::EmaTrailOffsetTicks,
                        ));
                    }
                }
            }

            if let Some(auto_trail) = self.displayed_auto_trail() {
                lines.push(Line::from(format_auto_trail_preview(auto_trail)));
                if let Some(live_line) = format_auto_trail_live(auto_trail) {
                    lines.push(Line::from(live_line));
                }
            }

            let controls_hint = if show_protection_controls {
                "Type numbers or use Left/Right for numeric fields. Left/Right/Enter toggles timing and reversal mode. Zero TP/SL disables them."
            } else {
                "Type numbers or use Left/Right for numeric fields. Left/Right/Enter toggles timing and reversal mode."
            };
            lines.push(Line::from(controls_hint));
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
        let continue_label = if self.automated_strategy_affordance_visible() {
            "[Enter] Continue To Dashboard / Arm Strategy"
        } else {
            "[Enter] Continue To Dashboard (Monitor Only)"
        };
        lines.push(styled_line(
            continue_label.to_string(),
            self.focus == Focus::StrategyContinue,
        ));
        lines
    }

    pub(in crate::app) fn strategy_notes_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![Line::from("TUI strategy setup supports Native Rust.")];

        match self.strategy.kind {
            StrategyKind::Native => {
                lines.extend([
                    Line::from("Native can run on closed bars or live bars."),
                    Line::from("Closed-bar delay waits extra completed bars before entry."),
                    Line::from("Direct reversal is fastest."),
                    Line::from("Flatten > Confirm > Enter is safer."),
                    Line::from("CloseAll > Enter submits close-all and the broker-owned reverse entry immediately."),
                ]);
                if self.native_protection_controls_visible() {
                    lines.push(Line::from("TP/SL/trailing stop settings are in ticks."));
                }
                lines.extend([
                    Line::from("Blockout can flatten before close and hold until reopen."),
                    Line::from("Controls: Up/Down move, Left/Right edit."),
                ]);
                if self.automated_strategy_affordance_visible() {
                    lines.push(Line::from("Enter on Continue arms the strategy."));
                } else {
                    lines.push(Line::from(
                        "Enter on Continue opens the dashboard in monitor mode.",
                    ));
                }
            }
            StrategyKind::Lua => lines.extend([
                Line::from("Lua can load from file or editor."),
                Line::from("Controls: Up/Down move, Enter loads or continues."),
                Line::from("Lua normal: h/j/k/l move, i/a/o/x edit."),
                Line::from("Lua insert: type, Enter newline, Esc exit."),
            ]),
            StrategyKind::MachineLearning => lines.extend([
                Line::from("Machine Learning strategies are not available in the TUI."),
                Line::from("Controls: Up/Down move, Enter continues."),
            ]),
        }

        lines
    }

    pub(in crate::app) fn strategy_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Selected: {}", self.strategy.summary_label())),
            Line::from(match self.strategy.kind {
                StrategyKind::Native if self.automated_strategy_affordance_visible() => {
                    format!(
                        "Native Rust {} is active and can submit automated market orders.",
                        self.strategy.native_strategy.label()
                    )
                }
                StrategyKind::Native => format!(
                    "Native Rust {} is active for monitor-only observation.",
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
            lines.push(Line::from(self.native_summary_for_display()));
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
}
