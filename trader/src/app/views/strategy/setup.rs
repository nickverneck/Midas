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

    pub(in crate::app) fn strategy_notes_lines(&self) -> Vec<Line<'static>> {
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

    pub(in crate::app) fn strategy_preview_lines(&self) -> Vec<Line<'static>> {
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
}
