use super::super::super::*;

impl App {
    pub(in crate::app) fn strategy_detail_title(&self) -> String {
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

    pub(in crate::app) fn strategy_detail_lines(&self) -> Vec<Line<'static>> {
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
}
