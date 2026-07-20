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
            let show_protection = self.native_protection_controls_visible();
            let mut lines = match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    let mut detail = vec![
                        Line::from("HMA Angle Strategy"),
                        Line::from(format!("Type: {}", NativeStrategyKind::HmaAngle.label())),
                        Line::from(format!(
                            "Params: len={} min_angle={:.1} lookback={} bars_required={}",
                            self.strategy.native_hma.hma_length,
                            self.strategy.native_hma.min_angle,
                            self.strategy.native_hma.angle_lookback,
                            self.strategy.native_hma.bars_required_to_trade
                        )),
                        if show_protection {
                            Line::from(format!(
                                "Flags: longs_only={} inverted={} trailing={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_hma.longs_only),
                                bool_label(self.strategy.native_hma.inverted),
                                bool_label(self.strategy.native_hma.use_trailing_stop),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        } else {
                            Line::from(format!(
                                "Flags: longs_only={} inverted={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_hma.longs_only),
                                bool_label(self.strategy.native_hma.inverted),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        },
                    ];
                    if show_protection {
                        let risk = if self.strategy.native_hma.use_trailing_stop {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                                self.strategy.native_hma.take_profit_ticks,
                                self.strategy.native_hma.stop_loss_ticks,
                                self.strategy.native_hma.trail_trigger_ticks,
                                self.strategy.native_hma.trail_offset_ticks,
                            )
                        } else {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0}",
                                self.strategy.native_hma.take_profit_ticks,
                                self.strategy.native_hma.stop_loss_ticks,
                            )
                        };
                        detail.push(Line::from(risk));
                    }
                    detail.extend([
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
                            "Blockout: {} {:.0}m before the inferred session close.",
                            bool_label(self.strategy.blockout_enabled),
                            self.strategy.blockout_minutes_before_close
                        )),
                    ]);
                    detail
                }
                NativeStrategyKind::EmaCross => {
                    let mut detail = vec![
                        Line::from("EMA Crossover Strategy"),
                        Line::from(format!("Type: {}", NativeStrategyKind::EmaCross.label())),
                        Line::from(format!(
                            "Params: fast={} slow={}",
                            self.strategy.native_ema.fast_length,
                            self.strategy.native_ema.slow_length,
                        )),
                        if show_protection {
                            Line::from(format!(
                                "Flags: inverted={} trailing={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_ema.inverted),
                                bool_label(self.strategy.native_ema.use_trailing_stop),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        } else {
                            Line::from(format!(
                                "Flags: inverted={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_ema.inverted),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        },
                    ];
                    if show_protection {
                        let risk = if self.strategy.native_ema.use_trailing_stop {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                                self.strategy.native_ema.take_profit_ticks,
                                self.strategy.native_ema.stop_loss_ticks,
                                self.strategy.native_ema.trail_trigger_ticks,
                                self.strategy.native_ema.trail_offset_ticks,
                            )
                        } else {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0}",
                                self.strategy.native_ema.take_profit_ticks,
                                self.strategy.native_ema.stop_loss_ticks,
                            )
                        };
                        detail.push(Line::from(risk));
                    }
                    detail.extend([
                        Line::from(""),
                        Line::from("Signal logic"),
                        Line::from("Buy: fast EMA crosses above slow EMA."),
                        Line::from("Sell: fast EMA crosses below slow EMA."),
                        Line::from("Inverted swaps buy/sell decisions before order routing."),
                        Line::from(format!(
                            "Blockout: {} {:.0}m before the inferred session close.",
                            bool_label(self.strategy.blockout_enabled),
                            self.strategy.blockout_minutes_before_close
                        )),
                    ]);
                    detail
                }
                NativeStrategyKind::HmaCross => {
                    let mut detail = vec![
                        Line::from("HMA Crossover Strategy"),
                        Line::from(format!("Type: {}", NativeStrategyKind::HmaCross.label())),
                        Line::from(format!(
                            "Params: fast={} slow={}",
                            self.strategy.native_hma_cross.fast_length,
                            self.strategy.native_hma_cross.slow_length,
                        )),
                        if show_protection {
                            Line::from(format!(
                                "Flags: inverted={} trailing={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_hma_cross.inverted),
                                bool_label(self.strategy.native_hma_cross.use_trailing_stop),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        } else {
                            Line::from(format!(
                                "Flags: inverted={} timing={} delay={} path={} reversal={}",
                                bool_label(self.strategy.native_hma_cross.inverted),
                                self.strategy.native_signal_timing.label(),
                                self.strategy.native_signal_delay_bars,
                                self.strategy.native_execution_path.label(),
                                self.strategy.native_reversal_mode.label(),
                            ))
                        },
                    ];
                    if show_protection {
                        let risk = if self.strategy.native_hma_cross.use_trailing_stop {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0} trail_trigger={:.0} trail_offset={:.0}",
                                self.strategy.native_hma_cross.take_profit_ticks,
                                self.strategy.native_hma_cross.stop_loss_ticks,
                                self.strategy.native_hma_cross.trail_trigger_ticks,
                                self.strategy.native_hma_cross.trail_offset_ticks,
                            )
                        } else {
                            format!(
                                "Risk: tp_ticks={:.0} sl_ticks={:.0}",
                                self.strategy.native_hma_cross.take_profit_ticks,
                                self.strategy.native_hma_cross.stop_loss_ticks,
                            )
                        };
                        detail.push(Line::from(risk));
                    }
                    detail.extend([
                        Line::from(""),
                        Line::from("Signal logic"),
                        Line::from("Buy: fast HMA crosses above slow HMA."),
                        Line::from("Sell: fast HMA crosses below slow HMA."),
                        Line::from("Inverted swaps buy/sell decisions before order routing."),
                        Line::from(format!(
                            "Blockout: {} {:.0}m before the inferred session close.",
                            bool_label(self.strategy.blockout_enabled),
                            self.strategy.blockout_minutes_before_close
                        )),
                    ]);
                    detail
                }
            };
            if show_protection {
                lines.extend([
                    Line::from(
                        "TP/SL are broker-native and keyed from the confirmed broker entry price.",
                    ),
                    Line::from(
                        "Chart labels TP*/SL* are projected from the native config until broker protection sync lands.",
                    ),
                    Line::from("Trailing stop uses Tradovate broker auto-trail when enabled."),
                ]);
            }
            if let Some(auto_trail) = self.displayed_auto_trail() {
                lines.push(Line::from(format_auto_trail_preview(auto_trail)));
                if let Some(live_line) = format_auto_trail_live(auto_trail) {
                    lines.push(Line::from(live_line));
                }
            }
            lines.extend([
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
