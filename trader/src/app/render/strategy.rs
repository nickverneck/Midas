use super::*;

impl App {
    pub(in crate::app) fn handle_strategy_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                self.clear_strategy_numeric_input();
                self.focus = self.prev_strategy_focus();
                return;
            }
            KeyCode::Down | KeyCode::Tab => {
                self.clear_strategy_numeric_input();
                self.focus = self.next_strategy_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::StrategyKind => match key.code {
                KeyCode::Left => {
                    self.strategy.kind = self.strategy.kind.prev();
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.kind = self.strategy.kind.next();
                }
                _ => {}
            },
            Focus::NativeStrategy => match key.code {
                KeyCode::Left => {
                    self.strategy.native_strategy = self.strategy.native_strategy.prev();
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_strategy = self.strategy.native_strategy.next();
                }
                _ => {}
            },
            Focus::NativeSignalTiming => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_signal_timing =
                        self.strategy.native_signal_timing.toggle();
                }
                _ => {}
            },
            Focus::NativeSignalDelayBars => {
                let _ = edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::NativeSignalDelayBars,
                    &mut self.strategy.native_signal_delay_bars,
                    key,
                    0,
                    1,
                );
            }
            Focus::NativeExecutionPath => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_execution_path =
                        self.strategy.native_execution_path.toggle();
                }
                _ => {}
            },
            Focus::NativeReversalMode => match key.code {
                KeyCode::Left => {
                    self.strategy.native_reversal_mode = self.strategy.native_reversal_mode.prev();
                }
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.native_reversal_mode = self.strategy.native_reversal_mode.next();
                }
                _ => {}
            },
            Focus::NativeBlockoutEnabled => {
                let _ = toggle_bool(&mut self.strategy.blockout_enabled, key);
            }
            Focus::NativeBlockoutMinutes => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::NativeBlockoutMinutes,
                    &mut self.strategy.blockout_minutes_before_close,
                    key,
                    0.0,
                    5.0,
                );
            }
            Focus::OrderQty => {
                let _ = edit_strategy_i32(
                    &mut self.strategy_numeric_input,
                    Focus::OrderQty,
                    &mut self.strategy.order_qty,
                    key,
                    1,
                    1,
                );
            }
            Focus::HmaLength => {
                let _ = edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaLength,
                    &mut self.strategy.native_hma.hma_length,
                    key,
                    2,
                    1,
                );
            }
            Focus::HmaMinAngle => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaMinAngle,
                    &mut self.strategy.native_hma.min_angle,
                    key,
                    0.0,
                    0.5,
                );
            }
            Focus::HmaAngleLookback => {
                let _ = edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaAngleLookback,
                    &mut self.strategy.native_hma.angle_lookback,
                    key,
                    1,
                    1,
                );
            }
            Focus::HmaBarsRequired => {
                let _ = edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::HmaBarsRequired,
                    &mut self.strategy.native_hma.bars_required_to_trade,
                    key,
                    1,
                    1,
                );
            }
            Focus::HmaLongsOnly => {
                let _ = toggle_bool(&mut self.strategy.native_hma.longs_only, key);
            }
            Focus::HmaInverted => {
                let _ = toggle_bool(&mut self.strategy.native_hma.inverted, key);
            }
            Focus::HmaTakeProfitTicks => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTakeProfitTicks,
                    &mut self.strategy.native_hma.take_profit_ticks,
                    key,
                    0.0,
                    1.0,
                );
            }
            Focus::HmaStopLossTicks => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaStopLossTicks,
                    &mut self.strategy.native_hma.stop_loss_ticks,
                    key,
                    0.0,
                    1.0,
                );
            }
            Focus::HmaTrailingStop => {
                let _ = toggle_bool(&mut self.strategy.native_hma.use_trailing_stop, key);
            }
            Focus::HmaTrailTriggerTicks => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTrailTriggerTicks,
                    &mut self.strategy.native_hma.trail_trigger_ticks,
                    key,
                    0.0,
                    1.0,
                );
            }
            Focus::HmaTrailOffsetTicks => {
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::HmaTrailOffsetTicks,
                    &mut self.strategy.native_hma.trail_offset_ticks,
                    key,
                    0.0,
                    1.0,
                );
            }
            Focus::EmaFastLength => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_usize(
                        &mut self.strategy_numeric_input,
                        Focus::EmaFastLength,
                        &mut self.strategy.native_hma_cross.fast_length,
                        key,
                        1,
                        1,
                    ),
                    _ => edit_strategy_usize(
                        &mut self.strategy_numeric_input,
                        Focus::EmaFastLength,
                        &mut self.strategy.native_ema.fast_length,
                        key,
                        1,
                        1,
                    ),
                };
            }
            Focus::EmaSlowLength => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_usize(
                        &mut self.strategy_numeric_input,
                        Focus::EmaSlowLength,
                        &mut self.strategy.native_hma_cross.slow_length,
                        key,
                        1,
                        1,
                    ),
                    _ => edit_strategy_usize(
                        &mut self.strategy_numeric_input,
                        Focus::EmaSlowLength,
                        &mut self.strategy.native_ema.slow_length,
                        key,
                        1,
                        1,
                    ),
                };
            }
            Focus::EmaInverted => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => {
                        toggle_bool(&mut self.strategy.native_hma_cross.inverted, key)
                    }
                    _ => toggle_bool(&mut self.strategy.native_ema.inverted, key),
                };
            }
            Focus::EmaTakeProfitTicks => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTakeProfitTicks,
                        &mut self.strategy.native_hma_cross.take_profit_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                    _ => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTakeProfitTicks,
                        &mut self.strategy.native_ema.take_profit_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                };
            }
            Focus::EmaStopLossTicks => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaStopLossTicks,
                        &mut self.strategy.native_hma_cross.stop_loss_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                    _ => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaStopLossTicks,
                        &mut self.strategy.native_ema.stop_loss_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                };
            }
            Focus::EmaTrailingStop => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => {
                        toggle_bool(&mut self.strategy.native_hma_cross.use_trailing_stop, key)
                    }
                    _ => toggle_bool(&mut self.strategy.native_ema.use_trailing_stop, key),
                };
            }
            Focus::EmaTrailTriggerTicks => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTrailTriggerTicks,
                        &mut self.strategy.native_hma_cross.trail_trigger_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                    _ => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTrailTriggerTicks,
                        &mut self.strategy.native_ema.trail_trigger_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                };
            }
            Focus::EmaTrailOffsetTicks => {
                let _ = match self.strategy.native_strategy {
                    NativeStrategyKind::HmaCross => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTrailOffsetTicks,
                        &mut self.strategy.native_hma_cross.trail_offset_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                    _ => edit_strategy_float(
                        &mut self.strategy_numeric_input,
                        Focus::EmaTrailOffsetTicks,
                        &mut self.strategy.native_ema.trail_offset_ticks,
                        key,
                        0.0,
                        1.0,
                    ),
                };
            }
            Focus::LuaSourceMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.strategy.lua_source_mode = self.strategy.lua_source_mode.toggle();
                }
                _ => {}
            },
            Focus::LuaFilePath => {
                if key.code == KeyCode::Enter {
                    match self.strategy.load_lua_file() {
                        Ok(lines) => self.push_log(format!(
                            "Loaded Lua file `{}` ({} lines)",
                            self.strategy.lua_file_path, lines
                        )),
                        Err(err) => self.push_log(format!(
                            "ERROR: failed to load Lua file `{}`: {}",
                            self.strategy.lua_file_path, err
                        )),
                    }
                } else {
                    edit_string(&mut self.strategy.lua_file_path, key);
                }
            }
            Focus::LuaEditor => {
                let _ = self.strategy.lua_editor.handle_key(key);
            }
            Focus::StrategyContinue => {
                if key.code == KeyCode::Enter {
                    self.screen = Screen::Dashboard;
                    self.focus = Focus::AccountList;
                    self.sync_selected_account(cmd_tx);
                    if self.capabilities.automated_orders {
                        self.arm_native_strategy(cmd_tx);
                    } else {
                        self.push_log(format!(
                            "{} automation is not enabled yet; opening the dashboard in monitor mode.",
                            self.selected_broker.label()
                        ));
                    }
                    self.push_log(format!(
                        "Strategy selected: {}",
                        self.strategy.summary_label()
                    ));
                    if self.strategy.kind == StrategyKind::Native {
                        self.push_log(format!(
                            "Strategy config: {}",
                            self.native_summary_for_display()
                        ));
                    }
                }
            }
            Focus::EngineList
            | Focus::Env
            | Focus::AuthMode
            | Focus::LogMode
            | Focus::TokenOverride
            | Focus::Username
            | Focus::Password
            | Focus::ApiKey
            | Focus::AppId
            | Focus::AppVersion
            | Focus::Cid
            | Focus::Secret
            | Focus::TokenPath
            | Focus::BrokerList
            | Focus::Connect
            | Focus::ReplayMode
            | Focus::AccountList
            | Focus::InstrumentQuery
            | Focus::BarTypeToggle
            | Focus::BarValue
            | Focus::CandleModeToggle
            | Focus::ContractList => {}
        }
    }

    pub(in crate::app) fn render_strategy_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(area);

        let notes_lines = self.strategy_notes_lines();
        let notes_height = (notes_lines.len() as u16).saturating_add(2);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(notes_height)])
            .split(columns[0]);

        let setup_lines = self.strategy_setup_lines();
        let setup_scroll = focused_paragraph_scroll_offset(&setup_lines, left[0]);
        let setup = Paragraph::new(setup_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Strategy Setup"),
            )
            .scroll((setup_scroll, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(setup, left[0]);

        let notes = Paragraph::new(notes_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Execution Priority"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(notes, left[1]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(16), Constraint::Length(10)])
            .split(columns[1]);

        let editor = Paragraph::new(self.strategy_detail_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(self.strategy_detail_title()),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(editor, right[0]);

        let preview = Paragraph::new(self.strategy_preview_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Selected Strategy"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(preview, right[1]);
    }
}
