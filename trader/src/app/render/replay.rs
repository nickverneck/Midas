use super::*;

impl App {
    pub(in crate::app) fn handle_replay_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match key.code {
            KeyCode::BackTab => {
                self.focus = self.prev_replay_focus();
                return;
            }
            KeyCode::Tab => {
                self.focus = self.next_replay_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::BarTypeToggle => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    self.bar_type = match key.code {
                        KeyCode::Left => self.bar_type.previous_kind(),
                        _ => self.bar_type.next_kind(),
                    };
                    return;
                }
                KeyCode::Up => {
                    self.focus = self.prev_replay_focus();
                    return;
                }
                KeyCode::Down | KeyCode::Enter => {
                    self.focus = self.next_replay_focus();
                    return;
                }
                _ => {}
            },
            Focus::BarValue => {
                match key.code {
                    KeyCode::Up => {
                        self.focus = self.prev_replay_focus();
                        return;
                    }
                    KeyCode::Down | KeyCode::Enter => {
                        self.focus = self.next_replay_focus();
                        return;
                    }
                    _ => {}
                }

                let mut value = self.bar_type.value() as usize;
                if edit_strategy_usize(
                    &mut self.strategy_numeric_input,
                    Focus::BarValue,
                    &mut value,
                    key,
                    1,
                    1,
                ) {
                    self.bar_type = self
                        .bar_type
                        .with_value(value.min(u32::MAX as usize) as u32);
                    return;
                }
            }
            Focus::CandleModeToggle => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    if !self.candle_mode_controls_visible() {
                        self.candle_mode = CandleMode::Standard;
                        self.focus = self.next_replay_focus();
                        return;
                    }
                    self.candle_mode = self.candle_mode.toggle();
                    return;
                }
                KeyCode::Up => {
                    self.focus = self.prev_replay_focus();
                    return;
                }
                KeyCode::Down | KeyCode::Enter => {
                    self.focus = self.next_replay_focus();
                    return;
                }
                _ => {}
            },
            #[cfg(feature = "replay")]
            Focus::ReplayDataset => {
                let count = self.replay_cache_library.datasets.len();
                if count == 0 {
                    if matches!(key.code, KeyCode::Up | KeyCode::Enter) {
                        self.focus = self.prev_replay_focus();
                    } else if key.code == KeyCode::Down {
                        self.focus = self.next_replay_focus();
                    }
                    return;
                }
                match key.code {
                    KeyCode::Up | KeyCode::Down => {
                        let current = self.replay_dataset_index.unwrap_or(0);
                        let next = if key.code == KeyCode::Up {
                            current.checked_sub(1).unwrap_or(count - 1)
                        } else {
                            (current + 1) % count
                        };
                        self.replay_dataset_index = Some(next);
                        return;
                    }
                    KeyCode::Char('a') => {
                        self.replay_dataset_index = None;
                        return;
                    }
                    KeyCode::Enter => {
                        self.focus = self.next_replay_focus();
                        return;
                    }
                    _ => {}
                }
            }
            Focus::ReplayMode => {
                if matches!(key.code, KeyCode::Char('d') | KeyCode::Char('D')) {
                    self.start_replay_download(cmd_tx);
                    return;
                }
                if matches!(key.code, KeyCode::Enter | KeyCode::Char(' ')) {
                    self.start_replay_mode(cmd_tx);
                    return;
                }
                match key.code {
                    KeyCode::Up => {
                        self.focus = self.prev_replay_focus();
                        return;
                    }
                    KeyCode::Down => {
                        self.focus = self.next_replay_focus();
                        return;
                    }
                    _ => {}
                }
            }
            Focus::EngineList
            | Focus::BrokerList
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
            | Focus::Connect
            | Focus::StrategyKind
            | Focus::OrderQty
            | Focus::NativeStrategy
            | Focus::NativeSignalTiming
            | Focus::NativeSignalDelayBars
            | Focus::NativeExecutionPath
            | Focus::NativeReversalMode
            | Focus::NativeBlockoutEnabled
            | Focus::NativeBlockoutMinutes
            | Focus::HmaLength
            | Focus::HmaMinAngle
            | Focus::HmaAngleLookback
            | Focus::HmaBarsRequired
            | Focus::HmaLongsOnly
            | Focus::HmaInverted
            | Focus::HmaTakeProfitTicks
            | Focus::HmaStopLossTicks
            | Focus::HmaTrailingStop
            | Focus::HmaTrailTriggerTicks
            | Focus::HmaTrailOffsetTicks
            | Focus::EmaFastLength
            | Focus::EmaSlowLength
            | Focus::EmaInverted
            | Focus::EmaTakeProfitTicks
            | Focus::EmaStopLossTicks
            | Focus::EmaTrailingStop
            | Focus::EmaTrailTriggerTicks
            | Focus::EmaTrailOffsetTicks
            | Focus::LuaSourceMode
            | Focus::LuaFilePath
            | Focus::LuaEditor
            | Focus::StrategyContinue
            | Focus::AccountList
            | Focus::InstrumentQuery
            | Focus::ContractList => {}
        }
    }

    fn start_replay_mode(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if !self.replay_dataset_available() {
            self.status = format!(
                "Replay dataset missing: no matching cache and no local file at {}",
                self.base_config.replay_file_path.display()
            );
            self.push_log(self.status.clone());
            return;
        }

        if !self.replay_selected_bar_supported() {
            self.status = "Replay volume bars require a matching server-bar cache; this local file is price-only."
                .to_string();
            self.push_log(self.status.clone());
            return;
        }

        let _ = cmd_tx.send(ServiceCommand::EnterReplayMode {
            config: self.current_config(),
            bar_type: self.bar_type,
            candle_mode: self.effective_candle_mode(),
            replay_dataset_manifest: self.replay_selected_dataset_manifest(),
        });
        self.push_log(format!(
            "Replay mode requested: {} ({})",
            self.bar_type.mode_label(self.effective_candle_mode()),
            if self.replay_cache_can_serve_selected_bar() {
                "cache"
            } else {
                "local file"
            }
        ));
    }

    #[cfg(feature = "replay")]
    fn start_replay_download(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let Some(index) = self.replay_dataset_index else {
            self.status = "Select an owned dataset before downloading or extending it.".to_string();
            self.push_log(self.status.clone());
            return;
        };
        let Some(dataset) = self.replay_cache_library.datasets.get(index) else {
            self.status = "Selected replay dataset is no longer available.".to_string();
            self.push_log(self.status.clone());
            return;
        };
        let source_kind = match dataset.manifest.source_kind.badge() {
            "server-bars" | "raw-ticks" => dataset.manifest.source_kind.badge().to_string(),
            other => {
                self.status = format!("Cannot download source kind {other} from this screen.");
                self.push_log(self.status.clone());
                return;
            }
        };
        let _ = cmd_tx.send(ServiceCommand::DownloadReplayData {
            config: self.current_config(),
            instrument: dataset.manifest.instrument.symbol.clone(),
            contract: dataset.manifest.contract.symbol.clone(),
            start_date: dataset.manifest.coverage.start.date_naive(),
            end_date: dataset.manifest.coverage.end.date_naive(),
            source_kind,
            bar_type: self.bar_type,
        });
        self.status = format!(
            "Replay download requested for {} {}.",
            dataset.manifest.instrument.symbol, dataset.manifest.contract.symbol
        );
        self.push_log(self.status.clone());
    }

    #[cfg(not(feature = "replay"))]
    fn start_replay_download(&mut self, _cmd_tx: &UnboundedSender<ServiceCommand>) {
        self.status = "Replay downloader requires a replay-enabled build.".to_string();
        self.push_log(self.status.clone());
    }

    #[cfg(feature = "replay")]
    fn replay_selected_dataset_manifest(&self) -> Option<std::path::PathBuf> {
        self.replay_dataset_index
            .and_then(|index| self.replay_cache_library.datasets.get(index))
            .map(|dataset| dataset.manifest_path.clone())
    }

    #[cfg(not(feature = "replay"))]
    fn replay_selected_dataset_manifest(&self) -> Option<std::path::PathBuf> {
        None
    }

    pub(in crate::app) fn render_replay_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(area);

        let dataset = Paragraph::new(self.replay_dataset_library_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Dataset Library"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(dataset, columns[0]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(10), Constraint::Min(8)])
            .split(columns[1]);

        let controls = Paragraph::new(self.replay_market_control_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Market"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(controls, right[0]);

        let run = Paragraph::new(self.replay_run_control_lines())
            .block(Block::default().borders(Borders::ALL).title("Run"))
            .wrap(Wrap { trim: true });
        frame.render_widget(run, right[1]);
    }
}
