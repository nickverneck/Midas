use super::*;

impl App {
    #[cfg(not(feature = "replay"))]
    fn open_replay_dataset_views(&mut self) {
        self.status = "Saved dataset views require a replay-enabled build.".to_string();
    }

    #[cfg(not(feature = "replay"))]
    fn replay_selected_dataset_view_path(&self) -> Option<std::path::PathBuf> {
        None
    }

    pub(in crate::app) fn handle_replay_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Downloader {
            self.handle_replay_downloader_key(key, cmd_tx);
            return;
        }
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::DatasetViews {
            self.handle_replay_dataset_views_key(key);
            return;
        }

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
                if matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V')) {
                    self.open_replay_dataset_views();
                    return;
                }
                if matches!(key.code, KeyCode::Char('n') | KeyCode::Char('N')) {
                    self.open_new_replay_downloader();
                    return;
                }
                if matches!(key.code, KeyCode::Char('d') | KeyCode::Char('D')) {
                    self.open_selected_replay_downloader();
                    return;
                }
                let count = self.replay_cache_library.datasets.len();
                if count == 0 {
                    if key.code == KeyCode::Enter {
                        self.open_new_replay_downloader();
                    } else if key.code == KeyCode::Up {
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
                        self.replay_dataset_view_path = None;
                        return;
                    }
                    KeyCode::Char('a') => {
                        self.replay_dataset_index = None;
                        self.replay_dataset_view_path = None;
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
                if matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V')) {
                    self.open_replay_dataset_views();
                    return;
                }
                if matches!(key.code, KeyCode::Char('d') | KeyCode::Char('D')) {
                    self.open_selected_replay_downloader();
                    return;
                }
                if matches!(key.code, KeyCode::Char('n') | KeyCode::Char('N')) {
                    self.open_new_replay_downloader();
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
            #[cfg(feature = "replay")]
            Focus::ReplayDownloadProvider
            | Focus::ReplayDownloadEnv
            | Focus::ReplayDownloadInstrument
            | Focus::ReplayDownloadContract
            | Focus::ReplayDownloadStart
            | Focus::ReplayDownloadEnd
            | Focus::ReplayDownloadSource
            | Focus::ReplayDownloadBarType
            | Focus::ReplayDownloadBarValue
            | Focus::ReplayDownloadCandleMode
            | Focus::ReplayDownloadName
            | Focus::ReplayDownloadTags
            | Focus::ReplayDownloadCacheRoot
            | Focus::ReplayDownloadSubmit => {}
            #[cfg(feature = "replay")]
            Focus::ReplayViewList
            | Focus::ReplayViewId
            | Focus::ReplayViewPreset
            | Focus::ReplayViewTradingDate
            | Focus::ReplayViewStart
            | Focus::ReplayViewEnd
            | Focus::ReplayViewTimezone
            | Focus::ReplayViewWarmupMinutes
            | Focus::ReplayViewSave => {}
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
            replay_dataset_view: self.replay_selected_dataset_view_path(),
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
}
