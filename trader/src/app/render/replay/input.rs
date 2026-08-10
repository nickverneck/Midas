use super::*;

impl App {
    #[cfg(feature = "replay")]
    pub(in crate::app) fn open_replay_setup(&mut self) {
        if let Some(bar_type) = self.replay_dataset_bar_type {
            self.bar_type = bar_type;
            self.candle_mode = self.bar_type.effective_candle_mode(self.candle_mode);
        }
        self.replay_view = ReplayView::Setup;
        self.focus = Focus::BarTypeToggle;
        self.clear_strategy_numeric_input();
        self.status = "Replay setup: configure the market and run settings.".to_string();
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn open_replay_library(&mut self) {
        self.replay_view = ReplayView::Library;
        self.sync_replay_dataset_filter();
        self.focus = if self.replay_filtered_dataset_indices().is_empty() {
            Focus::ReplayInstrumentQuery
        } else {
            Focus::ReplayDataset
        };
        self.clear_strategy_numeric_input();
    }

    #[cfg(feature = "replay")]
    fn navigate_replay_setup(&mut self, key: KeyCode) -> bool {
        if self.replay_view != ReplayView::Setup {
            return false;
        }
        match key {
            KeyCode::Up => {
                if self.focus == Focus::BarTypeToggle {
                    self.open_replay_library();
                } else {
                    self.focus = self.prev_replay_focus();
                }
                true
            }
            KeyCode::Down | KeyCode::Enter => {
                self.focus = self.next_replay_focus();
                true
            }
            _ => false,
        }
    }

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
                self.clear_strategy_numeric_input();
                #[cfg(feature = "replay")]
                if self.replay_view == ReplayView::Setup && self.focus == Focus::BarTypeToggle {
                    self.open_replay_library();
                    return;
                }
                self.focus = self.prev_replay_focus();
                return;
            }
            KeyCode::Tab => {
                self.clear_strategy_numeric_input();
                #[cfg(feature = "replay")]
                if self.replay_view == ReplayView::Library
                    && (self.focus == Focus::ReplayDataset
                        || (self.focus == Focus::ReplayInstrumentQuery
                            && self.replay_filtered_dataset_indices().is_empty()))
                {
                    self.open_replay_setup();
                    return;
                }
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
                    #[cfg(feature = "replay")]
                    if self.navigate_replay_setup(KeyCode::Up) {
                        return;
                    }
                    self.focus = self.prev_replay_focus();
                    return;
                }
                KeyCode::Down | KeyCode::Enter => {
                    #[cfg(feature = "replay")]
                    if self.navigate_replay_setup(key.code) {
                        return;
                    }
                    self.focus = self.next_replay_focus();
                    return;
                }
                _ => {}
            },
            Focus::BarValue => {
                match key.code {
                    KeyCode::Up => {
                        #[cfg(feature = "replay")]
                        if self.navigate_replay_setup(KeyCode::Up) {
                            return;
                        }
                        self.focus = self.prev_replay_focus();
                        return;
                    }
                    KeyCode::Down | KeyCode::Enter => {
                        #[cfg(feature = "replay")]
                        if self.navigate_replay_setup(key.code) {
                            return;
                        }
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
                    #[cfg(feature = "replay")]
                    if self.navigate_replay_setup(KeyCode::Up) {
                        return;
                    }
                    self.focus = self.prev_replay_focus();
                    return;
                }
                KeyCode::Down | KeyCode::Enter => {
                    #[cfg(feature = "replay")]
                    if self.navigate_replay_setup(key.code) {
                        return;
                    }
                    self.focus = self.next_replay_focus();
                    return;
                }
                _ => {}
            },
            #[cfg(feature = "replay")]
            Focus::ReplayInitialCapital => {
                if self.navigate_replay_setup(key.code) {
                    return;
                }
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::ReplayInitialCapital,
                    &mut self.base_config.replay_initial_capital,
                    key,
                    1.0,
                    1_000.0,
                );
            }
            #[cfg(feature = "replay")]
            Focus::ReplayMarginPerContract => {
                if self.navigate_replay_setup(key.code) {
                    return;
                }
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::ReplayMarginPerContract,
                    &mut self.base_config.replay_margin_per_contract,
                    key,
                    0.0,
                    100.0,
                );
            }
            #[cfg(feature = "replay")]
            Focus::ReplaySafetyBuffer => {
                if self.navigate_replay_setup(key.code) {
                    return;
                }
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::ReplaySafetyBuffer,
                    &mut self.base_config.replay_safety_buffer,
                    key,
                    0.0,
                    100.0,
                );
            }
            #[cfg(feature = "replay")]
            Focus::ReplaySafetyBufferPercent => {
                if self.navigate_replay_setup(key.code) {
                    return;
                }
                let _ = edit_strategy_float(
                    &mut self.strategy_numeric_input,
                    Focus::ReplaySafetyBufferPercent,
                    &mut self.base_config.replay_safety_buffer_percent,
                    key,
                    0.0,
                    1.0,
                );
            }
            #[cfg(feature = "replay")]
            Focus::ReplayInstrumentQuery => {
                match key.code {
                    KeyCode::Up => {
                        self.focus = self.prev_replay_focus();
                        return;
                    }
                    KeyCode::Down | KeyCode::Enter => {
                        let options = self.replay_dataset_options();
                        if options.is_empty() {
                            if key.code == KeyCode::Enter {
                                self.open_new_replay_downloader();
                            }
                        } else {
                            let selected =
                                self.replay_selected_option_position(&options).unwrap_or(0);
                            let (index, bar_type) = options[selected];
                            self.select_replay_dataset_option(index, bar_type);
                            self.focus = Focus::ReplayDataset;
                        }
                        return;
                    }
                    KeyCode::Right => {
                        self.focus = Focus::ReplayDataset;
                        return;
                    }
                    _ => {}
                }
                edit_string(&mut self.replay_instrument_query, key);
                self.sync_replay_dataset_filter();
            }
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
                let options = self.replay_dataset_options();
                let count = options.len();
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
                        if count == 1 && self.replay_dataset_index.is_some() {
                            if key.code == KeyCode::Down {
                                if let Some((index, bar_type)) = options.first().copied() {
                                    self.select_replay_dataset_option(index, bar_type);
                                }
                                self.open_replay_setup();
                            } else {
                                self.focus = Focus::ReplayInstrumentQuery;
                            }
                            return;
                        }
                        let current = self.replay_selected_option_position(&options).unwrap_or(0);
                        let next_position = if key.code == KeyCode::Up {
                            current.checked_sub(1).unwrap_or(count - 1)
                        } else {
                            (current + 1) % count
                        };
                        if let Some((index, bar_type)) = options.get(next_position).copied() {
                            self.select_replay_dataset_option(index, bar_type);
                        }
                        return;
                    }
                    KeyCode::Char('a') => {
                        self.replay_dataset_index = None;
                        self.replay_dataset_bar_type = None;
                        self.replay_dataset_view_path = None;
                        return;
                    }
                    KeyCode::Left => {
                        self.focus = Focus::ReplayInstrumentQuery;
                        return;
                    }
                    KeyCode::Right | KeyCode::Enter => {
                        let selected = self.replay_selected_option_position(&options).unwrap_or(0);
                        if let Some((index, bar_type)) = options.get(selected).copied() {
                            self.select_replay_dataset_option(index, bar_type);
                        }
                        self.open_replay_setup();
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
            | Focus::VolumeHmaLookbackBars
            | Focus::VolumeHmaInvertBelowRatio
            | Focus::AdxLength
            | Focus::AdxEntryThreshold
            | Focus::AdxExitThreshold
            | Focus::AdxDiImbalance
            | Focus::AdxSlopeLookback
            | Focus::AdxDominanceBars
            | Focus::AdxBreakoutLookback
            | Focus::AdxInverted
            | Focus::AdxTakeProfitTicks
            | Focus::AdxStopLossTicks
            | Focus::AdxTrailingStop
            | Focus::AdxTrailTriggerTicks
            | Focus::AdxTrailOffsetTicks
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
