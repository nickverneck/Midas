use super::*;

impl App {
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
    pub(in crate::app) fn open_new_replay_downloader(&mut self) {
        if self.retain_active_replay_downloader() {
            return;
        }
        let config = self.current_config();
        self.replay_downloader = ReplayDownloaderState::new(&config);
        self.replay_view = ReplayView::Downloader;
        self.focus = Focus::ReplayDownloadInstrument;
    }

    #[cfg(not(feature = "replay"))]
    pub(in crate::app) fn open_new_replay_downloader(&mut self) {
        self.status = "Replay downloader requires a replay-enabled build.".to_string();
        self.push_log(self.status.clone());
    }

    #[cfg(feature = "replay")]
    fn open_selected_replay_downloader(&mut self) {
        if self.retain_active_replay_downloader() {
            return;
        }
        let Some(index) = self.replay_dataset_index else {
            self.open_new_replay_downloader();
            return;
        };
        let Some(dataset) = self.replay_cache_library.datasets.get(index) else {
            self.status = "Selected replay dataset is no longer available.".to_string();
            self.push_log(self.status.clone());
            return;
        };
        let manifest = dataset.manifest.clone();
        let config = self.current_config();
        let mut downloader = ReplayDownloaderState::new(&config);
        downloader.workflow = ReplayDownloadWorkflow::Extend;
        downloader.target = Some(ReplayDownloadCacheTarget {
            dataset_dir: dataset.dataset_dir.clone(),
            manifest_path: dataset.manifest_path.clone(),
        });
        downloader.provider = manifest.provider;
        downloader.env = manifest.env;
        downloader.instrument_query = manifest.instrument.symbol.clone();
        downloader.exact_contract =
            manifest
                .contract
                .id
                .filter(|id| *id > 0)
                .map(|id| ContractSuggestion {
                    id,
                    name: manifest.contract.symbol.clone(),
                    description: "Owned cached contract".to_string(),
                    raw: manifest
                        .contract_metadata
                        .as_ref()
                        .map(|metadata| metadata.contract.payload.clone())
                        .unwrap_or(serde_json::Value::Null),
                });
        downloader.start_date = manifest
            .coverage
            .start
            .date_naive()
            .format("%Y-%m-%d")
            .to_string();
        downloader.end_date = manifest
            .coverage
            .end
            .date_naive()
            .format("%Y-%m-%d")
            .to_string();
        let sources = manifest.downloadable_source_kinds();
        downloader.source_kind = match sources.as_slice() {
            [source] => Some(*source),
            _ => None,
        };
        if let Some(bar_type) = manifest.files.iter().find_map(|file| {
            (file.source_kind == ReplayCacheSourceKind::ServerBars)
                .then_some(file.market_shape.bar_type)
                .flatten()
        }) {
            downloader.bar_type = bar_type;
        }
        downloader.display_name = manifest.display_name;
        downloader.tags = manifest.tags.join(", ");
        if let Some(suggestion) = manifest
            .contract_metadata
            .and_then(|metadata| metadata.suggested_coverage)
        {
            downloader.suggestion_basis = Some(suggestion.basis);
        }
        downloader.phase = if downloader.exact_contract.is_some() {
            ReplayDownloadPhase::Ready
        } else {
            ReplayDownloadPhase::Idle
        };
        downloader.phase_message = if downloader.exact_contract.is_none() {
            "Cached contract has no provider ID. Search and select the exact contract before extending."
                .to_string()
        } else if downloader.source_kind.is_some() {
            "Existing identity and coverage loaded. Edit fields or start the download.".to_string()
        } else {
            "Mixed dataset loaded. Explicitly choose server bars or raw ticks.".to_string()
        };
        self.replay_downloader = downloader;
        self.replay_view = ReplayView::Downloader;
        self.focus = if self.replay_downloader.exact_contract.is_none() {
            Focus::ReplayDownloadInstrument
        } else if self.replay_downloader.source_kind.is_some() {
            Focus::ReplayDownloadStart
        } else {
            Focus::ReplayDownloadSource
        };
    }

    #[cfg(not(feature = "replay"))]
    fn open_selected_replay_downloader(&mut self) {
        self.open_new_replay_downloader();
    }

    #[cfg(feature = "replay")]
    fn handle_replay_downloader_key(
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

        if self.replay_downloader.phase.is_busy() {
            return;
        }

        match self.focus {
            Focus::ReplayDownloadProvider => {
                if matches!(key.code, KeyCode::Down | KeyCode::Enter) {
                    self.focus = self.next_replay_focus();
                }
            }
            Focus::ReplayDownloadEnv => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    self.replay_downloader.cancel_active_operation(cmd_tx);
                    self.replay_downloader.env = self.replay_downloader.env.toggle();
                    self.replay_downloader.exact_contract = None;
                    self.replay_downloader.contract_results.clear();
                    self.replay_downloader.phase = ReplayDownloadPhase::Idle;
                    self.replay_downloader.phase_message =
                        "Environment changed; search and select the exact contract again."
                            .to_string();
                }
                KeyCode::Down | KeyCode::Enter => self.focus = self.next_replay_focus(),
                KeyCode::Up => self.focus = self.prev_replay_focus(),
                _ => {}
            },
            Focus::ReplayDownloadInstrument => {
                if key.code == KeyCode::Enter {
                    self.search_replay_download_contracts(cmd_tx);
                } else if key.code == KeyCode::Down {
                    self.focus = self.next_replay_focus();
                } else if key.code == KeyCode::Up {
                    self.focus = self.prev_replay_focus();
                } else {
                    let previous = self.replay_downloader.instrument_query.clone();
                    edit_string(&mut self.replay_downloader.instrument_query, key);
                    if previous != self.replay_downloader.instrument_query {
                        self.replay_downloader.cancel_active_operation(cmd_tx);
                        self.replay_downloader.exact_contract = None;
                        self.replay_downloader.contract_results.clear();
                    }
                }
            }
            Focus::ReplayDownloadContract => match key.code {
                KeyCode::Up => {
                    self.replay_downloader.selected_contract =
                        self.replay_downloader.selected_contract.saturating_sub(1);
                }
                KeyCode::Down => {
                    if self.replay_downloader.selected_contract + 1
                        < self.replay_downloader.contract_results.len()
                    {
                        self.replay_downloader.selected_contract += 1;
                    }
                }
                KeyCode::Enter => self.inspect_selected_replay_download_contract(cmd_tx),
                _ => {}
            },
            Focus::ReplayDownloadStart => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    edit_string(&mut self.replay_downloader.start_date, key);
                }
            }
            Focus::ReplayDownloadEnd => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    edit_string(&mut self.replay_downloader.end_date, key);
                }
            }
            Focus::ReplayDownloadSource => match key.code {
                KeyCode::Left => {
                    self.replay_downloader.source_kind = Some(ReplayCacheSourceKind::ServerBars)
                }
                KeyCode::Right => {
                    self.replay_downloader.source_kind = Some(ReplayCacheSourceKind::RawTicks)
                }
                KeyCode::Up => self.focus = self.prev_replay_focus(),
                KeyCode::Down | KeyCode::Enter => self.focus = self.next_replay_focus(),
                _ => {}
            },
            Focus::ReplayDownloadBarType => match key.code {
                KeyCode::Left => {
                    self.replay_downloader.bar_type =
                        self.replay_downloader.bar_type.previous_kind()
                }
                KeyCode::Right => {
                    self.replay_downloader.bar_type = self.replay_downloader.bar_type.next_kind()
                }
                KeyCode::Up => self.focus = self.prev_replay_focus(),
                KeyCode::Down | KeyCode::Enter => self.focus = self.next_replay_focus(),
                _ => {}
            },
            Focus::ReplayDownloadBarValue => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    let mut value = self.replay_downloader.bar_type.value() as usize;
                    if edit_strategy_usize(
                        &mut self.strategy_numeric_input,
                        Focus::ReplayDownloadBarValue,
                        &mut value,
                        key,
                        1,
                        1,
                    ) {
                        self.replay_downloader.bar_type = self
                            .replay_downloader
                            .bar_type
                            .with_value(value.min(u32::MAX as usize) as u32);
                    }
                }
            }
            Focus::ReplayDownloadCandleMode => match key.code {
                KeyCode::Left | KeyCode::Right => {
                    self.replay_downloader.candle_mode = self.replay_downloader.candle_mode.toggle()
                }
                KeyCode::Up => self.focus = self.prev_replay_focus(),
                KeyCode::Down | KeyCode::Enter => self.focus = self.next_replay_focus(),
                _ => {}
            },
            Focus::ReplayDownloadName => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    edit_string(&mut self.replay_downloader.display_name, key);
                }
            }
            Focus::ReplayDownloadTags => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    edit_string(&mut self.replay_downloader.tags, key);
                }
            }
            Focus::ReplayDownloadCacheRoot => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    self.focus = if key.code == KeyCode::Up {
                        self.prev_replay_focus()
                    } else {
                        self.next_replay_focus()
                    };
                } else {
                    edit_string(&mut self.replay_downloader.cache_root, key);
                }
            }
            Focus::ReplayDownloadSubmit => {
                if matches!(key.code, KeyCode::Enter | KeyCode::Char(' ')) {
                    self.submit_replay_download(cmd_tx);
                } else if key.code == KeyCode::Up {
                    self.focus = self.prev_replay_focus();
                }
            }
            _ => {}
        }
    }

    #[cfg(feature = "replay")]
    fn replay_downloader_config(&self) -> Result<AppConfig, String> {
        let cache_root = self.replay_downloader.cache_root.trim();
        if cache_root.is_empty() {
            return Err("Cache root cannot be empty.".to_string());
        }
        let mut config = self.current_config();
        config.broker = self.replay_downloader.provider;
        config.env = self.replay_downloader.env;
        config.replay_cache_dir = std::path::PathBuf::from(cache_root);
        Ok(config)
    }

    #[cfg(feature = "replay")]
    fn search_replay_download_contracts(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let query = self.replay_downloader.instrument_query.trim().to_string();
        if query.is_empty() {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message = "Enter an instrument query first.".to_string();
            return;
        }
        let config = match self.replay_downloader_config() {
            Ok(config) => config,
            Err(message) => {
                self.replay_downloader.phase = ReplayDownloadPhase::Failed;
                self.replay_downloader.phase_message = message;
                return;
            }
        };
        let operation_id = self.replay_downloader.begin_operation();
        let _ = cmd_tx.send(ServiceCommand::SearchReplayDownloadContracts {
            operation_id,
            config,
            query: query.clone(),
            limit: self.base_config.contract_suggest_limit.max(12),
        });
        self.replay_downloader.phase = ReplayDownloadPhase::Searching;
        self.replay_downloader.phase_message = format!("Searching for `{query}`...");
    }

    #[cfg(feature = "replay")]
    fn inspect_selected_replay_download_contract(
        &mut self,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        let Some(contract) = self
            .replay_downloader
            .contract_results
            .get(self.replay_downloader.selected_contract)
            .cloned()
        else {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message = "Select a contract result first.".to_string();
            return;
        };
        let config = match self.replay_downloader_config() {
            Ok(config) => config,
            Err(message) => {
                self.replay_downloader.phase = ReplayDownloadPhase::Failed;
                self.replay_downloader.phase_message = message;
                return;
            }
        };
        let operation_id = self.replay_downloader.begin_operation();
        let _ = cmd_tx.send(ServiceCommand::InspectReplayDownloadContract {
            operation_id,
            config,
            contract,
        });
        self.replay_downloader.phase = ReplayDownloadPhase::InspectingContract;
        self.replay_downloader.phase_message =
            "Loading contract metadata and broad coverage suggestion...".to_string();
    }

    #[cfg(feature = "replay")]
    fn submit_replay_download(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let Some(contract) = self.replay_downloader.exact_contract.clone() else {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message =
                "Search and select an exact contract before downloading.".to_string();
            return;
        };
        if contract.id <= 0 {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message =
                "Selected contract is missing a valid provider ID. Search and select it again."
                    .to_string();
            return;
        }
        let Some(source_kind) = self.replay_downloader.source_kind else {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message =
                "Choose server bars or raw ticks explicitly.".to_string();
            return;
        };
        let start_date = match chrono::NaiveDate::parse_from_str(
            self.replay_downloader.start_date.trim(),
            "%Y-%m-%d",
        ) {
            Ok(date) => date,
            Err(_) => {
                self.replay_downloader.phase = ReplayDownloadPhase::Failed;
                self.replay_downloader.phase_message =
                    "Start date must use YYYY-MM-DD.".to_string();
                return;
            }
        };
        let end_date = match chrono::NaiveDate::parse_from_str(
            self.replay_downloader.end_date.trim(),
            "%Y-%m-%d",
        ) {
            Ok(date) if date >= start_date => date,
            _ => {
                self.replay_downloader.phase = ReplayDownloadPhase::Failed;
                self.replay_downloader.phase_message =
                    "End date must use YYYY-MM-DD and be on or after start.".to_string();
                return;
            }
        };
        let instrument = self.replay_downloader.instrument_query.trim().to_string();
        if instrument.is_empty() {
            self.replay_downloader.phase = ReplayDownloadPhase::Failed;
            self.replay_downloader.phase_message = "Instrument cannot be empty.".to_string();
            return;
        }
        let config = match self.replay_downloader_config() {
            Ok(config) => config,
            Err(message) => {
                self.replay_downloader.phase = ReplayDownloadPhase::Failed;
                self.replay_downloader.phase_message = message;
                return;
            }
        };
        let display_name = self.replay_downloader.display_name.trim().to_string();
        let tags = self
            .replay_downloader
            .tags
            .split(',')
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .map(ToString::to_string)
            .collect();
        let operation_id = self.replay_downloader.begin_operation();
        let _ = cmd_tx.send(ServiceCommand::DownloadReplayData {
            operation_id,
            config,
            instrument,
            contract,
            target: self.replay_downloader.target.clone(),
            start_date,
            end_date,
            source_kind: source_kind.badge().to_string(),
            bar_type: self.replay_downloader.bar_type,
            candle_mode: self.replay_downloader.candle_mode,
            display_name: (!display_name.is_empty()).then_some(display_name),
            tags,
        });
        self.replay_downloader.phase = ReplayDownloadPhase::Authenticating;
        self.replay_downloader.phase_message = "Download queued; authenticating...".to_string();
        self.replay_downloader.actual_rows = None;
        self.replay_downloader.actual_bytes = None;
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
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Downloader {
            self.render_replay_downloader_screen(frame, area);
            return;
        }

        let compact = area.width < 120 || area.height < 28;
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(area);

        let dataset_lines = if compact {
            self.replay_dataset_library_compact_lines()
        } else {
            self.replay_dataset_library_lines()
        };
        let dataset = Paragraph::new(dataset_lines)
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

        let market_lines = if compact {
            self.replay_market_control_compact_lines()
        } else {
            self.replay_market_control_lines()
        };
        let controls = Paragraph::new(market_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Replay Market"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(controls, right[0]);

        let run_lines = if compact {
            self.replay_run_control_compact_lines()
        } else {
            self.replay_run_control_lines()
        };
        let run = Paragraph::new(run_lines)
            .block(Block::default().borders(Borders::ALL).title("Run"))
            .wrap(Wrap { trim: true });
        frame.render_widget(run, right[1]);
    }

    #[cfg(feature = "replay")]
    fn render_replay_downloader_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let compact = area.width < 120 || area.height < 28;
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(if compact {
                [Constraint::Percentage(58), Constraint::Percentage(42)]
            } else {
                [Constraint::Percentage(52), Constraint::Percentage(48)]
            })
            .split(area);
        let downloader = &self.replay_downloader;
        let source_label = downloader
            .source_kind
            .map(|source| source.label())
            .unwrap_or("choose explicitly");
        let exact_label = downloader
            .exact_contract
            .as_ref()
            .map(|contract| format!("{} (#{} exact)", contract.name, contract.id))
            .unwrap_or_else(|| "not selected".to_string());
        let mut form_lines = vec![
            Line::from(format!(
                "Workflow: {} | Esc returns to library",
                match downloader.workflow {
                    ReplayDownloadWorkflow::New => "New dataset",
                    ReplayDownloadWorkflow::Extend => "Extend / refresh",
                }
            )),
            styled_line(
                format!("Provider: {} (only supported)", downloader.provider.label()),
                self.focus == Focus::ReplayDownloadProvider,
            ),
            styled_line(
                format!("Environment: {}", downloader.env.label()),
                self.focus == Focus::ReplayDownloadEnv,
            ),
            styled_line(
                format!("Instrument query: {}", downloader.instrument_query),
                self.focus == Focus::ReplayDownloadInstrument,
            ),
            styled_line(
                format!("Exact contract: {exact_label}"),
                self.focus == Focus::ReplayDownloadContract,
            ),
            styled_line(
                format!("Coverage start: {}", downloader.start_date),
                self.focus == Focus::ReplayDownloadStart,
            ),
            styled_line(
                format!("Coverage end: {} (inclusive)", downloader.end_date),
                self.focus == Focus::ReplayDownloadEnd,
            ),
            styled_line(
                format!("Source: {source_label} (Left bars / Right ticks)"),
                self.focus == Focus::ReplayDownloadSource,
            ),
        ];
        if downloader.source_kind == Some(ReplayCacheSourceKind::ServerBars) {
            form_lines.extend([
                styled_line(
                    format!("Server bar: {}", downloader.bar_type.kind().label()),
                    self.focus == Focus::ReplayDownloadBarType,
                ),
                styled_line(
                    format!("Bar value: {}", downloader.bar_type.value()),
                    self.focus == Focus::ReplayDownloadBarValue,
                ),
                styled_line(
                    format!("Candles: {}", downloader.candle_mode.label()),
                    self.focus == Focus::ReplayDownloadCandleMode,
                ),
            ]);
        }
        form_lines.extend([
            styled_line(
                format!("Dataset name: {}", downloader.display_name),
                self.focus == Focus::ReplayDownloadName,
            ),
            styled_line(
                format!("Tags (comma-separated): {}", downloader.tags),
                self.focus == Focus::ReplayDownloadTags,
            ),
            styled_line(
                format!("Cache root: {}", downloader.cache_root),
                self.focus == Focus::ReplayDownloadCacheRoot,
            ),
            styled_line(
                "[Enter] Start read-only download".to_string(),
                self.focus == Focus::ReplayDownloadSubmit,
            ),
        ]);
        let form_scroll = focused_paragraph_scroll_offset(&form_lines, columns[0]);
        let form = Paragraph::new(form_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Dataset Downloader"),
            )
            .scroll((form_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(form, columns[0]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if compact {
                [Constraint::Length(6), Constraint::Min(5)]
            } else {
                [Constraint::Percentage(52), Constraint::Percentage(48)]
            })
            .split(columns[1]);
        let results = if downloader.contract_results.is_empty() {
            vec![ListItem::new(Line::from("Enter on Instrument to search"))]
        } else {
            downloader
                .contract_results
                .iter()
                .enumerate()
                .map(|(index, contract)| {
                    ListItem::new(styled_line(
                        format!(
                            "{} #{} | {}",
                            contract.name, contract.id, contract.description
                        ),
                        self.focus == Focus::ReplayDownloadContract
                            && index == downloader.selected_contract,
                    ))
                })
                .collect()
        };
        let contracts = List::new(results)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Exact Contract"),
            )
            .scroll_padding(1);
        let mut contract_state = focused_list_state(
            self.focus == Focus::ReplayDownloadContract,
            downloader.selected_contract,
            downloader.contract_results.len(),
        );
        frame.render_stateful_widget(contracts, right[0], &mut contract_state);

        let estimate = match (downloader.estimated_rows, downloader.estimated_bytes) {
            (Some(rows), Some(bytes)) => format!("{rows} rows / {bytes} bytes"),
            (Some(rows), None) => format!("{rows} rows / bytes unavailable"),
            (None, Some(bytes)) => format!("rows unavailable / {bytes} bytes"),
            (None, None) => "unavailable until provider responds".to_string(),
        };
        let actual = match (downloader.actual_rows, downloader.actual_bytes) {
            (Some(rows), Some(bytes)) => format!("{rows} rows / {bytes} bytes"),
            _ => "pending".to_string(),
        };
        let mut status_lines = vec![
            Line::from(format!("Phase: {}", downloader.phase.label())),
            Line::from(downloader.phase_message.clone()),
            Line::from(format!("Estimate: {estimate}")),
            Line::from(format!("Actual: {actual}")),
        ];
        if let Some(basis) = &downloader.suggestion_basis {
            status_lines.push(Line::from(format!("Range basis: {basis}")));
        }
        status_lines.push(Line::from(
            "Read-only REST + market-data WS; no user/account/order streams.",
        ));
        match downloader.phase {
            ReplayDownloadPhase::Searching
            | ReplayDownloadPhase::InspectingContract
            | ReplayDownloadPhase::Authenticating
            | ReplayDownloadPhase::Downloading => {
                status_lines.push(Line::from("Esc requests cancellation."));
                status_lines.push(Line::from("Wait for acknowledgement."));
            }
            ReplayDownloadPhase::Cancelling => {
                status_lines.push(Line::from("Cancellation requested."));
                status_lines.push(Line::from("Waiting for acknowledgement."));
            }
            ReplayDownloadPhase::WritingCache => {
                status_lines.push(Line::from("Commit is non-interruptible."));
                status_lines.push(Line::from("Esc will not hide this job."));
            }
            ReplayDownloadPhase::Idle
            | ReplayDownloadPhase::Ready
            | ReplayDownloadPhase::Busy
            | ReplayDownloadPhase::Cancelled
            | ReplayDownloadPhase::Complete
            | ReplayDownloadPhase::Failed => {}
        }
        let status = Paragraph::new(status_lines)
            .block(Block::default().borders(Borders::ALL).title("Download Job"))
            .wrap(Wrap { trim: true });
        frame.render_widget(status, right[1]);
    }
}
