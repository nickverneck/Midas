use super::*;

impl App {
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
    pub(super) fn open_selected_replay_downloader(&mut self) {
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
    pub(super) fn open_selected_replay_downloader(&mut self) {
        self.open_new_replay_downloader();
    }

    #[cfg(feature = "replay")]
    pub(super) fn handle_replay_downloader_key(
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
    pub(super) fn replay_selected_dataset_manifest(&self) -> Option<std::path::PathBuf> {
        self.replay_dataset_index
            .and_then(|index| self.replay_cache_library.datasets.get(index))
            .map(|dataset| dataset.manifest_path.clone())
    }

    #[cfg(not(feature = "replay"))]
    pub(super) fn replay_selected_dataset_manifest(&self) -> Option<std::path::PathBuf> {
        None
    }
}
