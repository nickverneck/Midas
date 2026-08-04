impl App {
    pub fn new(config: AppConfig) -> Self {
        #[cfg(feature = "replay")]
        let mut config = config;
        #[cfg(feature = "replay")]
        let replay_cache_root_was_resolved = {
            let configured_cache_root = config.replay_cache_dir.clone();
            let resolved_cache_root = replay_cache_root_for_app(&configured_cache_root);
            let changed = resolved_cache_root != configured_cache_root;
            if changed {
                config.replay_cache_dir = resolved_cache_root;
            }
            changed
        };
        let session_stats_enabled = config.session_stats_enabled;
        let candle_mode = config.candle_mode;
        let available_brokers = compiled_brokers().to_vec();
        let selected_broker = if available_brokers.contains(&config.broker) {
            config.broker
        } else {
            default_broker()
        };
        let form = FormState::from_config(&config);
        #[cfg(feature = "replay")]
        let replay_cache_library = ReplayCacheLibrary::scan(&config.replay_cache_dir);
        #[cfg(feature = "replay")]
        let replay_downloader = ReplayDownloaderState::new(&config);
        #[cfg(feature = "replay")]
        let replay_analytics = ReplayAnalyticsState::new(config.replay_result_dir.clone());
        let mut app = Self {
            base_config: config,
            #[cfg(feature = "replay")]
            replay_cache_library,
            available_brokers,
            selected_broker,
            capabilities: BrokerCapabilities::default(),
            form,
            strategy: StrategyState::new(),
            screen: Screen::EngineSelect,
            focus: Focus::EngineList,
            running_engines: Vec::new(),
            engine_summaries: Vec::new(),
            selected_engine: 0,
            engine_creation_enabled: true,
            pending_engine_lifecycle_confirmation: None,
            pending_engine_create_mode: None,
            pending_engine_selection_action: None,
            engine_socket_path: None,
            active_engine_key: None,
            should_quit: false,
            status: "Idle".to_string(),
            accounts: Vec::new(),
            account_snapshots: Vec::new(),
            selected_account: 0,
            instrument_query: String::new(),
            bar_type: BarType::default(),
            candle_mode,
            contract_results: Vec::new(),
            selected_contract: 0,
            pending_contract_override: None,
            market: MarketSnapshot::default(),
            logs: VecDeque::new(),
            persisted_logs: VecDeque::new(),
            last_saved_log_path: None,
            session_stats: SessionStatsState::new(session_stats_enabled),
            engine_history: None,
            session_stats_show_fees: true,
            dashboard_visuals_enabled: false,
            strategy_runtime: StrategyRuntimeState::default(),
            strategy_numeric_input: None,
            latency: LatencySnapshot::default(),
            session_mode: EngineCreateMode::Broker,
            session_kind: SessionKind::Live,
            replay_speed: ReplaySpeed::default(),
            replay_execution_ledger: ReplayExecutionLedgerSummary::default(),
            #[cfg(feature = "replay")]
            replay_dataset_index: None,
            #[cfg(feature = "replay")]
            replay_dataset_bar_type: None,
            #[cfg(feature = "replay")]
            replay_instrument_query: String::new(),
            #[cfg(feature = "replay")]
            replay_dataset_view_path: None,
            #[cfg(feature = "replay")]
            replay_view: ReplayView::Library,
            #[cfg(feature = "replay")]
            replay_downloader,
            #[cfg(feature = "replay")]
            replay_dataset_views: ReplayDatasetViewsState::default(),
            #[cfg(feature = "replay")]
            replay_analytics,
            analytics_return_screen: Screen::EngineSelect,
            last_log_at: None,
            last_market_update_at: None,
        };
        app.normalize_market_controls_for_broker();
        app.status = "Select an engine to continue or create a new one.".to_string();
        app.push_log(format!(
            "Broker support compiled in: {}.",
            app.available_brokers
                .iter()
                .map(|broker| broker.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        app.push_log("Dashboard visual overlays can be toggled with v.".to_string());
        app.push_log(
            "Native HMA Angle and EMA Crossover strategies can auto-trade on closed bars or live forming bars once armed from Strategy."
                .to_string(),
        );
        app.push_log(
            if app.selected_broker == BrokerKind::Tradovate && cfg!(feature = "replay") {
                "Replay mode available on F7: local price ticks can derive seconds, minutes, tick-count, and range bars."
            } else {
                "Replay mode is only available on Tradovate builds with `--features replay`."
            }
            .to_string(),
        );
        #[cfg(feature = "replay")]
        if replay_cache_root_was_resolved {
            app.push_log(format!(
                "Replay cache root resolved to {}.",
                app.base_config.replay_cache_dir.display()
            ));
        }
        app.push_log(
            if app.session_stats.enabled {
                "Session stats tracking enabled: F6 opens balance-delta stats and F5/Ctrl+S includes them in saved logs."
            } else {
                "Session stats tracking disabled by config; set `session_stats_enabled = true` or TRADER_SESSION_STATS_ENABLED=1 to enable it."
            }
            .to_string(),
        );
        app
    }

    pub fn set_running_engines(&mut self, engines: Vec<RunningEngine>) {
        let mut previous = std::mem::take(&mut self.engine_summaries);
        self.engine_summaries = engines
            .iter()
            .map(|engine| {
                let key = EngineKey::from_socket_path(&engine.socket_path);
                if let Some(index) = previous.iter().position(|summary| summary.key == key) {
                    let mut summary = previous.swap_remove(index);
                    summary.refresh_from_running_engine(engine);
                    summary
                } else {
                    EngineSummary::from_running_engine(engine)
                }
            })
            .collect();
        self.running_engines = engines;
        self.clamp_selected_engine();
    }

    pub fn set_engine_creation_enabled(&mut self, enabled: bool) {
        self.engine_creation_enabled = enabled;
        self.clamp_selected_engine();
    }

    pub(crate) fn take_engine_selection_action(&mut self) -> Option<EngineSelectionAction> {
        self.pending_engine_selection_action.take()
    }

    #[cfg(test)]
    pub fn enter_engine_session(&mut self, socket_path: PathBuf) {
        let engine_key = EngineKey::from_socket_path(&socket_path);
        self.enter_engine_session_for_key(engine_key, socket_path);
    }

    pub fn enter_engine_session_for_key(&mut self, engine_key: EngineKey, socket_path: PathBuf) {
        let mode = self.engine_create_mode_for_key(&engine_key);
        self.enter_engine_session_for_key_with_mode(
            engine_key,
            socket_path,
            mode,
        );
    }

    pub fn enter_engine_session_for_key_with_mode(
        &mut self,
        engine_key: EngineKey,
        socket_path: PathBuf,
        mode: EngineCreateMode,
    ) {
        self.observe_live_engine_socket(socket_path.clone());
        let active_key = engine_key.clone();
        self.active_engine_key = Some(engine_key);
        self.engine_socket_path = Some(socket_path.clone());
        if let Some(summary) = self
            .engine_summaries
            .iter_mut()
            .find(|summary| summary.key == active_key)
        {
            summary.set_session_kind(match mode {
                EngineCreateMode::Broker => SessionKind::Live,
                EngineCreateMode::Replay => SessionKind::Replay,
            });
        }
        match mode {
            EngineCreateMode::Broker => self.move_to_initial_broker_screen(),
            EngineCreateMode::Replay => self.move_to_replay_screen(),
        }
        self.push_log(format!(
            "Attached to engine socket {}.",
            socket_path.display()
        ));
    }

    pub fn leave_active_engine_session(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.active_engine_key = None;
        self.engine_socket_path = None;
        self.screen = Screen::EngineSelect;
        self.focus = Focus::EngineList;
        self.pending_engine_create_mode = None;
        self.capabilities = BrokerCapabilities::default();
        self.session_mode = EngineCreateMode::Broker;
        self.session_kind = SessionKind::Live;
        self.accounts.clear();
        self.account_snapshots.clear();
        self.contract_results.clear();
        self.market = MarketSnapshot::default();
        self.strategy_runtime = StrategyRuntimeState::default();
        self.latency = LatencySnapshot::default();
        self.replay_speed = ReplaySpeed::default();
        self.replay_execution_ledger = ReplayExecutionLedgerSummary::default();
        self.last_market_update_at = None;
        self.status = message.clone();
        self.push_log(message);
    }

    pub fn observe_live_engine_socket(&mut self, socket_path: PathBuf) {
        let key = EngineKey::from_socket_path(&socket_path);
        if self
            .engine_summaries
            .iter()
            .any(|summary| summary.key == key)
        {
            return;
        }
        self.engine_summaries
            .push(EngineSummary::live_socket(socket_path));
    }

    fn move_to_initial_broker_screen(&mut self) {
        self.session_mode = EngineCreateMode::Broker;
        if self.available_brokers.len() > 1 {
            self.screen = Screen::BrokerSelect;
            self.focus = Focus::BrokerList;
            self.status = format!(
                "Select a broker to continue. Current: {}",
                self.selected_broker.label()
            );
        } else {
            self.screen = Screen::Login;
            self.focus = Focus::Env;
            self.status = format!("Login for {}", self.selected_broker.label());
        }
    }

    #[cfg(feature = "replay")]
    fn move_to_replay_screen(&mut self) {
        // Replay is currently backed by Tradovate's local replay service.  Do
        // not carry a broker choice from the live workflow into this mode.
        self.selected_broker = BrokerKind::Tradovate;
        self.session_mode = EngineCreateMode::Replay;
        self.session_kind = SessionKind::Replay;
        self.screen = Screen::Replay;
        self.focus = Focus::ReplayInstrumentQuery;
        self.replay_view = ReplayView::Library;
        self.replay_instrument_query.clear();
        if let Some((index, bar_type)) = self.replay_dataset_options().first().copied() {
            self.select_replay_dataset_option(index, bar_type);
        } else {
            self.replay_dataset_index = None;
            self.replay_dataset_bar_type = None;
        }
        self.status = "Replay engine ready; select a cached dataset.".to_string();
    }

    #[cfg(not(feature = "replay"))]
    fn move_to_replay_screen(&mut self) {
        // This branch is unreachable from the picker because the replay mode
        // option is only offered by replay-enabled builds. Keep a safe live
        // fallback for callers/tests compiled without that feature.
        self.move_to_initial_broker_screen();
    }

    pub fn awaiting_broker_selection(&self) -> bool {
        self.screen == Screen::BrokerSelect
    }

    pub fn current_config(&self) -> AppConfig {
        let mut cfg = self.base_config.clone();
        cfg.broker = self.selected_broker;
        cfg.env = self.form.env;
        cfg.auth_mode = self.form.auth_mode;
        cfg.log_mode = self.form.log_mode;
        cfg.token_override = self.form.token_override.clone();
        cfg.username = self.form.username.clone();
        cfg.password = self.form.password.clone();
        cfg.api_key = self.form.api_key.clone();
        cfg.app_id = self.form.app_id.clone();
        cfg.app_version = self.form.app_version.clone();
        cfg.cid = self.form.cid.clone();
        cfg.secret = self.form.secret.clone();
        cfg.token_path = self.form.token_path.clone().into();
        cfg.candle_mode = self.effective_candle_mode();
        cfg
    }

    pub fn handle_service_event(
        &mut self,
        event: ServiceEvent,
        _cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        match event {
            ServiceEvent::Status(message) => {
                self.status = message.clone();
                self.push_log(message);
            }
            ServiceEvent::DebugLog(message) => {
                if self.form.log_mode == LogMode::Debug {
                    self.push_log(format!("DEBUG: {message}"));
                }
            }
            ServiceEvent::BrokerRejection(message) => {
                self.status = format!("Rejected: {message}");
                self.push_log(format!("REJECTED: {message}"));
            }
            ServiceEvent::Error(message) => {
                self.status = format!("Error: {message}");
                self.push_log(format!("ERROR: {message}"));
            }
            ServiceEvent::Connected {
                broker,
                env,
                user_name,
                auth_mode,
                session_kind,
                capabilities,
            } => {
                self.selected_broker = broker;
                self.capabilities = capabilities;
                self.normalize_market_controls_for_broker();
                self.form.env = env;
                self.form.auth_mode = auth_mode;
                self.session_kind = session_kind;
                self.session_mode = if session_kind == SessionKind::Replay {
                    EngineCreateMode::Replay
                } else {
                    EngineCreateMode::Broker
                };
                self.replay_speed = ReplaySpeed::default();
                self.replay_execution_ledger = ReplayExecutionLedgerSummary::default();
                if session_kind == SessionKind::Replay {
                    self.screen = Screen::Strategy;
                    self.focus = Focus::StrategyKind;
                } else {
                    self.screen = Screen::Selection;
                    self.focus = Focus::AccountList;
                }
                self.status = match user_name {
                    Some(name) => format!(
                        "Connected to {} {} as {}",
                        broker.label(),
                        env.label(),
                        name
                    ),
                    None => format!("Connected to {} {}", broker.label(), env.label()),
                };
                self.push_log(self.status.clone());
            }
            ServiceEvent::Disconnected => {
                let replay_mode = self.session_mode == EngineCreateMode::Replay;
                if self.engine_socket_path.is_none() {
                    self.screen = Screen::EngineSelect;
                    self.focus = Focus::EngineList;
                    self.session_mode = EngineCreateMode::Broker;
                    self.session_kind = SessionKind::Live;
                } else if replay_mode {
                    #[cfg(feature = "replay")]
                    {
                        self.screen = Screen::Replay;
                        self.focus = Focus::ReplayInstrumentQuery;
                        self.replay_view = ReplayView::Library;
                    }
                    #[cfg(not(feature = "replay"))]
                    {
                        self.screen = Screen::EngineSelect;
                        self.focus = Focus::EngineList;
                    }
                } else if self.available_brokers.len() > 1 {
                    self.screen = Screen::BrokerSelect;
                    self.focus = Focus::BrokerList;
                } else {
                    self.screen = Screen::Login;
                    self.focus = Focus::Env;
                }
                self.capabilities = BrokerCapabilities::default();
                if !replay_mode {
                    self.session_mode = EngineCreateMode::Broker;
                    self.session_kind = SessionKind::Live;
                } else {
                    self.session_kind = SessionKind::Replay;
                }
                self.accounts.clear();
                self.account_snapshots.clear();
                self.engine_history = None;
                self.contract_results.clear();
                self.market = MarketSnapshot::default();
                self.strategy_runtime = StrategyRuntimeState::default();
                self.latency = LatencySnapshot::default();
                self.replay_speed = ReplaySpeed::default();
                self.replay_execution_ledger = ReplayExecutionLedgerSummary::default();
                self.last_market_update_at = None;
                self.status = "Disconnected".to_string();
                self.push_log("Disconnected".to_string());
            }
            ServiceEvent::AccountsLoaded(accounts) => {
                self.accounts = accounts;
                if self.selected_account >= self.accounts.len() {
                    self.selected_account = 0;
                }
                self.push_log(format!("Loaded {} accounts", self.accounts.len()));
            }
            ServiceEvent::AccountSnapshotsLoaded(snapshots) => {
                self.record_session_stats(&snapshots);
                self.account_snapshots = snapshots;
            }
            ServiceEvent::ContractSearchResults { query, results } => {
                self.contract_results = results;
                self.selected_contract = 0;
                self.pending_contract_override = None;
                self.push_log(format!(
                    "Contract search `{query}` returned {} result(s)",
                    self.contract_results.len()
                ));
            }
            ServiceEvent::MarketSnapshot(snapshot) => {
                if snapshot.contract_id.is_some() || !snapshot.bars.is_empty() {
                    self.candle_mode = snapshot.candle_mode;
                }
                self.market = snapshot;
                self.last_market_update_at = Some(Instant::now());
            }
            ServiceEvent::TradeMarkersUpdated(markers) => {
                self.market.trade_markers = markers;
            }
            ServiceEvent::EngineHistoryUpdated(history) => {
                self.engine_history = Some(history);
            }
            ServiceEvent::Latency(snapshot) => {
                self.latency = snapshot;
            }
            ServiceEvent::ExecutionState(snapshot) => {
                self.strategy.apply_execution_config(&snapshot.config);
                self.strategy_runtime.armed = snapshot.runtime.armed;
                self.strategy_runtime.last_closed_bar_ts = snapshot.runtime.last_closed_bar_ts;
                self.strategy_runtime.pending_target_qty = snapshot.runtime.pending_target_qty;
                self.strategy_runtime.last_summary = snapshot.runtime.last_summary;
                if let Some(selected_account_id) = snapshot.selected_account_id {
                    if let Some(index) = self
                        .accounts
                        .iter()
                        .position(|account| account.id == selected_account_id)
                    {
                        self.selected_account = index;
                    }
                }
            }
            ServiceEvent::ExecutionProbe(_) => {}
            ServiceEvent::ReplaySpeedUpdated(speed) => {
                self.replay_speed = speed;
            }
            ServiceEvent::ReplayExecutionLedgerUpdated(summary) => {
                self.replay_execution_ledger = summary;
            }
            ServiceEvent::ReplayExecutionLedgerSnapshot(snapshot) => {
                self.replay_execution_ledger = snapshot.summary();
            }
            ServiceEvent::ReplayResultSaved {
                run_id,
                result_path,
                status,
                fill_count,
                trade_count,
            } => {
                self.status = format!(
                    "Replay {status}: {run_id} saved to {} ({} fills, {} trades)",
                    result_path.display(),
                    fill_count,
                    trade_count
                );
                self.push_log(self.status.clone());
            }
            ServiceEvent::ReplayDownloadProgress {
                operation_id,
                phase,
                message,
                estimated_rows,
                estimated_bytes,
            } => {
                #[cfg(feature = "replay")]
                {
                    if !self.replay_downloader.accepts(operation_id) {
                        return;
                    }
                    self.replay_downloader.phase = phase;
                    self.replay_downloader.phase_message = message.clone();
                    self.replay_downloader.estimated_rows = estimated_rows;
                    self.replay_downloader.estimated_bytes = estimated_bytes;
                }
                #[cfg(not(feature = "replay"))]
                let _ = (operation_id, phase, estimated_rows, estimated_bytes);
                self.status = message.clone();
                self.push_log(message);
            }
            ServiceEvent::ReplayDownloadContractSearchResults {
                operation_id,
                query,
                results,
            } => {
                #[cfg(feature = "replay")]
                {
                    if !self.replay_downloader.accepts(operation_id) {
                        return;
                    }
                    self.replay_downloader.invalidate_operation();
                    self.replay_downloader.contract_results = results;
                    self.replay_downloader.selected_contract = 0;
                    self.replay_downloader.exact_contract = None;
                    self.replay_downloader.phase = ReplayDownloadPhase::Idle;
                    self.replay_downloader.phase_message = format!(
                        "Search `{query}` returned {} contract(s). Select one exactly.",
                        self.replay_downloader.contract_results.len()
                    );
                }
                #[cfg(not(feature = "replay"))]
                let _ = (operation_id, results);
                self.push_log(format!("Replay contract search completed for `{query}`."));
            }
            ServiceEvent::ReplayDownloadContractInspected {
                operation_id,
                contract,
                suggested_start_date,
                suggested_end_date,
                suggestion_basis,
            } => {
                #[cfg(feature = "replay")]
                {
                    if !self.replay_downloader.accepts(operation_id) {
                        return;
                    }
                    self.replay_downloader.invalidate_operation();
                    self.replay_downloader.exact_contract = Some(contract.clone());
                    if self.replay_downloader.workflow == ReplayDownloadWorkflow::New {
                        if let Some(start) = suggested_start_date {
                            self.replay_downloader.start_date =
                                start.format("%Y-%m-%d").to_string();
                        }
                        if let Some(end) = suggested_end_date {
                            self.replay_downloader.end_date = end.format("%Y-%m-%d").to_string();
                        }
                    }
                    self.replay_downloader.suggestion_basis = suggestion_basis;
                    self.replay_downloader.phase = ReplayDownloadPhase::Ready;
                    self.replay_downloader.phase_message = if suggested_start_date.is_some()
                        && suggested_end_date.is_some()
                    {
                        "Exact contract selected; broad estimated coverage was applied and remains editable."
                            .to_string()
                    } else {
                        "Exact contract selected; coverage estimate unavailable, so dates remain user supplied."
                            .to_string()
                    };
                    self.focus = Focus::ReplayDownloadStart;
                }
                #[cfg(not(feature = "replay"))]
                let _ = (
                    operation_id,
                    suggested_start_date,
                    suggested_end_date,
                    suggestion_basis,
                );
                self.push_log(format!(
                    "Replay downloader selected exact contract {} (#{}).",
                    contract.name, contract.id
                ));
            }
            ServiceEvent::ReplayDownloadCompleted {
                operation_id,
                cache_root,
                manifest_path,
                data_path,
                rows,
                bytes,
            } => {
                #[cfg(not(feature = "replay"))]
                let _ = (&operation_id, &cache_root);
                #[cfg(feature = "replay")]
                {
                    if !self.replay_downloader.accepts(operation_id) {
                        return;
                    }
                    let downloaded_bar_type = self.replay_downloader.bar_type;
                    self.replay_downloader.invalidate_operation();
                    self.base_config.replay_cache_dir = cache_root.clone();
                    self.replay_cache_library = ReplayCacheLibrary::scan(&cache_root);
                    if let Some(index) = self
                        .replay_cache_library
                        .datasets
                        .iter()
                        .position(|dataset| dataset.manifest_path == manifest_path)
                    {
                        self.select_replay_dataset_option(index, Some(downloaded_bar_type));
                    } else {
                        self.replay_dataset_index = None;
                        self.replay_dataset_bar_type = None;
                    }
                    self.replay_dataset_view_path = None;
                    self.replay_downloader.phase = ReplayDownloadPhase::Complete;
                    self.replay_downloader.phase_message =
                        "Cache committed and dataset library refreshed.".to_string();
                    self.replay_downloader.actual_rows = Some(rows);
                    self.replay_downloader.actual_bytes = Some(bytes);
                }
                self.status = format!(
                    "Replay download complete: {rows} rows, {bytes} bytes ({})",
                    manifest_path.display()
                );
                self.push_log(format!(
                    "Replay cache updated: manifest={} data={}",
                    manifest_path.display(),
                    data_path.display()
                ));
            }
            ServiceEvent::ReplayDownloadFailed {
                operation_id,
                phase,
                message,
            } => {
                #[cfg(not(feature = "replay"))]
                let _ = operation_id;
                #[cfg(feature = "replay")]
                {
                    if !self.replay_downloader.accepts(operation_id) {
                        return;
                    }
                    self.replay_downloader.invalidate_operation();
                    self.replay_downloader.phase = match phase {
                        ReplayDownloadPhase::Busy | ReplayDownloadPhase::Cancelled => phase,
                        _ => ReplayDownloadPhase::Failed,
                    };
                    self.replay_downloader.phase_message =
                        format!("{} failed: {message}", phase.label());
                }
                self.status = format!("Replay download error: {message}");
                self.push_log(format!("ERROR: replay {}: {message}", phase.label()));
            }
        }
    }

    pub fn handle_engine_service_event(
        &mut self,
        engine_key: EngineKey,
        event: ServiceEvent,
        is_active_detail: bool,
        cmd_tx: &UnboundedSender<ServiceCommand>,
    ) {
        if let Some(summary) = self
            .engine_summaries
            .iter_mut()
            .find(|summary| summary.key == engine_key)
        {
            summary.apply_event(&event);
        }

        if is_active_detail {
            self.handle_service_event(event, cmd_tx);
        }
    }

    pub fn handle_engine_receiver_closed(
        &mut self,
        engine_key: &EngineKey,
        is_active_detail: bool,
    ) {
        let message = self.engine_receiver_closed_message(engine_key);
        if let Some(summary) = self
            .engine_summaries
            .iter_mut()
            .find(|summary| &summary.key == engine_key)
        {
            summary.mark_receiver_closed(message.clone());
        }
        if is_active_detail {
            self.leave_active_engine_session(message);
        }
    }
}

#[cfg(feature = "replay")]
fn replay_cache_root_for_app(configured: &std::path::Path) -> std::path::PathBuf {
    // Keep explicit cache paths authoritative. The fallback is only for the
    // built-in per-user default, so a configured deployment path is never
    // silently replaced by a repository-local cache.
    if cfg!(test)
        || std::env::var_os("TRADER_DATA_CACHE_DIR").is_some()
        || !configured.ends_with(".local/share/trader/replay-cache")
    {
        return configured.to_path_buf();
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(".run/replay-cache"));
    }
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".run/replay-cache"));

    candidates
        .into_iter()
        .find(|candidate| !crate::replay_cache::ReplayCacheLibrary::scan(candidate).datasets.is_empty())
        .unwrap_or_else(|| configured.to_path_buf())
}
