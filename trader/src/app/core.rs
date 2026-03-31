impl App {
    pub fn new(config: AppConfig) -> Self {
        let available_brokers = compiled_brokers().to_vec();
        let selected_broker = if available_brokers.contains(&config.broker) {
            config.broker
        } else {
            default_broker()
        };
        let form = FormState::from_config(&config);
        let mut app = Self {
            base_config: config,
            available_brokers,
            selected_broker,
            capabilities: BrokerCapabilities::default(),
            form,
            strategy: StrategyState::new(),
            screen: Screen::Login,
            focus: Focus::Env,
            should_quit: false,
            status: "Idle".to_string(),
            accounts: Vec::new(),
            account_snapshots: Vec::new(),
            selected_account: 0,
            instrument_query: String::new(),
            bar_type: BarType::default(),
            contract_results: Vec::new(),
            selected_contract: 0,
            market: MarketSnapshot::default(),
            logs: VecDeque::new(),
            dashboard_visuals_enabled: false,
            strategy_runtime: StrategyRuntimeState::default(),
            strategy_numeric_input: None,
            latency: LatencySnapshot::default(),
            session_kind: SessionKind::Live,
            replay_speed: ReplaySpeed::default(),
            last_log_at: None,
            last_market_update_at: None,
        };
        if app.available_brokers.len() > 1 {
            app.screen = Screen::BrokerSelect;
            app.focus = Focus::BrokerList;
            app.status = format!("Select a broker to continue. Current: {}", app.selected_broker.label());
        }
        app.push_log(
            format!(
                "Broker support compiled in: {}.",
                app.available_brokers
                    .iter()
                    .map(|broker| broker.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
        app.push_log("Dashboard hotkeys enabled: b buy, s sell, c close, v visuals.".to_string());
        app.push_log(
            "Native HMA Angle and EMA Crossover strategies can auto-trade on closed bars or live forming bars once armed from Strategy."
                .to_string(),
        );
        app.push_log(
            if app.selected_broker == BrokerKind::Tradovate && cfg!(feature = "replay") {
                "Replay mode available from Login: local ES tick data can stream 1 Range or 1 Min bars."
            } else {
                "Replay mode is only available on Tradovate builds with `--features replay`."
            }
            .to_string(),
        );
        app
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
                self.form.env = env;
                self.form.auth_mode = auth_mode;
                self.session_kind = session_kind;
                self.replay_speed = ReplaySpeed::default();
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
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                self.capabilities = BrokerCapabilities::default();
                self.session_kind = SessionKind::Live;
                self.accounts.clear();
                self.account_snapshots.clear();
                self.contract_results.clear();
                self.market = MarketSnapshot::default();
                self.strategy_runtime = StrategyRuntimeState::default();
                self.latency = LatencySnapshot::default();
                self.replay_speed = ReplaySpeed::default();
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
                self.account_snapshots = snapshots;
            }
            ServiceEvent::ContractSearchResults { query, results } => {
                self.contract_results = results;
                self.selected_contract = 0;
                self.push_log(format!(
                    "Contract search `{query}` returned {} result(s)",
                    self.contract_results.len()
                ));
            }
            ServiceEvent::MarketSnapshot(snapshot) => {
                self.market = snapshot;
                self.last_market_update_at = Some(Instant::now());
            }
            ServiceEvent::TradeMarkersUpdated(markers) => {
                self.market.trade_markers = markers;
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
            ServiceEvent::ReplaySpeedUpdated(speed) => {
                self.replay_speed = speed;
            }
        }
    }
}
