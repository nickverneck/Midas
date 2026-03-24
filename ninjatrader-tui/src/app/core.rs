impl App {
    pub fn new(config: AppConfig) -> Self {
        let form = FormState::from_config(&config);
        let mut app = Self {
            base_config: config,
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
            strategy_runtime: StrategyRuntimeState::default(),
            strategy_numeric_input: None,
            latency: LatencySnapshot::default(),
            last_market_update_at: None,
        };
        app.push_log(
            "Phase 1 enabled: auth, account selection, contract search, 1m or 1 range history/live."
                .to_string(),
        );
        app.push_log("Dashboard hotkeys enabled: b buy, s sell, c close.".to_string());
        app.push_log(
            "Native HMA Angle and EMA Crossover strategies can auto-trade closed 1m bars once armed from Strategy."
                .to_string(),
        );
        app
    }
    pub fn current_config(&self) -> AppConfig {
        let mut cfg = self.base_config.clone();
        cfg.env = self.form.env;
        cfg.auth_mode = self.form.auth_mode;
        cfg.token_override = self.form.token_override.clone();
        cfg.username = self.form.username.clone();
        cfg.password = self.form.password.clone();
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
            ServiceEvent::Error(message) => {
                self.status = format!("Error: {message}");
                self.push_log(format!("ERROR: {message}"));
            }
            ServiceEvent::Connected {
                env,
                user_name,
                auth_mode,
            } => {
                self.form.env = env;
                self.form.auth_mode = auth_mode;
                self.screen = Screen::Selection;
                self.focus = Focus::AccountList;
                self.status = match user_name {
                    Some(name) => format!("Connected to {} as {}", env.label(), name),
                    None => format!("Connected to {}", env.label()),
                };
                self.push_log(self.status.clone());
            }
            ServiceEvent::Disconnected => {
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                self.accounts.clear();
                self.account_snapshots.clear();
                self.contract_results.clear();
                self.market = MarketSnapshot::default();
                self.strategy_runtime = StrategyRuntimeState::default();
                self.latency = LatencySnapshot::default();
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
        }
    }

}
