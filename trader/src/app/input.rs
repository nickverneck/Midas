impl App {
    pub fn handle_key(&mut self, key: KeyEvent, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }

        if key.code == KeyCode::F(5)
            || (key.modifiers.contains(KeyModifiers::CONTROL)
                && key.code == KeyCode::Char('s'))
        {
            self.save_logs_to_file();
            return;
        }

        if self.screen == Screen::EngineSelect {
            if key.code == KeyCode::F(7) && self.replay_affordance_visible() {
                self.status =
                    "Attach or create an engine first, then press F7 for Replay.".to_string();
                self.push_log(self.status.clone());
                return;
            }
            self.handle_engine_select_key(key);
            return;
        }

        if self.screen == Screen::Login
            && !self.is_text_focus()
            && matches!(key.code, KeyCode::Char('r') | KeyCode::Char('R'))
        {
            if !self.replay_affordance_visible() {
                return;
            }
            self.screen = Screen::Replay;
            self.focus = Focus::BarTypeToggle;
            return;
        }

        match key.code {
            KeyCode::F(1) => {
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                return;
            }
            KeyCode::F(2) => {
                self.screen = Screen::Selection;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::F(3) => {
                self.screen = Screen::Strategy;
                self.focus = Focus::StrategyKind;
                return;
            }
            KeyCode::F(4) => {
                self.screen = Screen::Dashboard;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::F(6) => {
                if !self.session_stats_affordance_visible() {
                    return;
                }
                self.screen = Screen::Stats;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::F(7) => {
                if !self.replay_affordance_visible() {
                    return;
                }
                self.screen = Screen::Replay;
                self.focus = Focus::BarTypeToggle;
                return;
            }
            KeyCode::Esc => {
                if self.screen == Screen::Replay {
                    self.screen = Screen::Login;
                    self.focus = Focus::Env;
                } else if self.screen == Screen::Login && self.available_brokers.len() > 1 {
                    self.screen = Screen::BrokerSelect;
                    self.focus = Focus::BrokerList;
                } else {
                    self.screen = Screen::Login;
                    self.focus = Focus::Env;
                }
                return;
            }
            _ => {}
        }

        if !self.is_text_focus() && key.code == KeyCode::Char('q') {
            self.should_quit = true;
            return;
        }

        if !self.is_free_text_focus()
            && matches!(key.code, KeyCode::Char('d') | KeyCode::Char('D'))
        {
            if !self.automated_strategy_affordance_visible() {
                return;
            }
            self.manual_disarm_native_strategy(cmd_tx);
            return;
        }

        match self.screen {
            Screen::EngineSelect => self.handle_engine_select_key(key),
            Screen::BrokerSelect => self.handle_broker_select_key(key),
            Screen::Login => self.handle_login_key(key, cmd_tx),
            Screen::Replay => self.handle_replay_key(key, cmd_tx),
            Screen::Strategy => self.handle_strategy_key(key, cmd_tx),
            Screen::Selection => self.handle_selection_key(key, cmd_tx),
            Screen::Dashboard => self.handle_dashboard_key(key, cmd_tx),
            Screen::Stats => self.handle_session_stats_key(key, cmd_tx),
        }
    }
}
