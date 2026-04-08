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

        if self.screen == Screen::Login
            && !self.is_text_focus()
            && matches!(key.code, KeyCode::Char('r') | KeyCode::Char('R'))
        {
            if !self.broker_supports_replay() {
                self.push_log(format!(
                    "{} does not support replay mode in this build.",
                    self.selected_broker.label()
                ));
                return;
            }
            self.bar_type = BarType::Range1;
            let _ = cmd_tx.send(ServiceCommand::EnterReplayMode {
                config: self.current_config(),
                bar_type: self.bar_type,
            });
            self.push_log("Replay mode requested".to_string());
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
                self.screen = Screen::Stats;
                self.focus = Focus::AccountList;
                return;
            }
            KeyCode::Esc => {
                if self.screen == Screen::Login && self.available_brokers.len() > 1 {
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

        match self.screen {
            Screen::BrokerSelect => self.handle_broker_select_key(key),
            Screen::Login => self.handle_login_key(key, cmd_tx),
            Screen::Strategy => self.handle_strategy_key(key, cmd_tx),
            Screen::Selection => self.handle_selection_key(key, cmd_tx),
            Screen::Dashboard => self.handle_dashboard_key(key, cmd_tx),
            Screen::Stats => self.handle_session_stats_key(key, cmd_tx),
        }
    }
}
