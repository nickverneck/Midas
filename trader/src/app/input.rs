impl App {
    pub fn handle_key(&mut self, key: KeyEvent, cmd_tx: &ServiceCommandSender) {
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

        #[cfg(feature = "replay")]
        if matches!(
            key.code,
            KeyCode::F(1)
                | KeyCode::F(2)
                | KeyCode::F(3)
                | KeyCode::F(4)
                | KeyCode::F(6)
                | KeyCode::F(7)
                | KeyCode::F(8)
        ) && self.retain_active_replay_downloader()
        {
            return;
        }

        if self.screen == Screen::EngineSelect {
            if key.code == KeyCode::F(7) && self.replay_affordance_visible() {
                self.status =
                    "Choose Replay from the create-engine modal.".to_string();
                self.push_log(self.status.clone());
                return;
            }
            if key.code == KeyCode::F(8) && self.analytics_affordance_visible() {
                self.analytics_return_screen = self.screen;
                self.screen = Screen::Analytics;
                #[cfg(feature = "replay")]
                self.replay_analytics.refresh();
                return;
            }
            self.handle_engine_select_key(key);
            return;
        }

        if self.screen == Screen::Login
            && !self.is_text_focus()
            && matches!(key.code, KeyCode::Char('r') | KeyCode::Char('R'))
        {
            // Replay is selected at engine creation; the broker login screen
            // intentionally has no shortcut into it.
            return;
        }

        match key.code {
            KeyCode::F(1) => {
                if self.replay_navigation_active() {
                    return;
                }
                self.screen = Screen::Login;
                self.focus = Focus::Env;
                return;
            }
            KeyCode::F(2) => {
                if self.replay_navigation_active() {
                    return;
                }
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
                if !self.replay_navigation_active() {
                    return;
                }
                self.screen = Screen::Replay;
                #[cfg(feature = "replay")]
                {
                    self.replay_view = ReplayView::Library;
                }
                #[cfg(feature = "replay")]
                {
                    self.focus = Focus::ReplayInstrumentQuery;
                }
                return;
            }
            KeyCode::F(8) => {
                if !self.analytics_affordance_visible() {
                    return;
                }
                self.analytics_return_screen = self.screen;
                self.screen = Screen::Analytics;
                #[cfg(feature = "replay")]
                self.replay_analytics.refresh();
                return;
            }
            KeyCode::Esc => {
                #[cfg(feature = "replay")]
                if self.screen == Screen::Replay && self.replay_view == ReplayView::Downloader {
                    if let Some(operation_id) = self.replay_downloader.active_operation_id {
                        if self.replay_downloader.phase != ReplayDownloadPhase::Cancelling {
                            let _ = cmd_tx.send(ServiceCommand::CancelReplayDownloadOperation {
                                operation_id,
                            });
                        }
                        if self.replay_downloader.phase == ReplayDownloadPhase::WritingCache {
                            self.replay_downloader.phase_message =
                                "The current atomic cache commit will finish; cancellation will stop any remaining resumable chunks."
                                    .to_string();
                        } else {
                            self.replay_downloader.phase = ReplayDownloadPhase::Cancelling;
                            self.replay_downloader.phase_message =
                                "Cancellation requested; waiting for the downloader to confirm whether cache commit began."
                                    .to_string();
                        }
                        return;
                    }
                    self.replay_view = ReplayView::Library;
                    self.focus = Focus::ReplayInstrumentQuery;
                    return;
                }
                #[cfg(feature = "replay")]
                if self.screen == Screen::Replay && self.replay_view == ReplayView::DatasetViews {
                    if self.replay_dataset_views.editor.take().is_some() {
                        self.focus = Focus::ReplayViewList;
                        self.replay_dataset_views.message =
                            "Edit cancelled; no view changes were saved.".to_string();
                    } else {
                        self.replay_view = ReplayView::Library;
                        self.focus = Focus::ReplayInstrumentQuery;
                    }
                    return;
                }
                #[cfg(feature = "replay")]
                if self.screen == Screen::Replay && self.replay_view == ReplayView::Setup {
                    self.open_replay_library();
                    self.status = "Replay setup closed; select a cached dataset.".to_string();
                    return;
                }
                if self.screen == Screen::Replay {
                    self.screen = Screen::EngineSelect;
                    self.focus = Focus::EngineList;
                    self.status = "Replay workflow closed; select an engine to continue.".to_string();
                } else if self.screen == Screen::Analytics {
                    self.screen = self.analytics_return_screen;
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

        if self.screen != Screen::Replay
            && !self.is_free_text_focus()
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
            Screen::Analytics => self.handle_analytics_key(key),
        }
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn retain_active_replay_downloader(&mut self) -> bool {
        if self.replay_downloader.active_operation_id.is_none() {
            return false;
        }

        self.screen = Screen::Replay;
        self.replay_view = ReplayView::Downloader;
        if !matches!(
            self.focus,
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
                | Focus::ReplayDownloadSubmit
        ) {
            self.focus = Focus::ReplayDownloadSubmit;
        }
        self.status = match self.replay_downloader.phase {
            ReplayDownloadPhase::WritingCache =>
                "Replay cache commit is active; wait for completion before navigating.",
            ReplayDownloadPhase::Cancelling =>
                "Replay cancellation is pending; wait for confirmation before navigating.",
            _ => "Replay downloader is active; press Esc to cancel or wait for completion.",
        }
        .to_string();
        true
    }
}
