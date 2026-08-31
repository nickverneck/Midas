use super::*;

impl App {
    pub(in crate::app) fn handle_login_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &ServiceCommandSender,
    ) {
        match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                self.focus = self.prev_login_focus();
                return;
            }
            KeyCode::Down | KeyCode::Tab => {
                self.focus = self.next_login_focus();
                return;
            }
            _ => {}
        }

        match self.focus {
            Focus::Env => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.form.env = self.form.env.toggle();
                }
                _ => {}
            },
            Focus::AuthMode => match key.code {
                KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.form.auth_mode = self.form.auth_mode.toggle();
                }
                _ => {}
            },
            Focus::LogMode => match key.code {
                KeyCode::Left => self.form.log_mode = self.form.log_mode.previous(),
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    self.form.log_mode = self.form.log_mode.next()
                }
                _ => {}
            },
            Focus::Connect => {
                if key.code == KeyCode::Enter {
                    let cfg = self.current_config();
                    let _ = cmd_tx.send(ServiceCommand::Connect(cfg));
                    self.push_log("Connect requested".to_string());
                }
            }
            Focus::ReplayMode => {
                if key.code == KeyCode::Enter {
                    if !self.replay_navigation_active() {
                        return;
                    }
                    self.screen = Screen::Replay;
                    self.focus = Focus::BarTypeToggle;
                }
            }
            Focus::TokenOverride => edit_string(&mut self.form.token_override, key),
            Focus::Username => edit_string(&mut self.form.username, key),
            Focus::Password => edit_string(&mut self.form.password, key),
            Focus::ApiKey => edit_string(&mut self.form.api_key, key),
            Focus::AppId => edit_string(&mut self.form.app_id, key),
            Focus::AppVersion => edit_string(&mut self.form.app_version, key),
            Focus::Cid => edit_string(&mut self.form.cid, key),
            Focus::Secret => edit_string(&mut self.form.secret, key),
            Focus::TokenPath => edit_string(&mut self.form.token_path, key),
            Focus::EngineList
            | Focus::BrokerList
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
            | Focus::BarTypeToggle
            | Focus::BarValue
            | Focus::CandleModeToggle
            | Focus::ContractList => {}
            #[cfg(feature = "replay")]
            Focus::ReplayInstrumentQuery
            | Focus::ReplayDataset
            | Focus::ReplayInitialCapital
            | Focus::ReplayMarginPerContract
            | Focus::ReplaySafetyBuffer
            | Focus::ReplaySafetyBufferPercent
            | Focus::ReplayViewList
            | Focus::ReplayViewId
            | Focus::ReplayViewPreset
            | Focus::ReplayViewTradingDate
            | Focus::ReplayViewStart
            | Focus::ReplayViewEnd
            | Focus::ReplayViewTimezone
            | Focus::ReplayViewWarmupMinutes
            | Focus::ReplayViewSave
            | Focus::ReplayDownloadProvider
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

    pub(in crate::app) fn render_login_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(area);

        let connection_lines = self.connection_lines();
        let connection_scroll = focused_paragraph_scroll_offset(&connection_lines, columns[0]);
        let login = Paragraph::new(connection_lines)
            .block(Block::default().borders(Borders::ALL).title("Login"))
            .scroll((connection_scroll, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(login, columns[0]);

        let side = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(14), Constraint::Min(10)])
            .split(columns[1]);

        let notes = Paragraph::new(self.login_notes_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Auth Priority"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(notes, side[0]);

        let status = Paragraph::new(self.login_status_lines())
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        frame.render_widget(status, side[1]);
    }
}
