impl LogEntry {
    fn render_line(&self) -> String {
        format!(
            "[{} | {}] {}",
            self.timestamp.format("%H:%M:%S%.3f"),
            format_log_elapsed(self.elapsed_since_previous),
            self.message
        )
    }
}

fn format_log_elapsed(duration: Option<std::time::Duration>) -> String {
    match duration.map(|value| value.as_millis()) {
        Some(value) if value >= 60_000 => format!("+{:.1}m", value as f64 / 60_000.0),
        Some(value) if value >= 1_000 => format!("+{:.1}s", value as f64 / 1_000.0),
        Some(value) => format!("+{value}ms"),
        None => "+0ms".to_string(),
    }
}

impl App {
    fn set_replay_speed(
        &mut self,
        cmd_tx: &UnboundedSender<ServiceCommand>,
        speed: ReplaySpeed,
    ) {
        if self.session_kind != SessionKind::Replay || self.replay_speed == speed {
            return;
        }
        self.replay_speed = speed;
        let _ = cmd_tx.send(ServiceCommand::SetReplaySpeed { speed });
    }

    fn sync_selected_account(&self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if let Some(account) = self.accounts.get(self.selected_account) {
            let _ = cmd_tx.send(ServiceCommand::SelectAccount {
                account_id: account.id,
            });
        }
    }

    fn selected_snapshot(&self) -> Option<&AccountSnapshot> {
        let account = self.accounts.get(self.selected_account)?;
        self.snapshot_for_account(account.id)
    }
    fn snapshot_for_account(&self, account_id: i64) -> Option<&AccountSnapshot> {
        self.account_snapshots
            .iter()
            .find(|snapshot| snapshot.account_id == account_id)
    }

    fn login_focus_order(&self) -> Vec<Focus> {
        vec![
            Focus::Env,
            Focus::AuthMode,
            Focus::LogMode,
            Focus::TokenPath,
            Focus::TokenOverride,
            Focus::Username,
            Focus::Password,
            Focus::AppId,
            Focus::AppVersion,
            Focus::Cid,
            Focus::Secret,
            Focus::Connect,
            Focus::ReplayMode,
        ]
    }

    fn strategy_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::StrategyKind, Focus::OrderQty];
        if self.strategy.kind == StrategyKind::Native {
            order.push(Focus::NativeStrategy);
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => order.extend([
                    Focus::HmaLength,
                    Focus::HmaMinAngle,
                    Focus::HmaAngleLookback,
                    Focus::HmaBarsRequired,
                    Focus::HmaLongsOnly,
                    Focus::HmaInverted,
                    Focus::HmaTakeProfitTicks,
                    Focus::HmaStopLossTicks,
                    Focus::HmaTrailingStop,
                    Focus::HmaTrailTriggerTicks,
                    Focus::HmaTrailOffsetTicks,
                ]),
                NativeStrategyKind::EmaCross => order.extend([
                    Focus::EmaFastLength,
                    Focus::EmaSlowLength,
                    Focus::EmaInverted,
                    Focus::EmaTakeProfitTicks,
                    Focus::EmaStopLossTicks,
                    Focus::EmaTrailingStop,
                    Focus::EmaTrailTriggerTicks,
                    Focus::EmaTrailOffsetTicks,
                ]),
            }
        } else if self.strategy.kind == StrategyKind::Lua {
            order.push(Focus::LuaSourceMode);
            match self.strategy.lua_source_mode {
                LuaSourceMode::File => order.push(Focus::LuaFilePath),
                LuaSourceMode::Editor => order.push(Focus::LuaEditor),
            }
        }
        order.push(Focus::StrategyContinue);
        order
    }

    fn selection_focus_order(&self) -> Vec<Focus> {
        vec![
            Focus::AccountList,
            Focus::BarTypeToggle,
            Focus::InstrumentQuery,
            Focus::ContractList,
        ]
    }

    fn next_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn next_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn next_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    fn is_text_focus(&self) -> bool {
        matches!(
            self.focus,
            Focus::TokenOverride
                | Focus::Username
                | Focus::Password
                | Focus::AppId
                | Focus::AppVersion
                | Focus::Cid
                | Focus::Secret
                | Focus::TokenPath
                | Focus::OrderQty
                | Focus::HmaLength
                | Focus::HmaMinAngle
                | Focus::HmaAngleLookback
                | Focus::HmaBarsRequired
                | Focus::HmaTakeProfitTicks
                | Focus::HmaStopLossTicks
                | Focus::HmaTrailTriggerTicks
                | Focus::HmaTrailOffsetTicks
                | Focus::EmaFastLength
                | Focus::EmaSlowLength
                | Focus::EmaTakeProfitTicks
                | Focus::EmaStopLossTicks
                | Focus::EmaTrailTriggerTicks
                | Focus::EmaTrailOffsetTicks
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::InstrumentQuery
        )
    }

    fn push_log(&mut self, message: String) {
        let now = std::time::Instant::now();
        while self.logs.len() >= 200 {
            self.logs.pop_front();
        }
        self.logs.push_back(LogEntry {
            timestamp: chrono::Local::now(),
            elapsed_since_previous: self.last_log_at.map(|last| now.duration_since(last)),
            message,
        });
        self.last_log_at = Some(now);
    }

    fn save_logs_to_file(&mut self) {
        match self.persist_logs_to_file() {
            Ok(path) => {
                let message = format!("Saved logs to {}", path.display());
                self.status = message.clone();
                self.push_log(message);
            }
            Err(err) => {
                let message = format!("ERROR: failed to save logs: {err}");
                self.status = message.clone();
                self.push_log(message);
            }
        }
    }

    fn persist_logs_to_file(&self) -> std::io::Result<std::path::PathBuf> {
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let dir = std::path::PathBuf::from(".run").join("ninjatrader-tui-logs");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("session-{timestamp}.txt"));

        let screen = match self.screen {
            Screen::Login => "Login",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
        };
        let selected_account = self
            .accounts
            .get(self.selected_account)
            .map(|account| account.name.as_str())
            .unwrap_or("none");
        let selected_contract = self
            .contract_results
            .get(self.selected_contract)
            .map(|contract| contract.name.as_str())
            .or(self.market.contract_name.as_deref())
            .unwrap_or("none");

        let mut body = String::new();
        body.push_str(&format!("saved_at_utc: {timestamp}\n"));
        body.push_str(&format!("screen: {screen}\n"));
        body.push_str(&format!("status: {}\n", self.status));
        body.push_str(&format!("env: {}\n", self.form.env.label()));
        body.push_str(&format!("auth_mode: {}\n", self.form.auth_mode.label()));
        body.push_str(&format!("log_mode: {}\n", self.form.log_mode.label()));
        body.push_str(&format!("strategy: {}\n", self.strategy.summary_label()));
        body.push_str(&format!("selected_account: {selected_account}\n"));
        body.push_str(&format!("selected_contract: {selected_contract}\n"));
        body.push_str(&format!("bar_type: {}\n", self.bar_type.label()));
        body.push_str("log_format: [HH:MM:SS.mmm local | +elapsed_since_previous] message\n");
        body.push('\n');

        for entry in &self.logs {
            body.push_str(&entry.render_line());
            body.push('\n');
        }

        std::fs::write(&path, body)?;
        Ok(path)
    }

    fn clear_strategy_numeric_input(&mut self) {
        self.strategy_numeric_input = None;
    }

    fn strategy_numeric_value(&self, focus: Focus, fallback: String) -> String {
        self.strategy_numeric_input
            .as_ref()
            .filter(|draft| draft.focus == focus)
            .map(|draft| draft.value.clone())
            .unwrap_or(fallback)
    }

    fn market_update_age_ms(&self) -> Option<u64> {
        self.last_market_update_at
            .map(|instant| instant.elapsed().as_millis() as u64)
    }

    fn latency_summary(&self) -> String {
        format!(
            "REST {} | Submit {} | Seen {} | Ack {} | Fill {} | Market {}",
            format_latency_ms(self.latency.rest_rtt_ms),
            format_latency_ms(self.latency.last_order_ack_ms),
            format_latency_ms(self.latency.last_order_seen_ms),
            format_latency_ms(self.latency.last_exec_report_ms),
            format_latency_ms(self.latency.last_fill_ms),
            format_age_ms(self.market_update_age_ms()),
        )
    }

    fn sync_execution_strategy_config(&self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        let _ = cmd_tx.send(ServiceCommand::SetExecutionStrategyConfig(
            self.strategy.execution_config(),
        ));
    }

    fn disarm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::DisarmExecutionStrategy {
            reason: "Native strategy config changed; press Continue to re-arm.".to_string(),
        });
    }

    fn arm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::ArmExecutionStrategy);
    }

    fn closed_bars(&self) -> &[crate::tradovate::Bar] {
        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        &self.market.bars[..closed_len]
    }

    fn latest_closed_bar_ts(&self) -> Option<i64> {
        self.closed_bars().last().map(|bar| bar.ts_ns)
    }

    fn session_window_at(&self, ts_ns: i64) -> Option<InstrumentSessionWindow> {
        self.market
            .session_profile
            .map(|profile| profile.evaluate(ts_ns))
    }

    fn latest_session_window(&self) -> Option<InstrumentSessionWindow> {
        self.latest_closed_bar_ts()
            .and_then(|ts_ns| self.session_window_at(ts_ns))
    }

    fn session_gate_summary(&self) -> String {
        let Some(profile) = self.market.session_profile else {
            return "n/a".to_string();
        };
        let Some(window) = self.latest_session_window() else {
            return format!("{} awaiting bars", profile.label());
        };

        if !window.session_open {
            return format!("{} closed; holding until reopen", profile.label());
        }

        let minutes_to_close = window
            .minutes_to_close
            .map(|minutes| format!("{minutes:.0}m"))
            .unwrap_or_else(|| "n/a".to_string());
        if window.hold_entries {
            format!(
                "{} hold active; flattening before close ({} left)",
                profile.label(),
                minutes_to_close
            )
        } else {
            format!("{} open; {} to close", profile.label(), minutes_to_close)
        }
    }

    fn strategy_runtime_summary(&self) -> String {
        if !self.strategy_runtime.last_summary.is_empty() {
            return self.strategy_runtime.last_summary.clone();
        }
        if self.strategy_runtime.armed {
            "Armed".to_string()
        } else {
            "Disarmed".to_string()
        }
    }

    fn effective_market_position_qty(&self) -> i32 {
        self.strategy_runtime.pending_target_qty.unwrap_or_else(|| {
            self.selected_snapshot()
                .and_then(|snapshot| snapshot.market_position_qty)
                .unwrap_or(0.0)
                .round() as i32
        })
    }

}
