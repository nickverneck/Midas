impl App {
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
            Focus::TokenPath,
            Focus::TokenOverride,
            Focus::Username,
            Focus::Password,
            Focus::AppId,
            Focus::AppVersion,
            Focus::Cid,
            Focus::Secret,
            Focus::Connect,
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
        while self.logs.len() >= 200 {
            self.logs.pop_front();
        }
        self.logs.push_back(message);
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
