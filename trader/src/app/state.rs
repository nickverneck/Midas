#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct DisplayedTradeLevels {
    entry_price: Option<f64>,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    take_profit_projected: bool,
    stop_price_projected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DisplayedAutoTrail {
    trigger_ticks: f64,
    offset_ticks: f64,
    initial_stop_ticks_from_entry: f64,
    first_stop_ticks_from_entry: f64,
    has_fixed_stop: bool,
    initial_stop_price: Option<f64>,
    trigger_price: Option<f64>,
    first_stop_price: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrategyReadinessStatus {
    ReadyToArm,
    MonitorOnly,
    PreviewOnly,
    NeedsAttention,
}

impl StrategyReadinessStatus {
    fn label(self) -> &'static str {
        match self {
            Self::ReadyToArm => "Ready to arm",
            Self::MonitorOnly => "Monitor only",
            Self::PreviewOnly => "Preview only",
            Self::NeedsAttention => "Needs attention",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StrategyReadiness {
    status: StrategyReadinessStatus,
    blockers: Vec<String>,
    warnings: Vec<String>,
    adjustments: Vec<String>,
}

impl StrategyReadiness {
    fn can_arm(&self) -> bool {
        self.status == StrategyReadinessStatus::ReadyToArm
    }
}

impl LogEntry {
    fn render_line(&self) -> String {
        self.render_line_with_message(&self.message)
    }

    fn render_persisted_line(&self) -> String {
        let message = sanitize_persisted_log_message(&self.message);
        self.render_line_with_message(&message)
    }

    fn render_line_with_message(&self, message: &str) -> String {
        format!(
            "[{} | {}] {}",
            self.timestamp.format("%H:%M:%S%.3f"),
            format_log_elapsed(self.elapsed_since_previous),
            message
        )
    }
}

fn sanitize_persisted_log_message(message: &str) -> String {
    if let Some(prefix) = raw_http_failure_prefix(message) {
        return format!("{prefix}[redacted broker response]");
    }
    redact_structured_log_segments(message)
}

fn raw_http_failure_prefix(message: &str) -> Option<&str> {
    let body_start = message.find("): ")?;
    let before_status = &message[..body_start];
    if before_status.contains(" failed (") {
        Some(&message[..body_start + "): ".len()])
    } else {
        None
    }
}

fn redact_structured_log_segments(message: &str) -> String {
    let mut redacted = String::with_capacity(message.len());
    let mut index = 0;
    while index < message.len() {
        let rest = &message[index..];
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch == '{' || (ch == '[' && starts_like_json_array(rest)) {
            if let Some(end) = structured_segment_end(rest, ch) {
                redacted.push_str("[redacted structured data]");
                index += end;
                continue;
            }
        }
        redacted.push(ch);
        index += ch.len_utf8();
    }
    redacted
}

fn starts_like_json_array(segment: &str) -> bool {
    let Some(rest) = segment.strip_prefix('[') else {
        return false;
    };
    matches!(
        rest.chars().find(|ch| !ch.is_whitespace()),
        Some('{' | '[' | '"' | ']' | '-' | '0'..='9' | 't' | 'f' | 'n')
    )
}

fn structured_segment_end(segment: &str, opener: char) -> Option<usize> {
    let closer = match opener {
        '{' => '}',
        '[' => ']',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in segment.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            value if value == opener => depth += 1,
            value if value == closer => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(idx + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
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
    fn broker_supports_replay(&self) -> bool {
        self.selected_broker == BrokerKind::Tradovate && cfg!(feature = "replay")
    }

    fn broker_supports_bar_type_selection(&self) -> bool {
        self.selected_broker == BrokerKind::Tradovate
    }

    fn broker_supports_heikin_ashi(&self) -> bool {
        self.selected_broker == BrokerKind::Tradovate
    }

    fn replay_affordance_visible(&self) -> bool {
        self.broker_supports_replay()
    }

    fn session_stats_affordance_visible(&self) -> bool {
        self.session_stats.enabled
    }

    fn manual_order_affordance_visible(&self) -> bool {
        cfg!(feature = "manual-orders") && self.capabilities.manual_orders
    }

    fn automated_strategy_affordance_visible(&self) -> bool {
        self.capabilities.automated_orders
    }

    fn engine_create_affordance_visible(&self) -> bool {
        self.engine_creation_enabled
    }

    fn engine_close_and_kill_affordance_visible(&self) -> bool {
        cfg!(feature = "manual-orders")
    }

    fn engine_select_item_count(&self) -> usize {
        self.engine_summaries.len() + usize::from(self.engine_create_affordance_visible())
    }

    fn clamp_selected_engine(&mut self) {
        let item_count = self.engine_select_item_count();
        if item_count == 0 {
            self.selected_engine = 0;
        } else if self.selected_engine >= item_count {
            self.selected_engine = item_count - 1;
        }
    }

    fn bar_type_controls_visible(&self) -> bool {
        self.broker_supports_bar_type_selection()
    }

    fn candle_mode_controls_visible(&self) -> bool {
        self.broker_supports_heikin_ashi() && self.bar_type.supports_candle_mode()
    }

    fn visible_strategy_kinds(&self) -> &'static [StrategyKind] {
        &[StrategyKind::Native]
    }

    fn next_visible_strategy_kind(&self) -> StrategyKind {
        let kinds = self.visible_strategy_kinds();
        let index = kinds
            .iter()
            .position(|kind| *kind == self.strategy.kind)
            .unwrap_or(0);
        kinds[(index + 1) % kinds.len()]
    }

    fn prev_visible_strategy_kind(&self) -> StrategyKind {
        let kinds = self.visible_strategy_kinds();
        let index = kinds
            .iter()
            .position(|kind| *kind == self.strategy.kind)
            .unwrap_or(0);
        kinds[(index + kinds.len() - 1) % kinds.len()]
    }

    fn normalize_market_controls_for_broker(&mut self) {
        if !self.broker_supports_bar_type_selection() {
            self.bar_type = BarType::default();
        }
        if !self.broker_supports_heikin_ashi() || !self.bar_type.supports_candle_mode() {
            self.candle_mode = CandleMode::Standard;
        }
    }

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
        let mut order = vec![
            Focus::Env,
            Focus::AuthMode,
            Focus::LogMode,
            Focus::TokenPath,
            Focus::TokenOverride,
            Focus::Username,
            Focus::Password,
            Focus::Connect,
        ];
        if self.replay_affordance_visible() {
            order.push(Focus::ReplayMode);
        }
        match self.selected_broker {
            BrokerKind::Ironbeam => {
                order.insert(7, Focus::ApiKey);
            }
            BrokerKind::Tradovate => {
                order.splice(
                    7..7,
                    [
                        Focus::AppId,
                        Focus::AppVersion,
                        Focus::Cid,
                        Focus::Secret,
                    ],
                );
            }
        }
        order
    }

    fn strategy_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::StrategyKind, Focus::OrderQty];
        if self.strategy.kind == StrategyKind::Native {
            let show_protection_controls = self.native_protection_controls_visible();
            order.push(Focus::NativeStrategy);
            order.push(Focus::NativeSignalTiming);
            order.push(Focus::NativeSignalDelayBars);
            order.push(Focus::NativeExecutionPath);
            order.push(Focus::NativeReversalMode);
            order.push(Focus::NativeBlockoutEnabled);
            order.push(Focus::NativeBlockoutMinutes);
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    order.extend([
                        Focus::HmaLength,
                        Focus::HmaMinAngle,
                        Focus::HmaAngleLookback,
                        Focus::HmaBarsRequired,
                        Focus::HmaLongsOnly,
                        Focus::HmaInverted,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::HmaTakeProfitTicks,
                            Focus::HmaStopLossTicks,
                            Focus::HmaTrailingStop,
                        ]);
                        if self.strategy.native_hma.use_trailing_stop {
                            order.extend([
                                Focus::HmaTrailTriggerTicks,
                                Focus::HmaTrailOffsetTicks,
                            ]);
                        }
                    }
                }
                NativeStrategyKind::EmaCross | NativeStrategyKind::HmaCross => {
                    order.extend([
                        Focus::EmaFastLength,
                        Focus::EmaSlowLength,
                        Focus::EmaInverted,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::EmaTakeProfitTicks,
                            Focus::EmaStopLossTicks,
                            Focus::EmaTrailingStop,
                        ]);
                        let use_trailing_stop = match self.strategy.native_strategy {
                            NativeStrategyKind::HmaCross => {
                                self.strategy.native_hma_cross.use_trailing_stop
                            }
                            _ => self.strategy.native_ema.use_trailing_stop,
                        };
                        if use_trailing_stop {
                            order.extend([
                                Focus::EmaTrailTriggerTicks,
                                Focus::EmaTrailOffsetTicks,
                            ]);
                        }
                    }
                }
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
        let mut order = vec![Focus::AccountList];
        if self.bar_type_controls_visible() {
            order.push(Focus::BarTypeToggle);
            order.push(Focus::BarValue);
        }
        if self.candle_mode_controls_visible() {
            order.push(Focus::CandleModeToggle);
        }
        order.extend([
            Focus::InstrumentQuery,
            Focus::ContractList,
        ]);
        order
    }

    fn replay_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::BarTypeToggle, Focus::BarValue];
        if self.candle_mode_controls_visible() {
            order.push(Focus::CandleModeToggle);
        }
        order.push(Focus::ReplayMode);
        order
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

    fn next_replay_focus(&self) -> Focus {
        let order = self.replay_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    fn prev_replay_focus(&self) -> Focus {
        let order = self.replay_focus_order();
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
                | Focus::ApiKey
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
                | Focus::NativeBlockoutMinutes
                | Focus::NativeSignalDelayBars
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::BarValue
                | Focus::InstrumentQuery
        )
    }

    fn is_free_text_focus(&self) -> bool {
        matches!(
            self.focus,
            Focus::TokenOverride
                | Focus::Username
                | Focus::Password
                | Focus::ApiKey
                | Focus::AppId
                | Focus::AppVersion
                | Focus::Cid
                | Focus::Secret
                | Focus::TokenPath
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::InstrumentQuery
        )
    }

    fn push_log(&mut self, message: String) {
        let now = std::time::Instant::now();
        let entry = LogEntry {
            timestamp: chrono::Local::now(),
            elapsed_since_previous: self.last_log_at.map(|last| now.duration_since(last)),
            message,
        };
        while self.logs.len() >= UI_LOG_ENTRY_LIMIT {
            self.logs.pop_front();
        }
        while self.persisted_logs.len() >= PERSISTED_LOG_ENTRY_LIMIT {
            self.persisted_logs.pop_front();
        }
        self.logs.push_back(entry.clone());
        self.persisted_logs.push_back(entry);
        self.last_log_at = Some(now);
    }

    fn save_logs_to_file(&mut self) {
        match self.persist_logs_to_file() {
            Ok(path) => {
                let message = format!("Saved logs to {}", path.display());
                self.last_saved_log_path = Some(path.clone());
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
        let dir = std::path::PathBuf::from(".run").join("trader-logs");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("session-{timestamp}.txt"));

        let body = self.build_persisted_log_body(&timestamp);
        std::fs::write(&path, body)?;
        Ok(path)
    }

    fn build_persisted_log_body(&self, timestamp: &str) -> String {
        let screen = match self.screen {
            Screen::EngineSelect => "Engine",
            Screen::BrokerSelect => "Broker",
            Screen::Login => "Login",
            Screen::Replay => "Replay",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
            Screen::Stats => "Stats",
        };
        let selected_account = self
            .accounts
            .get(self.selected_account)
            .map(|account| (account.id, account.name.as_str()));
        let selected_account_name = selected_account
            .map(|(_, name)| name)
            .unwrap_or("none");
        let selected_account_id = selected_account
            .map(|(id, _)| id.to_string())
            .unwrap_or_else(|| "none".to_string());
        let selected_contract = self.contract_results.get(self.selected_contract);
        let selected_contract_name = selected_contract
            .map(|contract| contract.name.as_str())
            .or(self.market.contract_name.as_deref())
            .unwrap_or("none");
        let selected_contract_id = selected_contract
            .map(|contract| contract.id.to_string())
            .or_else(|| self.market.contract_id.map(|id| id.to_string()))
            .unwrap_or_else(|| "none".to_string());

        let mut body = String::new();
        body.push_str(&format!("saved_at_utc: {timestamp}\n"));
        body.push_str(&format!("screen: {screen}\n"));
        body.push_str(&format!(
            "status: {}\n",
            sanitize_persisted_log_message(&self.status)
        ));
        body.push_str(&format!(
            "engine_socket: {}\n",
            self.active_engine_socket_label()
        ));
        body.push_str(&format!("engine_id: {}\n", self.active_engine_id_label()));
        body.push_str(&format!(
            "engine_connection_state: {}\n",
            self.active_engine_connection_state_label()
        ));
        body.push_str(&format!(
            "active_engine_key_display: {}\n",
            self.active_engine_key_display_label()
        ));
        body.push_str(&format!(
            "active_engine_key: {}\n",
            self.active_engine_key_display_label()
        ));
        body.push_str(&format!(
            "other_live_engine_count: {}\n",
            self.other_live_engine_count()
        ));
        body.push_str(&format!(
            "other_live_engines: {}\n",
            self.other_live_engine_count()
        ));
        body.push_str(&format!("broker: {}\n", self.selected_broker.label()));
        body.push_str(&format!("env: {}\n", self.form.env.label()));
        body.push_str(&format!("auth_mode: {}\n", self.form.auth_mode.label()));
        body.push_str(&format!("log_mode: {}\n", self.form.log_mode.label()));
        body.push_str(&format!("session_kind: {}\n", self.session_kind.label()));
        body.push_str(&format!("replay_speed: {}\n", self.replay_speed.label()));
        body.push_str(&format!(
            "capability_replay: {}\n",
            self.capabilities.replay
        ));
        body.push_str(&format!(
            "capability_manual_orders: {}\n",
            self.capabilities.manual_orders
        ));
        body.push_str(&format!(
            "capability_automated_orders: {}\n",
            self.capabilities.automated_orders
        ));
        body.push_str(&format!(
            "capability_native_protection: {}\n",
            self.capabilities.native_protection
        ));
        body.push_str(&format!("strategy: {}\n", self.strategy.summary_label()));
        body.push_str(&format!("strategy_order_qty: {}\n", self.strategy.order_qty));
        body.push_str(&format!(
            "order_time_in_force: {}\n",
            self.base_config.time_in_force
        ));
        body.push_str(&format!(
            "strategy_armed: {}\n",
            self.strategy_runtime.armed
        ));
        body.push_str(&format!(
            "pending_target: {}\n",
            self.pending_target_label()
        ));
        body.push_str(&format!(
            "last_strategy_summary: {}\n",
            sanitize_persisted_log_message(&self.last_strategy_summary_label())
        ));
        body.push_str(&format!("latency_summary: {}\n", self.latency_summary()));
        body.push_str(&format!(
            "latency_rest_rtt_ms: {}\n",
            self.optional_u64_label(self.latency.rest_rtt_ms)
        ));
        body.push_str(&format!(
            "latency_last_order_seen_ms: {}\n",
            self.optional_u64_label(self.latency.last_order_seen_ms)
        ));
        body.push_str(&format!(
            "latency_last_order_ack_ms: {}\n",
            self.optional_u64_label(self.latency.last_order_ack_ms)
        ));
        body.push_str(&format!(
            "latency_last_exec_report_ms: {}\n",
            self.optional_u64_label(self.latency.last_exec_report_ms)
        ));
        body.push_str(&format!(
            "latency_last_fill_ms: {}\n",
            self.optional_u64_label(self.latency.last_fill_ms)
        ));
        body.push_str(&format!(
            "latency_last_signal_submit_ms: {}\n",
            self.optional_u64_label(self.latency.last_signal_submit_ms)
        ));
        body.push_str(&format!(
            "latency_last_signal_seen_ms: {}\n",
            self.optional_u64_label(self.latency.last_signal_seen_ms)
        ));
        body.push_str(&format!(
            "latency_last_signal_ack_ms: {}\n",
            self.optional_u64_label(self.latency.last_signal_ack_ms)
        ));
        body.push_str(&format!(
            "latency_last_signal_fill_ms: {}\n",
            self.optional_u64_label(self.latency.last_signal_fill_ms)
        ));
        body.push_str(&format!(
            "market_update_age: {}\n",
            format_age_ms(self.market_update_age_ms())
        ));
        body.push_str(&format!(
            "market_update_age_ms: {}\n",
            self.optional_u64_label(self.market_update_age_ms())
        ));
        body.push_str(&format!("selected_account: {selected_account_name}\n"));
        body.push_str(&format!("selected_account_id: {selected_account_id}\n"));
        body.push_str(&format!("selected_account_name: {selected_account_name}\n"));
        body.push_str(&format!("selected_contract: {selected_contract_name}\n"));
        body.push_str(&format!("selected_contract_id: {selected_contract_id}\n"));
        body.push_str(&format!("selected_contract_name: {selected_contract_name}\n"));
        body.push_str(&format!("bar_type: {}\n", self.bar_type.label()));
        if self.bar_type.supports_candle_mode() {
            body.push_str(&format!("candle_mode: {}\n", self.candle_mode.label()));
        }
        body.push_str("log_format: [HH:MM:SS.mmm local | +elapsed_since_previous] message\n");
        body.push('\n');
        body.push_str(&self.review_summary_log_section());
        body.push('\n');
        body.push_str(&self.recent_session_events_log_section(12));
        body.push('\n');
        body.push_str(&self.session_stats_log_section());
        body.push('\n');
        body.push_str("[logs]\n");

        for entry in &self.persisted_logs {
            body.push_str(&entry.render_persisted_line());
            body.push('\n');
        }

        body
    }

    fn active_engine_summary(&self) -> Option<&EngineSummary> {
        let active_key = self.active_engine_key.as_ref()?;
        self.engine_summaries
            .iter()
            .find(|summary| &summary.key == active_key)
    }

    fn active_engine_socket_label(&self) -> String {
        self.engine_socket_path
            .as_ref()
            .or_else(|| self.active_engine_summary().map(|summary| &summary.socket_path))
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    fn active_engine_socket_short_label(&self) -> String {
        self.active_engine_summary()
            .map(EngineSummary::socket_short_label)
            .or_else(|| {
                self.engine_socket_path.as_ref().map(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(ToString::to_string)
                        .unwrap_or_else(|| path.display().to_string())
                })
            })
            .unwrap_or_else(|| "none".to_string())
    }

    fn active_engine_id_label(&self) -> String {
        self.active_engine_summary()
            .and_then(|summary| summary.id)
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    fn active_engine_connection_state_label(&self) -> String {
        self.active_engine_summary()
            .map(|summary| summary.connection_state.label().to_string())
            .unwrap_or_else(|| {
                if self.engine_socket_path.is_some() {
                    "unknown".to_string()
                } else {
                    "none".to_string()
                }
            })
    }

    pub(in crate::app) fn active_engine_header_label(&self) -> String {
        if self.engine_socket_path.is_none() && self.active_engine_key.is_none() {
            return "none".to_string();
        }

        let id = self.active_engine_id_label();
        let socket = self.active_engine_socket_short_label();
        let identity = if id == "none" {
            socket
        } else {
            format!("#{id} {socket}")
        };
        let mut label = format!(
            "{} {} {}",
            identity,
            self.active_engine_connection_state_label(),
            self.session_kind.label()
        );
        let other_count = self.other_live_engine_count();
        if other_count > 0 {
            label.push_str(&format!(" +{other_count} other"));
        }
        label
    }

    fn engine_receiver_closed_message(&self, engine_key: &EngineKey) -> String {
        let Some(summary) = self
            .engine_summaries
            .iter()
            .find(|summary| &summary.key == engine_key)
        else {
            return format!(
                "Engine {} connection closed.",
                engine_key.display_label()
            );
        };

        format!(
            "Engine {} connection closed; last state was {}.",
            summary.identity_label(),
            summary.connection_state.label()
        )
    }

    fn active_engine_key_display_label(&self) -> String {
        self.active_engine_key
            .as_ref()
            .map(EngineKey::display_label)
            .unwrap_or_else(|| "none".to_string())
    }

    fn active_engine_stats_identity_key(&self) -> String {
        self.active_engine_key
            .as_ref()
            .map(|key| format!("engine:{}", key.display_label()))
            .unwrap_or_else(|| "embedded".to_string())
    }

    fn other_live_engine_count(&self) -> usize {
        self.engine_summaries
            .iter()
            .filter(|summary| {
                self.active_engine_key
                    .as_ref()
                    .is_none_or(|active_key| &summary.key != active_key)
                    && matches!(
                        summary.connection_state,
                        EngineConnectionState::Observing | EngineConnectionState::Connected
                    )
            })
            .count()
    }

    fn optional_u64_label(&self, value: Option<u64>) -> String {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    }

    fn pending_target_label(&self) -> String {
        self.strategy_runtime
            .pending_target_qty
            .map(|qty| qty.to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    fn last_strategy_summary_label(&self) -> String {
        if self.strategy_runtime.last_summary.is_empty() {
            "none".to_string()
        } else {
            self.strategy_runtime.last_summary.clone()
        }
    }

    fn persisted_log_counts(&self) -> (usize, usize) {
        self.persisted_logs
            .iter()
            .fold((0, 0), |(errors, debug), entry| {
                (
                    errors + usize::from(entry.message.starts_with("ERROR:")),
                    debug + usize::from(entry.message.starts_with("DEBUG:")),
                )
            })
    }

    fn review_summary_log_section(&self) -> String {
        let (error_count, debug_count) = self.persisted_log_counts();
        let mut body = String::new();
        body.push_str("[review_summary]\n");
        body.push_str(&format!(
            "final_status: {}\n",
            sanitize_persisted_log_message(&self.status)
        ));
        body.push_str(&format!(
            "strategy_runtime_summary: {}\n",
            sanitize_persisted_log_message(&self.strategy_runtime_summary())
        ));
        body.push_str(&format!("pending_target: {}\n", self.pending_target_label()));
        body.push_str(&format!("latency_summary: {}\n", self.latency_summary()));
        if let Some(stats) = self.selected_session_stats() {
            body.push_str(&format!(
                "selected_account_session_pnl: {}\n",
                format_signed_money(Some(stats.session_pnl()))
            ));
            body.push_str(&format!(
                "selected_account_trade_pnl_ex_fees: {}\n",
                format_signed_money(Some(stats.trade_pnl_ex_fees()))
            ));
            body.push_str(&format!(
                "selected_account_total_fees: {}\n",
                format_signed_money(Some(stats.total_fees))
            ));
        } else {
            body.push_str("selected_account_session_pnl: n/a\n");
            body.push_str("selected_account_trade_pnl_ex_fees: n/a\n");
            body.push_str("selected_account_total_fees: n/a\n");
        }
        if let Some(snapshot) = self.selected_snapshot() {
            body.push_str(&format!(
                "selected_account_realized_pnl: {}\n",
                format_signed_money(snapshot.realized_pnl)
            ));
            body.push_str(&format!(
                "selected_account_unrealized_pnl: {}\n",
                format_signed_money(snapshot.unrealized_pnl)
            ));
        } else {
            body.push_str("selected_account_realized_pnl: n/a\n");
            body.push_str("selected_account_unrealized_pnl: n/a\n");
        }
        body.push_str(&format!("persisted_error_count: {error_count}\n"));
        body.push_str(&format!("persisted_debug_count: {debug_count}\n"));
        body
    }

    fn recent_session_events_log_section(&self, limit: usize) -> String {
        let mut body = String::new();
        body.push_str("[recent_session_events]\n");
        body.push_str(&format!(
            "fees_visible: {}\n",
            if self.session_stats_show_fees {
                "shown"
            } else {
                "hidden"
            }
        ));
        for line in self.session_stats_event_lines(limit) {
            body.push_str(&line.to_string());
            body.push('\n');
        }
        body
    }

    fn clear_strategy_numeric_input(&mut self) {
        self.strategy_numeric_input = None;
    }

    fn bar_value_text(&self) -> String {
        self.strategy_numeric_value(Focus::BarValue, self.bar_type.value().to_string())
    }

    fn effective_candle_mode(&self) -> CandleMode {
        self.bar_type.effective_candle_mode(self.candle_mode)
    }

    fn market_data_title_prefix(&self) -> String {
        self.bar_type.mode_label(self.effective_candle_mode())
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

    fn strategy_readiness(&self) -> StrategyReadiness {
        let mut blockers = Vec::new();
        let mut warnings = Vec::new();
        let mut adjustments = Vec::new();

        if self.strategy.kind != StrategyKind::Native {
            return StrategyReadiness {
                status: StrategyReadinessStatus::PreviewOnly,
                blockers: vec![format!(
                    "{} is not executable from the TUI yet.",
                    self.strategy.kind.label()
                )],
                warnings,
                adjustments,
            };
        }

        if self.strategy.order_qty <= 0 {
            blockers.push("Order Qty must be at least 1.".to_string());
        }

        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                if self.strategy.native_hma.hma_length < 2 {
                    blockers.push("HMA Length must be at least 2.".to_string());
                }
                if self.strategy.native_hma.angle_lookback == 0 {
                    blockers.push("Angle Lookback must be at least 1.".to_string());
                }
                if self.strategy.native_hma.bars_required_to_trade == 0 {
                    blockers.push("Bars Required must be at least 1.".to_string());
                }
            }
            NativeStrategyKind::EmaCross => {
                let fast = self.strategy.native_ema.fast_length;
                let slow = self.strategy.native_ema.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast EMA Length ({fast}) must be less than Slow EMA Length ({slow})."
                    ));
                }
            }
            NativeStrategyKind::HmaCross => {
                let fast = self.strategy.native_hma_cross.fast_length;
                let slow = self.strategy.native_hma_cross.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast HMA Length ({fast}) must be less than Slow HMA Length ({slow})."
                    ));
                }
            }
        }

        if self.strategy.native_signal_timing == NativeSignalTiming::ClosedBar
            && self.market.bars.is_empty()
        {
            warnings.push("Closed-bar timing will wait for completed bars.".to_string());
        }
        if self.native_protection_controls_visible() && !self.active_native_uses_broker_owned_protection() {
            warnings.push("No TP/SL/trailing protection is configured.".to_string());
        }

        if self.active_native_uses_broker_owned_protection()
            && !self.capabilities.native_protection
        {
            blockers.push("Native protection is unavailable for this engine.".to_string());
        } else {
            if self.active_native_uses_broker_owned_protection()
                && self.strategy.native_reversal_mode == NativeReversalMode::Direct
            {
                adjustments.push(
                    "Direct reversal will switch to CloseAll > Enter for broker-owned protection."
                        .to_string(),
                );
            }
            if self.active_native_requires_guarded_path()
                && self.strategy.native_execution_path != NativeExecutionPath::Guarded
            {
                adjustments.push(
                    "Execution Path will switch to Guarded for this reversal/protection setup."
                        .to_string(),
                );
            }
        }

        if !blockers.is_empty() {
            return StrategyReadiness {
                status: StrategyReadinessStatus::NeedsAttention,
                blockers,
                warnings,
                adjustments,
            };
        }

        if !self.automated_strategy_affordance_visible() {
            blockers.push(format!(
                "{} automation is unavailable.",
                self.selected_broker.label()
            ));
        }
        if self.accounts.get(self.selected_account).is_none() {
            blockers.push("Select an account before arming.".to_string());
        }
        if !self.selected_contract_ready() {
            blockers.push("Select a contract before arming.".to_string());
        }

        StrategyReadiness {
            status: if blockers.is_empty() {
                StrategyReadinessStatus::ReadyToArm
            } else {
                StrategyReadinessStatus::MonitorOnly
            },
            blockers,
            warnings,
            adjustments,
        }
    }

    fn selected_contract_ready(&self) -> bool {
        self.contract_results.get(self.selected_contract).is_some()
            || self.market.contract_id.is_some()
            || self.market.contract_name.is_some()
    }

    fn strategy_continue_label(&self) -> String {
        match self.strategy_readiness().status {
            StrategyReadinessStatus::ReadyToArm => {
                "[Enter] Continue / Arm Native Strategy".to_string()
            }
            StrategyReadinessStatus::MonitorOnly => "[Enter] Continue / Monitor Only".to_string(),
            StrategyReadinessStatus::PreviewOnly => "[Enter] Continue / Preview Only".to_string(),
            StrategyReadinessStatus::NeedsAttention => "[Enter] Review Strategy Setup".to_string(),
        }
    }

    fn sync_execution_strategy_config(&self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if !self.capabilities.automated_orders {
            return;
        }
        let _ = cmd_tx.send(ServiceCommand::SetExecutionStrategyConfig(
            self.strategy.execution_config(),
        ));
    }

    fn native_protection_controls_visible(&self) -> bool {
        if self.strategy.kind != StrategyKind::Native {
            return false;
        }
        if !self.capabilities.native_protection {
            return false;
        }
        if self.selected_broker != BrokerKind::Tradovate {
            return true;
        }
        self.strategy.native_execution_path == NativeExecutionPath::Guarded
            && self.strategy.native_reversal_mode != NativeReversalMode::Direct
    }

    fn native_summary_for_display(&self) -> String {
        if self.native_protection_controls_visible() {
            self.strategy.native_summary()
        } else {
            self.strategy.native_summary_without_protection()
        }
    }

    fn active_native_uses_broker_owned_protection(&self) -> bool {
        if self.strategy.kind != StrategyKind::Native {
            return false;
        }
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self.strategy.native_hma.uses_native_protection(),
            NativeStrategyKind::EmaCross => self.strategy.native_ema.uses_native_protection(),
            NativeStrategyKind::HmaCross => {
                self.strategy.native_hma_cross.uses_native_protection()
            }
        }
    }

    fn active_native_requires_guarded_path(&self) -> bool {
        self.strategy.kind == StrategyKind::Native
            && (self.strategy.native_reversal_mode != NativeReversalMode::Direct
                || self.active_native_uses_broker_owned_protection())
    }

    fn normalize_native_reversal_mode_before_arm(&mut self) -> Option<&'static str> {
        if !self.active_native_uses_broker_owned_protection()
            || self.strategy.native_reversal_mode != NativeReversalMode::Direct
        {
            return None;
        }
        let previous = self.strategy.native_reversal_mode.label();
        self.strategy.native_reversal_mode = NativeReversalMode::CloseAllEnter;
        Some(previous)
    }

    fn normalize_native_execution_path_before_arm(&mut self) -> Option<&'static str> {
        if !self.active_native_requires_guarded_path()
            || self.strategy.native_execution_path == NativeExecutionPath::Guarded
        {
            return None;
        }
        let previous = self.strategy.native_execution_path.label();
        self.strategy.native_execution_path = NativeExecutionPath::Guarded;
        Some(previous)
    }

    fn manual_disarm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if !self.capabilities.automated_orders {
            return;
        }
        let _ = cmd_tx.send(ServiceCommand::DisarmExecutionStrategy {
            reason: "Manual strategy disarm requested.".to_string(),
        });
        self.push_log("Manual strategy disarm requested.".to_string());
    }

    fn arm_native_strategy(&mut self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if !self.capabilities.automated_orders {
            return;
        }
        if let Some(previous_mode) = self.normalize_native_reversal_mode_before_arm() {
            self.push_log(format!(
                "Reversal Mode switched to CloseAll > Enter before arming; {previous_mode} cannot attach broker-owned TP/SL/trailing protection."
            ));
        }
        if let Some(previous_path) = self.normalize_native_execution_path_before_arm() {
            self.push_log(format!(
                "Execution Path switched to Guarded before arming; {previous_path} ignores broker protection or non-direct reversal settings."
            ));
        }
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::ArmExecutionStrategy);
    }

    fn closed_bars(&self) -> &[crate::broker::Bar] {
        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        &self.market.bars[..closed_len]
    }

    fn latest_closed_bar_ts(&self) -> Option<i64> {
        self.closed_bars().last().map(|bar| bar.ts_ns)
    }

    fn session_window_at(&self, ts_ns: i64) -> Option<InstrumentSessionWindow> {
        self.market
            .session_profile
            .map(|profile| {
                profile.evaluate_with_blockout(
                    ts_ns,
                    self.strategy.blockout_minutes_before_close,
                )
            })
    }

    fn latest_session_window(&self) -> Option<InstrumentSessionWindow> {
        self.latest_closed_bar_ts()
            .and_then(|ts_ns| self.session_window_at(ts_ns))
    }

    fn session_gate_summary(&self) -> String {
        if !self.strategy.blockout_enabled {
            return "blockout off".to_string();
        }

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
        if self.strategy.blockout_enabled && window.hold_entries {
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
        if !self.capabilities.automated_orders {
            return format!("{} monitor-only", self.selected_broker.label());
        }
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

    fn displayed_trade_levels(&self) -> DisplayedTradeLevels {
        let entry_price = self
            .selected_snapshot()
            .and_then(|snapshot| snapshot.market_entry_price)
            .filter(|price| price.is_finite());
        let signed_qty = self
            .selected_snapshot()
            .and_then(|snapshot| snapshot.market_position_qty)
            .map(|qty| qty.round() as i32)
            .filter(|qty| *qty != 0);
        let mut levels = DisplayedTradeLevels {
            entry_price,
            take_profit_price: self
                .selected_snapshot()
                .and_then(|snapshot| snapshot.selected_contract_take_profit_price)
                .filter(|price| price.is_finite()),
            stop_price: self
                .selected_snapshot()
                .and_then(|snapshot| snapshot.selected_contract_stop_price)
                .filter(|price| price.is_finite()),
            ..DisplayedTradeLevels::default()
        };

        let (Some(entry_price), Some(signed_qty)) = (entry_price, signed_qty) else {
            return levels;
        };

        if levels.take_profit_price.is_none() {
            levels.take_profit_price = self.projected_native_take_profit_price(entry_price, signed_qty);
            levels.take_profit_projected = levels.take_profit_price.is_some();
        }
        if levels.stop_price.is_none() {
            levels.stop_price = self.projected_native_stop_price(entry_price, signed_qty);
            levels.stop_price_projected = levels.stop_price.is_some();
        }

        levels
    }

    fn displayed_auto_trail(&self) -> Option<DisplayedAutoTrail> {
        if self.strategy.kind != StrategyKind::Native {
            return None;
        }
        if !self.native_protection_controls_visible() {
            return None;
        }

        let (use_trailing_stop, trigger_ticks, offset_ticks, stop_loss_ticks) =
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => (
                    self.strategy.native_hma.use_trailing_stop,
                    self.strategy.native_hma.trail_trigger_ticks,
                    self.strategy.native_hma.trail_offset_ticks,
                    self.strategy.native_hma.stop_loss_ticks,
                ),
                NativeStrategyKind::EmaCross => (
                    self.strategy.native_ema.use_trailing_stop,
                    self.strategy.native_ema.trail_trigger_ticks,
                    self.strategy.native_ema.trail_offset_ticks,
                    self.strategy.native_ema.stop_loss_ticks,
                ),
                NativeStrategyKind::HmaCross => (
                    self.strategy.native_hma_cross.use_trailing_stop,
                    self.strategy.native_hma_cross.trail_trigger_ticks,
                    self.strategy.native_hma_cross.trail_offset_ticks,
                    self.strategy.native_hma_cross.stop_loss_ticks,
                ),
            };
        if !use_trailing_stop || trigger_ticks <= 0.0 || offset_ticks <= 0.0 {
            return None;
        }

        let has_fixed_stop = stop_loss_ticks > 0.0;
        let initial_stop_distance_ticks = if has_fixed_stop {
            stop_loss_ticks
        } else {
            trigger_ticks + offset_ticks
        };
        let initial_stop_ticks_from_entry = -initial_stop_distance_ticks;
        let first_stop_ticks_from_entry = trigger_ticks - offset_ticks;
        let (initial_stop_price, trigger_price, first_stop_price) = self
            .market
            .tick_size
            .filter(|tick| tick.is_finite() && *tick > 0.0)
            .and_then(|tick_size| {
                let entry_price = self
                    .selected_snapshot()
                    .and_then(|snapshot| snapshot.market_entry_price)
                    .filter(|price| price.is_finite())?;
                let signed_qty = self
                    .selected_snapshot()
                    .and_then(|snapshot| snapshot.market_position_qty)
                    .map(|qty| qty.round() as i32)
                    .filter(|qty| *qty != 0)?;
                let direction = if signed_qty > 0 { 1.0 } else { -1.0 };
                Some((
                    entry_price + direction * initial_stop_ticks_from_entry * tick_size,
                    entry_price + direction * trigger_ticks * tick_size,
                    entry_price + direction * first_stop_ticks_from_entry * tick_size,
                ))
            })
            .map_or(
                (None, None, None),
                |(initial_stop_price, trigger_price, first_stop_price)| {
                    (
                        Some(initial_stop_price),
                        Some(trigger_price),
                        Some(first_stop_price),
                    )
                },
            );

        Some(DisplayedAutoTrail {
            trigger_ticks,
            offset_ticks,
            initial_stop_ticks_from_entry,
            first_stop_ticks_from_entry,
            has_fixed_stop,
            initial_stop_price,
            trigger_price,
            first_stop_price,
        })
    }

    fn projected_native_take_profit_price(&self, entry_price: f64, signed_qty: i32) -> Option<f64> {
        if self.strategy.kind != StrategyKind::Native || !entry_price.is_finite() || signed_qty == 0 {
            return None;
        }

        let offset = match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                self.strategy.native_hma.take_profit_offset(self.market.tick_size)?
            }
            NativeStrategyKind::EmaCross => {
                self.strategy.native_ema.take_profit_offset(self.market.tick_size)?
            }
            NativeStrategyKind::HmaCross => self
                .strategy
                .native_hma_cross
                .take_profit_offset(self.market.tick_size)?,
        };

        Some(if signed_qty > 0 {
            entry_price + offset
        } else {
            entry_price - offset
        })
    }

    fn projected_native_stop_price(&self, entry_price: f64, signed_qty: i32) -> Option<f64> {
        if self.strategy.kind != StrategyKind::Native || !entry_price.is_finite() || signed_qty == 0 {
            return None;
        }

        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let mut runtime = crate::strategies::hma_angle::HmaAngleExecutionState::default();
                self.strategy
                    .native_hma
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_hma
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::EmaCross => {
                let mut runtime = crate::strategies::ema_cross::EmaCrossExecutionState::default();
                self.strategy
                    .native_ema
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_ema
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::HmaCross => {
                let mut runtime =
                    crate::strategies::hma_cross::HmaCrossExecutionState::default();
                self.strategy
                    .native_hma_cross
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_hma_cross
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
        }
        .filter(|price| price.is_finite())
    }

}
