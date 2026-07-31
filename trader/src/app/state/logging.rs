use super::super::*;
use crate::broker::ReplayEngineMode;

impl LogEntry {
    pub(in crate::app) fn render_line(&self) -> String {
        self.render_line_with_message(&self.message)
    }

    pub(in crate::app) fn render_persisted_line(&self) -> String {
        let message = sanitize_persisted_log_message(&self.message);
        self.render_line_with_message(&message)
    }

    pub(in crate::app) fn render_line_with_message(&self, message: &str) -> String {
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
    pub(in crate::app) fn push_log(&mut self, message: String) {
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

    pub(in crate::app) fn save_logs_to_file(&mut self) {
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

    pub(in crate::app) fn persist_logs_to_file(&self) -> std::io::Result<std::path::PathBuf> {
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let dir = std::path::PathBuf::from(".run").join("trader-logs");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("session-{timestamp}.txt"));

        let body = self.build_persisted_log_body(&timestamp);
        std::fs::write(&path, body)?;
        Ok(path)
    }

    pub(in crate::app) fn build_persisted_log_body(&self, timestamp: &str) -> String {
        let screen = match self.screen {
            Screen::EngineSelect => "Engine",
            Screen::BrokerSelect => "Broker",
            Screen::Login => "Login",
            Screen::Replay => "Replay",
            Screen::Selection => "Selection",
            Screen::Strategy => "Strategy",
            Screen::Dashboard => "Dashboard",
            Screen::Stats => "Stats",
            Screen::Analytics => "Analytics",
        };
        let selected_account = self
            .accounts
            .get(self.selected_account)
            .map(|account| (account.id, account.name.as_str()));
        let selected_account_name = selected_account.map(|(_, name)| name).unwrap_or("none");
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
            "replay_engine_mode: {}\n",
            self.base_config.replay_engine_mode.label()
        ));
        body.push_str(&format!(
            "replay_fill_model: {}\n",
            match self.base_config.replay_engine_mode {
                ReplayEngineMode::Legacy => "legacy_reference_price",
                ReplayEngineMode::Deterministic =>
                    self.base_config.replay_fill_model.config_label(),
            }
        ));
        body.push_str(&format!(
            "replay_dom_file_path: {}\n",
            self.base_config
                .replay_dom_file_path
                .as_deref()
                .map_or("(not configured)".to_string(), |path| path
                    .display()
                    .to_string())
        ));
        body.push_str(&format!(
            "replay_latency_model: {}\n",
            self.base_config.replay_latency_model.label()
        ));
        body.push_str(&format!(
            "replay_fixed_latency_ms: {}\n",
            self.base_config.replay_fixed_latency_ms
        ));
        body.push_str(&format!(
            "replay_observed_latency_sample_count: {}\n",
            self.base_config.replay_observed_latency_ms.len()
        ));
        body.push_str(&format!(
            "replay_latency_seed: {}\n",
            self.base_config.replay_latency_seed
        ));
        body.push_str(&format!(
            "replay_bar_protection_policy: {}\n",
            self.base_config.replay_bar_protection_policy.label()
        ));
        body.push_str(&format!(
            "replay_ledger_schema_version: {}\n",
            self.replay_execution_ledger.schema_version
        ));
        body.push_str(&format!(
            "replay_ledger_fill_model: {}\n",
            self.replay_execution_ledger.fill_model.config_label()
        ));
        body.push_str(&format!(
            "replay_ledger_fill_count: {}\n",
            self.replay_execution_ledger.fill_count
        ));
        body.push_str(&format!(
            "replay_ledger_gross_realized_pnl: {:.8}\n",
            self.replay_execution_ledger.gross_realized_pnl
        ));
        body.push_str(&format!(
            "replay_signal_source: {}\n",
            self.replay_execution_ledger.signal_source
        ));
        if let Some(window) = self.market.replay_window.as_ref() {
            body.push_str(&format!("replay_window_preset: {}\n", window.preset));
            body.push_str(&format!(
                "replay_window_input_timezone: {}\n",
                window.input_timezone
            ));
            body.push_str(&format!(
                "replay_warmup_start_utc: {}\n",
                window.warmup_start
            ));
            body.push_str(&format!(
                "replay_evaluation_start_utc: {}\n",
                window.evaluation_start
            ));
            body.push_str(&format!(
                "replay_evaluation_end_utc: {}\n",
                window.evaluation_end
            ));
            body.push_str(&format!(
                "replay_evaluation_local_range: {}\n",
                window.local_range_label()
            ));
            body.push_str(&format!("replay_warmup_rows: {}\n", window.warmup_rows));
            body.push_str(&format!(
                "replay_evaluation_rows_total: {}\n",
                window.evaluation_rows_total
            ));
            body.push_str(&format!(
                "replay_evaluation_rows_processed: {}\n",
                window.evaluation_rows_processed
            ));
            body.push_str(&format!(
                "replay_evaluation_rows_remaining: {}\n",
                window.evaluation_rows_remaining()
            ));
        } else {
            body.push_str("replay_window_preset: n/a\n");
            body.push_str("replay_window_input_timezone: n/a\n");
            body.push_str("replay_warmup_start_utc: n/a\n");
            body.push_str("replay_evaluation_start_utc: n/a\n");
            body.push_str("replay_evaluation_end_utc: n/a\n");
            body.push_str("replay_evaluation_local_range: n/a\n");
            body.push_str("replay_warmup_rows: n/a\n");
            body.push_str("replay_evaluation_rows_total: n/a\n");
            body.push_str("replay_evaluation_rows_processed: n/a\n");
            body.push_str("replay_evaluation_rows_remaining: n/a\n");
        }
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
        body.push_str(&format!(
            "strategy_order_qty: {}\n",
            self.strategy.order_qty
        ));
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
        body.push_str(&format!(
            "selected_contract_name: {selected_contract_name}\n"
        ));
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

    pub(in crate::app) fn optional_u64_label(&self, value: Option<u64>) -> String {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    }

    pub(in crate::app) fn pending_target_label(&self) -> String {
        self.strategy_runtime
            .pending_target_qty
            .map(|qty| qty.to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    pub(in crate::app) fn last_strategy_summary_label(&self) -> String {
        if self.strategy_runtime.last_summary.is_empty() {
            "none".to_string()
        } else {
            self.strategy_runtime.last_summary.clone()
        }
    }

    pub(in crate::app) fn persisted_log_counts(&self) -> (usize, usize) {
        self.persisted_logs
            .iter()
            .fold((0, 0), |(errors, debug), entry| {
                (
                    errors + usize::from(entry.message.starts_with("ERROR:")),
                    debug + usize::from(entry.message.starts_with("DEBUG:")),
                )
            })
    }

    pub(in crate::app) fn review_summary_log_section(&self) -> String {
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
        body.push_str(&format!(
            "pending_target: {}\n",
            self.pending_target_label()
        ));
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

    pub(in crate::app) fn recent_session_events_log_section(&self, limit: usize) -> String {
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
}
