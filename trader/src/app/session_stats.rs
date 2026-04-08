const SESSION_STATS_DELTA_EPSILON: f64 = 0.005;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionStatSource {
    Balance,
    CashBalance,
    NetLiq,
}

impl SessionStatSource {
    fn label(self) -> &'static str {
        match self {
            Self::Balance => "balance",
            Self::CashBalance => "cash_balance",
            Self::NetLiq => "net_liq",
        }
    }
}

#[derive(Debug, Clone)]
struct SessionBalanceEvent {
    recorded_at_utc: chrono::DateTime<chrono::Utc>,
    source: SessionStatSource,
    previous_value: f64,
    current_value: f64,
    delta: f64,
}

#[derive(Debug, Clone)]
struct AccountSessionStats {
    account_id: i64,
    account_name: String,
    broker: BrokerKind,
    env: TradingEnvironment,
    session_kind: SessionKind,
    source: SessionStatSource,
    started_at_utc: chrono::DateTime<chrono::Utc>,
    last_updated_at_utc: chrono::DateTime<chrono::Utc>,
    sample_count: usize,
    start_value: f64,
    current_value: f64,
    last_delta: Option<f64>,
    wins: usize,
    losses: usize,
    flat_moves: usize,
    gross_wins: f64,
    gross_losses: f64,
    max_win: Option<f64>,
    max_loss: Option<f64>,
    events: Vec<SessionBalanceEvent>,
}

impl AccountSessionStats {
    fn new(
        snapshot: &AccountSnapshot,
        broker: BrokerKind,
        env: TradingEnvironment,
        session_kind: SessionKind,
        source: SessionStatSource,
        value: f64,
        captured_at_utc: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            account_id: snapshot.account_id,
            account_name: snapshot.account_name.clone(),
            broker,
            env,
            session_kind,
            source,
            started_at_utc: captured_at_utc,
            last_updated_at_utc: captured_at_utc,
            sample_count: 1,
            start_value: value,
            current_value: value,
            last_delta: None,
            wins: 0,
            losses: 0,
            flat_moves: 0,
            gross_wins: 0.0,
            gross_losses: 0.0,
            max_win: None,
            max_loss: None,
            events: Vec::new(),
        }
    }

    fn apply_snapshot(
        &mut self,
        snapshot: &AccountSnapshot,
        broker: BrokerKind,
        env: TradingEnvironment,
        session_kind: SessionKind,
        source: SessionStatSource,
        value: f64,
        captured_at_utc: chrono::DateTime<chrono::Utc>,
    ) {
        self.account_name = snapshot.account_name.clone();
        self.broker = broker;
        self.env = env;
        self.session_kind = session_kind;
        self.last_updated_at_utc = captured_at_utc;
        self.sample_count += 1;

        if self.source != source {
            self.source = source;
            self.current_value = value;
            self.last_delta = None;
            return;
        }

        let delta = value - self.current_value;
        if delta.abs() < SESSION_STATS_DELTA_EPSILON {
            self.current_value = value;
            self.last_delta = Some(0.0);
            self.flat_moves += 1;
            return;
        }

        let previous_value = self.current_value;
        self.current_value = value;
        self.last_delta = Some(delta);
        self.events.push(SessionBalanceEvent {
            recorded_at_utc: captured_at_utc,
            source,
            previous_value,
            current_value: value,
            delta,
        });

        if delta > 0.0 {
            self.wins += 1;
            self.gross_wins += delta;
            self.max_win = Some(
                self.max_win
                    .map(|value| value.max(delta))
                    .unwrap_or(delta),
            );
        } else {
            let loss = delta.abs();
            self.losses += 1;
            self.gross_losses += loss;
            self.max_loss = Some(
                self.max_loss
                    .map(|value| value.max(loss))
                    .unwrap_or(loss),
            );
        }
    }

    fn session_pnl(&self) -> f64 {
        self.current_value - self.start_value
    }

    fn move_count(&self) -> usize {
        self.wins + self.losses + self.flat_moves
    }

    fn avg_win(&self) -> Option<f64> {
        (self.wins > 0).then_some(self.gross_wins / self.wins as f64)
    }

    fn avg_loss(&self) -> Option<f64> {
        (self.losses > 0).then_some(self.gross_losses / self.losses as f64)
    }

    fn avg_loss_signed(&self) -> Option<f64> {
        self.avg_loss().map(|value| -value)
    }

    fn max_loss_signed(&self) -> Option<f64> {
        self.max_loss.map(|value| -value)
    }

    fn win_rate(&self) -> Option<f64> {
        let total = self.wins + self.losses;
        (total > 0).then_some(self.wins as f64 / total as f64)
    }

    fn profit_factor(&self) -> Option<f64> {
        if self.gross_losses > 0.0 {
            Some(self.gross_wins / self.gross_losses)
        } else if self.gross_wins > 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        }
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }
}

#[derive(Debug, Clone)]
struct SessionStatsState {
    enabled: bool,
    started_at_utc: chrono::DateTime<chrono::Utc>,
    accounts: std::collections::BTreeMap<String, AccountSessionStats>,
}

impl SessionStatsState {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            started_at_utc: chrono::Utc::now(),
            accounts: std::collections::BTreeMap::new(),
        }
    }

    fn tracked_accounts(&self) -> usize {
        self.accounts.len()
    }

    fn total_event_count(&self) -> usize {
        self.accounts
            .values()
            .map(AccountSessionStats::event_count)
            .sum()
    }
}

fn tracked_balance_value(snapshot: &AccountSnapshot) -> Option<(SessionStatSource, f64)> {
    let (source, value) = snapshot
        .balance
        .map(|value| (SessionStatSource::Balance, value))
        .or_else(|| {
            snapshot
                .cash_balance
                .map(|value| (SessionStatSource::CashBalance, value))
        })
        .or_else(|| snapshot.net_liq.map(|value| (SessionStatSource::NetLiq, value)))?;
    value.is_finite().then_some((source, value))
}

fn format_percent(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{:.1}%", value * 100.0),
        None => "n/a".to_string(),
    }
}

fn format_ratio(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_infinite() => "inf".to_string(),
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_session_stats_timestamp(
    value: chrono::DateTime<chrono::Utc>,
    include_date: bool,
) -> String {
    let local = value.with_timezone(&chrono::Local);
    if include_date {
        local.format("%Y-%m-%d %H:%M:%S %Z").to_string()
    } else {
        local.format("%H:%M:%S").to_string()
    }
}

impl App {
    fn session_stats_key_for_account_id(&self, account_id: i64) -> String {
        format!(
            "{}|{}|{}|{}",
            self.selected_broker.label(),
            self.form.env.label(),
            self.session_kind.label(),
            account_id
        )
    }

    fn record_session_stats(&mut self, snapshots: &[AccountSnapshot]) {
        if !self.session_stats.enabled {
            return;
        }

        let captured_at_utc = chrono::Utc::now();
        for snapshot in snapshots {
            let Some((source, value)) = tracked_balance_value(snapshot) else {
                continue;
            };
            let key = self.session_stats_key_for_account_id(snapshot.account_id);
            match self.session_stats.accounts.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(AccountSessionStats::new(
                        snapshot,
                        self.selected_broker,
                        self.form.env,
                        self.session_kind,
                        source,
                        value,
                        captured_at_utc,
                    ));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().apply_snapshot(
                        snapshot,
                        self.selected_broker,
                        self.form.env,
                        self.session_kind,
                        source,
                        value,
                        captured_at_utc,
                    );
                }
            }
        }
    }

    fn selected_session_stats(&self) -> Option<&AccountSessionStats> {
        let account_id = self.accounts.get(self.selected_account).map(|account| account.id)?;
        self.session_stats
            .accounts
            .get(&self.session_stats_key_for_account_id(account_id))
    }

    fn session_stats_enabled_label(&self) -> &'static str {
        if self.session_stats.enabled {
            "enabled"
        } else {
            "disabled"
        }
    }

    fn session_stats_log_section(&self) -> String {
        let mut body = String::new();
        body.push_str("[session_stats]\n");
        body.push_str(&format!("enabled: {}\n", self.session_stats.enabled));
        body.push_str(&format!(
            "tracker_started_at: {}\n",
            self.session_stats.started_at_utc.to_rfc3339()
        ));
        body.push_str(&format!(
            "tracked_accounts: {}\n",
            self.session_stats.tracked_accounts()
        ));
        body.push_str(&format!(
            "event_count: {}\n",
            self.session_stats.total_event_count()
        ));
        body.push('\n');

        if !self.session_stats.enabled {
            body.push_str("Session stats tracking was disabled for this run.\n");
            return body;
        }

        if self.session_stats.accounts.is_empty() {
            body.push_str("No trackable account balance samples were captured.\n");
            return body;
        }

        for stats in self.session_stats.accounts.values() {
            body.push_str("[session_stats.account]\n");
            body.push_str(&format!("account_id: {}\n", stats.account_id));
            body.push_str(&format!("account_name: {}\n", stats.account_name));
            body.push_str(&format!("broker: {}\n", stats.broker.label()));
            body.push_str(&format!("env: {}\n", stats.env.label()));
            body.push_str(&format!("mode: {}\n", stats.session_kind.label()));
            body.push_str(&format!("source: {}\n", stats.source.label()));
            body.push_str(&format!(
                "started_at: {}\n",
                stats.started_at_utc.to_rfc3339()
            ));
            body.push_str(&format!(
                "last_updated_at: {}\n",
                stats.last_updated_at_utc.to_rfc3339()
            ));
            body.push_str(&format!("samples: {}\n", stats.sample_count));
            body.push_str(&format!("moves: {}\n", stats.move_count()));
            body.push_str(&format!("wins: {}\n", stats.wins));
            body.push_str(&format!("losses: {}\n", stats.losses));
            body.push_str(&format!("flats: {}\n", stats.flat_moves));
            body.push_str(&format!("start_value: {:.2}\n", stats.start_value));
            body.push_str(&format!("current_value: {:.2}\n", stats.current_value));
            body.push_str(&format!(
                "session_pnl: {}\n",
                format_signed_money(Some(stats.session_pnl()))
            ));
            body.push_str(&format!(
                "avg_win: {}\n",
                format_signed_money(stats.avg_win())
            ));
            body.push_str(&format!(
                "max_win: {}\n",
                format_signed_money(stats.max_win)
            ));
            body.push_str(&format!(
                "avg_loss: {}\n",
                format_signed_money(stats.avg_loss_signed())
            ));
            body.push_str(&format!(
                "max_loss: {}\n",
                format_signed_money(stats.max_loss_signed())
            ));
            body.push_str(&format!("win_rate: {}\n", format_percent(stats.win_rate())));
            body.push_str(&format!(
                "profit_factor: {}\n",
                format_ratio(stats.profit_factor())
            ));
            body.push_str("events:\n");
            if stats.events.is_empty() {
                body.push_str("  none\n");
            } else {
                for event in &stats.events {
                    body.push_str(&format!(
                        "  {} source={} prev={:.2} current={:.2} delta={}\n",
                        event.recorded_at_utc.to_rfc3339(),
                        event.source.label(),
                        event.previous_value,
                        event.current_value,
                        format_signed_money(Some(event.delta)),
                    ));
                }
            }
            body.push('\n');
        }

        body
    }
}
