const SESSION_STATS_DELTA_EPSILON: f64 = 0.005;
const SESSION_STATS_FEE_MATCH_EPSILON: f64 = 0.015;
const SESSION_STATS_KNOWN_FEE_AMOUNTS: &[f64] = &[0.35, 0.56, 0.91];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionTradeSide {
    Long,
    Short,
    Flat,
    Unknown,
}

impl SessionTradeSide {
    fn label(self) -> &'static str {
        match self {
            Self::Long => "long",
            Self::Short => "short",
            Self::Flat => "flat",
            Self::Unknown => "unknown",
        }
    }

    fn from_signed_qty(qty: Option<f64>) -> Self {
        let Some(qty) = qty.filter(|value| value.is_finite()) else {
            return Self::Unknown;
        };
        if qty > 0.0 {
            Self::Long
        } else if qty < 0.0 {
            Self::Short
        } else {
            Self::Flat
        }
    }

    fn non_flat(self) -> Option<Self> {
        matches!(self, Self::Long | Self::Short).then_some(self)
    }
}

#[derive(Debug, Clone, Default)]
struct SessionSideDeltaStats {
    events: usize,
    wins: usize,
    losses: usize,
    pnl: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct SessionHourlyDeltaStats {
    events: usize,
    wins: usize,
    losses: usize,
    raw_pnl: f64,
    trade_pnl: f64,
    fees: f64,
}

impl SessionHourlyDeltaStats {
    fn record(&mut self, event: &SessionBalanceEvent) {
        self.events += 1;
        self.raw_pnl += event.delta;
        self.trade_pnl += event.trade_delta;
        self.fees += event.fee_delta;
        if event.trade_delta > SESSION_STATS_DELTA_EPSILON {
            self.wins += 1;
        } else if event.trade_delta < -SESSION_STATS_DELTA_EPSILON {
            self.losses += 1;
        }
    }
}

impl SessionSideDeltaStats {
    fn record(&mut self, delta: f64) {
        self.events += 1;
        self.pnl += delta;
        if delta > 0.0 {
            self.wins += 1;
        } else {
            self.losses += 1;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionBalanceEventKind {
    Trade,
    Fee,
    Mixed,
}

impl SessionBalanceEventKind {
    fn label(self) -> &'static str {
        match self {
            Self::Trade => "trade",
            Self::Fee => "fee",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SessionFeeContext {
    tick_value: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct SessionDeltaClassification {
    kind: SessionBalanceEventKind,
    fee_delta: f64,
    trade_delta: f64,
}

#[derive(Debug, Clone)]
struct SessionBalanceEvent {
    recorded_at_utc: chrono::DateTime<chrono::Utc>,
    source: SessionStatSource,
    side: SessionTradeSide,
    previous_position_side: SessionTradeSide,
    current_position_side: SessionTradeSide,
    kind: SessionBalanceEventKind,
    previous_value: f64,
    current_value: f64,
    delta: f64,
    fee_delta: f64,
    trade_delta: f64,
}

#[derive(Debug, Clone)]
struct AccountSessionStats {
    engine_identity: String,
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
    last_position_side: SessionTradeSide,
    last_delta: Option<f64>,
    last_trade_delta: Option<f64>,
    wins: usize,
    losses: usize,
    flat_moves: usize,
    fee_events: usize,
    total_fees: f64,
    gross_wins: f64,
    gross_losses: f64,
    max_win: Option<f64>,
    max_loss: Option<f64>,
    long_side: SessionSideDeltaStats,
    short_side: SessionSideDeltaStats,
    flat_side: SessionSideDeltaStats,
    unknown_side: SessionSideDeltaStats,
    events: Vec<SessionBalanceEvent>,
}

impl AccountSessionStats {
    fn new(
        engine_identity: String,
        snapshot: &AccountSnapshot,
        broker: BrokerKind,
        env: TradingEnvironment,
        session_kind: SessionKind,
        source: SessionStatSource,
        value: f64,
        captured_at_utc: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            engine_identity,
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
            last_position_side: session_trade_side_from_snapshot(snapshot),
            last_delta: None,
            last_trade_delta: None,
            wins: 0,
            losses: 0,
            flat_moves: 0,
            fee_events: 0,
            total_fees: 0.0,
            gross_wins: 0.0,
            gross_losses: 0.0,
            max_win: None,
            max_loss: None,
            long_side: SessionSideDeltaStats::default(),
            short_side: SessionSideDeltaStats::default(),
            flat_side: SessionSideDeltaStats::default(),
            unknown_side: SessionSideDeltaStats::default(),
            events: Vec::new(),
        }
    }

    fn apply_snapshot(
        &mut self,
        engine_identity: String,
        snapshot: &AccountSnapshot,
        broker: BrokerKind,
        env: TradingEnvironment,
        session_kind: SessionKind,
        source: SessionStatSource,
        value: f64,
        fee_context: SessionFeeContext,
        captured_at_utc: chrono::DateTime<chrono::Utc>,
    ) {
        self.engine_identity = engine_identity;
        self.account_name = snapshot.account_name.clone();
        self.broker = broker;
        self.env = env;
        self.session_kind = session_kind;
        self.last_updated_at_utc = captured_at_utc;
        self.sample_count += 1;
        let previous_position_side = self.last_position_side;
        let current_position_side = session_trade_side_from_snapshot(snapshot);

        if self.source != source {
            self.source = source;
            self.current_value = value;
            self.last_delta = None;
            self.last_trade_delta = None;
            self.last_position_side = current_position_side;
            return;
        }

        let delta = value - self.current_value;
        if delta.abs() < SESSION_STATS_DELTA_EPSILON {
            self.current_value = value;
            self.last_delta = Some(0.0);
            self.last_trade_delta = Some(0.0);
            self.flat_moves += 1;
            self.last_position_side = current_position_side;
            return;
        }

        let previous_value = self.current_value;
        let side = session_trade_side_for_delta(source, previous_position_side, current_position_side);
        let classification = classify_session_balance_delta(delta, fee_context);
        self.current_value = value;
        self.last_delta = Some(delta);
        self.last_trade_delta = Some(classification.trade_delta);
        self.last_position_side = current_position_side;
        self.events.push(SessionBalanceEvent {
            recorded_at_utc: captured_at_utc,
            source,
            side,
            previous_position_side,
            current_position_side,
            kind: classification.kind,
            previous_value,
            current_value: value,
            delta,
            fee_delta: classification.fee_delta,
            trade_delta: classification.trade_delta,
        });

        if classification.fee_delta.abs() >= SESSION_STATS_DELTA_EPSILON {
            self.fee_events += 1;
            self.total_fees += classification.fee_delta;
        }

        let trade_delta = classification.trade_delta;
        if trade_delta.abs() < SESSION_STATS_DELTA_EPSILON {
            return;
        }

        self.side_stats_mut(side).record(trade_delta);

        if trade_delta > 0.0 {
            self.wins += 1;
            self.gross_wins += trade_delta;
            self.max_win = Some(
                self.max_win
                    .map(|value| value.max(trade_delta))
                    .unwrap_or(trade_delta),
            );
        } else {
            let loss = trade_delta.abs();
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

    fn trade_pnl_ex_fees(&self) -> f64 {
        self.events.iter().map(|event| event.trade_delta).sum()
    }

    fn elapsed_hours(&self) -> Option<f64> {
        let elapsed_ms = self
            .last_updated_at_utc
            .signed_duration_since(self.started_at_utc)
            .num_milliseconds();
        (elapsed_ms > 0).then_some(elapsed_ms as f64 / 3_600_000.0)
    }

    fn session_pnl_per_hour(&self) -> Option<f64> {
        self.elapsed_hours()
            .map(|hours| self.session_pnl() / hours)
    }

    fn trade_pnl_per_hour(&self) -> Option<f64> {
        self.elapsed_hours()
            .map(|hours| self.trade_pnl_ex_fees() / hours)
    }

    fn hourly_stats(&self) -> Vec<(usize, SessionHourlyDeltaStats)> {
        let mut buckets = [SessionHourlyDeltaStats::default(); 24];
        for event in &self.events {
            let hour = event
                .recorded_at_utc
                .with_timezone(&chrono::Local)
                .format("%H")
                .to_string()
                .parse::<usize>()
                .unwrap_or(0)
                .min(23);
            buckets[hour].record(event);
        }
        buckets
            .into_iter()
            .enumerate()
            .filter(|(_, stats)| stats.events > 0)
            .collect()
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

    fn side_stats(&self, side: SessionTradeSide) -> &SessionSideDeltaStats {
        match side {
            SessionTradeSide::Long => &self.long_side,
            SessionTradeSide::Short => &self.short_side,
            SessionTradeSide::Flat => &self.flat_side,
            SessionTradeSide::Unknown => &self.unknown_side,
        }
    }

    fn side_stats_mut(&mut self, side: SessionTradeSide) -> &mut SessionSideDeltaStats {
        match side {
            SessionTradeSide::Long => &mut self.long_side,
            SessionTradeSide::Short => &mut self.short_side,
            SessionTradeSide::Flat => &mut self.flat_side,
            SessionTradeSide::Unknown => &mut self.unknown_side,
        }
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

fn session_trade_side_from_snapshot(snapshot: &AccountSnapshot) -> SessionTradeSide {
    SessionTradeSide::from_signed_qty(snapshot.market_position_qty)
}

fn session_trade_side_for_delta(
    source: SessionStatSource,
    previous_side: SessionTradeSide,
    current_side: SessionTradeSide,
) -> SessionTradeSide {
    match source {
        SessionStatSource::Balance | SessionStatSource::CashBalance => previous_side
            .non_flat()
            .or_else(|| current_side.non_flat())
            .unwrap_or(current_side),
        SessionStatSource::NetLiq => current_side
            .non_flat()
            .or_else(|| previous_side.non_flat())
            .unwrap_or(current_side),
    }
}

fn format_session_position_transition(
    previous_side: SessionTradeSide,
    current_side: SessionTradeSide,
) -> String {
    format!("{}->{}", previous_side.label(), current_side.label())
}

fn classify_session_balance_delta(
    delta: f64,
    fee_context: SessionFeeContext,
) -> SessionDeltaClassification {
    if let Some(fee_amount) = matching_known_fee_amount(delta.abs()) {
        if delta < 0.0 {
            return SessionDeltaClassification {
                kind: SessionBalanceEventKind::Fee,
                fee_delta: -fee_amount,
                trade_delta: 0.0,
            };
        }
    }

    if let Some(tick_value) = fee_context.tick_value {
        if let Some(classification) = classify_mixed_fee_delta(delta, tick_value) {
            return classification;
        }
    }

    SessionDeltaClassification {
        kind: SessionBalanceEventKind::Trade,
        fee_delta: 0.0,
        trade_delta: delta,
    }
}

fn classify_mixed_fee_delta(delta: f64, tick_value: f64) -> Option<SessionDeltaClassification> {
    if !tick_value.is_finite() || tick_value <= SESSION_STATS_DELTA_EPSILON {
        return None;
    }

    SESSION_STATS_KNOWN_FEE_AMOUNTS
        .iter()
        .filter_map(|fee_amount| {
            let trade_delta = delta + fee_amount;
            if trade_delta.abs() < SESSION_STATS_DELTA_EPSILON {
                return None;
            }
            let ticks = trade_delta / tick_value;
            let rounded_ticks = ticks.round();
            if rounded_ticks.abs() < 1.0 {
                return None;
            }
            let expected_trade_delta = rounded_ticks * tick_value;
            let error = (trade_delta - expected_trade_delta).abs();
            (error <= SESSION_STATS_FEE_MATCH_EPSILON).then_some((
                error,
                SessionDeltaClassification {
                    kind: SessionBalanceEventKind::Mixed,
                    fee_delta: -*fee_amount,
                    trade_delta: expected_trade_delta,
                },
            ))
        })
        .min_by(|(left_error, _), (right_error, _)| {
            left_error
                .partial_cmp(right_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, classification)| classification)
}

fn matching_known_fee_amount(amount: f64) -> Option<f64> {
    SESSION_STATS_KNOWN_FEE_AMOUNTS
        .iter()
        .copied()
        .find(|fee_amount| (amount - fee_amount).abs() <= SESSION_STATS_FEE_MATCH_EPSILON)
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

fn format_money_per_hour(value: Option<f64>) -> String {
    value
        .map(|value| format!("{}/h", format_signed_money(Some(value))))
        .unwrap_or_else(|| "n/a".to_string())
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
            "{}|{}|{}|{}|{}",
            self.active_engine_stats_identity_key(),
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
        let fee_context = self.session_fee_context();
        let engine_identity = self.active_engine_stats_identity_key();
        for snapshot in snapshots {
            let Some((source, value)) = tracked_balance_value(snapshot) else {
                continue;
            };
            let key = self.session_stats_key_for_account_id(snapshot.account_id);
            match self.session_stats.accounts.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(AccountSessionStats::new(
                        engine_identity.clone(),
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
                        engine_identity.clone(),
                        snapshot,
                        self.selected_broker,
                        self.form.env,
                        self.session_kind,
                        source,
                        value,
                        fee_context,
                        captured_at_utc,
                    );
                }
            }
        }
    }

    fn session_fee_context(&self) -> SessionFeeContext {
        let tick_value = self
            .market
            .tick_size
            .zip(self.market.value_per_point)
            .map(|(tick_size, value_per_point)| (tick_size * value_per_point).abs())
            .filter(|value| value.is_finite() && *value > SESSION_STATS_DELTA_EPSILON);
        SessionFeeContext { tick_value }
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
            body.push_str(&format!("engine_identity: {}\n", stats.engine_identity));
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
            body.push_str(&format!("fee_events: {}\n", stats.fee_events));
            body.push_str(&format!(
                "total_fees: {}\n",
                format_signed_money(Some(stats.total_fees))
            ));
            for side in [
                SessionTradeSide::Long,
                SessionTradeSide::Short,
                SessionTradeSide::Flat,
                SessionTradeSide::Unknown,
            ] {
                let side_stats = stats.side_stats(side);
                body.push_str(&format!("{}_events: {}\n", side.label(), side_stats.events));
                body.push_str(&format!(
                    "{}_pnl: {}\n",
                    side.label(),
                    format_signed_money(Some(side_stats.pnl))
                ));
                body.push_str(&format!("{}_wins: {}\n", side.label(), side_stats.wins));
                body.push_str(&format!(
                    "{}_losses: {}\n",
                    side.label(),
                    side_stats.losses
                ));
            }
            body.push_str(&format!("start_value: {:.2}\n", stats.start_value));
            body.push_str(&format!("current_value: {:.2}\n", stats.current_value));
            body.push_str(&format!(
                "session_pnl: {}\n",
                format_signed_money(Some(stats.session_pnl()))
            ));
            body.push_str(&format!(
                "trade_pnl_ex_fees: {}\n",
                format_signed_money(Some(stats.trade_pnl_ex_fees()))
            ));
            body.push_str(&format!(
                "net_pnl_per_hour: {}\n",
                format_money_per_hour(stats.session_pnl_per_hour())
            ));
            body.push_str(&format!(
                "trade_pnl_per_hour: {}\n",
                format_money_per_hour(stats.trade_pnl_per_hour())
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
                        "  {} source={} side={} pos={} kind={} prev={:.2} current={:.2} delta={} trade_delta={} fee_delta={}\n",
                        event.recorded_at_utc.to_rfc3339(),
                        event.source.label(),
                        event.side.label(),
                        format_session_position_transition(
                            event.previous_position_side,
                            event.current_position_side
                        ),
                        event.kind.label(),
                        event.previous_value,
                        event.current_value,
                        format_signed_money(Some(event.delta)),
                        format_signed_money(Some(event.trade_delta)),
                        format_signed_money(Some(event.fee_delta)),
                    ));
                }
            }
            body.push_str("hourly_local:\n");
            let hourly_stats = stats.hourly_stats();
            if hourly_stats.is_empty() {
                body.push_str("  none\n");
            } else {
                for (hour, hourly) in hourly_stats {
                    body.push_str(&format!(
                        "  {hour:02}:00 events={} wins={} losses={} net={} trade={} fees={} trade_per_hour={}\n",
                        hourly.events,
                        hourly.wins,
                        hourly.losses,
                        format_signed_money(Some(hourly.raw_pnl)),
                        format_signed_money(Some(hourly.trade_pnl)),
                        format_signed_money(Some(hourly.fees)),
                        format_money_per_hour(Some(hourly.trade_pnl)),
                    ));
                }
            }
            body.push('\n');
        }

        body
    }
}
