use super::*;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct EngineKey(PathBuf);

impl EngineKey {
    pub(crate) fn from_socket_path(socket_path: &Path) -> Self {
        Self(socket_path.to_path_buf())
    }

    pub(crate) fn display_label(&self) -> String {
        self.0.display().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineConnectionState {
    Observing,
    Connected,
    Disconnected,
    Closed,
    Error,
    Stale,
}

impl EngineConnectionState {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Observing => "live",
            Self::Connected => "connected",
            Self::Disconnected => "broker off",
            Self::Closed => "closed",
            Self::Error => "error",
            Self::Stale => "stale",
        }
    }

    pub(crate) fn attachable(self) -> bool {
        matches!(
            self,
            Self::Observing | Self::Connected | Self::Disconnected | Self::Error
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EngineSummary {
    pub(crate) key: EngineKey,
    pub(crate) id: Option<u32>,
    pub(crate) socket_path: PathBuf,
    pub(crate) connection_state: EngineConnectionState,
    broker: Option<BrokerKind>,
    env: Option<TradingEnvironment>,
    auth_mode: Option<AuthMode>,
    session_kind: Option<SessionKind>,
    user_name: Option<String>,
    accounts: Vec<AccountInfo>,
    selected_account_id: Option<i64>,
    selected_account_name: Option<String>,
    selected_contract_name: Option<String>,
    market_contract_name: Option<String>,
    position_qty: Option<i32>,
    strategy_kind: Option<StrategyKind>,
    native_strategy: Option<NativeStrategyKind>,
    strategy_armed: Option<bool>,
    latency: LatencySnapshot,
    latest_status: Option<String>,
    latest_error: Option<String>,
}

impl EngineSummary {
    pub(crate) fn from_running_engine(engine: &RunningEngine) -> Self {
        Self {
            key: EngineKey::from_socket_path(&engine.socket_path),
            id: Some(engine.id),
            socket_path: engine.socket_path.clone(),
            connection_state: if engine.socket_is_live {
                EngineConnectionState::Observing
            } else {
                EngineConnectionState::Stale
            },
            broker: None,
            env: None,
            auth_mode: None,
            session_kind: None,
            user_name: None,
            accounts: Vec::new(),
            selected_account_id: None,
            selected_account_name: None,
            selected_contract_name: None,
            market_contract_name: None,
            position_qty: None,
            strategy_kind: None,
            native_strategy: None,
            strategy_armed: None,
            latency: LatencySnapshot::default(),
            latest_status: None,
            latest_error: None,
        }
    }

    pub(crate) fn live_socket(socket_path: PathBuf) -> Self {
        Self {
            key: EngineKey::from_socket_path(&socket_path),
            id: None,
            socket_path,
            connection_state: EngineConnectionState::Observing,
            broker: None,
            env: None,
            auth_mode: None,
            session_kind: None,
            user_name: None,
            accounts: Vec::new(),
            selected_account_id: None,
            selected_account_name: None,
            selected_contract_name: None,
            market_contract_name: None,
            position_qty: None,
            strategy_kind: None,
            native_strategy: None,
            strategy_armed: None,
            latency: LatencySnapshot::default(),
            latest_status: None,
            latest_error: None,
        }
    }

    pub(crate) fn refresh_from_running_engine(&mut self, engine: &RunningEngine) {
        self.key = EngineKey::from_socket_path(&engine.socket_path);
        self.id = Some(engine.id);
        self.socket_path = engine.socket_path.clone();
        if engine.socket_is_live {
            if matches!(
                self.connection_state,
                EngineConnectionState::Stale | EngineConnectionState::Closed
            ) {
                self.connection_state = EngineConnectionState::Observing;
            }
        } else {
            self.connection_state = EngineConnectionState::Stale;
            if self.latest_status.is_none() {
                self.latest_status = Some(format!(
                    "Socket {} is unavailable.",
                    self.socket_path.display()
                ));
            }
        }
    }

    pub(crate) fn apply_event(&mut self, event: &ServiceEvent) {
        match event {
            ServiceEvent::Status(message) => {
                self.latest_status = Some(message.clone());
            }
            ServiceEvent::DebugLog(_) => {}
            ServiceEvent::Error(message) => {
                self.connection_state = EngineConnectionState::Error;
                self.latest_error = Some(message.clone());
                self.latest_status = Some(message.clone());
            }
            ServiceEvent::Connected {
                broker,
                env,
                user_name,
                auth_mode,
                session_kind,
                ..
            } => {
                self.connection_state = EngineConnectionState::Connected;
                self.broker = Some(*broker);
                self.env = Some(*env);
                self.user_name = user_name.clone();
                self.auth_mode = Some(*auth_mode);
                self.session_kind = Some(*session_kind);
                self.latest_error = None;
                self.latest_status = Some(match user_name {
                    Some(name) => format!("connected as {name}"),
                    None => "connected".to_string(),
                });
            }
            ServiceEvent::Disconnected => {
                self.mark_disconnected("Disconnected");
            }
            ServiceEvent::AccountsLoaded(accounts) => {
                self.accounts = accounts.clone();
                self.sync_selected_account_name();
                self.latest_status = Some(format!("{} account(s)", self.accounts.len()));
            }
            ServiceEvent::AccountSnapshotsLoaded(snapshots) => {
                self.apply_account_snapshots(snapshots);
            }
            ServiceEvent::ContractSearchResults { .. } => {}
            ServiceEvent::MarketSnapshot(snapshot) => {
                self.market_contract_name = snapshot.contract_name.clone();
                if snapshot.contract_name.is_some() {
                    self.selected_contract_name = snapshot.contract_name.clone();
                }
                if !snapshot.status.is_empty() {
                    self.latest_status = Some(snapshot.status.clone());
                }
            }
            ServiceEvent::TradeMarkersUpdated(_) => {}
            ServiceEvent::Latency(snapshot) => {
                self.latency = *snapshot;
            }
            ServiceEvent::ExecutionState(snapshot) => {
                self.strategy_kind = Some(snapshot.config.kind);
                self.native_strategy = Some(snapshot.config.native_strategy);
                self.strategy_armed = Some(snapshot.runtime.armed);
                self.selected_account_id = snapshot.selected_account_id;
                self.sync_selected_account_name();
                if let Some(name) = &snapshot.selected_contract_name {
                    self.selected_contract_name = Some(name.clone());
                }
                self.position_qty = Some(snapshot.market_position_qty);
                if !snapshot.runtime.last_summary.is_empty() {
                    self.latest_status = Some(snapshot.runtime.last_summary.clone());
                }
            }
            ServiceEvent::ExecutionProbe(_) => {}
            ServiceEvent::ReplaySpeedUpdated(_) => {}
        }
    }

    pub(crate) fn mark_disconnected(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.connection_state = EngineConnectionState::Disconnected;
        self.latest_status = Some(message);
    }

    pub(crate) fn mark_receiver_closed(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.connection_state = EngineConnectionState::Closed;
        self.latest_status = Some(message);
    }

    pub(crate) fn broker_mode_label(&self) -> String {
        match (self.broker, self.env, self.session_kind) {
            (Some(broker), Some(env), Some(session_kind)) => {
                format!(
                    "{} {} {}",
                    broker.label(),
                    env.label(),
                    session_kind.label()
                )
            }
            (Some(broker), Some(env), None) => format!("{} {}", broker.label(), env.label()),
            (Some(broker), None, Some(session_kind)) => {
                format!("{} {}", broker.label(), session_kind.label())
            }
            (Some(broker), None, None) => broker.label().to_string(),
            _ => "-".to_string(),
        }
    }

    pub(crate) fn account_label(&self) -> String {
        if let Some(name) = &self.selected_account_name {
            return name.clone();
        }
        if let Some(id) = self.selected_account_id {
            return format!("#{id}");
        }
        if !self.accounts.is_empty() {
            return format!("{} acct", self.accounts.len());
        }
        "-".to_string()
    }

    pub(crate) fn instrument_label(&self) -> String {
        self.selected_contract_name
            .as_ref()
            .or(self.market_contract_name.as_ref())
            .cloned()
            .unwrap_or_else(|| "-".to_string())
    }

    pub(crate) fn position_label(&self) -> String {
        self.position_qty
            .map(|qty| qty.to_string())
            .unwrap_or_else(|| "-".to_string())
    }

    pub(crate) fn strategy_label(&self) -> String {
        let strategy = match (self.strategy_kind, self.native_strategy) {
            (Some(StrategyKind::Native), Some(native)) => native.label().to_string(),
            (Some(kind), _) => kind.label().to_string(),
            _ => "-".to_string(),
        };
        match self.strategy_armed {
            Some(true) => format!("{strategy} armed"),
            Some(false) => format!("{strategy} idle"),
            None => strategy,
        }
    }

    pub(crate) fn latency_label(&self) -> String {
        self.latency
            .rest_rtt_ms
            .map(|ms| format!("{ms}ms"))
            .unwrap_or_else(|| "-".to_string())
    }

    pub(crate) fn status_label(&self) -> String {
        self.latest_error
            .as_ref()
            .or(self.latest_status.as_ref())
            .cloned()
            .unwrap_or_else(|| "-".to_string())
    }

    fn sync_selected_account_name(&mut self) {
        self.selected_account_name = self.selected_account_id.and_then(|selected_id| {
            self.accounts
                .iter()
                .find(|account| account.id == selected_id)
                .map(|account| account.name.clone())
        });
    }

    fn apply_account_snapshots(&mut self, snapshots: &[AccountSnapshot]) {
        let selected = self
            .selected_account_id
            .and_then(|id| snapshots.iter().find(|snapshot| snapshot.account_id == id))
            .or_else(|| snapshots.first());
        if let Some(snapshot) = selected {
            self.selected_account_id.get_or_insert(snapshot.account_id);
            self.selected_account_name = Some(snapshot.account_name.clone());
            self.position_qty = snapshot
                .market_position_qty
                .or(snapshot.open_position_qty)
                .map(|qty| qty.round() as i32);
        }
    }
}
