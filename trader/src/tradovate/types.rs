#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccessTokenResponse {
    error_text: Option<String>,
    access_token: Option<String>,
    md_access_token: Option<String>,
    expiration_time: Option<String>,
    user_id: Option<i64>,
    name: Option<String>,
    has_live: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenCacheFile {
    token: String,
    #[serde(rename = "accessToken")]
    access_token: Option<String>,
    #[serde(rename = "mdAccessToken")]
    md_access_token: Option<String>,
    #[serde(rename = "expirationTime")]
    expiration_time: Option<String>,
    #[serde(rename = "userId")]
    user_id: Option<i64>,
    name: Option<String>,
    #[serde(rename = "hasLive")]
    has_live: Option<bool>,
}

#[derive(Debug, Clone)]
struct TokenBundle {
    access_token: String,
    md_access_token: String,
    expiration_time: Option<String>,
    user_id: Option<i64>,
    user_name: Option<String>,
}

struct ServiceState {
    client: Client,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_tx: tokio::sync::watch::Sender<ReplaySpeed>,
    replay_speed: ReplaySpeed,
    session: Option<SessionState>,
    replay: Option<replay::ReplayState>,
    user_task: Option<JoinHandle<()>>,
    market_task: Option<JoinHandle<()>>,
    rest_probe_task: Option<JoinHandle<()>>,
    latency: LatencySnapshot,
    snapshot_revision: u64,
}

struct SessionState {
    cfg: AppConfig,
    session_kind: SessionKind,
    replay_enabled: bool,
    tokens: TokenBundle,
    accounts: Vec<AccountInfo>,
    request_tx: UnboundedSender<UserSocketCommand>,
    execution_config: ExecutionStrategyConfig,
    execution_runtime: ExecutionRuntimeState,
    pending_signal_context: Option<PendingSignalLatencyContext>,
    order_latency_tracker: Option<OrderLatencyTracker>,
    order_submit_in_flight: bool,
    protection_sync_in_flight: bool,
    pending_protection_sync: Option<DesiredNativeProtection>,
    user_store: UserSyncStore,
    selected_account_id: Option<i64>,
    selected_contract: Option<ContractSuggestion>,
    bar_type: BarType,
    market: MarketSnapshot,
    managed_protection: BTreeMap<StrategyProtectionKey, ManagedProtectionOrders>,
    active_order_strategy: Option<TrackedOrderStrategy>,
    next_strategy_order_nonce: u64,
}

#[derive(Debug, Clone)]
struct PendingSignalLatencyContext {
    started_at: time::Instant,
    description: String,
}

#[derive(Debug, Clone)]
struct PendingNativeReversalEntry {
    target_qty: i32,
    reason: String,
}

#[derive(Debug, Clone, Default)]
struct ExecutionRuntimeState {
    armed: bool,
    last_closed_bar_ts: Option<i64>,
    pending_target_qty: Option<i32>,
    pending_reversal_entry: Option<PendingNativeReversalEntry>,
    last_summary: String,
    hma_execution: HmaAngleExecutionState,
    ema_execution: EmaCrossExecutionState,
}

impl ExecutionRuntimeState {
    fn snapshot(&self) -> ExecutionRuntimeSnapshot {
        ExecutionRuntimeSnapshot {
            armed: self.armed,
            last_closed_bar_ts: self.last_closed_bar_ts,
            pending_target_qty: self.pending_target_qty,
            last_summary: self.last_summary.clone(),
        }
    }

    fn reset_execution(&mut self) {
        self.pending_reversal_entry = None;
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
    }
}

#[derive(Clone, Default)]
struct UserSyncStore {
    accounts: BTreeMap<i64, Value>,
    risk: BTreeMap<i64, Value>,
    cash: BTreeMap<i64, Value>,
    positions: BTreeMap<i64, BTreeMap<i64, Value>>,
    orders: BTreeMap<i64, BTreeMap<i64, Value>>,
    fills: BTreeMap<i64, BTreeMap<i64, Value>>,
    order_strategies: BTreeMap<i64, Value>,
    order_strategy_links: BTreeMap<i64, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StrategyProtectionKey {
    account_id: i64,
    contract_id: i64,
}

#[derive(Debug, Clone)]
struct ManagedProtectionOrders {
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    last_requested_take_profit_price: Option<f64>,
    last_requested_stop_price: Option<f64>,
    take_profit_cl_ord_id: Option<String>,
    stop_cl_ord_id: Option<String>,
    take_profit_order_id: Option<i64>,
    stop_order_id: Option<i64>,
}

#[derive(Debug, Clone)]
struct TrackedOrderStrategy {
    key: StrategyProtectionKey,
    order_strategy_id: i64,
    target_qty: i32,
}

const TOKEN_REFRESH_LEAD_SECS: i64 = 300;
const SESSION_MAINTENANCE_INTERVAL_SECS: u64 = 30;
const ENGINE_MARKET_BAR_LIMIT: usize = 4_096;
const UI_MARKET_BAR_LIMIT: usize = 256;
