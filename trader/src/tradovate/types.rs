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

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenFileSnapshot {
    path: PathBuf,
    modified: Option<SystemTime>,
    len: u64,
    content_hash: u64,
}

struct ServiceState {
    client: Client,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_tx: tokio::sync::watch::Sender<ReplaySpeed>,
    replay_speed: ReplaySpeed,
    replay_execution_ledger: replay::ReplayExecutionLedgerState,
    session: Option<SessionState>,
    replay: Option<replay::ReplayState>,
    user_task: Option<JoinHandle<()>>,
    market_task: Option<JoinHandle<()>>,
    rest_probe_task: Option<JoinHandle<()>>,
    replay_lookup_job: Option<ReplayLookupJob>,
    replay_download_job: Option<ReplayDownloadJob>,
    latency: LatencySnapshot,
    snapshot_revision: u64,
}

struct ReplayLookupJob {
    operation_id: ReplayDownloadOperationId,
    task: JoinHandle<()>,
}

struct ReplayDownloadJob {
    operation_id: ReplayDownloadOperationId,
    cancel_tx: tokio::sync::watch::Sender<bool>,
    stage: Arc<AtomicU8>,
    task: JoinHandle<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ReplayDownloadJobStage {
    Network = 0,
    CancelRequested = 1,
    Committing = 2,
    Finished = 3,
}

impl ReplayDownloadJob {
    fn stage(&self) -> ReplayDownloadJobStage {
        ReplayDownloadJobStage::from_raw(self.stage.load(Ordering::Acquire))
    }

    fn request_cancellation(&self) -> ReplayDownloadJobStage {
        match self.stage.compare_exchange(
            ReplayDownloadJobStage::Network as u8,
            ReplayDownloadJobStage::CancelRequested as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                let _ = self.cancel_tx.send(true);
                ReplayDownloadJobStage::CancelRequested
            }
            Err(stage) => {
                let stage = ReplayDownloadJobStage::from_raw(stage);
                if stage == ReplayDownloadJobStage::Committing {
                    // An atomic chunk/final-manifest commit must finish, but a
                    // resumable download should stop before its next chunk.
                    let _ = self.cancel_tx.send(true);
                }
                stage
            }
        }
    }
}

impl ReplayDownloadJobStage {
    fn from_raw(stage: u8) -> Self {
        match stage {
            0 => ReplayDownloadJobStage::Network,
            1 => ReplayDownloadJobStage::CancelRequested,
            2 => ReplayDownloadJobStage::Committing,
            _ => ReplayDownloadJobStage::Finished,
        }
    }
}

struct SessionState {
    cfg: AppConfig,
    session_kind: SessionKind,
    replay_enabled: bool,
    tokens: TokenBundle,
    token_file_snapshot: Option<TokenFileSnapshot>,
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
    candle_mode: CandleMode,
    market: MarketSnapshot,
    managed_protection: BTreeMap<StrategyProtectionKey, ManagedProtectionOrders>,
    active_order_strategy: Option<TrackedOrderStrategy>,
    next_strategy_order_nonce: u64,
    engine_run: Option<EngineRunState>,
}

#[derive(Debug, Clone)]
struct EngineRunState {
    run_id: String,
    order_prefix: String,
    started_at_utc: DateTime<Utc>,
    account_id: i64,
    account_name: String,
    contract_id: i64,
    contract_name: String,
    owned_order_ids: BTreeSet<i64>,
    history: EngineHistorySnapshot,
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
    started_at: time::Instant,
    flat_seen_at: Option<time::Instant>,
}

#[derive(Debug, Clone, Default)]
struct ExecutionRuntimeState {
    armed: bool,
    last_closed_bar_ts: Option<i64>,
    last_closed_bar_fingerprint: Option<u64>,
    last_dispatched_signal_bar_ts: Option<i64>,
    last_dispatched_entry_signal: Option<StrategySignal>,
    pending_target_qty: Option<i32>,
    pending_reversal_entry: Option<PendingNativeReversalEntry>,
    last_summary: String,
    hma_execution: HmaAngleExecutionState,
    ema_execution: EmaCrossExecutionState,
    hma_cross_execution: HmaCrossExecutionState,
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
        self.last_dispatched_signal_bar_ts = None;
        self.last_dispatched_entry_signal = None;
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
        self.hma_cross_execution = HmaCrossExecutionState::default();
    }
}

#[derive(Clone, Default)]
struct UserSyncStore {
    accounts: BTreeMap<i64, Value>,
    risk: BTreeMap<i64, Value>,
    cash: BTreeMap<i64, Value>,
    positions: BTreeMap<i64, BTreeMap<i64, Value>>,
    orders: BTreeMap<i64, BTreeMap<i64, Value>>,
    commands: BTreeMap<i64, Value>,
    command_reports: BTreeMap<i64, Value>,
    reported_command_rejections: BTreeSet<i64>,
    fills: BTreeMap<i64, BTreeMap<i64, Value>>,
    history_fills: BTreeMap<i64, Value>,
    fill_fees: BTreeMap<i64, Value>,
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

const TOKEN_REFRESH_LEAD_SECS: i64 = 900;
const SESSION_MAINTENANCE_INTERVAL_SECS: u64 = 30;
const ENGINE_MARKET_BAR_LIMIT: usize = 4_096;
const UI_MARKET_BAR_LIMIT: usize = 256;
