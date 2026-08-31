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

/// The command queues are intentionally bounded at the producer boundary.
/// Order-producing code must observe a full or closed queue synchronously;
/// putting an unbounded compatibility queue in front of the worker only moves
/// the memory problem downstream.
pub(crate) const BROKER_COMMAND_QUEUE_CAPACITY: usize = 256;
pub(crate) const USER_SOCKET_COMMAND_QUEUE_CAPACITY: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandSendError {
    Full,
    Closed,
}

impl std::fmt::Display for CommandSendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Full => "queue is full",
            Self::Closed => "queue is closed",
        })
    }
}

impl std::error::Error for CommandSendError {}

/// Object-safe producer boundary used by order/execution code. The concrete
/// production senders below wrap bounded Tokio channels; test-only adapters
/// keep the existing unit tests that use unbounded channels source-compatible.
pub(crate) trait BrokerCommandSink: Send + Sync {
    fn send(&self, command: BrokerCommand) -> Result<(), CommandSendError>;
}

pub(crate) trait UserSocketCommandSink: Send + Sync {
    fn send(&self, command: UserSocketCommand) -> Result<(), CommandSendError>;
}

#[cfg(not(test))]
#[derive(Clone)]
pub(crate) struct BrokerCommandSender(tokio::sync::mpsc::Sender<BrokerCommand>);

#[cfg(test)]
pub(crate) type BrokerCommandSender = UnboundedSender<BrokerCommand>;

#[cfg(not(test))]
impl BrokerCommandSender {
    fn bounded(sender: tokio::sync::mpsc::Sender<BrokerCommand>) -> Self {
        Self(sender)
    }

    /// Compatibility with the existing service command handlers. This is a
    /// non-blocking bounded send, not an unbounded Tokio send.
    pub(crate) fn send(&self, command: BrokerCommand) -> Result<(), CommandSendError> {
        self.0.try_send(command).map_err(|error| match error {
            tokio::sync::mpsc::error::TrySendError::Full(_) => CommandSendError::Full,
            tokio::sync::mpsc::error::TrySendError::Closed(_) => CommandSendError::Closed,
        })
    }
}

#[cfg(not(test))]
impl BrokerCommandSink for BrokerCommandSender {
    fn send(&self, command: BrokerCommand) -> Result<(), CommandSendError> {
        self.send(command)
    }
}

#[cfg(test)]
impl BrokerCommandSink for UnboundedSender<BrokerCommand> {
    fn send(&self, command: BrokerCommand) -> Result<(), CommandSendError> {
        self.send(command).map_err(|_| CommandSendError::Closed)
    }
}

#[cfg(not(test))]
#[derive(Clone)]
pub(crate) struct UserSocketCommandSender(tokio::sync::mpsc::Sender<UserSocketCommand>);

#[cfg(test)]
pub(crate) type UserSocketCommandSender = UnboundedSender<UserSocketCommand>;

#[cfg(not(test))]
impl UserSocketCommandSender {
    fn bounded(sender: tokio::sync::mpsc::Sender<UserSocketCommand>) -> Self {
        Self(sender)
    }
}

#[cfg(not(test))]
impl UserSocketCommandSink for UserSocketCommandSender {
    fn send(&self, command: UserSocketCommand) -> Result<(), CommandSendError> {
        self.0.try_send(command).map_err(|error| match error {
            tokio::sync::mpsc::error::TrySendError::Full(_) => CommandSendError::Full,
            tokio::sync::mpsc::error::TrySendError::Closed(_) => CommandSendError::Closed,
        })
    }
}

#[cfg(test)]
impl UserSocketCommandSink for UnboundedSender<UserSocketCommand> {
    fn send(&self, command: UserSocketCommand) -> Result<(), CommandSendError> {
        self.send(command).map_err(|_| CommandSendError::Closed)
    }
}

#[cfg(not(test))]
pub(crate) type BrokerCommandReceiver = tokio::sync::mpsc::Receiver<BrokerCommand>;

#[cfg(test)]
pub(crate) type BrokerCommandReceiver = UnboundedReceiver<BrokerCommand>;

#[cfg(not(test))]
pub(crate) type UserSocketCommandReceiver = tokio::sync::mpsc::Receiver<UserSocketCommand>;

#[cfg(test)]
pub(crate) type UserSocketCommandReceiver = UnboundedReceiver<UserSocketCommand>;

pub(crate) fn broker_command_channel() -> (BrokerCommandSender, BrokerCommandReceiver) {
    #[cfg(not(test))]
    {
        let (sender, receiver) = tokio::sync::mpsc::channel(BROKER_COMMAND_QUEUE_CAPACITY);
        (BrokerCommandSender::bounded(sender), receiver)
    }
    #[cfg(test)]
    {
        tokio::sync::mpsc::unbounded_channel()
    }
}

pub(crate) fn user_socket_command_channel() -> (UserSocketCommandSender, UserSocketCommandReceiver)
{
    #[cfg(not(test))]
    {
        let (sender, receiver) = tokio::sync::mpsc::channel(USER_SOCKET_COMMAND_QUEUE_CAPACITY);
        (UserSocketCommandSender::bounded(sender), receiver)
    }
    #[cfg(test)]
    {
        tokio::sync::mpsc::unbounded_channel()
    }
}

struct ServiceState {
    client: Client,
    broker_tx: BrokerCommandSender,
    replay_speed_tx: tokio::sync::watch::Sender<ReplaySpeed>,
    replay_speed: ReplaySpeed,
    replay_execution_ledger: replay::ReplayExecutionLedgerState,
    /// Keep the broker gateway owned by the service.  Dropping its join
    /// handle would detach a websocket/order task when the service exits.
    broker_task: Option<JoinHandle<()>>,
    session: Option<SessionState>,
    replay: Option<replay::ReplayState>,
    user_task: Option<JoinHandle<()>>,
    market_task: Option<JoinHandle<()>>,
    rest_probe_task: Option<JoinHandle<()>>,
    /// Snapshot construction clones bounded-but-heavy account/fill state.
    /// Keep the task owned so session replacement can cancel and await it
    /// instead of letting a detached build outlive the service.
    snapshot_task: Option<JoinHandle<()>>,
    replay_lookup_job: Option<ReplayLookupJob>,
    replay_download_job: Option<ReplayDownloadJob>,
    latency: LatencySnapshot,
    snapshot_generation: u64,
    snapshot_revision: u64,
    /// A refresh requested during the owned build is represented by one bit
    /// of state and serviced after that build completes.
    snapshot_refresh_pending: bool,
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
    request_tx: UserSocketCommandSender,
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
    market_update_sequence: Option<u64>,
    market_update_kind: MarketHistoryUpdate,
    hma_execution: HmaAngleExecutionState,
    ema_execution: EmaCrossExecutionState,
    hma_cross_execution: HmaCrossExecutionState,
    volume_hma_cross_execution: HmaCrossExecutionState,
    volume_ema_cross_execution: EmaCrossExecutionState,
    adx_execution: crate::strategies::adx::AdxExecutionState,
    /// Replay-only structured strategy decision rows. Kept in runtime state
    /// so live session construction remains allocation-free and unchanged.
    replay_signal_diagnostics: Vec<crate::broker::ReplaySignalDiagnostic>,
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
        self.market_update_sequence = None;
        self.market_update_kind = MarketHistoryUpdate::Snapshot;
        self.last_dispatched_signal_bar_ts = None;
        self.last_dispatched_entry_signal = None;
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
        self.hma_cross_execution = HmaCrossExecutionState::default();
        self.volume_hma_cross_execution = HmaCrossExecutionState::default();
        self.volume_ema_cross_execution = EmaCrossExecutionState::default();
        self.adx_execution = crate::strategies::adx::AdxExecutionState::default();
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
    /// Fees for entities evicted from the bounded raw caches. This preserves
    /// account fee totals without retaining the full broker payload forever.
    evicted_fee_totals: BTreeMap<i64, f64>,
    /// For an evicted fee whose fill is still cached, retain only the winning
    /// amount and account. This prevents the fill commission fallback from
    /// being counted alongside the already-aggregated explicit fee.
    evicted_fee_by_fill: BTreeMap<i64, (i64, f64)>,
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
pub(crate) const ENGINE_MARKET_BAR_LIMIT: usize = 4_096;
// These maps are a recent-state cache used for account/F6 snapshots, not a
// durable ledger. Keep a generous window while preventing an all-day stream
// from retaining every historical entity forever.
pub(crate) const USER_STORE_HISTORY_FILL_LIMIT: usize = 50_000;
pub(crate) const USER_STORE_FILL_FEE_LIMIT: usize = 50_000;
pub(crate) const USER_STORE_REPLAY_FILL_LIMIT: usize = 50_000;
pub(crate) const USER_STORE_EVICTED_FEE_LIMIT: usize = 100_000;
pub(crate) const USER_STORE_COMMAND_LIMIT: usize = 8_192;
pub(crate) const USER_STORE_COMMAND_REPORT_LIMIT: usize = 8_192;
pub(crate) const USER_STORE_ORDER_STRATEGY_LIMIT: usize = 8_192;
pub(crate) const USER_STORE_ORDER_STRATEGY_LINK_LIMIT: usize = 8_192;
pub(crate) const USER_STORE_POSITION_LIMIT: usize = 4_096;
pub(crate) const USER_STORE_ORDER_LIMIT: usize = 50_000;
// Keep enough closed bars for the dashboard to render long native indicators.
// A 300-period HMA needs 317 bars before its first finite value (plus one
// prior bar for crossover display), while the chart itself still only paints
// the latest 180 bars.  This is display-only; the execution path retains its
// independent ENGINE_MARKET_BAR_LIMIT history.
const UI_MARKET_BAR_LIMIT: usize = 512;
