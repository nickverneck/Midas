#[cfg(feature = "replay")]
use super::ReplayFrameSet;
use super::{
    AccountInfo, AccountSnapshot, BarType, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, EngineHistorySnapshot, EngineHistorySummary, ExecutionProbeSnapshot,
    LatencySnapshot, MarketSnapshot, ReplayDownloadCacheTarget, ReplayDownloadOperationId,
    ReplayExecutionLedgerSnapshot, ReplayExecutionLedgerSummary, ReplaySpeed, TradeMarker,
};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategy::{ExecutionStateSnapshot, ExecutionStrategyConfig};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(feature = "replay")]
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceCommand {
    Connect(AppConfig),
    EnterReplayMode {
        config: AppConfig,
        bar_type: BarType,
        candle_mode: CandleMode,
        replay_dataset_manifest: Option<PathBuf>,
        #[serde(default)]
        replay_dataset_view: Option<PathBuf>,
    },
    /// Replay-sweep-only entry point that reuses immutable bars/frames while
    /// keeping broker, strategy, and ledger state private to the child.
    #[cfg(feature = "replay")]
    EnterReplayModeWithSharedFrames {
        config: AppConfig,
        bar_type: BarType,
        candle_mode: CandleMode,
        replay_dataset_manifest: Option<PathBuf>,
        #[serde(default)]
        replay_dataset_view: Option<PathBuf>,
        #[serde(skip)]
        replay_shared_frames: Option<Arc<ReplayFrameSet>>,
    },
    DownloadReplayData {
        operation_id: ReplayDownloadOperationId,
        config: AppConfig,
        instrument: String,
        contract: ContractSuggestion,
        target: Option<ReplayDownloadCacheTarget>,
        start_date: NaiveDate,
        end_date: NaiveDate,
        source_kind: String,
        bar_type: BarType,
        candle_mode: CandleMode,
        display_name: Option<String>,
        tags: Vec<String>,
    },
    SearchReplayDownloadContracts {
        operation_id: ReplayDownloadOperationId,
        config: AppConfig,
        query: String,
        limit: usize,
    },
    InspectReplayDownloadContract {
        operation_id: ReplayDownloadOperationId,
        config: AppConfig,
        contract: ContractSuggestion,
    },
    CancelReplayDownloadOperation {
        operation_id: ReplayDownloadOperationId,
    },
    /// Request the state already held by the engine. This is deliberately
    /// distinct from ReplayState, which is also used by TUI hydration and
    /// may refresh broker history.
    InspectState,
    ReplayState,
    SelectAccount {
        account_id: i64,
    },
    SearchContracts {
        query: String,
        limit: usize,
    },
    SubscribeBars {
        contract: ContractSuggestion,
        bar_type: BarType,
        candle_mode: CandleMode,
    },
    SetReplaySpeed {
        speed: ReplaySpeed,
    },
    ManualOrder {
        action: ManualOrderAction,
    },
    SetTargetPosition {
        target_qty: i32,
        automated: bool,
        reason: String,
    },
    ProfileLegacyOrderStrategyTarget {
        target_qty: i32,
        reason: String,
    },
    SyncNativeProtection {
        signed_qty: i32,
        take_profit_price: Option<f64>,
        stop_price: Option<f64>,
        reason: String,
    },
    SetExecutionStrategyConfig(ExecutionStrategyConfig),
    ArmExecutionStrategy,
    DisarmExecutionStrategy {
        reason: String,
    },
    ProbeExecution {
        tag: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ManualOrderAction {
    Buy,
    Sell,
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceEvent {
    Status(String),
    DebugLog(String),
    BrokerRejection(String),
    Error(String),
    Connected {
        broker: BrokerKind,
        env: TradingEnvironment,
        user_name: Option<String>,
        auth_mode: AuthMode,
        session_kind: SessionKind,
        capabilities: BrokerCapabilities,
    },
    Disconnected,
    AccountsLoaded(Vec<AccountInfo>),
    AccountSnapshotsLoaded(Vec<AccountSnapshot>),
    ContractSearchResults {
        query: String,
        results: Vec<ContractSuggestion>,
    },
    MarketSnapshot(MarketSnapshot),
    TradeMarkersUpdated(Vec<TradeMarker>),
    EngineHistoryUpdated(EngineHistorySnapshot),
    /// A bounded, read-only snapshot for inspection tools. It contains no
    /// raw account/fill payloads and must be answered from cached state.
    StateInspected(EngineInspectionSnapshot),
    Latency(LatencySnapshot),
    ExecutionState(ExecutionStateSnapshot),
    ExecutionProbe(ExecutionProbeSnapshot),
    ReplaySpeedUpdated(ReplaySpeed),
    ReplayExecutionLedgerUpdated(ReplayExecutionLedgerSummary),
    ReplayExecutionLedgerSnapshot(ReplayExecutionLedgerSnapshot),
    /// Emitted after the replay worker has durably written its result
    /// artifacts. Headless sweep workers use this as the completion barrier.
    ReplayResultSaved {
        run_id: String,
        result_path: PathBuf,
        status: String,
        fill_count: usize,
        trade_count: usize,
    },
    ReplayDownloadProgress {
        operation_id: ReplayDownloadOperationId,
        phase: ReplayDownloadPhase,
        message: String,
        estimated_rows: Option<u64>,
        estimated_bytes: Option<u64>,
    },
    ReplayDownloadContractSearchResults {
        operation_id: ReplayDownloadOperationId,
        query: String,
        results: Vec<ContractSuggestion>,
    },
    ReplayDownloadContractInspected {
        operation_id: ReplayDownloadOperationId,
        contract: ContractSuggestion,
        suggested_start_date: Option<NaiveDate>,
        suggested_end_date: Option<NaiveDate>,
        suggestion_basis: Option<String>,
    },
    ReplayDownloadCompleted {
        operation_id: ReplayDownloadOperationId,
        cache_root: PathBuf,
        manifest_path: PathBuf,
        data_path: PathBuf,
        rows: u64,
        bytes: u64,
    },
    ReplayDownloadFailed {
        operation_id: ReplayDownloadOperationId,
        phase: ReplayDownloadPhase,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineInspectionSnapshot {
    pub broker: BrokerKind,
    pub env: TradingEnvironment,
    pub session_kind: SessionKind,
    pub account_id: Option<i64>,
    pub account_name: Option<String>,
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub execution: ExecutionStateSnapshot,
    pub history: Option<EngineHistorySummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDownloadPhase {
    Idle,
    Searching,
    InspectingContract,
    Ready,
    Authenticating,
    Downloading,
    WritingCache,
    Busy,
    Cancelling,
    Cancelled,
    Complete,
    Failed,
}

impl ReplayDownloadPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Searching => "Searching",
            Self::InspectingContract => "Inspecting contract",
            Self::Ready => "Ready",
            Self::Authenticating => "Authenticating",
            Self::Downloading => "Downloading",
            Self::WritingCache => "Writing cache",
            Self::Busy => "Busy",
            Self::Cancelling => "Cancelling",
            Self::Cancelled => "Cancelled",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
        }
    }

    #[cfg(feature = "replay")]
    pub fn is_busy(self) -> bool {
        matches!(
            self,
            Self::Searching
                | Self::InspectingContract
                | Self::Authenticating
                | Self::Downloading
                | Self::WritingCache
                | Self::Cancelling
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionKind {
    Live,
    Replay,
}

impl SessionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Replay => "Replay",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspection_command_and_event_are_wire_serializable() {
        let command = serde_json::to_string(&ServiceCommand::InspectState)
            .expect("serialize inspection command");
        assert!(matches!(
            serde_json::from_str::<ServiceCommand>(&command),
            Ok(ServiceCommand::InspectState)
        ));

        let event = ServiceEvent::StateInspected(EngineInspectionSnapshot {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            session_kind: SessionKind::Live,
            account_id: Some(42),
            account_name: Some("SIM42".to_string()),
            contract_id: Some(7),
            contract_name: Some("GCZ6".to_string()),
            execution: ExecutionStateSnapshot::default(),
            history: None,
        });
        let encoded = serde_json::to_string(&event).expect("serialize inspection event");
        assert!(encoded.contains("StateInspected"));
        assert!(!encoded.contains("\"fills\""));
        assert!(matches!(
            serde_json::from_str::<ServiceEvent>(&encoded),
            Ok(ServiceEvent::StateInspected(snapshot)) if snapshot.account_id == Some(42)
        ));
    }
}
