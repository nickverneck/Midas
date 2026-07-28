use super::*;
use crate::broker::ReplayDownloadCacheTarget;

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownloadRequest {
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub bar_type: BarType,
}

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownload {
    pub contract: ContractSuggestion,
    pub bars: Vec<Bar>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub contract_metadata: ReplayCacheContractMetadata,
    pub warnings: Vec<String>,
    pub telemetry: HistoricalDownloadTelemetry,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownloadRequest {
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownload {
    pub contract: ContractSuggestion,
    pub ticks: Vec<ReplayCacheRawTickRow>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub contract_metadata: ReplayCacheContractMetadata,
    pub warnings: Vec<String>,
    pub telemetry: HistoricalDownloadTelemetry,
}

#[derive(Debug, Clone)]
pub struct TradovateReplayContractInspection {
    pub contract: ContractSuggestion,
    pub suggested_coverage: Option<ReplayCacheSuggestedCoverage>,
}

#[derive(Debug, Clone)]
pub struct TradovateChunkedRawTickCacheRequest {
    pub instrument: String,
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub target: Option<ReplayDownloadCacheTarget>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub cache_root: PathBuf,
    pub chunk_duration: chrono::Duration,
    pub minimum_split: chrono::Duration,
    pub display_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradovateChunkedRawTickPhase {
    Resuming,
    Downloading,
    Retrying,
    Splitting,
    CommittingChunk,
    Finalizing,
}

#[derive(Debug, Clone)]
pub struct TradovateChunkedRawTickProgress {
    pub phase: TradovateChunkedRawTickPhase,
    pub window: Option<DownloadWindow>,
    pub completed_chunks: usize,
    pub total_leaf_chunks: usize,
    pub message: String,
}

/// Read-only authenticated context shared by bounded historical requests. The
/// market-data token intentionally has no Debug/Serialize surface.
pub struct TradovateReplayDownloadSession {
    pub(super) cfg: AppConfig,
    pub(super) md_access_token: String,
    pub(super) contract: ContractSuggestion,
    pub(super) tick_specs: ReplayCacheTickSpecs,
    pub(super) session_template: Option<String>,
    pub(super) contract_metadata: ReplayCacheContractMetadata,
    pub(super) warnings: Vec<String>,
    pub(super) raw_tick_last_request: tokio::sync::Mutex<Option<tokio::time::Instant>>,
}
