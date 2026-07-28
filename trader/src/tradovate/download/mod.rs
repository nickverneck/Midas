use super::*;
use crate::replay_cache::{
    ReplayCacheChunkedWriteOutcome, ReplayCacheContract, ReplayCacheContractMetadata,
    ReplayCacheInstrument, ReplayCacheMetadataAccount, ReplayCacheMetadataContext,
    ReplayCacheMetadataSnapshot, ReplayCacheRawTickCheckpointIdentity,
    ReplayCacheRawTickChunkPlanWrite, ReplayCacheRawTickRow, ReplayCacheSuggestedCoverage,
    ReplayCacheTickSpecs, finalize_raw_tick_chunk_cache, prepare_raw_tick_chunk_cache,
    record_raw_tick_chunk_attempt, split_raw_tick_checkpoint_chunk, write_raw_tick_chunk_cache,
};
use crate::replay_download::{
    DownloadCapClassification, DownloadCompletionEvidence, DownloadWindow,
    HistoricalDownloadFailure, HistoricalDownloadFailureKind, HistoricalDownloadTelemetry,
    plan_fixed_windows,
};

mod chunked;
mod metadata;
mod protocol;
mod session;
mod types;
mod websocket;

pub use chunked::download_replay_raw_ticks_chunked_to_cache;
pub use metadata::{inspect_replay_download_contract, search_replay_download_contracts};
pub use protocol::{
    extract_historical_raw_ticks_from_chart_message,
    extract_historical_server_bars_from_chart_message, raw_tick_chart_request_body,
    server_bar_chart_request_body,
};
pub use session::{
    download_replay_raw_ticks, download_replay_raw_ticks_after_auth, download_replay_server_bars,
    download_replay_server_bars_after_auth, prepare_replay_download_session,
    prepare_replay_download_session_after_auth,
};
pub use types::*;

#[cfg(test)]
mod tests;
