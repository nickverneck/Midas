use super::*;

#[cfg(feature = "replay")]
mod bars;
#[cfg(feature = "replay")]
mod fees;
#[cfg(feature = "replay")]
mod instrument;
mod ledger;
mod load;
#[cfg(feature = "replay")]
mod results;
#[cfg(feature = "replay")]
mod risk;
mod state;
#[cfg(feature = "replay")]
mod sweep;
#[cfg(feature = "replay")]
mod sweep_analytics;
#[cfg(feature = "replay")]
mod sweep_runner;
#[cfg(feature = "replay")]
mod ticks;
#[cfg(feature = "replay")]
pub(crate) mod virtual_time;
mod worker;

#[cfg(feature = "replay")]
pub(crate) use fees::ReplayFeeSchedule;
pub(crate) use ledger::ReplayExecutionLedgerState;
#[cfg(feature = "replay")]
pub(crate) use ledger::ReplayLedgerMarketContext;
pub(crate) use load::load_replay_state;
#[cfg(feature = "replay")]
pub(crate) use results::{
    ReplayFeeScenario, ReplayResultEntry, ReplayResultInput, ReplayResultLibrarySnapshot,
    ReplayResultStatus, ReplayTradeExcursion, analyze_replay_margin, load_replay_result_entries,
    load_replay_signal_diagnostics, reprice_replay_result, write_replay_result,
};
#[cfg(feature = "replay")]
pub(crate) use risk::ReplayMarginConfig;
pub(crate) use state::{ReplayState, replay_accounts, replay_contract, search_replay_contracts};
#[cfg(feature = "replay")]
#[allow(unused_imports)]
pub(crate) use sweep::{
    ReplaySweepChildSpec, ReplaySweepConstraint, ReplaySweepGuardrailReport, ReplaySweepGuardrails,
    ReplaySweepOutputFormat, ReplaySweepParameter, ReplaySweepPlan, ReplaySweepResourceEstimate,
    ReplaySweepSpec,
};
#[cfg(feature = "replay")]
pub(crate) use sweep_analytics::{
    ReplaySweepRankingDocument, ReplaySweepRankingEntry, ReplaySweepRankingLibrarySnapshot,
    ReplaySweepRankingMetric, ReplaySweepRankingOptions, ReplaySweepRankingRow,
    load_replay_sweep_ranking_entries, rank_replay_sweep,
};
#[cfg(feature = "replay")]
pub(crate) use sweep_runner::run_replay_sweep;
pub(crate) use worker::spawn_replay_market_task;

#[cfg(all(test, feature = "replay"))]
mod tests;
