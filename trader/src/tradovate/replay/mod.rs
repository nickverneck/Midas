use super::*;

#[cfg(feature = "replay")]
mod bars;
#[cfg(feature = "replay")]
mod broker_schedule;
mod candle_accel;
#[cfg(feature = "replay")]
mod fees;
#[cfg(feature = "replay")]
mod instrument;
mod ledger;
#[cfg(feature = "replay")]
mod liquidation;
mod load;
#[cfg(feature = "replay")]
mod result_parquet;
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
mod sweep_parquet;
#[cfg(feature = "replay")]
mod sweep_performance;
#[cfg(feature = "replay")]
mod sweep_runner;
#[cfg(feature = "replay")]
mod ticks;
#[cfg(feature = "replay")]
pub(crate) mod virtual_time;
#[cfg(feature = "replay")]
mod walk_forward;
#[cfg(feature = "replay")]
mod walk_forward_eval;
mod worker;

#[cfg(feature = "replay")]
pub(crate) use broker_schedule::ReplayBrokerSchedule;
pub(crate) use candle_accel::{
    ReplayAcceleration, ReplayAccelerationDevice, ReplayAccelerationStatus, ReplayEmaBatchResult,
    ReplayEmaLastPair, ema_last_batch, probe_acceleration,
};
#[cfg(feature = "replay")]
pub(crate) use fees::ReplayFeeSchedule;
pub(crate) use ledger::ReplayExecutionLedgerState;
#[cfg(feature = "replay")]
pub(crate) use ledger::ReplayLedgerMarketContext;
#[cfg(feature = "replay")]
pub(crate) use liquidation::ReplayLiquidationConfig;
pub(crate) use load::load_replay_state;
#[cfg(feature = "replay")]
pub(crate) use results::{
    ReplayEquityPoint, ReplayFeeScenario, ReplayResultEntry, ReplayResultInput,
    ReplayResultLibrarySnapshot, ReplayResultStatus, ReplayTradeExcursion, analyze_replay_margin,
    load_replay_equity, load_replay_result_entries, load_replay_signal_diagnostics,
    reprice_replay_result, simulate_replay_liquidation_result, write_replay_result,
};
#[cfg(feature = "replay")]
pub(crate) use risk::ReplayMarginConfig;
pub(crate) use state::{ReplayState, replay_accounts, replay_contract, search_replay_contracts};
#[cfg(feature = "replay")]
#[allow(unused_imports)]
pub(crate) use sweep::{
    ReplaySweepChildSpec, ReplaySweepConstraint, ReplaySweepExecutionMode,
    ReplaySweepGuardrailReport, ReplaySweepGuardrails, ReplaySweepOutputFormat,
    ReplaySweepParameter, ReplaySweepPlan, ReplaySweepResourceEstimate, ReplaySweepSpec,
};
#[cfg(feature = "replay")]
pub(crate) use sweep_analytics::{
    ReplaySweepRankingDocument, ReplaySweepRankingEntry, ReplaySweepRankingLibrarySnapshot,
    ReplaySweepRankingMetric, ReplaySweepRankingOptions, ReplaySweepRankingRow,
    load_replay_sweep_ranking_entries, rank_replay_sweep,
};
#[cfg(feature = "replay")]
pub(crate) use sweep_performance::{
    ReplaySweepPerformanceOptions, ReplaySweepPerformanceReport, replay_sweep_performance_csv,
    replay_sweep_performance_json, run_replay_sweep_performance,
    write_replay_sweep_performance_report,
};
#[cfg(feature = "replay")]
pub(crate) use sweep_runner::{run_replay_sweep, run_replay_sweep_with_mode};
#[cfg(feature = "replay")]
pub(crate) use walk_forward::{
    ReplayWalkForwardOptions, ReplayWalkForwardPhase, ReplayWalkForwardWindow,
    plan_replay_walk_forward,
};
#[cfg(feature = "replay")]
pub(crate) use walk_forward_eval::{
    ReplayWalkForwardEvaluationDocument, ReplayWalkForwardEvaluationOptions,
    ReplayWalkForwardSelectionPolicy, evaluate_replay_walk_forward,
};
pub(crate) use worker::spawn_replay_market_task;

#[cfg(all(test, feature = "replay"))]
mod tests;
