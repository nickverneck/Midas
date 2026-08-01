use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "trader")]
#[command(about = "Ratatui terminal client for Tradovate and Ironbeam")]
pub(crate) struct Cli {
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,

    #[arg(long, default_value = ".run/trader-engine.sock")]
    pub(crate) engine_socket: PathBuf,

    #[arg(long)]
    pub(crate) no_spawn_engine: bool,

    #[command(subcommand)]
    pub(crate) mode: Option<Mode>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct SwipeProfileArgs {
    #[arg(long, default_value = "DEMO")]
    pub(crate) account_filter: String,
    #[arg(long, default_value = "ES")]
    pub(crate) contract_query: String,
    #[arg(long)]
    pub(crate) contract_exact: Option<String>,
    #[arg(long, value_delimiter = ',', default_values_t = [50_u64, 250, 500, 1_000, 5_000])]
    pub(crate) delays_ms: Vec<u64>,
    #[arg(long, default_value_t = 10)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 400.0)]
    pub(crate) take_profit_ticks: f64,
    #[arg(long, default_value_t = 400.0)]
    pub(crate) stop_loss_ticks: f64,
    #[arg(long, default_value_t = 1)]
    pub(crate) order_qty: i32,
    #[arg(long, default_value_t = 20_000)]
    pub(crate) settle_timeout_ms: u64,
    #[arg(long)]
    pub(crate) output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ReplayDownloadArgs {
    /// Instrument root symbol, for example MES or ES.
    #[arg(long)]
    pub(crate) instrument: String,
    /// Exact contract symbol, for example MESU6.
    #[arg(long)]
    pub(crate) contract: String,
    /// Inclusive start date in YYYY-MM-DD.
    #[arg(long)]
    pub(crate) start: String,
    /// Inclusive end date in YYYY-MM-DD.
    #[arg(long)]
    pub(crate) end: String,
    /// Data source kind: server-bars or raw-ticks.
    #[arg(long, default_value = "server-bars")]
    pub(crate) source_kind: String,
    /// Server-bar kind: minute, second, tick, volume, or range.
    #[arg(long, default_value = "minute")]
    pub(crate) bar_kind: String,
    /// Server-bar value, such as 1 for 1 minute or 6500 for volume.
    #[arg(long, default_value_t = 1)]
    pub(crate) bar_value: u32,
    /// Chart mode: ohlc or heikin-ashi.
    #[arg(long, default_value = "ohlc")]
    pub(crate) chart_mode: String,
    /// Optional cache root. Defaults to replay_cache_dir / TRADER_DATA_CACHE_DIR.
    #[arg(long)]
    pub(crate) cache_dir: Option<PathBuf>,
    /// Optional user-facing dataset name stored in the cache manifest.
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Optional dataset tag. Repeat --tag to store more than one.
    #[arg(long = "tag")]
    pub(crate) tags: Vec<String>,
    /// Maximum initial raw-tick request chunk size in minutes.
    #[arg(long, default_value_t = 60)]
    pub(crate) raw_chunk_minutes: u32,
    /// Smallest raw-tick chunk produced after an incomplete request.
    #[arg(long, default_value_t = 5)]
    pub(crate) raw_minimum_split_minutes: u32,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct RepriceReplayResultArgs {
    /// Existing result.json to reprice without replaying market data.
    #[arg(long)]
    pub(crate) result: PathBuf,
    /// Fee schedule name stored as the active scenario.
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Imported broker schedule JSON. When set, fee components/name come from this file.
    #[arg(long)]
    pub(crate) schedule_file: Option<PathBuf>,
    /// Currency label for the fee components.
    #[arg(long, default_value = "USD")]
    pub(crate) currency: String,
    /// Broker commission per contract per side.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) commission_per_contract: f64,
    /// Exchange fee per contract per side.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) exchange_per_contract: f64,
    /// Clearing fee per contract per side.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) clearing_per_contract: f64,
    /// Regulatory/NFA fee per contract per side.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) regulatory_per_contract: f64,
    /// Miscellaneous fee or rebate per contract per side.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) misc_per_contract: f64,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ImportReplayBrokerScheduleArgs {
    /// Replay cache manifest or raw broker metadata JSON.
    #[arg(long)]
    pub(crate) metadata: PathBuf,
    /// Output path for the normalized replay broker schedule JSON.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Scenario name stored in the imported fee schedule.
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Currency label for imported fee components.
    #[arg(long, default_value = "USD")]
    pub(crate) currency: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct AnalyzeReplayMarginArgs {
    /// Existing result.json to analyze without replaying market data.
    #[arg(long)]
    pub(crate) result: PathBuf,
    /// Imported broker schedule JSON. When set, its margin requirement is used.
    #[arg(long)]
    pub(crate) schedule_file: Option<PathBuf>,
    /// Margin model label stored in the result metadata.
    #[arg(long, default_value = "fixed_per_contract")]
    pub(crate) model: String,
    /// Currency label for the margin and account-size assumptions.
    #[arg(long, default_value = "USD")]
    pub(crate) currency: String,
    /// Margin requirement per open contract.
    #[arg(long)]
    pub(crate) margin_per_contract: Option<f64>,
    /// Fixed safety buffer added above the margin requirement.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) safety_buffer: f64,
    /// Percentage safety buffer applied to the margin requirement.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) safety_buffer_percent: f64,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct SimulateReplayLiquidationArgs {
    /// Existing completed result.json to analyze without replaying market data.
    #[arg(long)]
    pub(crate) result: PathBuf,
    /// Imported broker schedule JSON. When set, its margin requirement is used.
    #[arg(long)]
    pub(crate) schedule_file: Option<PathBuf>,
    /// Margin requirement per open contract.
    #[arg(long)]
    pub(crate) margin_per_contract: Option<f64>,
    /// Fixed safety buffer above the margin requirement.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) safety_buffer: f64,
    /// Percentage safety buffer above the margin requirement.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) safety_buffer_percent: f64,
    /// Slippage in points applied to the simulated liquidation fill.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) slippage_points: f64,
    /// Optional saved fee scenario to use instead of the active scenario.
    #[arg(long)]
    pub(crate) fee_scenario: Option<String>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ValidateReplaySweepArgs {
    /// JSON sweep specification to validate and expand before execution.
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Optional JSON path for the fully expanded, rerunnable child plan.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct RunReplaySweepArgs {
    /// JSON sweep specification to execute against the cached dataset view.
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Re-run child IDs even when a completed result.json already exists.
    #[arg(long)]
    pub(crate) no_resume: bool,
    /// Confirm that a large sweep may start after reviewing its estimate.
    #[arg(long)]
    pub(crate) allow_large: bool,
    /// Explicitly bypass hard resource guardrails for this launch.
    #[arg(long)]
    pub(crate) override_guardrails: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct RankReplaySweepArgs {
    /// Existing sweep-summary.json produced by run-replay-sweep.
    #[arg(long)]
    pub(crate) summary: PathBuf,
    /// Optional sweep-plan.json used to recover parameter values from older results.
    #[arg(long)]
    pub(crate) plan: Option<PathBuf>,
    /// Ranking metric; defaults to robustness rather than raw PnL.
    #[arg(long, default_value = "robustness")]
    pub(crate) metric: String,
    /// Fee scenario name, or `active` (the default) for the saved active scenario.
    #[arg(long)]
    pub(crate) fee_scenario: Option<String>,
    /// Maximum number of rows to print/write.
    #[arg(long, default_value_t = 20)]
    pub(crate) limit: usize,
    /// Exclude candidates with fewer closed trades than this value.
    #[arg(long, default_value_t = 0)]
    pub(crate) min_closed_trades: usize,
    /// Exclude candidates whose drawdown percentage exceeds this value.
    #[arg(long)]
    pub(crate) max_drawdown_pct: Option<f64>,
    /// Optional JSON output path for the ranking document.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Optional CSV output path for the ranking rows.
    #[arg(long)]
    pub(crate) csv: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PlanReplayWalkForwardArgs {
    /// JSON replay sweep specification to split into chronological phases.
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Output path for the persisted walk-forward plan.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Fraction of the base evaluation range assigned to training.
    #[arg(long, default_value_t = 0.70)]
    pub(crate) train_fraction: f64,
    /// Fraction of the base evaluation range assigned to validation.
    #[arg(long, default_value_t = 0.15)]
    pub(crate) validation_fraction: f64,
    /// Fraction of the base evaluation range assigned to out-of-sample testing.
    #[arg(long, default_value_t = 0.15)]
    pub(crate) test_fraction: f64,
    /// Maximum number of rolling chronological folds to materialize.
    #[arg(long, default_value_t = 1)]
    pub(crate) folds: usize,
    /// Optional fraction of the base range by which each next fold advances.
    /// Defaults to the test fraction.
    #[arg(long)]
    pub(crate) step_fraction: Option<f64>,
    /// Gap in seconds inserted between train/validation and validation/test.
    #[arg(long, default_value_t = 0)]
    pub(crate) purge_seconds: u64,
    /// Indicator warmup seconds for generated views. Defaults to the source view policy.
    #[arg(long)]
    pub(crate) warmup_seconds: Option<u64>,
    /// Root directory for generated phase sweep specifications.
    #[arg(long)]
    pub(crate) output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct EvaluateReplayWalkForwardArgs {
    /// Persisted walk-forward plan whose phase sweep artifacts should be evaluated.
    #[arg(long)]
    pub(crate) plan: PathBuf,
    /// Ranking metric used to select a candidate from train/validation results.
    #[arg(long, default_value = "robustness")]
    pub(crate) metric: String,
    /// Saved fee-scenario name, or `active` (the default) for each result's active scenario.
    #[arg(long)]
    pub(crate) fee_scenario: Option<String>,
    /// Selection policy: `train` or `train_then_validation`.
    #[arg(long, default_value = "train")]
    pub(crate) selection_policy: String,
    /// Optional JSON output path for the evaluation document.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Optional CSV output path for candidate/phase rows.
    #[arg(long)]
    pub(crate) csv: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ProfileReplaySweepArgs {
    /// JSON replay sweep specification to execute as a bounded performance probe.
    #[arg(long)]
    pub(crate) spec: PathBuf,
    /// Run at most this many deterministic prefix combinations.
    #[arg(long)]
    pub(crate) sample_runs: Option<usize>,
    /// Explicit child-output directory. Omit to use a fresh isolated directory.
    #[arg(long)]
    pub(crate) output_dir: Option<PathBuf>,
    /// Reuse completed child artifacts when --output-dir points at an existing tree.
    #[arg(long)]
    pub(crate) resume: bool,
    /// Confirm that a large probe may start after reviewing its estimate.
    #[arg(long)]
    pub(crate) allow_large: bool,
    /// Explicitly bypass hard resource guardrails for this launch.
    #[arg(long)]
    pub(crate) override_guardrails: bool,
    /// Optional JSON output path for the empirical report.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
    /// Optional CSV output path for the empirical child rows.
    #[arg(long)]
    pub(crate) csv: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct CaptureReplayDomArgs {
    /// Exact contract symbol, for example MESU6 or GCZ6.
    #[arg(long)]
    pub(crate) contract: String,
    /// Inclusive UTC start timestamp in RFC3339 form.
    #[arg(long)]
    pub(crate) start: String,
    /// Exclusive UTC end timestamp in RFC3339 form.
    #[arg(long)]
    pub(crate) end: String,
    /// JSONL output path for full-book snapshots.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Market Replay speed percentage (0-400).
    #[arg(long, default_value_t = 400)]
    pub(crate) speed: u16,
    /// Initial balance used to initialize the disposable Market Replay account.
    #[arg(long, default_value_t = 50_000.0)]
    pub(crate) initial_balance: f64,
    /// Permit replacing an existing output file.
    #[arg(long)]
    pub(crate) overwrite: bool,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct CaptureLiveDomArgs {
    /// Exact contract symbol, for example MESU6 or GCZ6.
    #[arg(long)]
    pub(crate) contract: String,
    /// Capture duration in seconds.
    #[arg(long, default_value_t = 300)]
    pub(crate) duration_seconds: u64,
    /// JSONL output path for full-book snapshots.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Permit replacing an existing output file.
    #[arg(long)]
    pub(crate) overwrite: bool,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Mode {
    /// Run the background engine server.
    Engine,
    /// List running engine processes on this host.
    List,
    /// Attach a full TUI session to a running engine by ID from `list`.
    Attach {
        /// Engine ID from `trader list` (PID).
        id: u32,
    },
    /// Kill one running engine by ID.
    Kill {
        /// Engine ID from `trader list` (PID).
        id: u32,
        /// Disarm, manually close the selected market, then kill the engine. Requires --features manual-orders.
        #[arg(short = 'c', long = "close")]
        close: bool,
    },
    /// Kill all running engines.
    #[command(name = "killall")]
    KillAll {
        /// Disarm, manually close each selected market, then kill engines. Requires --features manual-orders.
        #[arg(short = 'c', long = "close")]
        close: bool,
    },
    /// Run a non-TUI Tradovate sim swipe profiler for reversal paths.
    #[command(name = "swipe-profile")]
    SwipeProfile(SwipeProfileArgs),
    /// Download replay data into the local replay cache.
    #[command(name = "download-replay-data")]
    DownloadReplayData(ReplayDownloadArgs),
    /// Reprice a saved replay result with an accounting-only fee schedule.
    #[command(name = "reprice-replay-result")]
    RepriceReplayResult(RepriceReplayResultArgs),
    /// Analyze required starting capital for a saved replay result.
    #[command(name = "analyze-replay-margin")]
    AnalyzeReplayMargin(AnalyzeReplayMarginArgs),
    /// Normalize broker fee/margin metadata into a replay schedule JSON.
    #[command(name = "import-replay-broker-schedule")]
    ImportReplayBrokerSchedule(ImportReplayBrokerScheduleArgs),
    /// Simulate optional margin liquidation on a saved replay result.
    #[command(name = "simulate-replay-liquidation")]
    SimulateReplayLiquidation(SimulateReplayLiquidationArgs),
    /// Validate and expand a replay parameter sweep without executing it.
    #[command(name = "validate-replay-sweep")]
    ValidateReplaySweep(ValidateReplaySweepArgs),
    /// Execute a replay parameter sweep without starting the TUI.
    #[command(name = "run-replay-sweep")]
    RunReplaySweep(RunReplaySweepArgs),
    /// Rank and inspect completed replay sweep results without replaying data.
    #[command(name = "rank-replay-sweep")]
    RankReplaySweep(RankReplaySweepArgs),
    /// Persist bounded train/validation/test sweep specifications for walk-forward evaluation.
    #[command(name = "plan-replay-walk-forward")]
    PlanReplayWalkForward(PlanReplayWalkForwardArgs),
    /// Evaluate saved walk-forward phase results without replaying market data.
    #[command(name = "evaluate-replay-walk-forward")]
    EvaluateReplayWalkForward(EvaluateReplayWalkForwardArgs),
    /// Run a bounded empirical performance probe for a replay sweep.
    #[command(name = "profile-replay-sweep")]
    ProfileReplaySweep(ProfileReplaySweepArgs),
    /// Capture historical Level 2 snapshots through a Tradovate Market Replay session.
    #[command(name = "capture-replay-dom")]
    CaptureReplayDom(CaptureReplayDomArgs),
    /// Capture live Level 2 snapshots in a separate, opt-in process.
    #[command(name = "capture-live-dom")]
    CaptureLiveDom(CaptureLiveDomArgs),
}
