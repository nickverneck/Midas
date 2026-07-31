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
    /// Capture historical Level 2 snapshots through a Tradovate Market Replay session.
    #[command(name = "capture-replay-dom")]
    CaptureReplayDom(CaptureReplayDomArgs),
    /// Capture live Level 2 snapshots in a separate, opt-in process.
    #[command(name = "capture-live-dom")]
    CaptureLiveDom(CaptureLiveDomArgs),
}
