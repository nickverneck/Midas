mod protection;
mod proxy;
mod tick;

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "trader-replay-proxy")]
#[command(about = "Loopback-only Tradovate protocol proxy backed by deterministic replay data")]
struct Args {
    /// JSONL, CSV, or server-bar Parquet fixture for the strategy chart.
    /// Rows are consumed in file order.
    #[arg(long)]
    bars: std::path::PathBuf,

    /// Optional raw-tick JSONL, CSV, or replay-cache Parquet fixture. When
    /// supplied, every raw tick drives synthetic broker quotes, entry fills,
    /// TP/SL, and trailing protection. Use bars derived from this same tape
    /// for the native strategy chart.
    #[arg(long)]
    ticks: Option<std::path::PathBuf>,

    /// Optional replay-cache manifest used to verify contract and tick specs.
    /// When omitted, the proxy searches fixture parent directories for
    /// manifest.json.
    #[arg(long)]
    fixture_manifest: Option<std::path::PathBuf>,

    /// Permit hand-authored fixtures that have no discoverable manifest.
    #[arg(long)]
    allow_unverified_fixture: bool,

    /// Require every raw tick to contain a valid bid and ask. Without this
    /// flag trade-only ticks use an explicitly marked trade-price fallback.
    #[arg(long)]
    require_quote_ticks: bool,

    /// Maximum raw ticks to load. Zero means all ticks.
    #[arg(long, default_value_t = 0)]
    max_ticks: usize,

    /// How timestamps in the chart fixture relate to the raw-tick tape.
    /// `start` processes ticks strictly before a chart timestamp; `close`
    /// processes ticks through the timestamp before publishing that chart
    /// update. Use `close` for range/tick/volume bars whose timestamp is the
    /// closing source tick.
    #[arg(long, default_value = "start")]
    raw_tick_bar_timestamps: String,

    /// Resolution for an ambiguous protection quote. `stop` is conservative;
    /// `target` is optimistic. The default is conservative.
    #[arg(long, default_value = "stop")]
    protection_precedence: String,

    /// Loopback REST listener. The trader config should use this address with /v1.
    #[arg(long, default_value = "127.0.0.1:18100")]
    rest_bind: std::net::SocketAddr,

    /// Loopback user/account WebSocket listener.
    #[arg(long, default_value = "127.0.0.1:18101")]
    user_ws_bind: std::net::SocketAddr,

    /// Loopback market-data WebSocket listener.
    #[arg(long, default_value = "127.0.0.1:18102")]
    market_ws_bind: std::net::SocketAddr,

    /// Number of initial bars returned by md/getChart as historical data.
    /// Remaining rows are emitted as the realtime stream.
    #[arg(long, default_value_t = 500)]
    history_bars: usize,

    /// Stop after this many rows. Zero means all rows.
    #[arg(long, default_value_t = 0)]
    max_bars: usize,

    /// Hold the realtime stream after the historical chart response until
    /// this local marker path exists. This makes commit comparisons start at
    /// the same first live bar after the candidate has been configured.
    #[arg(long)]
    start_file: Option<std::path::PathBuf>,

    /// Virtual replay speed. Zero is unpaced; one follows source timestamps;
    /// two is twice as fast, and so on.
    #[arg(long, default_value_t = 0.0)]
    speed: f64,

    /// Optional cap on one inter-bar sleep, useful for sparse fixtures.
    #[arg(long, default_value_t = 5_000)]
    max_sleep_ms: u64,

    /// Delay between loop cycles when the first bar has no timestamp gap
    /// from the previous cycle's last bar.
    #[arg(long, default_value_t = 1_000)]
    loop_boundary_delay_ms: u64,

    /// Delay every REST response by this many milliseconds.
    #[arg(long, default_value_t = 0)]
    rest_delay_ms: u64,

    /// Delay user-WebSocket request acknowledgements by this many milliseconds.
    #[arg(long, default_value_t = 0)]
    ack_delay_ms: u64,

    /// Delay broker fill/entity publication after an order acknowledgement.
    #[arg(long, default_value_t = 0)]
    fill_delay_ms: u64,

    /// Emit a JSONL protocol trace at this path.
    #[arg(long)]
    trace: Option<std::path::PathBuf>,

    /// Maximum trace bytes. Zero is rejected to keep accidental long-running
    /// replays from filling the disk.
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    trace_max_bytes: usize,

    /// Repeat the realtime portion after it reaches EOF.
    #[arg(long)]
    loop_replay: bool,

    /// Synthetic account id returned by the REST and user-sync façades.
    #[arg(long, default_value_t = 910001)]
    account_id: i64,

    /// Synthetic account name returned by the REST and user-sync façades.
    #[arg(long, default_value = "PROXY_REPLAY")]
    account_name: String,

    /// Synthetic contract symbol returned by the REST and market façades.
    #[arg(long, default_value = "GCZ6")]
    contract: String,

    /// Synthetic contract id used by chart requests and orders.
    #[arg(long)]
    contract_id: Option<i64>,

    /// Contract tick size used for metadata and PnL.
    #[arg(long, default_value_t = 0.1)]
    tick_size: f64,

    /// Contract currency value of one full point.
    #[arg(long, default_value_t = 100.0)]
    value_per_point: f64,

    /// Initial synthetic cash balance.
    #[arg(long, default_value_t = 100_000.0)]
    initial_balance: f64,

    /// Maximum number of simultaneously connected clients per listener.
    #[arg(long, default_value_t = 1)]
    max_clients: usize,

    /// Maximum number of retained synthetic order/fill entities. Older
    /// terminal entities are evicted once this bound is reached.
    #[arg(long, default_value_t = 100_000)]
    max_state_entities: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    proxy::run(args).await
}
