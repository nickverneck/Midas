mod app;
mod broker;
mod config;
mod engine_control;
mod engine_registry;
mod engine_runtime;
mod ipc;
#[cfg(feature = "ironbeam")]
mod ironbeam;
#[cfg(feature = "replay")]
mod replay_cache;
mod strategies;
mod strategy;
mod strategy_debug;
#[cfg(feature = "tradovate")]
mod tradovate;

#[cfg(feature = "replay")]
use anyhow::Context;
use anyhow::{Result, bail};
use app::{App, EngineKey};
use broker::{ServiceCommand, ServiceEvent};
use clap::{Args, Parser, Subcommand};
use config::AppConfig;
use crossterm::event::{Event as CEvent, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use engine_control::{close_and_kill_engine, kill_engine_process};
use engine_registry::{list_running_engines, resolve_engine};
use engine_runtime::{EngineSession, unique_engine_socket_path};
use futures_util::StreamExt;
use ipc::run_engine_server;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug, Parser)]
#[command(name = "trader")]
#[command(about = "Ratatui terminal client for Tradovate and Ironbeam")]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long, default_value = ".run/trader-engine.sock")]
    engine_socket: PathBuf,

    #[arg(long)]
    no_spawn_engine: bool,

    #[command(subcommand)]
    mode: Option<Mode>,
}

#[derive(Debug, Clone, Args)]
struct SwipeProfileArgs {
    #[arg(long, default_value = "DEMO")]
    account_filter: String,

    #[arg(long, default_value = "ES")]
    contract_query: String,

    #[arg(long)]
    contract_exact: Option<String>,

    #[arg(long, value_delimiter = ',', default_values_t = [50_u64, 250, 500, 1_000, 5_000])]
    delays_ms: Vec<u64>,

    #[arg(long, default_value_t = 10)]
    iterations: usize,

    #[arg(long, default_value_t = 400.0)]
    take_profit_ticks: f64,

    #[arg(long, default_value_t = 400.0)]
    stop_loss_ticks: f64,

    #[arg(long, default_value_t = 1)]
    order_qty: i32,

    #[arg(long, default_value_t = 20_000)]
    settle_timeout_ms: u64,

    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
struct ReplayDownloadArgs {
    /// Instrument root symbol, for example MES or ES.
    #[arg(long)]
    instrument: String,

    /// Exact contract symbol, for example MESU6.
    #[arg(long)]
    contract: String,

    /// Inclusive start date in YYYY-MM-DD.
    #[arg(long)]
    start: String,

    /// Inclusive end date in YYYY-MM-DD.
    #[arg(long)]
    end: String,

    /// Data source kind: server-bars or raw-ticks.
    #[arg(long, default_value = "server-bars")]
    source_kind: String,

    /// Server-bar kind: minute, second, tick, volume, or range.
    #[arg(long, default_value = "minute")]
    bar_kind: String,

    /// Server-bar value, such as 1 for 1 minute or 6500 for volume.
    #[arg(long, default_value_t = 1)]
    bar_value: u32,

    /// Chart mode: ohlc or heikin-ashi.
    #[arg(long, default_value = "ohlc")]
    chart_mode: String,

    /// Optional cache root. Defaults to replay_cache_dir / TRADER_DATA_CACHE_DIR.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Subcommand)]
enum Mode {
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
    #[cfg(feature = "tradovate")]
    if let Some(Mode::SwipeProfile(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::run_swipe_profile(
            config,
            tradovate::SwipeProfileOptions {
                account_filter: args.account_filter,
                contract_query: args.contract_query,
                contract_exact: args.contract_exact,
                delays_ms: args.delays_ms,
                iterations_per_delay: args.iterations,
                take_profit_ticks: args.take_profit_ticks,
                stop_loss_ticks: args.stop_loss_ticks,
                order_qty: args.order_qty,
                settle_timeout_ms: args.settle_timeout_ms,
                output_dir: args.output_dir,
            },
        )
        .await;
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::SwipeProfile(_))) {
        bail!("tradovate support is not enabled in this build");
    }

    if let Some(Mode::DownloadReplayData(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return download_replay_data(&config, args).await;
    }

    if matches!(cli.mode, Some(Mode::Engine)) {
        return run_engine_server(&cli.engine_socket).await;
    }
    if matches!(cli.mode, Some(Mode::List)) {
        return list_engines();
    }
    if let Some(Mode::Kill { id, close }) = cli.mode {
        return kill_engine(id, close).await;
    }
    if let Some(Mode::KillAll { close }) = cli.mode {
        return kill_all_engines_cmd(close).await;
    }

    let attach_mode = if let Some(Mode::Attach { id }) = cli.mode {
        configure_attach_mode(&mut cli, id)?;
        true
    } else {
        false
    };

    let mut config = AppConfig::load(cli.config.as_deref())?;
    if attach_mode {
        config.autoconnect = false;
    }

    run_tui(&cli, config, attach_mode).await
}

async fn download_replay_data(config: &AppConfig, args: ReplayDownloadArgs) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (config, args);
        bail!("replay downloader requires `--features replay`");
    }

    #[cfg(feature = "replay")]
    {
        let plan = build_replay_download_plan(config, args)?;
        if config.broker != broker::BrokerKind::Tradovate {
            bail!("replay data downloads currently support Tradovate only");
        }

        #[cfg(not(feature = "tradovate"))]
        {
            let _ = plan;
            bail!("Tradovate server-bar replay downloads require `--features tradovate,replay`");
        }

        #[cfg(feature = "tradovate")]
        {
            match plan.source_kind {
                replay_cache::ReplayCacheSourceKind::ServerBars => {
                    let download = tradovate::download_replay_server_bars(
                        config,
                        tradovate::TradovateServerBarDownloadRequest {
                            contract: plan.contract.clone(),
                            start: plan.start,
                            end: plan.end,
                            bar_type: plan.bar_type,
                        },
                    )
                    .await?;
                    let outcome = replay_cache::write_server_bars_jsonl_cache(
                        replay_cache::ReplayCacheServerBarsWrite {
                            cache_root: plan.cache_root.clone(),
                            provider: config.broker,
                            env: config.env,
                            instrument: replay_cache::ReplayCacheInstrument {
                                symbol: plan.instrument.clone(),
                                name: None,
                                exchange: None,
                            },
                            contract: replay_cache::ReplayCacheContract {
                                symbol: download.contract.name.clone(),
                                id: Some(download.contract.id),
                                expiration: None,
                            },
                            request_start: plan.start,
                            request_end: plan.end,
                            source_kind: replay_cache::ReplayCacheSourceKind::ServerBars,
                            download_request: download.request_body,
                            bar_type: plan.bar_type,
                            tick_specs: download.tick_specs,
                            session_template: download.session_template,
                            bars: download.bars,
                            warnings: download.warnings,
                            notes: Some(
                                "Downloaded through Tradovate md/getChart only; no user sync or order path was started."
                                    .to_string(),
                            ),
                        },
                    )?;

                    println!("Replay server-bar download complete.");
                    println!("No user sync, account stream, or order path was started.");
                    println!("Provider: {}", config.broker.label());
                    println!("Environment: {}", config.env.label());
                    println!("Instrument: {}", plan.instrument);
                    println!("Contract: {}", plan.contract);
                    println!("Date range: {} to {}", plan.start_date, plan.end_date);
                    println!("Source kind: {}", plan.source_kind.label());
                    println!(
                        "Requested shape: {}",
                        plan.bar_type.mode_label(plan.chart_mode)
                    );
                    println!("Rows: {}", outcome.row_count);
                    println!("Data: {}", outcome.data_path.display());
                    println!("Manifest: {}", outcome.manifest_path.display());
                    Ok(())
                }
                replay_cache::ReplayCacheSourceKind::RawTicks => {
                    let download = tradovate::download_replay_raw_ticks(
                        config,
                        tradovate::TradovateRawTickDownloadRequest {
                            contract: plan.contract.clone(),
                            start: plan.start,
                            end: plan.end,
                        },
                    )
                    .await?;
                    let outcome = replay_cache::write_raw_ticks_parquet_cache(
                        replay_cache::ReplayCacheRawTicksWrite {
                            cache_root: plan.cache_root.clone(),
                            provider: config.broker,
                            env: config.env,
                            instrument: replay_cache::ReplayCacheInstrument {
                                symbol: plan.instrument.clone(),
                                name: None,
                                exchange: None,
                            },
                            contract: replay_cache::ReplayCacheContract {
                                symbol: download.contract.name.clone(),
                                id: Some(download.contract.id),
                                expiration: None,
                            },
                            request_start: plan.start,
                            request_end: plan.end,
                            download_request: download.request_body,
                            tick_specs: download.tick_specs,
                            session_template: download.session_template,
                            ticks: download.ticks,
                            warnings: download.warnings,
                            notes: Some(
                                "Downloaded raw ticks through Tradovate md/getChart only; no user sync or order path was started."
                                    .to_string(),
                            ),
                        },
                    )?;

                    println!("Replay raw tick download complete.");
                    println!("No user sync, account stream, or order path was started.");
                    println!("Provider: {}", config.broker.label());
                    println!("Environment: {}", config.env.label());
                    println!("Instrument: {}", plan.instrument);
                    println!("Contract: {}", plan.contract);
                    println!("Date range: {} to {}", plan.start_date, plan.end_date);
                    println!("Source kind: {}", plan.source_kind.label());
                    println!("Storage: parquet ({})", "snappy");
                    println!("Rows: {}", outcome.row_count);
                    println!("Data: {}", outcome.data_path.display());
                    println!("Manifest: {}", outcome.manifest_path.display());
                    Ok(())
                }
                other => bail!("unsupported replay download source kind: {}", other.label()),
            }
        }
    }
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayDownloadPlan {
    instrument: String,
    contract: String,
    start_date: chrono::NaiveDate,
    end_date: chrono::NaiveDate,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    source_kind: replay_cache::ReplayCacheSourceKind,
    bar_type: broker::BarType,
    chart_mode: broker::CandleMode,
    cache_root: PathBuf,
}

#[cfg(feature = "replay")]
fn build_replay_download_plan(
    config: &AppConfig,
    args: ReplayDownloadArgs,
) -> Result<ReplayDownloadPlan> {
    let instrument = args.instrument.trim();
    let contract = args.contract.trim();
    if instrument.is_empty() {
        bail!("--instrument cannot be empty");
    }
    if contract.is_empty() {
        bail!("--contract cannot be empty");
    }
    if args.bar_value == 0 {
        bail!("--bar-value must be > 0");
    }

    let start_date = chrono::NaiveDate::parse_from_str(&args.start, "%Y-%m-%d")
        .with_context(|| format!("parse --start {}", args.start))?;
    let end_date = chrono::NaiveDate::parse_from_str(&args.end, "%Y-%m-%d")
        .with_context(|| format!("parse --end {}", args.end))?;
    if end_date < start_date {
        bail!("--end must be on or after --start");
    }

    let source_kind = parse_replay_download_source_kind(&args.source_kind)?;
    let bar_type = parse_replay_download_bar_type(&args.bar_kind, args.bar_value)?;
    let chart_mode = parse_replay_download_chart_mode(&args.chart_mode)?;
    let cache_root = args
        .cache_dir
        .unwrap_or_else(|| config.replay_cache_dir.clone());
    let start = start_date
        .and_hms_opt(0, 0, 0)
        .context("build replay download start timestamp")?
        .and_utc();
    let end = end_date
        .succ_opt()
        .context("build replay download exclusive end date")?
        .and_hms_opt(0, 0, 0)
        .context("build replay download end timestamp")?
        .and_utc();
    Ok(ReplayDownloadPlan {
        instrument: instrument.to_string(),
        contract: contract.to_string(),
        start_date,
        end_date,
        start,
        end,
        source_kind,
        bar_type,
        chart_mode,
        cache_root,
    })
}

#[cfg(feature = "replay")]
fn parse_replay_download_source_kind(raw: &str) -> Result<replay_cache::ReplayCacheSourceKind> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "server-bars" | "bars" => Ok(replay_cache::ReplayCacheSourceKind::ServerBars),
        "raw-ticks" | "ticks" | "tick" => Ok(replay_cache::ReplayCacheSourceKind::RawTicks),
        other => bail!("invalid --source-kind `{other}`; use server-bars or raw-ticks"),
    }
}

#[cfg(feature = "replay")]
fn parse_replay_download_bar_type(raw: &str, value: u32) -> Result<broker::BarType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "minute" | "min" | "m" => Ok(broker::BarType::minute(value)),
        "second" | "sec" | "s" => Ok(broker::BarType::second(value)),
        "tick" | "tick-count" | "ticks" => Ok(broker::BarType::tick(value)),
        "volume" | "vol" => Ok(broker::BarType::volume(value)),
        "range" => Ok(broker::BarType::range(value)),
        other => bail!("invalid --bar-kind `{other}`"),
    }
}

#[cfg(feature = "replay")]
fn parse_replay_download_chart_mode(raw: &str) -> Result<broker::CandleMode> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "ohlc" | "standard" | "regular" => Ok(broker::CandleMode::Standard),
        "heikin-ashi" | "heikin" | "heiken-ashi" | "heiken" => Ok(broker::CandleMode::HeikinAshi),
        other => bail!("invalid --chart-mode `{other}`; use ohlc or heikin-ashi"),
    }
}

fn list_engines() -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    println!("ID\tSTATUS\tSOCKET\tCWD");
    for engine in engines {
        let status = if engine.socket_is_live {
            "live"
        } else {
            "stale"
        };
        println!(
            "{}\t{}\t{}\t{}",
            engine.id,
            status,
            engine.socket_path.display(),
            engine.cwd.display()
        );
    }
    Ok(())
}

async fn kill_engine(id: u32, close: bool) -> Result<()> {
    if close {
        close_and_kill_engine(id).await?;
        println!("Closed the selected market and killed engine {id}.");
    } else {
        kill_engine_process(id).await?;
        println!("Killed engine {id}.");
    }
    Ok(())
}

async fn kill_all_engines_cmd(close: bool) -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    let mut failures = Vec::new();
    for engine in engines {
        let result = if close {
            close_and_kill_engine(engine.id).await
        } else {
            kill_engine_process(engine.id).await
        };
        match result {
            Ok(()) => {
                if close {
                    println!(
                        "Closed the selected market and killed engine {}.",
                        engine.id
                    );
                } else {
                    println!("Killed engine {}.", engine.id);
                }
            }
            Err(err) => failures.push(format!("{}: {err}", engine.id)),
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        bail!("failed to stop some engines: {}", failures.join("; "))
    }
}

fn configure_attach_mode(cli: &mut Cli, id: u32) -> Result<()> {
    let engine = resolve_engine(id)?;
    if !engine.socket_is_live {
        bail!(
            "engine {} is running but its socket {} is unavailable",
            engine.id,
            engine.socket_path.display()
        );
    }

    cli.engine_socket = engine.socket_path;
    cli.no_spawn_engine = true;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineEntryMode {
    AttachExisting,
    CreateNew,
}

fn should_autoconnect_engine_session(
    config: &AppConfig,
    awaiting_broker_selection: bool,
    mode: EngineEntryMode,
) -> bool {
    config.autoconnect && mode == EngineEntryMode::CreateNew && !awaiting_broker_selection
}

async fn run_tui(cli: &Cli, config: AppConfig, attach_mode: bool) -> Result<()> {
    let running_engines = list_running_engines()?;
    let direct_session = if attach_mode {
        Some(connect_or_spawn_engine(cli).await?)
    } else {
        None
    };

    let mut terminal = init_terminal()?;
    let mut app = App::new(config.clone());
    let startup_engines = running_engines
        .iter()
        .map(|engine| {
            (
                EngineKey::from_socket_path(&engine.socket_path),
                engine.socket_path.clone(),
                engine.socket_is_live,
            )
        })
        .collect::<Vec<_>>();
    app.set_running_engines(running_engines);
    app.set_engine_creation_enabled(!cli.no_spawn_engine);
    let mut event_stream = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(125));
    let (dummy_cmd_tx, _dummy_cmd_rx) = mpsc::unbounded_channel();
    let (engine_event_tx, mut engine_event_rx) = mpsc::unbounded_channel();
    let mut engine_sessions = HashMap::<EngineKey, ObservedEngineSession>::new();
    let mut active_engine_key = None::<EngineKey>;
    tick.tick().await;

    if let Some(session) = direct_session {
        let engine_key = EngineKey::from_socket_path(&cli.engine_socket);
        insert_observed_engine_session(
            &mut engine_sessions,
            engine_key.clone(),
            session,
            &engine_event_tx,
        );
        enter_engine_session(
            &mut app,
            &engine_sessions,
            &mut active_engine_key,
            engine_key,
            cli.engine_socket.clone(),
            &config,
            EngineEntryMode::AttachExisting,
        );
    }

    observe_running_engine_sessions(
        startup_engines,
        &mut app,
        &mut engine_sessions,
        &engine_event_tx,
        &dummy_cmd_tx,
    )
    .await;

    loop {
        terminal.draw(|frame| app.draw(frame))?;

        tokio::select! {
            _ = tick.tick() => {}
            maybe_event = event_stream.next() => {
                match maybe_event {
                    Some(Ok(CEvent::Key(key))) => {
                        {
                            let active_cmd_tx = active_engine_key
                                .as_ref()
                                .and_then(|engine_key| engine_sessions.get(engine_key))
                                .map(|session| &session.cmd_tx)
                                .unwrap_or(&dummy_cmd_tx);
                            app.handle_key(key, active_cmd_tx);
                        }
                        if let Some(action) = app.take_engine_selection_action() {
                            match action {
                                app::EngineSelectionAction::Attach { .. }
                                | app::EngineSelectionAction::CreateNew => {
                                    match connect_selected_engine(
                                        cli,
                                        action,
                                        &engine_event_tx,
                                        &mut engine_sessions,
                                    )
                                    .await
                                    {
                                        Ok((engine_key, socket_path, mode)) => {
                                            enter_engine_session(
                                                &mut app,
                                                &engine_sessions,
                                                &mut active_engine_key,
                                                engine_key,
                                                socket_path,
                                                &config,
                                                mode,
                                            );
                                        }
                                        Err(err) => {
                                            let active_cmd_tx = active_engine_key
                                                .as_ref()
                                                .and_then(|engine_key| engine_sessions.get(engine_key))
                                                .map(|session| &session.cmd_tx)
                                                .unwrap_or(&dummy_cmd_tx);
                                            app.handle_service_event(
                                                ServiceEvent::Error(format!("Engine selection failed: {err}")),
                                                active_cmd_tx,
                                            );
                                        }
                                    }
                                }
                                app::EngineSelectionAction::Refresh => {
                                    if let Err(err) = refresh_engine_overview(
                                        &mut app,
                                        &mut engine_sessions,
                                        &engine_event_tx,
                                        &dummy_cmd_tx,
                                        true,
                                    )
                                    .await
                                    {
                                        let active_cmd_tx = active_engine_key
                                            .as_ref()
                                            .and_then(|engine_key| engine_sessions.get(engine_key))
                                            .map(|session| &session.cmd_tx)
                                            .unwrap_or(&dummy_cmd_tx);
                                        app.handle_service_event(
                                            ServiceEvent::Error(format!("Engine refresh failed: {err}")),
                                            active_cmd_tx,
                                        );
                                    }
                                }
                                app::EngineSelectionAction::Kill { id } => {
                                    spawn_engine_lifecycle_action(
                                        app::EngineLifecycleAction::Kill,
                                        id,
                                        &engine_event_tx,
                                    );
                                }
                                app::EngineSelectionAction::CloseAndKill { id } => {
                                    spawn_engine_lifecycle_action(
                                        app::EngineLifecycleAction::CloseAndKill,
                                        id,
                                        &engine_event_tx,
                                    );
                                }
                            }
                        }
                    }
                    Some(Ok(CEvent::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        let active_cmd_tx = active_engine_key
                            .as_ref()
                            .and_then(|engine_key| engine_sessions.get(engine_key))
                            .map(|session| &session.cmd_tx)
                            .unwrap_or(&dummy_cmd_tx);
                        app.handle_service_event(ServiceEvent::Error(err.to_string()), active_cmd_tx);
                    }
                    None => break,
                }
            }
            maybe_service = engine_event_rx.recv() => {
                let Some(message) = maybe_service else {
                    break;
                };
                match message {
                    EngineRelayMessage::Event(envelope) => {
                        let is_active_detail =
                            active_engine_key.as_ref() == Some(&envelope.engine_key);
                        let active_cmd_tx = active_engine_key
                            .as_ref()
                            .and_then(|engine_key| engine_sessions.get(engine_key))
                            .map(|session| &session.cmd_tx)
                            .unwrap_or(&dummy_cmd_tx);
                        app.handle_engine_service_event(
                            envelope.engine_key,
                            envelope.event,
                            is_active_detail,
                            active_cmd_tx,
                        );
                    }
                    EngineRelayMessage::Closed { engine_key } => {
                        let is_active_detail = active_engine_key.as_ref() == Some(&engine_key);
                        app.handle_engine_receiver_closed(
                            &engine_key,
                            is_active_detail,
                        );
                        engine_sessions.remove(&engine_key);
                        if is_active_detail {
                            active_engine_key = None;
                        }
                    }
                    EngineRelayMessage::LifecycleCompleted { action, id, result } => {
                        let active_cmd_tx = active_engine_key
                            .as_ref()
                            .and_then(|engine_key| engine_sessions.get(engine_key))
                            .map(|session| session.cmd_tx.clone())
                            .unwrap_or_else(|| dummy_cmd_tx.clone());
                        match result {
                            Ok(()) => {
                                app.handle_service_event(
                                    ServiceEvent::Status(engine_lifecycle_success_message(action, id)),
                                    &active_cmd_tx,
                                );
                                if let Err(err) = refresh_engine_overview(
                                    &mut app,
                                    &mut engine_sessions,
                                    &engine_event_tx,
                                    &dummy_cmd_tx,
                                    false,
                                )
                                .await
                                {
                                    app.handle_service_event(
                                        ServiceEvent::Error(format!("Engine refresh failed: {err}")),
                                        &active_cmd_tx,
                                    );
                                }
                            }
                            Err(err) => {
                                app.handle_service_event(
                                    ServiceEvent::Error(format!(
                                        "{} for engine {id} failed: {err}",
                                            engine_lifecycle_failure_label(action)
                                    )),
                                    &active_cmd_tx,
                                );
                            }
                        }
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}

async fn connect_or_spawn_engine(cli: &Cli) -> Result<EngineSession> {
    engine_runtime::connect_or_spawn_engine(
        cli.config.as_deref(),
        &cli.engine_socket,
        cli.no_spawn_engine,
    )
    .await
}

async fn connect_selected_engine(
    cli: &Cli,
    action: app::EngineSelectionAction,
    engine_event_tx: &mpsc::UnboundedSender<EngineRelayMessage>,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
) -> Result<(EngineKey, PathBuf, EngineEntryMode)> {
    match action {
        app::EngineSelectionAction::Attach {
            engine_key,
            socket_path,
        } => {
            if !engine_sessions.contains_key(&engine_key) {
                let session = engine_runtime::connect_existing_engine(&socket_path).await?;
                insert_observed_engine_session(
                    engine_sessions,
                    engine_key.clone(),
                    session,
                    engine_event_tx,
                );
            }
            Ok((engine_key, socket_path, EngineEntryMode::AttachExisting))
        }
        app::EngineSelectionAction::CreateNew => {
            if cli.no_spawn_engine {
                bail!("engine creation is disabled by --no-spawn-engine");
            }
            let socket_path = unique_engine_socket_path(&cli.engine_socket);
            let session =
                engine_runtime::spawn_and_connect_engine(cli.config.as_deref(), &socket_path)
                    .await?;
            let engine_key = EngineKey::from_socket_path(&socket_path);
            insert_observed_engine_session(
                engine_sessions,
                engine_key.clone(),
                session,
                engine_event_tx,
            );
            Ok((engine_key, socket_path, EngineEntryMode::CreateNew))
        }
        app::EngineSelectionAction::Refresh
        | app::EngineSelectionAction::Kill { .. }
        | app::EngineSelectionAction::CloseAndKill { .. } => {
            bail!("unsupported engine connection action")
        }
    }
}

async fn refresh_engine_overview(
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    engine_event_tx: &mpsc::UnboundedSender<EngineRelayMessage>,
    dummy_cmd_tx: &mpsc::UnboundedSender<ServiceCommand>,
    announce: bool,
) -> Result<()> {
    let running_engines = list_running_engines()?;
    let entries = running_engines
        .iter()
        .map(|engine| {
            (
                EngineKey::from_socket_path(&engine.socket_path),
                engine.socket_path.clone(),
                engine.socket_is_live,
            )
        })
        .collect::<Vec<_>>();
    app.set_running_engines(running_engines);
    observe_running_engine_sessions(entries, app, engine_sessions, engine_event_tx, dummy_cmd_tx)
        .await;
    if announce {
        app.handle_service_event(
            ServiceEvent::Status("Engine list refreshed.".to_string()),
            dummy_cmd_tx,
        );
    }
    Ok(())
}

async fn observe_running_engine_sessions(
    entries: Vec<(EngineKey, PathBuf, bool)>,
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    engine_event_tx: &mpsc::UnboundedSender<EngineRelayMessage>,
    dummy_cmd_tx: &mpsc::UnboundedSender<ServiceCommand>,
) {
    for (engine_key, socket_path, socket_is_live) in entries {
        if !socket_is_live || engine_sessions.contains_key(&engine_key) {
            continue;
        }
        match engine_runtime::connect_existing_engine(&socket_path).await {
            Ok(session) => {
                insert_observed_engine_session(
                    engine_sessions,
                    engine_key,
                    session,
                    engine_event_tx,
                );
            }
            Err(err) => {
                app.handle_engine_service_event(
                    engine_key,
                    ServiceEvent::Error(format!("Engine observer failed: {err}")),
                    false,
                    dummy_cmd_tx,
                );
            }
        }
    }
}

fn spawn_engine_lifecycle_action(
    action: app::EngineLifecycleAction,
    id: u32,
    engine_event_tx: &mpsc::UnboundedSender<EngineRelayMessage>,
) {
    let engine_event_tx = engine_event_tx.clone();
    tokio::spawn(async move {
        let result = match action {
            app::EngineLifecycleAction::Kill => kill_engine_process(id).await,
            app::EngineLifecycleAction::CloseAndKill => close_and_kill_engine(id).await,
        }
        .map_err(|err| err.to_string());
        let _ = engine_event_tx.send(EngineRelayMessage::LifecycleCompleted { action, id, result });
    });
}

fn engine_lifecycle_success_message(action: app::EngineLifecycleAction, id: u32) -> String {
    match action {
        app::EngineLifecycleAction::Kill => format!("Killed engine {id}."),
        app::EngineLifecycleAction::CloseAndKill => {
            format!("Closed the selected market and killed engine {id}.")
        }
    }
}

fn engine_lifecycle_failure_label(action: app::EngineLifecycleAction) -> &'static str {
    match action {
        app::EngineLifecycleAction::Kill => "Kill",
        app::EngineLifecycleAction::CloseAndKill => "Close and kill",
    }
}

fn enter_engine_session(
    app: &mut App,
    engine_sessions: &HashMap<EngineKey, ObservedEngineSession>,
    active_engine_key: &mut Option<EngineKey>,
    engine_key: EngineKey,
    socket_path: PathBuf,
    config: &AppConfig,
    mode: EngineEntryMode,
) {
    *active_engine_key = Some(engine_key.clone());
    app.enter_engine_session_for_key(engine_key.clone(), socket_path);
    if let Some(session) = engine_sessions.get(&engine_key) {
        let _ = session.cmd_tx.send(ServiceCommand::ReplayState);
    }
    if should_autoconnect_engine_session(config, app.awaiting_broker_selection(), mode) {
        if let Some(session) = engine_sessions.get(&engine_key) {
            let _ = session.cmd_tx.send(ServiceCommand::Connect(config.clone()));
        }
    }
}

#[derive(Debug)]
struct ObservedEngineSession {
    cmd_tx: mpsc::UnboundedSender<ServiceCommand>,
    _child: Option<tokio::process::Child>,
    _relay_task: JoinHandle<()>,
}

#[derive(Debug)]
struct EngineEventEnvelope {
    engine_key: EngineKey,
    event: ServiceEvent,
}

#[derive(Debug)]
enum EngineRelayMessage {
    Event(EngineEventEnvelope),
    Closed {
        engine_key: EngineKey,
    },
    LifecycleCompleted {
        action: app::EngineLifecycleAction,
        id: u32,
        result: Result<(), String>,
    },
}

fn insert_observed_engine_session(
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    engine_key: EngineKey,
    session: EngineSession,
    engine_event_tx: &mpsc::UnboundedSender<EngineRelayMessage>,
) {
    let relay_task = spawn_engine_event_relay(
        engine_key.clone(),
        session.event_rx,
        engine_event_tx.clone(),
    );
    engine_sessions.insert(
        engine_key,
        ObservedEngineSession {
            cmd_tx: session.cmd_tx,
            _child: session.child,
            _relay_task: relay_task,
        },
    );
}

fn spawn_engine_event_relay(
    engine_key: EngineKey,
    mut event_rx: mpsc::UnboundedReceiver<ServiceEvent>,
    engine_event_tx: mpsc::UnboundedSender<EngineRelayMessage>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            if engine_event_tx
                .send(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: engine_key.clone(),
                    event,
                }))
                .is_err()
            {
                return;
            }
        }
        let _ = engine_event_tx.send(EngineRelayMessage::Closed { engine_key });
    })
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    terminal.clear()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_attach_does_not_autoconnect_existing_session() {
        let config = AppConfig {
            autoconnect: true,
            ..AppConfig::default()
        };

        assert!(!should_autoconnect_engine_session(
            &config,
            false,
            EngineEntryMode::AttachExisting
        ));
    }

    #[test]
    fn engine_create_can_autoconnect_when_broker_is_already_selected() {
        let config = AppConfig {
            autoconnect: true,
            ..AppConfig::default()
        };

        assert!(should_autoconnect_engine_session(
            &config,
            false,
            EngineEntryMode::CreateNew
        ));
    }

    #[test]
    fn engine_create_does_not_autoconnect_while_broker_picker_is_visible() {
        let config = AppConfig {
            autoconnect: true,
            ..AppConfig::default()
        };

        assert!(!should_autoconnect_engine_session(
            &config,
            true,
            EngineEntryMode::CreateNew
        ));
    }

    #[tokio::test]
    async fn engine_event_relay_tags_events_and_reports_closed() {
        let key =
            EngineKey::from_socket_path(PathBuf::from("/tmp/trader-engine-99.sock").as_path());
        let (service_tx, service_rx) = mpsc::unbounded_channel();
        let (relay_tx, mut relay_rx) = mpsc::unbounded_channel();
        let _relay_task = spawn_engine_event_relay(key.clone(), service_rx, relay_tx);

        service_tx
            .send(ServiceEvent::Status("ready".to_string()))
            .expect("send service event");
        drop(service_tx);

        match tokio::time::timeout(Duration::from_secs(1), relay_rx.recv())
            .await
            .expect("relay event timed out")
            .expect("expected relay event")
        {
            EngineRelayMessage::Event(envelope) => {
                assert_eq!(envelope.engine_key, key);
                assert!(
                    matches!(envelope.event, ServiceEvent::Status(message) if message == "ready")
                );
            }
            EngineRelayMessage::Closed { .. } => panic!("expected tagged service event first"),
            EngineRelayMessage::LifecycleCompleted { .. } => {
                panic!("expected tagged service event first")
            }
        }

        match tokio::time::timeout(Duration::from_secs(1), relay_rx.recv())
            .await
            .expect("relay close timed out")
            .expect("expected relay close")
        {
            EngineRelayMessage::Closed { engine_key } => {
                assert_eq!(engine_key, key);
            }
            EngineRelayMessage::Event(_) => panic!("expected relay close after sender drop"),
            EngineRelayMessage::LifecycleCompleted { .. } => {
                panic!("expected relay close after sender drop")
            }
        }
    }

    #[cfg(feature = "replay")]
    #[test]
    fn replay_download_planner_parses_safe_request_parts() {
        assert!(matches!(
            parse_replay_download_source_kind("raw-ticks").expect("source kind"),
            replay_cache::ReplayCacheSourceKind::RawTicks
        ));
        assert_eq!(
            parse_replay_download_bar_type("volume", 6500).expect("bar type"),
            broker::BarType::volume(6500)
        );
        assert_eq!(
            parse_replay_download_chart_mode("heikin-ashi").expect("chart mode"),
            broker::CandleMode::HeikinAshi
        );
        assert!(parse_replay_download_source_kind("dom").is_err());
    }

    #[cfg(feature = "replay")]
    #[test]
    fn replay_download_planner_rejects_empty_or_zero_request_parts() {
        let config = AppConfig::default();
        let valid = ReplayDownloadArgs {
            instrument: "MES".to_string(),
            contract: "MESU6".to_string(),
            start: "2026-07-23".to_string(),
            end: "2026-07-24".to_string(),
            source_kind: "server-bars".to_string(),
            bar_kind: "minute".to_string(),
            bar_value: 1,
            chart_mode: "ohlc".to_string(),
            cache_dir: None,
        };

        let mut blank_contract = valid.clone();
        blank_contract.contract = " ".to_string();
        assert!(build_replay_download_plan(&config, blank_contract).is_err());

        let mut zero_bar_value = valid;
        zero_bar_value.bar_value = 0;
        assert!(build_replay_download_plan(&config, zero_bar_value).is_err());
    }

    #[cfg(feature = "replay")]
    #[test]
    fn replay_download_plan_uses_inclusive_end_date_and_cache_root() {
        let mut config = AppConfig::default();
        config.replay_cache_dir = PathBuf::from("/tmp/trader-cache-test");
        let plan = build_replay_download_plan(
            &config,
            ReplayDownloadArgs {
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
                start: "2026-07-23".to_string(),
                end: "2026-07-24".to_string(),
                source_kind: "server-bars".to_string(),
                bar_kind: "minute".to_string(),
                bar_value: 5,
                chart_mode: "heikin-ashi".to_string(),
                cache_dir: None,
            },
        )
        .expect("download plan");

        assert_eq!(plan.start.to_rfc3339(), "2026-07-23T00:00:00+00:00");
        assert_eq!(plan.end.to_rfc3339(), "2026-07-25T00:00:00+00:00");
        assert_eq!(plan.cache_root, PathBuf::from("/tmp/trader-cache-test"));
        assert_eq!(plan.bar_type, broker::BarType::minute(5));
        assert_eq!(plan.chart_mode, broker::CandleMode::HeikinAshi);
    }

    #[cfg(feature = "replay")]
    #[test]
    fn replay_download_plan_accepts_raw_tick_source_kind() {
        let config = AppConfig::default();
        let plan = build_replay_download_plan(
            &config,
            ReplayDownloadArgs {
                instrument: "MES".to_string(),
                contract: "MESU6".to_string(),
                start: "2026-07-23".to_string(),
                end: "2026-07-24".to_string(),
                source_kind: "raw-ticks".to_string(),
                bar_kind: "minute".to_string(),
                bar_value: 1,
                chart_mode: "ohlc".to_string(),
                cache_dir: None,
            },
        )
        .expect("raw tick download plan");

        assert_eq!(
            plan.source_kind,
            replay_cache::ReplayCacheSourceKind::RawTicks
        );
        assert_eq!(plan.start.to_rfc3339(), "2026-07-23T00:00:00+00:00");
        assert_eq!(plan.end.to_rfc3339(), "2026-07-25T00:00:00+00:00");
    }
}
