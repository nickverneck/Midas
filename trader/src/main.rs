mod app;
mod broker;
mod config;
mod engine_control;
mod engine_registry;
mod engine_runtime;
mod ipc;
#[cfg(feature = "ironbeam")]
mod ironbeam;
mod strategies;
mod strategy;
#[cfg(feature = "tradovate")]
mod tradovate;

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
        /// Disarm the strategy, close the selected market, then kill the engine.
        #[arg(short = 'c', long = "close")]
        close: bool,
    },
    /// Kill all running engines.
    #[command(name = "killall")]
    KillAll {
        /// Disarm each strategy, close the selected market, then kill the engine.
        #[arg(short = 'c', long = "close")]
        close: bool,
    },
    /// Run a non-TUI Tradovate sim swipe profiler for reversal paths.
    #[command(name = "swipe-profile")]
    SwipeProfile(SwipeProfileArgs),
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

    for (engine_key, socket_path, socket_is_live) in startup_engines {
        if !socket_is_live || engine_sessions.contains_key(&engine_key) {
            continue;
        }
        match engine_runtime::connect_existing_engine(&socket_path).await {
            Ok(session) => {
                insert_observed_engine_session(
                    &mut engine_sessions,
                    engine_key,
                    session,
                    &engine_event_tx,
                );
            }
            Err(err) => {
                app.handle_engine_service_event(
                    engine_key,
                    ServiceEvent::Error(format!("Engine observer failed: {err}")),
                    false,
                    &dummy_cmd_tx,
                );
            }
        }
    }

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
    Closed { engine_key: EngineKey },
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
        }
    }
}
