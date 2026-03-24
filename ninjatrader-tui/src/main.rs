mod app;
mod config;
mod engine_control;
mod engine_registry;
mod ipc;
mod strategies;
mod strategy;
mod tradovate;

use anyhow::{Result, bail};
use app::App;
use clap::{Parser, Subcommand};
use config::AppConfig;
use crossterm::event::{Event as CEvent, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use engine_control::{close_and_kill_engine, kill_engine_process};
use engine_registry::{list_running_engines, resolve_engine};
use futures_util::StreamExt;
use ipc::{connect_client, run_engine_server};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tradovate::ServiceCommand;

#[derive(Debug, Parser)]
#[command(name = "ninjatrader-tui")]
#[command(about = "Ratatui terminal client for Tradovate / NinjaTrader")]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long, default_value = ".run/ninjatrader-engine.sock")]
    engine_socket: PathBuf,

    #[arg(long)]
    no_spawn_engine: bool,

    #[command(subcommand)]
    mode: Option<Mode>,
}

#[derive(Debug, Clone, Copy, Subcommand)]
enum Mode {
    /// Run the background engine server.
    Engine,
    /// List running engine processes on this host.
    List,
    /// Attach a full TUI session to a running engine by ID from `list`.
    Attach {
        /// Engine ID from `ninjatrader-tui list` (PID).
        id: u32,
    },
    /// Kill one running engine by ID.
    Kill {
        /// Engine ID from `ninjatrader-tui list` (PID).
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
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

    run_tui(&cli, config).await
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

async fn run_tui(cli: &Cli, config: AppConfig) -> Result<()> {
    let (_engine_child, cmd_tx, mut event_rx) = connect_or_spawn_engine(cli).await?;

    let mut terminal = init_terminal()?;
    let mut app = App::new(config.clone());
    let mut event_stream = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(125));
    tick.tick().await;

    if config.autoconnect {
        let _ = cmd_tx.send(ServiceCommand::Connect(config));
    }

    loop {
        terminal.draw(|frame| app.draw(frame))?;

        tokio::select! {
            _ = tick.tick() => {}
            maybe_event = event_stream.next() => {
                match maybe_event {
                    Some(Ok(CEvent::Key(key))) => app.handle_key(key, &cmd_tx),
                    Some(Ok(CEvent::Resize(_, _))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        app.handle_service_event(
                            tradovate::ServiceEvent::Error(err.to_string()),
                            &cmd_tx,
                        );
                    }
                    None => break,
                }
            }
            maybe_service = event_rx.recv() => {
                if let Some(event) = maybe_service {
                    app.handle_service_event(event, &cmd_tx);
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

async fn connect_or_spawn_engine(
    cli: &Cli,
) -> Result<(
    Option<tokio::process::Child>,
    mpsc::UnboundedSender<ServiceCommand>,
    mpsc::UnboundedReceiver<tradovate::ServiceEvent>,
)> {
    if let Ok((cmd_tx, event_rx)) = connect_client(&cli.engine_socket).await {
        let _ = cmd_tx.send(ServiceCommand::ReplayState);
        return Ok((None, cmd_tx, event_rx));
    }
    if cli.no_spawn_engine {
        bail!(
            "engine socket {} is unavailable and --no-spawn-engine was set",
            cli.engine_socket.display()
        );
    }

    let mut child = spawn_engine_process(cli)?;
    for _ in 0..50 {
        if let Ok((cmd_tx, event_rx)) = connect_client(&cli.engine_socket).await {
            let _ = cmd_tx.send(ServiceCommand::ReplayState);
            return Ok((Some(child), cmd_tx, event_rx));
        }
        sleep(Duration::from_millis(100)).await;
    }

    let _ = child.start_kill();
    bail!(
        "timed out waiting for engine socket {}",
        cli.engine_socket.display()
    );
}

fn spawn_engine_process(cli: &Cli) -> Result<tokio::process::Child> {
    let current_exe = std::env::current_exe()?;
    let mut command = tokio::process::Command::new(current_exe);
    if let Some(config_path) = cli.config.as_ref() {
        command.arg("--config").arg(config_path);
    }
    command
        .arg("--engine-socket")
        .arg(&cli.engine_socket)
        .arg("engine")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    Ok(command.spawn()?)
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
