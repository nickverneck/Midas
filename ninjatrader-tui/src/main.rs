mod app;
mod config;
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
    Engine,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.mode, Some(Mode::Engine)) {
        return run_engine_server(&cli.engine_socket).await;
    }

    let config = AppConfig::load(cli.config.as_deref())?;
    let (_engine_child, cmd_tx, mut event_rx) = connect_or_spawn_engine(&cli).await?;

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
