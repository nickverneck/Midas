mod app;
mod automation;
mod config;
mod strategies;
mod strategy;
mod tradovate;

use anyhow::Result;
use app::App;
use clap::Parser;
use config::AppConfig;
use crossterm::event::{Event as CEvent, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tradovate::{ServiceCommand, service_loop};

#[derive(Debug, Parser)]
#[command(name = "ninjatrader-tui")]
#[command(about = "Ratatui terminal client for Tradovate / NinjaTrader")]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = AppConfig::load(cli.config.as_deref())?;

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    tokio::spawn(service_loop(cmd_rx, event_tx));

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
