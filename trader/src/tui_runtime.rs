use crate::app::{self, App, EngineKey};
use crate::broker::ServiceEvent;
use crate::cli::Cli;
use crate::config::AppConfig;
use crate::engine_registry::list_running_engines;
use crate::engine_session::{
    EngineEntryMode, EngineRelayMessage, ObservedEngineSession, connect_or_spawn_engine,
    connect_selected_engine, engine_lifecycle_failure_label, engine_lifecycle_success_message,
    enter_engine_session, insert_observed_engine_session, observe_running_engine_sessions,
    refresh_engine_overview, spawn_engine_lifecycle_action,
};
use anyhow::Result;
use crossterm::event::{Event as CEvent, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) async fn run_tui(cli: &Cli, config: AppConfig, attach_mode: bool) -> Result<()> {
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
                        app.handle_engine_receiver_closed(&engine_key, is_active_detail);
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
