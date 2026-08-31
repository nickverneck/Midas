use crate::app::{self, App, EngineKey};
use crate::broker::{
    SERVICE_COMMAND_QUEUE_CAPACITY, ServiceCommandSender, ServiceEvent, service_command_channel,
};
use crate::cli::Cli;
use crate::config::AppConfig;
use crate::engine_registry::list_running_engines;
use crate::engine_runtime::{ATTACH_HYDRATION_TIMEOUT, AttachHydration, AttachHydrationProgress};
use crate::engine_session::{
    EngineEntryMode, EngineRelayMessage, EngineRelayQueue, ObservedEngineSession,
    connect_or_spawn_engine, connect_selected_engine, engine_lifecycle_failure_label,
    engine_lifecycle_success_message, enter_engine_session, insert_observed_engine_session,
    observe_running_engine_sessions, refresh_engine_overview, shutdown_observed_engine_sessions,
    spawn_engine_lifecycle_action,
};
use anyhow::{Result, bail};
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
use tokio::time::{Instant as TokioInstant, timeout_at};

const ENGINE_EVENT_BATCH_LIMIT: usize = 256;

pub(crate) async fn run_tui(cli: &Cli, config: AppConfig, attach_mode: bool) -> Result<()> {
    let running_engines = list_running_engines()?;
    let direct_session = if attach_mode {
        Some(connect_or_spawn_engine(cli).await?)
    } else {
        None
    };

    let mut terminal = TerminalGuard::new(init_terminal()?);
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
    let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(SERVICE_COMMAND_QUEUE_CAPACITY);
    let engine_event_queue = EngineRelayQueue::new();
    let mut engine_sessions = HashMap::<EngineKey, ObservedEngineSession>::new();
    let mut active_engine_key = None::<EngineKey>;
    tick.tick().await;

    if let Some(session) = direct_session {
        let engine_key = EngineKey::from_socket_path(&cli.engine_socket);
        insert_observed_engine_session(
            &mut engine_sessions,
            engine_key.clone(),
            session,
            &engine_event_queue,
        )
        .await;
        if let Err(err) = hydrate_and_enter_existing_engine(
            &mut app,
            &mut engine_sessions,
            &mut active_engine_key,
            engine_key,
            cli.engine_socket.clone(),
            &config,
            &engine_event_queue,
            &dummy_cmd_tx,
        )
        .await
        {
            return Err(err);
        }
    }

    observe_running_engine_sessions(
        startup_engines,
        &mut app,
        &mut engine_sessions,
        &engine_event_queue,
        &dummy_cmd_tx,
    )
    .await;

    let loop_result = async {
        loop {
            terminal.terminal_mut().draw(|frame| app.draw(frame))?;

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
                                | app::EngineSelectionAction::CreateNew { .. } => {
                                    match connect_selected_engine(
                                        cli,
                                        action,
                                        &engine_event_queue,
                                        &mut engine_sessions,
                                    )
                                    .await
                                    {
                                        Ok((engine_key, socket_path, mode)) => {
                                            if mode == EngineEntryMode::AttachExisting {
                                                if let Err(err) = hydrate_and_enter_existing_engine(
                                                    &mut app,
                                                    &mut engine_sessions,
                                                    &mut active_engine_key,
                                                    engine_key,
                                                    socket_path,
                                                    &config,
                                                    &engine_event_queue,
                                                    &dummy_cmd_tx,
                                                )
                                                .await
                                                {
                                                    let active_cmd_tx = active_engine_key
                                                        .as_ref()
                                                        .and_then(|active_key| {
                                                            engine_sessions.get(active_key)
                                                        })
                                                        .map(|session| &session.cmd_tx)
                                                        .unwrap_or(&dummy_cmd_tx);
                                                    app.handle_service_event(
                                                        ServiceEvent::Error(format!(
                                                            "Engine attach failed: {err}"
                                                        )),
                                                        active_cmd_tx,
                                                    );
                                                }
                                            } else {
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
                                        &engine_event_queue,
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
                                        &engine_event_queue,
                                    );
                                }
                                app::EngineSelectionAction::CloseAndKill { id } => {
                                    spawn_engine_lifecycle_action(
                                        app::EngineLifecycleAction::CloseAndKill,
                                        id,
                                        &engine_event_queue,
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
            message = engine_event_queue.recv() => {
                handle_engine_relay_message(
                    message,
                    &mut app,
                    &mut engine_sessions,
                    &mut active_engine_key,
                    &engine_event_queue,
                    &dummy_cmd_tx,
                )
                .await;

                // A busy range-bar engine can produce more UI updates than a
                // single render cycle can consume. Drain a bounded batch so
                // the UI catches up instead of rendering one stale event per
                // 125 ms tick forever. The bound keeps keyboard input and
                // rendering responsive even when a client reconnects to a
                // previously busy engine.
                for _ in 1..ENGINE_EVENT_BATCH_LIMIT {
                    if let Some(message) = engine_event_queue.try_recv() {
                            handle_engine_relay_message(
                                message,
                                &mut app,
                                &mut engine_sessions,
                                &mut active_engine_key,
                                &engine_event_queue,
                                &dummy_cmd_tx,
                            )
                            .await;
                    } else {
                        break;
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;

    // Restore the terminal before cleanup can await a spawned child. If draw
    // or the event loop returned an error, the guard still restores it in its
    // destructor.
    let restore_result = terminal.restore();
    shutdown_observed_engine_sessions(&mut engine_sessions).await;
    loop_result?;
    restore_result?;
    Ok(())
}

/// Drain the first ReplayState response before activating an existing engine
/// in the UI. The relay already owns the engine session, so this gate consumes
/// only the target generation, updates its overview summary, and then replays
/// the same initial events once after activation. This prevents the default
/// App state from ever selecting or rendering a stale broker/strategy screen.
async fn hydrate_and_enter_existing_engine(
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    active_engine_key: &mut Option<EngineKey>,
    engine_key: EngineKey,
    socket_path: std::path::PathBuf,
    config: &AppConfig,
    engine_event_queue: &std::sync::Arc<EngineRelayQueue>,
    dummy_cmd_tx: &ServiceCommandSender,
) -> Result<()> {
    hydrate_and_enter_existing_engine_until(
        app,
        engine_sessions,
        active_engine_key,
        engine_key,
        socket_path,
        config,
        engine_event_queue,
        dummy_cmd_tx,
        TokioInstant::now() + ATTACH_HYDRATION_TIMEOUT,
    )
    .await
}

async fn hydrate_and_enter_existing_engine_until(
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    active_engine_key: &mut Option<EngineKey>,
    engine_key: EngineKey,
    socket_path: std::path::PathBuf,
    config: &AppConfig,
    engine_event_queue: &std::sync::Arc<EngineRelayQueue>,
    dummy_cmd_tx: &ServiceCommandSender,
    deadline: TokioInstant,
) -> Result<()> {
    app.observe_live_engine_socket(socket_path.clone());
    let Some(generation) = engine_sessions
        .get(&engine_key)
        .map(|session| session.generation)
    else {
        app.handle_service_event(
            ServiceEvent::Error(format!(
                "Cannot attach to engine {}; its session is unavailable.",
                socket_path.display()
            )),
            dummy_cmd_tx,
        );
        bail!(
            "Cannot attach to engine {}; its session is unavailable.",
            socket_path.display()
        );
    };

    let mut hydration = AttachHydration::default();
    let mut initial_events = Vec::new();

    loop {
        let message = match timeout_at(deadline, engine_event_queue.recv()).await {
            Ok(message) => message,
            Err(_) => {
                let message = format!(
                    "Timed out waiting for engine {} state hydration.",
                    socket_path.display()
                );
                fail_engine_hydration(
                    app,
                    engine_sessions,
                    engine_event_queue,
                    active_engine_key,
                    &engine_key,
                    generation,
                    message.clone(),
                    dummy_cmd_tx,
                )
                .await;
                bail!(message);
            }
        };

        match message {
            EngineRelayMessage::Event(envelope)
                if envelope.engine_key == engine_key && envelope.generation == generation =>
            {
                let progress = hydration.observe(&envelope.event);
                let event = envelope.event;
                let summary_cmd_tx = active_engine_key
                    .as_ref()
                    .and_then(|active_key| engine_sessions.get(active_key))
                    .map(|session| session.cmd_tx.clone())
                    .unwrap_or_else(|| dummy_cmd_tx.clone());
                app.handle_engine_service_event_sequenced(
                    engine_key.clone(),
                    event.clone(),
                    false,
                    &summary_cmd_tx,
                    envelope.sequence,
                );
                if progress == AttachHydrationProgress::Terminal {
                    let message = match event {
                        ServiceEvent::Disconnected => format!(
                            "Engine {} disconnected before state hydration completed.",
                            socket_path.display()
                        ),
                        ServiceEvent::Error(message) => format!(
                            "Engine {} failed during state hydration: {message}",
                            socket_path.display()
                        ),
                        _ => format!(
                            "Engine {} terminated before state hydration completed.",
                            socket_path.display()
                        ),
                    };
                    fail_engine_hydration(
                        app,
                        engine_sessions,
                        engine_event_queue,
                        active_engine_key,
                        &engine_key,
                        generation,
                        message.clone(),
                        dummy_cmd_tx,
                    )
                    .await;
                    bail!(message);
                }
                initial_events.push(event);

                if progress == AttachHydrationProgress::Pending {
                    continue;
                }

                enter_engine_session(
                    app,
                    engine_sessions,
                    active_engine_key,
                    engine_key.clone(),
                    socket_path.clone(),
                    config,
                    EngineEntryMode::AttachExisting,
                );
                let Some(session) = engine_sessions.get(&engine_key) else {
                    bail!(
                        "Engine {} session disappeared during state hydration.",
                        socket_path.display()
                    );
                };
                let detail_cmd_tx = session.cmd_tx.clone();
                for event in initial_events {
                    app.handle_engine_service_event(
                        engine_key.clone(),
                        event,
                        true,
                        &detail_cmd_tx,
                    );
                }
                return Ok(());
            }
            other => {
                // Other observed engines may also be producing startup
                // events. Keep their existing behavior, but give every await
                // reachable here the same absolute hydration deadline so a
                // refresh/reconnect cannot hold direct attach forever.
                match timeout_at(
                    deadline,
                    handle_engine_relay_message(
                        other,
                        app,
                        engine_sessions,
                        active_engine_key,
                        engine_event_queue,
                        dummy_cmd_tx,
                    ),
                )
                .await
                {
                    Ok(()) => {}
                    Err(_) => {
                        let message = format!(
                            "Timed out while processing engine {} state hydration.",
                            socket_path.display()
                        );
                        fail_engine_hydration(
                            app,
                            engine_sessions,
                            engine_event_queue,
                            active_engine_key,
                            &engine_key,
                            generation,
                            message.clone(),
                            dummy_cmd_tx,
                        )
                        .await;
                        bail!(message);
                    }
                }
                if !engine_sessions.contains_key(&engine_key) {
                    bail!(
                        "Engine {} session ended during state hydration.",
                        socket_path.display()
                    );
                }
            }
        }
    }
}

async fn fail_engine_hydration(
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    engine_event_queue: &std::sync::Arc<EngineRelayQueue>,
    active_engine_key: &mut Option<EngineKey>,
    engine_key: &EngineKey,
    generation: u64,
    message: String,
    dummy_cmd_tx: &ServiceCommandSender,
) {
    app.handle_service_event(ServiceEvent::Error(message), dummy_cmd_tx);

    // Remove only the exact failed session. A newer attach may already have
    // replaced this key; its relay, child, and active TUI state must remain
    // untouched.
    let failed_session = engine_sessions
        .get(engine_key)
        .filter(|session| session.generation == generation)
        .is_some()
        .then(|| {
            engine_sessions
                .remove(engine_key)
                .expect("session still present")
        });
    if let Some(session) = failed_session {
        // Invalidate queued events for this generation before stopping the
        // relay. The next attach will call begin_connection and receive a
        // fresh generation.
        let _ = invalidate_failed_attach_generation(engine_event_queue, engine_key, generation);
        session.shutdown().await;
        if active_engine_key.as_ref() == Some(engine_key) {
            *active_engine_key = None;
        }
    }
}

fn invalidate_failed_attach_generation(
    engine_event_queue: &std::sync::Arc<EngineRelayQueue>,
    engine_key: &EngineKey,
    generation: u64,
) -> bool {
    // Invalidate only the failed connection.  A newer attach may already have
    // replaced it by the time timeout/terminal cleanup runs; in that case the
    // newer generation and its resources must remain untouched.
    engine_event_queue.invalidate_connection(engine_key, generation)
}

async fn handle_engine_relay_message(
    message: EngineRelayMessage,
    app: &mut App,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    active_engine_key: &mut Option<EngineKey>,
    engine_event_queue: &std::sync::Arc<EngineRelayQueue>,
    dummy_cmd_tx: &ServiceCommandSender,
) {
    match message {
        EngineRelayMessage::Event(envelope) => {
            let Some(session) = engine_sessions.get(&envelope.engine_key) else {
                return;
            };
            // A detached relay may still have unread IPC events.  Its
            // generation is invalid after reconnect, so it must not update
            // either the engine picker summary or active Dashboard/F6 state.
            if session.generation != envelope.generation {
                return;
            }
            let is_active_detail = active_engine_key.as_ref() == Some(&envelope.engine_key);
            let active_cmd_tx = active_engine_key
                .as_ref()
                .and_then(|engine_key| engine_sessions.get(engine_key))
                .map(|session| &session.cmd_tx)
                .unwrap_or(dummy_cmd_tx);
            app.handle_engine_service_event_sequenced(
                envelope.engine_key,
                envelope.event,
                is_active_detail,
                active_cmd_tx,
                envelope.sequence,
            );
        }
        EngineRelayMessage::Closed {
            engine_key,
            generation,
        } => {
            let Some(session) = engine_sessions.get(&engine_key) else {
                return;
            };
            if session.generation != generation {
                return;
            }
            let is_active_detail = active_engine_key.as_ref() == Some(&engine_key);
            app.handle_engine_receiver_closed(&engine_key, is_active_detail);
            if let Some(session) = engine_sessions.remove(&engine_key) {
                session.shutdown().await;
            }
            if is_active_detail {
                *active_engine_key = None;
            }
        }
        EngineRelayMessage::RelayOverflow {
            engine_key,
            generation,
            reason,
        } => {
            let active_cmd_tx = active_engine_key
                .as_ref()
                .and_then(|key| engine_sessions.get(key))
                .map(|session| session.cmd_tx.clone())
                .unwrap_or_else(|| dummy_cmd_tx.clone());
            app.handle_service_event(ServiceEvent::Error(reason), &active_cmd_tx);
            if let Some(engine_key) = engine_key {
                if let Some(session) = engine_sessions.get(&engine_key)
                    && generation != Some(session.generation)
                {
                    return;
                }
                let is_active_detail = active_engine_key.as_ref() == Some(&engine_key);
                if let Some(session) = engine_sessions.remove(&engine_key) {
                    session.shutdown().await;
                }
                if is_active_detail {
                    app.handle_engine_receiver_closed(&engine_key, true);
                    *active_engine_key = None;
                }
            } else {
                // A relay-wide overflow invalidates every generation. Do not
                // leave any of their IPC relays waiting on a closed queue.
                let was_active = active_engine_key.take();
                for (_, session) in std::mem::take(engine_sessions) {
                    session.shutdown().await;
                }
                if let Some(engine_key) = was_active {
                    app.handle_engine_receiver_closed(&engine_key, true);
                }
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
                        app,
                        engine_sessions,
                        engine_event_queue,
                        dummy_cmd_tx,
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

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    restored: bool,
}

impl TerminalGuard {
    fn new(terminal: Terminal<CrosstermBackend<Stdout>>) -> Self {
        Self {
            terminal,
            restored: false,
        }
    }

    fn terminal_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }

    fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        restore_terminal(&mut self.terminal)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.restored {
            let _ = restore_terminal(&mut self.terminal);
            self.restored = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{
        AccountInfo, AccountSnapshot, Bar, BarType, BrokerCapabilities, BrokerKind, CandleMode,
        MarketSnapshot, ServiceEvent, SessionKind, service_command_channel, service_event_channel,
    };
    use crate::config::{AuthMode, TradingEnvironment};
    use crate::engine_runtime::EngineSession;
    use crate::engine_session::EngineEventEnvelope;
    use crate::strategy::{ExecutionRuntimeSnapshot, ExecutionStateSnapshot, NativeStrategyKind};
    use ratatui::{Terminal, backend::TestBackend};
    use tokio::time::timeout;

    fn test_engine_session() -> EngineSession {
        let (cmd_tx, _cmd_rx) = service_command_channel(1);
        let (_event_tx, event_rx) = service_event_channel(1);
        EngineSession {
            child: None,
            cmd_tx,
            event_rx,
        }
    }

    fn test_connected_event() -> ServiceEvent {
        ServiceEvent::Connected {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_name: None,
            auth_mode: AuthMode::TokenFile,
            session_kind: SessionKind::Live,
            capabilities: BrokerCapabilities {
                automated_orders: true,
                ..BrokerCapabilities::default()
            },
        }
    }

    #[test]
    fn failed_attach_generation_cannot_publish_after_cleanup() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/attach-timeout.sock"));
        let generation = queue.begin_connection(key.clone());

        assert!(
            queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation,
                sequence: 1,
                event: ServiceEvent::Status("before failure".to_string()),
            }))
        );

        assert!(invalidate_failed_attach_generation(
            &queue, &key, generation
        ));

        assert!(
            !queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation,
                sequence: 2,
                event: ServiceEvent::Status("after failure".to_string()),
            }))
        );
        assert!(queue.try_recv().is_none());

        let replacement_generation = queue.begin_connection(key.clone());
        assert_ne!(replacement_generation, generation);
        assert!(
            queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key,
                generation: replacement_generation,
                sequence: 3,
                event: ServiceEvent::Status("replacement connection".to_string()),
            }))
        );
    }

    #[test]
    fn failed_attach_does_not_invalidate_a_newer_generation() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/attach-reconnect.sock"));
        let failed_generation = queue.begin_connection(key.clone());
        let current_generation = queue.begin_connection(key.clone());

        assert!(!invalidate_failed_attach_generation(
            &queue,
            &key,
            failed_generation
        ));
        assert!(
            queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key,
                generation: current_generation,
                sequence: 1,
                event: ServiceEvent::Status("new connection".to_string()),
            }))
        );
    }

    #[tokio::test]
    async fn silent_accepted_engine_times_out_and_cleans_up_session() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/silent-engine.sock"));
        let mut sessions = HashMap::new();
        insert_observed_engine_session(&mut sessions, key.clone(), test_engine_session(), &queue)
            .await;
        let mut app = App::new(AppConfig::default());
        let mut active = None;
        let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(1);

        let result = timeout(
            Duration::from_secs(1),
            hydrate_and_enter_existing_engine_until(
                &mut app,
                &mut sessions,
                &mut active,
                key,
                std::path::PathBuf::from("/tmp/silent-engine.sock"),
                &AppConfig::default(),
                &queue,
                &dummy_cmd_tx,
                TokioInstant::now() + Duration::from_millis(10),
            ),
        )
        .await
        .expect("silent hydration should be bounded");

        assert!(result.is_err());
        assert!(sessions.is_empty());
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn terminal_engine_disconnect_returns_error_and_cleans_up_session() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/terminal-engine.sock"));
        let mut sessions = HashMap::new();
        insert_observed_engine_session(&mut sessions, key.clone(), test_engine_session(), &queue)
            .await;
        let generation = sessions[&key].generation;
        assert!(
            queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation,
                sequence: 1,
                event: ServiceEvent::Disconnected,
            }))
        );
        let mut app = App::new(AppConfig::default());
        let mut active = None;
        let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(1);

        let result = hydrate_and_enter_existing_engine_until(
            &mut app,
            &mut sessions,
            &mut active,
            key,
            std::path::PathBuf::from("/tmp/terminal-engine.sock"),
            &AppConfig::default(),
            &queue,
            &dummy_cmd_tx,
            TokioInstant::now() + Duration::from_secs(1),
        )
        .await;

        assert!(result.is_err());
        assert!(sessions.is_empty());
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn closed_engine_event_reaps_owned_child() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/closed-engine.sock"));
        let mut sessions = HashMap::new();
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn test child");
        let child_pid = child.id().expect("test child pid");
        let (cmd_tx, _cmd_rx) = service_command_channel(1);
        let (event_tx, event_rx) = service_event_channel(1);
        insert_observed_engine_session(
            &mut sessions,
            key.clone(),
            EngineSession {
                child: Some(child),
                cmd_tx,
                event_rx,
            },
            &queue,
        )
        .await;
        let generation = sessions[&key].generation;
        let mut app = App::new(AppConfig::default());
        let mut active = Some(key.clone());
        let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(1);

        assert!(queue.publish(EngineRelayMessage::Closed {
            engine_key: key.clone(),
            generation,
        }));
        handle_engine_relay_message(
            queue.try_recv().expect("closed event"),
            &mut app,
            &mut sessions,
            &mut active,
            &queue,
            &dummy_cmd_tx,
        )
        .await;

        assert!(sessions.is_empty());
        assert!(active.is_none());
        assert_ne!(unsafe { libc::kill(child_pid as i32, 0) }, 0);
        drop(event_tx);
    }

    #[tokio::test]
    async fn relay_overflow_reaps_owned_child() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/overflow-engine.sock"));
        let mut sessions = HashMap::new();
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn test child");
        let child_pid = child.id().expect("test child pid");
        let (cmd_tx, _cmd_rx) = service_command_channel(1);
        let (event_tx, event_rx) = service_event_channel(1);
        insert_observed_engine_session(
            &mut sessions,
            key.clone(),
            EngineSession {
                child: Some(child),
                cmd_tx,
                event_rx,
            },
            &queue,
        )
        .await;
        let generation = sessions[&key].generation;
        let mut app = App::new(AppConfig::default());
        let mut active = Some(key.clone());
        let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(1);

        assert!(queue.publish(EngineRelayMessage::RelayOverflow {
            engine_key: Some(key.clone()),
            generation: Some(generation),
            reason: "test overflow".to_string(),
        }));
        handle_engine_relay_message(
            queue.try_recv().expect("overflow event"),
            &mut app,
            &mut sessions,
            &mut active,
            &queue,
            &dummy_cmd_tx,
        )
        .await;

        assert!(sessions.is_empty());
        assert!(active.is_none());
        assert_ne!(unsafe { libc::kill(child_pid as i32, 0) }, 0);
        drop(event_tx);
    }

    #[tokio::test]
    async fn successful_attach_hydrates_before_entering_session() {
        let queue = EngineRelayQueue::new();
        let key = EngineKey::from_socket_path(std::path::Path::new("/tmp/ready-engine.sock"));
        let mut sessions = HashMap::new();
        insert_observed_engine_session(&mut sessions, key.clone(), test_engine_session(), &queue)
            .await;
        let generation = sessions[&key].generation;
        let account = AccountInfo {
            id: 42,
            name: "Hydrated SIM".to_string(),
            raw: serde_json::json!({}),
        };
        let account_snapshot = AccountSnapshot {
            account_id: account.id,
            account_name: account.name.clone(),
            balance: Some(100_000.0),
            cash_balance: Some(100_000.0),
            net_liq: Some(100_000.0),
            realized_pnl: Some(125.0),
            unrealized_pnl: None,
            fees: None,
            intraday_margin: None,
            open_position_qty: Some(2.0),
            market_position_qty: Some(2.0),
            market_entry_price: Some(5000.25),
            selected_contract_take_profit_price: Some(5005.25),
            selected_contract_stop_price: Some(4997.25),
            raw_account: None,
            raw_risk: None,
            raw_cash: None,
            raw_positions: Vec::new(),
        };
        let market = MarketSnapshot {
            contract_id: Some(99),
            contract_name: Some("ESZ6".to_string()),
            candle_mode: CandleMode::Standard,
            bars: vec![Bar {
                ts_ns: 1,
                open: 5000.0,
                high: 5002.0,
                low: 4999.0,
                close: 5001.0,
                volume: Some(10.0),
            }],
            history_loaded: 1,
            live_bars: 1,
            status: "hydrated market".to_string(),
            ..MarketSnapshot::default()
        };
        let mut execution_config = ExecutionStateSnapshot::default().config;
        execution_config.native_strategy = NativeStrategyKind::HmaCross;
        for (sequence, event) in [
            (1, test_connected_event()),
            (2, ServiceEvent::AccountsLoaded(vec![account])),
            (
                3,
                ServiceEvent::AccountSnapshotsLoaded(vec![account_snapshot]),
            ),
            (4, ServiceEvent::MarketSnapshot(market)),
            (
                5,
                ServiceEvent::ExecutionState(ExecutionStateSnapshot {
                    config: execution_config,
                    runtime: ExecutionRuntimeSnapshot {
                        armed: true,
                        last_summary: "hydrated strategy".to_string(),
                        ..ExecutionRuntimeSnapshot::default()
                    },
                    bar_type: Some(BarType::range(10)),
                    candle_mode: Some(CandleMode::Standard),
                    selected_account_id: Some(42),
                    selected_contract_name: Some("ESZ6".to_string()),
                    market_position_qty: 2,
                    market_entry_price: Some(5000.25),
                    selected_contract_take_profit_price: Some(5005.25),
                    selected_contract_stop_price: Some(4997.25),
                }),
            ),
        ] {
            assert!(
                queue.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: key.clone(),
                    generation,
                    sequence,
                    event,
                }))
            );
        }
        let mut app = App::new(AppConfig::default());
        let mut active = None;
        let (dummy_cmd_tx, _dummy_cmd_rx) = service_command_channel(1);

        let result = hydrate_and_enter_existing_engine_until(
            &mut app,
            &mut sessions,
            &mut active,
            key.clone(),
            std::path::PathBuf::from("/tmp/ready-engine.sock"),
            &AppConfig::default(),
            &queue,
            &dummy_cmd_tx,
            TokioInstant::now() + Duration::from_secs(1),
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(active, Some(key));
        assert_eq!(sessions.len(), 1);

        let mut terminal = Terminal::new(TestBackend::new(160, 60)).expect("test terminal");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("render hydrated session");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .chunks(160)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("Selected account: Hydrated SIM"));
        assert!(rendered.contains("Selected position: 2.00"));
        assert!(
            rendered.contains("10 Range Market Data [ESZ6] hist=1 live=1"),
            "rendered dashboard:\n{rendered}"
        );
        assert!(rendered.contains("Strategy: Native Rust / HMA Crossover"));
        assert!(rendered.contains("Strategy Status: hydrated strategy"));
    }
}
