use crate::app::{self, App, EngineKey};
use crate::broker::{ServiceCommand, ServiceEvent};
use crate::cli::Cli;
use crate::config::AppConfig;
use crate::engine_control::{close_and_kill_engine, kill_engine_process};
use crate::engine_registry::list_running_engines;
use crate::engine_runtime::{self, EngineSession, unique_engine_socket_path};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineEntryMode {
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

pub(crate) async fn connect_or_spawn_engine(cli: &Cli) -> Result<EngineSession> {
    engine_runtime::connect_or_spawn_engine(
        cli.config.as_deref(),
        &cli.engine_socket,
        cli.no_spawn_engine,
    )
    .await
}

pub(crate) async fn connect_selected_engine(
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

pub(crate) async fn refresh_engine_overview(
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

pub(crate) async fn observe_running_engine_sessions(
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

pub(crate) fn spawn_engine_lifecycle_action(
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

pub(crate) fn engine_lifecycle_success_message(
    action: app::EngineLifecycleAction,
    id: u32,
) -> String {
    match action {
        app::EngineLifecycleAction::Kill => format!("Killed engine {id}."),
        app::EngineLifecycleAction::CloseAndKill => {
            format!("Closed the selected market and killed engine {id}.")
        }
    }
}

pub(crate) fn engine_lifecycle_failure_label(action: app::EngineLifecycleAction) -> &'static str {
    match action {
        app::EngineLifecycleAction::Kill => "Kill",
        app::EngineLifecycleAction::CloseAndKill => "Close and kill",
    }
}

pub(crate) fn enter_engine_session(
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
pub(crate) struct ObservedEngineSession {
    pub(crate) cmd_tx: mpsc::UnboundedSender<ServiceCommand>,
    _child: Option<tokio::process::Child>,
    _relay_task: JoinHandle<()>,
}

#[derive(Debug)]
pub(crate) struct EngineEventEnvelope {
    pub(crate) engine_key: EngineKey,
    pub(crate) event: ServiceEvent,
}

#[derive(Debug)]
pub(crate) enum EngineRelayMessage {
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

pub(crate) fn insert_observed_engine_session(
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
            _ => panic!("expected tagged service event first"),
        }

        match tokio::time::timeout(Duration::from_secs(1), relay_rx.recv())
            .await
            .expect("relay close timed out")
            .expect("expected relay close")
        {
            EngineRelayMessage::Closed { engine_key } => assert_eq!(engine_key, key),
            _ => panic!("expected relay close after sender drop"),
        }
    }
}
