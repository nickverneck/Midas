use crate::app::{self, App, EngineKey};
#[cfg(test)]
use crate::broker::service_event_channel;
use crate::broker::{ServiceCommand, ServiceCommandSender, ServiceEvent, ServiceEventReceiver};
use crate::cli::Cli;
use crate::config::AppConfig;
use crate::engine_control::{close_and_kill_engine, kill_engine_process};
use crate::engine_registry::list_running_engines;
use crate::engine_runtime::{self, EngineSession, unique_engine_socket_path};
use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

const ENGINE_RELAY_CONTROL_CAPACITY: usize = 256;

static NEXT_CONNECTION_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_RELAY_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineEntryMode {
    AttachExisting,
    CreateNew { mode: app::EngineCreateMode },
}

fn should_autoconnect_engine_session(
    config: &AppConfig,
    awaiting_broker_selection: bool,
    mode: EngineEntryMode,
) -> bool {
    config.autoconnect
        && matches!(
            mode,
            EngineEntryMode::CreateNew {
                mode: app::EngineCreateMode::Broker
            }
        )
        && !awaiting_broker_selection
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
    engine_event_tx: &Arc<EngineRelayQueue>,
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
) -> Result<(EngineKey, PathBuf, EngineEntryMode)> {
    match action {
        app::EngineSelectionAction::Attach {
            engine_key,
            socket_path,
        } => {
            // Always create a fresh IPC connection for an attach. An overview
            // observer may have buffered an older ReplayState; replacing it
            // gives the attach a new generation and makes those events unable
            // to satisfy hydration.
            let session = engine_runtime::connect_existing_engine(&socket_path).await?;
            insert_observed_engine_session(
                engine_sessions,
                engine_key.clone(),
                session,
                engine_event_tx,
            )
            .await;
            Ok((engine_key, socket_path, EngineEntryMode::AttachExisting))
        }
        app::EngineSelectionAction::CreateNew { mode } => {
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
            )
            .await;
            Ok((engine_key, socket_path, EngineEntryMode::CreateNew { mode }))
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
    engine_event_tx: &Arc<EngineRelayQueue>,
    dummy_cmd_tx: &ServiceCommandSender,
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
    engine_event_tx: &Arc<EngineRelayQueue>,
    dummy_cmd_tx: &ServiceCommandSender,
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
                )
                .await;
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
    engine_event_tx: &Arc<EngineRelayQueue>,
) {
    let engine_event_tx = engine_event_tx.clone();
    tokio::spawn(async move {
        let result = match action {
            app::EngineLifecycleAction::Kill => kill_engine_process(id).await,
            app::EngineLifecycleAction::CloseAndKill => close_and_kill_engine(id).await,
        }
        .map_err(|err| err.to_string());
        let _ =
            engine_event_tx.publish(EngineRelayMessage::LifecycleCompleted { action, id, result });
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
    let ui_mode = match mode {
        EngineEntryMode::AttachExisting => app.engine_create_mode_for_key(&engine_key),
        EngineEntryMode::CreateNew { mode } => mode,
    };
    app.enter_engine_session_for_key_with_mode(engine_key.clone(), socket_path, ui_mode);
    // `connect_existing_engine`/`spawn_and_connect_engine` already request
    // ReplayState before returning the session.  Sending it again here used
    // to duplicate a large snapshot burst exactly when a detached TUI was
    // reattached.  The relay mailbox below coalesces state, but avoiding the
    // duplicate request also makes the attach ordering deterministic.
    if should_autoconnect_engine_session(config, app.awaiting_broker_selection(), mode) {
        if let Some(session) = engine_sessions.get(&engine_key) {
            let _ = session.cmd_tx.send(ServiceCommand::Connect(config.clone()));
        }
    }
}

#[derive(Debug)]
pub(crate) struct ObservedEngineSession {
    pub(crate) cmd_tx: ServiceCommandSender,
    pub(crate) generation: u64,
    _child: Option<tokio::process::Child>,
    _relay_task: JoinHandle<()>,
}

impl ObservedEngineSession {
    /// Stop every task/process owned by this observation before it leaves the
    /// session map. Dropping `cmd_tx` closes the IPC command lifecycle; the
    /// client reader/writer tasks use that same lifecycle and therefore stop
    /// without changing the broker order path. The relay is explicitly
    /// aborted and awaited, and a spawned engine is killed and reaped.
    pub(crate) async fn shutdown(self) {
        let ObservedEngineSession {
            cmd_tx,
            _child,
            _relay_task,
            ..
        } = self;

        drop(cmd_tx);
        _relay_task.abort();
        let _ = _relay_task.await;

        if let Some(mut child) = _child {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

#[derive(Debug)]
pub(crate) struct EngineEventEnvelope {
    pub(crate) engine_key: EngineKey,
    pub(crate) generation: u64,
    pub(crate) sequence: u64,
    pub(crate) event: ServiceEvent,
}

#[derive(Debug)]
pub(crate) enum EngineRelayMessage {
    Event(EngineEventEnvelope),
    Closed {
        engine_key: EngineKey,
        generation: u64,
    },
    RelayOverflow {
        engine_key: Option<EngineKey>,
        generation: Option<u64>,
        reason: String,
    },
    LifecycleCompleted {
        action: app::EngineLifecycleAction,
        id: u32,
        result: Result<(), String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LatestEventKind {
    Market,
    TradeMarkers,
    EngineHistory,
    Latency,
    ExecutionState,
    ExecutionProbe,
    ReplaySpeed,
    ReplayLedger,
    ReplayProgress,
}

#[derive(Debug)]
struct QueuedRelayMessage {
    sequence: u64,
    message: EngineRelayMessage,
}

#[derive(Default)]
struct RelayQueueState {
    control: VecDeque<QueuedRelayMessage>,
    latest: HashMap<(EngineKey, u64, LatestEventKind), QueuedRelayMessage>,
    active_generations: HashMap<EngineKey, u64>,
}

/// TUI-facing relay mailbox.  State observations have one slot per engine,
/// generation, and event kind; ordered control events retain bounded FIFO
/// delivery.  This keeps a detached/re-attached TUI from replaying hours of
/// stale snapshots while still preserving fills, errors, and lifecycle data.
pub(crate) struct EngineRelayQueue {
    state: Mutex<RelayQueueState>,
    notify: Notify,
    generation_notify: Notify,
}

impl EngineRelayQueue {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(RelayQueueState::default()),
            notify: Notify::new(),
            generation_notify: Notify::new(),
        })
    }

    pub(crate) fn begin_connection(&self, engine_key: EngineKey) -> u64 {
        let generation = NEXT_CONNECTION_GENERATION.fetch_add(1, Ordering::Relaxed);
        let mut state = self.state.lock().expect("engine relay mutex poisoned");
        let replaced_generation = state
            .active_generations
            .insert(engine_key.clone(), generation);
        state
            .control
            .retain(|queued| !message_belongs_to(&queued.message, &engine_key, None));
        state.latest.retain(|(key, _, _), _| key != &engine_key);
        drop(state);
        self.notify.notify_one();
        if replaced_generation.is_some() {
            self.generation_notify.notify_waiters();
        }
        generation
    }

    pub(crate) fn invalidate_connection(&self, engine_key: &EngineKey, generation: u64) -> bool {
        let mut state = self.state.lock().expect("engine relay mutex poisoned");
        if state.active_generations.get(engine_key) != Some(&generation) {
            return false;
        }

        state.active_generations.remove(engine_key);
        state
            .control
            .retain(|queued| !message_belongs_to(&queued.message, engine_key, Some(generation)));
        state.latest.retain(|(key, queued_generation, _), _| {
            key != engine_key || *queued_generation != generation
        });
        drop(state);
        self.notify.notify_one();
        self.generation_notify.notify_waiters();
        true
    }

    pub(crate) fn publish(&self, message: EngineRelayMessage) -> bool {
        let mut state = self.state.lock().expect("engine relay mutex poisoned");
        let mut generation_invalidated = false;
        match message {
            EngineRelayMessage::Event(envelope) => {
                if state.active_generations.get(&envelope.engine_key) != Some(&envelope.generation)
                {
                    return false;
                }
                let engine_key = envelope.engine_key.clone();
                let generation = envelope.generation;
                let kind = latest_event_kind_for_service_event(&envelope.event);
                let queued = QueuedRelayMessage {
                    sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                    message: EngineRelayMessage::Event(envelope),
                };
                if let Some(kind) = kind {
                    state.latest.insert((engine_key, generation, kind), queued);
                } else if !push_control(&mut state, queued) {
                    force_overflow(&mut state, Some(engine_key), Some(generation));
                    generation_invalidated = true;
                }
            }
            EngineRelayMessage::Closed {
                engine_key,
                generation,
            } => {
                if state.active_generations.get(&engine_key) != Some(&generation) {
                    return false;
                }
                state.active_generations.remove(&engine_key);
                generation_invalidated = true;
                state.latest.retain(|(key, event_generation, _), _| {
                    key != &engine_key || *event_generation != generation
                });
                let queued = QueuedRelayMessage {
                    sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                    message: EngineRelayMessage::Closed {
                        engine_key: engine_key.clone(),
                        generation,
                    },
                };
                if !push_control(&mut state, queued) {
                    force_overflow(&mut state, Some(engine_key), Some(generation));
                }
            }
            message @ (EngineRelayMessage::RelayOverflow { .. }
            | EngineRelayMessage::LifecycleCompleted { .. }) => {
                let queued = QueuedRelayMessage {
                    sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                    message,
                };
                if !push_control(&mut state, queued) {
                    force_overflow(&mut state, None, None);
                    generation_invalidated = true;
                }
            }
        }
        drop(state);
        self.notify.notify_one();
        if generation_invalidated {
            self.generation_notify.notify_waiters();
        }
        true
    }

    pub(crate) fn try_recv(&self) -> Option<EngineRelayMessage> {
        let mut state = self.state.lock().expect("engine relay mutex poisoned");
        let latest_key = state
            .latest
            .iter()
            .min_by_key(|(_, queued)| queued.sequence)
            .map(|(key, _)| key.clone());
        let take_latest = match (state.control.front(), latest_key.as_ref()) {
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (Some(control), Some(key)) => {
                control.sequence > state.latest.get(key).expect("latest key exists").sequence
            }
            (None, None) => return None,
        };
        if take_latest {
            state
                .latest
                .remove(&latest_key.expect("latest key exists"))
                .map(|queued| queued.message)
        } else {
            state.control.pop_front().map(|queued| queued.message)
        }
    }

    pub(crate) async fn recv(&self) -> EngineRelayMessage {
        loop {
            if let Some(message) = self.try_recv() {
                return message;
            }
            self.notify.notified().await;
        }
    }

    fn is_generation_active(&self, engine_key: &EngineKey, generation: u64) -> bool {
        self.state
            .lock()
            .expect("engine relay mutex poisoned")
            .active_generations
            .get(engine_key)
            == Some(&generation)
    }

    async fn wait_for_generation_invalidation(&self, engine_key: &EngineKey, generation: u64) {
        loop {
            let notified = self.generation_notify.notified();
            if !self.is_generation_active(engine_key, generation) {
                return;
            }
            notified.await;
        }
    }
}

fn push_control(state: &mut RelayQueueState, message: QueuedRelayMessage) -> bool {
    if state.control.len() >= ENGINE_RELAY_CONTROL_CAPACITY {
        return false;
    }
    state.control.push_back(message);
    true
}

fn force_overflow(
    state: &mut RelayQueueState,
    engine_key: Option<EngineKey>,
    generation: Option<u64>,
) {
    if let Some(engine_key) = engine_key {
        // Make the producing relay observe an invalid generation on its next
        // message and stop. Remove only this generation's queued messages so
        // another engine's fills, account events, and control messages remain
        // available to the TUI.
        if state.active_generations.get(&engine_key) == generation.as_ref() {
            state.active_generations.remove(&engine_key);
        }
        state
            .control
            .retain(|queued| !message_belongs_to(&queued.message, &engine_key, generation));
        state.latest.retain(|(key, queued_generation, _), _| {
            key != &engine_key || Some(*queued_generation) != generation
        });

        let notification = QueuedRelayMessage {
            sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
            message: EngineRelayMessage::RelayOverflow {
                engine_key: Some(engine_key),
                generation,
                reason: "TUI relay control queue overflow; engine observation disconnected"
                    .to_string(),
            },
        };
        if push_control(state, notification) {
            return;
        }

        // The remaining control queue is full of other engines' messages. A
        // per-engine notification cannot be admitted without evicting them,
        // so escalate to an explicit relay-wide invalidation/error instead of
        // silently dropping unaffected messages.
    }

    force_relay_wide_overflow(state);
}

fn force_relay_wide_overflow(state: &mut RelayQueueState) {
    state.active_generations.clear();
    state.control.clear();
    state.latest.clear();
    state.control.push_back(QueuedRelayMessage {
        sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        message: EngineRelayMessage::RelayOverflow {
            engine_key: None,
            generation: None,
            reason: "TUI relay control queue overflow".to_string(),
        },
    });
}

fn message_belongs_to(
    message: &EngineRelayMessage,
    engine_key: &EngineKey,
    generation: Option<u64>,
) -> bool {
    match message {
        EngineRelayMessage::Event(envelope) => {
            &envelope.engine_key == engine_key
                && generation.is_none_or(|expected| expected == envelope.generation)
        }
        EngineRelayMessage::Closed {
            engine_key: message_key,
            generation: message_generation,
        } => {
            message_key == engine_key
                && generation.is_none_or(|expected| expected == *message_generation)
        }
        EngineRelayMessage::RelayOverflow {
            engine_key: Some(message_key),
            generation: message_generation,
            ..
        } => {
            message_key == engine_key
                && generation.is_none_or(|expected| Some(expected) == *message_generation)
        }
        _ => false,
    }
}

fn latest_event_kind_for_service_event(event: &ServiceEvent) -> Option<LatestEventKind> {
    Some(match event {
        // Account snapshots contain balance/fill deltas consumed by F6. Keep
        // them in the bounded lossless stream so coalescing cannot rewrite
        // session statistics. The engine IPC layer already publishes the
        // latest account snapshot to a newly attached client.
        ServiceEvent::AccountSnapshotsLoaded(_) => return None,
        ServiceEvent::MarketSnapshot(_) => LatestEventKind::Market,
        ServiceEvent::TradeMarkersUpdated(_) => LatestEventKind::TradeMarkers,
        ServiceEvent::EngineHistoryUpdated(_) => LatestEventKind::EngineHistory,
        ServiceEvent::Latency(_) => LatestEventKind::Latency,
        ServiceEvent::ExecutionState(_) => LatestEventKind::ExecutionState,
        ServiceEvent::ExecutionProbe(_) => LatestEventKind::ExecutionProbe,
        ServiceEvent::ReplaySpeedUpdated(_) => LatestEventKind::ReplaySpeed,
        ServiceEvent::ReplayExecutionLedgerUpdated(_)
        | ServiceEvent::ReplayExecutionLedgerSnapshot(_) => LatestEventKind::ReplayLedger,
        ServiceEvent::ReplayDownloadProgress { .. } => LatestEventKind::ReplayProgress,
        _ => return None,
    })
}

pub(crate) async fn insert_observed_engine_session(
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
    engine_key: EngineKey,
    session: EngineSession,
    engine_event_tx: &Arc<EngineRelayQueue>,
) {
    if let Some(previous) = engine_sessions.remove(&engine_key) {
        previous.shutdown().await;
    }
    let generation = engine_event_tx.begin_connection(engine_key.clone());
    let relay_task = spawn_engine_event_relay(
        engine_key.clone(),
        generation,
        session.event_rx,
        engine_event_tx.clone(),
    );
    engine_sessions.insert(
        engine_key,
        ObservedEngineSession {
            cmd_tx: session.cmd_tx,
            generation,
            _child: session.child,
            _relay_task: relay_task,
        },
    );
}

pub(crate) async fn shutdown_observed_engine_sessions(
    engine_sessions: &mut HashMap<EngineKey, ObservedEngineSession>,
) {
    let sessions = std::mem::take(engine_sessions);
    for (_, session) in sessions {
        session.shutdown().await;
    }
}

fn spawn_engine_event_relay(
    engine_key: EngineKey,
    generation: u64,
    mut event_rx: ServiceEventReceiver,
    engine_event_tx: Arc<EngineRelayQueue>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let next_event = tokio::select! {
                event = event_rx.recv() => event,
                _ = engine_event_tx.wait_for_generation_invalidation(&engine_key, generation) => {
                    return;
                }
            };
            let Some(event) = next_event else {
                break;
            };
            let message = EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: engine_key.clone(),
                generation,
                sequence: NEXT_RELAY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                event,
            });
            if !engine_event_tx.publish(message) {
                return;
            }
        }
        let _ = engine_event_tx.publish(EngineRelayMessage::Closed {
            engine_key,
            generation,
        });
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
            EngineEntryMode::CreateNew {
                mode: app::EngineCreateMode::Broker,
            }
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
            EngineEntryMode::CreateNew {
                mode: app::EngineCreateMode::Broker,
            }
        ));
    }

    #[tokio::test]
    async fn engine_event_relay_tags_events_and_reports_closed() {
        let key =
            EngineKey::from_socket_path(PathBuf::from("/tmp/trader-engine-99.sock").as_path());
        let (service_tx, service_rx) = service_event_channel(8);
        let relay = EngineRelayQueue::new();
        let generation = relay.begin_connection(key.clone());
        let _relay_task =
            spawn_engine_event_relay(key.clone(), generation, service_rx, relay.clone());

        service_tx
            .send(ServiceEvent::Status("ready".to_string()))
            .expect("send service event");
        drop(service_tx);

        match tokio::time::timeout(Duration::from_secs(1), relay.recv())
            .await
            .expect("relay event timed out")
        {
            EngineRelayMessage::Event(envelope) => {
                assert_eq!(envelope.engine_key, key);
                assert_eq!(envelope.generation, generation);
                assert!(
                    matches!(envelope.event, ServiceEvent::Status(message) if message == "ready")
                );
            }
            _ => panic!("expected tagged service event first"),
        }

        match tokio::time::timeout(Duration::from_secs(1), relay.recv()).await {
            Ok(EngineRelayMessage::Closed {
                engine_key,
                generation: closed_generation,
            }) => {
                assert_eq!(engine_key, key);
                assert_eq!(closed_generation, generation);
            }
            _ => panic!("expected relay close after sender drop"),
        }
    }

    #[tokio::test]
    async fn relay_latest_state_replaces_a_long_detached_stream() {
        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/relay-detach.sock").as_path());
        let relay = EngineRelayQueue::new();
        let generation = relay.begin_connection(key.clone());

        for index in 0..10_000 {
            assert!(
                relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: key.clone(),
                    generation,
                    sequence: index + 1,
                    event: ServiceEvent::Latency(crate::broker::LatencySnapshot {
                        last_order_ack_ms: Some(index as u64),
                        ..Default::default()
                    }),
                }))
            );
        }

        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::Latency(snapshot),
                ..
            })) if snapshot.last_order_ack_ms == Some(9999)
        ));
        assert!(relay.try_recv().is_none());
    }

    #[tokio::test]
    async fn reconnect_generation_invalidates_old_relay_events() {
        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/relay-reconnect.sock").as_path());
        let relay = EngineRelayQueue::new();
        let old_generation = relay.begin_connection(key.clone());
        let new_generation = relay.begin_connection(key.clone());

        assert!(
            !relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation: old_generation,
                sequence: 1,
                event: ServiceEvent::DebugLog("old".to_string()),
            }))
        );
        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation: new_generation,
                sequence: 2,
                event: ServiceEvent::DebugLog("new".to_string()),
            }))
        );
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                generation,
                event: ServiceEvent::DebugLog(message),
                ..
            })) if generation == new_generation && message == "new"
        ));
    }

    #[tokio::test]
    async fn invalidated_relay_stops_without_waiting_for_another_event() {
        let key =
            EngineKey::from_socket_path(PathBuf::from("/tmp/relay-invalidate.sock").as_path());
        let relay = EngineRelayQueue::new();
        let generation = relay.begin_connection(key.clone());
        let (event_tx, event_rx) = service_event_channel(1);
        let relay_task = spawn_engine_event_relay(key.clone(), generation, event_rx, relay.clone());

        assert!(relay.invalidate_connection(&key, generation));
        tokio::time::timeout(Duration::from_secs(1), relay_task)
            .await
            .expect("invalidated relay did not stop")
            .expect("invalidated relay task panicked");
        drop(event_tx);
    }

    #[test]
    fn fresh_generation_clears_stale_reattach_state() {
        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/relay-fresh.sock").as_path());
        let relay = EngineRelayQueue::new();
        let old_generation = relay.begin_connection(key.clone());

        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation: old_generation,
                sequence: 1,
                event: ServiceEvent::ExecutionState(
                    crate::strategy::ExecutionStateSnapshot::default()
                ),
            }))
        );
        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation: old_generation,
                sequence: 2,
                event: ServiceEvent::Status("stale attach state".to_string()),
            }))
        );

        let new_generation = relay.begin_connection(key.clone());
        assert_ne!(old_generation, new_generation);
        assert!(relay.try_recv().is_none());
        assert!(
            !relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation: old_generation,
                sequence: 3,
                event: ServiceEvent::Status("old generation".to_string()),
            }))
        );
        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key,
                generation: new_generation,
                sequence: 4,
                event: ServiceEvent::Status("fresh attach state".to_string()),
            }))
        );
    }

    #[tokio::test]
    async fn relay_keeps_control_events_fifo() {
        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/relay-control.sock").as_path());
        let relay = EngineRelayQueue::new();
        let generation = relay.begin_connection(key.clone());
        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key.clone(),
                generation,
                sequence: 1,
                event: ServiceEvent::Status("first".to_string()),
            }))
        );
        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: key,
                generation,
                sequence: 2,
                event: ServiceEvent::Status("second".to_string()),
            }))
        );

        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::Status(message), ..
            })) if message == "first"
        ));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::Status(message), ..
            })) if message == "second"
        ));
    }

    #[tokio::test]
    async fn per_engine_overflow_preserves_other_engine_control_events() {
        let affected_key = EngineKey::from_socket_path(
            PathBuf::from("/tmp/relay-overflow-affected.sock").as_path(),
        );
        let unaffected_key = EngineKey::from_socket_path(
            PathBuf::from("/tmp/relay-overflow-unaffected.sock").as_path(),
        );
        let relay = EngineRelayQueue::new();
        let affected_generation = relay.begin_connection(affected_key.clone());
        let unaffected_generation = relay.begin_connection(unaffected_key.clone());

        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: unaffected_key.clone(),
                generation: unaffected_generation,
                sequence: 1,
                event: ServiceEvent::Status("unaffected before overflow".to_string()),
            }))
        );
        for index in 0..(ENGINE_RELAY_CONTROL_CAPACITY - 1) {
            assert!(
                relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: affected_key.clone(),
                    generation: affected_generation,
                    sequence: index as u64 + 2,
                    event: ServiceEvent::Status(format!("affected {index}")),
                }))
            );
        }

        let invalidation_relay = relay.clone();
        let invalidation_key = affected_key.clone();
        let invalidation = tokio::spawn(async move {
            invalidation_relay
                .wait_for_generation_invalidation(&invalidation_key, affected_generation)
                .await;
        });
        tokio::task::yield_now().await;

        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: affected_key.clone(),
                generation: affected_generation,
                sequence: ENGINE_RELAY_CONTROL_CAPACITY as u64 + 1,
                event: ServiceEvent::Status("causes overflow".to_string()),
            }))
        );
        tokio::time::timeout(Duration::from_secs(1), invalidation)
            .await
            .expect("affected generation invalidation timed out")
            .expect("invalidation waiter panicked");

        assert!(!relay.is_generation_active(&affected_key, affected_generation));
        assert!(relay.is_generation_active(&unaffected_key, unaffected_generation));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key,
                event: ServiceEvent::Status(message),
                ..
            })) if engine_key == unaffected_key && message == "unaffected before overflow"
        ));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::RelayOverflow {
                engine_key: Some(engine_key),
                generation: Some(generation),
                ..
            }) if engine_key == affected_key && generation == affected_generation
        ));

        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: unaffected_key.clone(),
                generation: unaffected_generation,
                sequence: 10_000,
                event: ServiceEvent::Status("unaffected after overflow".to_string()),
            }))
        );
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::Status(message), ..
            })) if message == "unaffected after overflow"
        ));
    }

    #[test]
    fn per_engine_overflow_escalates_when_other_engines_fill_control_queue() {
        let affected_key = EngineKey::from_socket_path(
            PathBuf::from("/tmp/relay-overflow-global-a.sock").as_path(),
        );
        let unaffected_key = EngineKey::from_socket_path(
            PathBuf::from("/tmp/relay-overflow-global-b.sock").as_path(),
        );
        let relay = EngineRelayQueue::new();
        let affected_generation = relay.begin_connection(affected_key.clone());
        let unaffected_generation = relay.begin_connection(unaffected_key.clone());

        for index in 0..ENGINE_RELAY_CONTROL_CAPACITY {
            assert!(
                relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: unaffected_key.clone(),
                    generation: unaffected_generation,
                    sequence: index as u64 + 1,
                    event: ServiceEvent::Status(format!("unaffected {index}")),
                }))
            );
        }

        assert!(
            relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                engine_key: affected_key.clone(),
                generation: affected_generation,
                sequence: ENGINE_RELAY_CONTROL_CAPACITY as u64 + 1,
                event: ServiceEvent::Status("causes relay-wide overflow".to_string()),
            }))
        );

        assert!(!relay.is_generation_active(&affected_key, affected_generation));
        assert!(!relay.is_generation_active(&unaffected_key, unaffected_generation));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::RelayOverflow {
                engine_key: None,
                generation: None,
                reason,
            }) if reason == "TUI relay control queue overflow"
        ));
        assert!(relay.try_recv().is_none());
    }

    #[tokio::test]
    async fn failed_attach_cleanup_requests_child_kill() {
        let (cmd_tx, _cmd_rx) = crate::broker::service_command_channel(1);
        let (_event_tx, event_rx) = service_event_channel(1);
        let relay_task = tokio::spawn(async move {
            let mut event_rx = event_rx;
            let _ = event_rx.recv().await;
        });
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn test child");

        let session = ObservedEngineSession {
            cmd_tx,
            generation: 7,
            _child: Some(child),
            _relay_task: relay_task,
        };

        session.shutdown().await;
    }

    #[tokio::test]
    async fn replacing_session_reaps_the_owned_child() {
        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/replace-session.sock").as_path());
        let relay = EngineRelayQueue::new();
        let mut sessions = HashMap::new();
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn test child");
        let child_pid = child.id().expect("test child pid");
        let (old_cmd_tx, _old_cmd_rx) = crate::broker::service_command_channel(1);
        let (_old_event_tx, old_event_rx) = service_event_channel(1);

        insert_observed_engine_session(
            &mut sessions,
            key.clone(),
            EngineSession {
                child: Some(child),
                cmd_tx: old_cmd_tx,
                event_rx: old_event_rx,
            },
            &relay,
        )
        .await;

        let (new_cmd_tx, _new_cmd_rx) = crate::broker::service_command_channel(1);
        let (_new_event_tx, new_event_rx) = service_event_channel(1);
        insert_observed_engine_session(
            &mut sessions,
            key,
            EngineSession {
                child: None,
                cmd_tx: new_cmd_tx,
                event_rx: new_event_rx,
            },
            &relay,
        )
        .await;

        assert_eq!(sessions.len(), 1);
        assert_ne!(unsafe { libc::kill(child_pid as i32, 0) }, 0);
        shutdown_observed_engine_sessions(&mut sessions).await;
    }

    #[test]
    fn relay_keeps_fill_logs_and_account_snapshots_lossless() {
        assert!(
            latest_event_kind_for_service_event(&ServiceEvent::DebugLog("fill detail".to_string()))
                .is_none()
        );
        assert!(
            latest_event_kind_for_service_event(&ServiceEvent::AccountSnapshotsLoaded(Vec::new()))
                .is_none()
        );

        let key = EngineKey::from_socket_path(PathBuf::from("/tmp/relay-critical.sock").as_path());
        let relay = EngineRelayQueue::new();
        let generation = relay.begin_connection(key.clone());
        for event in [
            ServiceEvent::DebugLog("fill detail".to_string()),
            ServiceEvent::Status("protection acknowledged".to_string()),
            ServiceEvent::AccountSnapshotsLoaded(Vec::new()),
        ] {
            assert!(
                relay.publish(EngineRelayMessage::Event(EngineEventEnvelope {
                    engine_key: key.clone(),
                    generation,
                    sequence: 1,
                    event,
                }))
            );
        }

        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::DebugLog(message), ..
            })) if message == "fill detail"
        ));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::Status(message), ..
            })) if message == "protection acknowledged"
        ));
        assert!(matches!(
            relay.try_recv(),
            Some(EngineRelayMessage::Event(EngineEventEnvelope {
                event: ServiceEvent::AccountSnapshotsLoaded(_),
                ..
            }))
        ));
    }
}
