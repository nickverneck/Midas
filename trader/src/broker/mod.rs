mod types;

pub use types::*;

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{
    mpsc::{self, Receiver, Sender},
    watch,
};
use tokio::task::JoinHandle;

#[cfg(not(any(feature = "tradovate", feature = "ironbeam")))]
compile_error!("`trader` requires at least one broker feature enabled.");

pub fn compiled_brokers() -> &'static [BrokerKind] {
    #[cfg(all(feature = "tradovate", feature = "ironbeam"))]
    {
        &[BrokerKind::Tradovate, BrokerKind::Ironbeam]
    }
    #[cfg(all(feature = "tradovate", not(feature = "ironbeam")))]
    {
        &[BrokerKind::Tradovate]
    }
    #[cfg(all(feature = "ironbeam", not(feature = "tradovate")))]
    {
        &[BrokerKind::Ironbeam]
    }
}

pub fn default_broker() -> BrokerKind {
    compiled_brokers()
        .first()
        .copied()
        .expect("at least one broker feature must be enabled")
}

pub fn supports_broker(kind: BrokerKind) -> bool {
    compiled_brokers().contains(&kind)
}

struct ActiveBrokerService {
    kind: BrokerKind,
    cmd_tx: ServiceCommandSender,
    task: JoinHandle<()>,
}

impl ActiveBrokerService {
    async fn shutdown(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

/// Synchronous, bounded command handle used by the engine and its direct
/// callers.  The existing call sites intentionally submit commands from
/// non-async UI code, so this wrapper exposes the same synchronous `send`
/// shape while making queue admission explicit through `TrySendError`.
#[derive(Clone, Debug)]
pub(crate) struct ServiceCommandSender {
    inner: Sender<ServiceCommand>,
}

impl ServiceCommandSender {
    pub(crate) fn send(
        &self,
        command: ServiceCommand,
    ) -> Result<(), mpsc::error::TrySendError<ServiceCommand>> {
        self.inner.try_send(command)
    }
}

pub(crate) type ServiceCommandReceiver = Receiver<ServiceCommand>;

#[derive(Debug, Default)]
struct ServiceEventOverflow {
    dropped: AtomicU64,
    notify: tokio::sync::Notify,
}

/// Bounded event handle with observable overflow.  Event producers are mostly
/// synchronous state-machine code, therefore `send` is deliberately a
/// non-blocking `try_send`.  A full queue is never silently treated as normal:
/// the failed send is returned and the shared overflow counter wakes the IPC
/// supervisor, which disconnects affected clients explicitly.
#[derive(Clone, Debug)]
pub(crate) struct ServiceEventSender {
    inner: Sender<ServiceEvent>,
    overflow: Arc<ServiceEventOverflow>,
}

impl ServiceEventSender {
    pub(crate) fn send(
        &self,
        event: ServiceEvent,
    ) -> Result<(), mpsc::error::TrySendError<ServiceEvent>> {
        match self.inner.try_send(event) {
            Ok(()) => Ok(()),
            Err(error @ TrySendError::Full(_)) => {
                self.overflow.dropped.fetch_add(1, Ordering::Relaxed);
                self.overflow.notify.notify_one();
                Err(error)
            }
            Err(error @ TrySendError::Closed(_)) => Err(error),
        }
    }

    pub(crate) fn take_overflow_count(&self) -> u64 {
        self.overflow.dropped.swap(0, Ordering::AcqRel)
    }

    pub(crate) async fn overflow_notified(&self) {
        self.overflow.notify.notified().await;
    }
}

pub(crate) type ServiceEventReceiver = Receiver<ServiceEvent>;

pub(crate) fn service_command_channel(
    capacity: usize,
) -> (ServiceCommandSender, ServiceCommandReceiver) {
    let (inner, receiver) = mpsc::channel(capacity);
    (ServiceCommandSender { inner }, receiver)
}

pub(crate) fn service_event_channel(capacity: usize) -> (ServiceEventSender, ServiceEventReceiver) {
    let (inner, receiver) = mpsc::channel(capacity);
    (
        ServiceEventSender {
            inner,
            overflow: Arc::new(ServiceEventOverflow::default()),
        },
        receiver,
    )
}

pub(crate) const SERVICE_COMMAND_QUEUE_CAPACITY: usize = 128;
pub(crate) const SERVICE_EVENT_QUEUE_CAPACITY: usize = 256;
const BACKEND_COMMAND_QUEUE_CAPACITY: usize = 128;

pub async fn service_loop(
    cmd_rx: ServiceCommandReceiver,
    event_tx: ServiceEventSender,
    market_tx: watch::Sender<MarketSnapshot>,
) {
    service_loop_with_backend_factory(cmd_rx, event_tx, market_tx, spawn_backend_service).await;
}

async fn service_loop_with_backend_factory<F>(
    mut cmd_rx: ServiceCommandReceiver,
    event_tx: ServiceEventSender,
    market_tx: watch::Sender<MarketSnapshot>,
    mut spawn_backend: F,
) where
    F: FnMut(BrokerKind, ServiceEventSender, watch::Sender<MarketSnapshot>) -> ActiveBrokerService,
{
    let mut active = None::<ActiveBrokerService>;

    loop {
        let next = if let Some(active_service) = active.as_mut() {
            tokio::select! {
                biased;
                task_result = &mut active_service.task => {
                    ServiceLoopInput::BackendTerminated {
                        kind: active_service.kind,
                        task_result,
                    }
                }
                _ = event_tx.overflow_notified() => ServiceLoopInput::EventOverflow,
                cmd = cmd_rx.recv() => ServiceLoopInput::Command(cmd),
            }
        } else {
            tokio::select! {
                biased;
                _ = event_tx.overflow_notified() => ServiceLoopInput::EventOverflow,
                cmd = cmd_rx.recv() => ServiceLoopInput::Command(cmd),
            }
        };

        let ServiceLoopInput::Command(Some(cmd)) = next else {
            match next {
                ServiceLoopInput::BackendTerminated { kind, task_result } => {
                    let reason = match task_result {
                        Ok(()) => "exited unexpectedly".to_string(),
                        Err(error) if error.is_panic() => "panicked".to_string(),
                        Err(error) => format!("terminated ({error})"),
                    };
                    let _ = event_tx.send(ServiceEvent::Error(format!(
                        "{} backend service {reason}; disconnecting.",
                        kind.label()
                    )));
                    // `task_result` was produced by polling this handle to
                    // completion. It must be dropped, not awaited again.
                    active.take();
                    let _ = market_tx.send(MarketSnapshot::default());
                }
                ServiceLoopInput::EventOverflow => {}
                ServiceLoopInput::Command(None) => {}
                ServiceLoopInput::Command(Some(_)) => unreachable!(),
            }
            break;
        };
        if let Some(kind) = command_broker(cmd_ref(&cmd)) {
            if !supports_broker(kind) {
                let _ = event_tx.send(ServiceEvent::Error(format!(
                    "{} support is not enabled in this build",
                    kind.label()
                )));
                continue;
            }
            let needs_restart = active
                .as_ref()
                .map(|service| service.kind != kind)
                .unwrap_or(true);
            if needs_restart {
                if let Some(previous) = active.take() {
                    previous.shutdown().await;
                }
                active = Some(spawn_backend(kind, event_tx.clone(), market_tx.clone()));
            }
        }

        if active.is_none() {
            match cmd {
                ServiceCommand::InspectState | ServiceCommand::ReplayState => {
                    let _ = event_tx.send(ServiceEvent::Disconnected);
                }
                _ => {
                    let _ = event_tx.send(ServiceEvent::Error(
                        "Select a broker and connect first.".to_string(),
                    ));
                }
            }
            continue;
        }

        let send_result = if let Some(active_service) = active.as_ref() {
            let kind = active_service.kind;
            Some((active_service.cmd_tx.send(cmd), kind))
        } else {
            None
        };
        if let Some((Err(_), kind)) = send_result {
            let _ = event_tx.send(ServiceEvent::Error(format!(
                "{} engine is unavailable; reconnect to restart it.",
                kind.label()
            )));
            if let Some(previous) = active.take() {
                previous.shutdown().await;
            }
            let _ = market_tx.send(MarketSnapshot::default());
        }
    }

    if let Some(previous) = active.take() {
        previous.shutdown().await;
    }
}

enum ServiceLoopInput {
    Command(Option<ServiceCommand>),
    EventOverflow,
    BackendTerminated {
        kind: BrokerKind,
        task_result: Result<(), tokio::task::JoinError>,
    },
}

fn command_broker(cmd: &ServiceCommand) -> Option<BrokerKind> {
    match cmd {
        ServiceCommand::Connect(cfg) => Some(cfg.broker),
        ServiceCommand::EnterReplayMode { config, .. } => Some(config.broker),
        #[cfg(feature = "replay")]
        ServiceCommand::EnterReplayModeWithSharedFrames { config, .. } => Some(config.broker),
        _ => None,
    }
}

fn cmd_ref(cmd: &ServiceCommand) -> &ServiceCommand {
    cmd
}

fn spawn_backend_service(
    kind: BrokerKind,
    event_tx: ServiceEventSender,
    market_tx: watch::Sender<MarketSnapshot>,
) -> ActiveBrokerService {
    let (backend_cmd_tx, backend_cmd_rx) = service_command_channel(BACKEND_COMMAND_QUEUE_CAPACITY);
    let task = match kind {
        BrokerKind::Tradovate => {
            #[cfg(feature = "tradovate")]
            {
                tokio::spawn(crate::tradovate::service_loop(
                    backend_cmd_rx,
                    event_tx,
                    market_tx,
                ))
            }
            #[cfg(not(feature = "tradovate"))]
            unreachable!("broker feature gating should prevent spawning disabled Tradovate backend")
        }
        BrokerKind::Ironbeam => {
            #[cfg(feature = "ironbeam")]
            {
                tokio::spawn(crate::ironbeam::service_loop(
                    backend_cmd_rx,
                    event_tx,
                    market_tx,
                ))
            }
            #[cfg(not(feature = "ironbeam"))]
            unreachable!("broker feature gating should prevent spawning disabled Ironbeam backend")
        }
    };

    ActiveBrokerService {
        kind,
        cmd_tx: backend_cmd_tx,
        task,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use std::sync::{Arc, Mutex};
    use tokio::sync::{Notify, oneshot};
    use tokio::time::{Duration, timeout};

    struct DropSignal {
        dropped: Arc<Mutex<Vec<BrokerKind>>>,
        kind: BrokerKind,
        notify: Arc<Notify>,
    }

    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.dropped
                .lock()
                .expect("drop signal lock")
                .push(self.kind);
            self.notify.notify_one();
        }
    }

    struct AbortSignal(Option<oneshot::Sender<()>>);

    impl Drop for AbortSignal {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    fn connect_command(kind: BrokerKind) -> ServiceCommand {
        ServiceCommand::Connect(AppConfig {
            broker: kind,
            ..AppConfig::default()
        })
    }

    #[tokio::test]
    async fn service_loop_reports_backend_termination_without_followup_command() {
        let kind = default_broker();
        let (cmd_tx, cmd_rx) = service_command_channel(2);
        let (event_tx, mut event_rx) = service_event_channel(2);
        let (market_tx, _) = watch::channel(MarketSnapshot::default());

        let service_task = tokio::spawn(service_loop_with_backend_factory(
            cmd_rx,
            event_tx,
            market_tx,
            move |kind, _event_tx, _market_tx| {
                let (backend_cmd_tx, mut backend_cmd_rx) = service_command_channel(1);
                let task = tokio::spawn(async move {
                    let _ = backend_cmd_rx.recv().await;
                });
                ActiveBrokerService {
                    kind,
                    cmd_tx: backend_cmd_tx,
                    task,
                }
            },
        ));

        cmd_tx
            .send(connect_command(kind))
            .expect("connect command fits");

        let event = timeout(Duration::from_secs(1), event_rx.recv())
            .await
            .expect("backend termination event timeout")
            .expect("backend termination event");
        assert!(matches!(
            event,
            ServiceEvent::Error(message)
                if message.contains("backend service exited unexpectedly")
                    && message.contains(kind.label())
        ));
        timeout(Duration::from_secs(1), service_task)
            .await
            .expect("service loop termination timeout")
            .expect("service loop task failed");
    }

    #[tokio::test]
    async fn service_loop_aborts_backend_cleanly_when_command_channel_closes() {
        let kind = default_broker();
        let (cmd_tx, cmd_rx) = service_command_channel(2);
        let (backend_started_tx, backend_started_rx) = oneshot::channel();
        let (backend_dropped_tx, backend_dropped_rx) = oneshot::channel();
        let (event_tx, _event_rx) = service_event_channel(2);
        let (market_tx, _) = watch::channel(MarketSnapshot::default());
        let mut backend_started_tx = Some(backend_started_tx);
        let mut backend_dropped_tx = Some(backend_dropped_tx);

        let service_task = tokio::spawn(service_loop_with_backend_factory(
            cmd_rx,
            event_tx,
            market_tx,
            move |kind, _event_tx, _market_tx| {
                let (backend_cmd_tx, mut backend_cmd_rx) = service_command_channel(1);
                let started = backend_started_tx
                    .take()
                    .expect("backend factory called once");
                let dropped = backend_dropped_tx
                    .take()
                    .expect("backend factory called once");
                let task = tokio::spawn(async move {
                    let _backend_cmd_rx = &mut backend_cmd_rx;
                    let _drop_signal = AbortSignal(Some(dropped));
                    let _ = started.send(());
                    std::future::pending::<()>().await;
                });
                ActiveBrokerService {
                    kind,
                    cmd_tx: backend_cmd_tx,
                    task,
                }
            },
        ));

        cmd_tx
            .send(connect_command(kind))
            .expect("connect command fits");
        timeout(Duration::from_secs(1), backend_started_rx)
            .await
            .expect("backend start timeout")
            .expect("backend start signal");
        drop(cmd_tx);

        timeout(Duration::from_secs(1), service_task)
            .await
            .expect("service loop shutdown timeout")
            .expect("service loop task failed");
        backend_dropped_rx.await.expect("backend drop signal");
    }

    #[tokio::test]
    async fn service_loop_switches_brokers_before_shutting_down() {
        let kinds = compiled_brokers();
        if kinds.len() < 2 {
            return;
        }
        let first = kinds[0];
        let second = kinds[1];
        let (cmd_tx, cmd_rx) = service_command_channel(4);
        let (event_tx, _event_rx) = service_event_channel(4);
        let (market_tx, _) = watch::channel(MarketSnapshot::default());
        let (first_started_tx, first_started_rx) = oneshot::channel();
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let notify = Arc::new(Notify::new());
        let mut first_started_tx = Some(first_started_tx);
        let dropped_for_factory = Arc::clone(&dropped);
        let notify_for_factory = Arc::clone(&notify);

        let service_task = tokio::spawn(service_loop_with_backend_factory(
            cmd_rx,
            event_tx,
            market_tx,
            move |kind, _event_tx, _market_tx| {
                let (backend_cmd_tx, mut backend_cmd_rx) = service_command_channel(1);
                let started = first_started_tx.take();
                let dropped = Arc::clone(&dropped_for_factory);
                let notify = Arc::clone(&notify_for_factory);
                let task = tokio::spawn(async move {
                    let _backend_cmd_rx = &mut backend_cmd_rx;
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    let _drop_signal = DropSignal {
                        dropped,
                        kind,
                        notify,
                    };
                    std::future::pending::<()>().await;
                });
                ActiveBrokerService {
                    kind,
                    cmd_tx: backend_cmd_tx,
                    task,
                }
            },
        ));

        cmd_tx
            .send(connect_command(first))
            .expect("first connect command fits");
        timeout(Duration::from_secs(1), first_started_rx)
            .await
            .expect("first backend start timeout")
            .expect("first backend start signal");
        cmd_tx
            .send(connect_command(second))
            .expect("second connect command fits");
        timeout(Duration::from_secs(1), notify.notified())
            .await
            .expect("old backend shutdown timeout");
        assert_eq!(
            dropped.lock().expect("drop signal lock").as_slice(),
            &[first]
        );

        drop(cmd_tx);
        timeout(Duration::from_secs(1), service_task)
            .await
            .expect("service loop shutdown timeout")
            .expect("service loop task failed");
        assert_eq!(
            dropped.lock().expect("drop signal lock").as_slice(),
            &[first, second]
        );
    }
}
