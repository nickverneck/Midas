mod types;

pub use types::*;

use tokio::sync::{
    mpsc::{self, UnboundedSender},
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
    cmd_tx: UnboundedSender<ServiceCommand>,
    task: JoinHandle<()>,
}

pub async fn service_loop(
    mut cmd_rx: mpsc::UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: watch::Sender<MarketSnapshot>,
) {
    let mut active = None::<ActiveBrokerService>;

    while let Some(cmd) = cmd_rx.recv().await {
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
                    previous.task.abort();
                }
                active = Some(spawn_backend_service(
                    kind,
                    event_tx.clone(),
                    market_tx.clone(),
                ));
            }
        }

        let Some(active_service) = active.as_ref() else {
            match cmd {
                ServiceCommand::ReplayState => {
                    let _ = event_tx.send(ServiceEvent::Disconnected);
                }
                _ => {
                    let _ = event_tx.send(ServiceEvent::Error(
                        "Select a broker and connect first.".to_string(),
                    ));
                }
            }
            continue;
        };

        if active_service.cmd_tx.send(cmd).is_err() {
            let _ = event_tx.send(ServiceEvent::Error(format!(
                "{} engine is unavailable; reconnect to restart it.",
                active_service.kind.label()
            )));
            active = None;
            let _ = market_tx.send(MarketSnapshot::default());
        }
    }

    if let Some(previous) = active.take() {
        previous.task.abort();
    }
}

fn command_broker(cmd: &ServiceCommand) -> Option<BrokerKind> {
    match cmd {
        ServiceCommand::Connect(cfg) => Some(cfg.broker),
        ServiceCommand::EnterReplayMode { config, .. } => Some(config.broker),
        _ => None,
    }
}

fn cmd_ref(cmd: &ServiceCommand) -> &ServiceCommand {
    cmd
}

fn spawn_backend_service(
    kind: BrokerKind,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: watch::Sender<MarketSnapshot>,
) -> ActiveBrokerService {
    let (backend_cmd_tx, backend_cmd_rx) = mpsc::unbounded_channel();
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
