use crate::tradovate::{ServiceCommand, ServiceEvent, service_loop};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{
    mpsc::{self, UnboundedReceiver, UnboundedSender},
    watch,
};

#[derive(Debug, Serialize, Deserialize)]
enum ClientWireMessage {
    Command(ServiceCommand),
}

#[derive(Debug, Serialize, Deserialize)]
enum ServerWireMessage {
    Event(ServiceEvent),
}

pub async fn run_engine_server(socket_path: &Path) -> Result<()> {
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create engine socket dir {}", parent.display()))?;
    }
    if socket_path.exists() {
        fs::remove_file(socket_path)
            .with_context(|| format!("remove stale engine socket {}", socket_path.display()))?;
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("bind engine socket {}", socket_path.display()))?;
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (market_tx, _) = watch::channel(crate::tradovate::MarketSnapshot::default());
    tokio::spawn(service_loop(cmd_rx, event_tx, market_tx.clone()));

    let mut clients = Vec::<UnboundedSender<ServiceEvent>>::new();
    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept
                    .with_context(|| format!("accept engine socket {}", socket_path.display()))?;
                let client_tx = spawn_server_client(stream, cmd_tx.clone(), market_tx.subscribe());
                clients.push(client_tx);
                let _ = cmd_tx.send(ServiceCommand::ReplayState);
            }
            maybe_event = event_rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                clients.retain(|client_tx| client_tx.send(event.clone()).is_ok());
            }
        }
    }

    Ok(())
}

pub async fn connect_client(
    socket_path: &Path,
) -> Result<(
    UnboundedSender<ServiceCommand>,
    UnboundedReceiver<ServiceEvent>,
)> {
    let stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("connect engine socket {}", socket_path.display()))?;
    let (read_half, mut write_half) = stream.into_split();

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<ServiceCommand>();
    let (event_tx, event_rx) = mpsc::unbounded_channel::<ServiceEvent>();
    let reader_event_tx = event_tx.clone();

    tokio::spawn(async move {
        while let Some(command) = cmd_rx.recv().await {
            let Ok(message) = serde_json::to_string(&ClientWireMessage::Command(command)) else {
                continue;
            };
            if write_half.write_all(message.as_bytes()).await.is_err() {
                break;
            }
            if write_half.write_all(b"\n").await.is_err() {
                break;
            }
        }
    });

    tokio::spawn(async move {
        let mut lines = BufReader::new(read_half).lines();
        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => {
                    let _ = reader_event_tx
                        .send(ServiceEvent::Error("Engine connection closed.".to_string()));
                    break;
                }
                Err(err) => {
                    let _ = reader_event_tx.send(ServiceEvent::Error(format!(
                        "Engine IPC read failed: {err}"
                    )));
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<ServerWireMessage>(&line) {
                Ok(ServerWireMessage::Event(event)) => {
                    if reader_event_tx.send(event).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = reader_event_tx.send(ServiceEvent::Error(format!(
                        "Engine IPC decode failed: {err}"
                    )));
                }
            }
        }
    });

    Ok((cmd_tx, event_rx))
}

fn spawn_server_client(
    stream: UnixStream,
    cmd_tx: UnboundedSender<ServiceCommand>,
    mut market_rx: watch::Receiver<crate::tradovate::MarketSnapshot>,
) -> UnboundedSender<ServiceEvent> {
    let (read_half, mut write_half) = stream.into_split();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ServiceEvent>();

    tokio::spawn(async move {
        let initial_snapshot = market_rx.borrow().clone();
        let Ok(message) = serde_json::to_string(&ServerWireMessage::Event(
            ServiceEvent::MarketSnapshot(initial_snapshot),
        )) else {
            return;
        };
        if write_half.write_all(message.as_bytes()).await.is_err() {
            return;
        }
        if write_half.write_all(b"\n").await.is_err() {
            return;
        }

        loop {
            let next = tokio::select! {
                maybe_event = event_rx.recv() => maybe_event,
                changed = market_rx.changed() => {
                    if changed.is_err() {
                        None
                    } else {
                        Some(ServiceEvent::MarketSnapshot(market_rx.borrow().clone()))
                    }
                }
            };

            let Some(event) = next else {
                break;
            };

            let Ok(message) = serde_json::to_string(&ServerWireMessage::Event(event)) else {
                continue;
            };
            if write_half.write_all(message.as_bytes()).await.is_err() {
                break;
            }
            if write_half.write_all(b"\n").await.is_err() {
                break;
            }
        }
    });

    tokio::spawn(async move {
        let mut lines = BufReader::new(read_half).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<ClientWireMessage>(&line) {
                Ok(ClientWireMessage::Command(command)) => {
                    let _ = cmd_tx.send(command);
                }
                Err(_) => {}
            }
        }
    });

    event_tx
}
