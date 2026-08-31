use crate::broker::{
    MarketSnapshot, SERVICE_COMMAND_QUEUE_CAPACITY, SERVICE_EVENT_QUEUE_CAPACITY, ServiceCommand,
    ServiceCommandSender, ServiceEvent, ServiceEventReceiver, service_command_channel,
    service_event_channel, service_loop,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{
    Notify,
    mpsc::{self, Sender},
    watch,
};
use tokio::time::{Duration, timeout};

/// A slow client must never make the engine retain an unbounded stream of
/// state snapshots.  Each state kind has one latest-value slot.  Intermediate
/// values are intentionally coalesced; the market stream itself continues to
/// use the existing `watch` channel below.
const SERVER_CLIENT_CONTROL_QUEUE_CAPACITY: usize = 128;
/// Commands are accepted per socket before they are forwarded to the broker
/// service.  A full queue is reported back to the client; it is never
/// silently discarded.
const SERVER_CLIENT_COMMAND_QUEUE_CAPACITY: usize = 64;
const SERVER_CLIENT_EVENT_QUEUE_CAPACITY: usize = 256;
const SERVER_CLIENT_WRITE_TIMEOUT: Duration = Duration::from_secs(2);
/// IPC is newline-delimited JSON.  Bound the complete wire line, including
/// its terminating newline, before parsing or writing it.  A large state
/// snapshot is rejected and the affected connection is closed rather than
/// allowing an attacker or a broken peer to grow a buffer indefinitely.
pub(crate) const MAX_IPC_LINE_BYTES: usize = 8 * 1024 * 1024;
/// Bound only the Unix-domain connection handshake. Once connected, normal
/// IPC reads and broker execution are not governed by this timeout.
pub(crate) const ENGINE_SOCKET_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Serialize, Deserialize)]
enum ClientWireMessage {
    Command(ServiceCommand),
}

#[derive(Debug, Serialize, Deserialize)]
enum ServerWireMessage {
    Event { sequence: u64, event: ServiceEvent },
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
    let (cmd_tx, cmd_rx) = service_command_channel(SERVICE_COMMAND_QUEUE_CAPACITY);
    let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _) = watch::channel(MarketSnapshot::default());
    let service_task = tokio::spawn(service_loop(cmd_rx, event_tx.clone(), market_tx.clone()));
    run_engine_server_loop(
        socket_path,
        listener,
        cmd_tx,
        event_tx,
        &mut event_rx,
        market_tx,
        service_task,
    )
    .await
}

async fn run_engine_server_loop(
    socket_path: &Path,
    listener: UnixListener,
    cmd_tx: ServiceCommandSender,
    event_tx: crate::broker::ServiceEventSender,
    event_rx: &mut ServiceEventReceiver,
    market_tx: watch::Sender<MarketSnapshot>,
    service_task: tokio::task::JoinHandle<()>,
) -> Result<()> {
    let sequence = Arc::new(AtomicU64::new(0));

    let cleanup_notify = Arc::new(Notify::new());
    let mut clients = Vec::<ServerClient>::new();
    let mut service_task = Some(service_task);
    let server_result: Result<()> = loop {
        tokio::select! {
            biased;
            service_result = service_task.as_mut().expect("engine service task is owned") => {
                let result = service_result.context("engine service task terminated");
                service_task.take();
                break result;
            }
            accept = listener.accept() => {
                let (stream, _) = match accept
                    .with_context(|| format!("accept engine socket {}", socket_path.display()))
                {
                    Ok(accepted) => accepted,
                    Err(error) => break Err(error),
                };
                let client = spawn_server_client(
                    stream,
                    cmd_tx.clone(),
                    market_tx.subscribe(),
                    sequence.clone(),
                    cleanup_notify.clone(),
                );
                clients.push(client);
            }
            _ = cleanup_notify.notified() => {
                clients.retain(ServerClient::is_alive);
            }
            _ = event_tx.overflow_notified() => {
                let dropped = event_tx.take_overflow_count();
                if dropped > 0 {
                    eprintln!(
                        "engine IPC event queue overflowed; disconnected {} client(s) after dropping {} event(s)",
                        clients.len(), dropped
                    );
                    for client in &clients {
                        client.lifecycle.close();
                    }
                    clients.retain(ServerClient::is_alive);
                }
            }
            maybe_event = event_rx.recv() => {
                let Some(event) = maybe_event else {
                    break Ok(());
                };
                let delivery = SequencedEvent {
                    sequence: sequence.fetch_add(1, Ordering::Relaxed),
                    event,
                };
                clients.retain(|client| client.try_send_delivery(delivery.clone()));
            }
        }
    };

    if let Some(service_task) = service_task {
        shutdown_server(service_task, &mut clients).await;
    } else {
        close_clients(&mut clients);
    }
    server_result
}

fn close_clients(clients: &mut Vec<ServerClient>) {
    for client in clients.iter() {
        client.lifecycle.close();
    }
    clients.clear();
}

async fn shutdown_server(
    service_task: tokio::task::JoinHandle<()>,
    clients: &mut Vec<ServerClient>,
) {
    close_clients(clients);

    service_task.abort();
    let _ = service_task.await;
}

pub async fn connect_client(
    socket_path: &Path,
) -> Result<(ServiceCommandSender, ServiceEventReceiver)> {
    let stream = timeout(
        ENGINE_SOCKET_CONNECT_TIMEOUT,
        UnixStream::connect(socket_path),
    )
    .await
    .with_context(|| {
        format!(
            "timed out connecting to engine socket {} after {:?}",
            socket_path.display(),
            ENGINE_SOCKET_CONNECT_TIMEOUT
        )
    })?
    .with_context(|| format!("connect engine socket {}", socket_path.display()))?;
    let (read_half, mut write_half) = stream.into_split();

    let (cmd_tx, mut cmd_rx) = service_command_channel(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
    let (event_tx, event_rx) = mpsc::channel::<ServiceEvent>(SERVER_CLIENT_EVENT_QUEUE_CAPACITY);
    let reader_event_tx = event_tx.clone();

    tokio::spawn(async move {
        while let Some(command) = cmd_rx.recv().await {
            let Ok(message) = serde_json::to_string(&ClientWireMessage::Command(command)) else {
                continue;
            };
            if !write_client_message(&mut write_half, &message).await {
                break;
            }
        }
    });

    tokio::spawn(async move {
        let mut reader = BufReader::new(read_half);
        loop {
            let line = match read_ipc_line(&mut reader).await {
                Ok(Some(line)) => line,
                Ok(None) => {
                    let _ = reader_event_tx
                        .try_send(ServiceEvent::Error("Engine connection closed.".to_string()));
                    break;
                }
                Err(err) => {
                    let _ = reader_event_tx.try_send(ServiceEvent::Error(format!(
                        "Engine IPC inbound frame rejected: {err}"
                    )));
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<ServerWireMessage>(&line) {
                Ok(ServerWireMessage::Event { event, .. }) => {
                    match reader_event_tx.try_send(event) {
                        Ok(()) => {}
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            eprintln!(
                                "engine IPC client event queue overflowed; closing event stream"
                            );
                            break;
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => break,
                    }
                }
                Err(err) => {
                    let _ = reader_event_tx.try_send(ServiceEvent::Error(format!(
                        "Engine IPC decode failed: {err}"
                    )));
                }
            }
        }
    });

    Ok((cmd_tx, event_rx))
}

#[derive(Clone)]
struct ServerClient {
    control_tx: Sender<SequencedEvent>,
    latest: LatestMailbox,
    #[allow(dead_code)]
    sequence: Arc<AtomicU64>,
    lifecycle: ClientLifecycle,
}

impl ServerClient {
    fn is_alive(&self) -> bool {
        self.lifecycle.is_alive()
    }

    #[allow(dead_code)]
    fn try_send(&self, event: ServiceEvent) -> bool {
        let delivery = SequencedEvent {
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed),
            event,
        };
        self.try_send_delivery(delivery)
    }

    fn try_send_delivery(&self, delivery: SequencedEvent) -> bool {
        // The writer task owns the receivers.  Once it has gone away, stop
        // retaining even a latest-value payload for this client.
        if !self.lifecycle.is_alive() || self.control_tx.is_closed() {
            return false;
        }

        // MarketSnapshot is deliberately delivered by market_rx below.  Do
        // not duplicate it through the service-event mailbox; this preserves
        // the pre-existing market watch semantics and its latest-value
        // behavior.
        if matches!(delivery.event, ServiceEvent::MarketSnapshot(_)) {
            return true;
        }

        if let Some(kind) = latest_event_kind(&delivery.event) {
            self.latest.replace(kind, delivery);
            true
        } else {
            // Control events are lossless while the queue has room.  An
            // explicit disconnect on overflow is safer than silently losing
            // an order, fill, error, protection, or lifecycle event.
            match self.control_tx.try_send(delivery) {
                Ok(()) => true,
                Err(mpsc::error::TrySendError::Full(_))
                | Err(mpsc::error::TrySendError::Closed(_)) => {
                    eprintln!(
                        "engine IPC client control queue overflowed or closed; disconnecting client"
                    );
                    self.lifecycle.close();
                    false
                }
            }
        }
    }
}

#[derive(Clone)]
struct ClientLifecycle {
    alive: Arc<AtomicBool>,
    closed_tx: watch::Sender<bool>,
    cleanup_notify: Arc<Notify>,
}

impl ClientLifecycle {
    fn new(cleanup_notify: Arc<Notify>) -> Self {
        let (closed_tx, _) = watch::channel(false);
        Self {
            alive: Arc::new(AtomicBool::new(true)),
            closed_tx,
            cleanup_notify,
        }
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    fn close(&self) {
        if self.alive.swap(false, Ordering::AcqRel) {
            let _ = self.closed_tx.send(true);
            self.cleanup_notify.notify_one();
        }
    }

    async fn closed(&self) {
        let mut closed = self.closed_tx.subscribe();
        if *closed.borrow() {
            return;
        }
        let _ = closed.changed().await;
    }
}

#[derive(Debug, Clone)]
struct SequencedEvent {
    /// Monotonic per-server generation used to document and preserve the
    /// relative order of lossless control events and state updates that are
    /// already pending when the writer chooses its next message.
    sequence: u64,
    event: ServiceEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatestEventKind {
    TradeMarkers,
    EngineHistory,
    Latency,
    ExecutionState,
    ReplaySpeed,
    ReplayExecutionLedger,
    ReplayExecutionLedgerSnapshot,
    ReplayDownloadProgress,
}

impl LatestEventKind {
    const COUNT: usize = 8;

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone)]
struct LatestMailbox {
    slots: Arc<Mutex<Vec<Option<SequencedEvent>>>>,
    notify: Arc<Notify>,
}

impl LatestMailbox {
    fn new() -> Self {
        Self {
            slots: Arc::new(Mutex::new(vec![None; LatestEventKind::COUNT])),
            notify: Arc::new(Notify::new()),
        }
    }

    fn replace(&self, kind: LatestEventKind, event: SequencedEvent) {
        let mut slots = self.slots.lock().expect("latest mailbox mutex poisoned");
        slots[kind.index()] = Some(event);
        drop(slots);
        self.notify.notify_one();
    }

    fn take_oldest(&self) -> Option<SequencedEvent> {
        let mut slots = self.slots.lock().expect("latest mailbox mutex poisoned");
        let index = slots
            .iter()
            .enumerate()
            .filter_map(|(index, event)| event.as_ref().map(|event| (index, event.sequence)))
            .min_by_key(|(_, sequence)| *sequence)
            .map(|(index, _)| index)?;
        slots[index].take()
    }

    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.slots
            .lock()
            .expect("latest mailbox mutex poisoned")
            .iter()
            .filter(|event| event.is_some())
            .count()
    }

    async fn notified(&self) {
        self.notify.notified().await;
    }
}

#[cfg(test)]
fn is_coalescible_event(event: &ServiceEvent) -> bool {
    latest_event_kind(event).is_some() || matches!(event, ServiceEvent::MarketSnapshot(_))
}

fn latest_event_kind(event: &ServiceEvent) -> Option<LatestEventKind> {
    Some(match event {
        ServiceEvent::TradeMarkersUpdated(_) => LatestEventKind::TradeMarkers,
        ServiceEvent::EngineHistoryUpdated(_) => LatestEventKind::EngineHistory,
        ServiceEvent::Latency(_) => LatestEventKind::Latency,
        ServiceEvent::ExecutionState(_) => LatestEventKind::ExecutionState,
        ServiceEvent::ReplaySpeedUpdated(_) => LatestEventKind::ReplaySpeed,
        ServiceEvent::ReplayExecutionLedgerUpdated(_) => LatestEventKind::ReplayExecutionLedger,
        ServiceEvent::ReplayExecutionLedgerSnapshot(_) => {
            LatestEventKind::ReplayExecutionLedgerSnapshot
        }
        ServiceEvent::ReplayDownloadProgress { .. } => LatestEventKind::ReplayDownloadProgress,
        _ => return None,
    })
}

fn spawn_server_client(
    stream: UnixStream,
    cmd_tx: ServiceCommandSender,
    mut market_rx: watch::Receiver<MarketSnapshot>,
    sequence: Arc<AtomicU64>,
    cleanup_notify: Arc<Notify>,
) -> ServerClient {
    let (read_half, mut write_half) = stream.into_split();
    let (control_tx, mut control_rx) =
        mpsc::channel::<SequencedEvent>(SERVER_CLIENT_CONTROL_QUEUE_CAPACITY);
    let latest = LatestMailbox::new();
    let writer_latest = latest.clone();
    let writer_sequence = sequence.clone();
    let lifecycle = ClientLifecycle::new(cleanup_notify);
    let writer_lifecycle = lifecycle.clone();
    let command_lifecycle = lifecycle.clone();
    let reader_lifecycle = lifecycle.clone();
    let command_sequence = sequence.clone();
    let reader_control_tx = control_tx.clone();
    let forward_control_tx = control_tx.clone();
    let forward_sequence = sequence.clone();
    let (command_tx, mut command_rx) =
        mpsc::channel::<ServiceCommand>(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = command_lifecycle.closed() => break,
                maybe_command = command_rx.recv() => {
                    let Some(command) = maybe_command else { break; };
                    if !forward_client_command(
                        &cmd_tx,
                        &forward_control_tx,
                        &forward_sequence,
                        command,
                    ) {
                        break;
                    }
                }
            }
        }
        command_lifecycle.close();
    });

    tokio::spawn(async move {
        let initial_snapshot = market_rx.borrow().clone();
        let initial = SequencedEvent {
            sequence: writer_sequence.fetch_add(1, Ordering::Relaxed),
            event: ServiceEvent::MarketSnapshot(initial_snapshot),
        };
        if !write_server_event(&mut write_half, initial, &writer_lifecycle).await {
            writer_lifecycle.close();
            return;
        }

        let mut pending_control: Option<SequencedEvent> = None;
        let mut pending_market: Option<SequencedEvent> = None;
        let mut pending_latest: Option<SequencedEvent> = None;
        let mut market_closed = false;

        loop {
            if !writer_lifecycle.is_alive() {
                break;
            }

            if pending_control.is_none() {
                match control_rx.try_recv() {
                    Ok(event) => pending_control = Some(event),
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                    Err(mpsc::error::TryRecvError::Empty) => {}
                }
            }

            if pending_latest.is_none() {
                pending_latest = writer_latest.take_oldest();
            }
            let next = match (
                pending_control.as_ref(),
                pending_market.as_ref(),
                pending_latest.as_ref(),
            ) {
                (None, None, None) => None,
                _ => {
                    let control = pending_control.as_ref().map(|event| (0, event.sequence));
                    let market = pending_market.as_ref().map(|event| (1, event.sequence));
                    let latest = pending_latest.as_ref().map(|event| (2, event.sequence));
                    match [control, market, latest]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(_, sequence)| *sequence)
                        .map(|(source, _)| source)
                    {
                        Some(0) => pending_control.take(),
                        Some(1) => pending_market.take(),
                        Some(2) => pending_latest.take(),
                        _ => None,
                    }
                }
            };

            if let Some(delivery) = next {
                if !write_server_event(&mut write_half, delivery, &writer_lifecycle).await {
                    writer_lifecycle.close();
                    break;
                }
                continue;
            }

            if market_closed {
                tokio::select! {
                    _ = writer_lifecycle.closed() => break,
                    maybe_event = control_rx.recv() => {
                        let Some(event) = maybe_event else { break; };
                        pending_control = Some(event);
                    }
                    _ = writer_latest.notified() => {}
                }
            } else {
                tokio::select! {
                    _ = writer_lifecycle.closed() => break,
                    maybe_event = control_rx.recv() => {
                        let Some(event) = maybe_event else { break; };
                        pending_control = Some(event);
                    }
                    _ = writer_latest.notified() => {}
                    changed = market_rx.changed() => {
                        if changed.is_err() {
                            market_closed = true;
                        } else {
                            pending_market = Some(SequencedEvent {
                                sequence: writer_sequence.fetch_add(1, Ordering::Relaxed),
                                event: ServiceEvent::MarketSnapshot(market_rx.borrow_and_update().clone()),
                            });
                        }
                    }
                }
            }
        }
        writer_lifecycle.close();
    });

    tokio::spawn(async move {
        let mut reader = BufReader::new(read_half);
        loop {
            let next_line = tokio::select! {
                _ = reader_lifecycle.closed() => break,
                line = read_ipc_line(&mut reader) => line,
            };
            match next_line {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<ClientWireMessage>(&line) {
                        Ok(ClientWireMessage::Command(command)) => {
                            if !try_enqueue_client_command(
                                &command_tx,
                                &reader_control_tx,
                                &command_sequence,
                                command,
                            ) {
                                break;
                            }
                        }
                        Err(_) => {}
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    eprintln!("engine IPC inbound frame rejected: {error}");
                    break;
                }
            }
        }
        reader_lifecycle.close();
    });

    ServerClient {
        control_tx,
        latest,
        sequence,
        lifecycle,
    }
}

fn try_enqueue_client_command(
    command_tx: &Sender<ServiceCommand>,
    control_tx: &Sender<SequencedEvent>,
    sequence: &AtomicU64,
    command: ServiceCommand,
) -> bool {
    match command_tx.try_send(command) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => enqueue_control_error(
            control_tx,
            sequence,
            "Engine IPC command queue is full; command was not accepted.",
        ),
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

fn forward_client_command(
    command_tx: &ServiceCommandSender,
    control_tx: &Sender<SequencedEvent>,
    sequence: &AtomicU64,
    command: ServiceCommand,
) -> bool {
    match command_tx.send(command) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => enqueue_control_error(
            control_tx,
            sequence,
            "Engine IPC service queue is full; command was not accepted.",
        ),
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

fn enqueue_control_error(
    control_tx: &Sender<SequencedEvent>,
    sequence: &AtomicU64,
    message: &str,
) -> bool {
    control_tx
        .try_send(SequencedEvent {
            sequence: sequence.fetch_add(1, Ordering::Relaxed),
            event: ServiceEvent::Error(message.to_string()),
        })
        .is_ok()
}

async fn write_server_event(
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
    delivery: SequencedEvent,
    lifecycle: &ClientLifecycle,
) -> bool {
    write_server_event_with_timeout(write_half, delivery, lifecycle, SERVER_CLIENT_WRITE_TIMEOUT)
        .await
}

async fn write_server_event_with_timeout(
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
    delivery: SequencedEvent,
    lifecycle: &ClientLifecycle,
    write_timeout: Duration,
) -> bool {
    let message = ServerWireMessage::Event {
        sequence: delivery.sequence,
        event: delivery.event,
    };
    let line = match encode_ipc_line(&message, "outbound server") {
        Ok(line) => line,
        Err(error) => {
            eprintln!("engine IPC outbound frame rejected: {error}");
            return false;
        }
    };
    timeout(write_timeout, async {
        tokio::select! {
            result = write_half.write_all(&line) => result.is_ok(),
            _ = lifecycle.closed() => false,
        }
    })
    .await
    .unwrap_or(false)
}

async fn write_client_message(
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
    message: &str,
) -> bool {
    let Some(line_bytes) = message.len().checked_add(1) else {
        eprintln!("engine IPC outbound client frame length overflowed");
        return false;
    };
    if line_bytes > MAX_IPC_LINE_BYTES {
        eprintln!(
            "engine IPC outbound client frame rejected: {line_bytes} bytes exceeds {} byte limit",
            MAX_IPC_LINE_BYTES
        );
        return false;
    }
    let mut line = Vec::with_capacity(line_bytes);
    line.extend_from_slice(message.as_bytes());
    line.push(b'\n');
    timeout(SERVER_CLIENT_WRITE_TIMEOUT, async {
        write_half.write_all(&line).await.is_ok()
    })
    .await
    .unwrap_or(false)
}

fn encode_ipc_line<T: Serialize>(message: &T, direction: &str) -> Result<Vec<u8>> {
    let mut line =
        serde_json::to_vec(message).with_context(|| format!("serialize {direction} IPC frame"))?;
    let line_bytes = line
        .len()
        .checked_add(1)
        .context("IPC frame length overflowed")?;
    if line_bytes > MAX_IPC_LINE_BYTES {
        bail!(
            "{direction} IPC frame is {line_bytes} bytes; maximum is {} bytes",
            MAX_IPC_LINE_BYTES
        );
    }
    line.push(b'\n');
    Ok(line)
}

/// Read one newline-delimited IPC frame without ever appending bytes beyond
/// the wire limit.  `AsyncBufReadExt::read_line` is intentionally not used:
/// it may allocate the entire oversized line before returning an error to the
/// caller.
async fn read_ipc_line<R>(reader: &mut R) -> std::io::Result<Option<String>>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = Vec::with_capacity(MAX_IPC_LINE_BYTES.min(4096));
    loop {
        let buffer = reader.fill_buf().await?;
        if buffer.is_empty() {
            if line.is_empty() {
                return Ok(None);
            }
            return String::from_utf8(line).map(Some).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("IPC line is not valid UTF-8: {error}"),
                )
            });
        }

        let bytes_to_consume = buffer
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|position| position + 1)
            .unwrap_or(buffer.len());
        let next_len = line.len().saturating_add(bytes_to_consume);
        if next_len > MAX_IPC_LINE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("IPC line exceeds {} byte limit", MAX_IPC_LINE_BYTES),
            ));
        }

        line.extend_from_slice(&buffer[..bytes_to_consume]);
        reader.consume(bytes_to_consume);
        if line.last().copied() == Some(b'\n') {
            line.pop();
            return String::from_utf8(line).map(Some).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("IPC line is not valid UTF-8: {error}"),
                )
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClientLifecycle, ClientWireMessage, LatestEventKind, LatestMailbox, MAX_IPC_LINE_BYTES,
        SERVER_CLIENT_COMMAND_QUEUE_CAPACITY, SERVER_CLIENT_CONTROL_QUEUE_CAPACITY, SequencedEvent,
        ServerClient, ServerWireMessage, connect_client, encode_ipc_line, forward_client_command,
        is_coalescible_event, latest_event_kind, read_ipc_line, run_engine_server,
        run_engine_server_loop, shutdown_server, spawn_server_client, try_enqueue_client_command,
        write_client_message, write_server_event_with_timeout,
    };
    use crate::broker::{
        LatencySnapshot, ServiceCommand, ServiceEvent, service_command_channel,
        service_event_channel,
    };
    use std::io::Cursor;
    use std::sync::{Arc, atomic::AtomicU64};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::sync::{mpsc, oneshot};
    use tokio::time::{Duration, timeout};

    fn test_client() -> (ServerClient, mpsc::Receiver<SequencedEvent>) {
        let (control_tx, control_rx) = mpsc::channel(SERVER_CLIENT_CONTROL_QUEUE_CAPACITY);
        let lifecycle = ClientLifecycle::new(Arc::new(tokio::sync::Notify::new()));
        let client = ServerClient {
            control_tx,
            latest: LatestMailbox::new(),
            sequence: Arc::new(AtomicU64::new(0)),
            lifecycle,
        };
        (client, control_rx)
    }

    #[test]
    fn high_rate_observability_events_are_coalescible() {
        assert!(is_coalescible_event(&ServiceEvent::Latency(
            LatencySnapshot::default()
        )));
        assert!(!is_coalescible_event(&ServiceEvent::DebugLog(
            "diagnostic".to_string()
        )));
        assert!(!is_coalescible_event(&ServiceEvent::Status(
            "order acknowledgement".to_string()
        )));
        assert!(is_coalescible_event(&ServiceEvent::ExecutionState(
            Default::default()
        )));
        assert!(!is_coalescible_event(
            &ServiceEvent::AccountSnapshotsLoaded(Vec::new())
        ));
        assert!(is_coalescible_event(&ServiceEvent::TradeMarkersUpdated(
            Vec::new()
        )));
    }

    #[tokio::test]
    async fn oversized_ipc_line_is_rejected_before_json_parse() {
        let mut bytes = vec![b'x'; MAX_IPC_LINE_BYTES];
        bytes.push(b'\n');
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));

        let error = read_ipc_line(&mut reader)
            .await
            .expect_err("oversized IPC line must be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("byte limit"));
    }

    #[test]
    fn oversized_outbound_ipc_frame_is_rejected_before_write() {
        let message = ServerWireMessage::Event {
            sequence: 0,
            event: ServiceEvent::DebugLog("x".repeat(MAX_IPC_LINE_BYTES)),
        };

        let error = encode_ipc_line(&message, "test outbound")
            .expect_err("oversized outbound IPC frame must be rejected");
        assert!(error.to_string().contains("maximum"));
    }

    #[tokio::test]
    async fn oversized_client_ipc_line_is_rejected_before_write() {
        let (stream, _peer) = UnixStream::pair().expect("unix socket pair");
        let (_, mut write_half) = stream.into_split();

        assert!(
            !write_client_message(&mut write_half, &"x".repeat(MAX_IPC_LINE_BYTES)).await,
            "oversized client IPC frames must not be written"
        );
    }

    #[test]
    fn safety_and_lifecycle_events_are_not_coalescible() {
        assert!(!is_coalescible_event(&ServiceEvent::Error(
            "broker unavailable".to_string()
        )));
        assert!(!is_coalescible_event(&ServiceEvent::Disconnected));
    }

    #[test]
    fn critical_events_are_lossless_and_ordered_while_capacity_is_available() {
        let (client, mut control_rx) = test_client();

        assert!(client.try_send(ServiceEvent::Status("order acknowledgement".to_string())));
        assert!(client.try_send(ServiceEvent::DebugLog("fill detail".to_string())));
        assert!(client.try_send(ServiceEvent::AccountSnapshotsLoaded(Vec::new())));
        assert_eq!(client.latest.pending_count(), 0);

        let first = control_rx.try_recv().expect("ack event");
        assert_eq!(first.sequence, 0);
        assert!(matches!(
            first.event,
            ServiceEvent::Status(message) if message == "order acknowledgement"
        ));

        let second = control_rx.try_recv().expect("fill event");
        assert_eq!(second.sequence, 1);
        assert!(matches!(
            second.event,
            ServiceEvent::DebugLog(message) if message == "fill detail"
        ));

        let third = control_rx.try_recv().expect("account snapshot event");
        assert_eq!(third.sequence, 2);
        assert!(matches!(
            third.event,
            ServiceEvent::AccountSnapshotsLoaded(_)
        ));
    }

    #[tokio::test]
    async fn service_event_queue_is_bounded_and_overflow_is_observable() {
        let (event_tx, mut event_rx) = service_event_channel(1);
        event_tx
            .send(ServiceEvent::Status("first".to_string()))
            .expect("first event fits");

        let overflow = event_tx
            .send(ServiceEvent::Status("second".to_string()))
            .expect_err("second event must report a full queue");
        assert!(matches!(
            overflow,
            mpsc::error::TrySendError::Full(ServiceEvent::Status(message)) if message == "second"
        ));
        assert_eq!(event_tx.take_overflow_count(), 1);
        timeout(Duration::from_secs(1), event_tx.overflow_notified())
            .await
            .expect("overflow notification");

        assert!(matches!(
            event_rx.recv().await,
            Some(ServiceEvent::Status(message)) if message == "first"
        ));
        assert!(event_rx.try_recv().is_err());
    }

    #[test]
    fn newest_state_replaces_old_intermediate_state() {
        let (client, _control_rx) = test_client();

        for index in 0..10_000 {
            assert!(client.try_send(ServiceEvent::Latency(LatencySnapshot {
                last_order_ack_ms: Some(index),
                ..LatencySnapshot::default()
            })));
        }

        assert_eq!(client.latest.pending_count(), 1);
        let delivered = client.latest.take_oldest().expect("latest state");
        assert!(matches!(
            delivered.event,
            ServiceEvent::Latency(snapshot) if snapshot.last_order_ack_ms == Some(9999)
        ));
    }

    #[tokio::test]
    async fn newest_state_is_eventually_delivered_to_the_writer() {
        let (client, _control_rx) = test_client();
        let latest = client.latest.clone();
        let writer = tokio::spawn(async move {
            latest.notified().await;
            latest.take_oldest().expect("latest state")
        });

        for index in 0..100 {
            assert!(client.try_send(ServiceEvent::Latency(LatencySnapshot {
                last_order_ack_ms: Some(index),
                ..LatencySnapshot::default()
            })));
        }

        let delivered = writer.await.expect("writer task");
        assert!(matches!(
            delivered.event,
            ServiceEvent::Latency(snapshot) if snapshot.last_order_ack_ms == Some(99)
        ));
    }

    #[test]
    fn latest_mailbox_is_bounded_by_the_number_of_state_kinds() {
        let (client, _control_rx) = test_client();

        for _ in 0..10_000 {
            assert!(client.try_send(ServiceEvent::Latency(LatencySnapshot::default(),)));
        }

        assert!(client.latest.pending_count() <= LatestEventKind::COUNT);
        assert_eq!(client.latest.pending_count(), 1);
    }

    #[test]
    fn control_events_remain_lossless_and_ordered_while_capacity_is_available() {
        let (client, mut control_rx) = test_client();

        for index in 0..3 {
            assert!(client.try_send(ServiceEvent::Error(format!("error {index}"))));
        }

        for index in 0..3 {
            let event = control_rx.try_recv().expect("ordered control event");
            assert_eq!(event.sequence, index);
            assert!(matches!(
                event.event,
                ServiceEvent::Error(message) if message == format!("error {index}")
            ));
        }
    }

    #[test]
    fn slow_client_is_disconnected_before_control_events_are_dropped() {
        let (client, control_rx) = test_client();

        for _ in 0..SERVER_CLIENT_CONTROL_QUEUE_CAPACITY {
            assert!(client.try_send(ServiceEvent::Error("error".to_string())));
        }

        assert!(!client.try_send(ServiceEvent::Error("overflow".to_string())));
        assert_eq!(control_rx.len(), SERVER_CLIENT_CONTROL_QUEUE_CAPACITY);
        assert!(!client.is_alive());
    }

    #[test]
    fn disconnected_client_is_handled_cleanly() {
        let (client, control_rx) = test_client();
        drop(control_rx);

        assert!(!client.try_send(ServiceEvent::Error("closed".to_string())));
        assert!(!client.try_send(ServiceEvent::DebugLog("closed".to_string())));
        assert_eq!(client.latest.pending_count(), 0);
    }

    #[test]
    fn market_snapshot_remains_on_the_dedicated_watch_path() {
        let (client, _control_rx) = test_client();

        assert_eq!(
            latest_event_kind(&ServiceEvent::MarketSnapshot(Default::default())),
            None
        );
        assert!(client.try_send(ServiceEvent::MarketSnapshot(Default::default())));
        assert_eq!(client.latest.pending_count(), 0);
    }

    #[test]
    fn command_overflow_reports_an_error_without_dropping_accepted_commands() {
        let (command_tx, command_rx) =
            mpsc::channel::<ServiceCommand>(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let (control_tx, mut control_rx) =
            mpsc::channel::<SequencedEvent>(SERVER_CLIENT_CONTROL_QUEUE_CAPACITY);
        let sequence = AtomicU64::new(0);

        for _ in 0..SERVER_CLIENT_COMMAND_QUEUE_CAPACITY {
            assert!(try_enqueue_client_command(
                &command_tx,
                &control_tx,
                &sequence,
                ServiceCommand::ReplayState,
            ));
        }
        assert!(try_enqueue_client_command(
            &command_tx,
            &control_tx,
            &sequence,
            ServiceCommand::ReplayState,
        ));
        assert_eq!(command_rx.len(), SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let error = control_rx.try_recv().expect("overflow error event");
        assert!(matches!(
            error.event,
            ServiceEvent::Error(message) if message.contains("command queue is full")
        ));
    }

    #[test]
    fn root_service_forwarding_reports_bounded_queue_overflow() {
        let (command_tx, mut command_rx) = service_command_channel(1);
        let (control_tx, mut control_rx) =
            mpsc::channel::<SequencedEvent>(SERVER_CLIENT_CONTROL_QUEUE_CAPACITY);
        let sequence = AtomicU64::new(0);

        assert!(forward_client_command(
            &command_tx,
            &control_tx,
            &sequence,
            ServiceCommand::ReplayState,
        ));
        assert!(forward_client_command(
            &command_tx,
            &control_tx,
            &sequence,
            ServiceCommand::ReplayState,
        ));
        assert!(matches!(
            command_rx.try_recv().expect("accepted command"),
            ServiceCommand::ReplayState
        ));

        let error = control_rx.try_recv().expect("root overflow error");
        assert!(matches!(
            error.event,
            ServiceEvent::Error(message) if message.contains("service queue is full")
        ));
    }

    #[tokio::test]
    async fn socket_writer_delivers_initial_snapshot_and_newest_state() {
        let (server_stream, client_stream) = UnixStream::pair().expect("unix socket pair");
        let (cmd_tx, _cmd_rx) = service_command_channel(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let (_market_tx, market_rx) =
            tokio::sync::watch::channel(crate::broker::MarketSnapshot::default());
        let cleanup = Arc::new(tokio::sync::Notify::new());
        let client = spawn_server_client(
            server_stream,
            cmd_tx,
            market_rx,
            Arc::new(AtomicU64::new(0)),
            cleanup,
        );
        let mut lines = tokio::io::BufReader::new(client_stream).lines();

        let initial = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("initial snapshot timeout")
            .expect("initial snapshot read")
            .expect("initial snapshot line");
        assert!(initial.contains("MarketSnapshot"));

        for index in 0..100 {
            assert!(client.try_send(ServiceEvent::Latency(LatencySnapshot {
                last_order_ack_ms: Some(index),
                ..LatencySnapshot::default()
            })));
        }
        let newest = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("state timeout")
            .expect("state read")
            .expect("state line");
        assert!(newest.contains("\"last_order_ack_ms\":99"));

        client.lifecycle.close();
    }

    #[tokio::test]
    async fn socket_writer_preserves_sequence_across_control_and_latest_mailboxes() {
        let (server_stream, client_stream) = UnixStream::pair().expect("unix socket pair");
        let (cmd_tx, _cmd_rx) = service_command_channel(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let (_market_tx, market_rx) =
            tokio::sync::watch::channel(crate::broker::MarketSnapshot::default());
        let client = spawn_server_client(
            server_stream,
            cmd_tx,
            market_rx,
            Arc::new(AtomicU64::new(0)),
            Arc::new(tokio::sync::Notify::new()),
        );
        let mut lines = tokio::io::BufReader::new(client_stream).lines();
        let _ = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("initial snapshot timeout")
            .expect("initial snapshot read");

        assert!(client.try_send(ServiceEvent::Error("control".to_string())));
        assert!(client.try_send(ServiceEvent::Latency(LatencySnapshot {
            last_order_ack_ms: Some(7),
            ..LatencySnapshot::default()
        })));
        let first = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("control timeout")
            .expect("control read")
            .expect("control line");
        let second = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("state timeout")
            .expect("state read")
            .expect("state line");
        assert!(first.contains("control"));
        assert!(second.contains("\"last_order_ack_ms\":7"));
        client.lifecycle.close();
    }

    #[tokio::test]
    async fn disconnected_reader_closes_writer_and_marks_client_dead_without_new_events() {
        let (server_stream, client_stream) = UnixStream::pair().expect("unix socket pair");
        let (cmd_tx, _cmd_rx) = service_command_channel(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let (_market_tx, market_rx) =
            tokio::sync::watch::channel(crate::broker::MarketSnapshot::default());
        let client = spawn_server_client(
            server_stream,
            cmd_tx,
            market_rx,
            Arc::new(AtomicU64::new(0)),
            Arc::new(tokio::sync::Notify::new()),
        );
        drop(client_stream);

        timeout(Duration::from_secs(1), async {
            while client.is_alive() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("client lifecycle cleanup timeout");
    }

    #[tokio::test]
    async fn blocked_socket_writer_times_out_without_retaining_the_client() {
        let (server_stream, _client_stream) = UnixStream::pair().expect("unix socket pair");
        let filler = vec![b'x'; 64 * 1024];
        let mut writes = 0;
        loop {
            match server_stream.try_write(&filler) {
                Ok(_) if writes < 1024 => writes += 1,
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("fill socket: {error}"),
            }
        }

        let (_, mut write_half) = server_stream.into_split();
        let lifecycle = ClientLifecycle::new(Arc::new(tokio::sync::Notify::new()));
        let result = timeout(
            Duration::from_secs(1),
            write_server_event_with_timeout(
                &mut write_half,
                SequencedEvent {
                    sequence: 0,
                    event: ServiceEvent::DebugLog("x".repeat(8 * 1024 * 1024)),
                },
                &lifecycle,
                Duration::from_millis(25),
            ),
        )
        .await
        .expect("blocked writer test timed out");

        assert!(
            !result,
            "a full socket must be bounded by the write timeout"
        );
    }

    #[tokio::test]
    async fn normal_attach_replay_state_produces_one_response() {
        let socket_path = std::env::temp_dir().join(format!(
            "trader-ipc-{}-{}.sock",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let server_path = socket_path.clone();
        let server = tokio::spawn(async move { run_engine_server(&server_path).await });

        timeout(Duration::from_secs(1), async {
            while !socket_path.exists() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("engine socket did not appear");

        let stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect engine socket");
        let (read_half, mut write_half) = stream.into_split();
        let mut lines = tokio::io::BufReader::new(read_half).lines();

        let initial = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("initial snapshot timeout")
            .expect("initial snapshot read")
            .expect("initial snapshot line");
        assert!(initial.contains("MarketSnapshot"));

        let command =
            serde_json::to_string(&ClientWireMessage::Command(ServiceCommand::ReplayState))
                .expect("encode replay-state command");
        write_half
            .write_all(format!("{command}\n").as_bytes())
            .await
            .expect("send replay-state command");

        let response = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("replay-state response timeout")
            .expect("replay-state response read")
            .expect("replay-state response line");
        assert!(matches!(
            serde_json::from_str::<ServerWireMessage>(&response),
            Ok(ServerWireMessage::Event {
                event: ServiceEvent::Disconnected,
                ..
            })
        ));

        match timeout(Duration::from_millis(100), lines.next_line()).await {
            Err(_) | Ok(Ok(None)) => {}
            Ok(Ok(Some(line))) => panic!("duplicate replay state response: {line}"),
            Ok(Err(error)) => panic!("read after replay-state response: {error}"),
        }

        server.abort();
        let _ = server.await;
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn service_task_termination_shuts_down_server_and_closes_client() {
        let socket_path = std::env::temp_dir().join(format!(
            "trader-ipc-service-exit-{}-{}.sock",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let listener = UnixListener::bind(&socket_path).expect("bind engine socket");
        let (cmd_tx, _cmd_rx) = service_command_channel(SERVER_CLIENT_COMMAND_QUEUE_CAPACITY);
        let (event_tx, mut event_rx) = service_event_channel(1);
        let (market_tx, _) = tokio::sync::watch::channel(crate::broker::MarketSnapshot::default());
        let (service_exit_tx, service_exit_rx) = oneshot::channel();
        let service_task = tokio::spawn(async move {
            service_exit_rx.await.expect("service exit signal");
        });
        let server_socket_path = socket_path.clone();
        let server = tokio::spawn(async move {
            run_engine_server_loop(
                &server_socket_path,
                listener,
                cmd_tx,
                event_tx,
                &mut event_rx,
                market_tx,
                service_task,
            )
            .await
        });

        let stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect engine socket");
        let (read_half, _write_half) = stream.into_split();
        let mut lines = tokio::io::BufReader::new(read_half).lines();
        let initial = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("initial snapshot timeout")
            .expect("initial snapshot read")
            .expect("initial snapshot line");
        assert!(initial.contains("MarketSnapshot"));

        service_exit_tx.send(()).expect("terminate service task");
        let closed = timeout(Duration::from_secs(1), lines.next_line())
            .await
            .expect("client closure timeout")
            .expect("client closure read");
        assert!(closed.is_none(), "client received data after service exit");

        let server_result = timeout(Duration::from_secs(1), server)
            .await
            .expect("server shutdown timeout")
            .expect("server task join");
        assert!(
            server_result.is_ok(),
            "normal service exit: {server_result:?}"
        );
        assert!(UnixStream::connect(&socket_path).await.is_err());
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn server_shutdown_closes_clients_and_awaits_service_task() {
        struct DropSignal(Option<oneshot::Sender<()>>);

        impl Drop for DropSignal {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        let (dropped_tx, dropped_rx) = oneshot::channel();
        let service_task = tokio::spawn(async move {
            let _drop_signal = DropSignal(Some(dropped_tx));
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;

        let (client, _control_rx) = test_client();
        let client_observer = client.clone();
        let mut clients = vec![client];

        shutdown_server(service_task, &mut clients).await;

        assert!(clients.is_empty());
        assert!(!client_observer.is_alive());
        timeout(Duration::from_secs(1), dropped_rx)
            .await
            .expect("service task was not awaited after abort")
            .expect("service task drop signal was canceled");
    }

    #[tokio::test]
    async fn silent_accepted_socket_returns_before_hydration_deadline() {
        let socket_path = std::env::temp_dir().join(format!(
            "trader-ipc-silent-{}-{}.sock",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let listener = UnixListener::bind(&socket_path).expect("bind silent engine socket");
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.expect("accept silent client");
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let connection = timeout(Duration::from_secs(1), connect_client(&socket_path))
            .await
            .expect("silent socket connection exceeded startup bound")
            .expect("connect silent engine socket");
        drop(connection);
        server.await.expect("silent socket server task");
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn concurrent_latest_replacements_leave_one_newest_slot() {
        let mailbox = LatestMailbox::new();
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let mut tasks = Vec::new();
        for task_id in 0..8 {
            let mailbox = mailbox.clone();
            let barrier = barrier.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                mailbox.replace(
                    LatestEventKind::Latency,
                    SequencedEvent {
                        sequence: task_id,
                        event: ServiceEvent::Latency(LatencySnapshot {
                            last_order_ack_ms: Some(task_id),
                            ..LatencySnapshot::default()
                        }),
                    },
                );
            }));
        }
        for task in tasks {
            task.await.expect("replacement task");
        }
        assert_eq!(mailbox.pending_count(), 1);
        assert!(matches!(
            mailbox.take_oldest().expect("latest event").event,
            ServiceEvent::Latency(_)
        ));
    }
}
