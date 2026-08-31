use crate::broker::{ServiceCommand, ServiceCommandSender, ServiceEvent, ServiceEventReceiver};
use crate::ipc::connect_client;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::{Instant, sleep_until, timeout_at};

pub struct EngineSession {
    pub child: Option<tokio::process::Child>,
    pub cmd_tx: ServiceCommandSender,
    pub event_rx: ServiceEventReceiver,
}

/// The attach path must not select a TUI workflow from the App defaults while
/// the engine's ReplayState is still in flight. This is a startup-only
/// timeout; it is not used by the broker/execution path and does not add a
/// delay after the initial state has arrived.
pub(crate) const ATTACH_HYDRATION_TIMEOUT: Duration = Duration::from_secs(5);
/// A spawned engine gets one bounded socket-startup window. This is separate
/// from broker execution and is never used to retry orders or strategy
/// decisions.
pub(crate) const ENGINE_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const ENGINE_STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttachHydrationProgress {
    Pending,
    Ready,
    Terminal,
}

#[derive(Debug, Default)]
pub(crate) struct AttachHydration {
    saw_connected: bool,
    saw_accounts: bool,
    saw_account_snapshots: bool,
    saw_execution_state: bool,
}

impl AttachHydration {
    pub(crate) fn observe(&mut self, event: &ServiceEvent) -> AttachHydrationProgress {
        match event {
            ServiceEvent::Connected { .. } => self.saw_connected = true,
            ServiceEvent::AccountsLoaded(_) => self.saw_accounts = true,
            ServiceEvent::AccountSnapshotsLoaded(_) => self.saw_account_snapshots = true,
            ServiceEvent::ExecutionState(_) => self.saw_execution_state = true,
            ServiceEvent::Disconnected | ServiceEvent::Error(_) => {
                return AttachHydrationProgress::Terminal;
            }
            _ => {}
        }

        if self.saw_connected
            && self.saw_accounts
            && self.saw_account_snapshots
            && self.saw_execution_state
        {
            AttachHydrationProgress::Ready
        } else {
            AttachHydrationProgress::Pending
        }
    }
}

pub async fn connect_existing_engine(socket_path: &Path) -> Result<EngineSession> {
    let (cmd_tx, event_rx) = connect_client(socket_path).await?;
    let _ = cmd_tx.send(ServiceCommand::ReplayState);
    Ok(EngineSession {
        child: None,
        cmd_tx,
        event_rx,
    })
}

pub async fn connect_or_spawn_engine(
    config_path: Option<&Path>,
    socket_path: &Path,
    no_spawn_engine: bool,
) -> Result<EngineSession> {
    if let Ok(session) = connect_existing_engine(socket_path).await {
        return Ok(session);
    }
    if no_spawn_engine {
        bail!(
            "engine socket {} is unavailable and --no-spawn-engine was set",
            socket_path.display()
        );
    }

    spawn_and_connect_engine(config_path, socket_path).await
}

pub async fn spawn_and_connect_engine(
    config_path: Option<&Path>,
    socket_path: &Path,
) -> Result<EngineSession> {
    let child = spawn_engine_process(config_path, socket_path)?;
    let deadline = Instant::now() + ENGINE_STARTUP_TIMEOUT;
    let connection = connect_engine_until(socket_path, deadline).await;
    let (cmd_tx, event_rx) = match connection {
        Ok(connection) => connection,
        Err(err) => {
            terminate_and_reap_child(child).await;
            return Err(err);
        }
    };
    let _ = cmd_tx.send(ServiceCommand::ReplayState);
    Ok(EngineSession {
        child: Some(child),
        cmd_tx,
        event_rx,
    })
}

async fn connect_engine_until(
    socket_path: &Path,
    deadline: Instant,
) -> Result<(ServiceCommandSender, ServiceEventReceiver)> {
    loop {
        match timeout_at(deadline, connect_client(socket_path)).await {
            Ok(Ok(connection)) => return Ok(connection),
            Ok(Err(_)) => {}
            Err(_) => break,
        }

        let now = Instant::now();
        if now >= deadline {
            break;
        }
        sleep_until(std::cmp::min(deadline, now + ENGINE_STARTUP_POLL_INTERVAL)).await;
    }

    bail!(
        "timed out waiting for engine socket {}",
        socket_path.display()
    );
}

async fn terminate_and_reap_child(mut child: tokio::process::Child) {
    let _ = child.start_kill();
    let _ = child.wait().await;
}

fn spawn_engine_process(
    config_path: Option<&Path>,
    socket_path: &Path,
) -> Result<tokio::process::Child> {
    let current_exe = std::env::current_exe()?;
    let mut command = tokio::process::Command::new(current_exe);
    if let Some(config_path) = config_path {
        command.arg("--config").arg(config_path);
    }
    command
        .arg("--engine-socket")
        .arg(socket_path)
        .arg("engine")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    Ok(command.spawn()?)
}

pub fn unique_engine_socket_path(base_socket_path: &Path) -> PathBuf {
    let parent = base_socket_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".run"));
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    parent.join(format!(
        "trader-engine-{}-{timestamp_ns}.sock",
        std::process::id()
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        AttachHydration, AttachHydrationProgress, ENGINE_STARTUP_TIMEOUT, connect_engine_until,
        connect_existing_engine, unique_engine_socket_path,
    };
    use crate::broker::{BrokerCapabilities, BrokerKind, ServiceEvent, SessionKind};
    use crate::config::{AuthMode, TradingEnvironment};
    use crate::strategy::ExecutionStateSnapshot;
    use std::path::Path;
    use std::time::Duration;
    use tokio::time::{Instant, timeout};

    #[test]
    fn unique_socket_uses_base_parent_and_sock_suffix() {
        let socket = unique_engine_socket_path(Path::new("/tmp/midas/base.sock"));

        assert_eq!(socket.parent(), Some(Path::new("/tmp/midas")));
        assert_eq!(
            socket.extension().and_then(|ext| ext.to_str()),
            Some("sock")
        );
        assert!(
            socket
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("trader-engine-"))
        );
    }

    #[test]
    fn unique_socket_defaults_to_run_dir_without_parent() {
        let socket = unique_engine_socket_path(Path::new("base.sock"));

        assert_eq!(socket.parent(), Some(Path::new(".run")));
    }

    #[test]
    fn attach_hydration_requires_all_state_events_in_any_order() {
        let mut hydration = AttachHydration::default();

        assert_eq!(
            hydration.observe(&ServiceEvent::ExecutionState(
                ExecutionStateSnapshot::default()
            )),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::AccountSnapshotsLoaded(Vec::new())),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::Connected {
                broker: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_name: None,
                auth_mode: AuthMode::TokenFile,
                session_kind: SessionKind::Live,
                capabilities: BrokerCapabilities::default(),
            }),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::AccountsLoaded(Vec::new())),
            AttachHydrationProgress::Ready
        );
    }

    #[test]
    fn attach_hydration_stays_pending_without_account_snapshots() {
        let mut hydration = AttachHydration::default();

        assert_eq!(
            hydration.observe(&ServiceEvent::Connected {
                broker: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_name: None,
                auth_mode: AuthMode::TokenFile,
                session_kind: SessionKind::Live,
                capabilities: BrokerCapabilities::default(),
            }),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::AccountsLoaded(Vec::new())),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::ExecutionState(
                ExecutionStateSnapshot::default()
            )),
            AttachHydrationProgress::Pending
        );
    }

    #[test]
    fn attach_hydration_stops_on_terminal_connection_event() {
        let mut hydration = AttachHydration::default();

        assert_eq!(
            hydration.observe(&ServiceEvent::Disconnected),
            AttachHydrationProgress::Terminal
        );
    }

    #[test]
    fn attach_hydration_stops_immediately_on_error_after_connect() {
        let mut hydration = AttachHydration::default();

        assert_eq!(
            hydration.observe(&ServiceEvent::Connected {
                broker: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_name: None,
                auth_mode: AuthMode::TokenFile,
                session_kind: SessionKind::Live,
                capabilities: BrokerCapabilities::default(),
            }),
            AttachHydrationProgress::Pending
        );
        assert_eq!(
            hydration.observe(&ServiceEvent::Error("engine stopped".to_string())),
            AttachHydrationProgress::Terminal
        );
    }

    #[tokio::test]
    async fn missing_engine_socket_fails_within_startup_bound() {
        let socket = std::env::temp_dir().join(format!(
            "trader-missing-engine-{}-{}.sock",
            std::process::id(),
            unique_engine_socket_path(Path::new("base.sock"))
                .file_name()
                .expect("unique socket name")
                .to_string_lossy()
        ));
        let result = timeout(Duration::from_secs(1), connect_existing_engine(&socket)).await;

        assert!(result.is_ok(), "socket acquisition exceeded test bound");
        assert!(result.expect("timeout result").is_err());
    }

    #[tokio::test]
    async fn startup_connection_uses_one_explicit_deadline() {
        let socket = std::env::temp_dir().join(format!(
            "trader-startup-deadline-{}-{}.sock",
            std::process::id(),
            unique_engine_socket_path(Path::new("base.sock"))
                .file_name()
                .expect("unique socket name")
                .to_string_lossy()
        ));
        let deadline = Instant::now() + Duration::from_millis(20);
        let result = connect_engine_until(&socket, deadline).await;

        assert!(result.is_err());
        assert!(ENGINE_STARTUP_TIMEOUT >= Duration::from_secs(1));
    }
}
