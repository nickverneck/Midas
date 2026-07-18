use crate::broker::{ServiceCommand, ServiceEvent};
use crate::ipc::connect_client;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::sleep;

pub struct EngineSession {
    pub child: Option<tokio::process::Child>,
    pub cmd_tx: mpsc::UnboundedSender<ServiceCommand>,
    pub event_rx: mpsc::UnboundedReceiver<ServiceEvent>,
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
    let mut child = spawn_engine_process(config_path, socket_path)?;
    for _ in 0..50 {
        if let Ok((cmd_tx, event_rx)) = connect_client(socket_path).await {
            let _ = cmd_tx.send(ServiceCommand::ReplayState);
            return Ok(EngineSession {
                child: Some(child),
                cmd_tx,
                event_rx,
            });
        }
        sleep(Duration::from_millis(100)).await;
    }

    let _ = child.start_kill();
    bail!(
        "timed out waiting for engine socket {}",
        socket_path.display()
    );
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
    use super::unique_engine_socket_path;
    use std::path::Path;

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
}
