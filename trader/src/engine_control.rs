use anyhow::Result;

pub async fn close_and_kill_engine(id: u32) -> Result<()> {
    imp::close_and_kill_engine(id).await
}

pub async fn kill_engine_process(id: u32) -> Result<()> {
    imp::kill_engine_process(id).await
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod imp {
    use crate::broker::{ManualOrderAction, ServiceCommand, ServiceEvent};
    use crate::engine_registry::resolve_engine;
    use crate::ipc::connect_client;
    use anyhow::{Context, Result, anyhow, bail};
    use std::fs;
    use std::os::unix::fs::FileTypeExt;
    use std::path::Path;
    use std::time::Duration;
    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
    use tokio::time::{Instant, sleep, timeout};

    const REPLAY_TIMEOUT: Duration = Duration::from_secs(2);
    const FLATTEN_TIMEOUT: Duration = Duration::from_secs(20);
    const PROCESS_EXIT_TIMEOUT: Duration = Duration::from_secs(5);
    const REPLAY_POLL_INTERVAL: Duration = Duration::from_millis(250);

    #[derive(Debug, Default)]
    struct ObservedEngineState {
        disconnected: bool,
        saw_execution_state: bool,
        selected_contract_name: Option<String>,
        market_position_qty: i32,
    }

    pub async fn close_and_kill_engine(id: u32) -> Result<()> {
        let engine = resolve_engine(id)?;
        flatten_engine(&engine.socket_path).await?;
        terminate_process(id, &engine.socket_path).await
    }

    pub async fn kill_engine_process(id: u32) -> Result<()> {
        let engine = resolve_engine(id)?;
        terminate_process(id, &engine.socket_path).await
    }

    async fn flatten_engine(socket_path: &Path) -> Result<()> {
        let (cmd_tx, mut event_rx) = connect_client(socket_path)
            .await
            .with_context(|| format!("connect engine socket {}", socket_path.display()))?;
        let mut observed = replay_engine_state(&cmd_tx, &mut event_rx).await?;

        if observed.disconnected || observed.market_position_qty == 0 {
            return Ok(());
        }

        cmd_tx
            .send(ServiceCommand::DisarmExecutionStrategy {
                reason: "CLI shutdown requested".to_string(),
            })
            .map_err(|_| anyhow!("send disarm command"))?;
        cmd_tx
            .send(ServiceCommand::ManualOrder {
                action: ManualOrderAction::Close,
            })
            .map_err(|_| anyhow!("send close command"))?;

        let deadline = Instant::now() + FLATTEN_TIMEOUT;
        while Instant::now() < deadline {
            observed = replay_engine_state(&cmd_tx, &mut event_rx).await?;
            if observed.disconnected || observed.market_position_qty == 0 {
                return Ok(());
            }
            sleep(REPLAY_POLL_INTERVAL).await;
        }

        let label = observed
            .selected_contract_name
            .as_deref()
            .unwrap_or("selected market");
        bail!("timed out waiting for engine position to flatten on {label}; engine left running")
    }

    async fn replay_engine_state(
        cmd_tx: &UnboundedSender<ServiceCommand>,
        event_rx: &mut UnboundedReceiver<ServiceEvent>,
    ) -> Result<ObservedEngineState> {
        cmd_tx
            .send(ServiceCommand::ReplayState)
            .map_err(|_| anyhow!("send replay-state command"))?;

        let deadline = Instant::now() + REPLAY_TIMEOUT;
        let mut observed = ObservedEngineState::default();
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match timeout(remaining, event_rx.recv()).await {
                Ok(Some(event)) => {
                    observe_event(&mut observed, event)?;
                    if observed.disconnected || observed.saw_execution_state {
                        return Ok(observed);
                    }
                }
                Ok(None) => bail!("engine IPC connection closed unexpectedly"),
                Err(_) => break,
            }
        }

        if observed.disconnected || observed.saw_execution_state {
            Ok(observed)
        } else {
            bail!("timed out waiting for replayed engine state")
        }
    }

    fn observe_event(observed: &mut ObservedEngineState, event: ServiceEvent) -> Result<()> {
        match event {
            ServiceEvent::Disconnected => {
                observed.disconnected = true;
            }
            ServiceEvent::ExecutionState(snapshot) => {
                observed.saw_execution_state = true;
                observed.selected_contract_name = snapshot.selected_contract_name;
                observed.market_position_qty = snapshot.market_position_qty;
            }
            ServiceEvent::Error(message) => {
                bail!("{message}");
            }
            _ => {}
        }
        Ok(())
    }

    async fn terminate_process(id: u32, socket_path: &Path) -> Result<()> {
        send_signal(id, libc::SIGTERM)?;
        if !wait_for_exit(id, PROCESS_EXIT_TIMEOUT).await {
            send_signal(id, libc::SIGKILL)?;
            if !wait_for_exit(id, PROCESS_EXIT_TIMEOUT).await {
                bail!("engine {id} did not exit after SIGKILL");
            }
        }
        cleanup_socket(socket_path)?;
        Ok(())
    }

    fn send_signal(id: u32, signal: i32) -> Result<()> {
        let rc = unsafe { libc::kill(id as i32, signal) };
        if rc == 0 {
            return Ok(());
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        Err(err).with_context(|| format!("signal engine {id} with {signal}"))
    }

    async fn wait_for_exit(id: u32, timeout_dur: Duration) -> bool {
        let deadline = Instant::now() + timeout_dur;
        while Instant::now() < deadline {
            if !process_exists(id) {
                return true;
            }
            sleep(Duration::from_millis(100)).await;
        }
        !process_exists(id)
    }

    fn process_exists(id: u32) -> bool {
        let rc = unsafe { libc::kill(id as i32, 0) };
        if rc == 0 {
            return true;
        }
        let err = std::io::Error::last_os_error();
        err.raw_os_error() != Some(libc::ESRCH)
    }

    fn cleanup_socket(socket_path: &Path) -> Result<()> {
        let Ok(metadata) = fs::metadata(socket_path) else {
            return Ok(());
        };
        if metadata.file_type().is_socket() {
            fs::remove_file(socket_path)
                .with_context(|| format!("remove engine socket {}", socket_path.display()))?;
        }
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod imp {
    use anyhow::{Result, bail};

    pub async fn close_and_kill_engine(_id: u32) -> Result<()> {
        bail!("`trader kill -c` is currently only supported on Linux and macOS");
    }

    pub async fn kill_engine_process(_id: u32) -> Result<()> {
        bail!("`trader kill` is currently only supported on Linux and macOS");
    }
}
