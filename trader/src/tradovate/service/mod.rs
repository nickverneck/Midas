use super::*;

mod commands;
mod debug;
mod internal;
mod maintenance;

use self::{commands::handle_command, internal::handle_internal, maintenance::maintain_session};

pub async fn service_loop(
    mut cmd_rx: ServiceCommandReceiver,
    event_tx: ServiceEventSender,
    market_tx: tokio::sync::watch::Sender<MarketSnapshot>,
) {
    initialize_internal_backpressure();
    let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    let (broker_tx, broker_rx) = broker_command_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    let broker_task = spawn_broker_gateway_task(broker_rx, internal_tx.clone());
    let mut state = ServiceState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        broker_tx,
        broker_task: Some(broker_task),
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
        session: None,
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        snapshot_task: None,
        replay_lookup_job: None,
        replay_download_job: None,
        latency: LatencySnapshot::default(),
        snapshot_generation: 0,
        snapshot_revision: 0,
        snapshot_refresh_pending: false,
    };
    let mut maintenance_tick =
        time::interval(Duration::from_secs(SESSION_MAINTENANCE_INTERVAL_SECS));
    maintenance_tick.tick().await;

    while let Some(next) = tokio::select! {
        biased;
        _ = internal_tx.overflow_notified() => Some(Either::InternalOverflow),
        _ = event_tx.overflow_notified() => Some(Either::EventOverflow),
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
        _ = maintenance_tick.tick() => Some(Either::MaintenanceTick),
    } {
        match next {
            Either::Command(cmd) => {
                if let Err(err) =
                    handle_command(cmd, &mut state, &event_tx, &market_tx, internal_tx.clone())
                        .await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::Internal(internal) => {
                if let Err(err) = handle_internal(
                    internal,
                    &mut state,
                    &event_tx,
                    &market_tx,
                    internal_tx.clone(),
                )
                .await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::MaintenanceTick => {
                if let Err(err) = maintain_session(&mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::InternalOverflow => {
                let count = internal_tx.take_overflow_count();
                let _ = event_tx.send(ServiceEvent::Error(format!(
                    "broker service internal event queue overflowed ({count} event(s)); disconnecting safely"
                )));
                break;
            }
            Either::EventOverflow => {
                // The IPC event lane is bounded and its producer has no safe
                // way to reconstruct a dropped account/fill transition. End
                // this backend session so the owner can reconnect from a
                // fresh authoritative snapshot instead of continuing with a
                // stale F6/account view.
                break;
            }
        }
    }

    shutdown_state(&mut state, &event_tx).await;
    if let Some(task) = state.broker_task.take() {
        task.abort();
        let _ = task.await;
    }
}

const PENDING_TARGET_WATCHDOG_DELAY_SECS: u64 = 2;

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
    MaintenanceTick,
    InternalOverflow,
    EventOverflow,
}

#[cfg(test)]
mod tests;
