use super::*;

mod commands;
mod debug;
mod internal;
mod maintenance;

use self::{
    commands::handle_command,
    internal::handle_internal,
    maintenance::maintain_session,
};

pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: tokio::sync::watch::Sender<MarketSnapshot>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    let _broker_task = spawn_broker_gateway_task(broker_rx, internal_tx.clone());
    let mut state = ServiceState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        broker_tx,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        session: None,
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    };
    let mut maintenance_tick =
        time::interval(Duration::from_secs(SESSION_MAINTENANCE_INTERVAL_SECS));
    maintenance_tick.tick().await;

    while let Some(next) = tokio::select! {
        biased;
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
        }
    }

    shutdown_state(&mut state, &event_tx);
}

const PENDING_TARGET_WATCHDOG_DELAY_SECS: u64 = 2;

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
    MaintenanceTick,
}

#[cfg(test)]
mod tests;
