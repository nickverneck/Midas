use super::*;

// The service owns snapshot_revision, so the high bit can track whether a
// snapshot build is already running. The generation separately prevents a
// detached build from an older session being accepted after reconnect.
const SNAPSHOT_IN_FLIGHT_MASK: u64 = 1 << 63;
const SNAPSHOT_REVISION_MASK: u64 = !SNAPSHOT_IN_FLIGHT_MASK;

pub(crate) fn spawn_user_sync_task(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> (UnboundedSender<UserSocketCommand>, JoinHandle<()>) {
    let (request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(user_sync_worker(
        cfg,
        tokens,
        account_ids,
        request_rx,
        internal_tx,
    ));
    (request_tx, task)
}

pub(crate) fn spawn_rest_probe_task(
    client: Client,
    cfg: AppConfig,
    access_token: String,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        interval.tick().await;
        loop {
            interval.tick().await;
            if let Ok(rest_rtt_ms) = measure_rest_rtt_ms(&client, &cfg.env, &access_token).await {
                let _ = internal_tx.send(InternalEvent::RestLatencyMeasured(rest_rtt_ms));
            }
        }
    })
}

pub(crate) fn request_snapshot_refresh(
    state: &mut ServiceState,
    internal_tx: &UnboundedSender<InternalEvent>,
) {
    let Some(session) = state.session.as_ref() else {
        return;
    };

    let in_flight = state.snapshot_revision & SNAPSHOT_IN_FLIGHT_MASK != 0;
    let revision = (state.snapshot_revision & SNAPSHOT_REVISION_MASK)
        .saturating_add(1)
        .min(SNAPSHOT_REVISION_MASK);
    state.snapshot_revision = revision
        | if in_flight {
            SNAPSHOT_IN_FLIGHT_MASK
        } else {
            0
        };

    // A refresh request while a build is running only advances the freshness
    // revision. The completion handler will start one follow-up build from
    // the latest service state, instead of queuing overlapping clones/scans.
    if in_flight {
        return;
    }

    let accounts = session.accounts.clone();
    let market = session.market.clone();
    let managed_protection = session.managed_protection.clone();
    let user_store = session.user_store.clone();
    let generation = state.snapshot_generation;
    let internal_tx = internal_tx.clone();
    tokio::spawn(async move {
        let snapshots = user_store.build_snapshots(&accounts, Some(&market), &managed_protection);
        let _ = internal_tx.send(InternalEvent::SnapshotsBuilt {
            generation,
            revision,
            snapshots,
        });
    });
}
