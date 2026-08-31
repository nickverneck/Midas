use super::*;

// The service owns snapshot_revision, so the high bit can track whether a
// snapshot build is already running. The generation separately prevents a
// detached build from an older session being accepted after reconnect.
const SNAPSHOT_IN_FLIGHT_MASK: u64 = 1 << 63;
const SNAPSHOT_REVISION_MASK: u64 = !SNAPSHOT_IN_FLIGHT_MASK;
#[cfg(test)]
const USER_SYNC_QUEUE_CAPACITY: usize = 256;

fn next_snapshot_revision(snapshot_revision: u64) -> (u64, bool) {
    let in_flight = snapshot_revision & SNAPSHOT_IN_FLIGHT_MASK != 0;
    // Revision identity is only used within one connection generation. Wrap
    // instead of saturating so the terminal value cannot make an ancient
    // delayed completion indistinguishable from a new build.
    let revision =
        (snapshot_revision & SNAPSHOT_REVISION_MASK).wrapping_add(1) & SNAPSHOT_REVISION_MASK;
    let revision = if revision == 0 { 1 } else { revision };
    (revision, in_flight)
}

/// Reserve one build. Refresh requests that arrive while it runs collapse into
/// one pending bit; they do not create a queue of revisions or invalidate the
/// active build. The active result is useful progress even when it is followed
/// by a newer build from the latest store state.
fn reserve_snapshot_refresh(
    snapshot_revision: &mut u64,
    snapshot_refresh_pending: &mut bool,
) -> Option<u64> {
    let (revision, in_flight) = next_snapshot_revision(*snapshot_revision);
    if in_flight {
        *snapshot_refresh_pending = true;
        return None;
    }
    *snapshot_refresh_pending = false;
    *snapshot_revision = revision | SNAPSHOT_IN_FLIGHT_MASK;
    Some(revision)
}

pub(crate) fn spawn_user_sync_task(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: InternalEventSender,
) -> (UserSocketCommandSender, JoinHandle<()>) {
    #[cfg(not(test))]
    {
        let (request_tx, request_rx) = user_socket_command_channel();
        let task = tokio::spawn(user_sync_worker(
            cfg,
            tokens,
            account_ids,
            request_rx,
            internal_tx,
        ));
        return (request_tx, task);
    }

    // Unit tests historically exercise this task with an unbounded sender.
    // Keep that adapter test-only so production order requests enter the
    // bounded channel directly and cannot accumulate before the worker.
    #[cfg(test)]
    {
        let (request_tx, mut legacy_request_rx) = tokio::sync::mpsc::unbounded_channel();
        let (bounded_request_tx, bounded_request_rx) =
            tokio::sync::mpsc::channel(USER_SYNC_QUEUE_CAPACITY);
        let task = tokio::spawn(async move {
            let worker = tokio::spawn(user_sync_worker(
                cfg,
                tokens,
                account_ids,
                bounded_request_rx,
                internal_tx.clone(),
            ));
            loop {
                if legacy_request_rx.len() >= USER_SYNC_QUEUE_CAPACITY {
                    let _ = internal_tx.send(InternalEvent::Error(
                    "legacy user websocket ingress exceeded its bounded admission window; disconnecting safely"
                        .to_string(),
                ));
                    break;
                }
                let Some(request) = legacy_request_rx.recv().await else {
                    break;
                };
                match bounded_request_tx.try_send(request) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(request)) => {
                        let _ = request.response_tx.send(Err(
                            "user websocket command queue overflowed; disconnecting safely"
                                .to_string(),
                        ));
                        let _ = internal_tx.send(InternalEvent::Error(
                            "user websocket command queue overflowed; disconnecting safely"
                                .to_string(),
                        ));
                        break;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(request)) => {
                        let _ = request
                            .response_tx
                            .send(Err("user websocket command worker is closed".to_string()));
                        break;
                    }
                }
            }
            drop(bounded_request_tx);
            let _ = worker.await;
        });
        (request_tx, task)
    }
}

pub(crate) fn spawn_rest_probe_task(
    client: Client,
    cfg: AppConfig,
    access_token: String,
    internal_tx: InternalEventSender,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        interval.tick().await;
        loop {
            interval.tick().await;
            let rest_url = cfg.broker_rest_url();
            if let Ok(rest_rtt_ms) = measure_rest_rtt_ms(&client, &rest_url, &access_token).await {
                let _ = internal_tx.send(InternalEvent::RestLatencyMeasured(rest_rtt_ms));
            }
        }
    })
}

pub(crate) fn request_snapshot_refresh(
    state: &mut ServiceState,
    internal_tx: &InternalEventSender,
) {
    let Some(session) = state.session.as_ref() else {
        return;
    };

    let Some(revision) = reserve_snapshot_refresh(
        &mut state.snapshot_revision,
        &mut state.snapshot_refresh_pending,
    ) else {
        // A refresh request while a build is running sets the one pending bit.
        // The completion handler will start one follow-up build from the
        // latest service state, instead of queuing overlapping clones/scans.
        return;
    };

    let accounts = session.accounts.clone();
    let market = session.market.clone();
    let managed_protection = session.managed_protection.clone();
    let user_store = session.user_store.clone();
    let generation = state.snapshot_generation;
    let internal_tx = internal_tx.clone();
    state.snapshot_task = Some(tokio::spawn(async move {
        let snapshots = user_store.build_snapshots(&accounts, Some(&market), &managed_protection);
        if let Err(failure) =
            send_snapshots_built(&internal_tx, generation, revision, snapshots).await
        {
            // Delivery failures use the same bounded critical lane. If this
            // completion cannot be admitted either, the existing overflow
            // supervisor terminates the service safely; it must never retry
            // or retain the snapshot payload.
            let _ = internal_tx.send(InternalEvent::SnapshotsBuildFailed {
                generation,
                revision,
                failure,
            });
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_snapshot_request_reserves_in_flight_and_follow_up_coalesces() {
        let mut snapshot_revision = 0;

        let mut snapshot_refresh_pending = false;
        assert_eq!(
            reserve_snapshot_refresh(&mut snapshot_revision, &mut snapshot_refresh_pending),
            Some(1)
        );
        assert_eq!(
            snapshot_revision,
            1 | SNAPSHOT_IN_FLIGHT_MASK,
            "the first request must mark the build in flight before spawning"
        );

        assert_eq!(
            reserve_snapshot_refresh(&mut snapshot_revision, &mut snapshot_refresh_pending),
            None
        );
        assert_eq!(
            snapshot_revision,
            1 | SNAPSHOT_IN_FLIGHT_MASK,
            "a follow-up is one bounded pending bit, not another revision"
        );
        assert!(snapshot_refresh_pending);
    }

    #[test]
    fn continuous_refresh_requests_remain_coalesced_and_admit_follow_up() {
        let mut snapshot_revision = 0;
        let mut snapshot_refresh_pending = false;

        assert_eq!(
            reserve_snapshot_refresh(&mut snapshot_revision, &mut snapshot_refresh_pending),
            Some(1)
        );
        for _ in 0..100_000 {
            assert_eq!(
                reserve_snapshot_refresh(&mut snapshot_revision, &mut snapshot_refresh_pending),
                None
            );
        }
        assert_eq!(snapshot_revision, 1 | SNAPSHOT_IN_FLIGHT_MASK);
        assert!(snapshot_refresh_pending);

        snapshot_revision &= SNAPSHOT_REVISION_MASK;
        assert_eq!(
            reserve_snapshot_refresh(&mut snapshot_revision, &mut snapshot_refresh_pending),
            Some(2)
        );
        assert!(!snapshot_refresh_pending);
    }

    #[test]
    fn snapshot_revision_wraps_without_reusing_in_flight_bit() {
        let (revision, in_flight) =
            next_snapshot_revision(SNAPSHOT_REVISION_MASK | SNAPSHOT_IN_FLIGHT_MASK);
        assert_eq!(revision, 1);
        assert!(in_flight);
        assert_ne!(
            revision | SNAPSHOT_IN_FLIGHT_MASK,
            revision,
            "a result cannot be current while another build is in flight"
        );
    }

    #[tokio::test]
    async fn request_snapshot_refresh_spawns_only_the_first_build() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let session = SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            token_file_snapshot: None,
            accounts: vec![AccountInfo {
                id: 42,
                name: "SIM".to_string(),
                raw: json!({}),
            }],
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: None,
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store: UserSyncStore::default(),
            selected_account_id: Some(42),
            selected_contract: Some(ContractSuggestion {
                id: 3570918,
                name: "ESM6".to_string(),
                description: "E-mini S&P".to_string(),
                raw: json!({}),
            }),
            bar_type: BarType::default(),
            candle_mode: CandleMode::Standard,
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
            engine_run: None,
        };
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (replay_speed_tx, _replay_speed_rx) =
            tokio::sync::watch::channel(ReplaySpeed::default());
        let mut state = ServiceState {
            client: Client::builder().build().expect("client"),
            broker_tx,
            broker_task: None,
            replay_speed_tx,
            replay_speed: ReplaySpeed::default(),
            replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
            session: Some(session),
            replay: None,
            user_task: None,
            market_task: None,
            rest_probe_task: None,
            replay_lookup_job: None,
            replay_download_job: None,
            latency: LatencySnapshot::default(),
            snapshot_generation: 0,
            snapshot_revision: 0,
            snapshot_refresh_pending: false,
            snapshot_task: None,
        };
        let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

        request_snapshot_refresh(&mut state, &internal_tx);
        assert_eq!(
            state.snapshot_revision,
            1 | SNAPSHOT_IN_FLIGHT_MASK,
            "the real request path must reserve before the worker is spawned"
        );

        request_snapshot_refresh(&mut state, &internal_tx);
        assert_eq!(
            state.snapshot_revision,
            1 | SNAPSHOT_IN_FLIGHT_MASK,
            "the real follow-up path must coalesce while the first build runs"
        );
        // The request is retained as one bounded pending bit. The active
        // completion will publish first and admit exactly one follow-up.
        // (The test does not consume the completion here, so the task remains
        // owned by the service state.)
        assert!(state.snapshot_refresh_pending);

        let first = tokio::time::timeout(Duration::from_secs(1), internal_rx.recv())
            .await
            .expect("first snapshot build should complete")
            .expect("internal event channel should remain open");
        assert!(matches!(
            first,
            InternalEvent::SnapshotsBuilt {
                generation: 0,
                revision: 1,
                ..
            }
        ));

        assert!(
            tokio::time::timeout(Duration::from_millis(50), internal_rx.recv())
                .await
                .is_err(),
            "the coalesced request must not spawn a second overlapping build"
        );
    }
}
