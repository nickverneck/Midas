#[cfg(feature = "replay")]
use super::commands::{
    begin_replay_cache_commit, cancel_replay_operation, reap_finished_replay_download,
    replace_replay_lookup,
};
use super::*;
#[cfg(feature = "replay")]
use crate::broker::{ReplayDownloadOperationId, ReplayDownloadPhase};
use crate::strategy::{NativeReversalMode, StrategyKind};
use serde_json::json;
use std::collections::BTreeMap;

fn test_session() -> SessionState {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    SessionState {
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
    }
}

fn test_state(session: SessionState) -> ServiceState {
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    ServiceState {
        client: Client::builder().build().expect("client"),
        broker_tx,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        replay_lookup_job: None,
        replay_download_job: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    }
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_lookup_replacement_keeps_exactly_one_owned_job() {
    struct ActiveGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for ActiveGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let mut state = test_state(test_session());
    let active = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for _ in 0..8 {
        replace_replay_lookup(&mut state).await;
        let counter = active.clone();
        let operation_id = ReplayDownloadOperationId::next();
        let task = tokio::spawn(async move {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _guard = ActiveGuard(counter);
            std::future::pending::<()>().await;
        });
        state.replay_lookup_job = Some(ReplayLookupJob { operation_id, task });
        tokio::task::yield_now().await;
        assert_eq!(active.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
    replace_replay_lookup(&mut state).await;
    assert!(state.replay_lookup_job.is_none());
    assert_eq!(active.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_download_is_single_flight_and_reports_operation_scoped_busy() {
    let mut state = test_state(test_session());
    let active_id = ReplayDownloadOperationId::next();
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);
    state.replay_download_job = Some(ReplayDownloadJob {
        operation_id: active_id,
        cancel_tx,
        stage: Arc::new(AtomicU8::new(ReplayDownloadJobStage::Network as u8)),
        task: tokio::spawn(std::future::pending()),
    });
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let rejected_id = ReplayDownloadOperationId::next();

    handle_command(
        ServiceCommand::DownloadReplayData {
            operation_id: rejected_id,
            config: AppConfig::default(),
            instrument: "MES".to_string(),
            contract: ContractSuggestion {
                id: 4_399_631,
                name: "MESU6".to_string(),
                description: "Micro E-mini S&P".to_string(),
                raw: json!({}),
            },
            target: None,
            start_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 23).expect("start"),
            end_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 23).expect("end"),
            source_kind: "server-bars".to_string(),
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            display_name: None,
            tags: Vec::new(),
        },
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("busy response");

    assert_eq!(
        state.replay_download_job.as_ref().unwrap().operation_id,
        active_id
    );
    assert!(matches!(
        event_rx.recv().await,
        Some(ServiceEvent::ReplayDownloadFailed {
            operation_id,
            phase: ReplayDownloadPhase::Busy,
            ..
        }) if operation_id == rejected_id
    ));
    shutdown_tasks(&mut state).await;
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_download_cancellation_before_commit_preserves_manifest() {
    let root = std::env::temp_dir().join(format!(
        "trader-replay-cancel-test-{}",
        ReplayDownloadOperationId::next().0
    ));
    std::fs::create_dir_all(&root).expect("test directory");
    let manifest_path = root.join("manifest.json");
    std::fs::write(&manifest_path, b"old manifest").expect("old manifest");
    let operation_id = ReplayDownloadOperationId::next();
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let stage = Arc::new(AtomicU8::new(ReplayDownloadJobStage::Network as u8));
    let task_stage = stage.clone();
    let ready = Arc::new(tokio::sync::Barrier::new(2));
    let release = Arc::new(tokio::sync::Barrier::new(2));
    let task_ready = ready.clone();
    let task_release = release.clone();
    let task_manifest = manifest_path.clone();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let task_event_tx = event_tx.clone();
    let task = tokio::spawn(async move {
        let _cancel_rx = cancel_rx;
        task_ready.wait().await;
        task_release.wait().await;
        match begin_replay_cache_commit(&task_stage) {
            Ok(()) => std::fs::write(task_manifest, b"new manifest").expect("new manifest"),
            Err((phase, err)) => {
                let _ = task_event_tx.send(ServiceEvent::ReplayDownloadFailed {
                    operation_id,
                    phase,
                    message: err.to_string(),
                });
            }
        }
    });
    let mut state = test_state(test_session());
    state.replay_download_job = Some(ReplayDownloadJob {
        operation_id,
        cancel_tx,
        stage,
        task,
    });

    ready.wait().await;
    cancel_replay_operation(operation_id, &mut state, &event_tx).await;
    assert_eq!(
        state
            .replay_download_job
            .as_ref()
            .map(ReplayDownloadJob::stage),
        Some(ReplayDownloadJobStage::CancelRequested)
    );
    release.wait().await;
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if state
                .replay_download_job
                .as_ref()
                .is_some_and(|job| job.task.is_finished())
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cancelled job completion");
    reap_finished_replay_download(&mut state).await;

    assert_eq!(
        std::fs::read(&manifest_path).expect("manifest"),
        b"old manifest"
    );
    assert!(state.replay_download_job.is_none());
    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::ReplayDownloadProgress {
            operation_id: reported,
            phase: ReplayDownloadPhase::Cancelling,
            ..
        } if *reported == operation_id
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::ReplayDownloadFailed {
            operation_id: reported,
            phase: ReplayDownloadPhase::Cancelled,
            ..
        } if *reported == operation_id
    )));
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_download_cancellation_during_commit_is_reported_and_not_aborted() {
    let mut state = test_state(test_session());
    let operation_id = ReplayDownloadOperationId::next();
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let stage = Arc::new(AtomicU8::new(ReplayDownloadJobStage::Network as u8));
    let task_stage = stage.clone();
    let commit_claimed = Arc::new(tokio::sync::Barrier::new(2));
    let finish_commit = Arc::new(tokio::sync::Barrier::new(2));
    let task_claimed = commit_claimed.clone();
    let task_finish = finish_commit.clone();
    let committed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let task_committed = committed.clone();
    state.replay_download_job = Some(ReplayDownloadJob {
        operation_id,
        cancel_tx,
        stage,
        task: tokio::spawn(async move {
            let _cancel_rx = cancel_rx;
            begin_replay_cache_commit(&task_stage).expect("worker claims commit");
            task_claimed.wait().await;
            task_finish.wait().await;
            task_committed.store(true, std::sync::atomic::Ordering::SeqCst);
        }),
    });
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    commit_claimed.wait().await;
    cancel_replay_operation(operation_id, &mut state, &event_tx).await;

    assert!(matches!(
        event_rx.recv().await,
        Some(ServiceEvent::ReplayDownloadProgress {
            operation_id: reported,
            phase: ReplayDownloadPhase::WritingCache,
            message,
            ..
        }) if reported == operation_id && message.contains("cannot interrupt")
    ));
    assert_eq!(
        state
            .replay_download_job
            .as_ref()
            .map(ReplayDownloadJob::stage),
        Some(ReplayDownloadJobStage::Committing)
    );
    assert!(
        !*state
            .replay_download_job
            .as_ref()
            .unwrap()
            .cancel_tx
            .borrow()
    );
    finish_commit.wait().await;
    let job = state.replay_download_job.take().expect("owned commit job");
    job.task.await.expect("commit task");
    assert!(committed.load(std::sync::atomic::Ordering::SeqCst));
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn shutdown_aborts_network_jobs_but_awaits_owned_cache_commit() {
    let mut state = test_state(test_session());
    let lookup_dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let lookup_flag = lookup_dropped.clone();
    struct DropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    state.replay_lookup_job = Some(ReplayLookupJob {
        operation_id: ReplayDownloadOperationId::next(),
        task: tokio::spawn(async move {
            let _guard = DropFlag(lookup_flag);
            std::future::pending::<()>().await;
        }),
    });
    tokio::task::yield_now().await;

    let commit_finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let commit_flag = commit_finished.clone();
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);
    state.replay_download_job = Some(ReplayDownloadJob {
        operation_id: ReplayDownloadOperationId::next(),
        cancel_tx,
        stage: Arc::new(AtomicU8::new(ReplayDownloadJobStage::Committing as u8)),
        task: tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            commit_flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }),
    });

    shutdown_tasks(&mut state).await;

    assert!(lookup_dropped.load(std::sync::atomic::Ordering::SeqCst));
    assert!(commit_finished.load(std::sync::atomic::Ordering::SeqCst));
    assert!(state.replay_lookup_job.is_none());
    assert!(state.replay_download_job.is_none());
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn shutdown_aborts_an_owned_network_stage_download() {
    struct DropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let mut state = test_state(test_session());
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let task_flag = dropped.clone();
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);
    state.replay_download_job = Some(ReplayDownloadJob {
        operation_id: ReplayDownloadOperationId::next(),
        cancel_tx,
        stage: Arc::new(AtomicU8::new(ReplayDownloadJobStage::Network as u8)),
        task: tokio::spawn(async move {
            let _guard = DropFlag(task_flag);
            std::future::pending::<()>().await;
        }),
    });
    tokio::task::yield_now().await;

    shutdown_tasks(&mut state).await;

    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    assert!(state.replay_download_job.is_none());
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_downloader_validation_failure_is_reported_without_starting_live_state() {
    let mut state = test_state(test_session());
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    let operation_id = ReplayDownloadOperationId::next();
    tokio::time::timeout(
        std::time::Duration::from_millis(50),
        handle_command(
            ServiceCommand::DownloadReplayData {
                operation_id,
                config: AppConfig::default(),
                instrument: "MES".to_string(),
                contract: ContractSuggestion {
                    id: 4_399_631,
                    name: "MESU6".to_string(),
                    description: "Micro E-mini S&P".to_string(),
                    raw: json!({"id": 4_399_631, "name": "MESU6"}),
                },
                target: None,
                start_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 24).expect("start"),
                end_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 23).expect("end"),
                source_kind: "server-bars".to_string(),
                bar_type: BarType::minute(1),
                candle_mode: CandleMode::HeikinAshi,
                display_name: Some("invalid range".to_string()),
                tags: vec!["test".to_string()],
            },
            &mut state,
            &event_tx,
            &market_tx,
            internal_tx,
        ),
    )
    .await
    .expect("replay job dispatch should not block the service loop")
    .expect("validation failure should be emitted as an event");

    match tokio::time::timeout(std::time::Duration::from_secs(1), event_rx.recv())
        .await
        .expect("download failure event timeout")
        .expect("download failure event")
    {
        ServiceEvent::ReplayDownloadFailed {
            operation_id: failed_operation_id,
            phase,
            message,
        } => {
            assert_eq!(failed_operation_id, operation_id);
            assert_eq!(phase, ReplayDownloadPhase::Ready);
            assert!(message.contains("end date cannot be before start date"));
        }
        other => panic!("expected replay download failure, got {other:?}"),
    }
    assert!(state.session.is_some());
    assert!(state.market_task.is_none());
    assert!(state.user_task.is_none());
}

#[cfg(feature = "replay")]
#[tokio::test]
async fn replay_read_only_jobs_dispatch_without_blocking_live_service_state() {
    let mut state = test_state(test_session());
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let mut config = AppConfig::default();
    config.token_path = std::env::temp_dir().join("trader-missing-replay-token.json");
    let exact_contract = ContractSuggestion {
        id: 4_399_631,
        name: "MESU6".to_string(),
        description: "Micro E-mini S&P".to_string(),
        raw: json!({"id": 4_399_631, "name": "MESU6"}),
    };
    let commands = vec![
        ServiceCommand::SearchReplayDownloadContracts {
            operation_id: ReplayDownloadOperationId::next(),
            config: config.clone(),
            query: "MES".to_string(),
            limit: 12,
        },
        ServiceCommand::InspectReplayDownloadContract {
            operation_id: ReplayDownloadOperationId::next(),
            config: config.clone(),
            contract: exact_contract.clone(),
        },
        ServiceCommand::DownloadReplayData {
            operation_id: ReplayDownloadOperationId::next(),
            config,
            instrument: "MES".to_string(),
            contract: exact_contract,
            target: None,
            start_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 23).expect("start"),
            end_date: chrono::NaiveDate::from_ymd_opt(2026, 7, 23).expect("end"),
            source_kind: "server-bars".to_string(),
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            display_name: None,
            tags: Vec::new(),
        },
    ];

    for command in commands {
        let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            handle_command(command, &mut state, &event_tx, &market_tx, internal_tx),
        )
        .await
        .expect("read-only replay command dispatch should return immediately")
        .expect("read-only replay command should dispatch");
    }

    let session = state.session.expect("live session state remains available");
    assert_eq!(session.selected_account_id, Some(42));
    assert_eq!(session.selected_contract.expect("contract").name, "ESM6");
    assert!(state.user_task.is_none());
    assert!(state.market_task.is_none());
}

#[tokio::test]
async fn config_sync_does_not_disarm_armed_native_strategy() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(123);
    session.execution_runtime.last_summary = "Armed before edit".to_string();
    let mut state = test_state(session);
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut config = ExecutionStrategyConfig::default();
    config.kind = StrategyKind::Native;
    config.native_ema.fast_length = 8;

    handle_command(
        ServiceCommand::SetExecutionStrategyConfig(config),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("config sync should succeed");

    let session = state.session.expect("session should persist");
    assert!(session.execution_runtime.armed);
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(123));
    assert_eq!(session.execution_runtime.last_summary, "Armed before edit");
    assert_eq!(session.execution_config.native_ema.fast_length, 8);
}

#[tokio::test]
async fn stale_market_order_interrupt_recovers_and_rearms_signal() {
    let stale_strategy_id = 453147950116_i64;
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 6400.0,
        high: 6401.0,
        low: 6399.0,
        close: 6400.5,
        volume: None,
    }];
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: stale_strategy_id,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-stale-direct-reversal".to_string(),
        order_id: Some(77),
        order_strategy_id: Some(stale_strategy_id),
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    session.user_store.order_strategies.insert(
        stale_strategy_id,
        json!({
            "id": stale_strategy_id,
            "accountId": 42,
            "contractId": 3570918,
            "status": "Working"
        }),
    );

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_internal(
        InternalEvent::BrokerOrderFailed(BrokerOrderFailure {
            endpoint: "orderStrategy/interruptorderstrategy",
            cl_ord_id: "midas-stale-direct-reversal".to_string(),
            message: format!(
                "strategy {stale_strategy_id} was already inactive; waiting for broker sync before retrying the reversal"
            ),
            target_qty: Some(-1),
            stale_interrupt: true,
        }),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("stale market interrupt should recover");

    let session = state.session.expect("session should persist");
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert!(session.active_order_strategy.is_none());
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(199));
    assert_eq!(
        session.execution_runtime.last_summary,
        "Previous strategy was already inactive; retrying current signal after broker sync."
    );
    assert!(
        !session
            .user_store
            .order_strategies
            .contains_key(&stale_strategy_id)
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("submit stale")
                && message.contains("endpoint orderStrategy/interruptorderstrategy")
                && message.contains("clOrdId midas-stale-direct-reversal")
                && message.contains("target Some(-1)")
                && message.contains("already inactive")
                && message.contains("pending target none")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Status(message)
            if message.contains("already inactive")
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ServiceEvent::Error(_)))
    );
}

#[tokio::test]
async fn order_strategy_submit_failure_debug_logs_request_and_target() {
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(-1);
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("hma_cross Sell (qty 1 -> -1)".to_string()),
        cl_ord_id: "midas-start-fail".to_string(),
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_internal(
        InternalEvent::OrderStrategyFailed(BrokerOrderStrategyFailure {
            endpoint: "orderStrategy/startorderstrategy",
            uuid: "midas-start-fail".to_string(),
            message: "broker rejected entry".to_string(),
            target_qty: -1,
            stale_interrupt: false,
        }),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("order strategy failure should be handled");

    let session = state.session.expect("session should persist");
    assert!(!session.order_submit_in_flight);
    assert!(session.order_latency_tracker.is_none());
    assert_eq!(session.execution_runtime.pending_target_qty, None);
    assert_eq!(
        session.execution_runtime.last_summary,
        "broker rejected entry"
    );

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message)
            if message.contains("submit failed")
                && message.contains("endpoint orderStrategy/startorderstrategy")
                && message.contains("uuid midas-start-fail")
                && message.contains("target -1")
                && message.contains("broker rejected entry")
                && message.contains("pending target none")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServiceEvent::Error(message)
            if message == "broker rejected entry"
    )));
}

#[tokio::test]
async fn pending_target_watchdog_respects_order_strategy_position_sync_grace() {
    let mut session = test_session();
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(200);
    session.market.history_loaded = 1;
    session.market.bars = vec![Bar {
        ts_ns: 200,
        open: 6400.0,
        high: 6401.0,
        low: 6399.0,
        close: 6400.5,
        volume: None,
    }];
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now() - Duration::from_secs(3),
        signal_started_at: Some(time::Instant::now()),
        signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
        cl_ord_id: "midas-strategy-position-sync".to_string(),
        order_id: Some(77),
        order_strategy_id: Some(88),
        seen_recorded: true,
        exec_report_recorded: true,
        fill_recorded: true,
    });

    let mut state = test_state(session);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_internal(
        InternalEvent::PendingTargetWatchdog,
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("watchdog should preserve order-strategy pending target during sync grace");

    let session = state.session.expect("session should persist");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert!(session.order_latency_tracker.is_some());
    assert!(session.execution_runtime.last_summary.is_empty());

    let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
    assert!(!events.iter().any(|event| matches!(
        event,
        ServiceEvent::DebugLog(message) if message.contains("pending target cleared")
    )));
}

#[tokio::test]
async fn set_target_position_records_pending_target_for_staged_reversal() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_config.native_hma.take_profit_ticks = 30.0;
    session.execution_config.native_hma.stop_loss_ticks = 30.0;
    session.execution_config.native_reversal_mode = NativeReversalMode::FlattenConfirmEnter;
    session.execution_runtime.armed = true;
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "netPrice": 6400.0
            }),
        )]),
    );
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: 1,
    });
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: None,
        signal_context: None,
        cl_ord_id: "midas-live-strategy".to_string(),
        order_id: None,
        order_strategy_id: Some(77),
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });

    let (broker_tx, mut _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    let mut state = ServiceState {
        client: Client::builder().build().expect("client"),
        broker_tx,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        replay_lookup_job: None,
        replay_download_job: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    };
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();

    handle_command(
        ServiceCommand::SetTargetPosition {
            target_qty: -1,
            automated: true,
            reason: "test staged reversal".to_string(),
        },
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("staged reversal target should queue");

    let session = state.session.expect("session should persist");
    assert_eq!(session.execution_runtime.pending_target_qty, Some(0));
    assert!(session.execution_runtime.pending_reversal_entry.is_some());
    assert!(session.order_submit_in_flight);
}
