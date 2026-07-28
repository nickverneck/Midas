#[cfg(feature = "replay")]
use super::super::commands::{
    begin_replay_cache_commit, cancel_replay_operation, reap_finished_replay_download,
    replace_replay_lookup,
};
use super::super::*;
use super::support::{test_session, test_state};
#[cfg(feature = "replay")]
use crate::broker::{ReplayDownloadOperationId, ReplayDownloadPhase};
use serde_json::json;

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
async fn replay_download_cancellation_during_commit_is_queued_and_commit_is_not_aborted() {
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
        *state
            .replay_download_job
            .as_ref()
            .unwrap()
            .cancel_tx
            .borrow(),
        "the current commit finishes, but remaining resumable chunks must observe cancellation"
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
