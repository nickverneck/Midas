use super::*;
use crate::replay_download::{
    DownloadCompletionEvidence, DownloadWindow, HistoricalDownloadTelemetry,
};
use serde_json::json;

fn chunk_plan(cache_root: PathBuf, request: DownloadWindow) -> ReplayCacheRawTickChunkPlanWrite {
    ReplayCacheRawTickChunkPlanWrite {
        cache_root,
        target: None,
        identity: ReplayCacheRawTickCheckpointIdentity {
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request,
            chunk_seconds: 3_600,
            request_protocol_version: 4,
        },
        download_request: json!({"md": "getChart", "source": "raw-ticks"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        warnings: Vec::new(),
        display_name: Some("resumable MES ticks".to_string()),
        tags: Some(vec!["checkpoint".to_string()]),
        notes: Some("test".to_string()),
    }
}

fn completed_telemetry(window: DownloadWindow) -> HistoricalDownloadTelemetry {
    let mut telemetry = HistoricalDownloadTelemetry::new(window);
    telemetry.end_of_history = true;
    telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
    telemetry
}

#[test]
fn completed_chunks_resume_without_redownload_and_publish_exact_coverage() {
    let root = temp_cache_dir("raw-checkpoint-resume");
    let request = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T02:00:00Z"))
        .expect("request");
    let windows = vec![
        DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T01:00:00Z"))
            .expect("first window"),
        DownloadWindow::new(dt("2026-07-23T01:00:00Z"), dt("2026-07-23T02:00:00Z"))
            .expect("second window"),
    ];
    let plan = chunk_plan(root.clone(), request);
    let initial = prepare_raw_tick_chunk_cache(&plan, &windows).expect("prepare checkpoint");
    assert_eq!(initial.checkpoint.pending_windows(), windows);

    write_raw_tick_chunk_cache(
        &plan,
        windows[0],
        vec![raw_tick("2026-07-23T00:30:00Z", Some(1), 100.0)],
        completed_telemetry(windows[0]),
    )
    .expect("commit first chunk");

    let resumed = prepare_raw_tick_chunk_cache(&plan, &windows).expect("resume checkpoint");
    assert_eq!(resumed.checkpoint.completed_leaf_count(), 1);
    assert_eq!(resumed.checkpoint.pending_windows(), vec![windows[1]]);

    write_raw_tick_chunk_cache(
        &plan,
        windows[1],
        vec![raw_tick("2026-07-23T01:30:00Z", Some(2), 100.25)],
        completed_telemetry(windows[1]),
    )
    .expect("commit second chunk");
    let outcome = finalize_raw_tick_chunk_cache(&plan).expect("publish manifest");

    assert_eq!(outcome.row_count, 2);
    assert_eq!(outcome.data_paths.len(), 2);
    let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
    assert_eq!(
        manifest.completed_raw_tick_coverage,
        Some(ReplayCacheCoverage {
            start: request.start,
            end: request.end,
            trading_date: Some(request.start.date_naive()),
        })
    );
    assert_eq!(manifest.completed_raw_tick_windows, windows);

    let resolved = ReplayCacheLibrary::scan(root)
        .resolve_unique_raw_ticks_parquet_files(None)
        .expect("resolve chunked cache")
        .expect("raw ticks");
    let mut ids = Vec::new();
    stream_resolved_raw_ticks(&resolved, None, |row| {
        ids.push(row.tick_id);
        Ok(())
    })
    .expect("stream chunked cache");
    assert_eq!(ids, vec![Some(1), Some(2)]);
}

#[test]
fn incomplete_provider_response_never_marks_a_chunk_complete() {
    let root = temp_cache_dir("raw-checkpoint-incomplete");
    let window = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T01:00:00Z"))
        .expect("window");
    let plan = chunk_plan(root, window);
    let state = prepare_raw_tick_chunk_cache(&plan, &[window]).expect("prepare checkpoint");
    let error = write_raw_tick_chunk_cache(
        &plan,
        window,
        vec![raw_tick("2026-07-23T00:30:00Z", Some(1), 100.0)],
        HistoricalDownloadTelemetry::new(window),
    )
    .expect_err("missing end-of-history must fail");
    assert!(
        error
            .to_string()
            .contains("explicit provider end-of-history")
    );
    let resumed = prepare_raw_tick_chunk_cache(&plan, &[window]).expect("reload checkpoint");
    assert_eq!(resumed.checkpoint.pending_windows(), vec![window]);
    assert_eq!(resumed.checkpoint.completed_leaf_count(), 0);
    assert_eq!(resumed.checkpoint_path, state.checkpoint_path);
}

#[test]
fn entirely_empty_provider_history_stays_retryable_and_never_publishes() {
    let root = temp_cache_dir("raw-checkpoint-empty-history");
    let window = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T01:00:00Z"))
        .expect("window");
    let plan = chunk_plan(root, window);
    prepare_raw_tick_chunk_cache(&plan, &[window]).expect("prepare checkpoint");
    write_raw_tick_chunk_cache(&plan, window, Vec::new(), completed_telemetry(window))
        .expect("record explicit empty history");

    let error = finalize_raw_tick_chunk_cache(&plan).expect_err("empty range must not publish");
    assert!(error.to_string().contains("zero provider ticks"));
    assert!(error.to_string().contains("remain retryable"));
    let resumed = prepare_raw_tick_chunk_cache(&plan, &[window]).expect("retry checkpoint");
    assert_eq!(resumed.checkpoint.pending_windows(), vec![window]);
    assert!(
        resumed.checkpoint.chunks[0]
            .attempts
            .iter()
            .any(|attempt| attempt["kind"] == "empty_provider_history")
    );
    assert!(!resumed.dataset_dir.join(MANIFEST_FILE_NAME).exists());
}

#[test]
fn distinct_coverage_requests_keep_independent_resumable_checkpoints() {
    let root = temp_cache_dir("raw-checkpoint-identities");
    let first = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T01:00:00Z"))
        .expect("first request");
    let extended = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T02:00:00Z"))
        .expect("extended request");
    let first_state = prepare_raw_tick_chunk_cache(&chunk_plan(root.clone(), first), &[first])
        .expect("first checkpoint");
    let extended_windows = vec![
        first,
        DownloadWindow::new(first.end, extended.end).expect("extension window"),
    ];
    let extended_state =
        prepare_raw_tick_chunk_cache(&chunk_plan(root, extended), &extended_windows)
            .expect("independent extended checkpoint");

    assert_ne!(first_state.checkpoint_path, extended_state.checkpoint_path);
    assert!(first_state.checkpoint_path.exists());
    assert!(extended_state.checkpoint_path.exists());
}

#[test]
fn splitting_records_sanitized_evidence_and_preserves_leaf_coverage() {
    let root = temp_cache_dir("raw-checkpoint-split");
    let request = DownloadWindow::new(dt("2026-07-23T00:00:00Z"), dt("2026-07-23T02:00:00Z"))
        .expect("request");
    let plan = chunk_plan(root, request);
    prepare_raw_tick_chunk_cache(&plan, &[request]).expect("prepare checkpoint");
    let evidence = json!({"kind": "size_rejected", "status": 413});
    let (left, right) = split_raw_tick_checkpoint_chunk(
        &plan,
        request,
        &[],
        chrono::Duration::minutes(5),
        evidence.clone(),
    )
    .expect("split checkpoint");
    let state = prepare_raw_tick_chunk_cache(&plan, &[request]).expect("reload split checkpoint");

    assert_eq!(left.start, request.start);
    assert_eq!(left.end, right.start);
    assert_eq!(right.end, request.end);
    assert_eq!(state.checkpoint.pending_windows(), vec![left, right]);
    let parent = state
        .checkpoint
        .chunks
        .iter()
        .find(|chunk| chunk.window == request)
        .expect("split parent");
    assert_eq!(parent.attempts, vec![evidence]);
}
