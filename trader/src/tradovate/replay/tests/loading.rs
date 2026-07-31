use super::super::load::*;
use super::super::*;
use super::support::*;
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheInstrument, ReplayCacheLibrary, ReplayCacheServerBarsWrite,
    ReplayCacheSourceKind, ReplayCacheTickSpecs, ReplayCacheTimeRange, ReplayDatasetSessionPreset,
    ReplayDatasetView, ReplayDatasetViewStore, ReplayDatasetWarmupPolicy,
    ReplayWarmupTradingPolicy, write_server_bars_jsonl_cache,
};
use std::path::{Path, PathBuf};

#[test]
fn replay_path_candidates_cover_launch_crate_and_workspace_roots() {
    let launch_dir = Path::new("/tmp/launch");
    let manifest_dir = Path::new("/tmp/workspace/trader");
    let candidates = replay_path_candidates(
        Path::new("market replay/ES 06-26.Last.txt"),
        Some(launch_dir),
        manifest_dir,
    );

    assert_eq!(
        candidates,
        vec![
            PathBuf::from("/tmp/launch/market replay/ES 06-26.Last.txt"),
            PathBuf::from("/tmp/workspace/trader/market replay/ES 06-26.Last.txt"),
            PathBuf::from("/tmp/workspace/market replay/ES 06-26.Last.txt"),
        ]
    );
}

#[test]
fn replay_path_candidates_support_workspace_relative_inputs() {
    let launch_dir = Path::new("/tmp/launch");
    let manifest_dir = Path::new("/tmp/workspace/trader");
    let candidates = replay_path_candidates(
        Path::new("trader/market replay/ES 06-26.Last.txt"),
        Some(launch_dir),
        manifest_dir,
    );

    assert!(
        candidates.contains(&PathBuf::from(
            "/tmp/workspace/trader/market replay/ES 06-26.Last.txt"
        )),
        "workspace-root-relative replay paths should be accepted"
    );
}

#[test]
fn load_replay_state_prefers_matching_cached_server_bars() {
    let cache_root = temp_cache_dir("cache-load");
    write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
        cache_root: cache_root.clone(),
        target: None,
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: "MES".to_string(),
            name: None,
            exchange: None,
        },
        contract: ReplayCacheContract {
            symbol: "MESU6".to_string(),
            id: Some(25866054),
            expiration: None,
        },
        request_start: dt("2026-07-23T00:00:00Z"),
        request_end: dt("2026-07-24T00:00:00Z"),
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({"source": "unit-test"}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        bars: vec![Bar {
            ts_ns: dt("2026-07-23T00:00:00Z")
                .timestamp_nanos_opt()
                .expect("timestamp ns"),
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: Some(1000.0),
        }],
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    })
    .expect("write cache");

    let mut cfg = AppConfig::default();
    cfg.replay_cache_dir = cache_root;
    cfg.replay_file_path = PathBuf::from("/tmp/trader-replay-missing-local.Last.txt");
    cfg.replay_initial_capital = 12_345.0;
    let state =
        load_replay_state_blocking(&cfg, BarType::minute(1), CandleMode::HeikinAshi, None, None)
            .expect("load replay state from cache");

    assert_eq!(replay_contract(&state).name, "MESU6");
    assert_eq!(
        state.account.raw["startingBalance"].as_f64(),
        Some(12_345.0)
    );
    let bars = state
        .bars_for_type(BarType::minute(1))
        .expect("cached bars");
    assert_eq!(bars.len(), 1);
    assert_eq!(bars[0].volume, Some(1000.0));
}

#[test]
fn load_replay_state_applies_saved_dataset_view_including_warmup() {
    let cache_root = temp_cache_dir("cache-view-load");
    let bars = (0..=4)
        .map(|minute| Bar {
            ts_ns: dt(&format!("2026-07-23T00:0{minute}:00Z"))
                .timestamp_nanos_opt()
                .expect("timestamp ns"),
            open: 100.0 + f64::from(minute),
            high: 101.0 + f64::from(minute),
            low: 99.0 + f64::from(minute),
            close: 100.5 + f64::from(minute),
            volume: Some(1000.0),
        })
        .collect();
    write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
        cache_root: cache_root.clone(),
        target: None,
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: "MES".to_string(),
            name: None,
            exchange: None,
        },
        contract: ReplayCacheContract {
            symbol: "MESU6".to_string(),
            id: Some(25866054),
            expiration: None,
        },
        request_start: dt("2026-07-23T00:00:00Z"),
        request_end: dt("2026-07-23T00:05:00Z"),
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({"source": "unit-test"}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        bars,
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    })
    .expect("write cache");

    let library = ReplayCacheLibrary::scan(&cache_root);
    let dataset = library.datasets.first().expect("dataset");
    let view = ReplayDatasetView::for_dataset(
        &cache_root,
        dataset,
        "opening_window",
        ReplayCacheTimeRange::new(dt("2026-07-23T00:02:00Z"), dt("2026-07-23T00:04:00Z"))
            .expect("evaluation range"),
        "UTC",
        ReplayDatasetSessionPreset::CustomUtc,
        ReplayDatasetWarmupPolicy {
            duration_seconds: 60,
            trading: ReplayWarmupTradingPolicy::FlatUntilEvaluation,
        },
    )
    .expect("view model");
    let view_path = ReplayDatasetViewStore::new(&cache_root)
        .save(&view)
        .expect("save view");

    let mut cfg = AppConfig::default();
    cfg.replay_cache_dir = cache_root.clone();
    let state = load_replay_state_blocking(
        &cfg,
        BarType::minute(1),
        CandleMode::Standard,
        None,
        Some(&view_path),
    )
    .expect("load replay state from view");
    let window = state.replay_window.as_ref().expect("replay window");
    assert_eq!(window.preset, "Custom UTC");
    assert_eq!(window.input_timezone, "UTC");
    assert_eq!(window.warmup_start, dt("2026-07-23T00:01:00Z"));
    assert_eq!(window.evaluation_start, dt("2026-07-23T00:02:00Z"));
    assert_eq!(window.evaluation_end, dt("2026-07-23T00:04:00Z"));
    let bars = state.bars_for_type(BarType::minute(1)).expect("view bars");

    assert_eq!(bars.len(), 3);
    assert_eq!(
        bars[0].ts_ns,
        dt("2026-07-23T00:01:00Z").timestamp_nanos_opt().unwrap()
    );
    assert_eq!(
        bars[2].ts_ns,
        dt("2026-07-23T00:03:00Z").timestamp_nanos_opt().unwrap()
    );

    std::fs::remove_dir_all(cache_root).expect("cleanup");
}
