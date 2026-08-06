use super::super::bars::*;
use super::super::load::replay_state_from_cached_raw_ticks;
use super::super::state::{ReplayDataSource, ReplayState};
use super::super::ticks::{ReplayTick, replay_tick_from_cache_row};
use super::super::*;
use super::support::*;
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheInstrument, ReplayCacheLibrary, ReplayCacheRawTickRow,
    ReplayCacheRawTicksWrite, ReplayCacheTickSpecs, write_raw_ticks_parquet_cache,
    write_raw_ticks_parquet_file_with_limits,
};
use std::sync::Arc;

#[test]
fn time_bar_builder_groups_ticks_by_interval() {
    let mut builder = TimeBarBuilder::new(2_000_000_000);
    let base = 1_700_000_000_000_000_000i64;
    builder.push_tick(base, 100.0);
    builder.push_tick(base + 1_000_000_000, 101.0);
    builder.push_tick(base + 2_000_000_000, 99.5);
    let bars = builder.finish();

    assert_eq!(bars.len(), 2);
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].high, 101.0);
    assert_eq!(bars[0].close, 101.0);
    assert_eq!(bars[1].open, 99.5);
}

#[test]
fn tick_count_bar_builder_groups_by_number_of_ticks() {
    let mut builder = TickCountBarBuilder::new(2);
    let base = 1_700_000_000_000_000_000i64;
    builder.push_tick(base, 100.0);
    builder.push_tick(base + 1, 101.0);
    builder.push_tick(base + 2, 99.5);
    let bars = builder.finish();

    assert_eq!(bars.len(), 2);
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].high, 101.0);
    assert_eq!(bars[0].close, 101.0);
    assert_eq!(bars[1].open, 99.5);
    assert_eq!(bars[1].close, 99.5);
}

#[test]
fn range_bar_builder_rolls_on_one_tick_range_breaks() {
    let mut builder = RangeBarBuilder::new(0.25);
    let base = 1_700_000_000_000_000_000i64;
    builder.push_tick(base, 100.0);
    builder.push_tick(base + 1, 100.25);
    builder.push_tick(base + 2, 100.5);
    let bars = builder.finish();

    assert!(bars.len() >= 2);
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].close, 100.25);
    assert_eq!(bars[1].open, 100.25);
}

#[test]
fn replay_state_derives_requested_local_file_bar_types() {
    let base = 1_700_000_000_000_000_000i64;
    let state = ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract: ContractSuggestion {
            id: 1,
            name: "MESU6".to_string(),
            description: "test replay".to_string(),
            raw: json!({}),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({}),
        },
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::PriceTicks(Arc::from(
            vec![
                ReplayTick {
                    ts_ns: base,
                    last: 100.0,
                    size: None,
                },
                ReplayTick {
                    ts_ns: base + 1_000_000_000,
                    last: 100.25,
                    size: None,
                },
                ReplayTick {
                    ts_ns: base + 60_000_000_000,
                    last: 100.5,
                    size: None,
                },
            ]
            .into_boxed_slice(),
        )),
        shared_frames: None,
    };

    assert_eq!(state.bars_for_type(BarType::second(2)).unwrap().len(), 2);
    assert_eq!(state.bars_for_type(BarType::minute(1)).unwrap().len(), 2);
    assert_eq!(state.bars_for_type(BarType::tick(2)).unwrap().len(), 2);
    assert!(state.bars_for_type(BarType::range(1)).unwrap().len() >= 2);
    assert!(
        state
            .bars_for_type(BarType::volume(100))
            .expect_err("volume is unsupported for price-only replay")
            .to_string()
            .contains("volume bars require trade size")
    );
}

#[test]
fn raw_tick_replay_derives_volume_bars_and_preserves_trade_volume() {
    let state = ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract: ContractSuggestion {
            id: 1,
            name: "MESU6".to_string(),
            description: "raw tick replay".to_string(),
            raw: json!({}),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({}),
        },
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::RawTicks(Arc::from(
            vec![
                ReplayTick {
                    ts_ns: 1,
                    last: 100.0,
                    size: Some(60.0),
                },
                ReplayTick {
                    ts_ns: 2,
                    last: 100.25,
                    size: Some(60.0),
                },
                ReplayTick {
                    ts_ns: 3,
                    last: 100.5,
                    size: Some(30.0),
                },
            ]
            .into_boxed_slice(),
        )),
        shared_frames: None,
    };

    let bars = state.bars_for_type(BarType::volume(100)).unwrap();
    assert_eq!(bars.len(), 2);
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].high, 100.25);
    assert_eq!(bars[0].close, 100.25);
    assert_eq!(bars[0].volume, Some(100.0));
    assert_eq!(bars[1].open, 100.25);
    assert_eq!(bars[1].close, 100.5);
    assert_eq!(bars[1].volume, Some(50.0));
}

#[tokio::test]
async fn cached_raw_tick_stream_matches_memory_derivation_for_every_bar_kind() {
    let cache_root = temp_cache_dir("streaming-parity");
    let base = dt("2026-07-23T00:00:00Z");
    let prices_and_sizes = [
        (100.0, 60.0),
        (100.25, 60.0),
        (100.5, 30.0),
        (100.0, 20.0),
        (99.75, 40.0),
    ];
    let rows = prices_and_sizes
        .iter()
        .enumerate()
        .map(|(index, (price, size))| {
            let timestamp = base + chrono::Duration::seconds(index as i64 * 20);
            ReplayCacheRawTickRow {
                timestamp,
                ts_ns: timestamp.timestamp_nanos_opt().expect("timestamp"),
                tick_id: Some(index as i64 + 1),
                price: *price,
                size: *size,
                bid_price: None,
                bid_size: None,
                ask_price: None,
                ask_size: None,
                chart_id: None,
                trade_date: None,
                packet_source: None,
                packet_base_ts_ms: None,
                packet_base_price_ticks: None,
            }
        })
        .collect::<Vec<_>>();
    let outcome = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
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
            id: Some(123),
            expiration: None,
        },
        request_start: base,
        request_end: base + chrono::Duration::minutes(2),
        download_request: json!({"md": "getChart"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks: rows.clone(),
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    })
    .expect("write raw cache");
    write_raw_ticks_parquet_file_with_limits(&outcome.data_path, &rows, 2, 1)
        .expect("rewrite fixture with multiple row groups");
    let bytes = std::fs::read(&outcome.data_path).expect("read rewritten fixture");
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut manifest = crate::replay_cache::ReplayCacheManifest::from_path(&outcome.manifest_path)
        .expect("load fixture manifest");
    manifest.files[0]
        .data_hash
        .as_mut()
        .expect("fixture data hash")
        .value = format!("{hash:016x}");
    std::fs::write(
        &outcome.manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize fixture manifest"),
    )
    .expect("update fixture manifest hash");
    let library = ReplayCacheLibrary::scan(&cache_root);
    let resolved = library
        .resolve_unique_raw_ticks_parquet_files(None)
        .expect("resolve cache")
        .expect("raw dataset");
    let cached_state =
        replay_state_from_cached_raw_ticks(resolved.clone(), None, None, None, 100_000.0)
            .expect("build cached state");
    let memory_ticks = rows
        .iter()
        .map(replay_tick_from_cache_row)
        .collect::<Vec<_>>();
    let memory_state = ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract: ContractSuggestion {
            id: 1,
            name: "MESU6".to_string(),
            description: "memory parity".to_string(),
            raw: json!({}),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({}),
        },
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::RawTicks(Arc::from(memory_ticks.into_boxed_slice())),
        shared_frames: None,
    };

    for bar_type in [
        BarType::second(30),
        BarType::minute(1),
        BarType::tick(2),
        BarType::volume(100),
        BarType::range(1),
    ] {
        assert_eq!(
            derive_cached_raw_tick_bars(&resolved, None, bar_type, 0.25).expect("streaming bars"),
            memory_state.bars_for_type(bar_type).expect("memory bars"),
            "{} parity",
            bar_type.label()
        );
        let buffered = cached_state
            .frames_for_type(bar_type)
            .expect("buffered cached frames");
        let mut stream = cached_state
            .frame_stream_for_type(bar_type)
            .expect("create cached frame stream");
        let mut streamed = Vec::new();
        while let Some(frame) = stream.next().await.expect("read cached frame") {
            streamed.push(frame);
        }
        stream.finish().await.expect("finish cached frame stream");
        assert_eq!(
            streamed.len(),
            buffered.len(),
            "{} frame count",
            bar_type.label()
        );
        for (streamed, buffered) in streamed.iter().zip(buffered.iter()) {
            assert_eq!(streamed.bar, buffered.bar, "{} bar", bar_type.label());
            assert_eq!(
                streamed.ticks.as_ref(),
                buffered.ticks.as_ref(),
                "{} ticks",
                bar_type.label()
            );
        }
    }
}

#[tokio::test]
async fn cached_raw_tick_replay_stream_survives_cache_refresh() {
    let cache_root = temp_cache_dir("refresh-lease");
    let base = dt("2026-07-23T00:00:00Z");
    let make_rows = |price_offset: f64| {
        (0..3)
            .map(|index| {
                let timestamp = base + chrono::Duration::seconds(index * 30);
                ReplayCacheRawTickRow {
                    timestamp,
                    ts_ns: timestamp.timestamp_nanos_opt().expect("timestamp"),
                    tick_id: Some(index + 1),
                    price: 100.0 + price_offset + index as f64 * 0.25,
                    size: 1.0,
                    bid_price: Some(100.0 + price_offset + index as f64 * 0.25 - 0.25),
                    bid_size: Some(1.0),
                    ask_price: Some(100.0 + price_offset + index as f64 * 0.25 + 0.25),
                    ask_size: Some(1.0),
                    chart_id: None,
                    trade_date: None,
                    packet_source: None,
                    packet_base_ts_ms: None,
                    packet_base_price_ticks: None,
                }
            })
            .collect::<Vec<_>>()
    };
    let write = |ticks| ReplayCacheRawTicksWrite {
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
            id: Some(123),
            expiration: None,
        },
        request_start: base,
        request_end: base + chrono::Duration::minutes(2),
        download_request: json!({"md": "getChart"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks,
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    };

    write_raw_ticks_parquet_cache(write(make_rows(0.0))).expect("write first cache");
    let library = ReplayCacheLibrary::scan(&cache_root);
    let resolved = library
        .resolve_unique_raw_ticks_parquet_files(None)
        .expect("resolve first cache")
        .expect("first dataset");
    let old_path = resolved.files[0].data_path.clone();
    let state = replay_state_from_cached_raw_ticks(resolved.clone(), None, None, None, 100_000.0)
        .expect("build replay state");
    let second_state = replay_state_from_cached_raw_ticks(resolved, None, None, None, 100_000.0)
        .expect("build second replay state");
    assert!(matches!(
        (&state.data, &second_state.data),
        (
            ReplayDataSource::CachedRawTicks { .. },
            ReplayDataSource::CachedRawTicks { .. }
        )
    ));

    let refreshed = write_raw_ticks_parquet_cache(write(make_rows(10.0))).expect("refresh cache");
    assert_ne!(refreshed.data_path, old_path);
    assert!(
        !old_path.exists(),
        "superseded pathname should be cleaned up"
    );

    let bars = state
        .bars_for_type(BarType::minute(1))
        .expect("leased replay remains readable after unlink");
    assert_eq!(bars.len(), 2);
    assert_eq!(bars[0].open, 100.0);
    assert_eq!(bars[0].close, 100.25);
    assert_eq!(bars[1].open, 100.5);
    let frames = state
        .frames_for_type(BarType::minute(1))
        .expect("leased tick frames");
    assert_eq!(frames[0].ticks[0].bid_price, Some(99.75));
    assert_eq!(frames[0].ticks[0].ask_price, Some(100.25));

    let mut stream = state
        .frame_stream_for_type(BarType::minute(1))
        .expect("create bounded tick frame stream");
    let mut streamed_frames = Vec::new();
    while let Some(frame) = stream.next().await.expect("read streamed frame") {
        streamed_frames.push(frame);
    }
    stream.finish().await.expect("finish bounded tick stream");
    assert_eq!(streamed_frames.len(), frames.len());
    for (streamed, buffered) in streamed_frames.iter().zip(frames.iter()) {
        assert_eq!(streamed.bar, buffered.bar);
        assert_eq!(streamed.ticks.as_ref(), buffered.ticks.as_ref());
        assert_eq!(streamed.dom_updates.as_ref(), buffered.dom_updates.as_ref());
    }

    let shared = state
        .shared_frame_set_for_type(BarType::minute(1), CandleMode::Standard)
        .expect("prepare shared immutable frames");
    assert_eq!(shared.frames.as_ref(), frames.as_slice());
    assert!(
        state
            .clone()
            .with_shared_frames(shared.clone(), BarType::second(1), CandleMode::Standard)
            .is_err()
    );
    let attached = state
        .clone()
        .with_shared_frames(shared.clone(), BarType::minute(1), CandleMode::Standard)
        .expect("attach shared immutable frames");
    let second_attached = state
        .with_shared_frames(shared.clone(), BarType::minute(1), CandleMode::Standard)
        .expect("attach shared immutable frames to second candidate");
    assert!(Arc::ptr_eq(
        attached
            .shared_frames
            .as_ref()
            .expect("first shared frames"),
        second_attached
            .shared_frames
            .as_ref()
            .expect("second shared frames")
    ));
    let reused_frames = attached
        .frames_for_type(BarType::minute(1))
        .expect("reuse shared immutable frames");
    assert_eq!(reused_frames.as_slice(), shared.frames.as_ref());
    let mut shared_stream = attached
        .frame_stream_for_type(BarType::minute(1))
        .expect("create shared frame stream");
    let mut shared_frames = Vec::new();
    while let Some(frame) = shared_stream.next().await.expect("read shared frame") {
        shared_frames.push(frame);
    }
    shared_stream
        .finish()
        .await
        .expect("finish shared frame stream");
    assert_eq!(shared_frames.as_slice(), shared.frames.as_ref());
}

#[test]
fn replay_state_keeps_duplicate_timestamp_derived_bars_distinct() {
    let base = 1_700_000_000_000_000_000i64;
    let state = ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract: ContractSuggestion {
            id: 1,
            name: "MESU6".to_string(),
            description: "test replay".to_string(),
            raw: json!({}),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({}),
        },
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::PriceTicks(Arc::from(
            vec![
                ReplayTick {
                    ts_ns: base,
                    last: 100.0,
                    size: None,
                },
                ReplayTick {
                    ts_ns: base,
                    last: 100.25,
                    size: None,
                },
                ReplayTick {
                    ts_ns: base,
                    last: 101.0,
                    size: None,
                },
            ]
            .into_boxed_slice(),
        )),
        shared_frames: None,
    };

    let tick_bars = state.bars_for_type(BarType::tick(1)).unwrap();
    assert_eq!(tick_bars.len(), 3);
    assert!(
        tick_bars
            .windows(2)
            .all(|window| window[0].ts_ns < window[1].ts_ns)
    );

    let range_bars = state.bars_for_type(BarType::range(1)).unwrap();
    assert!(range_bars.len() >= 3);
    assert!(
        range_bars
            .windows(2)
            .all(|window| window[0].ts_ns < window[1].ts_ns)
    );
}

#[test]
fn cached_server_bar_state_serves_only_cached_bar_shape() {
    let base = 1_700_000_000_000_000_000i64;
    let state = ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract: ContractSuggestion {
            id: 1,
            name: "MESU6".to_string(),
            description: "cached replay".to_string(),
            raw: json!({}),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({}),
        },
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::CachedServerBars {
            bar_type: BarType::volume(6500),
            source_label: "cache fixture".to_string(),
            bars: Arc::from(
                vec![Bar {
                    ts_ns: base,
                    open: 100.0,
                    high: 101.0,
                    low: 99.0,
                    close: 100.5,
                    volume: Some(6500.0),
                }]
                .into_boxed_slice(),
            ),
        },
        shared_frames: None,
    };

    assert_eq!(state.bars_for_type(BarType::volume(6500)).unwrap().len(), 1);
    assert!(
        state
            .bars_for_type(BarType::minute(1))
            .expect_err("cached server bars are exact-shape data")
            .to_string()
            .contains("contains 6500 Vol; requested 1 Min")
    );
}
