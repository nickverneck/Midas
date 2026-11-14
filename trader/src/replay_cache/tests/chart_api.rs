use super::*;
use crate::broker::transform_bars_for_candle_mode;
use serde_json::json;

#[test]
fn chart_adapter_reads_real_cached_bars_and_pages_them() {
    let root = temp_cache_dir("chart-api");
    let write = ReplayCacheServerBarsWrite {
        cache_root: root.clone(),
        target: None,
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: "MES".into(),
            name: None,
            exchange: None,
        },
        contract: ReplayCacheContract {
            symbol: "MESU6".into(),
            id: Some(123),
            expiration: None,
        },
        request_start: dt("2026-07-23T00:00:00Z"),
        request_end: dt("2026-07-24T00:00:00Z"),
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({"fixture": false}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: None,
        bars: vec![
            bar("2026-07-23T00:00:00Z", 100.0),
            bar("2026-07-23T00:01:00Z", 101.0),
            bar("2026-07-23T00:02:00Z", 105.0),
        ],
        warnings: Vec::new(),
        display_name: Some("cached MES".into()),
        tags: None,
        notes: None,
    };
    write_server_bars_parquet_cache(write).expect("write actual cache fixture");

    let catalog = chart_api_catalog(&root);
    assert_eq!(catalog.datasets.len(), 1);
    assert_eq!(catalog.datasets[0].source_status, "stored_cache");
    let result = chart_api_read_page(
        &root,
        &catalog.datasets[0].id,
        ChartApiSeries {
            input: ChartApiInput::ServerBars,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
        },
        None,
        None,
        1,
        1,
    )
    .expect("read chart page");
    assert_eq!(result.total, 3);
    assert_eq!(result.bars.len(), 1);
    assert_eq!(result.bars[0].close, 101.5);
    assert_eq!(result.source_status, "stored_cache");

    let start = dt("2026-07-23T00:01:00Z").timestamp_nanos_opt().unwrap();
    let one_sided = chart_api_read_page(
        &root,
        &catalog.datasets[0].id,
        ChartApiSeries {
            input: ChartApiInput::ServerBars,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
        },
        Some(start),
        None,
        0,
        10,
    )
    .expect("read open-ended range");
    assert_eq!(one_sided.total, 2);
    assert_eq!(one_sided.bars[0].timestamp_ns, start);

    let raw = [
        bar("2026-07-23T00:00:00Z", 100.0),
        bar("2026-07-23T00:01:00Z", 101.0),
        bar("2026-07-23T00:02:00Z", 105.0),
    ];
    let expected_ha = transform_bars_for_candle_mode(&raw, CandleMode::HeikinAshi);
    assert!(catalog.datasets[0].series.contains(&ChartApiSeries {
        input: ChartApiInput::ServerBars,
        bar_type: BarType::minute(1),
        candle_mode: CandleMode::HeikinAshi,
    }));
    let ha_page = chart_api_read_page(
        &root,
        &catalog.datasets[0].id,
        ChartApiSeries {
            input: ChartApiInput::ServerBars,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::HeikinAshi,
        },
        Some(start),
        Some(dt("2026-07-23T00:02:00Z").timestamp_nanos_opt().unwrap()),
        0,
        10,
    )
    .expect("read ranged Heikin Ashi candles");
    assert_eq!(ha_page.total, 1);
    assert_eq!(ha_page.bars[0].open, expected_ha[1].open);
    assert_eq!(ha_page.bars[0].close, expected_ha[1].close);

    let dataset = ReplayCacheLibrary::scan(&root).datasets.remove(0);
    let manifest_path = dataset.manifest_path;
    let original = dataset.manifest;
    let mut unsupported_format = original.clone();
    unsupported_format.files[0].format = ReplayCacheFileFormat::Other;
    write_manifest_and_assert_empty(&root, &manifest_path, unsupported_format);
    let mut unsupported_schema = original.clone();
    unsupported_schema.files[0].schema_version = Some(SERVER_BARS_SCHEMA_VERSION + 1);
    write_manifest_and_assert_empty(&root, &manifest_path, unsupported_schema);
    let mut file_error = original.clone();
    file_error.files[0].errors.push("bad data file".into());
    write_manifest_and_assert_empty(&root, &manifest_path, file_error);
    let mut manifest_error = original;
    manifest_error.errors.push("bad manifest".into());
    write_manifest_and_assert_empty(&root, &manifest_path, manifest_error);
}

fn write_manifest_and_assert_empty(
    root: &Path,
    manifest_path: &Path,
    manifest: ReplayCacheManifest,
) {
    std::fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap())
        .expect("write catalog regression manifest");
    assert!(chart_api_catalog(root).datasets.is_empty());
}

#[test]
fn chart_adapter_rejects_unbounded_or_invalid_requests() {
    let err = chart_api_read_page(
        "/missing",
        "id",
        ChartApiSeries {
            input: ChartApiInput::ServerBars,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
        },
        None,
        None,
        0,
        CHART_API_MAX_PAGE_SIZE + 1,
    )
    .expect_err("oversized pages rejected before file lookup");
    assert!(err.to_string().contains("chart page limit"));
}

#[test]
fn chart_adapter_aggregates_stored_raw_ticks_without_rewriting_duplicate_times() {
    let root = temp_cache_dir("chart-api-raw-ticks");
    let mut first = raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0);
    let mut second = raw_tick("2026-07-23T00:00:00Z", Some(2), 100.25);
    first.size = 2.0;
    second.size = 3.0;
    write_raw_ticks_parquet_cache(raw_tick_cache_write(root.clone(), vec![first, second]))
        .expect("write raw-tick fixture");

    let catalog = chart_api_catalog(&root);
    let dataset = &catalog.datasets[0];
    for kind in [
        BarKind::Minute,
        BarKind::Second,
        BarKind::Tick,
        BarKind::Volume,
        BarKind::Range,
    ] {
        assert!(dataset.series.iter().any(|series| {
            series.input == ChartApiInput::RawTicks && series.bar_type.kind() == kind
        }));
    }
    let tick = chart_api_read_page(
        &root,
        &dataset.id,
        ChartApiSeries {
            input: ChartApiInput::RawTicks,
            bar_type: BarType::tick(1),
            candle_mode: CandleMode::Standard,
        },
        None,
        None,
        0,
        10,
    )
    .expect("aggregate raw ticks as tick bars");
    assert_eq!(tick.source_status, "stored_raw_ticks");
    assert_eq!(tick.total, 2);
    assert_eq!(tick.bars[0].timestamp_ns, tick.bars[1].timestamp_ns);
    assert_eq!(tick.bars[0].close, 100.0);
    assert_eq!(tick.bars[1].close, 100.25);

    let volume = chart_api_read_page(
        &root,
        &dataset.id,
        ChartApiSeries {
            input: ChartApiInput::RawTicks,
            bar_type: BarType::volume(5),
            candle_mode: CandleMode::Standard,
        },
        None,
        None,
        0,
        10,
    )
    .expect("aggregate raw ticks as volume bars");
    assert_eq!(volume.total, 1);
    assert_eq!(volume.bars[0].volume, Some(5.0));
}
