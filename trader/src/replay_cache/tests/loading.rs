use super::*;
use serde_json::json;

#[test]
fn write_raw_ticks_parquet_cache_writes_manifest_without_direct_replay_support() {
    let root = temp_cache_dir("raw-parquet-cache");
    let outcome = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
        cache_root: root.clone(),
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
        request_start: dt("2026-07-23T00:00:00Z"),
        request_end: dt("2026-07-24T00:00:00Z"),
        download_request: json!({
            "md": "getChart",
            "chartDescription": {
                "underlyingType": "Tick",
                "elementSize": 1,
                "elementSizeUnit": "UnderlyingUnits"
            }
        }),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks: vec![
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
            raw_tick("2026-07-23T00:00:01Z", Some(1), 100.25),
        ],
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: Some("raw tick test".to_string()),
    })
    .expect("write raw tick cache");

    assert_eq!(outcome.row_count, 1);
    assert!(outcome.data_path.exists());
    assert_eq!(
        outcome
            .data_path
            .extension()
            .and_then(|value| value.to_str()),
        Some("parquet")
    );

    let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
    assert_eq!(manifest.source_kind, ReplayCacheSourceKind::RawTicks);
    assert_eq!(manifest.files[0].format, ReplayCacheFileFormat::Parquet);
    assert_eq!(
        manifest.files[0].schema_version,
        Some(RAW_TICKS_SCHEMA_VERSION)
    );
    assert_eq!(
        manifest.files[0].compression.as_deref(),
        Some(PARQUET_COMPRESSION_LABEL)
    );
    assert!(manifest.badges.contains(&"raw-ticks".to_string()));
    assert!(
        manifest
            .warnings
            .iter()
            .any(|warning| warning.contains("duplicate raw tick id"))
    );
    assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::Standard, None));

    let loaded_ticks = read_raw_ticks_parquet_file(&outcome.data_path).expect("read ticks");
    assert_eq!(loaded_ticks.len(), 1);
    assert_eq!(loaded_ticks[0].tick_id, Some(1));
}

#[test]
fn read_server_bars_jsonl_file_parses_sorts_and_deduplicates_rows() {
    let root = temp_cache_dir("read-jsonl");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("bars.jsonl");
    fs::write(
        &path,
        [
            bar_row_json("2026-07-23T00:01:00Z", 2.0),
            bar_row_json("2026-07-23T00:00:00Z", 1.0),
            bar_row_json("2026-07-23T00:01:00Z", 2.0),
        ]
        .join("\n"),
    )
    .expect("write jsonl");

    let bars = read_server_bars_jsonl_file(&path).expect("read bars");

    assert_eq!(bars.len(), 2);
    assert!(bars[0].ts_ns < bars[1].ts_ns);
    assert_eq!(bars[0].open, 1.0);
    assert_eq!(bars[1].open, 2.0);
}

#[test]
fn read_server_bars_jsonl_file_rejects_invalid_rows() {
    let root = temp_cache_dir("read-bad-jsonl");
    fs::create_dir_all(&root).expect("create temp dir");
    let bad_ohlc = root.join("bad-ohlc.jsonl");
    fs::write(
        &bad_ohlc,
        json!({
            "timestamp": "2026-07-23T00:00:00Z",
            "ts_ns": dt("2026-07-23T00:00:00Z").timestamp_nanos_opt().expect("timestamp ns"),
            "open": 10.0,
            "high": 9.0,
            "low": 8.0,
            "close": 8.5
        })
        .to_string(),
    )
    .expect("write bad ohlc");
    let err = read_server_bars_jsonl_file(&bad_ohlc).expect_err("bad ohlc should fail");
    assert!(err.to_string().contains("validate JSONL bar line 1"));

    let bad_timestamp = root.join("bad-timestamp.jsonl");
    fs::write(
        &bad_timestamp,
        json!({
            "timestamp": "2026-07-23T00:00:00Z",
            "ts_ns": dt("2026-07-23T00:01:00Z").timestamp_nanos_opt().expect("timestamp ns"),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5
        })
        .to_string(),
    )
    .expect("write bad timestamp");
    let err =
        read_server_bars_jsonl_file(&bad_timestamp).expect_err("timestamp mismatch should fail");
    assert!(err.to_string().contains("validate JSONL bar line 1"));
}

#[test]
fn server_bars_jsonl_resolver_rejects_path_escape() {
    let root = temp_cache_dir("resolve-escape");
    let mut manifest = sample_manifest();
    manifest.files[0].relative_path = PathBuf::from("../escape.jsonl");
    manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
    manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.files[0].market_shape.chart_mode = None;
    let dataset = ReplayCacheDataset {
        manifest_path: root.join(MANIFEST_FILE_NAME),
        dataset_dir: root,
        manifest,
    };

    let err = dataset
        .resolve_server_bars_jsonl_file(BarType::minute(1), CandleMode::Standard, None)
        .expect_err("escaping relative path should fail");

    assert!(
        err.to_string()
            .contains("not a safe manifest-relative path")
    );
}

#[test]
fn load_server_bars_jsonl_cache_file_validates_manifest_metadata() {
    let root = temp_cache_dir("metadata-mismatch");
    let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
    let data_dir = dataset_dir.join("server-bars");
    fs::create_dir_all(&data_dir).expect("create data dir");
    let relative_path = PathBuf::from("server-bars/bars.jsonl");
    fs::write(
        dataset_dir.join(&relative_path),
        bar_row_json("2026-07-23T00:00:00Z", 1.0),
    )
    .expect("write jsonl");

    let mut manifest = sample_manifest();
    manifest.files[0].relative_path = relative_path;
    manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
    manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.files[0].market_shape.chart_mode = None;
    manifest.files[0].row_count = 2;
    manifest.files[0].first_timestamp = dt("2026-07-23T00:00:00Z");
    manifest.files[0].last_timestamp = dt("2026-07-23T00:00:00Z");
    let dataset = ReplayCacheDataset {
        manifest_path: dataset_dir.join(MANIFEST_FILE_NAME),
        dataset_dir,
        manifest,
    };

    let err = load_server_bars_jsonl_cache_file(
        &dataset,
        BarType::minute(1),
        CandleMode::HeikinAshi,
        None,
    )
    .expect_err("row count mismatch should fail");

    assert!(err.to_string().contains("row count mismatch"));
}
