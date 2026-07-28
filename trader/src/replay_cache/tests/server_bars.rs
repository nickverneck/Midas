use super::*;
use serde_json::json;

#[test]
fn streaming_file_hash_matches_in_memory_hash_across_chunks() {
    let root = temp_cache_dir("streaming-hash");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("large.bin");
    let bytes = (0..(128 * 1024 + 17))
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    fs::write(&path, &bytes).expect("write hash fixture");

    assert_eq!(
        fnv1a64_file_hex(&path).expect("streaming hash"),
        fnv1a64_hex(&bytes)
    );
}

#[test]
fn server_bar_parquet_file_round_trips_and_is_preferred_over_jsonl() {
    let root = temp_cache_dir("server-parquet");
    let write = ReplayCacheServerBarsWrite {
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
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({"md": "getChart"}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        bars: vec![bar("2026-07-23T00:00:00Z", 100.0)],
        warnings: Vec::new(),
        display_name: Some("MES baseline".to_string()),
        tags: Some(vec![
            " hma ".to_string(),
            "baseline".to_string(),
            "hma".to_string(),
        ]),
        notes: None,
    };
    let parquet = write_server_bars_parquet_cache(write.clone()).expect("write parquet");
    let jsonl =
        write_server_bars_jsonl_cache(write.clone()).expect("write jsonl compatibility file");
    assert_eq!(parquet.row_count, 1);
    assert_eq!(jsonl.row_count, 1);
    assert_eq!(
        parquet.data_path.extension().and_then(|ext| ext.to_str()),
        Some("parquet")
    );

    let library = ReplayCacheLibrary::scan(root.clone());
    let loaded = library
        .load_first_server_bars(BarType::minute(1), CandleMode::Standard, None)
        .expect("load parquet-preferred server bars")
        .expect("server bars exist");
    assert_eq!(loaded.file.format, ReplayCacheFileFormat::Parquet);
    assert_eq!(loaded.bars[0].close, 100.5);

    let manifest = ReplayCacheManifest::from_path(&loaded.manifest_path).expect("manifest");
    assert!(
        manifest
            .files
            .iter()
            .any(|file| file.format == ReplayCacheFileFormat::Jsonl)
    );
    assert!(
        manifest
            .files
            .iter()
            .any(|file| file.format == ReplayCacheFileFormat::Parquet)
    );
    assert_eq!(manifest.row_count_total(), 2);
    assert_eq!(manifest.preferred_row_count_total(), 1);
    assert_eq!(manifest.display_name, "MES baseline");
    assert_eq!(manifest.tags, vec!["baseline", "hma"]);

    let old_parquet_path = parquet.data_path.clone();
    let mut refresh_write = write;
    refresh_write.bars = vec![
        bar("2026-07-23T00:00:00Z", 101.0),
        bar("2026-07-23T00:01:00Z", 102.0),
    ];
    let refreshed =
        write_server_bars_parquet_cache(refresh_write.clone()).expect("refresh parquet cache");
    assert_ne!(refreshed.data_path, old_parquet_path);
    assert!(!old_parquet_path.exists());
    assert!(refreshed.data_path.exists());
    let refreshed_manifest =
        ReplayCacheManifest::from_path(&refreshed.manifest_path).expect("refreshed manifest");
    assert_eq!(
        refreshed_manifest
            .files
            .iter()
            .filter(|file| file.format == ReplayCacheFileFormat::Parquet)
            .count(),
        1
    );
    assert_eq!(refreshed_manifest.preferred_row_count_total(), 2);

    let original_dataset_dir = refreshed.dataset_dir.clone();
    let mut backward_extension = refresh_write;
    backward_extension.target = Some(ReplayDownloadCacheTarget {
        dataset_dir: original_dataset_dir.clone(),
        manifest_path: refreshed.manifest_path.clone(),
    });
    backward_extension.request_start = dt("2026-07-22T00:00:00Z");
    backward_extension.request_end = dt("2026-07-25T00:00:00Z");
    backward_extension.bars = vec![
        bar("2026-07-22T00:00:00Z", 99.0),
        bar("2026-07-24T23:59:00Z", 103.0),
    ];
    let extended = write_server_bars_parquet_cache(backward_extension)
        .expect("extend selected dataset backward");
    assert_eq!(extended.dataset_dir, original_dataset_dir);
    assert!(!root.join("tradovate/sim/MES/MESU6/2026-07-22").exists());
    let extended_manifest =
        ReplayCacheManifest::from_path(&extended.manifest_path).expect("extended manifest");
    assert_eq!(extended_manifest.coverage.start, dt("2026-07-22T00:00:00Z"));
    assert_eq!(extended_manifest.coverage.end, dt("2026-07-24T23:59:00Z"));

    let identity_err = replay_cache_write_dataset_dir(
        &root,
        BrokerKind::Tradovate,
        TradingEnvironment::Sim,
        "ES",
        &ReplayCacheContract {
            symbol: "ESU6".to_string(),
            id: Some(999),
            expiration: None,
        },
        NaiveDate::from_ymd_opt(2026, 7, 22).expect("date"),
        Some(&ReplayDownloadCacheTarget {
            dataset_dir: extended.dataset_dir,
            manifest_path: extended.manifest_path,
        }),
    )
    .expect_err("extension target identity mismatch must fail closed");
    assert!(identity_err.to_string().contains("identity does not match"));
}

#[test]
fn cache_writer_rejects_extension_target_outside_configured_root() {
    let root = temp_cache_dir("target-root");
    let outside_root = temp_cache_dir("target-outside");
    fs::create_dir_all(&root).expect("create cache root");
    let outside_dataset = outside_root.join("dataset");
    fs::create_dir_all(&outside_dataset).expect("create outside dataset");
    let outside_manifest = outside_dataset.join(MANIFEST_FILE_NAME);
    fs::write(&outside_manifest, b"{}").expect("write outside manifest");

    let err = write_server_bars_parquet_cache(ReplayCacheServerBarsWrite {
        cache_root: root,
        target: Some(ReplayDownloadCacheTarget {
            dataset_dir: outside_dataset,
            manifest_path: outside_manifest,
        }),
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
        request_start: dt("2026-07-22T00:00:00Z"),
        request_end: dt("2026-07-23T00:00:00Z"),
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: None,
        bars: vec![bar("2026-07-22T00:00:00Z", 100.0)],
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    })
    .expect_err("target outside cache root must fail closed");

    assert!(err.to_string().contains("outside cache root"));
}

#[test]
fn manifest_marks_combined_server_bar_and_raw_tick_files_as_mixed() {
    let root = temp_cache_dir("mixed-sources");
    let server_write = ReplayCacheServerBarsWrite {
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
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({"md": "getChart"}),
        bar_type: BarType::minute(1),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        bars: vec![bar("2026-07-23T00:00:00Z", 100.0)],
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    };
    let server = write_server_bars_parquet_cache(server_write).expect("write server bars");
    write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
        cache_root: root,
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
        download_request: json!({"md": "getChart"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks: vec![raw_tick("2026-07-23T00:00:01Z", Some(1), 100.25)],
        warnings: Vec::new(),
        display_name: None,
        tags: None,
        notes: None,
    })
    .expect("write raw ticks");

    let manifest = ReplayCacheManifest::from_path(&server.manifest_path).expect("manifest");
    assert_eq!(manifest.source_kind, ReplayCacheSourceKind::Mixed);
    assert_eq!(
        manifest.downloadable_source_kinds(),
        vec![
            ReplayCacheSourceKind::ServerBars,
            ReplayCacheSourceKind::RawTicks
        ]
    );
    assert!(manifest.badges.contains(&"server-bars".to_string()));
    assert!(manifest.badges.contains(&"raw-ticks".to_string()));
}
