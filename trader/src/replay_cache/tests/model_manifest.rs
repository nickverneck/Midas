use super::*;
use serde_json::json;

#[test]
fn manifest_parsing_derives_library_badges_and_shapes() {
    let manifest = sample_manifest();
    let raw = serde_json::to_string(&manifest).expect("serialize manifest");
    let mut parsed: ReplayCacheManifest = serde_json::from_str(&raw).expect("parse manifest");
    parsed.normalize_derived_fields();

    assert_eq!(parsed.provider, BrokerKind::Tradovate);
    assert_eq!(parsed.row_count_total(), 390);
    assert_eq!(parsed.available_bar_shapes, vec![BarType::minute(1)]);
    assert_eq!(parsed.available_chart_modes, vec![CandleMode::HeikinAshi]);
    assert!(parsed.badges.contains(&"server-bars".to_string()));
    assert!(parsed.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
    assert!(!parsed.supports_replay(BarType::minute(5), CandleMode::HeikinAshi, None));
}

#[test]
fn manifest_round_trips_contract_metadata_sources_and_suggested_coverage() {
    let mut manifest = sample_manifest();
    manifest.contract_metadata = Some(ReplayCacheContractMetadata {
        context: ReplayCacheMetadataContext {
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_id: Some(42),
            user_name: Some("tester".to_string()),
            accounts: vec![ReplayCacheMetadataAccount {
                id: 7,
                name: "DEMO7".to_string(),
            }],
            accounts_endpoint: Some("account/list".to_string()),
            accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
        },
        contract: ReplayCacheMetadataSnapshot {
            endpoint: "contract/item".to_string(),
            fetched_at: dt("2026-07-24T00:00:00Z"),
            source_timestamp: Some(dt("2026-07-23T23:59:00Z")),
            payload: json!({
                "id": 123,
                "contractMaturityId": 62531,
                "providerTickSize": 0.25
            }),
        },
        maturity: Some(ReplayCacheMetadataSnapshot {
            endpoint: "contractMaturity/item".to_string(),
            fetched_at: dt("2026-07-24T00:00:01Z"),
            source_timestamp: None,
            payload: json!({
                "id": 62531,
                "productId": 1878809,
                "expirationDate": "2026-09-18T13:30:00Z"
            }),
        }),
        maturity_chain: None,
        product: None,
        product_sessions: None,
        product_margins: None,
        contract_margins: None,
        fee_params: Some(ReplayCacheMetadataSnapshot {
            endpoint: "contract/getproductfeeparams".to_string(),
            fetched_at: dt("2026-07-24T00:00:02Z"),
            source_timestamp: None,
            payload: json!({"commission": 0.39, "dayMargin": 1380.98}),
        }),
        suggested_coverage: Some(ReplayCacheSuggestedCoverage {
            start_date: NaiveDate::from_ymd_opt(2026, 6, 19).expect("date"),
            end_date: NaiveDate::from_ymd_opt(2026, 9, 18).expect("date"),
            basis: "adjacent maturities".to_string(),
            estimated: true,
        }),
    });

    let raw = serde_json::to_string(&manifest).expect("serialize manifest");
    let parsed: ReplayCacheManifest = serde_json::from_str(&raw).expect("parse manifest");
    let metadata = parsed.contract_metadata.expect("contract metadata");

    assert_eq!(metadata.context.accounts[0].name, "DEMO7");
    assert_eq!(metadata.contract.endpoint, "contract/item");
    assert_eq!(
        metadata.fee_params.expect("fee params").payload["commission"],
        0.39
    );
    assert_eq!(
        metadata
            .suggested_coverage
            .expect("coverage")
            .start_date
            .to_string(),
        "2026-06-19"
    );
}

#[test]
fn metadata_merge_reuses_missing_snapshots_only_for_the_same_complete_identity() {
    let existing = sample_contract_metadata();
    let mut incoming = existing.clone();
    incoming.context.accounts.reverse();
    incoming.product_sessions = None;
    incoming.product_margins = None;
    incoming.contract_margins = None;
    incoming.fee_params = None;
    incoming.suggested_coverage = None;

    let merged =
        merge_contract_metadata(Some(existing), Some(incoming)).expect("same identity metadata");

    assert!(merged.product_sessions.is_some());
    assert!(merged.product_margins.is_some());
    assert!(merged.contract_margins.is_some());
    assert!(merged.fee_params.is_some());
    assert!(merged.suggested_coverage.is_some());
}

#[test]
fn metadata_merge_does_not_cross_provider_or_account_contract_product_identity() {
    let existing = sample_contract_metadata();
    let partial = || {
        let mut metadata = sample_contract_metadata();
        metadata.product_sessions = None;
        metadata.product_margins = None;
        metadata.contract_margins = None;
        metadata.fee_params = None;
        metadata.suggested_coverage = None;
        metadata
    };
    let assert_not_inherited = |incoming: ReplayCacheContractMetadata| {
        let merged = merge_contract_metadata(Some(existing.clone()), Some(incoming))
            .expect("incoming metadata");
        assert!(merged.product_sessions.is_none());
        assert!(merged.product_margins.is_none());
        assert!(merged.contract_margins.is_none());
        assert!(merged.fee_params.is_none());
        assert!(merged.suggested_coverage.is_none());
    };

    let mut changed = partial();
    changed.context.provider = BrokerKind::Ironbeam;
    assert_not_inherited(changed);

    let mut changed = partial();
    changed.context.env = TradingEnvironment::Live;
    assert_not_inherited(changed);

    let mut changed = partial();
    changed.context.user_id = Some(99);
    assert_not_inherited(changed);

    let mut changed = partial();
    changed.context.accounts[0].id = 99;
    assert_not_inherited(changed);

    let mut changed = partial();
    changed.contract.payload["id"] = json!(456);
    assert_not_inherited(changed);

    let mut changed = partial();
    changed.product.as_mut().expect("product").payload["id"] = json!(999);
    changed.maturity.as_mut().expect("maturity").payload["productId"] = json!(999);
    assert_not_inherited(changed);
}

#[test]
fn metadata_merge_fails_closed_when_identity_is_unavailable() {
    let existing = sample_contract_metadata();
    let mut incoming = sample_contract_metadata();
    incoming.context.accounts.clear();
    incoming.fee_params = None;

    let merged =
        merge_contract_metadata(Some(existing.clone()), Some(incoming)).expect("incoming metadata");
    assert!(merged.fee_params.is_none());
    assert!(merge_contract_metadata(Some(existing), None).is_none());
}

#[test]
fn manifest_errors_make_dataset_unservable() {
    let mut manifest = sample_manifest();
    manifest.errors.push("partial download".to_string());

    assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));

    let mut manifest = sample_manifest();
    manifest.files[0]
        .errors
        .push("missing data page".to_string());

    assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
}

#[test]
fn cache_library_discovers_manifests_without_reading_data_files() {
    let root = temp_cache_dir("scan");
    let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
    fs::create_dir_all(dataset_dir.join("server-bars")).expect("create dirs");
    fs::write(
        dataset_dir.join(MANIFEST_FILE_NAME),
        serde_json::to_vec_pretty(&sample_manifest()).expect("serialize manifest"),
    )
    .expect("write manifest");
    fs::write(
        dataset_dir.join("server-bars/1m-heikin.parquet"),
        b"not parquet",
    )
    .expect("write ignored data file");

    let library = ReplayCacheLibrary::scan(&root);

    assert_eq!(library.datasets.len(), 1);
    assert!(library.warnings.is_empty());
    assert_eq!(
        library.datasets[0].manifest.display_name,
        "MESU6 2026-07-23 1m HA"
    );
    assert!(
        library
            .first_serving(BarType::minute(1), CandleMode::HeikinAshi, None)
            .is_some()
    );
}

#[test]
fn server_bar_cache_path_shape_is_deterministic_and_sanitized() {
    let root = PathBuf::from("/tmp/cache-root");
    let dir = replay_cache_dataset_dir(
        &root,
        BrokerKind::Tradovate,
        TradingEnvironment::Sim,
        "ME S",
        "MES/U6",
        NaiveDate::from_ymd_opt(2026, 7, 23).expect("date"),
    );
    let relative = server_bars_relative_path(
        dt("2026-07-23T00:00:00Z"),
        dt("2026-07-25T00:00:00Z"),
        BarType::volume(6500),
    );

    assert_eq!(
        dir,
        PathBuf::from("/tmp/cache-root/tradovate/sim/ME_S/MESU6/2026-07-23")
    );
    assert_eq!(
        relative,
        PathBuf::from("server-bars/2026-07-23_to_2026-07-25_6500volume.jsonl")
    );
}

#[test]
fn normalize_server_bars_filters_sorts_and_deduplicates_rows() {
    let rows = normalize_server_bar_rows(vec![
        bar("2026-07-23T00:01:00Z", 2.0),
        Bar {
            open: f64::NAN,
            ..bar("2026-07-23T00:02:00Z", 3.0)
        },
        bar("2026-07-23T00:00:00Z", 1.0),
        bar("2026-07-23T00:01:00Z", 4.0),
    ]);

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].open, 1.0);
    assert_eq!(rows[1].open, 2.0);
}

#[test]
fn normalize_raw_ticks_sorts_deduplicates_ids_and_drops_bad_rows() {
    let mut bad = raw_tick("2026-07-23T00:02:00Z", Some(3), 100.0);
    bad.size = f64::NAN;
    let normalized = normalize_raw_tick_rows(vec![
        raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
        bad,
        raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
        raw_tick("2026-07-23T00:01:01Z", Some(2), 102.0),
    ]);

    assert_eq!(normalized.rows.len(), 2);
    assert_eq!(normalized.duplicate_tick_ids, 1);
    assert_eq!(normalized.dropped_rows, 1);
    assert_eq!(normalized.rows[0].tick_id, Some(1));
    assert_eq!(normalized.rows[1].tick_id, Some(2));
}

#[test]
fn raw_tick_loader_fails_closed_when_multiple_datasets_match() {
    let root = temp_cache_dir("ambiguous-raw-ticks");
    let mut first_manifest = sample_manifest();
    first_manifest.display_name = "MESU6 raw ticks".to_string();
    first_manifest.source_kind = ReplayCacheSourceKind::RawTicks;
    first_manifest.files[0].source_kind = ReplayCacheSourceKind::RawTicks;
    first_manifest.files[0].format = ReplayCacheFileFormat::Parquet;
    first_manifest.files[0].schema_version = Some(RAW_TICKS_SCHEMA_VERSION);
    first_manifest.files[0].data_hash = Some(ReplayCacheDataHash {
        algorithm: "fnv1a64".to_string(),
        value: "0000000000000000".to_string(),
    });
    first_manifest.files[0].market_shape = ReplayCacheMarketShape {
        bar_type: None,
        chart_mode: None,
        session_template: Some("Globex".to_string()),
    };

    let mut second_manifest = first_manifest.clone();
    second_manifest.display_name = "ESU6 raw ticks".to_string();
    second_manifest.instrument.symbol = "ES".to_string();
    second_manifest.contract.symbol = "ESU6".to_string();

    let library = ReplayCacheLibrary {
        root: root.clone(),
        datasets: vec![
            ReplayCacheDataset {
                manifest_path: root.join("first/manifest.json"),
                dataset_dir: root.join("first"),
                manifest: first_manifest,
            },
            ReplayCacheDataset {
                manifest_path: root.join("second/manifest.json"),
                dataset_dir: root.join("second"),
                manifest: second_manifest,
            },
        ],
        warnings: Vec::new(),
    };

    let err = library
        .load_unique_raw_ticks_parquet(None)
        .expect_err("ambiguous raw tick datasets must not be selected implicitly");
    assert!(
        err.to_string()
            .contains("multiple cached raw-tick datasets")
    );
    assert!(err.to_string().contains("MESU6 raw ticks"));
    assert!(err.to_string().contains("ESU6 raw ticks"));
}
