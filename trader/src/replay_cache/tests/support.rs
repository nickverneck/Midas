use super::*;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn dt(raw: &str) -> DateTime<Utc> {
    raw.parse().expect("valid timestamp")
}

pub(super) fn temp_cache_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("trader-cache-{name}-{nonce}"))
}

pub(super) fn sample_manifest() -> ReplayCacheManifest {
    ReplayCacheManifest {
        manifest_version: MANIFEST_VERSION,
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: "MES".to_string(),
            name: Some("Micro E-mini S&P 500".to_string()),
            exchange: Some("CME".to_string()),
        },
        contract: ReplayCacheContract {
            symbol: "MESU6".to_string(),
            id: Some(123),
            expiration: NaiveDate::from_ymd_opt(2026, 9, 18),
        },
        display_name: "MESU6 2026-07-23 1m HA".to_string(),
        coverage: ReplayCacheCoverage {
            start: dt("2026-07-23T13:30:00Z"),
            end: dt("2026-07-23T20:00:00Z"),
            trading_date: NaiveDate::from_ymd_opt(2026, 7, 23),
        },
        completed_raw_tick_coverage: None,
        completed_raw_tick_windows: Vec::new(),
        source_kind: ReplayCacheSourceKind::ServerBars,
        download_request: json!({
            "md": "getChart",
            "chartDescription": BarType::minute(1).chart_description(),
        }),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        files: vec![ReplayCacheDataFile {
            relative_path: PathBuf::from("server-bars/1m-heikin.parquet"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            format: ReplayCacheFileFormat::Parquet,
            schema_version: Some(1),
            compression: Some("zstd".to_string()),
            market_shape: ReplayCacheMarketShape {
                bar_type: Some(BarType::minute(1)),
                chart_mode: Some(CandleMode::HeikinAshi),
                session_template: Some("Globex".to_string()),
            },
            row_count: 390,
            first_timestamp: dt("2026-07-23T13:30:00Z"),
            last_timestamp: dt("2026-07-23T20:00:00Z"),
            request_start: None,
            request_end: None,
            data_hash: Some(ReplayCacheDataHash {
                algorithm: "sha256".to_string(),
                value: "abc123".to_string(),
            }),
            warnings: Vec::new(),
            errors: Vec::new(),
        }],
        app: Some(ReplayCacheAppMetadata {
            app_version: Some("0.1.0".to_string()),
            git_commit: Some("test".to_string()),
            generated_at: Some(dt("2026-07-24T00:00:00Z")),
        }),
        warnings: Vec::new(),
        errors: Vec::new(),
        badges: Vec::new(),
        available_bar_shapes: Vec::new(),
        available_chart_modes: Vec::new(),
        tags: vec!["regression".to_string()],
        notes: Some("sample".to_string()),
    }
}

pub(super) fn sample_contract_metadata() -> ReplayCacheContractMetadata {
    let snapshot = |endpoint: &str, payload: Value| ReplayCacheMetadataSnapshot {
        endpoint: endpoint.to_string(),
        fetched_at: dt("2026-07-24T00:00:00Z"),
        source_timestamp: None,
        payload,
    };
    ReplayCacheContractMetadata {
        context: ReplayCacheMetadataContext {
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_id: Some(42),
            user_name: Some("tester".to_string()),
            accounts: vec![
                ReplayCacheMetadataAccount {
                    id: 7,
                    name: "DEMO7".to_string(),
                },
                ReplayCacheMetadataAccount {
                    id: 9,
                    name: "DEMO9".to_string(),
                },
            ],
            accounts_endpoint: Some("account/list".to_string()),
            accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
        },
        contract: snapshot("contract/item", json!({"id": 123})),
        maturity: Some(snapshot(
            "contractMaturity/item",
            json!({"id": 62531, "productId": 1878809}),
        )),
        maturity_chain: Some(snapshot("contractMaturity/deps", json!([{"id": 62531}]))),
        product: Some(snapshot("product/item", json!({"id": 1878809}))),
        product_sessions: Some(snapshot("productSession/deps", json!([{"id": 1}]))),
        product_margins: Some(snapshot("productMargin/deps", json!([{"id": 2}]))),
        contract_margins: Some(snapshot("contractMargin/deps", json!([{"id": 3}]))),
        fee_params: Some(snapshot(
            "contract/getproductfeeparams",
            json!({"commission": 0.39}),
        )),
        suggested_coverage: Some(ReplayCacheSuggestedCoverage {
            start_date: NaiveDate::from_ymd_opt(2026, 6, 19).expect("date"),
            end_date: NaiveDate::from_ymd_opt(2026, 9, 18).expect("date"),
            basis: "adjacent maturities".to_string(),
            estimated: true,
        }),
    }
}

pub(super) fn bar(raw: &str, open: f64) -> Bar {
    Bar {
        ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
        open,
        high: open + 1.0,
        low: open - 1.0,
        close: open + 0.5,
        volume: Some(10.0),
    }
}

pub(super) fn bar_row_json(raw: &str, open: f64) -> String {
    json!({
        "timestamp": raw,
        "ts_ns": dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
        "open": open,
        "high": open + 1.0,
        "low": open - 1.0,
        "close": open + 0.5,
        "volume": 10.0
    })
    .to_string()
}

pub(super) fn raw_tick(raw: &str, tick_id: Option<i64>, price: f64) -> ReplayCacheRawTickRow {
    ReplayCacheRawTickRow {
        timestamp: dt(raw),
        ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
        tick_id,
        price,
        size: 2.0,
        bid_price: Some(price - 0.25),
        bid_size: Some(10.0),
        ask_price: Some(price),
        ask_size: Some(12.0),
        chart_id: Some(77),
        trade_date: Some(20260723),
        packet_source: Some("db".to_string()),
        packet_base_ts_ms: Some(1_785_000_000_000),
        packet_base_price_ticks: Some(29_700),
    }
}

pub(super) fn sequential_raw_ticks(count: usize) -> Vec<ReplayCacheRawTickRow> {
    let base = dt("2026-07-23T00:00:00Z")
        .timestamp_nanos_opt()
        .expect("timestamp ns");
    (0..count)
        .map(|index| {
            let ts_ns = base + index as i64;
            ReplayCacheRawTickRow {
                timestamp: DateTime::<Utc>::from_timestamp_nanos(ts_ns),
                ts_ns,
                tick_id: Some(index as i64 + 1),
                price: 100.0 + (index % 8) as f64 * 0.25,
                size: 1.0,
                bid_price: None,
                bid_size: None,
                ask_price: None,
                ask_size: None,
                chart_id: Some(77),
                trade_date: Some(20260723),
                packet_source: Some("db".to_string()),
                packet_base_ts_ms: None,
                packet_base_price_ticks: None,
            }
        })
        .collect()
}

pub(super) fn raw_tick_cache_write(
    root: PathBuf,
    ticks: Vec<ReplayCacheRawTickRow>,
) -> ReplayCacheRawTicksWrite {
    ReplayCacheRawTicksWrite {
        cache_root: root,
        target: None,
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: "MES".to_string(),
            name: Some("Micro E-mini S&P 500".to_string()),
            exchange: Some("CME".to_string()),
        },
        contract: ReplayCacheContract {
            symbol: "MESU6".to_string(),
            id: Some(123),
            expiration: NaiveDate::from_ymd_opt(2026, 9, 18),
        },
        request_start: dt("2026-07-23T00:00:00Z"),
        request_end: dt("2026-07-24T00:00:00Z"),
        download_request: json!({"md": "getChart", "source": "raw-ticks"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks,
        warnings: Vec::new(),
        display_name: Some("MESU6 legacy raw ticks".to_string()),
        tags: None,
        notes: None,
    }
}

pub(super) fn write_raw_ticks_with_schema(
    path: &Path,
    rows: &[ReplayCacheRawTickRow],
    schema: Arc<Schema>,
    statistics: EnabledStatistics,
    row_group_rows: usize,
) {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(statistics)
        .set_max_row_group_size(row_group_rows)
        .set_write_batch_size(1)
        .build();
    let file = File::create(path).expect("create parquet fixture");
    let mut writer =
        ArrowWriter::try_new(file, schema.clone(), Some(props)).expect("create parquet writer");
    for row in rows {
        writer
            .write(&raw_ticks_record_batch(schema.clone(), std::slice::from_ref(row)).unwrap())
            .expect("write parquet row");
    }
    writer.close().expect("close parquet fixture");
}

pub(super) fn write_legacy_v1_raw_tick_dataset(
    root: &Path,
    rows: &[ReplayCacheRawTickRow],
) -> ReplayCacheDataset {
    let dataset_dir = root.join("legacy-v1");
    let relative_path = PathBuf::from("raw-ticks/legacy-v1.parquet");
    let data_path = dataset_dir.join(&relative_path);
    fs::create_dir_all(data_path.parent().expect("raw tick parent"))
        .expect("create legacy cache directory");
    let mut fields = raw_ticks_parquet_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[2] = Field::new("tick_id", DataType::Int64, true);
    write_raw_ticks_with_schema(
        &data_path,
        rows,
        Arc::new(Schema::new(fields)),
        EnabledStatistics::Chunk,
        2,
    );

    let mut manifest = sample_manifest();
    manifest.display_name = "MESU6 legacy v1 raw ticks".to_string();
    manifest.source_kind = ReplayCacheSourceKind::RawTicks;
    manifest.coverage = ReplayCacheCoverage {
        start: rows.first().expect("legacy rows").timestamp,
        end: rows.last().expect("legacy rows").timestamp,
        trading_date: Some(dt("2026-07-23T00:00:00Z").date_naive()),
    };
    manifest.files = vec![ReplayCacheDataFile {
        relative_path,
        source_kind: ReplayCacheSourceKind::RawTicks,
        format: ReplayCacheFileFormat::Parquet,
        schema_version: Some(RAW_TICKS_LEGACY_SCHEMA_VERSION),
        compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
        market_shape: ReplayCacheMarketShape {
            bar_type: None,
            chart_mode: None,
            session_template: Some("Globex".to_string()),
        },
        row_count: rows.len() as u64,
        first_timestamp: rows.first().expect("legacy rows").timestamp,
        last_timestamp: rows.last().expect("legacy rows").timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: fnv1a64_file_hex(&data_path).expect("hash legacy parquet"),
        }),
        warnings: vec!["Legacy schema v1 nullable tick IDs.".to_string()],
        errors: Vec::new(),
    }];
    manifest.badges = manifest.derived_badges();
    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize legacy manifest"),
    )
    .expect("write legacy manifest");
    ReplayCacheDataset {
        manifest_path,
        dataset_dir,
        manifest,
    }
}
