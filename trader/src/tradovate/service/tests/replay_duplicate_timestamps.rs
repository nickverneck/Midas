use super::super::*;
use super::support::{test_session, test_state};
use crate::broker::{ReplayEngineMode, ReplayFillModel};
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheInstrument, ReplayCacheRawTickRow, ReplayCacheRawTicksWrite,
    ReplayCacheTickSpecs, write_raw_ticks_parquet_cache,
};
use crate::tradovate::replay::{replay_accounts, replay_contract};
use chrono::{DateTime, TimeZone, Utc};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn fixture_tick(ts: DateTime<Utc>, tick_id: i64, price: f64) -> ReplayCacheRawTickRow {
    ReplayCacheRawTickRow {
        timestamp: ts,
        ts_ns: ts.timestamp_nanos_opt().expect("timestamp nanoseconds"),
        tick_id: Some(tick_id),
        price,
        size: 1.0,
        bid_price: Some(price - 0.25),
        bid_size: Some(10.0),
        ask_price: Some(price + 0.25),
        ask_size: Some(10.0),
        chart_id: Some(1),
        trade_date: None,
        packet_source: Some("cached-fixture".to_string()),
        packet_base_ts_ms: None,
        packet_base_price_ticks: None,
    }
}

#[tokio::test]
async fn cached_volume_replay_evaluates_duplicate_timestamp_bars_and_orders_fills() {
    use tokio::time::timeout;

    let cache_root: PathBuf = std::env::temp_dir().join(format!(
        "trader-replay-duplicate-volume-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos()
    ));
    let t1 = Utc
        .with_ymd_and_hms(2026, 7, 23, 14, 0, 0)
        .single()
        .unwrap();
    let t2 = Utc
        .with_ymd_and_hms(2026, 7, 23, 14, 0, 1)
        .single()
        .unwrap();
    let t3 = Utc
        .with_ymd_and_hms(2026, 7, 23, 14, 0, 2)
        .single()
        .unwrap();
    let ticks = vec![
        fixture_tick(t1, 1, 100.0),
        fixture_tick(t1, 2, 100.0), // Flat warmup bar.
        fixture_tick(t2, 3, 101.0),
        fixture_tick(t2, 4, 100.0), // Bearish bar at t2.
        fixture_tick(t2, 5, 100.0),
        fixture_tick(t2, 6, 101.0), // Bullish bar at the same t2.
        fixture_tick(t3, 7, 101.0),
        fixture_tick(t3, 8, 101.0), // Allows the second order to fill.
    ];
    let _cache = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
        cache_root: cache_root.clone(),
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
            id: Some(25866054),
            expiration: None,
        },
        request_start: t1,
        request_end: t3 + chrono::Duration::seconds(1),
        download_request: serde_json::json!({"source": "duplicate-timestamp-e2e"}),
        tick_specs: ReplayCacheTickSpecs {
            tick_size: 0.25,
            value_per_point: 5.0,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks,
        warnings: Vec::new(),
        display_name: Some("duplicate timestamp execution fixture".to_string()),
        tags: None,
        notes: None,
    })
    .expect("write cached raw-tick fixture");

    let bar_type = BarType::volume(2);
    let mut cfg = AppConfig::default();
    cfg.replay_cache_dir = cache_root.clone();
    cfg.replay_file_path = PathBuf::from("/missing/local/replay.Last.txt");
    cfg.history_bars = 1;
    cfg.replay_headless = true;
    cfg.replay_engine_mode = ReplayEngineMode::Deterministic;
    cfg.replay_fill_model = ReplayFillModel::RawBarOpen;
    cfg.replay_signal_diagnostics = true;
    cfg.replay_result_dir = cache_root.join("results");
    let replay = replay::load_replay_state(&cfg, bar_type, CandleMode::Standard, None, None)
        .await
        .expect("load actual cached replay fixture");
    let bars = replay.bars_for_type(bar_type).expect("derived volume bars");
    assert_eq!(bars.len(), 4);
    assert_eq!(bars[1].ts_ns, bars[2].ts_ns);
    assert!(bars[1].close < bars[1].open);
    assert!(bars[2].close > bars[2].open);

    let accounts = replay_accounts(&replay);
    let contract = replay_contract(&replay);
    let mut session = test_session();
    session.cfg = cfg.clone();
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_config.kind = StrategyKind::Lua;
    session.execution_config.native_execution_path = NativeExecutionPath::Guarded;
    session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;
    session.execution_config.lua.source = r#"
        function on_bar(ctx, bar)
            if bar.close > bar.open then return 'buy' end
            if bar.close < bar.open then return 'sell' end
            return 'hold'
        end
    "#
    .into();
    session.selected_account_id = accounts.first().map(|account| account.id);
    session.accounts = accounts;
    session.selected_contract = Some(contract.clone());
    session.bar_type = bar_type;
    session.candle_mode = CandleMode::Standard;
    let mut state = test_state(session);
    state.replay = Some(replay.clone());
    state.replay_execution_ledger = replay::ReplayExecutionLedgerState::new_with_fill_config(
        cfg.replay_engine_mode,
        cfg.replay_fill_model,
        &cfg.replay_latency_config(),
        cfg.replay_bar_protection_policy,
        bar_type,
        CandleMode::Standard,
    );

    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    state.broker_tx = broker_tx.clone();
    let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    let broker_task = spawn_broker_gateway_task(broker_rx, internal_tx.clone());
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (speed_tx, speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    drop(speed_tx);

    handle_command(
        ServiceCommand::ArmExecutionStrategy,
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx.clone(),
    )
    .await
    .expect("arm cached replay strategy");
    let worker = replay::spawn_replay_market_task(
        replay,
        cfg,
        contract,
        bar_type,
        CandleMode::Standard,
        broker_tx,
        speed_rx,
        internal_tx.clone(),
    );

    timeout(Duration::from_secs(10), async {
        loop {
            let event = internal_rx.recv().await.expect("worker/broker event");
            let completed = matches!(event, InternalEvent::ReplayCompleted { .. });
            handle_internal(
                event,
                &mut state,
                &event_tx,
                &market_tx,
                internal_tx.clone(),
            )
            .await
            .expect("handle worker or broker event");
            if completed {
                break;
            }
        }
    })
    .await
    .expect("cached replay worker completes within timeout");
    worker.await.expect("replay worker joins");
    broker_task.abort();

    let session = state.session.as_ref().expect("replay session remains");
    let duplicate_rows = session
        .execution_runtime
        .replay_signal_diagnostics
        .iter()
        .filter(|row| row.bar_timestamp_ns == bars[1].ts_ns)
        .collect::<Vec<_>>();
    assert_eq!(duplicate_rows.len(), 2, "both same-time bars are evaluated");
    assert_eq!(duplicate_rows[0].signal, "Sell");
    assert_eq!(duplicate_rows[1].signal, "Buy");
    assert_eq!(duplicate_rows[0].bar_close, bars[1].close);
    assert_eq!(duplicate_rows[1].bar_close, bars[2].close);

    let fills = &state.replay_execution_ledger.snapshot().fills;
    assert_eq!(fills.len(), 2, "both targets reach the simulated broker");
    assert_eq!(fills[0].side, "Sell");
    assert_eq!(fills[1].side, "Buy");
    assert_eq!(fills[0].signal_timestamp_ns, Some(bars[1].ts_ns));
    assert_eq!(fills[1].signal_timestamp_ns, Some(bars[2].ts_ns));
    assert_eq!(fills[0].fill_timestamp_ns, bars[1].ts_ns);
    assert_eq!(fills[1].fill_timestamp_ns, bars[3].ts_ns);
    assert!(fills[0].sequence < fills[1].sequence);

    std::fs::remove_dir_all(cache_root).expect("remove test replay cache");
}
