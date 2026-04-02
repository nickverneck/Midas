use super::*;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn contract_position_qty_matches_selected_contract() {
    let mut store = UserSyncStore::default();
    store.positions.insert(
        42,
        BTreeMap::from([
            (
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 2
                }),
            ),
            (
                2,
                json!({
                    "id": 2,
                    "accountId": 42,
                    "symbol": "ESM6",
                    "netPos": -1
                }),
            ),
        ]),
    );

    let contract = ContractSuggestion {
        id: 3570918,
        name: "ESH6".to_string(),
        description: String::new(),
        raw: json!({ "contractMaturityId": 53951 }),
    };

    assert_eq!(store.contract_position_qty(42, &contract), Some(2.0));
}

#[test]
fn contract_position_qty_uses_position_side_when_netpos_missing() {
    let mut store = UserSyncStore::default();
    store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "qty": 1,
                "side": "Short"
            }),
        )]),
    );

    let contract = ContractSuggestion {
        id: 3570918,
        name: "ESM6".to_string(),
        description: String::new(),
        raw: json!({ "contractMaturityId": 53951 }),
    };

    assert_eq!(store.contract_position_qty(42, &contract), Some(-1.0));
}

#[test]
fn contract_position_qty_uses_best_match_when_duplicate_position_records_overlap() {
    let mut store = UserSyncStore::default();
    store.positions.insert(
        42,
        BTreeMap::from([
            (
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 6000.0
                }),
            ),
            (
                2,
                json!({
                    "id": 2,
                    "accountId": 42,
                    "symbol": "ESM6",
                    "netPos": 1,
                    "avgPrice": 5999.0
                }),
            ),
        ]),
    );

    let contract = ContractSuggestion {
        id: 3570918,
        name: "ESM6".to_string(),
        description: String::new(),
        raw: json!({ "contractMaturityId": 53951 }),
    };

    assert_eq!(store.contract_position_qty(42, &contract), Some(1.0));
}

#[test]
fn fallback_unrealized_pnl_uses_latest_close_and_value_per_point() {
    let market = MarketSnapshot {
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        bars: vec![Bar {
            ts_ns: 0,
            open: 6725.0,
            high: 6730.0,
            low: 6724.0,
            close: 6727.25,
        }],
        trade_markers: Vec::new(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 1,
        live_bars: 0,
        status: String::new(),
    };

    let positions = vec![json!({
        "accountId": 42,
        "contractId": 3570918,
        "netPos": 1,
        "netPrice": 6725.75
    })];

    assert_eq!(fallback_unrealized_pnl(&positions, &market), Some(75.0));
}

#[test]
fn fallback_unrealized_pnl_ignores_duplicate_symbol_only_position_records() {
    let market = MarketSnapshot {
        contract_id: Some(3570918),
        contract_name: Some("ESM6".to_string()),
        bars: vec![Bar {
            ts_ns: 0,
            open: 6725.0,
            high: 6730.0,
            low: 6724.0,
            close: 6727.25,
        }],
        value_per_point: Some(50.0),
        ..MarketSnapshot::default()
    };
    let positions = vec![
        json!({
            "id": 1,
            "accountId": 42,
            "contractId": 3570918,
            "netPos": 1,
            "netPrice": 6725.75
        }),
        json!({
            "id": 2,
            "accountId": 42,
            "symbol": "ESM6",
            "netPos": 1,
            "netPrice": 6724.75
        }),
    ];

    assert_eq!(fallback_unrealized_pnl(&positions, &market), Some(75.0));
}

#[test]
fn build_market_update_emits_snapshot_then_forming_delta() {
    let contract = ContractSuggestion {
        id: 3570918,
        name: "ESH6".to_string(),
        description: "ES Jun 2026".to_string(),
        raw: json!({ "id": 3570918 }),
    };
    let closed_bar = Bar {
        ts_ns: 1,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    };
    let forming_bar = Bar {
        ts_ns: 2,
        open: 5000.5,
        high: 5002.0,
        low: 5000.25,
        close: 5001.5,
    };
    let mut series = LiveSeries::new();
    series.closed_bars.push(closed_bar.clone());

    let initial = build_market_update(
        &contract,
        None,
        series.closed_bars.len(),
        0,
        "status".to_string(),
        0,
        None,
        None,
        &series,
    )
    .expect("initial snapshot should be emitted");
    assert!(matches!(
        initial.bars,
        MarketBarsUpdate::Snapshot {
            ref closed_bars,
            forming_bar: None
        } if closed_bars == &vec![closed_bar.clone()]
    ));

    let before_last_closed = series.closed_bars.last().cloned();
    series.forming_bar = Some(forming_bar.clone());
    let update = build_market_update(
        &contract,
        None,
        series.closed_bars.len(),
        0,
        "status".to_string(),
        series.closed_bars.len(),
        before_last_closed,
        None,
        &series,
    )
    .expect("forming update should be emitted");
    assert!(matches!(
        update.bars,
        MarketBarsUpdate::Forming { forming_bar: ref bar } if bar == &forming_bar
    ));
}

#[test]
fn apply_market_update_keeps_bars_incremental() {
    let closed_bar = Bar {
        ts_ns: 1,
        open: 5000.0,
        high: 5001.0,
        low: 4999.0,
        close: 5000.5,
    };
    let forming_bar = Bar {
        ts_ns: 2,
        open: 5000.5,
        high: 5002.0,
        low: 5000.25,
        close: 5001.5,
    };
    let next_forming_bar = Bar {
        ts_ns: 3,
        open: 5001.5,
        high: 5003.0,
        low: 5001.0,
        close: 5002.5,
    };
    let marker = TradeMarker {
        fill_id: Some(7),
        account_id: Some(42),
        contract_id: Some(3570918),
        contract_name: Some("ESH6".to_string()),
        ts_ns: 1,
        price: 5000.5,
        qty: 1,
        side: TradeMarkerSide::Buy,
    };
    let mut market = MarketSnapshot {
        trade_markers: vec![marker],
        ..MarketSnapshot::default()
    };

    let initial = MarketUpdate {
        contract_id: 3570918,
        contract_name: "ESH6".to_string(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 1,
        live_bars: 0,
        status: "initial".to_string(),
        bars: MarketBarsUpdate::Snapshot {
            closed_bars: vec![closed_bar.clone()],
            forming_bar: Some(forming_bar.clone()),
        },
    };
    assert!(apply_market_update(&mut market, initial));
    assert_eq!(market.history_loaded, 1);
    assert_eq!(market.bars, vec![closed_bar.clone(), forming_bar.clone()]);
    assert_eq!(market.trade_markers.len(), 1);

    let next = MarketUpdate {
        contract_id: 3570918,
        contract_name: "ESH6".to_string(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 2,
        live_bars: 1,
        status: "realtime".to_string(),
        bars: MarketBarsUpdate::Closed {
            closed_bar: forming_bar.clone(),
            forming_bar: Some(next_forming_bar.clone()),
        },
    };
    assert!(apply_market_update(&mut market, next));
    assert_eq!(market.history_loaded, 2);
    assert_eq!(market.bars, vec![closed_bar, forming_bar, next_forming_bar]);
    assert_eq!(market.trade_markers.len(), 1);
}

#[test]
fn apply_market_update_drops_oldest_closed_bar_when_window_is_full() {
    let bar = |ts_ns| Bar {
        ts_ns,
        open: 5000.0 + ts_ns as f64,
        high: 5001.0 + ts_ns as f64,
        low: 4999.0 + ts_ns as f64,
        close: 5000.5 + ts_ns as f64,
    };

    let mut market = MarketSnapshot {
        bars: vec![bar(1), bar(2)],
        history_loaded: 2,
        ..MarketSnapshot::default()
    };
    let update = MarketUpdate {
        contract_id: 3570918,
        contract_name: "ESH6".to_string(),
        session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
        value_per_point: Some(50.0),
        tick_size: Some(0.25),
        history_loaded: 2,
        live_bars: 1,
        status: "realtime".to_string(),
        bars: MarketBarsUpdate::Closed {
            closed_bar: bar(3),
            forming_bar: None,
        },
    };

    assert!(apply_market_update(&mut market, update));
    assert_eq!(market.history_loaded, 2);
    assert_eq!(market.bars, vec![bar(2), bar(3)]);
}

#[test]
fn display_market_snapshot_trims_to_recent_closed_bars_and_keeps_forming_bar() {
    let bar = |ts_ns| Bar {
        ts_ns,
        open: 5000.0 + ts_ns as f64,
        high: 5001.0 + ts_ns as f64,
        low: 4999.0 + ts_ns as f64,
        close: 5000.5 + ts_ns as f64,
    };

    let mut bars = (1..=300).map(bar).collect::<Vec<_>>();
    bars.push(bar(301));
    let market = MarketSnapshot {
        bars,
        history_loaded: 300,
        status: "streaming".to_string(),
        ..MarketSnapshot::default()
    };

    let snapshot = display_market_snapshot(&market);
    assert_eq!(snapshot.history_loaded, UI_MARKET_BAR_LIMIT);
    assert_eq!(snapshot.bars.len(), UI_MARKET_BAR_LIMIT + 1);
    assert_eq!(snapshot.bars.first().map(|bar| bar.ts_ns), Some(45));
    assert_eq!(snapshot.bars.last().map(|bar| bar.ts_ns), Some(301));
    assert_eq!(snapshot.status, "streaming");
}

#[test]
fn futures_globex_preclose_window_holds_entries() {
    let dt = New_York
        .with_ymd_and_hms(2026, 3, 9, 16, 50, 0)
        .single()
        .unwrap()
        .with_timezone(&Utc);
    let window =
        InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

    assert!(window.session_open);
    assert!(window.hold_entries);
    assert!(window.minutes_to_close.unwrap() <= 10.0);
}

#[test]
fn futures_globex_daily_break_holds_until_reopen() {
    let dt = New_York
        .with_ymd_and_hms(2026, 3, 9, 17, 30, 0)
        .single()
        .unwrap()
        .with_timezone(&Utc);
    let window =
        InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

    assert!(!window.session_open);
    assert!(window.hold_entries);
    assert_eq!(window.minutes_to_close, None);
}

#[test]
fn futures_globex_reopens_after_break() {
    let dt = New_York
        .with_ymd_and_hms(2026, 3, 9, 18, 5, 0)
        .single()
        .unwrap()
        .with_timezone(&Utc);
    let window =
        InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

    assert!(window.session_open);
    assert!(!window.hold_entries);
    assert!(window.minutes_to_close.unwrap() > 1_300.0);
}

#[test]
fn equity_rth_preclose_window_holds_entries() {
    let dt = New_York
        .with_ymd_and_hms(2026, 3, 9, 15, 50, 0)
        .single()
        .unwrap()
        .with_timezone(&Utc);
    let window = InstrumentSessionProfile::EquityRth.evaluate(dt.timestamp_nanos_opt().unwrap());

    assert!(window.session_open);
    assert!(window.hold_entries);
    assert!(window.minutes_to_close.unwrap() <= 10.0);
}
