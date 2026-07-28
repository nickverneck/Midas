use super::super::protocol::historical_raw_tick_timestamp_bounds;
use super::super::*;
use super::support::dt;
use serde_json::json;

#[test]
fn extracts_only_historical_bars_inside_requested_range() {
    let item = json!({
        "d": {
            "charts": [
                {
                    "id": 7,
                    "bars": [
                        {"timestamp":"2026-07-22T23:59:00Z","open":1,"high":1,"low":1,"close":1,"volume":1},
                        {"timestamp":"2026-07-23T00:00:00Z","open":2,"high":3,"low":1,"close":2.5,"upVolume":5,"downVolume":4}
                    ]
                },
                {
                    "id": 8,
                    "bars": [
                        {"timestamp":"2026-07-23T00:00:00Z","open":9,"high":9,"low":9,"close":9,"volume":1}
                    ]
                },
                {"id": 7, "eoh": true}
            ]
        }
    });

    let (bars, eoh) = extract_historical_server_bars_from_chart_message(
        &item,
        Some(7),
        dt("2026-07-23T00:00:00Z"),
        dt("2026-07-24T00:00:00Z"),
    );

    assert!(eoh);
    assert_eq!(bars.len(), 1);
    assert_eq!(bars[0].open, 2.0);
    assert_eq!(bars[0].volume, Some(9.0));
}

#[test]
fn extracts_raw_ticks_from_historical_tick_packets() {
    let item = json!({
        "d": {
            "charts": [
                {
                    "id": 7,
                    "s": "db",
                    "td": 20260723,
                    "bp": 29700,
                    "bt": 1784764800000i64,
                    "ts": 0.25,
                    "tks": [
                        {"t": 0, "p": 0, "s": 1, "b": -1, "a": 0, "bs": 12, "as": 14, "id": 1001},
                        {"t": 1, "p": 1, "s": 2, "id": 1002},
                        {"t": -1, "p": 9, "s": 1, "id": 999}
                    ]
                },
                {
                    "id": 8,
                    "bp": 999,
                    "bt": 1784764800000i64,
                    "ts": 0.25,
                    "tks": [
                        {"t": 0, "p": 0, "s": 1, "id": 2001}
                    ]
                },
                {"id": 7, "eoh": true}
            ]
        }
    });

    let (ticks, eoh) = extract_historical_raw_ticks_from_chart_message(
        &item,
        Some(7),
        dt("2026-07-23T00:00:00Z"),
        dt("2026-07-23T00:00:01Z"),
        None,
    );

    assert!(eoh);
    assert_eq!(ticks.len(), 2);
    assert_eq!(ticks[0].tick_id, Some(1001));
    assert_eq!(ticks[0].price, 7425.0);
    assert_eq!(ticks[0].bid_price, Some(7424.75));
    assert_eq!(ticks[0].ask_price, Some(7425.0));
    assert_eq!(ticks[0].size, 1.0);
    assert_eq!(ticks[0].chart_id, Some(7));
    assert_eq!(ticks[0].trade_date, Some(20260723));
    assert_eq!(ticks[0].packet_source.as_deref(), Some("db"));
    assert_eq!(ticks[1].tick_id, Some(1002));
    assert_eq!(ticks[1].price, 7425.25);
    let bounds = historical_raw_tick_timestamp_bounds(&item, Some(7)).expect("provider bounds");
    assert_eq!(bounds.0, dt("2026-07-22T23:59:59.999Z"));
    assert_eq!(bounds.1, dt("2026-07-23T00:00:00.001Z"));
}
