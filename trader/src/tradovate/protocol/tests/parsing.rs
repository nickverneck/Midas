use super::*;
use serde_json::json;

#[test]
fn parse_bar_accepts_minute_precision_utc_timestamp() {
    let bar = parse_bar(&json!({
        "timestamp": "2026-03-11T22:38Z",
        "open": 6738.5,
        "high": 6739.5,
        "low": 6736.75,
        "close": 6738.0
    }))
    .expect("bar should parse");

    let expected_ts = chrono::DateTime::parse_from_rfc3339("2026-03-11T22:38:00Z")
        .unwrap()
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
        .unwrap();

    assert_eq!(bar.ts_ns, expected_ts);
    assert_eq!(bar.close, 6738.0);
}

#[test]
fn create_message_formats_body_only_requests_for_websocket() {
    let body = json!({
        "accountId": 42,
        "orderQty": 1
    });

    let actual = create_message("order/placeorder", 7, None, Some(&body));

    assert_eq!(
        actual,
        "order/placeorder\n7\n\n{\"accountId\":42,\"orderQty\":1}"
    );
}

#[test]
fn parse_socket_response_maps_status_and_payload() {
    let ok = json!({
        "i": 3,
        "s": 200,
        "d": { "orderId": 99 }
    });
    let err = json!({
        "i": 4,
        "s": 400,
        "d": "bad request"
    });

    assert_eq!(
        parse_socket_response(&ok).expect("success payload"),
        json!({ "orderId": 99 })
    );
    assert_eq!(
        parse_socket_response(&err).expect_err("error payload"),
        "websocket request failed (400): bad request"
    );
}
