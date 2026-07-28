use super::super::protocol::RAW_TICK_REQUEST_MAX_ELEMENTS;
use super::super::*;
use super::support::dt;
use serde_json::json;

#[test]
fn server_bar_request_uses_chart_timerange_without_account_streams() {
    let contract = ContractSuggestion {
        id: 123,
        name: "MESU6".to_string(),
        description: "Micro E-mini".to_string(),
        raw: json!({ "id": 123, "name": "MESU6" }),
    };
    let request = TradovateServerBarDownloadRequest {
        contract: "MESU6".to_string(),
        exact_contract: None,
        start: dt("2026-07-23T00:00:00Z"),
        end: dt("2026-07-24T00:00:00Z"),
        bar_type: BarType::minute(1),
    };

    let body = server_bar_chart_request_body(&contract, &request);

    assert_eq!(body["symbol"], 123);
    assert_eq!(
        body["chartDescription"],
        BarType::minute(1).chart_description()
    );
    assert_eq!(
        body["timeRange"]["asFarAsTimestamp"],
        "2026-07-23T00:00:00+00:00"
    );
    assert_eq!(
        body["timeRange"]["closestTimestamp"],
        "2026-07-24T00:00:00+00:00"
    );
    assert!(body.get("replayCache").is_none());
    assert!(body.get("accounts").is_none());
    assert!(body.get("entityTypes").is_none());
}

#[test]
fn raw_tick_request_uses_one_tick_chart_without_account_streams() {
    let contract = ContractSuggestion {
        id: 123,
        name: "MESU6".to_string(),
        description: "Micro E-mini".to_string(),
        raw: json!({ "id": 123, "name": "MESU6" }),
    };
    let request = TradovateRawTickDownloadRequest {
        contract: "MESU6".to_string(),
        exact_contract: None,
        start: dt("2026-07-23T00:00:00Z"),
        end: dt("2026-07-24T00:00:00Z"),
    };

    let body = raw_tick_chart_request_body(&contract, &request);

    assert_eq!(body["symbol"], "MESU6");
    assert_eq!(body["chartDescription"]["underlyingType"], "Tick");
    assert_eq!(body["chartDescription"]["elementSize"], 1);
    assert_eq!(
        body["chartDescription"]["elementSizeUnit"],
        "UnderlyingUnits"
    );
    assert_eq!(
        body["timeRange"]["asFarAsTimestamp"],
        "2026-07-23T00:00:00+00:00"
    );
    assert_eq!(
        body["timeRange"]["closestTimestamp"],
        "2026-07-24T00:00:00+00:00"
    );
    assert_eq!(
        body["timeRange"]["asMuchAsElements"],
        RAW_TICK_REQUEST_MAX_ELEMENTS
    );
    assert!(body.get("accounts").is_none());
    assert!(body.get("entityTypes").is_none());
}
