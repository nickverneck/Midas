use super::super::metadata::*;
use super::super::*;
use super::support::dt;
use serde_json::json;

#[tokio::test]
async fn exact_download_contract_is_used_without_symbol_search() {
    let client = Client::new();
    let exact = ContractSuggestion {
        id: 4_399_631,
        name: "MESU6".to_string(),
        description: "Micro E-mini".to_string(),
        raw: json!({"id": 4_399_631, "name": "MESU6"}),
    };

    let resolved = resolve_requested_download_contract(
        &client,
        &TradingEnvironment::Sim,
        "unused-token",
        "MESU6",
        Some(&exact),
        12,
    )
    .await
    .expect("exact contract should not require a search request");

    assert_eq!(resolved.id, exact.id);
    assert_eq!(resolved.name, exact.name);
}

#[tokio::test]
async fn exact_download_contract_rejects_name_mismatch_before_network_io() {
    let client = Client::new();
    let exact = ContractSuggestion {
        id: 4_399_631,
        name: "MESU6".to_string(),
        description: "Micro E-mini".to_string(),
        raw: json!({"id": 4_399_631, "name": "MESU6"}),
    };

    let err = resolve_requested_download_contract(
        &client,
        &TradingEnvironment::Sim,
        "unused-token",
        "ESU6",
        Some(&exact),
        12,
    )
    .await
    .expect_err("mismatched exact contract must fail closed");

    assert!(err.to_string().contains("does not match requested symbol"));
}

#[test]
fn provider_contract_snapshot_must_match_selected_exact_identity() {
    let contract = ContractSuggestion {
        id: 4_399_631,
        name: "MESU6".to_string(),
        description: "Micro E-mini".to_string(),
        raw: json!({}),
    };

    let id_err = validate_contract_identity(&contract, &json!({"id": 7, "name": "MESU6"}))
        .expect_err("provider id mismatch");
    assert!(
        id_err
            .to_string()
            .contains("does not match selected contract id")
    );

    let name_err = validate_contract_identity(&contract, &json!({"id": 4_399_631, "name": "ESU6"}))
        .expect_err("provider name mismatch");
    assert!(
        name_err
            .to_string()
            .contains("does not match selected contract name")
    );
}
#[test]
fn derives_broad_coverage_from_adjacent_maturity_expirations() {
    let selected = json!({
        "id": 62531,
        "expirationDate": "2026-09-18T13:30Z"
    });
    let chain = json!([
        {"id": 61200, "expirationDate": "2026-03-20T13:30Z"},
        {"id": 61800, "expirationDate": "2026-06-19T13:30Z"},
        {"id": 62531, "expirationDate": "2026-09-18T13:30Z"},
        {"id": 63200, "expirationDate": "2026-12-18T14:30Z"}
    ]);

    let coverage = suggested_contract_coverage(Some(&selected), Some(&chain))
        .expect("adjacent expiration coverage");

    assert_eq!(coverage.start_date.to_string(), "2026-06-19");
    assert_eq!(coverage.end_date.to_string(), "2026-09-18");
    assert!(coverage.estimated);
    assert!(coverage.basis.contains("actual liquidity"));
}

#[test]
fn broad_coverage_fails_closed_without_an_earlier_maturity() {
    let selected = json!({
        "id": 62531,
        "expirationDate": "2026-09-18T13:30:00Z"
    });
    let chain = json!([
        {"id": 62531, "expirationDate": "2026-09-18T13:30:00Z"},
        {"id": 63200, "expirationDate": "2026-12-18T14:30:00Z"}
    ]);

    assert!(suggested_contract_coverage(Some(&selected), Some(&chain)).is_none());
}

#[test]
fn metadata_source_timestamp_finds_provider_timestamp() {
    let payload = json!([{
        "id": 1,
        "initialMargin": 2761.96,
        "timestamp": "2026-07-24T03:50:43.545Z"
    }]);

    assert_eq!(
        metadata_source_timestamp(&payload),
        Some(dt("2026-07-24T03:50:43.545Z"))
    );
}

#[test]
fn market_specs_prefer_product_fields_and_keep_contract_tick_fallback() {
    let metadata = ReplayCacheContractMetadata {
        context: ReplayCacheMetadataContext {
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_id: Some(9),
            user_name: Some("tester".to_string()),
            accounts: Vec::new(),
            accounts_endpoint: Some("account/list".to_string()),
            accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
        },
        contract: ReplayCacheMetadataSnapshot {
            endpoint: "contract/item".to_string(),
            fetched_at: dt("2026-07-24T00:00:00Z"),
            source_timestamp: None,
            payload: json!({"providerTickSize": 0.5}),
        },
        maturity: None,
        maturity_chain: None,
        product: Some(ReplayCacheMetadataSnapshot {
            endpoint: "product/item".to_string(),
            fetched_at: dt("2026-07-24T00:00:01Z"),
            source_timestamp: None,
            payload: json!({
                "tickSize": 0.25,
                "valuePerPoint": 5.0,
                "productType": "Futures"
            }),
        }),
        product_sessions: None,
        product_margins: None,
        contract_margins: None,
        fee_params: None,
        suggested_coverage: None,
    };

    let specs = market_specs_from_metadata(&metadata).expect("metadata market specs");
    assert_eq!(specs.tick_size, Some(0.25));
    assert_eq!(specs.value_per_point, Some(5.0));
    assert!(specs.session_profile.is_some());
}

#[test]
fn optional_metadata_warning_names_the_failed_endpoint() {
    let warning =
        optional_metadata_warning("productMargin/deps", &anyhow::anyhow!("synthetic timeout"));

    assert!(warning.contains("productMargin/deps"));
    assert!(warning.contains("synthetic timeout"));
}

#[test]
fn authenticated_metadata_http_warnings_omit_provider_response_bodies() {
    let provider_body = r#"{"accessToken":"secret-token","accountName":"DEMO4769136"}"#;
    let error = sanitized_metadata_http_error(
        "account/list",
        reqwest::StatusCode::UNAUTHORIZED,
        provider_body,
    );
    let warning = optional_metadata_warning("account/list", &error);

    assert!(warning.contains("account/list"));
    assert!(warning.contains("HTTP 401"));
    assert!(warning.contains("authentication rejected"));
    assert!(warning.contains("provider response body omitted"));
    assert!(!warning.contains("accessToken"));
    assert!(!warning.contains("secret-token"));
    assert!(!warning.contains("DEMO4769136"));
}

#[test]
fn fee_metadata_request_uses_required_product_ids_array() {
    assert_eq!(
        fee_metadata_request_body(1_878_809),
        json!({"productIds": [1_878_809]})
    );
}
