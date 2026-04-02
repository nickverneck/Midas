use super::account::{
    selected_market_entry_price, selected_market_position_qty, selected_trade_markers,
};
use super::execution::target_qty_for_signal;
use super::state::{AccountState, ExecutionRuntimeState, IronbeamSession};
use super::support::{ironbeam_duration_code, order_is_active, stable_id};
use crate::broker::{AccountInfo, ContractSuggestion, MarketSnapshot, TradeMarkerSide};
use crate::config::AppConfig;
use crate::strategies::StrategySignal;
use crate::strategy::ExecutionStrategyConfig;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn duration_codes_match_supported_inputs() {
    assert_eq!(ironbeam_duration_code("Day"), "0");
    assert_eq!(ironbeam_duration_code("GTC"), "1");
    assert_eq!(ironbeam_duration_code("good till cancel"), "1");
}

#[test]
fn target_qty_for_signal_maps_expected_transitions() {
    assert_eq!(target_qty_for_signal(StrategySignal::Hold, 0, 1), None);
    assert_eq!(
        target_qty_for_signal(StrategySignal::EnterLong, 0, 2),
        Some(2)
    );
    assert_eq!(
        target_qty_for_signal(StrategySignal::EnterShort, 0, 3),
        Some(-3)
    );
    assert_eq!(
        target_qty_for_signal(StrategySignal::ExitLongOnShortSignal, 2, 1),
        Some(0)
    );
}

#[test]
fn active_order_statuses_exclude_terminal_states() {
    assert!(order_is_active(
        &json!({"status": "WORKING", "quantity": 1})
    ));
    assert!(!order_is_active(
        &json!({"status": "FILLED", "quantity": 1, "fillQuantity": 1})
    ));
    assert!(!order_is_active(
        &json!({"status": "CANCELED", "quantity": 1})
    ));
}

#[test]
fn selected_trade_markers_filter_to_selected_symbol() {
    let mut account_state = AccountState::default();
    account_state.fills.insert(
        "SIM".to_string(),
        BTreeMap::from([
            (
                "one".to_string(),
                json!({
                    "orderUpdateId": "one",
                    "accountId": "SIM",
                    "exchSym": "XCME:ESM6",
                    "side": "BUY",
                    "fillPrice": 5000.0,
                    "fillQuantity": 1,
                    "fillDate": "2026-03-30T14:30:00Z",
                }),
            ),
            (
                "two".to_string(),
                json!({
                    "orderUpdateId": "two",
                    "accountId": "SIM",
                    "exchSym": "XCME:NQM6",
                    "side": "SELL",
                    "fillPrice": 18000.0,
                    "fillQuantity": 1,
                    "fillDate": "2026-03-30T14:31:00Z",
                }),
            ),
        ]),
    );

    let markers = selected_trade_markers(
        &account_state,
        &ContractSuggestion {
            id: stable_id("XCME:ESM6"),
            name: "XCME:ESM6".to_string(),
            description: "ES".to_string(),
            raw: json!({"symbol": "XCME:ESM6"}),
        },
    );

    assert_eq!(markers.len(), 1);
    assert_eq!(markers[0].price, 5000.0);
    assert_eq!(markers[0].side, TradeMarkerSide::Buy);
}

#[test]
fn execution_runtime_snapshot_hides_internal_timers() {
    let mut runtime = ExecutionRuntimeState::default();
    runtime.armed = true;
    runtime.set_pending_target(Some(2));
    runtime.last_summary = "waiting".to_string();

    let snapshot = runtime.snapshot();

    assert!(snapshot.armed);
    assert_eq!(snapshot.pending_target_qty, Some(2));
    assert_eq!(snapshot.last_summary, "waiting");
}

#[test]
fn selected_position_qty_uses_selected_account_and_contract() {
    let contract = ContractSuggestion {
        id: stable_id("XCME:ESM6"),
        name: "XCME:ESM6".to_string(),
        description: "ES".to_string(),
        raw: json!({"symbol": "XCME:ESM6"}),
    };
    let session = IronbeamSession {
        cfg: AppConfig::default(),
        token: "token".to_string(),
        user_name: None,
        accounts: vec![AccountInfo {
            id: stable_id("SIM"),
            name: "SIM".to_string(),
            raw: json!({}),
        }],
        selected_account_id: Some(stable_id("SIM")),
        selected_contract: Some(contract.clone()),
        account_state: AccountState {
            positions: BTreeMap::from([(
                "SIM".to_string(),
                BTreeMap::from([(
                    "XCME:ESM6:LONG".to_string(),
                    json!({
                        "accountId": "SIM",
                        "exchSym": "XCME:ESM6",
                        "quantity": 2,
                        "side": "LONG",
                        "price": 5000.0,
                    }),
                )]),
            )]),
            ..AccountState::default()
        },
        account_snapshots: Vec::new(),
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        managed_protection: BTreeMap::new(),
        market: MarketSnapshot::default(),
        market_task: None,
    };

    assert_eq!(selected_market_position_qty(&session), 2);
    assert_eq!(selected_market_entry_price(&session), Some(5000.0));
}
