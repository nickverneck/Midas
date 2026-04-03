use super::*;

pub(super) use serde_json::{Value, json};
pub(super) use std::collections::BTreeMap;

mod account_sync;
mod pending_targets;
mod protection_sync;
mod replay;
mod snapshot;
mod strategy_loop;
mod strategy_tracking;

pub(super) fn test_session() -> SessionState {
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    SessionState {
        cfg: AppConfig::default(),
        session_kind: SessionKind::Live,
        replay_enabled: false,
        tokens: TokenBundle {
            access_token: "access".to_string(),
            md_access_token: "md".to_string(),
            expiration_time: None,
            user_id: None,
            user_name: None,
        },
        accounts: vec![AccountInfo {
            id: 42,
            name: "SIM".to_string(),
            raw: json!({}),
        }],
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store: UserSyncStore::default(),
        selected_account_id: Some(42),
        selected_contract: Some(ContractSuggestion {
            id: 3570918,
            name: "ESH6".to_string(),
            description: "E-mini S&P".to_string(),
            raw: json!({}),
        }),
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    }
}
