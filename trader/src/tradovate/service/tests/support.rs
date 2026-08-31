use super::super::*;
use serde_json::json;
use std::collections::BTreeMap;

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
        token_file_snapshot: None,
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
            name: "ESM6".to_string(),
            description: "E-mini S&P".to_string(),
            raw: json!({}),
        }),
        bar_type: BarType::default(),
        candle_mode: CandleMode::Standard,
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
        engine_run: None,
    }
}

pub(super) fn test_state(session: SessionState) -> ServiceState {
    let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) = tokio::sync::watch::channel(ReplaySpeed::default());
    ServiceState {
        client: Client::builder().build().expect("client"),
        broker_tx,
        broker_task: None,
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
        session: Some(session),
        replay: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        snapshot_task: None,
        replay_lookup_job: None,
        replay_download_job: None,
        latency: LatencySnapshot::default(),
        snapshot_generation: 0,
        snapshot_revision: 0,
        snapshot_refresh_pending: false,
    }
}
