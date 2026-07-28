use super::super::*;
use super::support::{test_session, test_state};

#[tokio::test]
async fn config_sync_does_not_disarm_armed_native_strategy() {
    let mut session = test_session();
    session.execution_config.kind = StrategyKind::Native;
    session.execution_runtime.armed = true;
    session.execution_runtime.pending_target_qty = Some(1);
    session.execution_runtime.last_closed_bar_ts = Some(123);
    session.execution_runtime.last_summary = "Armed before edit".to_string();
    let mut state = test_state(session);
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut config = ExecutionStrategyConfig::default();
    config.kind = StrategyKind::Native;
    config.native_ema.fast_length = 8;

    handle_command(
        ServiceCommand::SetExecutionStrategyConfig(config),
        &mut state,
        &event_tx,
        &market_tx,
        internal_tx,
    )
    .await
    .expect("config sync should succeed");

    let session = state.session.expect("session should persist");
    assert!(session.execution_runtime.armed);
    assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    assert_eq!(session.execution_runtime.last_closed_bar_ts, Some(123));
    assert_eq!(session.execution_runtime.last_summary, "Armed before edit");
    assert_eq!(session.execution_config.native_ema.fast_length, 8);
}
