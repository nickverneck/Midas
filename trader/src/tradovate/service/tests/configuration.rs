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
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
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

#[tokio::test]
async fn arm_command_runs_lua_only_in_cached_replay_and_reaches_replay_fill() {
    use tokio::time::{Duration, timeout};

    let mut session = test_session();
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_config.kind = StrategyKind::Lua;
    session.execution_config.lua.source =
        "function on_bar(ctx, bar) if bar.close > bar.open then return 'buy' end return 'hold' end"
            .into();
    session.execution_config.native_execution_path = NativeExecutionPath::Guarded;
    session.market.history_loaded = 1;
    session.market.bars = vec![crate::broker::Bar {
        ts_ns: 1,
        open: 100.0,
        high: 101.0,
        low: 99.0,
        close: 100.0,
        volume: Some(10.0),
    }];

    let mut state = test_state(session);
    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    state.broker_tx = broker_tx.clone();
    let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    let gateway = spawn_broker_gateway_task(broker_rx, internal_tx);
    let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
    let (service_internal_tx, _service_internal_rx) =
        internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

    handle_command(
        ServiceCommand::ArmExecutionStrategy,
        &mut state,
        &event_tx,
        &market_tx,
        service_internal_tx.clone(),
    )
    .await
    .expect("cached replay Lua arm command should succeed");
    assert!(state.session.as_ref().unwrap().execution_runtime.armed);

    {
        let session = state.session.as_mut().unwrap();
        session.market.bars.push(crate::broker::Bar {
            ts_ns: 2,
            open: 100.0,
            high: 102.0,
            low: 100.0,
            close: 101.5,
            volume: Some(12.0),
        });
        session.market.history_loaded = 2;
        maybe_run_execution_strategy(session, &broker_tx, &event_tx)
            .expect("armed Lua replay should evaluate and dispatch");
        assert_eq!(session.execution_runtime.pending_target_qty, Some(1));
    }
    let fill = timeout(Duration::from_secs(2), async {
        loop {
            if let InternalEvent::UserEntities(entities) =
                internal_rx.recv().await.expect("gateway event")
                && let Some(fill) = entities.iter().find(|entity| entity.entity_type == "fill")
            {
                break fill.clone();
            }
        }
    })
    .await
    .expect("arm command must lead to a replay fill");
    assert_eq!(fill.entity["source"], "replay");
    assert_eq!(fill.entity["buySell"], "Buy");
    assert_eq!(fill.entity["price"], 101.5);
    gateway.abort();

    for (session_kind, replay_enabled, hosted_replay_orders) in [
        (SessionKind::Live, false, false),
        (SessionKind::NinjaTraderReplay, true, true),
    ] {
        let mut rejected = test_session();
        rejected.session_kind = session_kind;
        rejected.replay_enabled = replay_enabled;
        rejected.hosted_replay_orders = hosted_replay_orders;
        rejected.hosted_replay_running = hosted_replay_orders;
        rejected.execution_config.kind = StrategyKind::Lua;
        rejected.execution_config.lua.source = "function on_bar(ctx, bar) return 'buy' end".into();
        let mut rejected_state = test_state(rejected);
        let err = handle_command(
            ServiceCommand::ArmExecutionStrategy,
            &mut rejected_state,
            &event_tx,
            &market_tx,
            service_internal_tx.clone(),
        )
        .await
        .expect_err("live and hosted replay Lua arms must fail closed");
        assert!(err.to_string().contains("cached Tradovate replay"));
        assert!(
            !rejected_state
                .session
                .as_ref()
                .unwrap()
                .execution_runtime
                .armed
        );
    }
}
