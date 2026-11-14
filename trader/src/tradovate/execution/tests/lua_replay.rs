use super::*;

#[test]
fn lua_mutable_evaluator_accumulates_while_immutable_paths_reject() {
    let mut session = test_session();
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_config.kind = StrategyKind::Lua;
    session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;
    session.execution_config.lua.source = r#"
        count = 0
        function on_bar(ctx, bar)
            count = count + 1
            if count >= 2 then return 'buy' end
            return 'hold'
        end
    "#
    .into();
    let bars = (0..2)
        .map(|i| crate::broker::Bar {
            ts_ns: i + 1,
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: Some(4.0),
        })
        .collect::<Vec<_>>();
    let (_, signal, _, _) = evaluate_active_execution_strategy_since_mut(
        &mut session,
        &bars,
        0,
        None,
        Some(10),
        MarketHistoryUpdate::Append,
    )
    .expect("mutable Lua evaluator runs");
    assert_eq!(signal, StrategySignal::EnterLong);

    let immutable = evaluate_active_execution_strategy(&session, &bars, 0).unwrap_err();
    assert!(immutable.to_string().contains("mutable stateful evaluator"));
    let immutable_since =
        evaluate_active_execution_strategy_since(&session, &bars, 0, None).unwrap_err();
    assert!(
        immutable_since
            .to_string()
            .contains("mutable stateful evaluator")
    );
}

#[test]
fn lua_replay_evaluates_each_bar_when_source_timestamps_repeat() {
    let mut session = test_session();
    session.session_kind = SessionKind::Replay;
    session.replay_enabled = true;
    session.execution_config.kind = StrategyKind::Lua;
    session.execution_config.native_signal_timing = NativeSignalTiming::ClosedBar;
    session.execution_config.lua.source = r#"
        count = 0
        function on_bar(ctx, bar)
            count = count + 1
            if count == 3 then return 'buy' end
            return 'hold'
        end
    "#
    .into();
    let bars = [
        crate::broker::Bar {
            ts_ns: 5,
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: Some(4.0),
        },
        crate::broker::Bar {
            ts_ns: 7,
            open: 1.5,
            high: 2.5,
            low: 1.0,
            close: 2.0,
            volume: Some(5.0),
        },
        crate::broker::Bar {
            ts_ns: 7,
            open: 2.0,
            high: 3.0,
            low: 1.5,
            close: 2.5,
            volume: Some(6.0),
        },
    ];

    let mut signals = Vec::new();
    for (index, bar) in bars.iter().enumerate() {
        let prefix = &bars[..=index];
        session.market.bars = prefix.to_vec();
        session.market.history_loaded = prefix.len();
        session.execution_runtime.replay_bar_ordinal = Some(index as u64);
        let (_, signal, _, _) = evaluate_active_execution_strategy_since_mut(
            &mut session,
            prefix,
            0,
            (index > 0).then_some(bar.ts_ns),
            Some(index as u64 + 1),
            MarketHistoryUpdate::Append,
        )
        .expect("Lua replay callback evaluates each source bar");
        signals.push(signal);
    }

    assert_eq!(
        signals,
        [
            StrategySignal::Hold,
            StrategySignal::Hold,
            StrategySignal::EnterLong
        ]
    );
}
