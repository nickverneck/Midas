use super::support::*;

#[cfg(feature = "replay")]
#[test]
fn compact_dashboard_explains_ninjatrader_wait_state_on_one_row() {
    let mut app = App::new(AppConfig::default());
    app.screen = Screen::Dashboard;
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.hosted_replay_stream_connected = true;
    app.capabilities.automated_orders = false;
    let backend = TestBackend::new(100, 32);
    let mut terminal = Terminal::new(backend).expect("test terminal");

    terminal
        .draw(|frame| app.draw(frame))
        .expect("draw compact Replay dashboard");
    let rows = rendered_terminal_rows(&terminal, 100);
    let status_rows = rows
        .iter()
        .filter(|row| row.contains("Replay: waiting for account"))
        .collect::<Vec<_>>();

    assert_eq!(status_rows.len(), 1);
    assert!(!status_rows[0].contains("Strategy Status:"));
}

#[cfg(feature = "replay")]
#[test]
fn hosted_replay_dashboard_calls_zero_speed_paused_and_explains_disarmed_clock() {
    let mut app = App::new(AppConfig::default());
    app.screen = Screen::Dashboard;
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.hosted_replay_stream_connected = true;
    app.hosted_replay_speed = Some(crate::broker::NinjaTraderReplaySpeed::new(0).unwrap());
    app.strategy_runtime.armed = false;

    let lines = rendered_text(app.dashboard_summary_lines()).join("\n");
    assert!(lines.contains("Mode: Hosted Replay"), "{lines}");
    assert!(lines.contains("Provider speed: paused (0%)"), "{lines}");
    assert!(
        lines.contains("Clock paused; choose on F3 and arm to advance."),
        "{lines}"
    );
    assert!(
        lines.contains("stops this app's Replay stream connection"),
        "{lines}"
    );
    assert!(!lines.contains("discards"), "{lines}");
}

#[cfg(feature = "replay")]
#[test]
fn hosted_replay_wait_status_is_concise_while_saved_logs_keep_feed_diagnostics() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.hosted_replay_stream_connected = true;
    app.hosted_replay_clock = Some(chrono::DateTime::from_timestamp(1_800_000_000, 0).unwrap());
    app.hosted_replay_speed = Some(crate::broker::NinjaTraderReplaySpeed::new(200).unwrap());
    app.strategy_runtime.armed = true;
    app.market.contract_name = Some("ESZ6".to_string());

    let detailed_status = "Waiting for Replay quotes for ESZ6; feed diagnostics: array frames=434, clock events=41, md items=0, quote rows=0, market updates=0, closed bars=0";
    app.handle_service_event(ServiceEvent::Status(detailed_status.to_string()), &cmd_tx);

    let lines = rendered_text(app.dashboard_summary_lines()).join("\n");
    assert!(
        lines.contains("Status: Waiting for Replay quotes for ESZ6; provider clock is advancing."),
        "{lines}"
    );
    assert!(!lines.contains("array frames="), "{lines}");
    assert!(!lines.contains("md items="), "{lines}");

    let saved_log = app.build_persisted_log_body("20261011T220000Z");
    assert!(saved_log.contains(detailed_status), "{saved_log}");
    assert!(saved_log.contains("md items=0"), "{saved_log}");
}

#[cfg(feature = "replay")]
#[test]
fn hosted_replay_failure_status_remains_precise_on_dashboard() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.status = "Error: NinjaTrader Replay provider closed the WebSocket unexpectedly".to_string();

    let lines = rendered_text(app.dashboard_summary_lines()).join("\n");
    assert!(
        lines.contains(
            "Status: Error: NinjaTrader Replay provider closed the WebSocket unexpectedly"
        ),
        "{lines}"
    );
}

#[cfg(feature = "replay")]
#[test]
fn stopped_hosted_replay_dashboard_hides_stale_armed_engine_summary() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.session_mode = EngineCreateMode::LiveReplay;
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.engine_socket_path = Some(PathBuf::from("/tmp/trader-engine-41.sock"));
    app.hosted_replay_stream_connected = true;
    app.capabilities.automated_orders = true;
    app.accounts = vec![account(1, "RPL123")];
    app.hosted_replay_clock = Some(chrono::DateTime::from_timestamp(1_800_000_000, 0).unwrap());
    app.strategy_runtime.armed = true;

    let engine = running_engine(41, true);
    let key = engine_key(engine.id);
    let mut summary = crate::app::EngineSummary::from_running_engine(&engine);
    let mut config = crate::strategy::ExecutionStrategyConfig::default();
    config.kind = StrategyKind::Native;
    config.native_strategy = NativeStrategyKind::EmaCross;
    summary.apply_event(&ServiceEvent::ExecutionState(ExecutionStateSnapshot {
        config,
        runtime: ExecutionRuntimeSnapshot {
            armed: true,
            ..ExecutionRuntimeSnapshot::default()
        },
        ..ExecutionStateSnapshot::default()
    }));
    app.engine_summaries.push(summary);
    app.active_engine_key = Some(key);

    app.handle_service_event(ServiceEvent::Disconnected, &cmd_tx);

    assert!(
        app.active_engine_summary()
            .is_some_and(crate::app::EngineSummary::strategy_is_armed)
    );
    let lines = rendered_text(app.dashboard_summary_lines()).join("\n");
    assert!(lines.contains("Replay stream: disconnected"), "{lines}");
    assert!(
        lines.contains("Engine strategy: inactive (Replay disconnected)"),
        "{lines}"
    );
    assert!(lines.contains("Provider clock: disconnected"), "{lines}");
    assert!(lines.contains("Provider speed: stopped"), "{lines}");
    assert!(
        lines.contains("Replay is stopped; configure a stream in Replay Setup to continue."),
        "{lines}"
    );
    assert!(!lines.contains("armed"), "{lines}");
    assert!(!lines.contains("Clock advances"), "{lines}");
    assert!(!lines.contains("x stops this app's Replay"), "{lines}");
}

#[cfg(feature = "replay")]
#[test]
fn hosted_replay_speed_error_gives_a_next_step_without_sending_change() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.hosted_replay_speed = Some(crate::broker::NinjaTraderReplaySpeed::new(0).unwrap());
    app.strategy_runtime.armed = false;

    app.handle_dashboard_key(key(KeyCode::Char(']')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
    assert!(app.status.contains("Choose an executable strategy with F3"));
    assert!(app.status.contains("continue to arm it"));
}

#[cfg(feature = "replay")]
#[test]
fn hosted_replay_connection_status_does_not_look_like_tradovate_simulation() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::Connected {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            user_name: Some("NinjaTrader Replay".to_string()),
            auth_mode: AuthMode::TokenFile,
            session_kind: crate::broker::SessionKind::NinjaTraderReplay,
            capabilities: BrokerCapabilities::default(),
        },
        &cmd_tx,
    );

    assert_eq!(app.status, "Connected to NinjaTrader Hosted Replay");
    assert!(!app.status.contains("Tradovate Simulation"));
}

#[test]
fn ninjatrader_replay_strategy_status_names_its_order_route_in_every_state() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.capabilities.automated_orders = false;
    app.accounts = vec![account(1, "NinjaTrader Replay")];
    app.hosted_replay_clock = Some(chrono::DateTime::from_timestamp(1_800_000_000, 0).unwrap());

    assert_eq!(
        app.strategy_runtime_summary(),
        "NinjaTrader Replay strategy orders unavailable"
    );

    app.capabilities.automated_orders = true;
    assert_eq!(
        app.strategy_runtime_summary(),
        "NinjaTrader Replay: strategy disarmed"
    );

    app.strategy_runtime.armed = true;
    assert_eq!(
        app.strategy_runtime_summary(),
        "NinjaTrader Replay: strategy armed"
    );

    app.strategy_runtime.last_summary = "Waiting for a closed bar".to_string();
    assert_eq!(
        app.strategy_runtime_summary(),
        "NinjaTrader Replay: Waiting for a closed bar"
    );
}

#[cfg(feature = "manual-orders")]
#[test]
fn dashboard_manual_orders_sync_selected_account_first() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;

    app.handle_dashboard_key(key(KeyCode::Char('b')), &cmd_tx);

    expect_select_account(&mut cmd_rx, 2);
    match cmd_rx.try_recv().expect("expected manual-order command") {
        ServiceCommand::ManualOrder {
            action: ManualOrderAction::Buy,
        } => {}
        _ => panic!("expected buy manual-order command"),
    }
}

#[cfg(not(feature = "manual-orders"))]
#[test]
fn dashboard_manual_order_hotkeys_disabled_without_feature() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    enable_tradovate_controls(&mut app);
    app.screen = Screen::Dashboard;
    app.accounts = vec![account(1, "DEMO4769136"), account(2, "CHMMMLE422")];
    app.selected_account = 1;

    assert!(!app.header_help_text().contains("b/s/c manual"));
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .all(|line| !line.contains("b/s/c"))
    );

    for ch in ['b', 'c', 's'] {
        app.handle_dashboard_key(key(KeyCode::Char(ch)), &cmd_tx);
    }

    assert!(cmd_rx.try_recv().is_err());
}

#[test]
fn dashboard_hides_manual_order_affordances_without_capability() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();
    app.screen = Screen::Dashboard;
    app.capabilities.manual_orders = false;
    app.capabilities.automated_orders = false;
    app.accounts = vec![account(1, "SIM")];
    app.account_snapshots = vec![account_snapshot(1, None, None, None, None)];

    assert!(!app.header_help_text().contains("b/s/c manual"));
    assert!(!app.header_help_text().contains("timing/reversal mode"));
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .all(|line| !line.contains("b/s/c"))
    );
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .any(|line| line.contains("Keys v"))
    );

    app.handle_dashboard_key(key(KeyCode::Char('b')), &cmd_tx);

    assert!(cmd_rx.try_recv().is_err());
}

#[test]
fn dashboard_selected_unrealized_pnl_reprices_from_live_market_without_account_refresh() {
    let mut app = App::new(AppConfig::default());
    app.accounts = vec![account(1, "SIM")];
    app.account_snapshots = vec![account_snapshot(1, Some(1.0), Some(5_000.0), None, None)];
    app.market.contract_id = Some(3_570_918);
    app.market.contract_name = Some("ESM6".to_string());
    app.market.value_per_point = Some(50.0);
    app.market.bars = vec![crate::broker::Bar {
        ts_ns: 1,
        open: 5_000.0,
        high: 5_001.0,
        low: 4_999.0,
        close: 5_001.0,
        volume: None,
    }];

    let snapshot = app.selected_snapshot().expect("selected snapshot");
    assert_eq!(app.selected_contract_unrealized_pnl(snapshot), Some(50.0));

    app.market.bars.last_mut().expect("forming bar").close = 5_002.0;

    let snapshot = app.selected_snapshot().expect("same account snapshot");
    assert_eq!(app.selected_contract_unrealized_pnl(snapshot), Some(100.0));
    assert!(
        rendered_text(app.stats_lines())
            .iter()
            .any(|line| line.contains("Selected unreal") && line.contains("100.00"))
    );
}

#[test]
fn dashboard_visual_toggle_updates_state_without_sending_commands() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, mut cmd_rx) = unbounded_channel();

    assert!(!app.dashboard_visuals_enabled);
    app.handle_dashboard_key(key(KeyCode::Char('v')), &cmd_tx);
    assert!(app.dashboard_visuals_enabled);
    assert!(cmd_rx.try_recv().is_err());

    app.handle_dashboard_key(key(KeyCode::Char('v')), &cmd_tx);
    assert!(!app.dashboard_visuals_enabled);
    assert!(cmd_rx.try_recv().is_err());
}

#[cfg(feature = "replay")]
#[test]
fn dashboard_chart_renders_compact_utc_time_and_ohlc_price_axes_at_100_columns() {
    use chrono::{NaiveDate, TimeZone, Utc};

    let mut app = App::new(AppConfig::default());
    app.market.contract_name = Some("ESZ6".to_string());
    app.market.tick_size = Some(0.25);
    app.market.history_loaded = 3;
    let start = Utc
        .from_utc_datetime(
            &NaiveDate::from_ymd_opt(2026, 10, 9)
                .expect("date")
                .and_hms_opt(9, 30, 0)
                .expect("time"),
        )
        .timestamp_nanos_opt()
        .expect("timestamp");
    app.market.bars = vec![
        crate::broker::Bar {
            ts_ns: start,
            open: 102.0,
            high: 104.0,
            low: 100.0,
            close: 103.0,
            volume: Some(1.0),
        },
        crate::broker::Bar {
            ts_ns: start + 60_000_000_000,
            open: 103.0,
            high: 108.0,
            low: 101.0,
            close: 106.0,
            volume: Some(1.0),
        },
        crate::broker::Bar {
            ts_ns: start + 120_000_000_000,
            open: 106.0,
            high: 107.0,
            low: 102.0,
            close: 104.0,
            volume: Some(1.0),
        },
    ];

    let mut terminal = Terminal::new(TestBackend::new(100, 44)).expect("terminal");
    terminal
        .draw(|frame| app.render_dashboard(frame, frame.area()))
        .expect("render dashboard");
    let rows = rendered_terminal_rows(&terminal, 100);
    let text = rows.join("\n");

    for visible_axis_label in [
        "Price", "Time UTC", "99.60", "104.00", "108.40", "09:30", "09:31", "09:32",
    ] {
        assert!(
            text.contains(visible_axis_label),
            "expected chart label {visible_axis_label:?} in:\n{text}"
        );
    }
}

#[cfg(feature = "replay")]
#[test]
fn dashboard_chart_shows_one_time_tick_without_duplicate_labels() {
    use chrono::{NaiveDate, TimeZone, Utc};

    let mut app = App::new(AppConfig::default());
    app.market.tick_size = Some(0.25);
    app.market.history_loaded = 1;
    let timestamp = Utc
        .from_utc_datetime(
            &NaiveDate::from_ymd_opt(2026, 10, 9)
                .expect("date")
                .and_hms_opt(9, 30, 0)
                .expect("time"),
        )
        .timestamp_nanos_opt()
        .expect("timestamp");
    app.market.bars = vec![crate::broker::Bar {
        ts_ns: timestamp,
        open: 5_000.0,
        high: 5_000.5,
        low: 4_999.75,
        close: 5_000.25,
        volume: Some(1.0),
    }];

    let mut terminal = Terminal::new(TestBackend::new(66, 30)).expect("terminal");
    terminal
        .draw(|frame| app.render_chart(frame, frame.area()))
        .expect("render chart");
    let text = rendered_terminal_rows(&terminal, 66).join("\n");
    assert_eq!(text.matches("09:30").count(), 1, "{text}");
    assert!(text.contains("Price"));
    assert!(text.contains("Time UTC"));
}

#[cfg(feature = "replay")]
#[test]
fn empty_hosted_replay_chart_waits_for_quotes_when_contract_is_selected() {
    let mut app = App::new(AppConfig::default());
    app.session_kind = crate::broker::SessionKind::NinjaTraderReplay;
    app.market.contract_name = Some("ESZ6".to_string());

    let mut terminal = Terminal::new(TestBackend::new(66, 12)).expect("terminal");
    terminal
        .draw(|frame| app.render_chart(frame, frame.area()))
        .expect("render hosted Replay chart");
    let text = rendered_terminal_rows(&terminal, 66).join("\n");

    assert!(text.contains("Waiting for Replay quotes."), "{text}");
    assert!(!text.contains("Select a contract"), "{text}");
}

#[cfg(feature = "replay")]
#[test]
fn empty_live_chart_still_prompts_for_a_contract() {
    let app = App::new(AppConfig::default());

    let mut terminal = Terminal::new(TestBackend::new(66, 12)).expect("terminal");
    terminal
        .draw(|frame| app.render_chart(frame, frame.area()))
        .expect("render empty live chart");
    let text = rendered_terminal_rows(&terminal, 66).join("\n");

    assert!(
        text.contains("Select a contract to load history + live bars."),
        "{text}"
    );
}

#[cfg(feature = "replay")]
#[test]
fn dashboard_chart_keeps_replay_tick_window_endpoints_distinct_when_truncated() {
    use chrono::DateTime;

    let mut app = App::new(AppConfig::default());
    app.market.contract_name = Some("ESZ6".to_string());
    app.market.tick_size = Some(0.25);
    app.market.history_loaded = 212;
    let start = DateTime::parse_from_rfc3339("2026-10-09T13:30:00Z")
        .expect("valid replay start")
        .timestamp_nanos_opt()
        .expect("timestamp");
    app.market.bars = (0..212)
        .map(|index| {
            // Match the captured 212-row one-minute replay window: first
            // update at .471s, final update at 59.839s.
            let ts_ns = start + 471_000_000 + index * 59_368_000_000 / 211;
            let close = 5_000.0 + (index % 7) as f64 * 0.25;
            crate::broker::Bar {
                ts_ns,
                open: close,
                high: close + 0.25,
                low: close - 0.25,
                close,
                volume: Some(1.0),
            }
        })
        .collect();

    // The dashboard intentionally plots only the newest 180 bars. Check the
    // actual terminal renderer at both review sizes, including the visible
    // viewport's true first and last timestamps.
    for width in [100_usize, 140] {
        let mut terminal = Terminal::new(TestBackend::new(width as u16, 44)).expect("terminal");
        terminal
            .draw(|frame| app.render_dashboard(frame, frame.area()))
            .expect("render dashboard");
        let text = rendered_terminal_rows(&terminal, width).join("\n");

        assert!(
            text.contains("13:30:09"),
            "first visible tick missing:\n{text}"
        );
        assert!(text.contains("13:30:34"), "middle tick missing:\n{text}");
        assert!(
            text.contains("13:30:59"),
            "last visible tick missing:\n{text}"
        );
        assert_eq!(text.matches("13:30:09").count(), 1, "{text}");
        assert_eq!(text.matches("13:30:59").count(), 1, "{text}");
        assert!(!text.contains("13:30:00"), "truncated tick leaked:\n{text}");
    }

    // The screenshot regression had the same displayed second at both ends.
    // If a market snapshot supplies repeated timestamps, preserve the visible
    // sequence positions so the endpoints do not look like one collapsed tick.
    let repeated_timestamp = start + 59_839_000_000;
    for bar in &mut app.market.bars {
        bar.ts_ns = repeated_timestamp;
    }
    for width in [100_usize, 140] {
        let mut terminal = Terminal::new(TestBackend::new(width as u16, 44)).expect("terminal");
        terminal
            .draw(|frame| app.render_dashboard(frame, frame.area()))
            .expect("render dashboard");
        let text = rendered_terminal_rows(&terminal, width).join("\n");

        assert!(
            text.contains("13:30:59#1"),
            "first tick label missing:\n{text}"
        );
        assert!(
            text.contains("13:30:59#91"),
            "middle tick label missing:\n{text}"
        );
        assert!(
            text.contains("13:30:59#180"),
            "last tick label missing:\n{text}"
        );
    }
}

#[cfg(feature = "replay")]
#[test]
fn dashboard_chart_time_labels_preserve_non_monotonic_and_equal_timestamps() {
    use chrono::DateTime;

    let app = App::new(AppConfig::default());
    let start = DateTime::parse_from_rfc3339("2026-10-09T13:30:00Z")
        .expect("valid timestamp")
        .timestamp_nanos_opt()
        .expect("timestamp");
    let bar = |ts_ns| crate::broker::Bar {
        ts_ns,
        open: 1.0,
        high: 1.0,
        low: 1.0,
        close: 1.0,
        volume: None,
    };

    assert_eq!(
        app.dashboard_chart_time_labels(&[
            bar(start + 59_000_000_000),
            bar(start),
            bar(start + 59_000_000_000),
        ]),
        ["13:30:59#1", "13:30:00#2", "13:30:59#3"]
    );
    assert_eq!(
        app.dashboard_chart_time_labels(&[bar(start), bar(start), bar(start)]),
        ["13:30:00#1", "13:30:00#2", "13:30:00#3"]
    );
}
