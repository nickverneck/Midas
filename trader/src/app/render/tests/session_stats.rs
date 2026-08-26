use super::support::*;

#[test]
fn session_stats_track_wins_losses_and_flats_from_balance_deltas() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for balance in [1_000.0, 1_015.0, 1_005.0, 1_030.0, 1_030.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.account_id, 7);
    assert_eq!(stats.sample_count, 5);
    assert_eq!(stats.wins, 2);
    assert_eq!(stats.losses, 1);
    assert_eq!(stats.flat_moves, 1);
    assert_eq!(stats.event_count(), 3);
    assert_eq!(stats.avg_win(), Some(20.0));
    assert_eq!(stats.max_win, Some(25.0));
    assert_eq!(stats.avg_loss_signed(), Some(-10.0));
    assert_eq!(stats.max_loss_signed(), Some(-10.0));
    assert_eq!(stats.session_pnl(), 30.0);
    assert_eq!(stats.win_rate(), Some(2.0 / 3.0));
}

#[test]
fn session_stats_reports_pnl_per_hour() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_000.0)]),
        &cmd_tx,
    );
    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 1_020.0)]),
        &cmd_tx,
    );

    let end = chrono::Utc::now();
    let stats = app
        .session_stats
        .accounts
        .values_mut()
        .next()
        .expect("expected stats");
    stats.started_at_utc = end - chrono::Duration::hours(2);
    stats.last_updated_at_utc = end;

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("PnL/H: Net +10.00/h  Trade +10.00/h"))
    );

    let event_lines = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_lines
            .iter()
            .any(|line| line == "Hourly PnL/H (local): Net | Trade | Fees")
    );
    assert!(event_lines.iter().any(|line| {
        line.contains("Net +20.00/h")
            && line.contains("Trade +20.00/h")
            && line.contains("Fees 0.00/h")
    }));

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("net_pnl_per_hour: +10.00/h"));
    assert!(body.contains("trade_pnl_per_hour: +10.00/h"));
    assert!(body.contains("hourly_local:"));
    assert!(body.contains("trade_per_hour=+20.00/h"));
}

#[test]
fn session_stats_attributes_balance_deltas_to_long_and_short_side() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.0, 1.0),
        balance_snapshot_with_position(7, "SIM", 1_015.0, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_015.0, -1.0),
        balance_snapshot_with_position(7, "SIM", 1_005.0, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.long_side.events, 1);
    assert_eq!(stats.long_side.wins, 1);
    assert_eq!(stats.long_side.losses, 0);
    assert_eq!(stats.long_side.pnl, 15.0);
    assert_eq!(stats.short_side.events, 1);
    assert_eq!(stats.short_side.wins, 0);
    assert_eq!(stats.short_side.losses, 1);
    assert_eq!(stats.short_side.pnl, -10.0);
    assert_eq!(stats.events[0].side, SessionTradeSide::Long);
    assert_eq!(
        stats.events[0].previous_position_side,
        SessionTradeSide::Long
    );
    assert_eq!(
        stats.events[0].current_position_side,
        SessionTradeSide::Flat
    );
    assert_eq!(stats.events[1].side, SessionTradeSide::Short);
    assert_eq!(
        stats.events[1].previous_position_side,
        SessionTradeSide::Short
    );
    assert_eq!(
        stats.events[1].current_position_side,
        SessionTradeSide::Flat
    );

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| { line.contains("Side PnL: Long +15.00 (1/0)  Short -10.00 (0/1)") })
    );

    let event_lines = app.session_stats_event_lines(8);
    let long_event_line = event_lines
        .iter()
        .find(|line| line.to_string().contains("balance long"))
        .expect("expected long balance event line");
    let short_event_line = event_lines
        .iter()
        .find(|line| line.to_string().contains("balance short"))
        .expect("expected short balance event line");
    assert!(line_span_with_fg(long_event_line, "long", Color::Cyan));
    assert!(line_span_with_fg(long_event_line, "1015.00", Color::Green));
    assert!(line_span_with_fg(long_event_line, "+15.00", Color::Green));
    assert!(line_span_with_fg(short_event_line, "short", Color::Magenta));
    assert!(line_span_with_fg(short_event_line, "1005.00", Color::Red));
    assert!(line_span_with_fg(short_event_line, "-10.00", Color::Red));

    let event_text = event_lines
        .iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance long pos long->flat"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance short pos short->flat"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("long_events: 1"));
    assert!(body.contains("long_pnl: +15.00"));
    assert!(body.contains("short_events: 1"));
    assert!(body.contains("short_pnl: -10.00"));
    assert!(body.contains("side=long"));
    assert!(body.contains("pos=long->flat"));
    assert!(body.contains("side=short"));
    assert!(body.contains("pos=short->flat"));
}

#[test]
fn session_stats_event_lines_include_position_transition_context() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.0, -1.0),
        balance_snapshot_with_position(7, "SIM", 1_012.5, 1.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.events[0].side, SessionTradeSide::Short);
    assert_eq!(
        stats.events[0].previous_position_side,
        SessionTradeSide::Short
    );
    assert_eq!(
        stats.events[0].current_position_side,
        SessionTradeSide::Long
    );

    let event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("balance short pos short->long"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("side=short"));
    assert!(body.contains("pos=short->long"));
}

#[test]
fn session_stats_filters_fee_only_and_mixed_fee_balance_deltas() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.market.tick_size = Some(0.25);
    app.market.value_per_point = Some(5.0);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.00, 1.0),
        balance_snapshot_with_position(7, "SIM", 1_005.69, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_004.78, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_004.78, -1.0),
        balance_snapshot_with_position(7, "SIM", 996.37, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.wins, 1);
    assert_eq!(stats.losses, 1);
    assert_eq!(stats.flat_moves, 1);
    assert_eq!(stats.fee_events, 3);
    assert_eq!(stats.event_count(), 3);
    assert_money_eq(stats.session_pnl(), -3.63);
    assert_money_eq(stats.trade_pnl_ex_fees(), -1.25);
    assert_money_eq(stats.total_fees, -2.38);
    assert_money_eq(stats.long_side.pnl, 6.25);
    assert_money_eq(stats.short_side.pnl, -7.50);
    assert_eq!(stats.long_side.wins, 1);
    assert_eq!(stats.short_side.losses, 1);
    assert_eq!(stats.events[0].kind, SessionBalanceEventKind::Mixed);
    assert_eq!(stats.events[1].kind, SessionBalanceEventKind::Fee);
    assert_eq!(stats.events[2].kind, SessionBalanceEventKind::Mixed);
    assert_money_eq(stats.events[0].trade_delta, 6.25);
    assert_money_eq(stats.events[0].fee_delta, -0.56);
    assert_money_eq(stats.events[1].trade_delta, 0.0);
    assert_money_eq(stats.events[1].fee_delta, -0.91);
    assert_money_eq(stats.events[2].trade_delta, -7.50);
    assert_money_eq(stats.events[2].fee_delta, -0.91);

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: -1.25  Fees: -2.38 (3)"))
    );
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Wins: 1  Losses: 1  Flats: 1  Fee Events: 3"))
    );

    let event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade -7.50 fees -0.91"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("fee trade 0.00 fees -0.91"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade +6.25 fees -0.56"))
    );

    app.handle_session_stats_key(key(KeyCode::Char('f')), &cmd_tx);
    assert!(!app.session_stats_show_fees);
    assert_eq!(app.status, "Session stats fee rows hidden.");

    let overview_text = rendered_text(app.session_stats_overview_lines());
    assert!(overview_text.iter().any(|line| line == "Fees: hidden"));

    let hidden_account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        hidden_account_lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: -1.25  Fees hidden"))
    );
    assert!(
        hidden_account_lines
            .iter()
            .any(|line| line.contains("Wins: 1  Losses: 1  Flats: 1"))
    );
    assert!(
        !hidden_account_lines
            .iter()
            .any(|line| line.contains("Fee Events"))
    );

    let hidden_event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line == "Hourly PnL/H (local): Net | Trade")
    );
    assert!(
        hidden_event_text
            .iter()
            .all(|line| !line.contains("Hourly PnL/H (local): Net | Trade | Fees"))
    );
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("| W/L 1/1 | 2 trade events"))
    );
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("balance short") && line.contains("trade -7.50"))
    );
    assert!(
        hidden_event_text
            .iter()
            .any(|line| line.contains("balance long") && line.contains("trade +6.25"))
    );
    assert!(
        !hidden_event_text
            .iter()
            .any(|line| line.contains("fee trade") || line.contains("fees"))
    );
    assert!(
        !hidden_event_text
            .iter()
            .any(|line| line.contains("mixed trade"))
    );

    app.handle_session_stats_key(key(KeyCode::Char('F')), &cmd_tx);
    assert!(app.session_stats_show_fees);
    assert_eq!(app.status, "Session stats fee rows shown.");

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("fee_events: 3"));
    assert!(body.contains("total_fees: -2.38"));
    assert!(body.contains("trade_pnl_ex_fees: -1.25"));
    assert!(body.contains("kind=fee"));
    assert!(body.contains("trade_delta=+6.25 fee_delta=-0.56"));
    assert!(body.contains("trade_delta=-7.50 fee_delta=-0.91"));
}

#[test]
fn session_stats_classifies_es_commissions_as_fees() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.market.tick_size = Some(0.25);
    app.market.value_per_point = Some(50.0);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    for snapshot in [
        balance_snapshot_with_position(7, "SIM", 1_000.00, 0.0),
        balance_snapshot_with_position(7, "SIM", 997.12, 1.0),
        balance_snapshot_with_position(7, "SIM", 1_056.74, 0.0),
        balance_snapshot_with_position(7, "SIM", 1_053.86, -1.0),
        balance_snapshot_with_position(7, "SIM", 1_025.98, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked session stats");
    assert_eq!(stats.wins, 1);
    assert_eq!(stats.losses, 1);
    assert_eq!(stats.flat_moves, 0);
    assert_eq!(stats.fee_events, 4);
    assert_eq!(stats.event_count(), 4);
    assert_money_eq(stats.session_pnl(), 25.98);
    assert_money_eq(stats.trade_pnl_ex_fees(), 37.50);
    assert_money_eq(stats.total_fees, -11.52);
    assert_money_eq(stats.long_side.pnl, 62.50);
    assert_money_eq(stats.short_side.pnl, -25.00);
    assert_eq!(stats.long_side.wins, 1);
    assert_eq!(stats.short_side.losses, 1);
    assert_eq!(stats.events[0].kind, SessionBalanceEventKind::Fee);
    assert_eq!(stats.events[1].kind, SessionBalanceEventKind::Mixed);
    assert_eq!(stats.events[2].kind, SessionBalanceEventKind::Fee);
    assert_eq!(stats.events[3].kind, SessionBalanceEventKind::Mixed);
    assert_money_eq(stats.events[0].trade_delta, 0.0);
    assert_money_eq(stats.events[0].fee_delta, -2.88);
    assert_money_eq(stats.events[1].trade_delta, 62.50);
    assert_money_eq(stats.events[1].fee_delta, -2.88);
    assert_money_eq(stats.events[2].trade_delta, 0.0);
    assert_money_eq(stats.events[2].fee_delta, -2.88);
    assert_money_eq(stats.events[3].trade_delta, -25.00);
    assert_money_eq(stats.events[3].fee_delta, -2.88);

    let account_lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: +37.50  Fees: -11.52 (4)"))
    );
    assert!(
        account_lines
            .iter()
            .any(|line| line.contains("Wins: 1  Losses: 1  Flats: 0  Fee Events: 4"))
    );

    let event_text = app
        .session_stats_event_lines(8)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("fee trade 0.00 fees -2.88"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade +62.50 fees -2.88"))
    );
    assert!(
        event_text
            .iter()
            .any(|line| line.contains("mixed trade -25.00 fees -2.88"))
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("fee_events: 4"));
    assert!(body.contains("total_fees: -11.52"));
    assert!(body.contains("trade_pnl_ex_fees: +37.50"));
    assert!(body.contains("trade_delta=+62.50 fee_delta=-2.88"));
    assert!(body.contains("trade_delta=-25.00 fee_delta=-2.88"));
}

#[test]
fn session_stats_uses_gc_snapshot_fees_and_ignores_mark_noise() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    let snapshot = |balance: f64, realized_pnl: f64, fees: f64, position: f64| {
        let mut snapshot = balance_snapshot_with_position(7, "SIM", balance, position);
        snapshot.realized_pnl = Some(realized_pnl);
        snapshot.fees = Some(fees);
        snapshot
    };

    for snapshot in [
        snapshot(1_000.00, 0.00, 0.00, 0.0),
        snapshot(996.90, 0.00, 3.10, 1.0),
        snapshot(995.30, 0.00, 3.10, 1.0),
        snapshot(1_192.20, 196.90, 6.20, 0.0),
    ] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![snapshot]),
            &cmd_tx,
        );
    }

    let stats = app
        .selected_session_stats()
        .expect("expected tracked GC session stats");
    assert_eq!(stats.wins, 1);
    assert_eq!(stats.losses, 0);
    assert_eq!(stats.fee_events, 2);
    assert_money_eq(stats.session_pnl(), 192.20);
    assert_money_eq(stats.trade_pnl_ex_fees(), 200.00);
    assert_money_eq(stats.total_fees, -6.20);
    assert_eq!(stats.events[0].kind, SessionBalanceEventKind::Fee);
    assert_eq!(stats.events[1].kind, SessionBalanceEventKind::Mark);
    assert_eq!(stats.events[2].kind, SessionBalanceEventKind::Mixed);
    assert_money_eq(stats.events[1].trade_delta, 0.0);

    let lines = app
        .selected_session_stats_lines()
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(
        lines
            .iter()
            .any(|line| line.contains("Trade PnL Ex Fees: +200.00  Fees: -6.20 (2)"))
    );
    assert!(lines.iter().any(|line| line.contains("Wins: 1  Losses: 0")));
}

#[test]
fn session_stats_identity_keeps_same_account_separate_by_active_engine() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.set_running_engines(vec![running_engine(10, true), running_engine(11, true)]);
    app.handle_service_event(
        ServiceEvent::AccountsLoaded(vec![account(7, "SIM")]),
        &cmd_tx,
    );

    let first_key = engine_key(10);
    app.enter_engine_session_for_key(
        first_key.clone(),
        PathBuf::from("/tmp/trader-engine-10.sock"),
    );
    for balance in [1_000.0, 1_010.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }
    assert_eq!(
        app.selected_session_stats()
            .expect("first engine stats")
            .session_pnl(),
        10.0
    );

    let second_key = engine_key(11);
    app.enter_engine_session_for_key(
        second_key.clone(),
        PathBuf::from("/tmp/trader-engine-11.sock"),
    );
    for balance in [2_000.0, 1_980.0] {
        app.handle_service_event(
            ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", balance)]),
            &cmd_tx,
        );
    }

    assert_eq!(app.session_stats.accounts.len(), 2);
    assert_eq!(
        app.selected_session_stats()
            .expect("second engine stats")
            .session_pnl(),
        -20.0
    );

    app.enter_engine_session_for_key(first_key, PathBuf::from("/tmp/trader-engine-10.sock"));
    assert_eq!(
        app.selected_session_stats()
            .expect("first engine stats after switch")
            .session_pnl(),
        10.0
    );

    let body = app.build_persisted_log_body("20260403T120000Z");
    assert!(body.contains("engine_identity: engine:/tmp/trader-engine-10.sock"));
    assert!(body.contains("engine_identity: engine:/tmp/trader-engine-11.sock"));
}

#[test]
fn account_snapshot_changes_do_not_mutate_broker_attributed_engine_history() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.accounts = vec![account(7, "SIM")];
    let history = EngineHistorySnapshot {
        run_id: "gc-run".to_string(),
        started_at_utc: chrono::Utc::now(),
        updated_at_utc: None,
        account_id: 7,
        account_name: "SIM".to_string(),
        contract_id: 4_095_561,
        contract_name: "GCQ6".to_string(),
        position_qty: 0,
        average_entry_price: None,
        realized_pnl: 240.0,
        unrealized_pnl: 0.0,
        fees: 2.0,
        wins: 1,
        losses: 0,
        fills: Vec::new(),
    };
    app.handle_service_event(ServiceEvent::EngineHistoryUpdated(history.clone()), &cmd_tx);

    app.handle_service_event(
        ServiceEvent::AccountSnapshotsLoaded(vec![balance_snapshot(7, "SIM", 101_000.0)]),
        &cmd_tx,
    );

    assert_eq!(app.engine_history.as_ref(), Some(&history));
    let lines = rendered_text(app.selected_session_stats_lines());
    assert!(lines.iter().any(|line| line.contains("GCQ6")));
    assert!(lines.iter().any(|line| line.contains("240.00")));
    assert!(
        lines
            .iter()
            .any(|line| line.contains("PnL/H: Net n/a  Trade n/a"))
    );
    assert!(lines.iter().all(|line| !line.contains("101000")));

    let event_lines = rendered_text(app.session_stats_event_lines(8));
    assert_eq!(
        event_lines,
        vec!["No broker-attributed fills for this engine run yet."]
    );
}

#[test]
fn broker_engine_history_restores_hourly_pnl_and_colorized_recent_fills() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.accounts = vec![account(7, "SIM")];

    let fill = |ts: &str, side, realized_pnl: f64, fill_id: i64| crate::broker::EngineHistoryFill {
        fill_id,
        order_id: fill_id + 100,
        ts_ns: chrono::DateTime::parse_from_rfc3339(ts)
            .expect("valid fill timestamp")
            .timestamp_nanos_opt()
            .expect("timestamp in range"),
        side,
        qty: 1,
        price: 4_570.0 + fill_id as f64,
        realized_pnl,
    };
    let fills = vec![
        fill(
            "2026-08-20T13:05:00Z",
            crate::broker::TradeMarkerSide::Buy,
            40.0,
            1,
        ),
        fill(
            "2026-08-20T13:35:00Z",
            crate::broker::TradeMarkerSide::Sell,
            10.0,
            2,
        ),
        fill(
            "2026-08-20T14:05:00Z",
            crate::broker::TradeMarkerSide::Sell,
            -25.0,
            3,
        ),
    ];
    let history = crate::broker::EngineHistorySnapshot {
        run_id: "gc-run".to_string(),
        started_at_utc: chrono::DateTime::parse_from_rfc3339("2026-08-20T12:05:00Z")
            .expect("valid start timestamp")
            .with_timezone(&chrono::Utc),
        updated_at_utc: None,
        account_id: 7,
        account_name: "SIM".to_string(),
        contract_id: 4_095_561,
        contract_name: "GCZ6".to_string(),
        position_qty: 0,
        average_entry_price: None,
        realized_pnl: 25.0,
        unrealized_pnl: 0.0,
        fees: 0.0,
        wins: 2,
        losses: 1,
        fills,
    };
    app.handle_service_event(ServiceEvent::EngineHistoryUpdated(history), &cmd_tx);

    let selected_lines = rendered_text(app.selected_session_stats_lines());
    assert!(
        selected_lines
            .iter()
            .any(|line| line.contains("PnL/H: Net +12.50/h  Trade +12.50/h"))
    );

    let event_lines = app.session_stats_event_lines(12);
    let event_text = rendered_text(event_lines.clone());
    assert!(
        event_text
            .iter()
            .any(|line| line == "Hourly Trade PnL/H (local, net of fees)")
    );
    let first_hour = chrono::DateTime::parse_from_rfc3339("2026-08-20T13:05:00Z")
        .expect("valid first timestamp")
        .with_timezone(&chrono::Local)
        .format("%H:00")
        .to_string();
    let second_hour = chrono::DateTime::parse_from_rfc3339("2026-08-20T14:05:00Z")
        .expect("valid second timestamp")
        .with_timezone(&chrono::Local)
        .format("%H:00")
        .to_string();
    let first_hour_line = event_lines
        .iter()
        .find(|line| line.to_string().starts_with(&first_hour))
        .expect("first local hour row");
    let second_hour_line = event_lines
        .iter()
        .find(|line| line.to_string().starts_with(&second_hour))
        .expect("second local hour row");
    assert!(first_hour_line.to_string().contains("+50.00/h (2 fills)"));
    assert!(second_hour_line.to_string().contains("-25.00/h (1 fills)"));
    assert!(line_span_with_fg(first_hour_line, "+50.00/h", Color::Green));
    assert!(line_span_with_fg(second_hour_line, "-25.00/h", Color::Red));

    let positive_fill = event_lines
        .iter()
        .find(|line| line.to_string().contains("fill 1 order 101"))
        .expect("positive recent fill row");
    let negative_fill = event_lines
        .iter()
        .find(|line| line.to_string().contains("fill 3 order 103"))
        .expect("negative recent fill row");
    assert!(line_span_with_fg(positive_fill, "+40.00", Color::Green));
    assert!(line_span_with_fg(negative_fill, "-25.00", Color::Red));
    assert!(line_span_with_fg(positive_fill, "BUY", Color::Cyan));
    assert!(line_span_with_fg(negative_fill, "SELL", Color::Magenta));

    let hourly_index = event_text
        .iter()
        .position(|line| line == "Hourly Trade PnL/H (local, net of fees)")
        .expect("hourly block");
    let recent_fill_index = event_text
        .iter()
        .position(|line| line.contains("fill 3 order 103"))
        .expect("recent fill row");
    assert!(recent_fill_index > hourly_index);
}

#[test]
fn broker_engine_history_keeps_a_recent_fill_when_hourly_rows_hit_the_limit() {
    let mut app = App::new(AppConfig::default());
    let (cmd_tx, _cmd_rx) = unbounded_channel();
    app.accounts = vec![account(7, "SIM")];
    let fills = (0..5)
        .map(|index| crate::broker::EngineHistoryFill {
            fill_id: index,
            order_id: index + 100,
            ts_ns: chrono::DateTime::parse_from_rfc3339(&format!(
                "2026-08-20T{:02}:00:00Z",
                10 + index
            ))
            .expect("valid fill timestamp")
            .timestamp_nanos_opt()
            .expect("timestamp in range"),
            side: crate::broker::TradeMarkerSide::Buy,
            qty: 1,
            price: 4_570.0,
            realized_pnl: index as f64,
        })
        .collect::<Vec<_>>();
    app.handle_service_event(
        ServiceEvent::EngineHistoryUpdated(crate::broker::EngineHistorySnapshot {
            run_id: "limit-run".to_string(),
            started_at_utc: chrono::Utc::now(),
            updated_at_utc: None,
            account_id: 7,
            account_name: "SIM".to_string(),
            contract_id: 1,
            contract_name: "GCZ6".to_string(),
            position_qty: 0,
            average_entry_price: None,
            realized_pnl: 10.0,
            unrealized_pnl: 0.0,
            fees: 0.0,
            wins: 1,
            losses: 1,
            fills,
        }),
        &cmd_tx,
    );

    let lines = app.session_stats_event_lines(4);
    assert_eq!(lines.len(), 4);
    assert!(
        lines
            .iter()
            .any(|line| line.to_string().contains("fill 4 order 104"))
    );
    assert!(
        lines
            .iter()
            .all(|line| !line.to_string().contains("fill 0 order 100"))
    );
}

#[test]
fn disabled_session_stats_screen_shows_enable_hint() {
    let mut config = AppConfig::default();
    config.session_stats_enabled = false;
    let app = App::new(config);

    let lines = app.selected_session_stats_lines();

    assert!(lines[0].to_string().contains("disabled"));
    assert!(
        app.session_stats_overview_lines()
            .iter()
            .any(|line| line.to_string().contains("TRADER_SESSION_STATS_ENABLED=1"))
    );
}
