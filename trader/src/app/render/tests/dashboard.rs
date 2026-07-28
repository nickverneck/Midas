use super::support::*;

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
