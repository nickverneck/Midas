#[cfg(feature = "replay")]
mod replay_analytics_tests {
    use super::super::support::*;
    use crate::tradovate::replay::{
        ReplaySweepRankingDocument, ReplaySweepRankingEntry, ReplaySweepRankingMetric,
        ReplaySweepRankingOptions, ReplaySweepRankingRow,
    };
    use std::collections::BTreeMap;

    #[test]
    fn f8_opens_saved_analytics_without_an_active_engine() {
        let mut app = App::new(AppConfig::default());
        enable_tradovate_controls(&mut app);
        let (tx, _rx) = unbounded_channel();

        app.handle_key(key(KeyCode::F(8)), &tx);

        assert_eq!(app.screen, Screen::Analytics);
        assert_eq!(app.analytics_return_screen, Screen::EngineSelect);
    }

    #[test]
    fn analytics_screen_renders_empty_result_guidance() {
        let mut config = AppConfig::default();
        config.replay_result_dir = std::env::temp_dir().join(format!(
            "trader-analytics-empty-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let mut app = App::new(config);
        enable_tradovate_controls(&mut app);
        app.screen = Screen::Analytics;
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).expect("terminal");

        terminal.draw(|frame| app.draw(frame)).expect("draw");

        let rows = rendered_terminal_rows(&terminal, 120).join("\n");
        assert!(rows.contains("Saved Replay Runs"));
        assert!(rows.contains("No saved replay result selected"));
    }

    #[test]
    fn analytics_signal_tab_shows_detail_and_cycles_filters() {
        let mut app = App::new(AppConfig::default());
        enable_tradovate_controls(&mut app);
        app.screen = Screen::Analytics;
        let dispatched = crate::broker::ReplaySignalDiagnostic {
            bar_timestamp_ns: 1_735_689_600_000_000_000,
            bar_open: 100.0,
            bar_high: 102.0,
            bar_low: 99.0,
            bar_close: 101.0,
            bar_index: Some(12),
            bar_count: 20,
            strategy: "ema_cross".to_string(),
            execution_path: "guarded".to_string(),
            signal_timing: "closed bar".to_string(),
            signal_delay_bars: 1,
            signal: "Buy".to_string(),
            raw_signal: "buy".to_string(),
            effective_signal: "buy".to_string(),
            raw_buy_signal: true,
            raw_sell_signal: false,
            effective_buy_signal: true,
            effective_sell_signal: false,
            current_position_qty: 0,
            effective_position_qty: 0,
            target_qty: Some(1),
            decision: "dispatching".to_string(),
            gate_reason: "target delta passed all execution gates".to_string(),
            order_action: Some("Buy".to_string()),
            order_qty: Some(1),
            indicator_name: "EMA".to_string(),
            previous_fast_indicator: Some(99.0),
            previous_slow_indicator: Some(100.0),
            fast_indicator: Some(101.0),
            slow_indicator: Some(100.5),
            auxiliary_name: None,
            auxiliary_value: None,
            hold_reason: None,
            strategy_detail: "Signal: Buy | fast crossed slow".to_string(),
            fingerprint: Some(123),
        };
        let mut no_op = dispatched.clone();
        no_op.decision = "target_already_actual".to_string();
        no_op.gate_reason = "target position already matches actual position".to_string();
        no_op.order_action = None;
        no_op.order_qty = None;
        app.replay_analytics.signals = vec![dispatched, no_op];
        app.replay_analytics.focus = AnalyticsFocus::Signals;

        let detail = rendered_text(app.analytics_selected_signal_lines()).join(" | ");
        assert!(detail.contains("Bar 12"));
        assert!(detail.contains("ema_cross"));
        assert!(detail.contains("Decision: dispatching"));
        assert!(detail.contains("Buy 1"));

        let mut terminal = Terminal::new(TestBackend::new(120, 40)).expect("terminal");
        terminal.draw(|frame| app.draw(frame)).expect("draw");
        let rows = rendered_terminal_rows(&terminal, 120).join("\n");
        assert!(rows.contains("Signal Diagnostics"));
        assert!(rows.contains("dispatching"));

        app.handle_analytics_key(key(KeyCode::Char('g')));
        assert_eq!(app.analytics_signal_filter_label(), "orders");
        assert!(
            app.analytics_signal_table_rows().len() == 1,
            "order filter should retain only the dispatched row"
        );
        app.handle_analytics_key(key(KeyCode::Char('g')));
        assert_eq!(app.analytics_signal_filter_label(), "blocked/gated");
        assert!(app.analytics_signal_table_rows().is_empty());
        assert!(
            rendered_text(app.analytics_selected_signal_lines())
                .join(" | ")
                .contains("No signal row matches")
        );
        app.handle_analytics_key(key(KeyCode::Char('g')));
        assert_eq!(app.analytics_signal_filter_label(), "non-hold signals");
        assert_eq!(app.analytics_signal_table_rows().len(), 2);
    }

    #[test]
    fn analytics_sweep_tab_renders_candidates_and_controls() {
        let mut app = App::new(AppConfig::default());
        enable_tradovate_controls(&mut app);
        let ranking = ReplaySweepRankingDocument {
            schema_version: 1,
            sweep_id: "mes-ema-grid".to_string(),
            name: "MES EMA grid".to_string(),
            generated_at_utc: chrono::Utc::now(),
            source_summary: PathBuf::from("sweep-summary.json"),
            metric: ReplaySweepRankingMetric::Robustness,
            fee_scenario: "active".to_string(),
            options: ReplaySweepRankingOptions::default(),
            total_completed_candidates: 1,
            filtered_candidates: 1,
            rows: vec![ReplaySweepRankingRow {
                rank: 1,
                run_id: "run-000001".to_string(),
                run_index: 0,
                fee_scenario: "primary".to_string(),
                parameter_values: BTreeMap::from([("fast".to_string(), json!(5))]),
                overrides: BTreeMap::from([("fast".to_string(), json!(5))]),
                metric_value: Some(4.5),
                robustness_score: Some(4.5),
                neighborhood_count: 2,
                neighborhood_median_quality: Some(4.0),
                gross_pnl: Some(100.0),
                net_pnl: Some(96.0),
                fees: Some(4.0),
                ending_equity: Some(10_096.0),
                return_on_initial_capital_pct: Some(0.96),
                required_starting_capital: Some(2_000.0),
                return_on_required_account_size_pct: Some(4.8),
                max_drawdown: Some(10.0),
                max_drawdown_pct: Some(0.1),
                profit_factor: Some(1.5),
                closed_trade_count: Some(12),
                win_rate_pct: Some(58.0),
                average_trade: Some(8.0),
                average_giveback: Some(2.0),
                median_giveback: Some(1.5),
                largest_giveback: Some(5.0),
                average_mfe_capture_ratio: Some(0.8),
                long_quantity: Some(7.0),
                short_quantity: Some(5.0),
            }],
            warnings: Vec::new(),
        };
        app.replay_analytics.sweep_entries = vec![ReplaySweepRankingEntry {
            path: PathBuf::from("runs/mes-ema-grid/sweep-ranking.json"),
            document: ranking.clone(),
        }];
        app.replay_analytics.sweep_ranking = Some(ranking);
        app.screen = Screen::Analytics;
        app.replay_analytics.focus = AnalyticsFocus::Sweeps;

        let mut terminal = Terminal::new(TestBackend::new(140, 45)).expect("terminal");
        terminal.draw(|frame| app.draw(frame)).expect("draw");
        let rows = rendered_terminal_rows(&terminal, 140).join("\n");
        assert!(rows.contains("Sweep Rankings"));
        assert!(rows.contains("Sweep Candidates"));
        assert!(rows.contains("run-000001"));
        assert!(
            rendered_text(app.analytics_selected_sweep_lines())
                .join(" | ")
                .contains("Overrides")
        );

        app.handle_analytics_key(key(KeyCode::Char('m')));
        assert_eq!(
            app.replay_analytics.sweep_metric,
            ReplaySweepRankingMetric::NetPnl
        );
        app.handle_analytics_key(key(KeyCode::Char('d')));
        assert_eq!(app.replay_analytics.sweep_max_drawdown_pct, Some(5.0));
        app.handle_analytics_key(key(KeyCode::Char('t')));
        assert_eq!(app.replay_analytics.sweep_min_closed_trades, 1);
        app.handle_analytics_key(key(KeyCode::Char('j')));
        assert_eq!(app.replay_analytics.selected_sweep_row, 0);
    }
}
