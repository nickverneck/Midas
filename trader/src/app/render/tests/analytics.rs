#[cfg(feature = "replay")]
mod replay_analytics_tests {
    use super::super::support::*;
    use crate::tradovate::replay::{
        ReplaySweepRankingDocument, ReplaySweepRankingEntry, ReplaySweepRankingMetric,
        ReplaySweepRankingOptions, ReplaySweepRankingRow,
    };
    use chrono::TimeZone;
    use std::collections::BTreeMap;

    fn enum_value<T: serde::Serialize>(value: T) -> serde_json::Value {
        serde_json::to_value(value).expect("serialize enum")
    }

    fn write_minimal_result(root: &std::path::Path, run_id: &str, completed_minute: u32) {
        let completed = chrono::Utc
            .with_ymd_and_hms(2026, 10, 10, 14, completed_minute, 0)
            .unwrap();
        let mut metadata = json!({
            "app_version": "test",
            "package_version": "test",
            "git_commit": null,
            "broker": "tradovate",
            "environment": enum_value(crate::config::TradingEnvironment::Sim),
            "run_mode": "cached_replay",
            "strategy": serde_json::to_value(crate::strategy::ExecutionStrategyConfig::default()).unwrap(),
            "signal_source": "cached_bars",
            "fill_model": enum_value(crate::broker::ReplayFillModel::RawBarOpen),
            "fill_price_sources": [],
            "execution_precisions": [],
            "latency_model": enum_value(crate::broker::ReplayLatencyModel::Fixed),
            "fixed_latency_ms": 0,
            "latency_seed": null,
            "observed_latency_sample_count": 0,
            "bar_protection_policy": enum_value(crate::broker::ReplayBarProtectionPolicy::Conservative),
        })
        .as_object()
        .expect("metadata object")
        .clone();
        metadata.extend(
            json!({
                "fee_model": "primary",
                "initial_capital": 10000.0,
                "account_currency": "USD",
                "margin_model": "fixed",
                "account_id": 1,
                "account_name": "test",
                "contract_id": 1,
                "contract_name": "ESZ6",
                "tick_size": 0.25,
                "value_per_point": 50.0,
                "bar_type": serde_json::to_value(crate::broker::BarType::minute(1)).unwrap(),
                "candle_mode": enum_value(crate::broker::CandleMode::Standard),
                "timezone": "UTC",
                "replay_file_path": "",
                "replay_dom_file_path": null,
                "replay_cache_dir": "",
                "dataset_view": null,
                "evaluation_start_utc": completed,
                "evaluation_end_utc": completed,
                "warmup_start_utc": null,
                "market_first_timestamp_ns": null,
                "market_last_timestamp_ns": null,
                "source_kind": "cached_server_bars",
                "source_label": "Cached Replay · ESZ6",
                "source_record_kind": "bars",
                "source_record_count": 12,
                "source_bar_count": 12,
                "warmup_rows": 4,
                "evaluation_rows_total": 8,
                "evaluation_rows_processed": 8
            })
            .as_object()
            .expect("metadata continuation")
            .clone(),
        );
        let result = json!({
            "schema_version": 3,
            "result_type": "replay_result",
            "run_id": run_id,
            "status": "completed",
            "error": null,
            "created_at_utc": completed,
            "started_at_utc": completed,
            "completed_at_utc": completed,
            "metadata": metadata,
            "summary": {
                "initial_capital": 10000.0,
                "ending_equity": 10000.0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "fees": 0.0,
                "return_on_initial_capital_pct": null,
                "fill_count": 0,
                "trade_count": 0,
                "closed_trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate_pct": null,
                "profit_factor": null,
                "max_drawdown": 0.0,
                "max_drawdown_pct": null,
                "max_open_position": 0.0,
                "precision": [],
                "exit_reason_counts": {},
                "evaluation_rows_processed": 8
            },
            "artifacts": {
                "result_json": "result.json",
                "trades_csv": "trades.csv",
                "fills_csv": "fills.csv",
                "equity_csv": "equity.csv"
            },
            "ledger": serde_json::to_value(crate::broker::ReplayExecutionLedgerSnapshot::default()).unwrap()
        });
        let run_dir = root.join(run_id);
        std::fs::create_dir_all(&run_dir).expect("create result directory");
        std::fs::write(
            run_dir.join("result.json"),
            serde_json::to_vec(&result).expect("serialize result fixture"),
        )
        .expect("write result fixture");
    }

    #[test]
    fn f8_opens_saved_analytics_without_an_active_engine() {
        let mut app = App::new(AppConfig::default());
        enable_tradovate_controls(&mut app);
        app.session_mode = EngineCreateMode::Replay;
        let (tx, _rx) = unbounded_channel();

        app.handle_key(key(KeyCode::F(8)), &tx);

        assert_eq!(app.screen, Screen::Analytics);
        assert_eq!(app.analytics_return_screen, Screen::EngineSelect);
    }

    #[test]
    fn opening_analytics_selects_newest_run_and_matching_detail() {
        let root = std::env::temp_dir().join(format!(
            "trader-analytics-latest-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        write_minimal_result(&root, "older-run", 10);
        write_minimal_result(&root, "newest-run", 11);

        let mut config = AppConfig::default();
        config.replay_result_dir = root.clone();
        let mut app = App::new(config);
        enable_tradovate_controls(&mut app);
        app.session_mode = EngineCreateMode::Replay;
        app.replay_analytics.selected_run = 1;
        app.replay_analytics.focus = AnalyticsFocus::Trades;
        let (tx, _rx) = unbounded_channel();

        app.handle_key(key(KeyCode::F(8)), &tx);

        assert_eq!(app.screen, Screen::Analytics);
        assert_eq!(app.replay_analytics.focus, AnalyticsFocus::Runs);
        assert_eq!(app.replay_analytics.selected_run, 0);
        assert_eq!(
            app.replay_analytics
                .selected_entry()
                .map(|entry| entry.document.run_id.as_str()),
            Some("newest-run")
        );
        assert!(
            rendered_text(app.analytics_selected_run_lines())
                .join(" | ")
                .contains("newest-run")
        );

        let mut terminal = Terminal::new(TestBackend::new(140, 45)).expect("terminal");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("draw analytics");
        let rows = rendered_terminal_rows(&terminal, 140).join("\n");
        assert!(rows.contains("> newest-run [ok]"), "{rows}");
        assert!(!rows.contains("> > newest-run [ok]"));
        assert!(rows.contains("Selected Run Detail"));
        assert!(rows.contains("newest-run | ESZ6"));
        assert!(rows.contains("Research Summary"));
        assert!(rows.contains("Equity Curve"));
        assert!(rows.contains("Trade Excursions"));

        std::fs::remove_dir_all(root).expect("remove analytics fixture");
    }

    #[test]
    fn compact_analytics_fits_80_by_24_and_scrolls_selected_run_summary() {
        let root = std::env::temp_dir().join(format!(
            "trader-analytics-compact-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        write_minimal_result(&root, "older-run", 10);
        write_minimal_result(&root, "newest-run", 11);

        let mut config = AppConfig::default();
        config.replay_result_dir = root.clone();
        let mut app = App::new(config);
        enable_tradovate_controls(&mut app);
        app.screen = Screen::Analytics;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");

        terminal
            .draw(|frame| app.draw(frame))
            .expect("draw compact analytics");
        let rows = rendered_terminal_rows(&terminal, 80).join("\n");
        assert!(rows.contains("Saved Replay Runs (2)"), "{rows}");
        assert!(rows.contains("Selected Run Summary"), "{rows}");
        assert!(rows.contains("> newest-run [ok]"), "{rows}");
        assert!(!rows.contains("> > newest-run"), "{rows}");
        assert!(rows.contains("Run: newest-run"), "{rows}");
        assert!(rows.contains("Source: Cached Replay · ESZ6"), "{rows}");
        assert!(rows.contains("Warmup: 4 | Evaluated: 8"), "{rows}");
        assert!(rows.contains("PgUp/PgDn scroll"), "{rows}");

        app.handle_analytics_key(key(KeyCode::PageDown));
        assert_eq!(app.replay_analytics.compact_scroll, 5);
        terminal
            .draw(|frame| app.draw(frame))
            .expect("draw scrolled summary");
        let scrolled_rows = rendered_terminal_rows(&terminal, 80).join("\n");
        assert!(
            !scrolled_rows.contains("Run: newest-run"),
            "{scrolled_rows}"
        );

        for _ in 0..10 {
            app.handle_analytics_key(key(KeyCode::PageDown));
        }
        terminal
            .draw(|frame| app.draw(frame))
            .expect("draw summary end");
        let end_rows = rendered_terminal_rows(&terminal, 80).join("\n");
        assert!(end_rows.contains("Margin: not configured"), "{end_rows}");

        for _ in 0..11 {
            app.handle_analytics_key(key(KeyCode::PageUp));
        }
        assert_eq!(app.replay_analytics.compact_scroll, 0);

        std::fs::remove_dir_all(root).expect("remove analytics fixture");
    }

    #[test]
    fn analytics_q_quits_even_when_stale_replay_text_focus_remains() {
        let mut app = App::new(AppConfig::default());
        app.screen = Screen::Analytics;
        app.focus = Focus::ReplayInitialCapital;
        let (tx, _rx) = unbounded_channel();

        app.handle_key(key(KeyCode::Char('q')), &tx);

        assert!(app.should_quit);
    }

    #[test]
    fn q_keeps_normal_login_text_editing() {
        let mut app = App::new(AppConfig::default());
        app.screen = Screen::Login;
        app.focus = Focus::Username;
        let (tx, _rx) = unbounded_channel();

        app.handle_key(key(KeyCode::Char('q')), &tx);

        assert!(!app.should_quit);
        assert_eq!(app.form.username, "q");
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
