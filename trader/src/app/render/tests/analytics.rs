#[cfg(feature = "replay")]
mod replay_analytics_tests {
    use super::super::support::*;

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
}
