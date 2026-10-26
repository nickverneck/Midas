use super::*;

mod analytics;
mod broker;
mod chrome;
mod dashboard;
mod engine;
mod login;
mod replay;
mod selection;
mod session_stats;
mod strategy;

#[cfg(test)]
mod tests;

impl App {
    pub fn draw(&self, frame: &mut Frame<'_>) {
        let compact_analytics = self.screen == Screen::Analytics && frame.area().width < 110;
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(if compact_analytics { 6 } else { 4 }),
                Constraint::Min(if compact_analytics { 0 } else { 20 }),
                Constraint::Length(if compact_analytics { 4 } else { 8 }),
            ])
            .split(frame.area());

        self.render_header(frame, layout[0]);
        match self.screen {
            Screen::EngineSelect => self.render_engine_select_screen(frame, layout[1]),
            Screen::BrokerSelect => self.render_broker_select_screen(frame, layout[1]),
            Screen::Login => self.render_login_screen(frame, layout[1]),
            Screen::Replay => self.render_replay_screen(frame, layout[1]),
            Screen::Strategy => self.render_strategy_screen(frame, layout[1]),
            Screen::Selection => self.render_selection_screen(frame, layout[1]),
            Screen::Dashboard => self.render_dashboard(frame, layout[1]),
            Screen::Stats => self.render_session_stats_screen(frame, layout[1]),
            Screen::Analytics => self.render_analytics_screen(frame, layout[1]),
        }
        self.render_logs(frame, layout[2]);
    }
}
