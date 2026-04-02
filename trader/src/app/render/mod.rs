use super::*;

mod broker;
mod chrome;
mod dashboard;
mod login;
mod selection;
mod strategy;

#[cfg(test)]
mod tests;

impl App {
    pub fn draw(&self, frame: &mut Frame<'_>) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(20),
                Constraint::Length(8),
            ])
            .split(frame.area());

        self.render_header(frame, layout[0]);
        match self.screen {
            Screen::BrokerSelect => self.render_broker_select_screen(frame, layout[1]),
            Screen::Login => self.render_login_screen(frame, layout[1]),
            Screen::Strategy => self.render_strategy_screen(frame, layout[1]),
            Screen::Selection => self.render_selection_screen(frame, layout[1]),
            Screen::Dashboard => self.render_dashboard(frame, layout[1]),
        }
        self.render_logs(frame, layout[2]);
    }
}
