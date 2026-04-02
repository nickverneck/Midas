use super::super::*;

impl App {
    pub(in crate::app) fn selection_summary_lines(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!(
                "Strategy Runtime: {}",
                self.strategy_runtime_summary()
            )),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(format!("Bar Type: {}", self.bar_type.label())),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(format!("Contract results: {}", self.contract_results.len())),
            Line::from(format!(
                "Last subscribed contract: {}",
                self.market
                    .contract_name
                    .clone()
                    .unwrap_or_else(|| "none".to_string())
            )),
            Line::from("F3 opens the monitoring dashboard."),
        ]
    }

    pub(in crate::app) fn selection_preview_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Status: {}", self.status)),
            Line::from(format!("Strategy: {}", self.strategy.summary_label())),
            Line::from(format!("Bar Type: {}", self.bar_type.label())),
            Line::from(format!("Query: {}", self.instrument_query)),
            Line::from(match self.accounts.get(self.selected_account) {
                Some(account) => format!("Selected account: {}", account.name),
                None => "Selected account: none".to_string(),
            }),
            Line::from(match self.contract_results.get(self.selected_contract) {
                Some(contract) => format!("Selected contract: {}", contract.name),
                None => "Selected contract: none".to_string(),
            }),
        ];

        if let Some(snapshot) = self.selected_snapshot() {
            lines.push(Line::from(format!(
                "Account net liq: {}",
                format_money(snapshot.net_liq.or(snapshot.balance))
            )));
            lines.push(Line::from(format!(
                "Selected contract qty: {}",
                format_quantity(snapshot.market_position_qty)
            )));
        }

        lines
    }
}
