use super::super::*;

impl App {
    pub(in crate::app) fn connection_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(format!("Broker: {}", self.selected_broker.label())),
            styled_line(
                format!("Env: {}", self.form.env.label()),
                self.focus == Focus::Env,
            ),
            styled_line(
                format!("Auth Mode: {}", self.form.auth_mode.label()),
                self.focus == Focus::AuthMode,
            ),
            styled_line(
                format!("Log Mode: {}", self.form.log_mode.label()),
                self.focus == Focus::LogMode,
            ),
            Line::from(""),
            Line::from("Token Sources"),
            styled_line(
                format!("Token Path: {}", self.form.token_path),
                self.focus == Focus::TokenPath,
            ),
            styled_line(
                format!(
                    "Token Override: {}",
                    display_token_override(
                        self.focus == Focus::TokenOverride,
                        &self.form.token_override,
                    )
                ),
                self.focus == Focus::TokenOverride,
            ),
            Line::from(""),
            Line::from("Credential Fallback"),
            styled_line(
                format!("Username: {}", self.form.username),
                self.focus == Focus::Username,
            ),
            styled_line(
                format!("Password: {}", mask(&self.form.password)),
                self.focus == Focus::Password,
            ),
            Line::from(""),
        ];

        match self.selected_broker {
            BrokerKind::Ironbeam => {
                lines.push(styled_line(
                    format!("API Key: {}", mask(&self.form.api_key)),
                    self.focus == Focus::ApiKey,
                ));
                lines.push(Line::from(""));
            }
            BrokerKind::Tradovate => {
                lines.push(styled_line(
                    format!("App ID: {}", self.form.app_id),
                    self.focus == Focus::AppId,
                ));
                lines.push(styled_line(
                    format!("App Version: {}", self.form.app_version),
                    self.focus == Focus::AppVersion,
                ));
                lines.push(styled_line(
                    format!("CID: {}", self.form.cid),
                    self.focus == Focus::Cid,
                ));
                lines.push(styled_line(
                    format!("Secret: {}", mask(&self.form.secret)),
                    self.focus == Focus::Secret,
                ));
                lines.push(Line::from(""));
            }
        }

        lines.push(styled_line(
            "[Enter] Connect / Refresh Session".to_string(),
            self.focus == Focus::Connect,
        ));
        lines.push(styled_line(
            if self.broker_supports_replay() {
                "[Enter] Replay Mode (local file, skips login)".to_string()
            } else {
                "[Enter] Replay Mode unavailable for this broker/build".to_string()
            },
            self.focus == Focus::ReplayMode,
        ));
        lines
    }

    pub(in crate::app) fn login_notes_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from("1. Token Override is used first when non-empty."),
            Line::from("2. Token File mode reads token_path, then session cache."),
            Line::from(match self.selected_broker {
                BrokerKind::Tradovate => {
                    "3. Credentials mode requests a fresh Tradovate access token."
                }
                BrokerKind::Ironbeam => {
                    "3. Credentials mode requests a fresh Ironbeam bearer token using username/password and optional api_key."
                }
            }),
            Line::from("4. Debug log mode adds submit/seen/ack/fill lifecycle lines."),
            Line::from(""),
            Line::from("Use Up/Down to move between fields."),
            Line::from("Use Left/Right on Env, Auth Mode, or Log Mode."),
            Line::from("Paste a token directly into Token Override when needed."),
            Line::from(if self.broker_supports_replay() {
                "Replay Mode loads the local tick file and starts on 1 Range bars by default."
            } else {
                "Replay Mode is only available on Tradovate builds with `--features replay`."
            }),
        ];

        if self.selected_broker == BrokerKind::Ironbeam {
            lines.push(Line::from(
                "Ironbeam currently uses 1-minute bars on the selection screen.",
            ));
        }

        lines
    }

    pub(in crate::app) fn login_status_lines(&self) -> Vec<Line<'static>> {
        let (rest_url, user_ws, market_ws) = match (self.selected_broker, self.form.env) {
            (BrokerKind::Tradovate, env) => (
                env.rest_url().to_string(),
                Some(env.user_ws_url().to_string()),
                Some(env.market_ws_url().to_string()),
            ),
            (BrokerKind::Ironbeam, TradingEnvironment::Sim) => (
                "https://demo.ironbeamapi.com/v2".to_string(),
                Some("wss://demo.ironbeamapi.com/v2/stream/{streamId}?token=...".to_string()),
                None,
            ),
            (BrokerKind::Ironbeam, TradingEnvironment::Live) => (
                "https://live.ironbeamapi.com/v2".to_string(),
                Some("wss://live.ironbeamapi.com/v2/stream/{streamId}?token=...".to_string()),
                None,
            ),
        };

        vec![
            Line::from(format!("Current status: {}", self.status)),
            Line::from(format!("Broker: {}", self.selected_broker.label())),
            Line::from(format!("Session Mode: {}", self.session_kind.label())),
            Line::from(format!("Environment REST: {rest_url}")),
            Line::from(format!("Log Mode: {}", self.form.log_mode.label())),
            Line::from(match user_ws {
                Some(url) => format!("User WebSocket: {url}"),
                None => "User WebSocket: n/a".to_string(),
            }),
            Line::from(match market_ws {
                Some(url) => format!("Market WebSocket: {url}"),
                None => "Market WebSocket: embedded stream".to_string(),
            }),
            Line::from(""),
            Line::from(format!("Accounts loaded: {}", self.accounts.len())),
            Line::from(match &self.market.contract_name {
                Some(name) => format!("Last contract: {name}"),
                None => "Last contract: none".to_string(),
            }),
        ]
    }
}
