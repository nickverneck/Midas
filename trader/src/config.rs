use crate::broker::{BrokerKind, default_broker, supports_broker};
use anyhow::{Context, Result, bail};
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradingEnvironment {
    Sim,
    Live,
}

impl TradingEnvironment {
    pub fn rest_url(self) -> &'static str {
        match self {
            Self::Sim => "https://demo.tradovateapi.com/v1",
            Self::Live => "https://live.tradovateapi.com/v1",
        }
    }

    pub fn user_ws_url(self) -> &'static str {
        match self {
            Self::Sim => "wss://demo.tradovateapi.com/v1/websocket",
            Self::Live => "wss://live.tradovateapi.com/v1/websocket",
        }
    }

    pub fn market_ws_url(self) -> &'static str {
        "wss://md.tradovateapi.com/v1/websocket"
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sim => "Simulation",
            Self::Live => "Live",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Sim => Self::Live,
            Self::Live => Self::Sim,
        }
    }
}

impl Default for TradingEnvironment {
    fn default() -> Self {
        Self::Sim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    TokenFile,
    Credentials,
}

impl AuthMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::TokenFile => "Token File",
            Self::Credentials => "Credentials",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::TokenFile => Self::Credentials,
            Self::Credentials => Self::TokenFile,
        }
    }
}

impl Default for AuthMode {
    fn default() -> Self {
        Self::TokenFile
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogMode {
    Default,
    Debug,
}

impl LogMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::Debug => "Debug",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Default => Self::Debug,
            Self::Debug => Self::Default,
        }
    }
}

impl Default for LogMode {
    fn default() -> Self {
        Self::Default
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub broker: BrokerKind,
    pub env: TradingEnvironment,
    pub auth_mode: AuthMode,
    pub log_mode: LogMode,
    pub session_stats_enabled: bool,
    pub token_override: String,
    pub username: String,
    pub password: String,
    pub api_key: String,
    pub app_id: String,
    pub app_version: String,
    pub cid: String,
    pub secret: String,
    pub custom_tag50: String,
    pub token_path: PathBuf,
    pub session_cache_path: PathBuf,
    pub history_bars: usize,
    pub heartbeat_ms: u64,
    pub contract_suggest_limit: usize,
    pub time_in_force: String,
    pub order_qty: i32,
    pub autoconnect: bool,
    pub replay_file_path: PathBuf,
    pub replay_bar_interval_ms: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            broker: default_broker(),
            env: TradingEnvironment::Sim,
            auth_mode: AuthMode::TokenFile,
            log_mode: LogMode::Default,
            session_stats_enabled: true,
            token_override: String::new(),
            username: String::new(),
            password: String::new(),
            api_key: String::new(),
            app_id: "Trader".to_string(),
            app_version: "0.1.0".to_string(),
            cid: String::new(),
            secret: String::new(),
            custom_tag50: String::new(),
            token_path: PathBuf::from(".auth/bearer-token.json"),
            session_cache_path: PathBuf::from(".auth/trader-session.json"),
            history_bars: 500,
            heartbeat_ms: 2500,
            contract_suggest_limit: 12,
            time_in_force: "Day".to_string(),
            order_qty: 1,
            autoconnect: false,
            replay_file_path: PathBuf::from("trader/market replay/ES 06-26.Last.txt"),
            replay_bar_interval_ms: 5,
        }
    }
}

impl AppConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        dotenv().ok();

        let mut cfg = Self::default();
        let config_path = path
            .map(PathBuf::from)
            .or_else(|| env_string_any(&["TRADER_CONFIG", "MIDAS_TUI_CONFIG"]).map(PathBuf::from));

        if let Some(path) = config_path {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("read config {}", path.display()))?;
            cfg = toml::from_str(&raw).with_context(|| format!("parse TOML {}", path.display()))?;
        }

        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn apply_env_overrides(&mut self) -> Result<()> {
        if let Some(raw) = env_string_any(&["TRADER_BROKER"]) {
            self.broker = parse_broker(&raw)?;
        }
        if let Some(raw) = env_string_any(&["TRADER_ENV", "MIDAS_TUI_ENV"]) {
            self.env = parse_env(&raw)?;
        }
        if let Some(raw) = env_string_any(&["TRADER_AUTH_MODE", "MIDAS_TUI_AUTH_MODE"]) {
            self.auth_mode = parse_auth_mode(&raw)?;
        }
        if let Some(raw) = env_string_any(&["TRADER_LOG_MODE", "MIDAS_TUI_LOG_MODE"]) {
            self.log_mode = parse_log_mode(&raw)?;
        }
        if let Some(raw) = env_bool_any(&[
            "TRADER_SESSION_STATS_ENABLED",
            "MIDAS_TUI_SESSION_STATS_ENABLED",
        ])? {
            self.session_stats_enabled = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_TOKEN_OVERRIDE", "MIDAS_TUI_TOKEN_OVERRIDE"]) {
            self.token_override = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_USERNAME", "MIDAS_TUI_USERNAME"]) {
            self.username = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_PASSWORD", "MIDAS_TUI_PASSWORD"]) {
            self.password = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_API_KEY"]) {
            self.api_key = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_APP_ID", "MIDAS_TUI_APP_ID"]) {
            self.app_id = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_APP_VERSION", "MIDAS_TUI_APP_VERSION"]) {
            self.app_version = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_CID", "MIDAS_TUI_CID"]) {
            self.cid = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_SECRET", "MIDAS_TUI_SECRET"]) {
            self.secret = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_CUSTOM_TAG50", "MIDAS_TUI_CUSTOM_TAG50"]) {
            self.custom_tag50 = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_TOKEN_PATH", "MIDAS_TUI_TOKEN_PATH"]) {
            self.token_path = PathBuf::from(raw);
        }
        if let Some(raw) =
            env_string_any(&["TRADER_SESSION_CACHE_PATH", "MIDAS_TUI_SESSION_CACHE_PATH"])
        {
            self.session_cache_path = PathBuf::from(raw);
        }
        if let Some(raw) =
            env_parse_any::<usize>(&["TRADER_HISTORY_BARS", "MIDAS_TUI_HISTORY_BARS"])?
        {
            self.history_bars = raw;
        }
        if let Some(raw) = env_parse_any::<u64>(&["TRADER_HEARTBEAT_MS", "MIDAS_TUI_HEARTBEAT_MS"])?
        {
            self.heartbeat_ms = raw;
        }
        if let Some(raw) = env_parse_any::<usize>(&[
            "TRADER_CONTRACT_SUGGEST_LIMIT",
            "MIDAS_TUI_CONTRACT_SUGGEST_LIMIT",
        ])? {
            self.contract_suggest_limit = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_TIME_IN_FORCE", "MIDAS_TUI_TIME_IN_FORCE"]) {
            self.time_in_force = raw;
        }
        if let Some(raw) = env_parse_any::<i32>(&["TRADER_ORDER_QTY", "MIDAS_TUI_ORDER_QTY"])? {
            self.order_qty = raw;
        }
        if let Some(raw) = env_bool_any(&["TRADER_AUTOCONNECT", "MIDAS_TUI_AUTOCONNECT"])? {
            self.autoconnect = raw;
        }
        if let Some(raw) =
            env_string_any(&["TRADER_REPLAY_FILE_PATH", "MIDAS_TUI_REPLAY_FILE_PATH"])
        {
            self.replay_file_path = PathBuf::from(raw);
        }
        if let Some(raw) = env_parse_any::<u64>(&[
            "TRADER_REPLAY_BAR_INTERVAL_MS",
            "MIDAS_TUI_REPLAY_BAR_INTERVAL_MS",
        ])? {
            self.replay_bar_interval_ms = raw;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if !supports_broker(self.broker) {
            bail!(
                "{} support is not enabled in this build",
                self.broker.label()
            );
        }
        if self.history_bars == 0 {
            bail!("history_bars must be > 0");
        }
        if self.heartbeat_ms == 0 {
            bail!("heartbeat_ms must be > 0");
        }
        if self.contract_suggest_limit == 0 {
            bail!("contract_suggest_limit must be > 0");
        }
        if self.order_qty <= 0 {
            bail!("order_qty must be > 0");
        }
        if self.replay_bar_interval_ms == 0 {
            bail!("replay_bar_interval_ms must be > 0");
        }
        if self.token_override.trim().is_empty() && matches!(self.auth_mode, AuthMode::Credentials)
        {
            if self.username.trim().is_empty() {
                bail!("username is required in credentials auth mode");
            }
            if self.password.trim().is_empty() {
                bail!("password is required in credentials auth mode");
            }
        }
        Ok(())
    }
}

fn parse_broker(raw: &str) -> Result<BrokerKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tradovate" | "ninjatrader" | "ninja_trader" => Ok(BrokerKind::Tradovate),
        "ironbeam" | "iron_beam" => Ok(BrokerKind::Ironbeam),
        other => bail!("invalid broker `{other}`"),
    }
}

fn parse_env(raw: &str) -> Result<TradingEnvironment> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "sim" | "demo" => Ok(TradingEnvironment::Sim),
        "live" => Ok(TradingEnvironment::Live),
        other => bail!("invalid env `{other}`"),
    }
}

fn parse_auth_mode(raw: &str) -> Result<AuthMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "token_file" | "token" => Ok(AuthMode::TokenFile),
        "credentials" | "creds" => Ok(AuthMode::Credentials),
        other => bail!("invalid auth mode `{other}`"),
    }
}

fn parse_log_mode(raw: &str) -> Result<LogMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "default" | "normal" => Ok(LogMode::Default),
        "debug" | "verbose" => Ok(LogMode::Debug),
        other => bail!("invalid log mode `{other}`"),
    }
}

fn env_string_any(keys: &[&str]) -> Option<String> {
    for key in keys {
        match env::var(key) {
            Ok(value) => {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            Err(_) => {}
        }
    }
    None
}

fn env_bool_any(keys: &[&str]) -> Result<Option<bool>> {
    let Some(raw) = env_string_any(keys) else {
        return Ok(None);
    };
    let parsed = match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => bail!("invalid boolean value `{other}` for {}", keys.join(" / ")),
    };
    Ok(Some(parsed))
}

fn env_parse_any<T>(keys: &[&str]) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(raw) = env_string_any(keys) else {
        return Ok(None);
    };
    let parsed = raw
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {} value `{raw}`: {err}", keys.join(" / ")))?;
    Ok(Some(parsed))
}
