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
    pub env: TradingEnvironment,
    pub auth_mode: AuthMode,
    pub log_mode: LogMode,
    pub token_override: String,
    pub username: String,
    pub password: String,
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
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            env: TradingEnvironment::Sim,
            auth_mode: AuthMode::TokenFile,
            log_mode: LogMode::Default,
            token_override: String::new(),
            username: String::new(),
            password: String::new(),
            app_id: "Midas TUI".to_string(),
            app_version: "0.1.0".to_string(),
            cid: String::new(),
            secret: String::new(),
            custom_tag50: String::new(),
            token_path: PathBuf::from(".auth/bearer-token.json"),
            session_cache_path: PathBuf::from(".auth/ninjatrader-tui-session.json"),
            history_bars: 500,
            heartbeat_ms: 2500,
            contract_suggest_limit: 12,
            time_in_force: "Day".to_string(),
            order_qty: 1,
            autoconnect: false,
        }
    }
}

impl AppConfig {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        dotenv().ok();

        let mut cfg = Self::default();
        let config_path = path
            .map(PathBuf::from)
            .or_else(|| env_string("MIDAS_TUI_CONFIG").map(PathBuf::from));

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
        if let Some(raw) = env_string("MIDAS_TUI_ENV") {
            self.env = parse_env(&raw)?;
        }
        if let Some(raw) = env_string("MIDAS_TUI_AUTH_MODE") {
            self.auth_mode = parse_auth_mode(&raw)?;
        }
        if let Some(raw) = env_string("MIDAS_TUI_LOG_MODE") {
            self.log_mode = parse_log_mode(&raw)?;
        }
        if let Some(raw) = env_string("MIDAS_TUI_TOKEN_OVERRIDE") {
            self.token_override = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_USERNAME") {
            self.username = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_PASSWORD") {
            self.password = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_APP_ID") {
            self.app_id = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_APP_VERSION") {
            self.app_version = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_CID") {
            self.cid = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_SECRET") {
            self.secret = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_CUSTOM_TAG50") {
            self.custom_tag50 = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_TOKEN_PATH") {
            self.token_path = PathBuf::from(raw);
        }
        if let Some(raw) = env_string("MIDAS_TUI_SESSION_CACHE_PATH") {
            self.session_cache_path = PathBuf::from(raw);
        }
        if let Some(raw) = env_parse::<usize>("MIDAS_TUI_HISTORY_BARS")? {
            self.history_bars = raw;
        }
        if let Some(raw) = env_parse::<u64>("MIDAS_TUI_HEARTBEAT_MS")? {
            self.heartbeat_ms = raw;
        }
        if let Some(raw) = env_parse::<usize>("MIDAS_TUI_CONTRACT_SUGGEST_LIMIT")? {
            self.contract_suggest_limit = raw;
        }
        if let Some(raw) = env_string("MIDAS_TUI_TIME_IN_FORCE") {
            self.time_in_force = raw;
        }
        if let Some(raw) = env_parse::<i32>("MIDAS_TUI_ORDER_QTY")? {
            self.order_qty = raw;
        }
        if let Some(raw) = env_bool("MIDAS_TUI_AUTOCONNECT")? {
            self.autoconnect = raw;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
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

fn env_string(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Err(_) => None,
    }
}

fn env_bool(key: &str) -> Result<Option<bool>> {
    let Some(raw) = env_string(key) else {
        return Ok(None);
    };
    let parsed = match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => bail!("invalid boolean value `{other}` for {key}"),
    };
    Ok(Some(parsed))
}

fn env_parse<T>(key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(raw) = env_string(key) else {
        return Ok(None);
    };
    let parsed = raw
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {key} value `{raw}`: {err}"))?;
    Ok(Some(parsed))
}
