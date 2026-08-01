use crate::broker::{
    BrokerKind, CandleMode, ReplayBarProtectionPolicy, ReplayEngineMode, ReplayFillModel,
    ReplayLatencyConfig, ReplayLatencyModel, default_broker, supports_broker,
};
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

    /// Market Replay uses a dedicated websocket. It replays the same market
    /// data operations (including `md/subscribeDOM`) against the initialized
    /// historical clock.
    pub fn replay_ws_url(self) -> &'static str {
        "wss://replay.tradovateapi.com/v1/websocket"
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
    pub candle_mode: CandleMode,
    pub autoconnect: bool,
    pub replay_file_path: PathBuf,
    /// Optional JSONL full-book snapshots for the Level 2 replay fill model.
    /// Empty/absent means the normal bar/tick/quote replay paths are used.
    pub replay_dom_file_path: Option<PathBuf>,
    pub replay_cache_dir: PathBuf,
    /// Root directory for durable replay result artifacts.
    pub replay_result_dir: PathBuf,
    /// Optional deterministic run id used by headless replay workers.
    /// Interactive replay sessions leave this unset and continue to receive
    /// a timestamp/process-derived id.
    #[serde(default)]
    pub replay_run_id: Option<String>,
    /// Disable wall-clock pacing for headless replay workers.
    #[serde(default)]
    pub replay_headless: bool,
    /// Starting account equity used by replay-only account and margin analytics.
    pub replay_initial_capital: f64,
    /// Currency label for replay account and margin analytics.
    pub replay_account_currency: String,
    /// Label for the replay margin assumption. The first supported calculation is fixed per contract.
    pub replay_margin_model: String,
    /// Fixed margin requirement per open contract. Zero disables automatic margin analytics.
    pub replay_margin_per_contract: f64,
    /// Fixed safety buffer added above replay margin requirements.
    pub replay_safety_buffer: f64,
    /// Percentage safety buffer applied to replay margin requirements.
    pub replay_safety_buffer_percent: f64,
    /// Optional number of selected replay bars to inspect after each exit for
    /// favorable continuation. Zero disables post-exit continuation analytics.
    pub replay_post_exit_continuation_bars: usize,
    /// Persist one structured strategy decision row per replay evaluation.
    /// Disabled by default to keep normal replay artifacts and live execution
    /// lightweight.
    pub replay_signal_diagnostics: bool,
    pub replay_bar_interval_ms: u64,
    pub replay_engine_mode: ReplayEngineMode,
    pub replay_fill_model: ReplayFillModel,
    pub replay_latency_model: ReplayLatencyModel,
    pub replay_fixed_latency_ms: u64,
    pub replay_observed_latency_ms: Vec<u64>,
    pub replay_latency_seed: u64,
    pub replay_bar_protection_policy: ReplayBarProtectionPolicy,
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
            candle_mode: CandleMode::Standard,
            autoconnect: false,
            replay_file_path: PathBuf::from("trader/market replay/ES 06-26.Last.txt"),
            replay_dom_file_path: None,
            replay_cache_dir: default_replay_cache_dir(),
            replay_result_dir: PathBuf::from(".run/replay-results"),
            replay_run_id: None,
            replay_headless: false,
            replay_initial_capital: 100_000.0,
            replay_account_currency: "USD".to_string(),
            replay_margin_model: "fixed_per_contract".to_string(),
            replay_margin_per_contract: 0.0,
            replay_safety_buffer: 0.0,
            replay_safety_buffer_percent: 0.0,
            replay_post_exit_continuation_bars: 0,
            replay_signal_diagnostics: false,
            replay_bar_interval_ms: 5,
            replay_engine_mode: ReplayEngineMode::default(),
            replay_fill_model: ReplayFillModel::RawBarOpen,
            replay_latency_model: ReplayLatencyModel::Fixed,
            replay_fixed_latency_ms: 0,
            replay_observed_latency_ms: Vec::new(),
            replay_latency_seed: 1,
            replay_bar_protection_policy: ReplayBarProtectionPolicy::Conservative,
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
        if let Some(raw) = env_string_any(&["TRADER_CANDLE_MODE", "MIDAS_TUI_CANDLE_MODE"]) {
            self.candle_mode = parse_candle_mode(&raw)?;
        }
        if let Some(raw) = env_bool_any(&["TRADER_AUTOCONNECT", "MIDAS_TUI_AUTOCONNECT"])? {
            self.autoconnect = raw;
        }
        if let Some(raw) =
            env_string_any(&["TRADER_REPLAY_FILE_PATH", "MIDAS_TUI_REPLAY_FILE_PATH"])
        {
            self.replay_file_path = PathBuf::from(raw);
        }
        if let Some(raw) = env_string_any(&[
            "TRADER_REPLAY_DOM_FILE_PATH",
            "MIDAS_TUI_REPLAY_DOM_FILE_PATH",
        ]) {
            self.replay_dom_file_path = (!raw.trim().is_empty()).then(|| PathBuf::from(raw));
        }
        if let Some(raw) = env_string_any(&["TRADER_DATA_CACHE_DIR"]) {
            self.replay_cache_dir = PathBuf::from(raw);
        }
        if let Some(raw) =
            env_string_any(&["TRADER_REPLAY_RESULT_DIR", "MIDAS_TUI_REPLAY_RESULT_DIR"])
        {
            self.replay_result_dir = PathBuf::from(raw);
        }
        if let Some(raw) = env_parse_any::<f64>(&[
            "TRADER_REPLAY_INITIAL_CAPITAL",
            "MIDAS_TUI_REPLAY_INITIAL_CAPITAL",
        ])? {
            self.replay_initial_capital = raw;
        }
        if let Some(raw) = env_string_any(&[
            "TRADER_REPLAY_ACCOUNT_CURRENCY",
            "MIDAS_TUI_REPLAY_ACCOUNT_CURRENCY",
        ]) {
            self.replay_account_currency = raw;
        }
        if let Some(raw) = env_string_any(&[
            "TRADER_REPLAY_MARGIN_MODEL",
            "MIDAS_TUI_REPLAY_MARGIN_MODEL",
        ]) {
            self.replay_margin_model = raw;
        }
        if let Some(raw) = env_parse_any::<f64>(&[
            "TRADER_REPLAY_MARGIN_PER_CONTRACT",
            "MIDAS_TUI_REPLAY_MARGIN_PER_CONTRACT",
        ])? {
            self.replay_margin_per_contract = raw;
        }
        if let Some(raw) = env_parse_any::<f64>(&[
            "TRADER_REPLAY_SAFETY_BUFFER",
            "MIDAS_TUI_REPLAY_SAFETY_BUFFER",
        ])? {
            self.replay_safety_buffer = raw;
        }
        if let Some(raw) = env_parse_any::<f64>(&[
            "TRADER_REPLAY_SAFETY_BUFFER_PERCENT",
            "MIDAS_TUI_REPLAY_SAFETY_BUFFER_PERCENT",
        ])? {
            self.replay_safety_buffer_percent = raw;
        }
        if let Some(raw) = env_parse_any::<usize>(&[
            "TRADER_REPLAY_POST_EXIT_CONTINUATION_BARS",
            "MIDAS_TUI_REPLAY_POST_EXIT_CONTINUATION_BARS",
        ])? {
            self.replay_post_exit_continuation_bars = raw;
        }
        if let Some(raw) = env_bool_any(&[
            "TRADER_REPLAY_SIGNAL_DIAGNOSTICS",
            "MIDAS_TUI_REPLAY_SIGNAL_DIAGNOSTICS",
        ])? {
            self.replay_signal_diagnostics = raw;
        }
        if let Some(raw) = env_parse_any::<u64>(&[
            "TRADER_REPLAY_BAR_INTERVAL_MS",
            "MIDAS_TUI_REPLAY_BAR_INTERVAL_MS",
        ])? {
            self.replay_bar_interval_ms = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_REPLAY_ENGINE_MODE"]) {
            self.replay_engine_mode = parse_replay_engine_mode(&raw)?;
        }
        if let Some(raw) = env_string_any(&["TRADER_REPLAY_FILL_MODEL"]) {
            self.replay_fill_model = parse_replay_fill_model(&raw)?;
        }
        if let Some(raw) = env_string_any(&["TRADER_REPLAY_LATENCY_MODEL"]) {
            self.replay_latency_model = parse_replay_latency_model(&raw)?;
        }
        if let Some(raw) = env_parse_any::<u64>(&["TRADER_REPLAY_FIXED_LATENCY_MS"])? {
            self.replay_fixed_latency_ms = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_REPLAY_OBSERVED_LATENCY_MS"]) {
            self.replay_observed_latency_ms =
                parse_u64_list(&raw, "TRADER_REPLAY_OBSERVED_LATENCY_MS")?;
        }
        if let Some(raw) = env_parse_any::<u64>(&["TRADER_REPLAY_LATENCY_SEED"])? {
            self.replay_latency_seed = raw;
        }
        if let Some(raw) = env_string_any(&["TRADER_REPLAY_BAR_PROTECTION_POLICY"]) {
            self.replay_bar_protection_policy = parse_replay_bar_protection_policy(&raw)?;
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<()> {
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
        if !self.replay_initial_capital.is_finite() || self.replay_initial_capital <= 0.0 {
            bail!("replay_initial_capital must be finite and greater than zero");
        }
        if self.replay_account_currency.trim().is_empty() {
            bail!("replay_account_currency cannot be empty");
        }
        if self
            .replay_run_id
            .as_deref()
            .is_some_and(|run_id| run_id.trim().is_empty())
        {
            bail!("replay_run_id cannot be empty when configured");
        }
        if self.replay_margin_model.trim().is_empty() {
            bail!("replay_margin_model cannot be empty");
        }
        if !self.replay_margin_per_contract.is_finite() || self.replay_margin_per_contract < 0.0 {
            bail!("replay_margin_per_contract must be finite and non-negative");
        }
        if !self.replay_safety_buffer.is_finite() || self.replay_safety_buffer < 0.0 {
            bail!("replay_safety_buffer must be finite and non-negative");
        }
        if !self.replay_safety_buffer_percent.is_finite() || self.replay_safety_buffer_percent < 0.0
        {
            bail!("replay_safety_buffer_percent must be finite and non-negative");
        }
        if self.replay_engine_mode == ReplayEngineMode::Deterministic {
            if self.replay_fill_model == ReplayFillModel::LegacyReferencePrice {
                bail!("deterministic replay cannot use legacy_reference_price fill model")
            }
            match self.replay_latency_model {
                ReplayLatencyModel::IgnoredLegacy => {
                    bail!("deterministic replay cannot use ignored_legacy latency")
                }
                ReplayLatencyModel::Fixed => {}
                ReplayLatencyModel::ObservedMean
                | ReplayLatencyModel::ObservedP95
                | ReplayLatencyModel::ObservedP99
                | ReplayLatencyModel::SeededObserved
                    if self.replay_observed_latency_ms.is_empty() =>
                {
                    bail!(
                        "{} replay latency requires replay_observed_latency_ms samples",
                        self.replay_latency_model.label()
                    )
                }
                _ => {}
            }
        }
        if self.replay_fill_model == ReplayFillModel::Dom
            && self.replay_engine_mode != ReplayEngineMode::Deterministic
        {
            bail!("Level 2 DOM replay requires deterministic replay_engine_mode")
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

    pub fn replay_latency_config(&self) -> ReplayLatencyConfig {
        ReplayLatencyConfig {
            model: self.replay_latency_model,
            fixed_latency_ms: self.replay_fixed_latency_ms,
            observed_samples_ms: self.replay_observed_latency_ms.clone(),
            seed: self.replay_latency_seed,
        }
    }
}

fn parse_replay_engine_mode(raw: &str) -> Result<ReplayEngineMode> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "legacy" | "compatibility" | "legacy_compatibility" => Ok(ReplayEngineMode::Legacy),
        "deterministic" | "virtual_time" | "deterministic_virtual_time" => {
            Ok(ReplayEngineMode::Deterministic)
        }
        other => bail!("invalid replay engine mode `{other}`; expected legacy or deterministic"),
    }
}

fn parse_replay_fill_model(raw: &str) -> Result<ReplayFillModel> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "legacy" | "legacy_reference_price" | "reference_price" => {
            Ok(ReplayFillModel::LegacyReferencePrice)
        }
        "raw_bar_open" | "next_bar_open" | "bar_open" => Ok(ReplayFillModel::RawBarOpen),
        "tick_bid_ask" | "ticks" | "tick" | "level1" | "level_1" => Ok(ReplayFillModel::TickBidAsk),
        "dom" | "level2" | "level_2" | "depth" | "l2" => Ok(ReplayFillModel::Dom),
        other => bail!(
            "invalid replay fill model `{other}`; expected legacy_reference_price, raw_bar_open, tick_bid_ask, or dom"
        ),
    }
}

fn parse_replay_latency_model(raw: &str) -> Result<ReplayLatencyModel> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "fixed" | "zero" => Ok(ReplayLatencyModel::Fixed),
        "observed_mean" | "mean" | "average" => Ok(ReplayLatencyModel::ObservedMean),
        "observed_p95" | "p95" => Ok(ReplayLatencyModel::ObservedP95),
        "observed_p99" | "p99" => Ok(ReplayLatencyModel::ObservedP99),
        "seeded_observed" | "seeded" | "distribution" => Ok(ReplayLatencyModel::SeededObserved),
        other => bail!(
            "invalid replay latency model `{other}`; expected fixed, observed_mean, observed_p95, observed_p99, or seeded_observed"
        ),
    }
}

fn parse_replay_bar_protection_policy(raw: &str) -> Result<ReplayBarProtectionPolicy> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "conservative" | "stop_first" => Ok(ReplayBarProtectionPolicy::Conservative),
        "optimistic" | "target_first" => Ok(ReplayBarProtectionPolicy::Optimistic),
        "nearest_open" | "nearest" => Ok(ReplayBarProtectionPolicy::NearestOpen),
        other => bail!(
            "invalid replay bar protection policy `{other}`; expected conservative, optimistic, or nearest_open"
        ),
    }
}

fn parse_u64_list(raw: &str, key: &str) -> Result<Vec<u64>> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("invalid {key} sample `{value}`"))
        })
        .collect()
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

fn parse_candle_mode(raw: &str) -> Result<CandleMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "standard" | "ohlc" | "regular" => Ok(CandleMode::Standard),
        "heikin_ashi" | "heikin-ashi" | "heikin" | "heiken_ashi" | "heiken-ashi" | "heiken" => {
            Ok(CandleMode::HeikinAshi)
        }
        other => bail!("invalid candle_mode `{other}`"),
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

fn default_replay_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".local/share/trader/replay-cache")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_config_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("trader-config-{name}-{nonce}.toml"))
    }

    #[test]
    fn config_loads_replay_cache_dir_from_file() {
        let path = temp_config_path("cache-root");
        let cache_dir = std::env::temp_dir().join("trader-cache-configured");
        fs::write(
            &path,
            format!("replay_cache_dir = '{}'\n", cache_dir.display()),
        )
        .expect("write config");

        let config = AppConfig::load(Some(&path)).expect("load config");

        assert_eq!(config.replay_cache_dir, cache_dir);
    }

    #[test]
    fn config_loads_replay_result_dir_from_file() {
        let path = temp_config_path("result-root");
        let result_dir = std::env::temp_dir().join("trader-replay-results-configured");
        fs::write(
            &path,
            format!("replay_result_dir = '{}'\n", result_dir.display()),
        )
        .expect("write config");

        let config = AppConfig::load(Some(&path)).expect("load config");

        assert_eq!(config.replay_result_dir, result_dir);
    }

    #[test]
    fn replay_account_and_margin_defaults_are_explicit() {
        let config: AppConfig = toml::from_str("").expect("default config");

        assert_eq!(config.replay_initial_capital, 100_000.0);
        assert_eq!(config.replay_account_currency, "USD");
        assert_eq!(config.replay_margin_model, "fixed_per_contract");
        assert_eq!(config.replay_margin_per_contract, 0.0);
        assert_eq!(config.replay_safety_buffer, 0.0);
        assert_eq!(config.replay_safety_buffer_percent, 0.0);
        assert_eq!(config.replay_post_exit_continuation_bars, 0);
        assert!(!config.replay_signal_diagnostics);
    }

    #[test]
    fn replay_account_and_margin_controls_load_from_file() {
        let config: AppConfig = toml::from_str(
            r#"
            replay_initial_capital = 25000.0
            replay_account_currency = "EUR"
            replay_margin_model = "fixed_per_contract"
            replay_margin_per_contract = 1400.0
            replay_safety_buffer = 250.0
            replay_safety_buffer_percent = 12.5
            replay_post_exit_continuation_bars = 8
            replay_signal_diagnostics = true
            "#,
        )
        .expect("configured replay account");

        assert_eq!(config.replay_initial_capital, 25_000.0);
        assert_eq!(config.replay_account_currency, "EUR");
        assert_eq!(config.replay_margin_per_contract, 1_400.0);
        assert_eq!(config.replay_safety_buffer, 250.0);
        assert_eq!(config.replay_safety_buffer_percent, 12.5);
        assert_eq!(config.replay_post_exit_continuation_bars, 8);
        assert!(config.replay_signal_diagnostics);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn default_replay_cache_dir_lives_outside_repo() {
        let config = AppConfig::default();

        assert!(
            config
                .replay_cache_dir
                .ends_with(".local/share/trader/replay-cache")
        );
        assert!(
            !config
                .replay_cache_dir
                .starts_with(env!("CARGO_MANIFEST_DIR"))
        );
    }

    #[test]
    fn replay_engine_defaults_to_legacy_compatibility() {
        let config: AppConfig = toml::from_str("").expect("default config");

        assert_eq!(config.replay_engine_mode, ReplayEngineMode::Legacy);
    }

    #[test]
    fn replay_engine_can_opt_into_deterministic_virtual_time() {
        let config: AppConfig =
            toml::from_str("replay_engine_mode = \"deterministic\"").expect("config");

        assert_eq!(config.replay_engine_mode, ReplayEngineMode::Deterministic);
        assert_eq!(config.replay_fill_model, ReplayFillModel::RawBarOpen);
    }

    #[test]
    fn deterministic_replay_can_select_tick_bid_ask_fills() {
        let config: AppConfig = toml::from_str(
            r#"
            replay_engine_mode = "deterministic"
            replay_fill_model = "tick_bid_ask"
            "#,
        )
        .expect("config");

        assert_eq!(config.replay_fill_model, ReplayFillModel::TickBidAsk);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn dom_replay_is_optional_and_loads_from_config() {
        let default_config: AppConfig = toml::from_str("").expect("default config");
        assert!(default_config.replay_dom_file_path.is_none());

        let configured: AppConfig = toml::from_str(
            r#"
            replay_engine_mode = "deterministic"
            replay_fill_model = "dom"
            replay_dom_file_path = "replay/es.dom.jsonl"
            "#,
        )
        .expect("config");

        assert_eq!(configured.replay_fill_model, ReplayFillModel::Dom);
        assert_eq!(
            configured.replay_dom_file_path,
            Some(PathBuf::from("replay/es.dom.jsonl"))
        );
        assert!(configured.validate().is_ok());
    }

    #[test]
    fn dom_replay_requires_deterministic_engine_only_when_selected() {
        let config = AppConfig {
            replay_fill_model: ReplayFillModel::Dom,
            ..AppConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn replay_fixed_latency_defaults_to_zero_and_loads_from_config() {
        let default_config: AppConfig = toml::from_str("").expect("default config");
        assert_eq!(default_config.replay_fixed_latency_ms, 0);

        let configured: AppConfig =
            toml::from_str("replay_fixed_latency_ms = 175").expect("config");
        assert_eq!(configured.replay_fixed_latency_ms, 175);
    }

    #[test]
    fn replay_observed_latency_and_protection_policy_load_from_config() {
        let configured: AppConfig = toml::from_str(
            r#"
            replay_engine_mode = "deterministic"
            replay_latency_model = "seeded_observed"
            replay_observed_latency_ms = [35, 60, 125]
            replay_latency_seed = 77
            replay_bar_protection_policy = "optimistic"
            "#,
        )
        .expect("config");

        assert_eq!(
            configured.replay_latency_model,
            ReplayLatencyModel::SeededObserved
        );
        assert_eq!(configured.replay_observed_latency_ms, vec![35, 60, 125]);
        assert_eq!(configured.replay_latency_seed, 77);
        assert_eq!(
            configured.replay_bar_protection_policy,
            ReplayBarProtectionPolicy::Optimistic
        );
        assert!(configured.validate().is_ok());
    }

    #[test]
    fn deterministic_observed_latency_requires_samples() {
        let config = AppConfig {
            replay_engine_mode: ReplayEngineMode::Deterministic,
            replay_latency_model: ReplayLatencyModel::ObservedP95,
            ..AppConfig::default()
        };

        let error = config.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires replay_observed_latency_ms")
        );
    }
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
