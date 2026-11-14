use crate::broker::{BarType, CandleMode};
use crate::strategies::adx::AdxConfig;
use crate::strategies::ema_cross::EmaCrossConfig;
use crate::strategies::heikin_ashi::HeikinAshiConfig;
use crate::strategies::hma_angle::HmaAngleConfig;
use crate::strategies::hma_cross::HmaCrossConfig;
use crate::strategies::volume_regime::{
    VolumeAdaptiveEmaCrossConfig, VolumeAdaptiveHmaCrossConfig,
};
use chrono::{DateTime, NaiveTime, Utc};
use chrono_tz::Tz;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::str::FromStr;

pub const DEFAULT_BLOCKOUT_MINUTES_BEFORE_CLOSE: f64 = 45.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrategyDailyWindow {
    pub enabled: bool,
    pub timezone: String,
    pub start_local: String,
    pub end_local: String,
    #[serde(default)]
    pub flatten_at_session_end: bool,
}

impl Default for StrategyDailyWindow {
    fn default() -> Self {
        Self {
            enabled: false,
            timezone: "America/New_York".into(),
            start_local: "08:30".into(),
            end_local: "11:15".into(),
            flatten_at_session_end: false,
        }
    }
}

impl StrategyDailyWindow {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        Tz::from_str(self.timezone.trim())
            .map_err(|_| anyhow::anyhow!("Unknown strategy-window timezone: {}", self.timezone))?;
        let start = NaiveTime::parse_from_str(self.start_local.trim(), "%H:%M").map_err(|_| {
            anyhow::anyhow!("Invalid strategy-window start; use HH:MM (00:00–23:59).")
        })?;
        let end = NaiveTime::parse_from_str(self.end_local.trim(), "%H:%M").map_err(|_| {
            anyhow::anyhow!("Invalid strategy-window end; use HH:MM (00:00–23:59).")
        })?;
        if start == end {
            anyhow::bail!("Strategy-window start and end must differ.");
        }
        Ok(())
    }

    /// Invalid configuration/timestamp fails closed for exposure-increasing entries.
    pub fn contains_timestamp(&self, ts_ns: i64) -> anyhow::Result<bool> {
        if !self.enabled {
            return Ok(true);
        }
        self.validate()?;
        if ts_ns <= 0 {
            return Ok(false);
        }
        let tz = Tz::from_str(self.timezone.trim())
            .map_err(|_| anyhow::anyhow!("Unknown strategy-window timezone: {}", self.timezone))?;
        let start = NaiveTime::parse_from_str(self.start_local.trim(), "%H:%M")?;
        let end = NaiveTime::parse_from_str(self.end_local.trim(), "%H:%M")?;
        let utc = DateTime::<Utc>::from_timestamp(
            ts_ns.div_euclid(1_000_000_000),
            ts_ns.rem_euclid(1_000_000_000) as u32,
        )
        .ok_or_else(|| {
            anyhow::anyhow!("Strategy-window timestamp is outside the supported UTC range.")
        })?;
        let local: DateTime<Tz> = utc.with_timezone(&tz);
        let time = local.time();
        Ok(if start < end {
            time >= start && time < end
        } else {
            time >= start || time < end
        })
    }

    /// Window-aware target check: outside the window, same-side reductions and
    /// flattening remain legal while new exposure and reversals are rejected.
    pub fn allows_target(
        &self,
        ts_ns: i64,
        current_qty: i32,
        target_qty: i32,
    ) -> anyhow::Result<bool> {
        if !self.enabled || self.contains_timestamp(ts_ns)? {
            return Ok(true);
        }
        Ok(target_qty.abs() <= current_qty.abs()
            && (target_qty == 0 || target_qty.signum() == current_qty.signum()))
    }

    /// Constrain a proposed position target while the daily window is closed.
    ///
    /// A zero target and same-side exposure reductions remain legal.  A
    /// reversal is converted to a flatten-only target, while a new entry or
    /// exposure increase is rejected.
    pub fn constrain_target_outside_window(current_qty: i32, target_qty: i32) -> Option<i32> {
        if target_qty == 0 {
            return Some(0);
        }
        if current_qty != 0
            && target_qty.signum() == current_qty.signum()
            && target_qty.abs() <= current_qty.abs()
        {
            return Some(target_qty);
        }
        if current_qty != 0 && target_qty.signum() != current_qty.signum() {
            return Some(0);
        }
        None
    }
}

/// Extra history is requested so recursive/rolling indicators are not seeded
/// at the first bar that the strategy can trade.  The strategy evaluator still
/// owns the exact warm-up gate; this is only the broker-history request policy.
pub const MARKET_HISTORY_STABILITY_MULTIPLIER: usize = 2;

/// Minimum extra context beyond the indicator's explicit warm-up boundary.
pub const MARKET_HISTORY_STABILITY_MARGIN: usize = 64;

/// Native indicators are rebuilt from the broker's historical snapshot after
/// a market-stream reconnect.  Keep enough history that recursive indicators
/// such as a slow EMA are effectively converged again instead of being
/// reseeded from a short window.  This stays below the engine's 4,096-bar
/// retained-market limit and is applied consistently on the initial
/// subscription and every automatic stream restart.
pub const MARKET_HISTORY_NATIVE_MINIMUM_BARS: usize = 3_200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyKind {
    Native,
    Lua,
    MachineLearning,
}

impl StrategyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "Native Rust",
            Self::Lua => "Lua",
            Self::MachineLearning => "Machine Learning",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeStrategyKind {
    HmaAngle,
    EmaCross,
    HmaCross,
    HeikinAshiColor,
    VolumeAdaptiveHmaCross,
    VolumeAdaptiveEmaCross,
    Adx,
}

impl NativeStrategyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HmaAngle => "HMA Angle",
            Self::EmaCross => "EMA Crossover",
            Self::HmaCross => "HMA Crossover",
            Self::HeikinAshiColor => "Heikin-Ashi Color",
            Self::VolumeAdaptiveHmaCross => "Volume-Adaptive HMA",
            Self::VolumeAdaptiveEmaCross => "Volume-Adaptive EMA",
            Self::Adx => "ADX Trend Regime",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::HmaAngle => "hma_angle",
            Self::EmaCross => "ema_cross",
            Self::HmaCross => "hma_cross",
            Self::HeikinAshiColor => "heikin_ashi_color",
            Self::VolumeAdaptiveHmaCross => "volume_adaptive_hma_cross",
            Self::VolumeAdaptiveEmaCross => "volume_adaptive_ema_cross",
            Self::Adx => "adx",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::HmaAngle => Self::EmaCross,
            Self::EmaCross => Self::HmaCross,
            Self::HmaCross => Self::HeikinAshiColor,
            Self::HeikinAshiColor => Self::VolumeAdaptiveHmaCross,
            Self::VolumeAdaptiveHmaCross => Self::VolumeAdaptiveEmaCross,
            Self::VolumeAdaptiveEmaCross => Self::Adx,
            Self::Adx => Self::HmaAngle,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::HmaAngle => Self::Adx,
            Self::EmaCross => Self::HmaAngle,
            Self::HmaCross => Self::EmaCross,
            Self::HeikinAshiColor => Self::HmaCross,
            Self::VolumeAdaptiveHmaCross => Self::HeikinAshiColor,
            Self::VolumeAdaptiveEmaCross => Self::VolumeAdaptiveHmaCross,
            Self::Adx => Self::VolumeAdaptiveEmaCross,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeSignalTiming {
    ClosedBar,
    LiveBar,
}

impl NativeSignalTiming {
    pub fn label(self) -> &'static str {
        match self {
            Self::ClosedBar => "Closed Bar",
            Self::LiveBar => "Live Bar",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::ClosedBar => Self::LiveBar,
            Self::LiveBar => Self::ClosedBar,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeExecutionPath {
    Guarded,
    SimpleDiagnostic,
    HmaDirect,
}

impl NativeExecutionPath {
    pub fn label(self) -> &'static str {
        match self {
            Self::Guarded => "Guarded",
            Self::SimpleDiagnostic => "Simple Diagnostic",
            Self::HmaDirect => "HMA Direct",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Guarded => Self::SimpleDiagnostic,
            Self::SimpleDiagnostic => Self::HmaDirect,
            Self::HmaDirect => Self::Guarded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeReversalMode {
    Direct,
    FlattenConfirmEnter,
    CloseAllEnter,
}

impl NativeReversalMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct => "Direct",
            Self::FlattenConfirmEnter => "Flatten > Confirm > Enter",
            Self::CloseAllEnter => "CloseAll > Enter",
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Direct => Self::CloseAllEnter,
            Self::FlattenConfirmEnter => Self::Direct,
            Self::CloseAllEnter => Self::FlattenConfirmEnter,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Direct => Self::FlattenConfirmEnter,
            Self::FlattenConfirmEnter => Self::CloseAllEnter,
            Self::CloseAllEnter => Self::Direct,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuaSourceMode {
    File,
    Editor,
}

impl LuaSourceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::File => "Load From File",
            Self::Editor => "Type In TUI",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::File => Self::Editor,
            Self::Editor => Self::File,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VimMode {
    Normal,
    Insert,
}

impl VimMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "NORMAL",
            Self::Insert => "INSERT",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VimEditor {
    lines: Vec<String>,
    row: usize,
    col: usize,
    mode: VimMode,
}

impl VimEditor {
    pub fn new_with_template() -> Self {
        Self::from_text(
            "function on_init(ctx)\n  -- optional setup\nend\n\nfunction on_bar(ctx, bar)\n  return 'hold'\nend\n",
        )
    }

    pub fn from_text(text: &str) -> Self {
        let mut lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
        if text.ends_with('\n') {
            lines.push(String::new());
        }
        if lines.is_empty() {
            lines.push(String::new());
        }
        Self {
            lines,
            row: 0,
            col: 0,
            mode: VimMode::Normal,
        }
    }

    pub fn mode(&self) -> VimMode {
        self.mode
    }

    pub fn cursor(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub fn set_text(&mut self, text: &str) {
        *self = Self::from_text(text);
    }

    pub fn visible_lines(&self, max_lines: usize) -> Vec<String> {
        let start = self.window_start(max_lines);
        let end = (start + max_lines).min(self.lines.len());
        self.lines[start..end].to_vec()
    }

    pub fn window_start(&self, max_lines: usize) -> usize {
        if max_lines == 0 {
            return 0;
        }
        self.row.saturating_sub(max_lines / 2)
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match self.mode {
            VimMode::Normal => self.handle_normal_mode(key),
            VimMode::Insert => self.handle_insert_mode(key),
        }
    }

    fn handle_normal_mode(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('i') => {
                self.mode = VimMode::Insert;
                true
            }
            KeyCode::Char('a') => {
                let line_len = self.current_line_len();
                self.col = (self.col + 1).min(line_len);
                self.mode = VimMode::Insert;
                true
            }
            KeyCode::Char('h') | KeyCode::Left => {
                self.col = self.col.saturating_sub(1);
                true
            }
            KeyCode::Char('l') | KeyCode::Right => {
                self.col = (self.col + 1).min(self.current_line_len());
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.row = self.row.saturating_sub(1);
                self.clamp_cursor();
                true
            }
            KeyCode::Char('j') | KeyCode::Down => {
                if self.row + 1 < self.lines.len() {
                    self.row += 1;
                }
                self.clamp_cursor();
                true
            }
            KeyCode::Char('0') => {
                self.col = 0;
                true
            }
            KeyCode::Char('$') => {
                self.col = self.current_line_len();
                true
            }
            KeyCode::Char('x') => {
                self.delete_char_under_cursor();
                true
            }
            KeyCode::Char('o') => {
                self.row += 1;
                self.lines.insert(self.row, String::new());
                self.col = 0;
                self.mode = VimMode::Insert;
                true
            }
            _ => false,
        }
    }

    fn handle_insert_mode(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                self.mode = VimMode::Normal;
                self.col = self.col.saturating_sub(1).min(self.current_line_len());
                true
            }
            KeyCode::Backspace => {
                self.backspace();
                true
            }
            KeyCode::Enter => {
                self.split_line();
                true
            }
            KeyCode::Left => {
                self.col = self.col.saturating_sub(1);
                true
            }
            KeyCode::Right => {
                self.col = (self.col + 1).min(self.current_line_len());
                true
            }
            KeyCode::Up => {
                self.row = self.row.saturating_sub(1);
                self.clamp_cursor();
                true
            }
            KeyCode::Down => {
                if self.row + 1 < self.lines.len() {
                    self.row += 1;
                }
                self.clamp_cursor();
                true
            }
            KeyCode::Tab => {
                self.insert_text("  ");
                true
            }
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.insert_char(ch);
                true
            }
            _ => false,
        }
    }

    fn insert_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.insert_char(ch);
        }
    }

    fn insert_char(&mut self, ch: char) {
        if let Some(line) = self.lines.get_mut(self.row) {
            let idx = self.col.min(line.len());
            line.insert(idx, ch);
            self.col = idx + ch.len_utf8();
        }
    }

    fn split_line(&mut self) {
        let idx = self.col.min(self.current_line_len());
        let tail = self.lines[self.row].split_off(idx);
        self.row += 1;
        self.lines.insert(self.row, tail);
        self.col = 0;
    }

    fn backspace(&mut self) {
        if self.col > 0 {
            if let Some(line) = self.lines.get_mut(self.row) {
                let idx = self.col - 1;
                line.remove(idx);
                self.col = idx;
            }
            return;
        }

        if self.row > 0 {
            let current = self.lines.remove(self.row);
            self.row -= 1;
            let prev_len = self.lines[self.row].len();
            self.lines[self.row].push_str(&current);
            self.col = prev_len;
        }
    }

    fn delete_char_under_cursor(&mut self) {
        let line_len = self.current_line_len();
        if self.col < line_len {
            self.lines[self.row].remove(self.col);
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
        }
    }

    fn current_line_len(&self) -> usize {
        self.lines.get(self.row).map(|line| line.len()).unwrap_or(0)
    }

    fn clamp_cursor(&mut self) {
        self.col = self.col.min(self.current_line_len());
    }
}

#[derive(Debug, Clone)]
pub struct StrategyState {
    pub kind: StrategyKind,
    pub native_strategy: NativeStrategyKind,
    pub native_signal_timing: NativeSignalTiming,
    pub native_signal_delay_bars: usize,
    pub native_execution_path: NativeExecutionPath,
    pub native_reversal_mode: NativeReversalMode,
    pub blockout_enabled: bool,
    pub blockout_minutes_before_close: f64,
    pub daily_window: StrategyDailyWindow,
    pub native_hma: HmaAngleConfig,
    pub native_ema: EmaCrossConfig,
    pub native_hma_cross: HmaCrossConfig,
    pub native_heikin_ashi: HeikinAshiConfig,
    pub native_volume_hma_cross: VolumeAdaptiveHmaCrossConfig,
    pub native_volume_ema_cross: VolumeAdaptiveEmaCrossConfig,
    pub native_adx: AdxConfig,
    pub order_qty: i32,
    pub lua_source_mode: LuaSourceMode,
    pub lua_file_path: String,
    pub lua_editor: VimEditor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionStrategyConfig {
    pub kind: StrategyKind,
    pub native_strategy: NativeStrategyKind,
    #[serde(default = "default_native_signal_timing")]
    pub native_signal_timing: NativeSignalTiming,
    #[serde(default = "default_native_signal_delay_bars")]
    pub native_signal_delay_bars: usize,
    #[serde(default = "default_native_execution_path")]
    pub native_execution_path: NativeExecutionPath,
    #[serde(default = "default_native_reversal_mode")]
    pub native_reversal_mode: NativeReversalMode,
    #[serde(default = "default_blockout_enabled")]
    pub blockout_enabled: bool,
    #[serde(default = "default_blockout_minutes_before_close")]
    pub blockout_minutes_before_close: f64,
    #[serde(default)]
    pub daily_window: StrategyDailyWindow,
    pub native_hma: HmaAngleConfig,
    pub native_ema: EmaCrossConfig,
    #[serde(default)]
    pub native_hma_cross: HmaCrossConfig,
    #[serde(default)]
    pub native_heikin_ashi: HeikinAshiConfig,
    #[serde(default)]
    pub native_volume_hma_cross: VolumeAdaptiveHmaCrossConfig,
    #[serde(default)]
    pub native_volume_ema_cross: VolumeAdaptiveEmaCrossConfig,
    #[serde(default)]
    pub native_adx: AdxConfig,
    #[serde(default)]
    pub lua: LuaStrategyConfig,
    #[serde(default = "default_order_qty")]
    pub order_qty: i32,
}

/// Serializable source and bounded-runtime settings for offline Lua replay.
/// The version is part of the strategy identity and rejects unknown contracts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LuaStrategyConfig {
    pub schema_version: u32,
    pub source: String,
    pub memory_limit_bytes: usize,
    pub instruction_limit: u64,
}

impl Default for LuaStrategyConfig {
    fn default() -> Self {
        Self {
            schema_version: 1,
            source: String::new(),
            memory_limit_bytes: 8 * 1024 * 1024,
            instruction_limit: 100_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketHistoryRequirement {
    pub configured_bars: usize,
    pub warmup_bars: usize,
    pub requested_bars: usize,
}

fn default_order_qty() -> i32 {
    1
}

fn default_native_signal_timing() -> NativeSignalTiming {
    NativeSignalTiming::ClosedBar
}

fn default_native_signal_delay_bars() -> usize {
    0
}

fn default_native_execution_path() -> NativeExecutionPath {
    NativeExecutionPath::Guarded
}

fn default_native_reversal_mode() -> NativeReversalMode {
    NativeReversalMode::Direct
}

fn default_blockout_enabled() -> bool {
    true
}

fn default_blockout_minutes_before_close() -> f64 {
    DEFAULT_BLOCKOUT_MINUTES_BEFORE_CLOSE
}

impl Default for ExecutionStrategyConfig {
    fn default() -> Self {
        Self {
            kind: StrategyKind::Native,
            native_strategy: NativeStrategyKind::HmaAngle,
            native_signal_timing: NativeSignalTiming::ClosedBar,
            native_signal_delay_bars: 0,
            native_execution_path: NativeExecutionPath::Guarded,
            native_reversal_mode: NativeReversalMode::Direct,
            blockout_enabled: true,
            blockout_minutes_before_close: DEFAULT_BLOCKOUT_MINUTES_BEFORE_CLOSE,
            daily_window: StrategyDailyWindow::default(),
            native_hma: HmaAngleConfig::default(),
            native_ema: EmaCrossConfig::default(),
            native_hma_cross: HmaCrossConfig::default(),
            native_heikin_ashi: HeikinAshiConfig::default(),
            native_volume_hma_cross: VolumeAdaptiveHmaCrossConfig::default(),
            native_volume_ema_cross: VolumeAdaptiveEmaCrossConfig::default(),
            native_adx: AdxConfig::default(),
            lua: LuaStrategyConfig::default(),
            order_qty: 1,
        }
    }
}

impl ExecutionStrategyConfig {
    /// Return the number of bars needed before the active strategy can make a
    /// valid first decision.  These are the same strategy-owned warm-up rules
    /// used by the evaluators, including optional gate/regime dependencies.
    pub fn strategy_warmup_bars(&self) -> usize {
        if self.kind != StrategyKind::Native {
            return 0;
        }

        match self.native_strategy {
            NativeStrategyKind::HmaAngle => self.native_hma.warmup_bars(),
            NativeStrategyKind::EmaCross => {
                self.native_ema
                    .warmup_bars()
                    .max(if self.native_ema.direction_gate.enabled {
                        self.native_ema.direction_gate.warmup_bars()
                    } else {
                        0
                    })
            }
            NativeStrategyKind::HmaCross => self.native_hma_cross.warmup_bars().max(
                if self.native_hma_cross.direction_gate.enabled {
                    self.native_hma_cross.direction_gate.warmup_bars()
                } else {
                    0
                },
            ),
            NativeStrategyKind::HeikinAshiColor => self.native_heikin_ashi.warmup_bars(),
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                self.native_volume_hma_cross.warmup_bars()
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                self.native_volume_ema_cross.warmup_bars()
            }
            NativeStrategyKind::Adx => self.native_adx.warmup_bars(),
        }
    }

    /// Calculate the broker history request for this strategy.
    ///
    /// The configured value remains the floor for compatibility.  A second
    /// warm-up window plus a fixed margin gives recursive indicators enough
    /// context to settle before the first live bar can create a crossover.
    pub fn market_history_requirement(&self, configured_bars: usize) -> MarketHistoryRequirement {
        let configured_bars = configured_bars.max(1);
        let warmup_bars = self.strategy_warmup_bars();
        let stability_bars = warmup_bars
            .saturating_mul(MARKET_HISTORY_STABILITY_MULTIPLIER)
            .max(warmup_bars.saturating_add(MARKET_HISTORY_STABILITY_MARGIN));
        let requested_bars = configured_bars.max(stability_bars);
        let requested_bars = if self.kind == StrategyKind::Native {
            requested_bars.max(MARKET_HISTORY_NATIVE_MINIMUM_BARS)
        } else {
            requested_bars
        };
        MarketHistoryRequirement {
            configured_bars,
            warmup_bars,
            requested_bars,
        }
    }

    /// Whether the active crossover strategy's directional EMA gate is using
    /// its gate-specific flat-only entry behavior.
    pub fn flat_only_direction_gate_active(&self) -> bool {
        let gate = match self.native_strategy {
            NativeStrategyKind::EmaCross => Some(&self.native_ema.direction_gate),
            NativeStrategyKind::HmaCross => Some(&self.native_hma_cross.direction_gate),
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                Some(&self.native_volume_ema_cross.ema_cross.direction_gate)
            }
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                Some(&self.native_volume_hma_cross.hma_cross.direction_gate)
            }
            _ => None,
        };
        gate.is_some_and(|gate| gate.enabled && gate.flat_only_entries)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecutionRuntimeSnapshot {
    pub armed: bool,
    pub last_closed_bar_ts: Option<i64>,
    pub pending_target_qty: Option<i32>,
    pub last_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionStateSnapshot {
    pub config: ExecutionStrategyConfig,
    pub runtime: ExecutionRuntimeSnapshot,
    /// The market shape currently owned by the engine. These are optional so
    /// a newer TUI can still attach to an older engine without replacing a
    /// known selection with the minute/OHLC defaults.
    #[serde(default)]
    pub bar_type: Option<BarType>,
    #[serde(default)]
    pub candle_mode: Option<CandleMode>,
    #[serde(default)]
    pub selected_account_id: Option<i64>,
    #[serde(default)]
    pub selected_contract_name: Option<String>,
    #[serde(default)]
    pub market_position_qty: i32,
    #[serde(default)]
    pub market_entry_price: Option<f64>,
    #[serde(default)]
    pub selected_contract_take_profit_price: Option<f64>,
    #[serde(default)]
    pub selected_contract_stop_price: Option<f64>,
}

impl Default for ExecutionStateSnapshot {
    fn default() -> Self {
        Self {
            config: ExecutionStrategyConfig::default(),
            runtime: ExecutionRuntimeSnapshot::default(),
            bar_type: None,
            candle_mode: None,
            selected_account_id: None,
            selected_contract_name: None,
            market_position_qty: 0,
            market_entry_price: None,
            selected_contract_take_profit_price: None,
            selected_contract_stop_price: None,
        }
    }
}

impl StrategyState {
    pub fn new() -> Self {
        Self {
            kind: StrategyKind::Native,
            native_strategy: NativeStrategyKind::HmaAngle,
            native_signal_timing: NativeSignalTiming::ClosedBar,
            native_signal_delay_bars: 0,
            native_execution_path: NativeExecutionPath::Guarded,
            native_reversal_mode: NativeReversalMode::Direct,
            blockout_enabled: true,
            blockout_minutes_before_close: DEFAULT_BLOCKOUT_MINUTES_BEFORE_CLOSE,
            daily_window: StrategyDailyWindow::default(),
            native_hma: HmaAngleConfig::default(),
            native_ema: EmaCrossConfig::default(),
            native_hma_cross: HmaCrossConfig::default(),
            native_heikin_ashi: HeikinAshiConfig::default(),
            native_volume_hma_cross: VolumeAdaptiveHmaCrossConfig::default(),
            native_volume_ema_cross: VolumeAdaptiveEmaCrossConfig::default(),
            native_adx: AdxConfig::default(),
            order_qty: 1,
            lua_source_mode: LuaSourceMode::Editor,
            lua_file_path: String::new(),
            lua_editor: VimEditor::new_with_template(),
        }
    }

    pub fn load_lua_file(&mut self) -> std::io::Result<usize> {
        let path = Path::new(self.lua_file_path.trim());
        let text = fs::read_to_string(path)?;
        let len = text.lines().count().max(1);
        self.lua_editor.set_text(&text);
        Ok(len)
    }

    pub fn summary_label(&self) -> String {
        match self.kind {
            StrategyKind::Native => {
                format!("Native Rust / {}", self.native_strategy.label())
            }
            StrategyKind::MachineLearning => "Machine Learning".to_string(),
            StrategyKind::Lua => match self.lua_source_mode {
                LuaSourceMode::File => {
                    let path = if self.lua_file_path.trim().is_empty() {
                        "no file"
                    } else {
                        self.lua_file_path.trim()
                    };
                    format!("Lua (file: {path})")
                }
                LuaSourceMode::Editor => {
                    format!("Lua (editor, {} lines)", self.lua_editor.line_count())
                }
            },
        }
    }

    pub fn native_summary(&self) -> String {
        match self.native_strategy {
            NativeStrategyKind::HmaAngle => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) len={} angle={:.1} lookback={} bars_required={} longs_only={} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::HmaAngle.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_hma.hma_length,
                self.native_hma.min_angle,
                self.native_hma.angle_lookback,
                self.native_hma.bars_required_to_trade,
                self.native_hma.longs_only,
                self.native_hma.take_profit_ticks,
                self.native_hma.stop_loss_ticks,
                self.native_hma.use_trailing_stop,
                self.native_hma.inverted,
            ),
            NativeStrategyKind::EmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::EmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_ema.fast_length,
                self.native_ema.slow_length,
                self.native_ema.take_profit_ticks,
                self.native_ema.stop_loss_ticks,
                self.native_ema.use_trailing_stop,
                self.native_ema.inverted,
            ),
            NativeStrategyKind::HmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::HmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_hma_cross.fast_length,
                self.native_hma_cross.slow_length,
                self.native_hma_cross.take_profit_ticks,
                self.native_hma_cross.stop_loss_ticks,
                self.native_hma_cross.use_trailing_stop,
                self.native_hma_cross.inverted,
            ),
            NativeStrategyKind::HeikinAshiColor => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) transitions=green-buy/red-sell inverted={}",
                NativeStrategyKind::HeikinAshiColor.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_heikin_ashi.inverted,
            ),
            NativeStrategyKind::VolumeAdaptiveHmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} lookback={} invert_below={:.2} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::VolumeAdaptiveHmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_volume_hma_cross.hma_cross.fast_length,
                self.native_volume_hma_cross.hma_cross.slow_length,
                self.native_volume_hma_cross.volume_regime.lookback_bars,
                self.native_volume_hma_cross
                    .volume_regime
                    .invert_below_relative_volume,
                self.native_volume_hma_cross.hma_cross.take_profit_ticks,
                self.native_volume_hma_cross.hma_cross.stop_loss_ticks,
                self.native_volume_hma_cross.hma_cross.use_trailing_stop,
                self.native_volume_hma_cross.hma_cross.inverted,
            ),
            NativeStrategyKind::VolumeAdaptiveEmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} lookback={} invert_below={:.2} gate_ema={} gate_enabled={} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::VolumeAdaptiveEmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_volume_ema_cross.ema_cross.fast_length,
                self.native_volume_ema_cross.ema_cross.slow_length,
                self.native_volume_ema_cross.volume_regime.lookback_bars,
                self.native_volume_ema_cross
                    .volume_regime
                    .invert_below_relative_volume,
                self.native_volume_ema_cross.ema_gate.ema_length,
                self.native_volume_ema_cross.ema_gate.enabled,
                self.native_volume_ema_cross.ema_cross.take_profit_ticks,
                self.native_volume_ema_cross.ema_cross.stop_loss_ticks,
                self.native_volume_ema_cross.ema_cross.use_trailing_stop,
                self.native_volume_ema_cross.ema_cross.inverted,
            ),
            NativeStrategyKind::Adx => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) len={} entry={:.1} exit={:.1} imbalance={:.2} slope={} dominance={} breakout={} tp={:.0} sl={:.0} trail={} inverted={}",
                NativeStrategyKind::Adx.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_adx.adx_length,
                self.native_adx.adx_entry_threshold,
                self.native_adx.adx_exit_threshold,
                self.native_adx.di_imbalance_threshold,
                self.native_adx.slope_lookback,
                self.native_adx.dominance_bars,
                self.native_adx.breakout_lookback,
                self.native_adx.take_profit_ticks,
                self.native_adx.stop_loss_ticks,
                self.native_adx.use_trailing_stop,
                self.native_adx.inverted,
            ),
        }
    }

    pub fn native_summary_without_protection(&self) -> String {
        match self.native_strategy {
            NativeStrategyKind::HmaAngle => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) len={} angle={:.1} lookback={} bars_required={} longs_only={} inverted={}",
                NativeStrategyKind::HmaAngle.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_hma.hma_length,
                self.native_hma.min_angle,
                self.native_hma.angle_lookback,
                self.native_hma.bars_required_to_trade,
                self.native_hma.longs_only,
                self.native_hma.inverted,
            ),
            NativeStrategyKind::EmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} inverted={}",
                NativeStrategyKind::EmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_ema.fast_length,
                self.native_ema.slow_length,
                self.native_ema.inverted,
            ),
            NativeStrategyKind::HmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} inverted={}",
                NativeStrategyKind::HmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_hma_cross.fast_length,
                self.native_hma_cross.slow_length,
                self.native_hma_cross.inverted,
            ),
            NativeStrategyKind::HeikinAshiColor => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) transitions=green-buy/red-sell inverted={}",
                NativeStrategyKind::HeikinAshiColor.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_heikin_ashi.inverted,
            ),
            NativeStrategyKind::VolumeAdaptiveHmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} lookback={} invert_below={:.2} inverted={}",
                NativeStrategyKind::VolumeAdaptiveHmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_volume_hma_cross.hma_cross.fast_length,
                self.native_volume_hma_cross.hma_cross.slow_length,
                self.native_volume_hma_cross.volume_regime.lookback_bars,
                self.native_volume_hma_cross
                    .volume_regime
                    .invert_below_relative_volume,
                self.native_volume_hma_cross.hma_cross.inverted,
            ),
            NativeStrategyKind::VolumeAdaptiveEmaCross => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) fast={} slow={} lookback={} invert_below={:.2} gate_ema={} gate_enabled={} inverted={}",
                NativeStrategyKind::VolumeAdaptiveEmaCross.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_volume_ema_cross.ema_cross.fast_length,
                self.native_volume_ema_cross.ema_cross.slow_length,
                self.native_volume_ema_cross.volume_regime.lookback_bars,
                self.native_volume_ema_cross
                    .volume_regime
                    .invert_below_relative_volume,
                self.native_volume_ema_cross.ema_gate.ema_length,
                self.native_volume_ema_cross.ema_gate.enabled,
                self.native_volume_ema_cross.ema_cross.inverted,
            ),
            NativeStrategyKind::Adx => format!(
                "{} | qty={} timing={} delay={} path={} reversal={} blockout={}({:.0}m) len={} entry={:.1} exit={:.1} imbalance={:.2} slope={} dominance={} breakout={} inverted={}",
                NativeStrategyKind::Adx.label(),
                self.order_qty,
                self.native_signal_timing.label(),
                self.native_signal_delay_bars,
                self.native_execution_path.label(),
                self.native_reversal_mode.label(),
                self.blockout_enabled,
                self.blockout_minutes_before_close,
                self.native_adx.adx_length,
                self.native_adx.adx_entry_threshold,
                self.native_adx.adx_exit_threshold,
                self.native_adx.di_imbalance_threshold,
                self.native_adx.slope_lookback,
                self.native_adx.dominance_bars,
                self.native_adx.breakout_lookback,
                self.native_adx.inverted,
            ),
        }
    }

    pub fn execution_config(&self) -> ExecutionStrategyConfig {
        ExecutionStrategyConfig {
            kind: self.kind,
            native_strategy: self.native_strategy,
            native_signal_timing: self.native_signal_timing,
            native_signal_delay_bars: self.native_signal_delay_bars,
            native_execution_path: self.native_execution_path,
            native_reversal_mode: self.native_reversal_mode,
            blockout_enabled: self.blockout_enabled,
            blockout_minutes_before_close: self.blockout_minutes_before_close,
            daily_window: self.daily_window.clone(),
            native_hma: self.native_hma.clone(),
            native_ema: self.native_ema.clone(),
            native_hma_cross: self.native_hma_cross.clone(),
            native_heikin_ashi: self.native_heikin_ashi.clone(),
            native_volume_hma_cross: self.native_volume_hma_cross.clone(),
            native_volume_ema_cross: self.native_volume_ema_cross.clone(),
            native_adx: self.native_adx.clone(),
            lua: LuaStrategyConfig {
                source: self.lua_editor.text(),
                ..LuaStrategyConfig::default()
            },
            order_qty: self.order_qty,
        }
    }

    pub fn apply_execution_config(&mut self, config: &ExecutionStrategyConfig) {
        self.kind = config.kind;
        self.native_strategy = config.native_strategy;
        self.native_signal_timing = config.native_signal_timing;
        self.native_signal_delay_bars = config.native_signal_delay_bars;
        self.native_execution_path = config.native_execution_path;
        self.native_reversal_mode = config.native_reversal_mode;
        self.blockout_enabled = config.blockout_enabled;
        self.blockout_minutes_before_close = config.blockout_minutes_before_close;
        self.daily_window = config.daily_window.clone();
        self.native_hma = config.native_hma.clone();
        self.native_ema = config.native_ema.clone();
        self.native_hma_cross = config.native_hma_cross.clone();
        self.native_heikin_ashi = config.native_heikin_ashi.clone();
        self.native_volume_hma_cross = config.native_volume_hma_cross.clone();
        self.native_volume_ema_cross = config.native_volume_ema_cross.clone();
        self.native_adx = config.native_adx.clone();
        if !config.lua.source.is_empty() {
            self.lua_editor.set_text(&config.lua.source);
        }
        self.order_qty = config.order_qty;
    }
}

#[cfg(test)]
mod market_history_tests {
    use super::*;

    #[test]
    fn lua_source_and_limits_round_trip_in_execution_config() {
        let mut config = ExecutionStrategyConfig::default();
        config.kind = StrategyKind::Lua;
        config.lua.source = "function on_bar(ctx, bar) return 'hold' end".into();
        config.lua.instruction_limit = 50_000;
        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: ExecutionStrategyConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.lua, config.lua);
    }

    #[test]
    fn history_request_grows_for_slow_ema() {
        let mut config = ExecutionStrategyConfig::default();
        config.native_strategy = NativeStrategyKind::EmaCross;
        config.native_ema.fast_length = 100;
        config.native_ema.slow_length = 620;

        let requirement = config.market_history_requirement(500);

        assert_eq!(requirement.configured_bars, 500);
        assert_eq!(requirement.warmup_bars, 621);
        assert_eq!(
            requirement.requested_bars,
            MARKET_HISTORY_NATIVE_MINIMUM_BARS
        );
    }

    #[test]
    fn enabled_direction_gate_can_extend_warmup() {
        let mut config = ExecutionStrategyConfig::default();
        config.native_strategy = NativeStrategyKind::EmaCross;
        config.native_ema.fast_length = 10;
        config.native_ema.slow_length = 30;
        config.native_ema.direction_gate.enabled = true;
        config.native_ema.direction_gate.fast_length = 100;
        config.native_ema.direction_gate.slow_length = 200;

        let requirement = config.market_history_requirement(500);

        assert_eq!(requirement.warmup_bars, 201);
        assert_eq!(
            requirement.requested_bars,
            MARKET_HISTORY_NATIVE_MINIMUM_BARS
        );
    }

    #[test]
    fn default_history_remains_the_floor_for_short_strategies() {
        let config = ExecutionStrategyConfig::default();
        let requirement = config.market_history_requirement(500);

        assert_eq!(requirement.configured_bars, 500);
        assert!(requirement.requested_bars >= 500);
        assert_eq!(
            requirement.requested_bars,
            MARKET_HISTORY_NATIVE_MINIMUM_BARS
        );
        assert!(requirement.requested_bars >= requirement.warmup_bars);
    }

    #[test]
    fn non_native_strategies_keep_the_configured_history_floor() {
        let mut config = ExecutionStrategyConfig::default();
        config.kind = StrategyKind::Lua;

        let requirement = config.market_history_requirement(500);

        assert_eq!(requirement.requested_bars, 500);
    }
}

#[cfg(test)]
mod daily_window_tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(local: &str) -> i64 {
        let tz = Tz::America__New_York;
        tz.from_local_datetime(
            &chrono::NaiveDateTime::parse_from_str(local, "%Y-%m-%d %H:%M").unwrap(),
        )
        .single()
        .unwrap()
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
        .unwrap()
    }

    #[test]
    fn old_execution_config_defaults_daily_window_off() {
        let mut value = serde_json::to_value(ExecutionStrategyConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("daily_window");
        let restored: ExecutionStrategyConfig = serde_json::from_value(value).unwrap();
        assert!(!restored.daily_window.enabled);
        assert_eq!(restored.daily_window.timezone, "America/New_York");
        assert_eq!(restored.daily_window.start_local, "08:30");
        assert!(!restored.daily_window.flatten_at_session_end);
    }

    #[test]
    fn daily_window_is_start_inclusive_end_exclusive_and_dst_aware() {
        let window = StrategyDailyWindow {
            enabled: true,
            ..Default::default()
        };
        assert!(window.contains_timestamp(ts("2026-01-15 08:30")).unwrap());
        assert!(!window.contains_timestamp(ts("2026-01-15 11:15")).unwrap());
        assert!(window.contains_timestamp(ts("2026-07-15 08:30")).unwrap());
        assert!(!window.contains_timestamp(ts("2026-07-15 08:29")).unwrap());
    }

    #[test]
    fn overnight_window_and_invalid_config_fail_safely() {
        let overnight = StrategyDailyWindow {
            enabled: true,
            start_local: "22:00".into(),
            end_local: "02:00".into(),
            ..Default::default()
        };
        assert!(
            overnight
                .contains_timestamp(ts("2026-01-15 23:00"))
                .unwrap()
        );
        assert!(
            overnight
                .contains_timestamp(ts("2026-01-16 01:00"))
                .unwrap()
        );
        assert!(
            !overnight
                .contains_timestamp(ts("2026-01-16 02:00"))
                .unwrap()
        );
        let invalid = StrategyDailyWindow {
            enabled: true,
            timezone: "Not/AZone".into(),
            ..Default::default()
        };
        assert!(invalid.contains_timestamp(ts("2026-01-15 09:00")).is_err());
        let equal = StrategyDailyWindow {
            enabled: true,
            end_local: "08:30".into(),
            ..Default::default()
        };
        assert!(equal.validate().is_err());
        assert!(
            !StrategyDailyWindow {
                enabled: true,
                start_local: "bad".into(),
                ..Default::default()
            }
            .contains_timestamp(ts("2026-01-15 09:00"))
            .unwrap_or(false)
        );
    }

    #[test]
    fn outside_window_blocks_entries_but_allows_reductions_and_default_keeps_flatten_off() {
        let window = StrategyDailyWindow {
            enabled: true,
            ..Default::default()
        };
        let outside = ts("2026-01-15 12:00");
        assert!(!window.allows_target(outside, 0, 1).unwrap());
        assert!(!window.allows_target(outside, 1, -1).unwrap());
        assert!(window.allows_target(outside, 1, 0).unwrap());
        assert!(window.allows_target(outside, 2, 1).unwrap());
        assert!(!window.flatten_at_session_end);
        assert_eq!(
            StrategyDailyWindow::constrain_target_outside_window(2, -1),
            Some(0)
        );
        assert_eq!(
            StrategyDailyWindow::constrain_target_outside_window(2, 1),
            Some(1)
        );
        assert_eq!(
            StrategyDailyWindow::constrain_target_outside_window(0, 1),
            None
        );
    }
}
