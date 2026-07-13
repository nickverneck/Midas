use crate::strategies::ema_cross::EmaCrossConfig;
use crate::strategies::hma_angle::HmaAngleConfig;
use crate::strategies::hma_cross::HmaCrossConfig;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const DEFAULT_BLOCKOUT_MINUTES_BEFORE_CLOSE: f64 = 45.0;

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

    pub fn next(self) -> Self {
        match self {
            Self::Native => Self::Lua,
            Self::Lua => Self::MachineLearning,
            Self::MachineLearning => Self::Native,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Native => Self::MachineLearning,
            Self::Lua => Self::Native,
            Self::MachineLearning => Self::Lua,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeStrategyKind {
    HmaAngle,
    EmaCross,
    HmaCross,
}

impl NativeStrategyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HmaAngle => "HMA Angle",
            Self::EmaCross => "EMA Crossover",
            Self::HmaCross => "HMA Crossover",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::HmaAngle => "hma_angle",
            Self::EmaCross => "ema_cross",
            Self::HmaCross => "hma_cross",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::HmaAngle => Self::EmaCross,
            Self::EmaCross => Self::HmaCross,
            Self::HmaCross => Self::HmaAngle,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::HmaAngle => Self::HmaCross,
            Self::EmaCross => Self::HmaAngle,
            Self::HmaCross => Self::EmaCross,
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
    pub native_hma: HmaAngleConfig,
    pub native_ema: EmaCrossConfig,
    pub native_hma_cross: HmaCrossConfig,
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
    pub native_hma: HmaAngleConfig,
    pub native_ema: EmaCrossConfig,
    #[serde(default)]
    pub native_hma_cross: HmaCrossConfig,
    #[serde(default = "default_order_qty")]
    pub order_qty: i32,
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
            native_hma: HmaAngleConfig::default(),
            native_ema: EmaCrossConfig::default(),
            native_hma_cross: HmaCrossConfig::default(),
            order_qty: 1,
        }
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
            native_hma: HmaAngleConfig::default(),
            native_ema: EmaCrossConfig::default(),
            native_hma_cross: HmaCrossConfig::default(),
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
            native_hma: self.native_hma.clone(),
            native_ema: self.native_ema.clone(),
            native_hma_cross: self.native_hma_cross.clone(),
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
        self.native_hma = config.native_hma.clone();
        self.native_ema = config.native_ema.clone();
        self.native_hma_cross = config.native_hma_cross.clone();
        self.order_qty = config.order_qty;
    }
}
