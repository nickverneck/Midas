//! Bounded, deterministic Lua callback evaluator used only by cached replay.

use mlua::{Error as LuaError, Function, HookTriggers, Lua, Value};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::broker::{Bar, MarketHistoryUpdate};
use crate::strategies::StrategySignal;
use crate::strategy::LuaStrategyConfig;

const MAX_SOURCE_BYTES: usize = 64 * 1024;
const MAX_MEMORY_BYTES: usize = 32 * 1024 * 1024;
const MAX_INSTRUCTIONS: u64 = 1_000_000;

pub(crate) fn validate_source(config: &LuaStrategyConfig) -> anyhow::Result<()> {
    LuaReplayRuntime::initialize(config).map(|_| ())
}

#[cfg(test)]
fn evaluate_bar(
    config: &LuaStrategyConfig,
    bar: &Bar,
    index: usize,
    current_qty: i32,
) -> anyhow::Result<StrategySignal> {
    let mut runtime = LuaReplayRuntime::new(config.clone());
    runtime.evaluate_bar(bar, index, current_qty)
}

/// One isolated interpreter per replay execution run. Cloning resets VM state,
/// so replay children and service clones never share mutable Lua globals.
pub(crate) struct LuaReplayRuntime {
    config: LuaStrategyConfig,
    state: std::sync::Mutex<RuntimeState>,
}

struct RuntimeState {
    lua: Option<Lua>,
    bootstrap_first: Option<BarIdentity>,
    required_bootstrap_first: Option<BarIdentity>,
    last_bar: Option<BarIdentity>,
    last_signal: Option<StrategySignal>,
    last_global_index: Option<usize>,
    next_global_index: usize,
    previous_window: Vec<BarIdentity>,
    last_input_index: Option<usize>,
    last_source_sequence: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BarIdentity {
    ts_ns: i64,
    open: u64,
    high: u64,
    low: u64,
    close: u64,
    volume: Option<u64>,
}

impl From<&Bar> for BarIdentity {
    fn from(bar: &Bar) -> Self {
        Self {
            ts_ns: bar.ts_ns,
            open: bar.open.to_bits(),
            high: bar.high.to_bits(),
            low: bar.low.to_bits(),
            close: bar.close.to_bits(),
            volume: bar.volume.map(f64::to_bits),
        }
    }
}

impl std::fmt::Debug for LuaReplayRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuaReplayRuntime").finish_non_exhaustive()
    }
}
impl Clone for LuaReplayRuntime {
    fn clone(&self) -> Self {
        let required_bootstrap_first = self
            .state
            .lock()
            .ok()
            .and_then(|state| state.bootstrap_first);
        Self {
            config: self.config.clone(),
            state: std::sync::Mutex::new(RuntimeState {
                lua: None,
                bootstrap_first: None,
                required_bootstrap_first,
                last_bar: None,
                last_signal: None,
                last_global_index: None,
                next_global_index: 0,
                previous_window: Vec::new(),
                last_input_index: None,
                last_source_sequence: None,
            }),
        }
    }
}
impl Default for LuaReplayRuntime {
    fn default() -> Self {
        Self::new(LuaStrategyConfig::default())
    }
}

impl LuaReplayRuntime {
    pub(crate) fn new(config: LuaStrategyConfig) -> Self {
        Self {
            config,
            state: std::sync::Mutex::new(RuntimeState {
                lua: None,
                bootstrap_first: None,
                required_bootstrap_first: None,
                last_bar: None,
                last_signal: None,
                last_global_index: None,
                next_global_index: 0,
                previous_window: Vec::new(),
                last_input_index: None,
                last_source_sequence: None,
            }),
        }
    }
    pub(crate) fn reset(&mut self, config: LuaStrategyConfig) {
        *self = Self::new(config);
    }
    pub(crate) fn ensure_config(&mut self, config: &LuaStrategyConfig) {
        if &self.config != config {
            self.reset(config.clone());
        }
    }

    pub(crate) fn ensure_bootstrap_history(
        &self,
        expected_history_rows: usize,
        available_rows: usize,
    ) -> anyhow::Result<()> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Lua replay runtime lock poisoned"))?;
        if state.lua.is_none() && expected_history_rows > available_rows {
            anyhow::bail!(
                "Lua replay bootstrap history is truncated: expected {expected_history_rows} rows but only {available_rows} are retained"
            );
        }
        Ok(())
    }

    fn initialize(config: &LuaStrategyConfig) -> anyhow::Result<Lua> {
        anyhow::ensure!(
            config.schema_version == 1,
            "unsupported Lua strategy schema version"
        );
        anyhow::ensure!(
            !config.source.trim().is_empty(),
            "Lua strategy source is empty"
        );
        anyhow::ensure!(
            config.source.len() <= MAX_SOURCE_BYTES,
            "Lua strategy source exceeds 64 KiB"
        );
        anyhow::ensure!(
            (64 * 1024..=MAX_MEMORY_BYTES).contains(&config.memory_limit_bytes),
            "Lua memory limit must be between 64 KiB and 32 MiB"
        );
        anyhow::ensure!(
            (1..=MAX_INSTRUCTIONS).contains(&config.instruction_limit),
            "Lua instruction limit must be between 1 and 1,000,000"
        );
        let lua = Lua::new();
        lua.set_memory_limit(config.memory_limit_bytes)?;
        {
            let globals = lua.globals();
            for key in [
                "os",
                "io",
                "package",
                "debug",
                "coroutine",
                "string",
                "table",
                "utf8",
                "dofile",
                "loadfile",
                "require",
                "load",
                "pcall",
                "xpcall",
                "collectgarbage",
                "print",
                "warn",
                "pairs",
                "next",
            ] {
                globals.set(key, Value::Nil)?;
            }
            let get_metatable: Function = globals.get("getmetatable")?;
            if let Value::Table(string_metatable) = get_metatable.call::<_, Value>("")? {
                string_metatable.set("__index", Value::Nil)?;
            }
            let math: mlua::Table = globals.get("math")?;
            math.set("random", Value::Nil)?;
            math.set("randomseed", Value::Nil)?;
            let load_count = Arc::new(AtomicU64::new(0));
            let count_hook = Arc::clone(&load_count);
            let limit = config.instruction_limit;
            lua.set_hook(
                HookTriggers {
                    every_nth_instruction: Some(100),
                    ..HookTriggers::default()
                },
                move |_, _| {
                    let next = count_hook
                        .fetch_add(100, Ordering::Relaxed)
                        .saturating_add(100);
                    if next > limit {
                        return Err(LuaError::RuntimeError(
                            "script instruction limit exceeded".to_string(),
                        ));
                    }
                    Ok(())
                },
            );
            lua.load(&config.source)
                .set_name("replay-strategy.lua")
                .exec()?;
            let _: Function = globals.get("on_bar")?;
        }
        Ok(lua)
    }

    fn evaluate_locked(
        &self,
        state: &mut RuntimeState,
        bar: &Bar,
        index: usize,
        current_qty: i32,
        warmup: bool,
    ) -> anyhow::Result<StrategySignal> {
        if state.lua.is_none() {
            state.lua = Some(Self::initialize(&self.config)?);
        }
        let lua = state.lua.as_ref().expect("initialized Lua VM");
        let count = Arc::new(AtomicU64::new(0));
        let count_hook = Arc::clone(&count);
        let instruction_limit = self.config.instruction_limit;
        lua.set_hook(
            HookTriggers {
                every_nth_instruction: Some(100),
                ..HookTriggers::default()
            },
            move |_, _| {
                let next = count_hook
                    .fetch_add(100, Ordering::Relaxed)
                    .saturating_add(100);
                if next > instruction_limit {
                    return Err(LuaError::RuntimeError(
                        "script instruction limit exceeded".to_string(),
                    ));
                }
                Ok(())
            },
        );
        let globals = lua.globals();
        let func: Function = globals.get("on_bar")?;
        let ctx = lua.create_table()?;
        ctx.set("bar_index", index as u64)?;
        ctx.set("is_warmup", warmup)?;
        if warmup {
            let mt = lua.create_table()?;
            mt.set(
                "__index",
                lua.create_function(|_, (_table, key): (Value, Value)| {
                    if matches!(key, Value::String(ref value) if value.to_str().ok() == Some("position_qty")) {
                        return Err(LuaError::RuntimeError(
                            "position_qty is unavailable while rebuilding Lua replay state".into(),
                        ));
                    }
                    Ok(Value::Nil)
                })?,
            )?;
            ctx.set_metatable(Some(mt));
        } else {
            ctx.set("position_qty", current_qty)?;
        }
        let data = lua.create_table()?;
        data.set("timestamp_ns", bar.ts_ns)?;
        data.set("open", bar.open)?;
        data.set("high", bar.high)?;
        data.set("low", bar.low)?;
        data.set("close", bar.close)?;
        data.set("volume", bar.volume)?;
        let result: Value = func.call((ctx, data))?;
        let action = match result {
            Value::String(value) => value.to_str()?.to_ascii_lowercase(),
            _ => anyhow::bail!("on_bar(ctx, bar) must return 'buy', 'sell', or 'hold'"),
        };
        let signal = match action.as_str() {
            "buy" if current_qty <= 0 => StrategySignal::EnterLong,
            "sell" if current_qty >= 0 => StrategySignal::EnterShort,
            "buy" | "sell" | "hold" => StrategySignal::Hold,
            _ => anyhow::bail!(
                "unsupported Lua action '{action}'; allowed actions are buy, sell, hold"
            ),
        };
        Ok(signal)
    }

    pub(crate) fn evaluate_bar(
        &mut self,
        bar: &Bar,
        index: usize,
        current_qty: i32,
    ) -> anyhow::Result<StrategySignal> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Lua replay runtime lock poisoned"))?;
        let signal = self.evaluate_locked(&mut state, bar, index, current_qty, false)?;
        let identity = BarIdentity::from(bar);
        state.bootstrap_first.get_or_insert(identity);
        state.last_bar = Some(identity);
        state.last_signal = Some(signal);
        state.last_global_index = Some(index);
        state.next_global_index = index.saturating_add(1);
        Ok(signal)
    }

    pub(crate) fn evaluate_with_prefix(
        &mut self,
        bars: &[Bar],
        index: usize,
        current_qty: i32,
        source_sequence: Option<u64>,
        update_kind: MarketHistoryUpdate,
    ) -> anyhow::Result<StrategySignal> {
        anyhow::ensure!(index < bars.len(), "Lua replay bar index out of range");
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Lua replay runtime lock poisoned"))?;
        let target_identity = BarIdentity::from(&bars[index]);
        if state.lua.is_none() {
            anyhow::ensure!(
                state.required_bootstrap_first.is_none_or(|expected| bars
                    .first()
                    .is_some_and(|first| BarIdentity::from(first) == expected)),
                "Lua replay clone cannot resume: original bootstrap bar was evicted from the retained window"
            );
            state.bootstrap_first = bars.first().map(BarIdentity::from);
            state.lua = Some(Self::initialize(&self.config)?);
            for (i, bar) in bars.iter().take(index).enumerate() {
                let signal = self.evaluate_locked(&mut state, bar, i, 0, true)?;
                state.last_bar = Some(BarIdentity::from(bar));
                state.last_signal = Some(signal);
                state.last_global_index = Some(i);
                state.next_global_index = i + 1;
            }
            state.previous_window = bars.iter().take(index).map(BarIdentity::from).collect();
        }

        let current_window: Vec<_> = bars.iter().take(index + 1).map(BarIdentity::from).collect();
        let mut signal = state.last_signal.unwrap_or(StrategySignal::Hold);
        if state.last_bar.is_none() {
            // Initial callback for a new one-bar series.
            signal = self.evaluate_locked(&mut state, &bars[index], 0, current_qty, false)?;
            state.bootstrap_first = Some(target_identity);
            state.last_bar = Some(target_identity);
            state.last_signal = Some(signal);
            state.last_global_index = Some(0);
            state.next_global_index = 1;
        } else {
            if let Some(last) = state.last_bar {
                if state.last_input_index == Some(index)
                    && target_identity.ts_ns == last.ts_ns
                    && target_identity != last
                    && update_kind != MarketHistoryUpdate::Append
                {
                    anyhow::bail!(
                        "Lua replay cannot process a revised forming bar; closed-bar append identity is required"
                    );
                }
            }
            let old_window = &state.previous_window;
            let same_input = current_window == *old_window
                && state.last_input_index == Some(index)
                && (update_kind == MarketHistoryUpdate::Unchanged
                    || source_sequence.is_some() && state.last_source_sequence == source_sequence);
            if same_input {
                return Ok(signal);
            }
            if current_window == *old_window && state.last_input_index == Some(index) {
                let confirmed_append = update_kind == MarketHistoryUpdate::Append
                    && source_sequence.is_some_and(|sequence| {
                        state
                            .last_source_sequence
                            .is_some_and(|previous| sequence > previous)
                    });
                anyhow::ensure!(
                    confirmed_append,
                    "Lua replay cannot identify an identical appended bar without an advanced append sequence"
                );
                let global_index = state.next_global_index;
                signal = self.evaluate_locked(
                    &mut state,
                    &bars[index],
                    global_index,
                    current_qty,
                    false,
                )?;
                state.last_bar = Some(target_identity);
                state.last_signal = Some(signal);
                state.last_global_index = Some(global_index);
                state.next_global_index = global_index.saturating_add(1);
                state.previous_window = current_window;
                state.last_input_index = Some(index);
                state.last_source_sequence = source_sequence;
                return Ok(signal);
            }
            let overlap = (1..=old_window.len().min(current_window.len()))
                .rev()
                .find(|len| old_window[old_window.len() - len..] == current_window[..*len]);
            let overlap = overlap.ok_or_else(|| anyhow::anyhow!(
                "Lua replay retained window no longer overlaps the last processed bar; state cannot be reconstructed safely"
            ))?;
            let anchor_index = overlap - 1;
            anyhow::ensure!(
                anchor_index < index,
                "Lua replay target does not advance beyond its stable bar anchor"
            );
            let unseen = &bars[anchor_index + 1..=index];
            for (offset, bar) in unseen.iter().enumerate() {
                let global_index = state.next_global_index;
                // Any unseen bar before the requested target is historical
                // warmup: its position cannot be inferred from today's account.
                let warmup = offset + 1 < unseen.len();
                let qty = if warmup { 0 } else { current_qty };
                signal = self.evaluate_locked(&mut state, bar, global_index, qty, warmup)?;
                state.last_bar = Some(BarIdentity::from(bar));
                state.last_signal = Some(signal);
                state.last_global_index = Some(global_index);
                state.next_global_index = global_index.saturating_add(1);
            }
        }
        state.previous_window = current_window;
        state.last_input_index = Some(index);
        state.last_source_sequence = source_sequence;
        let _ = update_kind; // Corrections never overlap the processed prefix and fail above.
        Ok(signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(source: &str) -> LuaStrategyConfig {
        LuaStrategyConfig {
            source: source.into(),
            ..Default::default()
        }
    }
    fn bar() -> Bar {
        bar_at(1)
    }
    fn bar_at(ts_ns: i64) -> Bar {
        Bar {
            ts_ns,
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: Some(4.0),
        }
    }

    #[test]
    fn validation_does_not_invoke_the_callback_on_fabricated_market_data() {
        let config = cfg(
            "function on_bar(ctx, bar) if bar.close <= 0 then error('invalid price') end return 'hold' end",
        );
        validate_source(&config).unwrap();
        assert_eq!(
            evaluate_bar(&config, &bar(), 0, 0).unwrap(),
            StrategySignal::Hold
        );
        assert!(validate_source(&cfg("on_bar = 1")).is_err());
        assert!(validate_source(&cfg("function on_bar(")).is_err());
    }

    #[test]
    fn evaluates_deterministic_callback_and_maps_targets() {
        let script = "function on_bar(ctx, bar) if bar.close > bar.open then return 'buy' end return 'hold' end";
        assert_eq!(
            evaluate_bar(&cfg(script), &bar(), 0, 0).unwrap(),
            StrategySignal::EnterLong
        );
        assert_eq!(
            evaluate_bar(&cfg(script), &bar(), 0, 1).unwrap(),
            StrategySignal::Hold
        );
    }

    #[test]
    fn blocks_io_and_runaway_scripts() {
        let err = evaluate_bar(
            &cfg("function on_bar(ctx, bar) return io.open('/tmp/x') end"),
            &bar(),
            0,
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("attempt to index"));
        let mut infinite = cfg("function on_bar(ctx, bar) while true do end end");
        infinite.instruction_limit = 1_000;
        let err = evaluate_bar(&infinite, &bar(), 0, 0).unwrap_err();
        assert!(err.to_string().contains("instruction limit"));
    }

    #[test]
    fn removes_quota_escape_hatches_and_unbounded_string_tables() {
        let script = r#"
            function on_bar(ctx, bar)
                if pcall ~= nil or xpcall ~= nil or coroutine ~= nil
                    or string ~= nil or table ~= nil or pairs ~= nil
                    or next ~= nil or collectgarbage ~= nil
                    or math.random ~= nil or getmetatable('').__index ~= nil then
                    return 'buy'
                end
                return 'hold'
            end
        "#;
        assert_eq!(
            evaluate_bar(&cfg(script), &bar(), 0, 0).unwrap(),
            StrategySignal::Hold
        );
    }

    #[test]
    fn state_accumulates_and_resumed_or_cloned_runs_replay_prefix() {
        let config = cfg(r#"
            count = 0
            function on_bar(ctx, bar)
                count = count + 1
                if count >= 3 then return 'buy' end
                return 'hold'
            end
        "#);
        let bars = [bar_at(1), bar_at(2), bar_at(3), bar_at(4)];
        let mut continuous = LuaReplayRuntime::new(config.clone());
        assert_eq!(
            continuous
                .evaluate_with_prefix(&bars, 0, 0, Some(1), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::Hold
        );
        assert_eq!(
            continuous
                .evaluate_with_prefix(&bars, 2, 0, Some(2), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
        let mut resumed = continuous.clone();
        assert_eq!(
            resumed
                .evaluate_with_prefix(&bars, 3, 0, Some(3), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
        // A sibling starts with independent globals and replays its own prefix.
        let mut sibling = LuaReplayRuntime::new(config);
        assert_eq!(
            sibling
                .evaluate_with_prefix(&bars, 1, 0, Some(4), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::Hold
        );
        assert_eq!(
            sibling
                .evaluate_with_prefix(&bars, 2, 0, Some(5), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
    }

    #[test]
    fn resumed_state_fails_closed_when_history_needs_unavailable_position() {
        let config = cfg(
            "function on_bar(ctx, bar) if ctx.position_qty == 0 then return 'buy' end return 'hold' end",
        );
        let mut resumed = LuaReplayRuntime::new(config);
        let error = resumed
            .evaluate_with_prefix(
                &[bar_at(1), bar_at(2)],
                1,
                1,
                Some(1),
                MarketHistoryUpdate::Append,
            )
            .unwrap_err();
        assert!(error.to_string().contains("unavailable"), "{error:?}");
    }

    #[test]
    fn rolling_window_requires_overlap_and_never_rebuilds_from_suffix() {
        let config = cfg("function on_bar(ctx, bar) return 'hold' end");
        let mut runtime = LuaReplayRuntime::new(config);
        let initial = [bar_at(10), bar_at(20)];
        runtime
            .evaluate_with_prefix(&initial, 1, 0, Some(1), MarketHistoryUpdate::Append)
            .unwrap();
        let overlapping = [bar_at(20), bar_at(30)];
        runtime
            .evaluate_with_prefix(&overlapping, 1, 0, Some(2), MarketHistoryUpdate::Append)
            .unwrap();
        let truncated = [bar_at(40), bar_at(50)];
        let error = runtime
            .evaluate_with_prefix(&truncated, 1, 0, Some(3), MarketHistoryUpdate::Append)
            .unwrap_err();
        assert!(error.to_string().contains("no longer overlaps"));
    }

    #[test]
    fn cloned_run_refuses_prefix_rebuild_after_bootstrap_bar_eviction() {
        let mut original =
            LuaReplayRuntime::new(cfg("function on_bar(ctx, bar) return 'hold' end"));
        let bars = [bar_at(10), bar_at(20)];
        original
            .evaluate_with_prefix(&bars, 1, 0, Some(1), MarketHistoryUpdate::Append)
            .unwrap();
        let mut cloned = original.clone();
        let truncated = [bar_at(20), bar_at(30)];
        let error = cloned
            .evaluate_with_prefix(&truncated, 1, 0, Some(2), MarketHistoryUpdate::Append)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("original bootstrap bar was evicted")
        );
    }

    #[test]
    fn repeated_identical_bars_are_distinct_by_series_position_but_same_event_is_idempotent() {
        let config = cfg(r#"
            count = 0
            function on_bar(ctx, bar)
                count = count + 1
                if count >= 2 then return 'buy' end
                return 'hold'
            end
        "#);
        let repeated = [bar_at(10), bar_at(10)];
        let mut runtime = LuaReplayRuntime::new(config);
        assert_eq!(
            runtime
                .evaluate_with_prefix(&repeated, 0, 0, Some(7), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::Hold
        );
        // The exact same update is idempotent.
        assert_eq!(
            runtime
                .evaluate_with_prefix(&repeated, 0, 0, Some(7), MarketHistoryUpdate::Unchanged)
                .unwrap(),
            StrategySignal::Hold
        );
        // Equal OHLCV at the next sequence position is a distinct bar.
        assert_eq!(
            runtime
                .evaluate_with_prefix(&repeated, 1, 0, Some(8), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
    }

    #[test]
    fn fixed_size_identical_window_advances_only_on_new_append_sequence() {
        let config = cfg(r#"
            count = 0
            function on_bar(ctx, bar)
                count = count + 1
                if count == 4 then return 'buy' end
                return 'hold'
            end
        "#);
        let bars = [bar_at(10), bar_at(10), bar_at(10)];
        let mut runtime = LuaReplayRuntime::new(config);
        assert_eq!(
            runtime
                .evaluate_with_prefix(&bars, 2, 0, Some(20), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::Hold
        );
        assert_eq!(
            runtime
                .evaluate_with_prefix(&bars, 2, 0, Some(21), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
        // Reprocessing the same append is idempotent, even though all bar
        // values and the fixed-size retained window are identical.
        assert_eq!(
            runtime
                .evaluate_with_prefix(&bars, 2, 0, Some(21), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::EnterLong
        );
        // A later real append advances the script exactly once again.
        assert_eq!(
            runtime
                .evaluate_with_prefix(&bars, 2, 0, Some(22), MarketHistoryUpdate::Append)
                .unwrap(),
            StrategySignal::Hold
        );
    }

    #[test]
    fn same_index_revision_fails_closed() {
        let mut runtime = LuaReplayRuntime::new(cfg("function on_bar(ctx, bar) return 'hold' end"));
        let original = [bar_at(1), bar_at(2)];
        runtime
            .evaluate_with_prefix(&original, 1, 0, Some(1), MarketHistoryUpdate::Append)
            .unwrap();
        let mut revised = original.clone();
        revised[1].close += 0.25;
        let error = runtime
            .evaluate_with_prefix(&revised, 1, 0, Some(2), MarketHistoryUpdate::Correction)
            .unwrap_err();
        assert!(error.to_string().contains("revised forming bar"));
    }

    #[test]
    fn missed_intermediate_bars_do_not_receive_the_current_position() {
        let config = cfg(
            "function on_bar(ctx, bar) if ctx.position_qty == nil then return 'hold' end return 'buy' end",
        );
        let bars = [bar_at(1), bar_at(2), bar_at(3)];
        let mut runtime = LuaReplayRuntime::new(config);
        runtime
            .evaluate_with_prefix(&bars, 0, 0, Some(1), MarketHistoryUpdate::Append)
            .unwrap();
        let error = runtime
            .evaluate_with_prefix(&bars, 2, 1, Some(2), MarketHistoryUpdate::Append)
            .unwrap_err();
        assert!(error.to_string().contains("position_qty is unavailable"));
    }
}
