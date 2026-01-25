//! Lua scripting runtime with safety limits for backtest strategies.

use std::cell::Cell;
use std::rc::Rc;

use anyhow::{bail, Result};
use mlua::{Error as LuaError, Function, HookTriggers, Lua, Table, Value};

use crate::env::Action;

#[derive(Debug, Clone, Copy)]
pub struct ScriptLimits {
    pub memory_bytes: Option<usize>,
    pub instruction_limit: Option<u64>,
    pub instruction_check_interval: u32,
}

impl Default for ScriptLimits {
    fn default() -> Self {
        Self {
            memory_bytes: Some(64 * 1024 * 1024),
            instruction_limit: Some(5_000_000),
            instruction_check_interval: 10_000,
        }
    }
}

pub struct ScriptRunner {
    lua: Lua,
    has_on_init: bool,
}

impl ScriptRunner {
    pub fn new(script: &str, limits: ScriptLimits) -> Result<Self> {
        let lua = Lua::new();
        sanitize_globals(&lua)?;
        apply_limits(&lua, limits)?;

        lua.load(script)
            .set_name("strategy.lua")
            .exec()?;

        let (has_on_init, has_on_bar) = {
            let globals = lua.globals();
            let init = globals.get::<_, Option<Function>>("on_init")?.is_some();
            let bar = globals.get::<_, Option<Function>>("on_bar")?.is_some();
            (init, bar)
        };

        if !has_on_bar {
            bail!("Lua script must define on_bar(ctx, bar)");
        }

        Ok(Self { lua, has_on_init })
    }

    pub fn lua(&self) -> &Lua {
        &self.lua
    }

    pub fn call_on_init(&self, ctx: &Table) -> Result<()> {
        if !self.has_on_init {
            return Ok(());
        }
        let func: Function = self.lua.globals().get("on_init")?;
        func.call::<_, ()>(ctx.clone())?;
        Ok(())
    }

    pub fn call_on_bar(&self, ctx: &Table, bar: &Table) -> Result<Action> {
        let func: Function = self.lua.globals().get("on_bar")?;
        let value: Value = func.call((ctx.clone(), bar.clone()))?;
        action_from_lua(value)
    }
}

fn sanitize_globals(lua: &Lua) -> Result<()> {
    let globals = lua.globals();
    for key in ["os", "io", "package", "debug", "dofile", "loadfile", "require", "load"] {
        globals.set(key, Value::Nil)?;
    }
    Ok(())
}

fn apply_limits(lua: &Lua, limits: ScriptLimits) -> Result<()> {
    if let Some(memory) = limits.memory_bytes {
        let _ = lua.set_memory_limit(memory);
    }

    if let Some(limit) = limits.instruction_limit {
        let interval = limits.instruction_check_interval.max(1) as u64;
        let counter = Rc::new(Cell::new(0_u64));
        let counter_hook = Rc::clone(&counter);
        let triggers = HookTriggers {
            every_nth_instruction: Some(limits.instruction_check_interval.max(1)),
            ..HookTriggers::default()
        };
        lua.set_hook(triggers, move |_lua, _debug| {
            let next = counter_hook.get().saturating_add(interval);
            if next > limit {
                return Err(LuaError::RuntimeError(
                    "script instruction limit exceeded".to_string(),
                ));
            }
            counter_hook.set(next);
            Ok(())
        });
    }

    Ok(())
}

fn action_from_lua(value: Value) -> Result<Action> {
    match value {
        Value::String(s) => {
            let action = s.to_str()?.to_ascii_lowercase();
            match action.as_str() {
                "buy" => Ok(Action::Buy),
                "sell" => Ok(Action::Sell),
                "hold" => Ok(Action::Hold),
                "revert" | "flip" => Ok(Action::Revert),
                other => bail!("Unknown action: {other}"),
            }
        }
        Value::Nil => bail!("on_bar must return an action string"),
        _ => bail!("on_bar must return an action string"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_on_bar() {
        let script = "function on_init(ctx) end";
        let err = ScriptRunner::new(script, ScriptLimits::default()).unwrap_err();
        assert!(err.to_string().contains("on_bar"));
    }

    #[test]
    fn returns_action() {
        let script = "function on_bar(ctx, bar) return 'buy' end";
        let runner = ScriptRunner::new(script, ScriptLimits::default()).unwrap();
        let lua = runner.lua();
        let ctx = lua.create_table().unwrap();
        let bar = lua.create_table().unwrap();
        let action = runner.call_on_bar(&ctx, &bar).unwrap();
        assert_eq!(action, Action::Buy);
    }
}
