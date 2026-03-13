#[derive(Debug, Clone)]
pub struct StrategyDescriptor {
    pub name: &'static str,
    pub priority: &'static str,
    pub status: &'static str,
    pub note: &'static str,
}

pub fn default_strategy_catalog() -> Vec<StrategyDescriptor> {
    vec![
        StrategyDescriptor {
            name: "Native Rust",
            priority: "P1",
            status: "active",
            note: "HMA Angle strategy is wired for live closed-bar execution and parameter editing.",
        },
        StrategyDescriptor {
            name: "Lua",
            priority: "P2",
            status: "scaffold",
            note: "Editor and file loading are in place; live execution wiring still follows native Rust.",
        },
        StrategyDescriptor {
            name: "PyTorch / GA-RL",
            priority: "P3",
            status: "backlog",
            note: "Load trained models only after native and Lua paths are stable.",
        },
    ]
}
