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
            status: "next",
            note: "Preferred backend. Intended to own realtime strategy execution first.",
        },
        StrategyDescriptor {
            name: "Lua",
            priority: "P2",
            status: "scaffold",
            note: "Follows after native Rust. Existing repo Lua runtime can be adapted later.",
        },
        StrategyDescriptor {
            name: "PyTorch / GA-RL",
            priority: "P3",
            status: "backlog",
            note: "Load trained models only after native and Lua paths are stable.",
        },
    ]
}
