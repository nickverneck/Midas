use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerKind {
    Tradovate,
    Ironbeam,
}

impl BrokerKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tradovate => "Tradovate",
            Self::Ironbeam => "Ironbeam",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrokerCapabilities {
    pub replay: bool,
    pub manual_orders: bool,
    pub automated_orders: bool,
    pub native_protection: bool,
}
