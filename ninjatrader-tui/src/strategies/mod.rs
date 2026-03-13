pub mod hma_angle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategySignal {
    Hold,
    EnterLong,
    EnterShort,
    ExitLongOnShortSignal,
}

impl StrategySignal {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hold => "Hold",
            Self::EnterLong => "Buy",
            Self::EnterShort => "Sell",
            Self::ExitLongOnShortSignal => "Exit Long",
        }
    }
}

pub fn side_from_signed_qty(qty: i32) -> Option<PositionSide> {
    if qty > 0 {
        Some(PositionSide::Long)
    } else if qty < 0 {
        Some(PositionSide::Short)
    } else {
        None
    }
}
