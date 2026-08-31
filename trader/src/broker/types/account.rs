use super::TradeMarkerSide;
use crate::strategy::ExecutionStateSnapshot;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub id: i64,
    pub name: String,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineHistoryFill {
    pub fill_id: i64,
    pub order_id: i64,
    pub ts_ns: i64,
    pub side: TradeMarkerSide,
    pub qty: i32,
    pub price: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineHistorySnapshot {
    pub run_id: String,
    pub started_at_utc: DateTime<Utc>,
    /// Last broker/market refresh used to build this snapshot. Optional for
    /// compatibility with older in-memory/replayed snapshots.
    #[serde(default)]
    pub updated_at_utc: Option<DateTime<Utc>>,
    pub account_id: i64,
    pub account_name: String,
    pub contract_id: i64,
    pub contract_name: String,
    pub position_qty: i32,
    pub average_entry_price: Option<f64>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees: f64,
    pub wins: usize,
    pub losses: usize,
    pub fills: Vec<EngineHistoryFill>,
}

/// Compact, read-only view of an engine's already-held history.  Inspection
/// clients such as `trader list` need the aggregates, not the complete fill
/// payload. Keeping this separate prevents a status query from copying and
/// serializing the full history while the engine is executing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineHistorySummary {
    pub run_id: String,
    pub started_at_utc: DateTime<Utc>,
    #[serde(default)]
    pub updated_at_utc: Option<DateTime<Utc>>,
    pub account_id: i64,
    pub account_name: String,
    pub contract_id: i64,
    pub contract_name: String,
    pub position_qty: i32,
    pub average_entry_price: Option<f64>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees: f64,
    pub wins: usize,
    pub losses: usize,
    pub fill_count: usize,
}

impl EngineHistorySnapshot {
    pub fn summary(&self) -> EngineHistorySummary {
        EngineHistorySummary {
            run_id: self.run_id.clone(),
            started_at_utc: self.started_at_utc,
            updated_at_utc: self.updated_at_utc,
            account_id: self.account_id,
            account_name: self.account_name.clone(),
            contract_id: self.contract_id,
            contract_name: self.contract_name.clone(),
            position_qty: self.position_qty,
            average_entry_price: self.average_entry_price,
            realized_pnl: self.realized_pnl,
            unrealized_pnl: self.unrealized_pnl,
            fees: self.fees,
            wins: self.wins,
            losses: self.losses,
            fill_count: self.fills.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub account_id: i64,
    pub account_name: String,
    pub balance: Option<f64>,
    pub cash_balance: Option<f64>,
    pub net_liq: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    /// Cumulative broker-reported fees/commissions, when the broker exposes
    /// them as explicit fill/account data. This is a positive magnitude.
    pub fees: Option<f64>,
    pub intraday_margin: Option<f64>,
    pub open_position_qty: Option<f64>,
    pub market_position_qty: Option<f64>,
    pub market_entry_price: Option<f64>,
    pub selected_contract_take_profit_price: Option<f64>,
    pub selected_contract_stop_price: Option<f64>,
    pub raw_account: Option<Value>,
    pub raw_risk: Option<Value>,
    pub raw_cash: Option<Value>,
    pub raw_positions: Vec<Value>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencySnapshot {
    pub rest_rtt_ms: Option<u64>,
    pub last_order_ack_ms: Option<u64>,
    pub last_order_seen_ms: Option<u64>,
    pub last_exec_report_ms: Option<u64>,
    pub last_fill_ms: Option<u64>,
    pub last_signal_submit_ms: Option<u64>,
    pub last_signal_seen_ms: Option<u64>,
    pub last_signal_ack_ms: Option<u64>,
    pub last_signal_fill_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeSnapshot {
    pub tag: String,
    pub captured_at_utc: DateTime<Utc>,
    pub execution_state: ExecutionStateSnapshot,
    pub latency: LatencySnapshot,
    pub order_submit_in_flight: bool,
    pub protection_sync_in_flight: bool,
    pub tracker_order_id: Option<i64>,
    pub tracker_order_is_active: bool,
    pub tracker_order_strategy_id: Option<i64>,
    pub tracker_strategy_has_live_orders: bool,
    pub tracker_within_strategy_grace: bool,
    pub tracked_order_strategy_id: Option<i64>,
    pub broker_order_strategy_id: Option<i64>,
    pub broker_order_strategy_status: Option<String>,
    pub broker_strategy_entry_order_qty: Option<i32>,
    pub broker_strategy_bracket_qtys: Vec<i32>,
    pub selected_working_orders: Vec<ExecutionProbeOrder>,
    pub linked_active_orders: Vec<ExecutionProbeOrder>,
    pub managed_protection: Option<ExecutionProbeManagedProtection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeOrder {
    pub order_id: Option<i64>,
    pub order_strategy_id: Option<i64>,
    pub cl_ord_id: Option<String>,
    pub order_type: Option<String>,
    pub action: Option<String>,
    pub order_qty: Option<i32>,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeManagedProtection {
    pub signed_qty: i32,
    pub take_profit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub take_profit_order_id: Option<i64>,
    pub stop_order_id: Option<i64>,
    pub take_profit_cl_ord_id: Option<String>,
    pub stop_cl_ord_id: Option<String>,
}
