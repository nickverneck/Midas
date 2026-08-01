//! Optional post-replay margin liquidation overlay.
//!
//! This is intentionally an accounting/research overlay.  It does not alter
//! the immutable replay ledger or the live broker path.  Once the configured
//! equity buffer is breached, open positions are flattened at the current
//! fill price plus configured slippage and later strategy fills are ignored,
//! which models a broker stopping the account after liquidation.

use super::fees::ReplayFeeSchedule;
use super::risk::ReplayMarginConfig;
use crate::broker::ReplayExecutionFill;
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) const REPLAY_LIQUIDATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayLiquidationConfig {
    pub(crate) margin: ReplayMarginConfig,
    pub(crate) slippage_points: f64,
}

impl Default for ReplayLiquidationConfig {
    fn default() -> Self {
        Self {
            margin: ReplayMarginConfig::default(),
            slippage_points: 0.0,
        }
    }
}

impl ReplayLiquidationConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.margin.validate()?;
        if !self.slippage_points.is_finite() || self.slippage_points < 0.0 {
            bail!("liquidation slippage_points must be finite and non-negative");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayLiquidationEvent {
    pub(crate) timestamp_ns: i64,
    pub(crate) account_id: i64,
    pub(crate) contract_id: i64,
    pub(crate) contract_name: String,
    pub(crate) side: String,
    pub(crate) quantity: f64,
    pub(crate) reference_price: f64,
    pub(crate) liquidation_price: f64,
    pub(crate) value_per_point: Option<f64>,
    pub(crate) gross_pnl: f64,
    pub(crate) fees: f64,
    pub(crate) net_pnl: f64,
    pub(crate) equity_before: f64,
    pub(crate) margin_requirement: f64,
    pub(crate) safety_buffer: f64,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayLiquidationAnalysis {
    pub(crate) schema_version: u32,
    pub(crate) fee_scenario: String,
    pub(crate) config: ReplayLiquidationConfig,
    pub(crate) triggered: bool,
    pub(crate) first_breach_timestamp_ns: Option<i64>,
    pub(crate) event_count: usize,
    pub(crate) ignored_fill_count: usize,
    pub(crate) equity_after: f64,
    pub(crate) net_pnl_after: f64,
    pub(crate) events: Vec<ReplayLiquidationEvent>,
}

impl Default for ReplayLiquidationAnalysis {
    fn default() -> Self {
        Self {
            schema_version: REPLAY_LIQUIDATION_SCHEMA_VERSION,
            fee_scenario: String::new(),
            config: ReplayLiquidationConfig::default(),
            triggered: false,
            first_breach_timestamp_ns: None,
            event_count: 0,
            ignored_fill_count: 0,
            equity_after: 0.0,
            net_pnl_after: 0.0,
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct OpenPosition {
    qty: f64,
    avg_price: f64,
    contract_name: String,
    value_per_point: Option<f64>,
}

#[derive(Debug, Clone)]
struct LastPrice {
    price: f64,
}

pub(crate) fn simulate_replay_liquidation(
    fills: &[ReplayExecutionFill],
    initial_capital: f64,
    _started_at_utc: DateTime<Utc>,
    fee_schedule: &ReplayFeeSchedule,
    fee_scenario: &str,
    config: &ReplayLiquidationConfig,
) -> Result<ReplayLiquidationAnalysis> {
    config.validate()?;
    fee_schedule.validate()?;
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        bail!("initial capital must be finite and greater than zero");
    }

    let mut positions = BTreeMap::<(i64, i64), OpenPosition>::new();
    let mut last_prices = BTreeMap::<(i64, i64), LastPrice>::new();
    let mut cumulative_net_pnl = 0.0;
    let mut events = Vec::new();
    let mut first_breach_timestamp_ns = None;
    let mut ignored_fill_count: usize = 0;
    let mut triggered = false;
    for fill in fills {
        if triggered {
            ignored_fill_count = ignored_fill_count.saturating_add(1);
            continue;
        }
        let quantity = fill.quantity.abs();
        if quantity <= f64::EPSILON {
            continue;
        }
        let signed = if fill.side.eq_ignore_ascii_case("buy") {
            quantity
        } else {
            -quantity
        };
        let key = (fill.account_id, fill.contract_id);
        apply_fill_to_position(
            &mut positions,
            key,
            signed,
            fill.price,
            fill.contract_name.clone(),
            fill.value_per_point,
        );
        last_prices.insert(key, LastPrice { price: fill.price });
        cumulative_net_pnl += fill.gross_realized_pnl_delta.unwrap_or_default()
            - fee_schedule.fee_for_quantity(quantity);

        let gross_position = positions
            .values()
            .map(|position| position.qty.abs())
            .sum::<f64>();
        let margin_requirement = gross_position * config.margin.margin_per_contract;
        let safety_buffer = config.margin.buffer_for_margin(margin_requirement);
        let equity =
            initial_capital + cumulative_net_pnl + mark_to_market_pnl(&positions, &last_prices);
        if gross_position > f64::EPSILON && equity < margin_requirement + safety_buffer {
            triggered = true;
            first_breach_timestamp_ns = Some(fill.fill_timestamp_ns);
            let open_positions = std::mem::take(&mut positions);
            for ((account_id, contract_id), position) in open_positions {
                let quantity = position.qty.abs();
                let reference_price = last_prices
                    .get(&(account_id, contract_id))
                    .map(|price| price.price)
                    .unwrap_or(fill.price);
                let liquidation_price = if position.qty > 0.0 {
                    reference_price - config.slippage_points
                } else {
                    reference_price + config.slippage_points
                };
                let points = if position.qty > 0.0 {
                    liquidation_price - position.avg_price
                } else {
                    position.avg_price - liquidation_price
                };
                let gross_pnl = position
                    .value_per_point
                    .map(|value| points * quantity * value)
                    .unwrap_or_default();
                let fees = fee_schedule.fee_for_quantity(quantity);
                let net_pnl = gross_pnl - fees;
                cumulative_net_pnl += net_pnl;
                events.push(ReplayLiquidationEvent {
                    timestamp_ns: fill.fill_timestamp_ns,
                    account_id,
                    contract_id,
                    contract_name: position.contract_name,
                    side: if position.qty > 0.0 {
                        "Sell".to_string()
                    } else {
                        "Buy".to_string()
                    },
                    quantity,
                    reference_price,
                    liquidation_price,
                    value_per_point: position.value_per_point,
                    gross_pnl,
                    fees,
                    net_pnl,
                    equity_before: equity,
                    margin_requirement,
                    safety_buffer,
                    reason: "margin_liquidation".to_string(),
                });
            }
        }
    }

    Ok(ReplayLiquidationAnalysis {
        schema_version: REPLAY_LIQUIDATION_SCHEMA_VERSION,
        fee_scenario: fee_scenario.to_string(),
        config: config.clone(),
        triggered,
        first_breach_timestamp_ns,
        event_count: events.len(),
        ignored_fill_count,
        equity_after: initial_capital + cumulative_net_pnl,
        net_pnl_after: cumulative_net_pnl,
        events,
    })
}

fn mark_to_market_pnl(
    positions: &BTreeMap<(i64, i64), OpenPosition>,
    last_prices: &BTreeMap<(i64, i64), LastPrice>,
) -> f64 {
    positions
        .iter()
        .filter_map(|(key, position)| {
            let value_per_point = position.value_per_point?;
            let price = last_prices.get(key)?.price;
            let points = if position.qty > 0.0 {
                price - position.avg_price
            } else {
                position.avg_price - price
            };
            Some(points * position.qty.abs() * value_per_point)
        })
        .sum()
}

fn apply_fill_to_position(
    positions: &mut BTreeMap<(i64, i64), OpenPosition>,
    key: (i64, i64),
    signed: f64,
    price: f64,
    contract_name: String,
    value_per_point: Option<f64>,
) {
    let Some(mut position) = positions.remove(&key) else {
        positions.insert(
            key,
            OpenPosition {
                qty: signed,
                avg_price: price,
                contract_name,
                value_per_point,
            },
        );
        return;
    };
    if position.qty.signum() == signed.signum() {
        let next_abs = position.qty.abs() + signed.abs();
        position.avg_price = (position.avg_price * position.qty.abs() + price * signed.abs())
            / next_abs.max(f64::EPSILON);
        position.qty = position.qty + signed;
        position.value_per_point = value_per_point.or(position.value_per_point);
    } else if position.qty.abs() > signed.abs() {
        position.qty += signed;
    } else if position.qty.abs() < signed.abs() {
        position.qty = position.qty + signed;
        position.avg_price = price;
        position.contract_name = contract_name;
        position.value_per_point = value_per_point.or(position.value_per_point);
    } else {
        position.qty = 0.0;
    }
    if position.qty.abs() > f64::EPSILON {
        positions.insert(key, position);
    }
}

pub(crate) fn liquidation_csv(analysis: &ReplayLiquidationAnalysis) -> Vec<u8> {
    let mut output = String::from(
        "timestamp_ns,account_id,contract_id,contract_name,side,quantity,reference_price,liquidation_price,value_per_point,gross_pnl,fees,net_pnl,equity_before,margin_requirement,safety_buffer,reason\n",
    );
    for event in &analysis.events {
        let fields = [
            event.timestamp_ns.to_string(),
            event.account_id.to_string(),
            event.contract_id.to_string(),
            event.contract_name.clone(),
            event.side.clone(),
            event.quantity.to_string(),
            event.reference_price.to_string(),
            event.liquidation_price.to_string(),
            event
                .value_per_point
                .map(|value| value.to_string())
                .unwrap_or_default(),
            event.gross_pnl.to_string(),
            event.fees.to_string(),
            event.net_pnl.to_string(),
            event.equity_before.to_string(),
            event.margin_requirement.to_string(),
            event.safety_buffer.to_string(),
            event.reason.clone(),
        ];
        for (index, field) in fields.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            if field.contains(',') || field.contains('"') {
                output.push('"');
                output.push_str(&field.replace('"', "\"\""));
                output.push('"');
            } else {
                output.push_str(field);
            }
        }
        output.push('\n');
    }
    output.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{ReplayExecutionPrecision, ReplayFillPriceSource};

    fn fill(id: i64, side: &str, qty: f64, price: f64, pnl: Option<f64>) -> ReplayExecutionFill {
        ReplayExecutionFill {
            sequence: id as u64,
            lifecycle_sequence: None,
            fill_id: id,
            order_id: id,
            order_strategy_id: None,
            protection_order_id: None,
            account_id: 1,
            contract_id: 2,
            contract_name: "MESU6".to_string(),
            side: side.to_string(),
            quantity: qty,
            price,
            signal_timestamp_ns: None,
            submission_timestamp_ns: None,
            exchange_arrival_timestamp_ns: None,
            acknowledgement_timestamp_ns: None,
            fill_timestamp_ns: id * 1_000,
            fill_price_source: ReplayFillPriceSource::RawBarOpen,
            execution_precision: ReplayExecutionPrecision::BarApproximate,
            exit_reason: None,
            latency_ms: 0,
            tick_size: Some(0.25),
            value_per_point: Some(5.0),
            gross_realized_pnl_delta: pnl,
        }
    }

    #[test]
    fn liquidation_flattens_positions_and_ignores_later_fills() {
        let config = ReplayLiquidationConfig {
            margin: ReplayMarginConfig {
                margin_per_contract: 1_000.0,
                ..ReplayMarginConfig::default()
            },
            slippage_points: 0.25,
        };
        let result = simulate_replay_liquidation(
            &[
                fill(1, "Buy", 1.0, 100.0, None),
                fill(2, "Buy", 1.0, 90.0, Some(-100.0)),
                fill(3, "Sell", 1.0, 80.0, Some(-50.0)),
            ],
            1_500.0,
            DateTime::from_timestamp(0, 0).unwrap(),
            &ReplayFeeSchedule::default(),
            "fee_neutral",
            &config,
        )
        .expect("liquidation simulation");

        assert!(result.triggered);
        assert_eq!(result.event_count, 1);
        assert_eq!(result.ignored_fill_count, 1);
        assert_eq!(result.events[0].side, "Sell");
        assert_eq!(result.events[0].quantity, 2.0);
    }
}
