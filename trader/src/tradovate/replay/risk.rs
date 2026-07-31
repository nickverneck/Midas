use super::fees::ReplayFeeSchedule;
use crate::broker::ReplayExecutionFill;
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) const REPLAY_MARGIN_ANALYSIS_SCHEMA_VERSION: u32 = 1;

/// Fixed-per-contract margin assumptions used by the first account-size
/// analytics slice.  The schedule is intentionally explicit because broker
/// intraday and overnight requirements can change independently of fills.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayMarginConfig {
    pub(crate) model: String,
    pub(crate) currency: String,
    pub(crate) margin_per_contract: f64,
    pub(crate) safety_buffer: f64,
    pub(crate) safety_buffer_percent: f64,
}

impl Default for ReplayMarginConfig {
    fn default() -> Self {
        Self {
            model: "fixed_per_contract".to_string(),
            currency: "USD".to_string(),
            margin_per_contract: 0.0,
            safety_buffer: 0.0,
            safety_buffer_percent: 0.0,
        }
    }
}

impl ReplayMarginConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            bail!("margin model cannot be empty");
        }
        if self.currency.trim().is_empty() {
            bail!("margin currency cannot be empty");
        }
        if !self.margin_per_contract.is_finite() || self.margin_per_contract <= 0.0 {
            bail!("margin_per_contract must be finite and greater than zero");
        }
        if !self.safety_buffer.is_finite() || self.safety_buffer < 0.0 {
            bail!("safety_buffer must be finite and non-negative");
        }
        if !self.safety_buffer_percent.is_finite() || self.safety_buffer_percent < 0.0 {
            bail!("safety_buffer_percent must be finite and non-negative");
        }
        Ok(())
    }

    pub(crate) fn buffer_for_margin(&self, margin: f64) -> f64 {
        self.safety_buffer + margin * self.safety_buffer_percent / 100.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayMarginAnalysis {
    pub(crate) schema_version: u32,
    pub(crate) fee_scenario: String,
    pub(crate) model: String,
    pub(crate) currency: String,
    pub(crate) margin_per_contract: f64,
    pub(crate) safety_buffer: f64,
    pub(crate) safety_buffer_percent: f64,
    pub(crate) initial_capital: f64,
    pub(crate) max_open_position: f64,
    pub(crate) peak_margin_requirement: f64,
    pub(crate) minimum_equity_buffer_over_margin: f64,
    pub(crate) required_starting_capital: f64,
    pub(crate) initial_capital_sufficient: bool,
    pub(crate) first_breach_timestamp_ns: Option<i64>,
}

impl Default for ReplayMarginAnalysis {
    fn default() -> Self {
        Self {
            schema_version: REPLAY_MARGIN_ANALYSIS_SCHEMA_VERSION,
            fee_scenario: String::new(),
            model: ReplayMarginConfig::default().model,
            currency: "USD".to_string(),
            margin_per_contract: 0.0,
            safety_buffer: 0.0,
            safety_buffer_percent: 0.0,
            initial_capital: 0.0,
            max_open_position: 0.0,
            peak_margin_requirement: 0.0,
            minimum_equity_buffer_over_margin: 0.0,
            required_starting_capital: 0.0,
            initial_capital_sufficient: false,
            first_breach_timestamp_ns: None,
        }
    }
}

/// Compute fixed-margin account-size metrics from the fee-neutral ledger and
/// the currently active accounting fee schedule.  Positions are tracked by
/// account and contract so unrelated instruments cannot change the selected
/// contract's margin path accidentally.
pub(crate) fn compute_margin_analysis(
    fills: &[ReplayExecutionFill],
    initial_capital: f64,
    started_at_utc: DateTime<Utc>,
    fee_schedule: &ReplayFeeSchedule,
    fee_scenario: &str,
    config: &ReplayMarginConfig,
) -> Result<ReplayMarginAnalysis> {
    config.validate()?;
    fee_schedule.validate()?;
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        bail!("initial capital must be finite and greater than zero");
    }

    let mut positions: BTreeMap<(i64, i64), f64> = BTreeMap::new();
    let mut cumulative_net_pnl = 0.0;
    let mut max_open_position: f64 = 0.0;
    let mut peak_margin_requirement: f64 = 0.0;
    let mut minimum_buffer = f64::INFINITY;
    let mut required_starting_capital: f64 = 0.0;
    let mut first_breach_timestamp_ns = None;

    record_margin_checkpoint(
        &positions,
        config,
        initial_capital,
        &mut max_open_position,
        &mut peak_margin_requirement,
        &mut minimum_buffer,
        &mut required_starting_capital,
        &mut first_breach_timestamp_ns,
        started_at_utc.timestamp_nanos_opt().unwrap_or_default(),
        cumulative_net_pnl,
    );
    for fill in fills {
        let quantity = fill.quantity.abs();
        if quantity <= f64::EPSILON {
            continue;
        }
        let signed = if fill.side.eq_ignore_ascii_case("buy") {
            quantity
        } else {
            -quantity
        };
        *positions
            .entry((fill.account_id, fill.contract_id))
            .or_default() += signed;
        cumulative_net_pnl += fill.gross_realized_pnl_delta.unwrap_or_default()
            - fee_schedule.fee_for_quantity(quantity);
        record_margin_checkpoint(
            &positions,
            config,
            initial_capital,
            &mut max_open_position,
            &mut peak_margin_requirement,
            &mut minimum_buffer,
            &mut required_starting_capital,
            &mut first_breach_timestamp_ns,
            fill.fill_timestamp_ns,
            cumulative_net_pnl,
        );
    }

    let minimum_equity_buffer_over_margin = if minimum_buffer.is_finite() {
        minimum_buffer
    } else {
        0.0
    };
    Ok(ReplayMarginAnalysis {
        schema_version: REPLAY_MARGIN_ANALYSIS_SCHEMA_VERSION,
        fee_scenario: fee_scenario.to_string(),
        model: config.model.clone(),
        currency: config.currency.clone(),
        margin_per_contract: config.margin_per_contract,
        safety_buffer: config.safety_buffer,
        safety_buffer_percent: config.safety_buffer_percent,
        initial_capital,
        max_open_position,
        peak_margin_requirement,
        minimum_equity_buffer_over_margin,
        required_starting_capital,
        initial_capital_sufficient: first_breach_timestamp_ns.is_none(),
        first_breach_timestamp_ns,
    })
}

fn record_margin_checkpoint(
    positions: &BTreeMap<(i64, i64), f64>,
    config: &ReplayMarginConfig,
    initial_capital: f64,
    max_open_position: &mut f64,
    peak_margin_requirement: &mut f64,
    minimum_buffer: &mut f64,
    required_starting_capital: &mut f64,
    first_breach_timestamp_ns: &mut Option<i64>,
    timestamp_ns: i64,
    cumulative_net_pnl: f64,
) {
    let gross_position = positions.values().map(|value| value.abs()).sum::<f64>();
    let margin = gross_position * config.margin_per_contract;
    let safety_buffer = config.buffer_for_margin(margin);
    let equity = initial_capital + cumulative_net_pnl;
    let equity_buffer = equity - margin - safety_buffer;
    *max_open_position = (*max_open_position).max(gross_position);
    *peak_margin_requirement = (*peak_margin_requirement).max(margin);
    *minimum_buffer = (*minimum_buffer).min(equity_buffer);
    *required_starting_capital =
        (*required_starting_capital).max((margin + safety_buffer - cumulative_net_pnl).max(0.0));
    if equity_buffer < 0.0 && first_breach_timestamp_ns.is_none() {
        *first_breach_timestamp_ns = Some(timestamp_ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{ReplayExecutionPrecision, ReplayFillPriceSource};

    fn fill(
        id: i64,
        side: &str,
        quantity: f64,
        timestamp_ns: i64,
        gross_pnl: Option<f64>,
    ) -> ReplayExecutionFill {
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
            quantity,
            price: 100.0,
            signal_timestamp_ns: None,
            submission_timestamp_ns: None,
            exchange_arrival_timestamp_ns: None,
            acknowledgement_timestamp_ns: None,
            fill_timestamp_ns: timestamp_ns,
            fill_price_source: ReplayFillPriceSource::RawBarOpen,
            execution_precision: ReplayExecutionPrecision::BarApproximate,
            exit_reason: None,
            latency_ms: 0,
            tick_size: Some(0.25),
            value_per_point: Some(5.0),
            gross_realized_pnl_delta: gross_pnl,
        }
    }

    #[test]
    fn margin_metrics_include_fee_overlay_and_gross_open_quantity() {
        let config = ReplayMarginConfig {
            margin_per_contract: 1_000.0,
            safety_buffer: 100.0,
            ..ReplayMarginConfig::default()
        };
        let schedule = ReplayFeeSchedule {
            commission_per_contract: 1.0,
            ..ReplayFeeSchedule::default()
        };
        let result = compute_margin_analysis(
            &[
                fill(1, "Buy", 2.0, 1_000_000_000, None),
                fill(2, "Sell", 1.0, 2_000_000_000, Some(50.0)),
            ],
            1_000.0,
            DateTime::from_timestamp(0, 0).unwrap(),
            &schedule,
            "broker_standard",
            &config,
        )
        .expect("margin analysis");

        assert_eq!(result.fee_scenario, "broker_standard");
        assert_eq!(result.max_open_position, 2.0);
        assert_eq!(result.peak_margin_requirement, 2_000.0);
        assert_eq!(result.required_starting_capital, 2_102.0);
        assert!(!result.initial_capital_sufficient);
        assert_eq!(result.first_breach_timestamp_ns, Some(1_000_000_000));
        assert!((result.minimum_equity_buffer_over_margin + 1_102.0).abs() < f64::EPSILON);
    }

    #[test]
    fn margin_config_rejects_zero_margin() {
        assert!(ReplayMarginConfig::default().validate().is_err());
    }
}
