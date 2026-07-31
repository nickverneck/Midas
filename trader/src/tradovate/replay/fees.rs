use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// A fee schedule applied as an accounting overlay to each simulated fill.
///
/// All components are per contract per side.  Keeping the components separate
/// makes a repriced result explainable while still allowing a compact total
/// calculation for normal fixed-quantity runs.  Negative components are
/// allowed for documented rebates; every value must remain finite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayFeeSchedule {
    pub(crate) name: String,
    pub(crate) currency: String,
    pub(crate) commission_per_contract: f64,
    pub(crate) exchange_per_contract: f64,
    pub(crate) clearing_per_contract: f64,
    pub(crate) regulatory_per_contract: f64,
    pub(crate) misc_per_contract: f64,
}

impl Default for ReplayFeeSchedule {
    fn default() -> Self {
        Self {
            name: "fee_neutral".to_string(),
            currency: "USD".to_string(),
            commission_per_contract: 0.0,
            exchange_per_contract: 0.0,
            clearing_per_contract: 0.0,
            regulatory_per_contract: 0.0,
            misc_per_contract: 0.0,
        }
    }
}

impl ReplayFeeSchedule {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("fee schedule name cannot be empty");
        }
        if self.currency.trim().is_empty() {
            bail!("fee schedule currency cannot be empty");
        }
        for (label, value) in self.components() {
            if !value.is_finite() {
                bail!("fee schedule {label} must be finite");
            }
        }
        if !self.total_per_contract().is_finite() {
            bail!("fee schedule total must be finite");
        }
        Ok(())
    }

    pub(crate) fn total_per_contract(&self) -> f64 {
        self.commission_per_contract
            + self.exchange_per_contract
            + self.clearing_per_contract
            + self.regulatory_per_contract
            + self.misc_per_contract
    }

    pub(crate) fn fee_for_quantity(&self, quantity: f64) -> f64 {
        quantity.abs() * self.total_per_contract()
    }

    fn components(&self) -> [(&'static str, f64); 5] {
        [
            ("commission_per_contract", self.commission_per_contract),
            ("exchange_per_contract", self.exchange_per_contract),
            ("clearing_per_contract", self.clearing_per_contract),
            ("regulatory_per_contract", self.regulatory_per_contract),
            ("misc_per_contract", self.misc_per_contract),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fee_schedule_charges_each_side_and_allows_rebates() {
        let schedule = ReplayFeeSchedule {
            name: "observed".to_string(),
            commission_per_contract: 0.75,
            exchange_per_contract: 0.25,
            clearing_per_contract: 0.10,
            regulatory_per_contract: 0.05,
            misc_per_contract: -0.05,
            ..ReplayFeeSchedule::default()
        };
        schedule.validate().expect("valid schedule");
        assert!((schedule.total_per_contract() - 1.10).abs() < f64::EPSILON);
        assert!((schedule.fee_for_quantity(-2.0) - 2.20).abs() < f64::EPSILON);
    }

    #[test]
    fn fee_schedule_rejects_non_finite_values() {
        let schedule = ReplayFeeSchedule {
            commission_per_contract: f64::NAN,
            ..ReplayFeeSchedule::default()
        };
        assert!(schedule.validate().is_err());
    }
}
