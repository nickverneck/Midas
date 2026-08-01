//! Import broker fee/margin metadata into an explicit replay overlay.
//!
//! Provider metadata shapes have changed over time, so this importer is
//! deliberately tolerant about field spelling and nesting while remaining
//! conservative: it never invents a fee or margin value when the source does
//! not contain one.

use super::fees::ReplayFeeSchedule;
use super::risk::ReplayMarginConfig;
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub(crate) const REPLAY_BROKER_SCHEDULE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayBrokerSchedule {
    pub(crate) schema_version: u32,
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) fetched_at_utc: Option<DateTime<Utc>>,
    pub(crate) fee: ReplayFeeSchedule,
    pub(crate) margin_model: Option<String>,
    pub(crate) margin_per_contract: Option<f64>,
    pub(crate) warnings: Vec<String>,
}

impl Default for ReplayBrokerSchedule {
    fn default() -> Self {
        Self {
            schema_version: REPLAY_BROKER_SCHEDULE_SCHEMA_VERSION,
            name: String::new(),
            source: String::new(),
            fetched_at_utc: None,
            fee: ReplayFeeSchedule::default(),
            margin_model: None,
            margin_per_contract: None,
            warnings: Vec::new(),
        }
    }
}

impl ReplayBrokerSchedule {
    pub(crate) fn from_metadata_path(
        path: &Path,
        name: Option<&str>,
        currency: &str,
    ) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("read broker metadata {}", path.display()))?;
        let payload: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse broker metadata {}", path.display()))?;
        Self::from_metadata_value(&payload, name, currency, path.display().to_string())
    }

    pub(crate) fn from_metadata_value(
        payload: &Value,
        name: Option<&str>,
        currency: &str,
        source: String,
    ) -> Result<Self> {
        if currency.trim().is_empty() {
            bail!("broker schedule currency cannot be empty");
        }
        let metadata = payload
            .get("contract_metadata")
            .or_else(|| payload.get("metadata"))
            .unwrap_or(payload);
        let fee_snapshot = metadata
            .get("fee_params")
            .and_then(snapshot_payload)
            .unwrap_or(metadata);
        let margin_snapshot = metadata
            .get("contract_margins")
            .and_then(snapshot_payload)
            .or_else(|| metadata.get("product_margins").and_then(snapshot_payload));

        let mut warnings = Vec::new();
        let commission = first_number(
            fee_snapshot,
            &[
                "commission",
                "commission_per_contract",
                "commission_per_side",
                "commission_fee",
                "commission_rate",
            ],
        );
        let exchange = first_number(
            fee_snapshot,
            &[
                "exchange",
                "exchange_fee",
                "exchange_per_contract",
                "exchange_rate",
            ],
        );
        let clearing = first_number(
            fee_snapshot,
            &[
                "clearing",
                "clearing_fee",
                "clearing_per_contract",
                "clearing_rate",
            ],
        );
        let regulatory = first_number(
            fee_snapshot,
            &[
                "regulatory",
                "regulatory_fee",
                "regulatory_per_contract",
                "regulatory_rate",
                "nfa_fee",
                "nfa_fees",
                "nfa_rate",
            ],
        );
        let misc = first_number(fee_snapshot, &["misc", "misc_fee", "miscellaneous_fee"]);
        let total = first_number(
            fee_snapshot,
            &[
                "total_fee",
                "total_fees",
                "total_per_contract",
                "total_rate",
                "fees",
                "fee",
            ],
        );
        let component_count = [commission, exchange, clearing, regulatory, misc]
            .into_iter()
            .flatten()
            .count();
        let misc = if component_count == 0 {
            if let Some(total) = total {
                warnings.push(
                    "broker metadata exposed only a total fee; it was stored as misc_per_contract"
                        .to_string(),
                );
                Some(total)
            } else {
                None
            }
        } else {
            misc
        };
        if component_count == 0 && misc.is_none() {
            bail!(
                "broker metadata {} contains no recognizable fee or total-fee value",
                source
            );
        }

        let schedule_name = name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("broker_import")
            .to_string();
        let fee = ReplayFeeSchedule {
            name: schedule_name.clone(),
            currency: currency.trim().to_string(),
            commission_per_contract: commission.unwrap_or_default(),
            exchange_per_contract: exchange.unwrap_or_default(),
            clearing_per_contract: clearing.unwrap_or_default(),
            regulatory_per_contract: regulatory.unwrap_or_default(),
            misc_per_contract: misc.unwrap_or_default(),
        };
        fee.validate()?;

        let margin_per_contract = margin_snapshot.and_then(|value| {
            first_number(
                value,
                &[
                    "initial_margin",
                    "initialmargin",
                    "intraday_margin",
                    "intradaymargin",
                    "day_trade_margin",
                    "daytrademargin",
                    "maintenance_margin",
                    "maintenancemargin",
                    "margin_requirement",
                    "margin",
                ],
            )
        });
        if margin_snapshot.is_some() && margin_per_contract.is_none() {
            warnings.push(
                "broker metadata contained margin snapshots but no recognizable margin value"
                    .to_string(),
            );
        }

        let schedule = Self {
            schema_version: REPLAY_BROKER_SCHEDULE_SCHEMA_VERSION,
            name: schedule_name,
            source,
            fetched_at_utc: first_datetime(metadata),
            fee,
            margin_model: margin_per_contract.map(|_| "broker_imported".to_string()),
            margin_per_contract,
            warnings,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("read broker schedule {}", path.display()))?;
        let schedule: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse broker schedule {}", path.display()))?;
        schedule.validate()?;
        Ok(schedule)
    }

    pub(crate) fn margin_config(&self) -> Option<ReplayMarginConfig> {
        self.margin_per_contract.map(|margin| ReplayMarginConfig {
            model: self
                .margin_model
                .clone()
                .unwrap_or_else(|| "broker_imported".to_string()),
            currency: self.fee.currency.clone(),
            margin_per_contract: margin,
            ..ReplayMarginConfig::default()
        })
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        fs::write(
            path,
            serde_json::to_vec_pretty(self).context("serialize broker schedule")?,
        )
        .with_context(|| format!("write broker schedule {}", path.display()))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version == 0 || self.schema_version > REPLAY_BROKER_SCHEDULE_SCHEMA_VERSION {
            bail!(
                "unsupported broker schedule schema version {}",
                self.schema_version
            );
        }
        if self.name.trim().is_empty() {
            bail!("broker schedule name cannot be empty");
        }
        if self.source.trim().is_empty() {
            bail!("broker schedule source cannot be empty");
        }
        self.fee.validate()?;
        if let Some(margin) = self.margin_per_contract
            && (!margin.is_finite() || margin <= 0.0)
        {
            bail!("broker schedule margin_per_contract must be finite and greater than zero");
        }
        Ok(())
    }
}

fn snapshot_payload(value: &Value) -> Option<&Value> {
    value.get("payload").or(Some(value))
}

fn first_number(value: &Value, keys: &[&str]) -> Option<f64> {
    let wanted = keys
        .iter()
        .map(|key| normalize_key(key))
        .collect::<Vec<_>>();
    find_value(value, &wanted).and_then(json_number)
}

fn find_value<'a>(value: &'a Value, wanted: &[String]) -> Option<&'a Value> {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                if wanted.iter().any(|wanted| *wanted == normalize_key(key)) {
                    if json_number(value).is_some() {
                        return Some(value);
                    }
                }
            }
            fields.values().find_map(|value| find_value(value, wanted))
        }
        Value::Array(items) => items.iter().find_map(|value| find_value(value, wanted)),
        _ => None,
    }
}

fn json_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
        .filter(|value| value.is_finite())
}

fn first_datetime(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::Object(fields) => {
            for key in ["source_timestamp", "fetched_at", "timestamp", "updated_at"] {
                if let Some(value) = fields.get(key).and_then(Value::as_str)
                    && let Ok(parsed) = DateTime::parse_from_rfc3339(value)
                {
                    return Some(parsed.with_timezone(&Utc));
                }
            }
            fields.values().find_map(first_datetime)
        }
        Value::Array(items) => items.iter().find_map(first_datetime),
        _ => None,
    }
}

fn normalize_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn imports_fee_and_margin_components_from_cache_metadata() {
        let value = json!({
            "contract_metadata": {
                "fee_params": {"payload": {"commission": 0.39, "exchangeFee": 0.18, "nfaFee": 0.02}},
                "contract_margins": {"payload": [{"initialMargin": 1430.0}]},
                "fetched_at": "2026-07-31T12:00:00Z"
            }
        });
        let schedule = ReplayBrokerSchedule::from_metadata_value(
            &value,
            Some("tradovate_standard"),
            "USD",
            "fixture".to_string(),
        )
        .expect("schedule import");
        assert_eq!(schedule.fee.name, "tradovate_standard");
        assert_eq!(schedule.fee.commission_per_contract, 0.39);
        assert_eq!(schedule.fee.exchange_per_contract, 0.18);
        assert_eq!(schedule.margin_per_contract, Some(1430.0));
        assert!(schedule.fetched_at_utc.is_some());
    }

    #[test]
    fn total_only_fee_is_kept_as_documented_misc_component() {
        let value = json!({"fee_params": {"payload": {"totalFees": 1.25}}});
        let schedule =
            ReplayBrokerSchedule::from_metadata_value(&value, None, "USD", "fixture".to_string())
                .expect("schedule import");
        assert_eq!(schedule.fee.misc_per_contract, 1.25);
        assert_eq!(schedule.warnings.len(), 1);
    }
}
