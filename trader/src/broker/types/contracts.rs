use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSuggestion {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub raw: Value,
}

pub const CONTRACT_OPENING_SAFETY_DAYS: i64 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractTradeStatus {
    Ready(String),
    Blocked(String),
    Unknown(String),
}

impl ContractTradeStatus {
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Blocked(_))
    }

    pub fn label(&self) -> String {
        match self {
            Self::Ready(detail) => format!("READY — {detail}"),
            Self::Blocked(detail) => format!("BLOCKED — {detail}"),
            Self::Unknown(detail) => format!("UNKNOWN — {detail}"),
        }
    }
}

impl ContractSuggestion {
    pub fn trade_status(&self) -> ContractTradeStatus {
        self.trade_status_at(Utc::now())
    }

    pub fn trade_status_at(&self, now: DateTime<Utc>) -> ContractTradeStatus {
        let Some(maturity) = self.raw.get("_midasContractMaturity") else {
            return ContractTradeStatus::Unknown(
                "maturity metadata unavailable; selection is allowed".to_string(),
            );
        };

        if maturity
            .get("archived")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return ContractTradeStatus::Blocked("contract maturity is archived".to_string());
        }

        if let Some(first_intent) = maturity_timestamp(maturity, "firstIntentDate") {
            let date = first_intent.format("%Y-%m-%d");
            if now >= first_intent {
                return ContractTradeStatus::Blocked(format!("first intent date passed on {date}"));
            }
            if now >= first_intent - chrono::Duration::days(CONTRACT_OPENING_SAFETY_DAYS) {
                return ContractTradeStatus::Blocked(format!(
                    "first intent {date} is within the {CONTRACT_OPENING_SAFETY_DAYS}-day opening safety window"
                ));
            }
            let days = (first_intent.date_naive() - now.date_naive()).num_days();
            return ContractTradeStatus::Ready(format!(
                "first intent {date} ({days} days away); volume becomes available after subscription"
            ));
        }

        if let Some(expiration) = maturity_timestamp(maturity, "expirationDate") {
            let date = expiration.format("%Y-%m-%d");
            if now >= expiration {
                return ContractTradeStatus::Blocked(format!("expired on {date}"));
            }
            if now >= expiration - chrono::Duration::days(CONTRACT_OPENING_SAFETY_DAYS) {
                return ContractTradeStatus::Blocked(format!(
                    "expiration {date} is within the {CONTRACT_OPENING_SAFETY_DAYS}-day opening safety window"
                ));
            }
            return ContractTradeStatus::Ready(format!(
                "expires {date}; first intent date unavailable"
            ));
        }

        ContractTradeStatus::Unknown("maturity dates unavailable; selection is allowed".to_string())
    }
}

fn maturity_timestamp(maturity: &Value, key: &str) -> Option<DateTime<Utc>> {
    let raw = maturity.get(key).and_then(Value::as_str)?;
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%MZ")
                .ok()
                .map(|timestamp| Utc.from_utc_datetime(&timestamp))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn contract_trade_status_blocks_first_intent_safety_window() {
        let contract = ContractSuggestion {
            id: 4_095_561,
            name: "GCQ6".to_string(),
            description: "Gold August 2026".to_string(),
            raw: json!({
                "contractMaturityId": 59107,
                "_midasContractMaturity": {
                    "id": 59107,
                    "expirationDate": "2026-08-27T17:30Z",
                    "firstIntentDate": "2026-07-31T00:00Z",
                    "archived": false
                }
            }),
        };
        let now = Utc.with_ymd_and_hms(2026, 7, 28, 12, 0, 0).unwrap();

        let status = contract.trade_status_at(now);

        assert!(status.is_blocked());
        assert!(status.label().contains("first intent 2026-07-31"));
        assert!(status.label().contains("5-day opening safety window"));
    }

    #[test]
    fn contract_trade_status_allows_maturity_outside_safety_window() {
        let contract = ContractSuggestion {
            id: 3_267_701,
            name: "GCZ6".to_string(),
            description: "Gold December 2026".to_string(),
            raw: json!({
                "contractMaturityId": 49223,
                "_midasContractMaturity": {
                    "id": 49223,
                    "expirationDate": "2026-12-29T18:30Z",
                    "firstIntentDate": "2026-11-30T00:00Z",
                    "archived": false
                }
            }),
        };
        let now = Utc.with_ymd_and_hms(2026, 7, 28, 12, 0, 0).unwrap();

        let status = contract.trade_status_at(now);

        assert!(!status.is_blocked());
        assert!(matches!(status, ContractTradeStatus::Ready(_)));
        assert!(status.label().contains("first intent 2026-11-30"));
    }
}
