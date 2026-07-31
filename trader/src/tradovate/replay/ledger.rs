use super::*;
use crate::broker::{
    REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION, ReplayBarProtectionPolicy, ReplayEngineMode,
    ReplayExecutionLedgerSnapshot, ReplayExecutionLedgerSummary, ReplayLatencyConfig,
    ReplayLatencyModel,
};
#[cfg(feature = "replay")]
use crate::broker::{ReplayExecutionFill, ReplayFillPriceSource};
#[cfg(feature = "replay")]
use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Default)]
struct ReplayLedgerPosition {
    quantity: f64,
    average_price: f64,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
pub(crate) struct ReplayLedgerMarketContext {
    pub contract_name: String,
    pub tick_size: Option<f64>,
    pub value_per_point: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReplayExecutionLedgerState {
    snapshot: ReplayExecutionLedgerSnapshot,
    #[cfg(feature = "replay")]
    seen_fill_ids: BTreeSet<i64>,
    #[cfg(feature = "replay")]
    positions: BTreeMap<(i64, i64), ReplayLedgerPosition>,
    #[cfg(feature = "replay")]
    next_sequence: u64,
}

impl ReplayExecutionLedgerState {
    pub fn new(
        engine_mode: ReplayEngineMode,
        fixed_latency_ms: u64,
        bar_type: BarType,
        candle_mode: CandleMode,
    ) -> Self {
        Self::new_with_config(
            engine_mode,
            &ReplayLatencyConfig {
                fixed_latency_ms,
                ..ReplayLatencyConfig::default()
            },
            ReplayBarProtectionPolicy::Conservative,
            bar_type,
            candle_mode,
        )
    }

    pub fn new_with_config(
        engine_mode: ReplayEngineMode,
        latency: &ReplayLatencyConfig,
        bar_protection_policy: ReplayBarProtectionPolicy,
        bar_type: BarType,
        candle_mode: CandleMode,
    ) -> Self {
        Self {
            snapshot: ReplayExecutionLedgerSnapshot {
                schema_version: REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION,
                fee_neutral: true,
                engine_mode,
                latency_model: match engine_mode {
                    ReplayEngineMode::Legacy => ReplayLatencyModel::IgnoredLegacy,
                    ReplayEngineMode::Deterministic => latency.model,
                },
                fixed_latency_ms: if engine_mode == ReplayEngineMode::Deterministic {
                    latency.fixed_latency_ms
                } else {
                    0
                },
                latency_seed: (engine_mode == ReplayEngineMode::Deterministic
                    && latency.model == ReplayLatencyModel::SeededObserved)
                    .then_some(latency.seed),
                observed_latency_sample_count: if engine_mode == ReplayEngineMode::Deterministic {
                    latency.observed_samples_ms.len()
                } else {
                    0
                },
                bar_protection_policy: if engine_mode == ReplayEngineMode::Deterministic {
                    bar_protection_policy
                } else {
                    ReplayBarProtectionPolicy::NearestOpen
                },
                signal_source: bar_type.mode_label(candle_mode),
                gross_realized_pnl: 0.0,
                fills: Vec::new(),
            },
            ..Self::default()
        }
    }

    pub fn snapshot(&self) -> &ReplayExecutionLedgerSnapshot {
        &self.snapshot
    }

    pub fn summary(&self) -> ReplayExecutionLedgerSummary {
        self.snapshot.summary()
    }

    #[cfg(feature = "replay")]
    pub fn append_entities(
        &mut self,
        entities: &[EntityEnvelope],
        market: &ReplayLedgerMarketContext,
    ) -> usize {
        let before = self.snapshot.fills.len();
        for envelope in entities {
            if envelope.deleted
                || !envelope.entity_type.eq_ignore_ascii_case("fill")
                || !is_replay_fill(&envelope.entity)
            {
                continue;
            }
            let Some(fill_id) = json_i64(&envelope.entity, "id") else {
                continue;
            };
            if !self.seen_fill_ids.insert(fill_id) {
                continue;
            }
            let Some(mut fill) =
                parse_fill(&envelope.entity, self.next_sequence, &self.snapshot, market)
            else {
                self.seen_fill_ids.remove(&fill_id);
                continue;
            };
            self.next_sequence = self.next_sequence.saturating_add(1);
            fill.gross_realized_pnl_delta = self.apply_gross_fill(&fill);
            if let Some(delta) = fill.gross_realized_pnl_delta {
                self.snapshot.gross_realized_pnl += delta;
            }
            self.snapshot.fills.push(fill);
        }
        self.snapshot.fills.len().saturating_sub(before)
    }

    #[cfg(feature = "replay")]
    fn apply_gross_fill(&mut self, fill: &ReplayExecutionFill) -> Option<f64> {
        let signed_fill_qty = match fill.side.to_ascii_lowercase().as_str() {
            "buy" => fill.quantity,
            "sell" => -fill.quantity,
            _ => return None,
        };
        let key = (fill.account_id, fill.contract_id);
        let position = self.positions.entry(key).or_default();
        let mut realized_points = 0.0;

        if position.quantity.abs() <= f64::EPSILON
            || position.quantity.signum() == signed_fill_qty.signum()
        {
            let next_abs = position.quantity.abs() + signed_fill_qty.abs();
            position.average_price = if position.quantity.abs() <= f64::EPSILON {
                fill.price
            } else {
                ((position.average_price * position.quantity.abs())
                    + (fill.price * signed_fill_qty.abs()))
                    / next_abs.max(1.0)
            };
            position.quantity += signed_fill_qty;
        } else {
            let prior_abs = position.quantity.abs();
            let close_qty = prior_abs.min(signed_fill_qty.abs());
            realized_points = if position.quantity > 0.0 {
                (fill.price - position.average_price) * close_qty
            } else {
                (position.average_price - fill.price) * close_qty
            };
            position.quantity += signed_fill_qty;
            if position.quantity.abs() <= f64::EPSILON {
                position.quantity = 0.0;
                position.average_price = 0.0;
            } else if signed_fill_qty.abs() > prior_abs {
                position.average_price = fill.price;
            }
        }

        fill.value_per_point
            .map(|value_per_point| realized_points * value_per_point)
    }
}

#[cfg(feature = "replay")]
fn parse_fill(
    value: &Value,
    sequence: u64,
    ledger: &ReplayExecutionLedgerSnapshot,
    market: &ReplayLedgerMarketContext,
) -> Option<ReplayExecutionFill> {
    let quantity = json_f64(value, &["qty", "quantity"])?.abs();
    let price = json_f64(value, &["price"])?;
    if !quantity.is_finite() || quantity <= 0.0 || !price.is_finite() {
        return None;
    }
    let side = value
        .get("buySell")
        .and_then(Value::as_str)
        .or_else(|| value.get("action").and_then(Value::as_str))?
        .to_string();
    let fill_timestamp_ns =
        json_i64(value, "replayFillTimestampNs").or_else(|| json_i64(value, "timestamp"))?;
    let contract_name = value
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| value.get("contractName").and_then(Value::as_str))
        .unwrap_or(&market.contract_name)
        .to_string();

    Some(ReplayExecutionFill {
        sequence,
        lifecycle_sequence: json_i64(value, "replayLifecycleSequence")
            .and_then(|number| u64::try_from(number).ok()),
        fill_id: json_i64(value, "id")?,
        order_id: json_i64(value, "orderId")?,
        order_strategy_id: json_i64(value, "orderStrategyId"),
        protection_order_id: json_i64(value, "replayProtectionOrderId"),
        account_id: json_i64(value, "accountId")?,
        contract_id: json_i64(value, "contractId")?,
        contract_name,
        side,
        quantity,
        price,
        signal_timestamp_ns: json_i64(value, "replaySignalTimestampNs"),
        submission_timestamp_ns: json_i64(value, "replaySubmissionTimestampNs"),
        exchange_arrival_timestamp_ns: json_i64(value, "replayExchangeArrivalTimestampNs"),
        acknowledgement_timestamp_ns: json_i64(value, "replayAcknowledgementTimestampNs"),
        fill_timestamp_ns,
        fill_price_source: parse_fill_source(value, ledger.engine_mode),
        exit_reason: value
            .get("replayExitReason")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        latency_ms: json_i64(value, "replayLatencyMs")
            .and_then(|number| u64::try_from(number).ok())
            .unwrap_or(ledger.fixed_latency_ms),
        tick_size: market.tick_size,
        value_per_point: market.value_per_point,
        gross_realized_pnl_delta: None,
    })
}

#[cfg(feature = "replay")]
fn parse_fill_source(value: &Value, engine_mode: ReplayEngineMode) -> ReplayFillPriceSource {
    match value
        .get("replayFillSource")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "raw_bar_open" => ReplayFillPriceSource::RawBarOpen,
        "raw_bar_ohlc" => ReplayFillPriceSource::RawBarOhlc,
        "legacy_reference_price" => ReplayFillPriceSource::LegacyReferencePrice,
        _ if engine_mode == ReplayEngineMode::Deterministic => ReplayFillPriceSource::RawBarOpen,
        _ => ReplayFillPriceSource::LegacyReferencePrice,
    }
}

#[cfg(feature = "replay")]
fn is_replay_fill(value: &Value) -> bool {
    value
        .get("source")
        .and_then(Value::as_str)
        .is_some_and(|source| source.eq_ignore_ascii_case("replay"))
}

#[cfg(feature = "replay")]
fn json_f64(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|number| {
            number
                .as_f64()
                .or_else(|| number.as_i64().map(|value| value as f64))
                .or_else(|| number.as_u64().map(|value| value as f64))
        })
    })
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;

    fn fill(entity: Value) -> EntityEnvelope {
        EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity,
        }
    }

    fn market() -> ReplayLedgerMarketContext {
        ReplayLedgerMarketContext {
            contract_name: "MESU6".to_string(),
            tick_size: Some(0.25),
            value_per_point: Some(5.0),
        }
    }

    #[test]
    fn ledger_is_append_only_deduplicated_and_fee_neutral() {
        let mut ledger = ReplayExecutionLedgerState::new(
            ReplayEngineMode::Deterministic,
            60,
            BarType::minute(1),
            CandleMode::HeikinAshi,
        );
        let entry = fill(json!({
            "id": 100,
            "accountId": 7,
            "contractId": 11,
            "orderId": 1000,
            "source": "replay",
            "symbol": "MESU6",
            "price": 5000.0,
            "qty": 1,
            "buySell": "Buy",
            "timestamp": 2_000,
            "replayLifecycleSequence": 1,
            "replayFillSource": "raw_bar_open",
            "replaySignalTimestampNs": 1_000,
            "replaySubmissionTimestampNs": 1_000,
            "replayExchangeArrivalTimestampNs": 1_060,
            "replayAcknowledgementTimestampNs": 2_000,
            "replayFillTimestampNs": 2_000,
            "replayLatencyMs": 60
        }));

        assert_eq!(
            ledger.append_entities(std::slice::from_ref(&entry), &market()),
            1
        );
        assert_eq!(ledger.append_entities(&[entry], &market()), 0);

        let snapshot = ledger.snapshot();
        assert!(snapshot.fee_neutral);
        assert_eq!(snapshot.fills.len(), 1);
        assert_eq!(snapshot.fills[0].sequence, 0);
        assert_eq!(snapshot.fills[0].lifecycle_sequence, Some(1));
        assert_eq!(
            snapshot.fills[0].fill_price_source,
            ReplayFillPriceSource::RawBarOpen
        );
        assert_eq!(snapshot.fills[0].gross_realized_pnl_delta, Some(0.0));
        assert!(
            !serde_json::to_string(snapshot)
                .unwrap()
                .contains("commission")
        );
    }

    #[test]
    fn ledger_v2_records_reproducible_latency_and_protection_configuration() {
        let latency = ReplayLatencyConfig {
            model: ReplayLatencyModel::SeededObserved,
            fixed_latency_ms: 0,
            observed_samples_ms: vec![25, 50, 100],
            seed: 91,
        };
        let ledger = ReplayExecutionLedgerState::new_with_config(
            ReplayEngineMode::Deterministic,
            &latency,
            ReplayBarProtectionPolicy::Optimistic,
            BarType::minute(1),
            CandleMode::Standard,
        );

        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.schema_version, 2);
        assert_eq!(snapshot.latency_model, ReplayLatencyModel::SeededObserved);
        assert_eq!(snapshot.latency_seed, Some(91));
        assert_eq!(snapshot.observed_latency_sample_count, 3);
        assert_eq!(
            snapshot.bar_protection_policy,
            ReplayBarProtectionPolicy::Optimistic
        );
    }

    #[test]
    fn ledger_records_protection_exit_and_gross_realized_pnl() {
        let mut ledger = ReplayExecutionLedgerState::new(
            ReplayEngineMode::Deterministic,
            0,
            BarType::minute(1),
            CandleMode::Standard,
        );
        let entities = vec![
            fill(json!({
                "id": 100,
                "accountId": 7,
                "contractId": 11,
                "orderId": 1000,
                "source": "replay",
                "symbol": "MESU6",
                "price": 5000.0,
                "qty": 2,
                "buySell": "Buy",
                "timestamp": 1_000,
                "replayFillSource": "raw_bar_open"
            })),
            fill(json!({
                "id": 101,
                "accountId": 7,
                "contractId": 11,
                "orderId": 1001,
                "orderStrategyId": 4000,
                "source": "replay",
                "symbol": "MESU6",
                "price": 5002.0,
                "qty": 2,
                "buySell": "Sell",
                "timestamp": 2_000,
                "replayFillSource": "raw_bar_ohlc",
                "replayProtectionOrderId": 1001,
                "replayExitReason": "take_profit"
            })),
        ];

        assert_eq!(ledger.append_entities(&entities, &market()), 2);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.gross_realized_pnl, 20.0);
        assert_eq!(snapshot.fills[1].gross_realized_pnl_delta, Some(20.0));
        assert_eq!(snapshot.fills[1].protection_order_id, Some(1001));
        assert_eq!(
            snapshot.fills[1].exit_reason.as_deref(),
            Some("take_profit")
        );
        assert_eq!(
            snapshot.fills[1].fill_price_source,
            ReplayFillPriceSource::RawBarOhlc
        );
    }

    #[test]
    fn ledger_position_accounting_is_isolated_by_account_and_contract() {
        let mut ledger = ReplayExecutionLedgerState::new(
            ReplayEngineMode::Deterministic,
            0,
            BarType::minute(1),
            CandleMode::Standard,
        );
        let entities = vec![
            fill(json!({
                "id": 100,
                "accountId": 7,
                "contractId": 11,
                "orderId": 1000,
                "source": "replay",
                "symbol": "MESU6",
                "price": 100.0,
                "qty": 1,
                "buySell": "Buy",
                "timestamp": 1_000
            })),
            fill(json!({
                "id": 101,
                "accountId": 7,
                "contractId": 12,
                "orderId": 1001,
                "source": "replay",
                "symbol": "GCU6",
                "price": 4000.0,
                "qty": 1,
                "buySell": "Buy",
                "timestamp": 1_100
            })),
            fill(json!({
                "id": 102,
                "accountId": 7,
                "contractId": 11,
                "orderId": 1002,
                "source": "replay",
                "symbol": "MESU6",
                "price": 101.0,
                "qty": 1,
                "buySell": "Sell",
                "timestamp": 1_200
            })),
        ];

        assert_eq!(ledger.append_entities(&entities, &market()), 3);
        assert_eq!(ledger.snapshot().gross_realized_pnl, 5.0);
        assert_eq!(
            ledger.snapshot().fills[2].gross_realized_pnl_delta,
            Some(5.0)
        );
    }
}
