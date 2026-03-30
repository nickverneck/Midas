fn extract_entity_envelopes(item: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if item.get("e").and_then(Value::as_str) == Some("props") {
        if let Some(d) = item.get("d") {
            let deleted = matches!(
                d.get("eventType").and_then(Value::as_str),
                Some("Deleted") | Some("deleted")
            );
            if let Some(entity_type) = d.get("entityType").and_then(Value::as_str) {
                if let Some(entity) = d.get("entity") {
                    out.push(EntityEnvelope {
                        entity_type: entity_type.to_string(),
                        deleted,
                        entity: entity.clone(),
                    });
                }
                if let Some(entities) = d.get("entities").and_then(Value::as_array) {
                    for entity in entities {
                        out.push(EntityEnvelope {
                            entity_type: entity_type.to_string(),
                            deleted,
                            entity: entity.clone(),
                        });
                    }
                }
            }
        }
    }

    if let Some(d) = item.get("d") {
        out.extend(extract_response_entities(d));
    }

    out
}

fn extract_response_entities(payload: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if let Some(items) = payload.as_array() {
        for item in items {
            out.extend(extract_response_entities(item));
        }
        return out;
    }

    let Some(obj) = payload.as_object() else {
        return out;
    };

    if let Some(entity_type) = obj.get("entityType").and_then(Value::as_str) {
        if let Some(entity) = obj.get("entity") {
            out.push(EntityEnvelope {
                entity_type: entity_type.to_string(),
                deleted: false,
                entity: entity.clone(),
            });
        }
        if let Some(entities) = obj.get("entities").and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: entity_type.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    for (key, plural) in [
        ("account", "accounts"),
        ("accountRiskStatus", "accountRiskStatuses"),
        ("cashBalance", "cashBalances"),
        ("position", "positions"),
        ("order", "orders"),
        ("orderStrategy", "orderStrategies"),
        ("orderStrategyLink", "orderStrategyLinks"),
        ("executionReport", "executionReports"),
        ("fill", "fills"),
    ] {
        if let Some(entity) = obj.get(key) {
            if entity.is_object() {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
        if let Some(entities) = obj.get(plural).and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    out
}

fn parse_status_code(msg: &Value) -> Option<i64> {
    if let Some(code) = msg.get("s").and_then(Value::as_i64) {
        return Some(code);
    }
    msg.get("s")
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<i64>().ok())
}

fn parse_frame(raw: &str) -> (char, Option<Value>) {
    let mut chars = raw.chars();
    let frame_type = chars.next().unwrap_or('\0');
    let offset = frame_type.len_utf8();
    let payload = raw.get(offset..).unwrap_or("");
    let value = if payload.is_empty() {
        None
    } else {
        serde_json::from_str(payload).ok()
    };
    (frame_type, value)
}

fn create_message(endpoint: &str, id: u64, query: Option<&str>, body: Option<&Value>) -> String {
    match (query, body) {
        (Some(query), Some(body)) => format!("{endpoint}\n{id}\n{query}\n{body}"),
        (Some(query), None) => format!("{endpoint}\n{id}\n{query}"),
        (None, Some(body)) => format!("{endpoint}\n{id}\n\n{body}"),
        (None, None) => format!("{endpoint}\n{id}\n\n"),
    }
}

fn parse_socket_response(message: &Value) -> Result<Value, String> {
    let Some(status) = parse_status_code(message) else {
        return Err("websocket response missing status code".to_string());
    };
    let payload = message.get("d").cloned().unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        Ok(payload)
    } else if let Some(text) = payload.as_str() {
        Err(format!("websocket request failed ({status}): {text}"))
    } else {
        Err(format!("websocket request failed ({status}): {payload}"))
    }
}

fn parse_bar(value: &Value) -> Option<Bar> {
    let ts = value.get("timestamp")?.as_str()?;
    let ts_ns = parse_bar_timestamp_ns(ts)?;
    Some(Bar {
        ts_ns,
        open: json_number(value, "open")?,
        high: json_number(value, "high")?,
        low: json_number(value, "low")?,
        close: json_number(value, "close")?,
    })
}

fn parse_bar_timestamp_ns(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::DateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M%:z")
                .map(|dt| dt.with_timezone(&Utc))
        })
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%MZ").map(|dt| dt.and_utc())
        })
        .ok()?
        .timestamp_nanos_opt()
}

fn json_number(value: &Value, key: &str) -> Option<f64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_f64() {
        return Some(v);
    }
    if let Some(v) = raw.as_i64() {
        return Some(v as f64);
    }
    if let Some(v) = raw.as_u64() {
        return Some(v as f64);
    }
    raw.as_str().and_then(|text| text.parse::<f64>().ok())
}

fn json_i64(value: &Value, key: &str) -> Option<i64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_i64() {
        return Some(v);
    }
    if let Some(v) = raw.as_u64() {
        return i64::try_from(v).ok();
    }
    raw.as_str().and_then(|text| text.parse::<i64>().ok())
}

fn sanitize_price(price: Option<f64>) -> Option<f64> {
    price.filter(|value| value.is_finite() && *value > 0.0)
}

fn with_cl_ord_id(mut payload: Value, cl_ord_id: Option<&str>) -> Value {
    if let Some(cl_ord_id) = cl_ord_id {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("clOrdId".to_string(), Value::String(cl_ord_id.to_string()));
        }
    }
    payload
}

fn prices_match(lhs: Option<f64>, rhs: Option<f64>) -> bool {
    match (lhs, rhs) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn known_order_id(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| json_i64(value, key))
}

fn first_known_order_id(value: &Value) -> Option<i64> {
    known_order_id(value, &["orderId", "id", "otherId", "stopOrderId"])
}

fn order_is_active(order: &Value) -> bool {
    let Some(status) = order
        .get("ordStatus")
        .and_then(Value::as_str)
        .or_else(|| order.get("status").and_then(Value::as_str))
    else {
        return true;
    };

    !matches!(
        status.to_ascii_lowercase().as_str(),
        "filled" | "cancelled" | "canceled" | "rejected" | "expired" | "stopped" | "finished"
    )
}

fn extract_entity_id(value: &Value) -> Option<i64> {
    json_i64(value, "id")
}

fn extract_account_id(entity_type: &str, value: &Value) -> Option<i64> {
    if entity_type.eq_ignore_ascii_case("account") {
        return json_i64(value, "id");
    }
    json_i64(value, "accountId")
        .or_else(|| {
            value
                .get("account")
                .and_then(|account| account.get("id"))
                .and_then(Value::as_i64)
        })
        .or_else(|| value.get("account").and_then(Value::as_i64))
        .or_else(|| json_i64(value, "id"))
}

fn empty_as_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn parse_bar_accepts_minute_precision_utc_timestamp() {
        let bar = parse_bar(&json!({
            "timestamp": "2026-03-11T22:38Z",
            "open": 6738.5,
            "high": 6739.5,
            "low": 6736.75,
            "close": 6738.0
        }))
        .expect("bar should parse");

        let expected_ts = chrono::DateTime::parse_from_rfc3339("2026-03-11T22:38:00Z")
            .unwrap()
            .with_timezone(&Utc)
            .timestamp_nanos_opt()
            .unwrap();

        assert_eq!(bar.ts_ns, expected_ts);
        assert_eq!(bar.close, 6738.0);
    }

    #[test]
    fn contract_position_qty_matches_selected_contract() {
        let mut store = UserSyncStore::default();
        store.positions.insert(
            42,
            BTreeMap::from([
                (
                    1,
                    json!({
                        "id": 1,
                        "accountId": 42,
                        "contractId": 3570918,
                        "netPos": 2
                    }),
                ),
                (
                    2,
                    json!({
                        "id": 2,
                        "accountId": 42,
                        "symbol": "ESM6",
                        "netPos": -1
                    }),
                ),
            ]),
        );

        let contract = ContractSuggestion {
            id: 3570918,
            name: "ESH6".to_string(),
            description: String::new(),
            raw: json!({ "contractMaturityId": 53951 }),
        };

        assert_eq!(store.contract_position_qty(42, &contract), Some(2.0));
    }

    #[test]
    fn contract_position_qty_uses_position_side_when_netpos_missing() {
        let mut store = UserSyncStore::default();
        store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "qty": 1,
                    "side": "Short"
                }),
            )]),
        );

        let contract = ContractSuggestion {
            id: 3570918,
            name: "ESM6".to_string(),
            description: String::new(),
            raw: json!({ "contractMaturityId": 53951 }),
        };

        assert_eq!(store.contract_position_qty(42, &contract), Some(-1.0));
    }

    #[test]
    fn fallback_unrealized_pnl_uses_latest_close_and_value_per_point() {
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 6725.0,
                high: 6730.0,
                low: 6724.0,
                close: 6727.25,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };

        let positions = vec![json!({
            "accountId": 42,
            "contractId": 3570918,
            "netPos": 1,
            "netPrice": 6725.75
        })];

        assert_eq!(fallback_unrealized_pnl(&positions, &market), Some(75.0));
    }

    #[test]
    fn build_market_update_emits_snapshot_then_forming_delta() {
        let contract = ContractSuggestion {
            id: 3570918,
            name: "ESH6".to_string(),
            description: "ES Jun 2026".to_string(),
            raw: json!({ "id": 3570918 }),
        };
        let closed_bar = Bar {
            ts_ns: 1,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        };
        let forming_bar = Bar {
            ts_ns: 2,
            open: 5000.5,
            high: 5002.0,
            low: 5000.25,
            close: 5001.5,
        };
        let mut series = LiveSeries::new();
        series.closed_bars.push(closed_bar.clone());

        let initial = build_market_update(
            &contract,
            None,
            series.closed_bars.len(),
            0,
            "status".to_string(),
            0,
            None,
            None,
            &series,
        )
        .expect("initial snapshot should be emitted");
        assert!(matches!(
            initial.bars,
            MarketBarsUpdate::Snapshot {
                ref closed_bars,
                forming_bar: None
            } if closed_bars == &vec![closed_bar.clone()]
        ));

        let before_last_closed = series.closed_bars.last().cloned();
        series.forming_bar = Some(forming_bar.clone());
        let update = build_market_update(
            &contract,
            None,
            series.closed_bars.len(),
            0,
            "status".to_string(),
            series.closed_bars.len(),
            before_last_closed,
            None,
            &series,
        )
        .expect("forming update should be emitted");
        assert!(matches!(
            update.bars,
            MarketBarsUpdate::Forming { forming_bar: ref bar } if bar == &forming_bar
        ));
    }

    #[test]
    fn apply_market_update_keeps_bars_incremental() {
        let closed_bar = Bar {
            ts_ns: 1,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        };
        let forming_bar = Bar {
            ts_ns: 2,
            open: 5000.5,
            high: 5002.0,
            low: 5000.25,
            close: 5001.5,
        };
        let next_forming_bar = Bar {
            ts_ns: 3,
            open: 5001.5,
            high: 5003.0,
            low: 5001.0,
            close: 5002.5,
        };
        let marker = TradeMarker {
            fill_id: Some(7),
            account_id: Some(42),
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            ts_ns: 1,
            price: 5000.5,
            qty: 1,
            side: TradeMarkerSide::Buy,
        };
        let mut market = MarketSnapshot {
            trade_markers: vec![marker],
            ..MarketSnapshot::default()
        };

        let initial = MarketUpdate {
            contract_id: 3570918,
            contract_name: "ESH6".to_string(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: "initial".to_string(),
            bars: MarketBarsUpdate::Snapshot {
                closed_bars: vec![closed_bar.clone()],
                forming_bar: Some(forming_bar.clone()),
            },
        };
        assert!(apply_market_update(&mut market, initial));
        assert_eq!(market.history_loaded, 1);
        assert_eq!(market.bars, vec![closed_bar.clone(), forming_bar.clone()]);
        assert_eq!(market.trade_markers.len(), 1);

        let next = MarketUpdate {
            contract_id: 3570918,
            contract_name: "ESH6".to_string(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 2,
            live_bars: 1,
            status: "realtime".to_string(),
            bars: MarketBarsUpdate::Closed {
                closed_bar: forming_bar.clone(),
                forming_bar: Some(next_forming_bar.clone()),
            },
        };
        assert!(apply_market_update(&mut market, next));
        assert_eq!(market.history_loaded, 2);
        assert_eq!(market.bars, vec![closed_bar, forming_bar, next_forming_bar]);
        assert_eq!(market.trade_markers.len(), 1);
    }

    #[test]
    fn apply_market_update_drops_oldest_closed_bar_when_window_is_full() {
        let bar = |ts_ns| Bar {
            ts_ns,
            open: 5000.0 + ts_ns as f64,
            high: 5001.0 + ts_ns as f64,
            low: 4999.0 + ts_ns as f64,
            close: 5000.5 + ts_ns as f64,
        };

        let mut market = MarketSnapshot {
            bars: vec![bar(1), bar(2)],
            history_loaded: 2,
            ..MarketSnapshot::default()
        };
        let update = MarketUpdate {
            contract_id: 3570918,
            contract_name: "ESH6".to_string(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 2,
            live_bars: 1,
            status: "realtime".to_string(),
            bars: MarketBarsUpdate::Closed {
                closed_bar: bar(3),
                forming_bar: None,
            },
        };

        assert!(apply_market_update(&mut market, update));
        assert_eq!(market.history_loaded, 2);
        assert_eq!(market.bars, vec![bar(2), bar(3)]);
    }

    #[test]
    fn display_market_snapshot_trims_to_recent_closed_bars_and_keeps_forming_bar() {
        let bar = |ts_ns| Bar {
            ts_ns,
            open: 5000.0 + ts_ns as f64,
            high: 5001.0 + ts_ns as f64,
            low: 4999.0 + ts_ns as f64,
            close: 5000.5 + ts_ns as f64,
        };

        let mut bars = (1..=300).map(bar).collect::<Vec<_>>();
        bars.push(bar(301));
        let market = MarketSnapshot {
            bars,
            history_loaded: 300,
            status: "streaming".to_string(),
            ..MarketSnapshot::default()
        };

        let snapshot = display_market_snapshot(&market);
        assert_eq!(snapshot.history_loaded, UI_MARKET_BAR_LIMIT);
        assert_eq!(snapshot.bars.len(), UI_MARKET_BAR_LIMIT + 1);
        assert_eq!(snapshot.bars.first().map(|bar| bar.ts_ns), Some(45));
        assert_eq!(snapshot.bars.last().map(|bar| bar.ts_ns), Some(301));
        assert_eq!(snapshot.status, "streaming");
    }

    #[test]
    fn jwt_expiration_time_reads_exp_claim() {
        let token = "eyJhbGciOiJub25lIn0.eyJleHAiOjE4OTM0NTYwMDB9.sig";
        let expires_at = jwt_expiration_time(token).expect("jwt exp should parse");

        let expected = DateTime::<Utc>::from_timestamp(1_893_456_000, 0).unwrap();
        assert_eq!(expires_at, expected);
    }

    #[test]
    fn token_refresh_due_uses_jwt_exp_when_expiration_time_missing() {
        let tokens = TokenBundle {
            access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
            md_access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
            expiration_time: None,
            user_id: Some(42),
            user_name: Some("demo".to_string()),
        };

        let now = DateTime::<Utc>::from_timestamp(1_773_436_044 - 60, 0).unwrap();
        assert!(token_refresh_due(&tokens, now));
    }

    #[test]
    fn create_message_formats_body_only_requests_for_websocket() {
        let body = json!({
            "accountId": 42,
            "orderQty": 1
        });

        let actual = create_message("order/placeorder", 7, None, Some(&body));

        assert_eq!(
            actual,
            "order/placeorder\n7\n\n{\"accountId\":42,\"orderQty\":1}"
        );
    }

    #[test]
    fn parse_socket_response_maps_status_and_payload() {
        let ok = json!({
            "i": 3,
            "s": 200,
            "d": { "orderId": 99 }
        });
        let err = json!({
            "i": 4,
            "s": 400,
            "d": "bad request"
        });

        assert_eq!(
            parse_socket_response(&ok).expect("success payload"),
            json!({ "orderId": 99 })
        );
        assert_eq!(
            parse_socket_response(&err).expect_err("error payload"),
            "websocket request failed (400): bad request"
        );
    }

    #[test]
    fn tracker_matches_entity_binds_order_id_from_cl_ord_id() {
        let mut tracker = OrderLatencyTracker {
            started_at: time::Instant::now(),
            signal_started_at: None,
            signal_context: None,
            cl_ord_id: "midas-1-entry".to_string(),
            order_id: None,
            order_strategy_id: None,
            seen_recorded: false,
            exec_report_recorded: false,
            fill_recorded: false,
        };
        let entity = json!({
            "orderId": 42,
            "clOrdId": "midas-1-entry"
        });

        assert!(tracker_matches_entity(&mut tracker, &entity));
        assert_eq!(tracker.order_id, Some(42));
    }

    #[test]
    fn update_latency_from_order_strategy_link_binds_order_id() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: Some(OrderLatencyTracker {
                started_at: time::Instant::now(),
                signal_started_at: None,
                signal_context: None,
                cl_ord_id: "midas-1-strategy".to_string(),
                order_id: None,
                order_strategy_id: Some(77),
                seen_recorded: false,
                exec_report_recorded: false,
                fill_recorded: false,
            }),
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store: UserSyncStore::default(),
            selected_account_id: None,
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
        };
        let mut latency = LatencySnapshot::default();
        let link = EntityEnvelope {
            entity_type: "orderStrategyLink".to_string(),
            deleted: false,
            entity: json!({
                "id": 700,
                "orderStrategyId": 77,
                "orderId": 42,
                "label": "entry"
            }),
        };

        assert!(!update_latency_from_envelope(&mut session, &mut latency, &link));
        assert_eq!(
            session
                .order_latency_tracker
                .as_ref()
                .and_then(|tracker| tracker.order_id),
            Some(42)
        );
    }

    #[test]
    fn update_latency_from_envelope_records_seen_ack_and_fill() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: Some(OrderLatencyTracker {
                started_at: time::Instant::now(),
                signal_started_at: Some(time::Instant::now()),
                signal_context: Some("ema_cross Buy (qty 0 -> 1)".to_string()),
                cl_ord_id: "midas-1-entry".to_string(),
                order_id: Some(42),
                order_strategy_id: None,
                seen_recorded: false,
                exec_report_recorded: false,
                fill_recorded: false,
            }),
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store: UserSyncStore::default(),
            selected_account_id: None,
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
        };
        let mut latency = LatencySnapshot::default();

        let order = EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42 }),
        };
        let exec_report = EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42, "execType": "New" }),
        };
        let fill = EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42, "price": 5000.25, "qty": 1 }),
        };

        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &order
        ));
        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &exec_report
        ));
        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &fill
        ));
        assert!(latency.last_order_seen_ms.is_some());
        assert!(latency.last_exec_report_ms.is_some());
        assert!(latency.last_fill_ms.is_some());
        assert!(latency.last_signal_seen_ms.is_some());
        assert!(latency.last_signal_ack_ms.is_some());
        assert!(latency.last_signal_fill_ms.is_some());
    }

    #[test]
    fn user_store_tracks_active_order_strategy_and_linked_orders() {
        let mut store = UserSyncStore::default();
        store.order_strategies.insert(
            77,
            json!({
                "id": 77,
                "accountId": 42,
                "contractId": 3570918,
                "status": "ActiveStrategy",
                "uuid": "midas-1710546400000-1-strategy"
            }),
        );
        store.orders.insert(
            42,
            BTreeMap::from([
                (
                    101,
                    json!({
                        "id": 101,
                        "accountId": 42,
                        "contractId": 3570918,
                        "orderType": "Limit",
                        "price": 5010.0
                    }),
                ),
                (
                    102,
                    json!({
                        "id": 102,
                        "accountId": 42,
                        "contractId": 3570918,
                        "orderType": "Stop",
                        "stopPrice": 4990.0
                    }),
                ),
            ]),
        );
        store.order_strategy_links.insert(
            1,
            json!({
                "id": 1,
                "orderStrategyId": 77,
                "orderId": 101,
                "label": "tp"
            }),
        );
        store.order_strategy_links.insert(
            2,
            json!({
                "id": 2,
                "orderStrategyId": 77,
                "orderId": 102,
                "label": "sl"
            }),
        );

        let strategy = store
            .find_active_order_strategy(42, 3570918)
            .expect("strategy should be tracked");
        let linked = store.linked_strategy_orders(42, 77);

        assert_eq!(extract_entity_id(strategy), Some(77));
        assert_eq!(linked.len(), 2);
        assert_eq!(linked[0].get("id").and_then(Value::as_i64), Some(101));
        assert_eq!(linked[1].get("id").and_then(Value::as_i64), Some(102));
    }

    #[test]
    fn user_store_treats_nonterminal_midas_order_strategy_status_as_active() {
        let mut store = UserSyncStore::default();
        store.order_strategies.insert(
            77,
            json!({
                "id": 77,
                "accountId": 42,
                "contractId": 3570918,
                "status": "Working",
                "uuid": "midas-1710546400000-1-strategy"
            }),
        );

        let strategy = store
            .find_active_order_strategy(42, 3570918)
            .expect("working strategy should be tracked");

        assert_eq!(extract_entity_id(strategy), Some(77));
    }

    #[test]
    fn user_store_recovers_midas_strategy_from_linked_active_orders() {
        let mut store = UserSyncStore::default();
        store.order_strategies.insert(
            77,
            json!({
                "id": 77,
                "status": "Working",
                "uuid": "midas-1710546400000-1-strategy"
            }),
        );
        store.orders.insert(
            42,
            BTreeMap::from([(
                101,
                json!({
                    "id": 101,
                    "accountId": 42,
                    "contractId": 3570918,
                    "orderType": "Stop",
                    "ordStatus": "Working"
                }),
            )]),
        );
        store.order_strategy_links.insert(
            1,
            json!({
                "id": 1,
                "orderStrategyId": 77,
                "orderId": 101,
                "label": "sl"
            }),
        );

        let strategy = store
            .find_active_order_strategy(42, 3570918)
            .expect("linked active order should recover the strategy");

        assert_eq!(extract_entity_id(strategy), Some(77));
    }

    #[test]
    fn user_store_ignores_terminal_midas_order_strategy_status() {
        let mut store = UserSyncStore::default();
        store.order_strategies.insert(
            77,
            json!({
                "id": 77,
                "accountId": 42,
                "contractId": 3570918,
                "status": "InterruptedStrategy",
                "uuid": "midas-1710546400000-1-strategy"
            }),
        );

        assert!(store.find_active_order_strategy(42, 3570918).is_none());
    }

    #[test]
    fn trade_marker_from_fill_uses_order_action_for_side_and_contract() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut user_store = UserSyncStore::default();
        user_store.orders.insert(
            42,
            BTreeMap::from([(
                77,
                json!({
                    "id": 77,
                    "accountId": 42,
                    "action": "Sell",
                    "contractId": 3570918,
                    "symbol": "ESH6"
                }),
            )]),
        );
        let session = SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: None,
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store,
            selected_account_id: Some(42),
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
        };
        let fill = json!({
            "id": 501,
            "accountId": 42,
            "orderId": 77,
            "price": 5000.25,
            "qty": 1,
            "timestamp": "2026-03-15T13:45:00Z"
        });

        let marker = trade_marker_from_fill(&session, &fill).expect("fill marker should resolve");

        assert_eq!(marker.fill_id, Some(501));
        assert_eq!(marker.account_id, Some(42));
        assert_eq!(marker.contract_id, Some(3570918));
        assert_eq!(marker.contract_name.as_deref(), Some("ESH6"));
        assert_eq!(marker.side, TradeMarkerSide::Sell);
        assert_eq!(marker.price, 5000.25);
        assert_eq!(marker.qty, 1);
    }

    #[test]
    fn record_trade_marker_deduplicates_fill_ids() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = SessionState {
            cfg: AppConfig::default(),
            session_kind: SessionKind::Live,
            replay_enabled: false,
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            pending_signal_context: None,
            order_latency_tracker: None,
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            pending_protection_sync: None,
            user_store: UserSyncStore::default(),
            selected_account_id: Some(42),
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            active_order_strategy: None,
            next_strategy_order_nonce: 1,
        };
        let marker = TradeMarker {
            fill_id: Some(501),
            account_id: Some(42),
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            ts_ns: 1,
            price: 5000.25,
            qty: 1,
            side: TradeMarkerSide::Buy,
        };

        assert!(record_trade_marker(&mut session, marker.clone()));
        assert!(!record_trade_marker(&mut session, marker));
        assert_eq!(session.market.trade_markers.len(), 1);
    }

    #[test]
    fn user_store_skips_live_fills_but_keeps_replay_fills() {
        let mut store = UserSyncStore::default();
        store.apply(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({
                "id": 11,
                "accountId": 42,
                "contractId": 3570918,
                "buySell": "Buy",
                "price": 5000.0,
                "qty": 1,
                "timestamp": 1
            }),
        });
        assert!(store.fills.get(&42).is_none());

        store.apply(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({
                "id": 12,
                "accountId": 42,
                "contractId": 3570918,
                "source": "replay",
                "buySell": "Buy",
                "price": 5000.0,
                "qty": 1,
                "timestamp": 2
            }),
        });
        assert_eq!(store.fills.get(&42).map(BTreeMap::len), Some(1));
    }

    #[test]
    fn build_snapshots_include_realized_pnl_and_protection_prices() {
        let mut store = UserSyncStore::default();
        store.risk.insert(
            42,
            json!({
                "accountId": 42,
                "balance": 10000.0,
                "realizedPnL": 125.5
            }),
        );
        store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "netPrice": 5000.0
                }),
            )]),
        );
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 4999.0,
                high: 5002.0,
                low: 4998.0,
                close: 5001.0,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };
        let managed_protection = BTreeMap::from([(
            StrategyProtectionKey {
                account_id: 42,
                contract_id: 3570918,
            },
            ManagedProtectionOrders {
                signed_qty: 1,
                take_profit_price: Some(5004.0),
                stop_price: Some(4998.0),
                last_requested_take_profit_price: Some(5004.0),
                last_requested_stop_price: Some(4998.0),
                take_profit_cl_ord_id: None,
                stop_cl_ord_id: None,
                take_profit_order_id: None,
                stop_order_id: None,
            },
        )]);
        let accounts = vec![AccountInfo {
            id: 42,
            name: "sim".to_string(),
            raw: json!({ "id": 42, "name": "sim" }),
        }];

        let snapshots = store.build_snapshots(&accounts, Some(&market), &managed_protection);
        let snapshot = snapshots.first().expect("snapshot should exist");

        assert_eq!(snapshot.realized_pnl, Some(125.5));
        assert_eq!(snapshot.market_entry_price, Some(5000.0));
        assert_eq!(snapshot.selected_contract_take_profit_price, Some(5004.0));
        assert_eq!(snapshot.selected_contract_stop_price, Some(4998.0));
    }

    #[test]
    fn replay_snapshots_mark_to_market_open_positions() {
        let mut store = UserSyncStore::default();
        store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "avgPrice": 5000.0
                }),
            )]),
        );
        store.fills.insert(
            42,
            BTreeMap::from([(
                11,
                json!({
                    "id": 11,
                    "accountId": 42,
                    "contractId": 3570918,
                    "buySell": "Buy",
                    "price": 5000.0,
                    "qty": 1,
                    "timestamp": 1
                }),
            )]),
        );
        let accounts = vec![AccountInfo {
            id: 42,
            name: "REPLAY".to_string(),
            raw: json!({
                "id": 42,
                "name": "REPLAY",
                "source": "replay",
                "startingBalance": 100000.0
            }),
        }];
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 5000.0,
                high: 5001.5,
                low: 4999.5,
                close: 5001.0,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };

        let snapshots = store.build_snapshots(&accounts, Some(&market), &BTreeMap::new());
        let snapshot = snapshots.first().expect("snapshot should exist");

        assert_eq!(snapshot.realized_pnl, Some(0.0));
        assert_eq!(snapshot.unrealized_pnl, Some(50.0));
        assert_eq!(snapshot.balance, Some(100000.0));
        assert_eq!(snapshot.cash_balance, Some(100000.0));
        assert_eq!(snapshot.net_liq, Some(100050.0));
    }

    #[test]
    fn replay_snapshots_roll_realized_pnl_into_balance() {
        let mut store = UserSyncStore::default();
        store.fills.insert(
            42,
            BTreeMap::from([
                (
                    11,
                    json!({
                        "id": 11,
                        "accountId": 42,
                        "contractId": 3570918,
                        "buySell": "Buy",
                        "price": 5000.0,
                        "qty": 1,
                        "timestamp": 1
                    }),
                ),
                (
                    12,
                    json!({
                        "id": 12,
                        "accountId": 42,
                        "contractId": 3570918,
                        "buySell": "Sell",
                        "price": 5002.0,
                        "qty": 1,
                        "timestamp": 2
                    }),
                ),
            ]),
        );
        let accounts = vec![AccountInfo {
            id: 42,
            name: "REPLAY".to_string(),
            raw: json!({
                "id": 42,
                "name": "REPLAY",
                "source": "replay",
                "startingBalance": 100000.0
            }),
        }];
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 5002.0,
                high: 5002.0,
                low: 5002.0,
                close: 5002.0,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };

        let snapshots = store.build_snapshots(&accounts, Some(&market), &BTreeMap::new());
        let snapshot = snapshots.first().expect("snapshot should exist");

        assert_eq!(snapshot.realized_pnl, Some(100.0));
        assert_eq!(snapshot.unrealized_pnl, Some(0.0));
        assert_eq!(snapshot.balance, Some(100100.0));
        assert_eq!(snapshot.cash_balance, Some(100100.0));
        assert_eq!(snapshot.net_liq, Some(100100.0));
    }

    #[test]
    fn replay_bar_fills_take_profit_and_clears_sibling_strategy_orders() {
        let mut broker = ReplayBrokerState::default();
        let key = StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        };
        broker.positions.insert(
            key,
            SimPosition {
                position_id: 30_001,
                qty: 1,
                avg_price: 6600.0,
                symbol: "ES 06-26".to_string(),
            },
        );

        let strategy_id = 40_001;
        broker.order_strategies.insert(
            strategy_id,
            SimOrderStrategyState {
                entity: json!({
                    "id": strategy_id,
                    "accountId": 42,
                    "contractId": 3570918,
                    "symbol": "ES 06-26",
                    "status": "Active",
                }),
                order_ids: vec![1001, 1002],
                link_ids: vec![5001, 5002],
            },
        );
        broker.active_orders.insert(
            1001,
            SimActiveOrder {
                order: json!({
                    "id": 1001,
                    "orderId": 1001,
                    "accountId": 42,
                    "contractId": 3570918,
                    "symbol": "ES 06-26",
                    "action": "Sell",
                    "orderQty": 1,
                    "orderType": "Limit",
                    "price": 6604.0,
                    "ordStatus": "Working",
                    "clOrdId": "midas-tp",
                    "orderStrategyId": strategy_id,
                }),
                link_id: Some(5001),
                strategy_id: Some(strategy_id),
            },
        );
        broker.active_orders.insert(
            1002,
            SimActiveOrder {
                order: json!({
                    "id": 1002,
                    "orderId": 1002,
                    "accountId": 42,
                    "contractId": 3570918,
                    "symbol": "ES 06-26",
                    "action": "Sell",
                    "orderQty": 1,
                    "orderType": "Stop",
                    "stopPrice": 6592.0,
                    "ordStatus": "Working",
                    "clOrdId": "midas-sl",
                    "orderStrategyId": strategy_id,
                }),
                link_id: Some(5002),
                strategy_id: Some(strategy_id),
            },
        );

        let events = broker.simulate_replay_bar(&Bar {
            ts_ns: 123,
            open: 6601.0,
            high: 6604.5,
            low: 6599.5,
            close: 6604.0,
        });

        assert_eq!(events.len(), 1);
        let envelopes = match &events[0] {
            InternalEvent::UserEntities(envelopes) => envelopes,
            _ => panic!("expected replay bar to emit user entities"),
        };

        assert!(envelopes.iter().any(|envelope| {
            envelope.entity_type == "fill" && json_i64(&envelope.entity, "orderId") == Some(1001)
        }));
        assert!(envelopes.iter().any(|envelope| {
            envelope.entity_type == "order"
                && !envelope.deleted
                && json_i64(&envelope.entity, "orderId") == Some(1001)
                && envelope.entity.get("ordStatus").and_then(Value::as_str) == Some("Filled")
        }));
        assert!(envelopes.iter().any(|envelope| {
            envelope.entity_type == "order"
                && envelope.deleted
                && json_i64(&envelope.entity, "orderId") == Some(1002)
        }));
        assert!(envelopes.iter().any(|envelope| {
            envelope.entity_type == "orderStrategy"
                && envelope.deleted
                && json_i64(&envelope.entity, "id") == Some(strategy_id)
        }));
        assert!(envelopes
            .iter()
            .any(|envelope| envelope.entity_type == "position" && envelope.deleted));
        assert!(broker.active_orders.is_empty());
        assert!(broker.order_strategies.is_empty());
        assert!(!broker.positions.contains_key(&key));
    }

    #[test]
    fn parse_expiration_time_accepts_rfc3339() {
        let parsed = parse_expiration_time("2026-03-13T15:04:05Z").expect("timestamp should parse");
        let expected = DateTime::parse_from_rfc3339("2026-03-13T15:04:05Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(parsed, expected);
    }

    #[test]
    fn futures_globex_preclose_window_holds_entries() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 16, 50, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(window.hold_entries);
        assert!(window.minutes_to_close.unwrap() <= 10.0);
    }

    #[test]
    fn futures_globex_daily_break_holds_until_reopen() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 17, 30, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(!window.session_open);
        assert!(window.hold_entries);
        assert_eq!(window.minutes_to_close, None);
    }

    #[test]
    fn futures_globex_reopens_after_break() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 18, 5, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(!window.hold_entries);
        assert!(window.minutes_to_close.unwrap() > 1_300.0);
    }

    #[test]
    fn equity_rth_preclose_window_holds_entries() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 15, 50, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::EquityRth.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(window.hold_entries);
        assert!(window.minutes_to_close.unwrap() <= 10.0);
    }
}
