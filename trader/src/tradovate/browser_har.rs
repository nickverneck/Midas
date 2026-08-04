//! Offline importer for NinjaTrader/Tradovate browser market-data HAR files.
//!
//! The browser uses the same multiplexed market-data WebSocket that the
//! application can use directly.  A HAR is useful for validating that feed
//! and for turning a captured session into replay artifacts, but it is not a
//! live browser bridge and it must never cause the captured authorization
//! token to be copied into a manifest.

use crate::broker::{BrokerKind, ReplayDomLevel, ReplayMarketDom};
use crate::config::{AppConfig, TradingEnvironment};
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheCoverage, ReplayCacheDataFile, ReplayCacheFileFormat,
    ReplayCacheInstrument, ReplayCacheManifest, ReplayCacheMarketShape, ReplayCacheRawTickRow,
    ReplayCacheRawTicksWrite, ReplayCacheSourceKind, ReplayCacheTickSpecs,
    replay_cache_dataset_dir, write_raw_ticks_parquet_cache,
};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BROWSER_TRADE_SOURCE: &str = "browser_quote_last";
const BROWSER_DATA_WARNING: &str = "Browser md/subscribequote last/ls updates are sampled provider quote updates, not a guaranteed complete trade tape; use the historical md/getChart tick downloader for exact tick history.";

#[derive(Debug, Clone)]
pub struct BrowserHarImportOptions {
    pub input: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub contracts: Vec<String>,
    pub environment: String,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct ContractDefinition {
    id: i64,
    name: String,
    maturity_id: Option<i64>,
    product_id: Option<i64>,
    expiration: Option<NaiveDate>,
    tick_size: Option<f64>,
    value_per_point: Option<f64>,
    product_name: Option<String>,
    product_description: Option<String>,
}

#[derive(Debug, Clone)]
struct ProductDefinition {
    name: Option<String>,
    description: Option<String>,
    value_per_point: Option<f64>,
    tick_size: Option<f64>,
}

#[derive(Debug, Clone)]
struct MaturityDefinition {
    product_id: Option<i64>,
    expiration: Option<NaiveDate>,
}

#[derive(Debug, Clone)]
struct CapturedMessage {
    direction: String,
    time: f64,
    data: String,
    order: u64,
}

#[derive(Debug, Clone)]
struct CapturedTrade {
    ts_ms: i64,
    arrival_order: u64,
    price: f64,
    size: f64,
    bid_price: Option<f64>,
    bid_size: Option<f64>,
    ask_price: Option<f64>,
    ask_size: Option<f64>,
}

#[derive(Debug, Clone)]
struct CapturedDom {
    ts_ms: i64,
    arrival_order: u64,
    snapshot: ReplayMarketDom,
}

#[derive(Debug, Clone, Copy, Default)]
struct QuoteState {
    bid_price: Option<f64>,
    bid_size: Option<f64>,
    ask_price: Option<f64>,
    ask_size: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct ContractCapture {
    trades: Vec<CapturedTrade>,
    dom: Vec<CapturedDom>,
}

#[derive(Debug, Deserialize)]
struct HarDocument {
    log: HarLog,
}

#[derive(Debug, Deserialize)]
struct HarLog {
    #[serde(default)]
    entries: Vec<HarEntry>,
}

#[derive(Debug, Deserialize)]
struct HarEntry {
    request: HarRequest,
    #[serde(rename = "_webSocketMessages", default)]
    web_socket_messages: Vec<HarWebSocketMessage>,
}

#[derive(Debug, Deserialize)]
struct HarRequest {
    url: String,
}

#[derive(Debug, Deserialize)]
struct HarWebSocketMessage {
    #[serde(rename = "type")]
    direction: String,
    #[serde(default)]
    time: f64,
    #[serde(default)]
    data: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImportEnvironment {
    value: TradingEnvironment,
    source: String,
}

/// Import the browser's observed quote/last and visible DOM streams into one
/// replay dataset per captured contract.
///
/// The resulting raw-tick Parquet rows deliberately carry a warning: the
/// browser quote channel has no provider sequence and does not expose every
/// execution.  The sidecar DOM JSONL is normalized to `ReplayMarketDom`, so it
/// can be supplied to the existing deterministic DOM replay path.
pub fn import_browser_har(config: &AppConfig, options: BrowserHarImportOptions) -> Result<()> {
    let input = options.input;
    let file = File::open(&input).with_context(|| format!("open HAR {}", input.display()))?;
    let document: HarDocument =
        serde_json::from_reader(file).with_context(|| format!("parse HAR {}", input.display()))?;
    if document.log.entries.is_empty() {
        bail!("HAR {} contains no network entries", input.display());
    }

    let all_messages = collect_messages(&document.log.entries);
    let market_messages = collect_market_messages(&document.log.entries);
    if market_messages.is_empty() {
        bail!(
            "HAR {} contains no Tradovate market-data WebSocket messages",
            input.display()
        );
    }

    let (contracts, products, maturities) = collect_contract_metadata(&all_messages);
    let subscribed_ids = collect_subscribed_symbols(&market_messages)?;
    if subscribed_ids.is_empty() {
        bail!("market-data WebSocket contains no md/subscribequote or md/subscribeDOM requests");
    }

    let environment = resolve_environment(&options.environment, &document.log.entries, config.env)?;
    let selected_names = normalize_contract_filter(&options.contracts);
    let mut definitions = BTreeMap::new();
    let mut warnings = Vec::new();
    for id in &subscribed_ids {
        let Some(mut definition) = contracts.get(id).cloned() else {
            warnings.push(format!(
                "Skipped subscribed symbol id {id}: no contract/finds metadata was present in the HAR"
            ));
            continue;
        };
        if let Some(maturity_id) = definition.maturity_id
            && let Some(maturity) = maturities.get(&maturity_id)
        {
            definition.product_id = definition.product_id.or(maturity.product_id);
            definition.expiration = definition.expiration.or(maturity.expiration);
        }
        if let Some(product_id) = definition.product_id
            && let Some(product) = products.get(&product_id)
        {
            definition.product_name = product.name.clone();
            definition.product_description = product.description.clone();
            definition.value_per_point = definition.value_per_point.or(product.value_per_point);
            definition.tick_size = definition.tick_size.or(product.tick_size);
        }
        if !selected_names.is_empty()
            && !selected_names.contains(&definition.name.to_ascii_uppercase())
        {
            continue;
        }
        definitions.insert(*id, definition);
    }
    if definitions.is_empty() {
        if selected_names.is_empty() {
            bail!("no subscribed contracts could be resolved from the HAR metadata");
        }
        bail!(
            "none of the requested contracts ({}) were subscribed and resolved in the HAR",
            selected_names
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let captures = parse_market_capture(&market_messages, definitions.keys().copied().collect());
    let cache_root = options
        .cache_dir
        .unwrap_or_else(|| config.replay_cache_dir.clone());
    let source_label = source_socket_label(&market_messages);
    let mut imported = 0usize;
    let mut total_trades = 0usize;
    let mut total_dom = 0usize;

    for (id, definition) in definitions {
        let capture = captures.get(&id).cloned().unwrap_or_default();
        if capture.trades.is_empty() && capture.dom.is_empty() {
            warnings.push(format!(
                "{} (id {}): subscribed but the HAR contains no quote/DOM updates",
                definition.name, id
            ));
            continue;
        }
        if capture.trades.is_empty() {
            warnings.push(format!(
                "{} (id {}): DOM was captured but no last-trade updates were present; no replay price dataset was written",
                definition.name, id
            ));
            continue;
        }
        let first_ms = capture
            .trades
            .iter()
            .map(|trade| trade.ts_ms)
            .chain(capture.dom.iter().map(|dom| dom.ts_ms))
            .min()
            .context("capture timestamp range is empty")?;
        let last_ms = capture
            .trades
            .iter()
            .map(|trade| trade.ts_ms)
            .chain(capture.dom.iter().map(|dom| dom.ts_ms))
            .max()
            .context("capture timestamp range is empty")?;
        let first = timestamp_from_millis(first_ms)?;
        let last = timestamp_from_millis(last_ms)?;
        let request_end = last
            .checked_add_signed(Duration::milliseconds(1))
            .unwrap_or(last);
        let instrument_symbol = instrument_root(&definition.name);
        let tick_size = definition
            .tick_size
            .filter(|value| value.is_finite() && *value > 0.0)
            .with_context(|| {
                format!(
                    "{} has no positive provider tick size in the HAR metadata",
                    definition.name
                )
            })?;
        let value_per_point = definition
            .value_per_point
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0);
        if definition.value_per_point.is_none() {
            warnings.push(format!(
                "{}: product valuePerPoint was absent; manifest uses 1.0 as a placeholder",
                definition.name
            ));
        }

        let dataset_dir = replay_cache_dataset_dir(
            &cache_root,
            BrokerKind::Tradovate,
            environment.value,
            &instrument_symbol,
            &definition.name,
            first.date_naive(),
        );
        let manifest_path = dataset_dir.join("manifest.json");
        if manifest_path.exists() && !options.overwrite {
            bail!(
                "replay dataset already exists at {}; pass --overwrite to replace it",
                manifest_path.display()
            );
        }

        let mut trades = capture.trades;
        trades.sort_by(|left, right| {
            left.ts_ms
                .cmp(&right.ts_ms)
                .then_with(|| left.arrival_order.cmp(&right.arrival_order))
        });
        let rows = trades
            .iter()
            .enumerate()
            .map(|(index, trade)| {
                let timestamp = timestamp_from_millis(trade.ts_ms)?;
                Ok(ReplayCacheRawTickRow {
                    timestamp,
                    ts_ns: timestamp
                        .timestamp_nanos_opt()
                        .context("browser quote timestamp is outside nanosecond range")?,
                    tick_id: Some(i64::try_from(index + 1).unwrap_or(i64::MAX)),
                    price: trade.price,
                    size: trade.size,
                    bid_price: trade.bid_price,
                    bid_size: trade.bid_size,
                    ask_price: trade.ask_price,
                    ask_size: trade.ask_size,
                    chart_id: None,
                    trade_date: timestamp.format("%Y%m%d").to_string().parse::<i32>().ok(),
                    packet_source: Some(BROWSER_TRADE_SOURCE.to_string()),
                    packet_base_ts_ms: None,
                    packet_base_price_ticks: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut raw_warnings = vec![BROWSER_DATA_WARNING.to_string()];
        if !capture.dom.is_empty() {
            raw_warnings.push(
                "The DOM sidecar is visible depth only; it is not a complete exchange order book."
                    .to_string(),
            );
        }
        let trade_count = rows.len();
        let write = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
            cache_root: cache_root.clone(),
            target: None,
            provider: BrokerKind::Tradovate,
            env: environment.value,
            instrument: ReplayCacheInstrument {
                symbol: instrument_symbol.clone(),
                name: definition
                    .product_description
                    .clone()
                    .or_else(|| definition.product_name.clone()),
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: definition.name.clone(),
                id: Some(definition.id),
                expiration: definition.expiration,
            },
            request_start: first,
            request_end,
            download_request: json!({
                "source": "ninjatrader_browser_har",
                "market_socket": source_label,
                "contract_id": definition.id,
                "quote_channel": "md/subscribequote",
                "trade_semantics": "sampled_last_trade_updates",
                "dom_channel": "md/subscribedom",
            }),
            tick_specs: ReplayCacheTickSpecs {
                tick_size,
                value_per_point,
            },
            contract_metadata: None,
            session_template: None,
            ticks: rows,
            warnings: raw_warnings,
            display_name: Some(format!(
                "{} browser sampled quote capture {}",
                definition.name,
                first.date_naive()
            )),
            tags: Some(vec![
                "browser-har".to_string(),
                "sampled-quote-feed".to_string(),
            ]),
            notes: Some(
                "Imported from a NinjaTrader browser HAR. q.last/ls updates are not guaranteed to contain every execution; use the historical tick downloader for complete tape replay.".to_string(),
            ),
        })?;

        let dom_count = write_dom_sidecar(
            &write.manifest_path,
            &write.dataset_dir,
            definition.id,
            capture.dom,
            source_label.as_str(),
        )?;
        imported += 1;
        total_trades += trade_count;
        total_dom += dom_count;
        println!(
            "Imported {} (id {}) -> {} ({} sampled trade updates, {} DOM snapshots).",
            definition.name,
            definition.id,
            write.manifest_path.display(),
            trade_count,
            dom_count
        );
    }

    if imported == 0 {
        bail!("HAR contained no importable price captures");
    }
    println!(
        "Browser HAR import complete: {} dataset(s), {} sampled trade updates, {} DOM snapshots.",
        imported, total_trades, total_dom
    );
    println!(
        "Environment: {} ({}) | Cache root: {}",
        environment.value.label(),
        environment.source,
        cache_root.display()
    );
    println!("Warning: {BROWSER_DATA_WARNING}");
    for warning in warnings {
        println!("Warning: {warning}");
    }
    Ok(())
}

fn collect_messages(entries: &[HarEntry]) -> Vec<CapturedMessage> {
    let mut messages = Vec::new();
    let mut order = 0_u64;
    for entry in entries {
        for message in &entry.web_socket_messages {
            messages.push(CapturedMessage {
                direction: message.direction.clone(),
                time: message.time,
                data: message.data.clone(),
                order,
            });
            order = order.saturating_add(1);
        }
    }
    messages
}

fn collect_market_messages(entries: &[HarEntry]) -> Vec<CapturedMessage> {
    let mut messages = Vec::new();
    let mut order = 0_u64;
    let mut market_entries = entries
        .iter()
        .filter(|entry| is_market_socket_url(&entry.request.url))
        .collect::<Vec<_>>();
    market_entries.sort_by(|left, right| {
        first_ws_time(left)
            .total_cmp(&first_ws_time(right))
            .then_with(|| left.request.url.cmp(&right.request.url))
    });
    for entry in market_entries {
        for message in &entry.web_socket_messages {
            messages.push(CapturedMessage {
                direction: message.direction.clone(),
                time: message.time,
                data: message.data.clone(),
                order,
            });
            order = order.saturating_add(1);
        }
    }
    messages.sort_by(|left, right| {
        left.time
            .total_cmp(&right.time)
            .then_with(|| left.order.cmp(&right.order))
    });
    for (index, message) in messages.iter_mut().enumerate() {
        message.order = u64::try_from(index).unwrap_or(u64::MAX);
    }
    messages
}

fn first_ws_time(entry: &HarEntry) -> f64 {
    entry
        .web_socket_messages
        .first()
        .map(|message| message.time)
        .unwrap_or(f64::INFINITY)
}

fn is_market_socket_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.contains("/v1/websocket")
        && (lower.contains("md-") || lower.contains("md.") || lower.contains("md_demo"))
}

fn source_socket_label(messages: &[CapturedMessage]) -> String {
    let _ = messages;
    "tradovate market WebSocket (HAR)".to_string()
}

fn parse_server_items(data: &str) -> Option<Vec<Value>> {
    let payload = data.strip_prefix('a')?;
    serde_json::from_str(payload).ok()
}

fn parse_client_frame(data: &str) -> Option<(&str, &str)> {
    let (endpoint, rest) = data.split_once('\n')?;
    let (_, body) = rest.split_once("\n\n")?;
    Some((endpoint, body))
}

fn collect_subscribed_symbols(messages: &[CapturedMessage]) -> Result<BTreeSet<i64>> {
    let mut symbols = BTreeSet::new();
    for message in messages {
        if message.direction != "send" {
            continue;
        }
        let Some((endpoint, body)) = parse_client_frame(&message.data) else {
            continue;
        };
        if !matches!(endpoint, "md/subscribequote" | "md/subscribedom") {
            continue;
        }
        let value: Value = serde_json::from_str(body)
            .with_context(|| format!("parse {endpoint} request in browser HAR"))?;
        if let Some(symbol) = json_i64(value.get("symbol")) {
            symbols.insert(symbol);
        }
    }
    Ok(symbols)
}

fn collect_contract_metadata(
    messages: &[CapturedMessage],
) -> (
    BTreeMap<i64, ContractDefinition>,
    BTreeMap<i64, ProductDefinition>,
    BTreeMap<i64, MaturityDefinition>,
) {
    let mut contracts = BTreeMap::<i64, ContractDefinition>::new();
    let mut products = BTreeMap::<i64, ProductDefinition>::new();
    let mut maturities = BTreeMap::<i64, MaturityDefinition>::new();
    for message in messages {
        if message.direction != "receive" {
            continue;
        }
        let Some(items) = parse_server_items(&message.data) else {
            continue;
        };
        for item in items {
            let Some(records) = item.get("d").and_then(Value::as_array) else {
                continue;
            };
            for record in records {
                let Some(id) = json_i64(record.get("id")) else {
                    continue;
                };
                if let Some(name) = record.get("name").and_then(Value::as_str)
                    && record.get("contractMaturityId").is_some()
                {
                    contracts.insert(
                        id,
                        ContractDefinition {
                            id,
                            name: name.to_ascii_uppercase(),
                            maturity_id: json_i64(record.get("contractMaturityId")),
                            product_id: None,
                            expiration: None,
                            tick_size: json_f64(record.get("providerTickSize")),
                            value_per_point: None,
                            product_name: None,
                            product_description: None,
                        },
                    );
                }
                if record.get("valuePerPoint").is_some() {
                    products.insert(
                        id,
                        ProductDefinition {
                            name: record
                                .get("name")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            description: record
                                .get("description")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            value_per_point: json_f64(record.get("valuePerPoint")),
                            tick_size: json_f64(record.get("tickSize")),
                        },
                    );
                }
                if record.get("expirationDate").is_some() || record.get("productId").is_some() {
                    maturities.insert(
                        id,
                        MaturityDefinition {
                            product_id: json_i64(record.get("productId")),
                            expiration: record
                                .get("expirationDate")
                                .and_then(Value::as_str)
                                .and_then(parse_date),
                        },
                    );
                }
            }
        }
    }
    (contracts, products, maturities)
}

fn parse_market_capture(
    messages: &[CapturedMessage],
    subscribed_ids: BTreeSet<i64>,
) -> BTreeMap<i64, ContractCapture> {
    let mut captures = subscribed_ids
        .iter()
        .map(|id| (*id, ContractCapture::default()))
        .collect::<BTreeMap<_, _>>();
    let mut quote_state = HashMap::<i64, QuoteState>::new();
    let mut arrival_order = 0_u64;
    for message in messages {
        if message.direction != "receive" {
            continue;
        }
        let Some(items) = parse_server_items(&message.data) else {
            continue;
        };
        for item in items {
            if item.get("e").and_then(Value::as_str) != Some("md") {
                continue;
            }
            let Some(data) = item.get("d") else {
                continue;
            };
            if let Some(quotes) = data.get("q").and_then(Value::as_array) {
                for quote in quotes {
                    let Some(id) = json_i64(quote.get("id")) else {
                        continue;
                    };
                    if !subscribed_ids.contains(&id) {
                        continue;
                    }
                    let state = quote_state.entry(id).or_default();
                    if let Some(value) = json_f64(quote.get("bid")) {
                        state.bid_price = positive(value);
                    }
                    if let Some(value) = json_f64(quote.get("bs")) {
                        state.bid_size = nonnegative(value);
                    }
                    if let Some(value) = json_f64(quote.get("ask")) {
                        state.ask_price = positive(value);
                    }
                    if let Some(value) = json_f64(quote.get("as")) {
                        state.ask_size = nonnegative(value);
                    }
                    let Some(last) = json_f64(quote.get("last")).and_then(positive) else {
                        arrival_order = arrival_order.saturating_add(1);
                        continue;
                    };
                    let Some(size) = json_f64(quote.get("ls")).and_then(positive) else {
                        arrival_order = arrival_order.saturating_add(1);
                        continue;
                    };
                    let Some(ts_ms) = json_i64(quote.get("t")) else {
                        arrival_order = arrival_order.saturating_add(1);
                        continue;
                    };
                    if let Some(capture) = captures.get_mut(&id) {
                        capture.trades.push(CapturedTrade {
                            ts_ms,
                            arrival_order,
                            price: last,
                            size,
                            bid_price: state.bid_price,
                            bid_size: state.bid_size,
                            ask_price: state.ask_price,
                            ask_size: state.ask_size,
                        });
                    }
                    arrival_order = arrival_order.saturating_add(1);
                }
            }
            if let Some(doms) = data.get("d").and_then(Value::as_array) {
                for dom in doms {
                    let Some(id) = json_i64(dom.get("id")) else {
                        continue;
                    };
                    if !subscribed_ids.contains(&id) {
                        continue;
                    }
                    let Some(ts_ms) = json_i64(dom.get("t")) else {
                        arrival_order = arrival_order.saturating_add(1);
                        continue;
                    };
                    let Some(snapshot) = parse_compact_dom(dom) else {
                        arrival_order = arrival_order.saturating_add(1);
                        continue;
                    };
                    if let Some(capture) = captures.get_mut(&id) {
                        capture.dom.push(CapturedDom {
                            ts_ms,
                            arrival_order,
                            snapshot,
                        });
                    }
                    arrival_order = arrival_order.saturating_add(1);
                }
            }
        }
    }
    captures
}

fn parse_compact_dom(value: &Value) -> Option<ReplayMarketDom> {
    let bids = parse_compact_levels(value.get("b"));
    let asks = parse_compact_levels(value.get("a"));
    if bids.is_empty() && asks.is_empty() {
        return None;
    }
    Some(ReplayMarketDom {
        ts_ns: json_i64(value.get("t"))?.checked_mul(1_000_000)?,
        bids,
        asks,
    })
}

fn parse_compact_levels(value: Option<&Value>) -> Vec<ReplayDomLevel> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|level| {
            let values = level.as_array()?;
            let price = json_f64(values.first())?.filter_finite_positive()?;
            let size = json_f64(values.get(1))?.filter_finite_nonnegative()?;
            (size > 0.0).then_some(ReplayDomLevel { price, size })
        })
        .collect()
}

fn write_dom_sidecar(
    manifest_path: &Path,
    dataset_dir: &Path,
    contract_id: i64,
    mut captured: Vec<CapturedDom>,
    source_label: &str,
) -> Result<usize> {
    if captured.is_empty() {
        return Ok(0);
    }
    captured.sort_by(|left, right| {
        left.ts_ms
            .cmp(&right.ts_ms)
            .then_with(|| left.arrival_order.cmp(&right.arrival_order))
    });
    fs::create_dir_all(dataset_dir)
        .with_context(|| format!("create browser DOM dataset {}", dataset_dir.display()))?;
    let relative_path = PathBuf::from("dom_stream.jsonl");
    let data_path = dataset_dir.join(&relative_path);
    let mut bytes = Vec::new();
    for item in &captured {
        serde_json::to_writer(&mut bytes, &item.snapshot).context("serialize browser DOM")?;
        bytes.push(b'\n');
    }
    write_file_atomically(&data_path, &bytes)?;

    let first_timestamp = timestamp_from_millis(captured[0].ts_ms)?;
    let last_timestamp = timestamp_from_millis(
        captured
            .last()
            .map(|item| item.ts_ms)
            .context("DOM capture unexpectedly empty")?,
    )?;
    let mut manifest = ReplayCacheManifest::from_path(manifest_path)
        .with_context(|| format!("load imported replay manifest {}", manifest_path.display()))?;
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::DomStream
            && file.relative_path == relative_path)
    });
    manifest.files.push(ReplayCacheDataFile {
        relative_path,
        source_kind: ReplayCacheSourceKind::DomStream,
        format: ReplayCacheFileFormat::Jsonl,
        schema_version: Some(1),
        compression: None,
        market_shape: ReplayCacheMarketShape {
            bar_type: None,
            chart_mode: None,
            session_template: None,
        },
        row_count: captured.len() as u64,
        first_timestamp,
        last_timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(crate::replay_cache::ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: fnv1a64_file_hex(&data_path)?,
        }),
        warnings: vec![
            format!(
                "Visible DOM snapshots captured for contract id {contract_id}; depth is widget-visible, not the complete exchange book."
            ),
            format!("Source: {source_label}"),
        ],
        errors: Vec::new(),
    });
    manifest
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    manifest.source_kind = ReplayCacheSourceKind::Mixed;
    manifest.coverage = ReplayCacheCoverage {
        start: manifest
            .files
            .iter()
            .map(|file| file.first_timestamp)
            .min()
            .unwrap_or(first_timestamp),
        end: manifest
            .files
            .iter()
            .map(|file| file.last_timestamp)
            .max()
            .unwrap_or(last_timestamp),
        trading_date: Some(first_timestamp.date_naive()),
    };
    manifest
        .warnings
        .push("Browser HAR contains visible DOM snapshots in dom_stream.jsonl; set replay_dom_file_path to this file for DOM fill replay.".to_string());
    manifest.warnings.sort();
    manifest.warnings.dedup();
    manifest.tags.push("visible-dom".to_string());
    manifest.tags.sort();
    manifest.tags.dedup();
    manifest.badges = manifest.derived_badges();
    write_manifest_atomically(manifest_path, &manifest)
        .with_context(|| format!("write browser DOM manifest {}", manifest_path.display()))?;
    Ok(captured.len())
}

fn write_manifest_atomically(path: &Path, manifest: &ReplayCacheManifest) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(manifest).context("serialize browser HAR manifest")?;
    let temp = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("manifest.json"),
        std::process::id()
    ));
    write_file_atomically(&temp, &bytes)?;
    fs::rename(&temp, path).with_context(|| {
        format!(
            "replace browser HAR manifest {} with {}",
            path.display(),
            temp.display()
        )
    })?;
    Ok(())
}

fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    {
        let mut file = File::create(&temp).with_context(|| format!("create {}", temp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("write {}", temp.display()))?;
        file.sync_all()
            .with_context(|| format!("flush {}", temp.display()))?;
    }
    fs::rename(&temp, path)
        .with_context(|| format!("replace {} with {}", path.display(), temp.display()))?;
    Ok(())
}

fn fnv1a64_file_hex(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {} for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn resolve_environment(
    raw: &str,
    entries: &[HarEntry],
    fallback: TradingEnvironment,
) -> Result<ImportEnvironment> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "sim" | "simulation" | "demo" => Ok(ImportEnvironment {
            value: TradingEnvironment::Sim,
            source: "--environment".to_string(),
        }),
        "live" | "production" => Ok(ImportEnvironment {
            value: TradingEnvironment::Live,
            source: "--environment".to_string(),
        }),
        "auto" | "" => {
            let market_url = entries
                .iter()
                .map(|entry| entry.request.url.to_ascii_lowercase())
                .find(|url| is_market_socket_url(url));
            if market_url
                .as_deref()
                .is_some_and(|url| url.contains("demo"))
            {
                Ok(ImportEnvironment {
                    value: TradingEnvironment::Sim,
                    source: "market WebSocket URL".to_string(),
                })
            } else if market_url.is_some() {
                Ok(ImportEnvironment {
                    value: TradingEnvironment::Live,
                    source: "market WebSocket URL".to_string(),
                })
            } else {
                Ok(ImportEnvironment {
                    value: fallback,
                    source: "configured environment".to_string(),
                })
            }
        }
        other => bail!("invalid --environment `{other}`; use auto, sim, or live"),
    }
}

fn normalize_contract_filter(values: &[String]) -> BTreeSet<String> {
    values
        .iter()
        .map(|value| value.trim().to_ascii_uppercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn instrument_root(contract: &str) -> String {
    let contract = contract.trim().to_ascii_uppercase();
    let chars = contract.chars().collect::<Vec<_>>();
    for index in 1..chars.len() {
        if !matches!(
            chars[index],
            'F' | 'G' | 'H' | 'J' | 'K' | 'M' | 'N' | 'Q' | 'U' | 'V' | 'X' | 'Z'
        ) {
            continue;
        }
        let suffix = &chars[index + 1..];
        if (1..=2).contains(&suffix.len())
            && suffix.iter().all(|character| character.is_ascii_digit())
        {
            return chars[..index].iter().collect();
        }
    }
    contract
}

fn timestamp_from_millis(value: i64) -> Result<DateTime<Utc>> {
    DateTime::<Utc>::from_timestamp_millis(value)
        .with_context(|| format!("invalid browser market timestamp {value} ms"))
}

fn parse_date(value: &str) -> Option<NaiveDate> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|date| date.date_naive())
        .or_else(|| NaiveDate::parse_from_str(value, "%Y-%m-%d").ok())
}

fn json_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(Value::as_i64).or_else(|| {
        value
            .and_then(Value::as_u64)
            .and_then(|value| i64::try_from(value).ok())
    })
}

fn json_f64(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .or_else(|| value.and_then(Value::as_i64).map(|value| value as f64))
        .or_else(|| value.and_then(Value::as_u64).map(|value| value as f64))
}

fn positive(value: f64) -> Option<f64> {
    value
        .is_finite()
        .then_some(value)
        .filter(|value| *value > 0.0)
}

fn nonnegative(value: f64) -> Option<f64> {
    value
        .is_finite()
        .then_some(value)
        .filter(|value| *value >= 0.0)
}

trait FiniteValue {
    fn filter_finite_positive(self) -> Option<f64>;
    fn filter_finite_nonnegative(self) -> Option<f64>;
}

impl FiniteValue for f64 {
    fn filter_finite_positive(self) -> Option<f64> {
        positive(self)
    }

    fn filter_finite_nonnegative(self) -> Option<f64> {
        nonnegative(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(direction: &str, data: &str, order: u64) -> CapturedMessage {
        CapturedMessage {
            direction: direction.to_string(),
            time: order as f64,
            data: data.to_string(),
            order,
        }
    }

    #[test]
    fn compact_dom_maps_visible_levels_to_replay_dom() {
        let value = json!({
            "id": 7,
            "t": 1_700_000_000_123_i64,
            "b": [[100.0, 4], [99.75, 2]],
            "a": [[100.25, 3], [100.5, 1]],
            "r": true
        });
        let dom = parse_compact_dom(&value).expect("DOM");
        assert_eq!(dom.ts_ns, 1_700_000_000_123_000_000);
        assert_eq!(
            dom.bids[0],
            ReplayDomLevel {
                price: 100.0,
                size: 4.0
            }
        );
        assert_eq!(
            dom.asks[1],
            ReplayDomLevel {
                price: 100.5,
                size: 1.0
            }
        );
    }

    #[test]
    fn market_parser_keeps_quote_last_rows_and_carries_latest_book() {
        let messages = vec![
            message(
                "receive",
                r#"a[{"e":"md","d":{"q":[{"id":7,"t":1700000000000,"bid":100,"bs":4,"ask":100.25,"as":2}]}}]"#,
                1,
            ),
            message(
                "receive",
                r#"a[{"e":"md","d":{"q":[{"id":7,"t":1700000000123,"last":100.25,"ls":3}]}}]"#,
                2,
            ),
        ];
        let captures = parse_market_capture(&messages, BTreeSet::from([7]));
        let trades = &captures.get(&7).expect("capture").trades;
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100.25);
        assert_eq!(trades[0].size, 3.0);
        assert_eq!(trades[0].bid_price, Some(100.0));
        assert_eq!(trades[0].ask_size, Some(2.0));
    }

    #[test]
    fn instrument_root_handles_common_futures_symbols() {
        assert_eq!(instrument_root("ESU6"), "ES");
        assert_eq!(instrument_root("MESU6"), "MES");
        assert_eq!(instrument_root("GCZ6"), "GC");
        assert_eq!(instrument_root("MGCQ6"), "MGC");
    }

    #[test]
    fn subscription_parser_ignores_heartbeats_and_reads_multiple_symbols() {
        let messages = vec![
            message("send", "[]", 1),
            message("send", "md/subscribequote\n1\n\n{\"symbol\":7}", 2),
            message("send", "md/subscribedom\n2\n\n{\"symbol\":8}", 3),
        ];
        assert_eq!(
            collect_subscribed_symbols(&messages).expect("subscriptions"),
            BTreeSet::from([7, 8])
        );
    }
}
