//! Raw-tick fixture loading for the replay proxy.
//!
//! This module deliberately does not sort or deduplicate input.  A replay
//! fixture is an execution trace, so source order is part of its meaning.  A
//! caller that wants timestamp ordering or duplicate rejection must select it
//! explicitly through [`TickLoadOptions`].

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use chrono::{DateTime, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

const MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_PARQUET_BATCH_ROWS: usize = 16_384;

/// The ordering contract applied after decoding a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TickOrdering {
    /// Keep rows exactly in source order.  Equal timestamps are valid.
    PreserveSource,
    /// Require non-decreasing `ts_ns`, while allowing multiple ticks at one
    /// timestamp.
    NonDecreasingTimestamp,
    /// Require strictly increasing `ts_ns`.
    StrictTimestamp,
}

/// The duplicate-ID policy applied after decoding a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DuplicateTickIds {
    /// Do not inspect IDs for duplicates.  Source rows are still preserved.
    Allow,
    /// Reject a repeated `tick_id`.  Rows without an ID are preserved because
    /// there is no stable identity with which to prove that they duplicate.
    Reject,
}

/// Controls for loading a raw-tick fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickLoadOptions {
    /// Zero means no limit.  The limit is applied in source order.
    pub max_ticks: usize,
    pub ordering: TickOrdering,
    pub duplicate_tick_ids: DuplicateTickIds,
}

impl Default for TickLoadOptions {
    fn default() -> Self {
        Self {
            max_ticks: 0,
            ordering: TickOrdering::PreserveSource,
            duplicate_tick_ids: DuplicateTickIds::Reject,
        }
    }
}

/// A raw trade/quote observation accepted by the fixture loader.
///
/// The fields match the proxy’s repository raw-tick Parquet schema.  The
/// metadata fields are optional because provider responses and hand-authored
/// JSONL/CSV fixtures do not always contain them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawTick {
    pub timestamp: DateTime<Utc>,
    pub ts_ns: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick_id: Option<i64>,
    pub price: f64,
    pub size: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trade_date: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_ts_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_price_ticks: Option<i64>,
}

/// Loaded ticks plus the source format used to decode them.
#[derive(Debug, Clone, PartialEq)]
pub struct TickFixture {
    ticks: Vec<RawTick>,
    pub format: TickFixtureFormat,
}

#[allow(dead_code)]
impl TickFixture {
    pub fn as_slice(&self) -> &[RawTick] {
        &self.ticks
    }

    pub fn into_ticks(self) -> Vec<RawTick> {
        self.ticks
    }

    pub fn len(&self) -> usize {
        self.ticks.len()
    }
}

/// Input encoding selected by [`load`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickFixtureFormat {
    Jsonl,
    Csv,
    Parquet,
}

/// Load JSONL, CSV, or the repository raw-tick Parquet schema based on the
/// file extension.  Unknown extensions are treated as JSONL.
pub fn load(path: &Path, options: TickLoadOptions) -> Result<TickFixture> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match extension.as_str() {
        "parquet" => load_parquet(path, options),
        "csv" => {
            let file = File::open(path)
                .with_context(|| format!("open raw-tick CSV fixture {}", path.display()))?;
            load_csv_reader(BufReader::new(file), options)
                .with_context(|| format!("decode raw-tick CSV fixture {}", path.display()))
        }
        _ => {
            let file = File::open(path)
                .with_context(|| format!("open raw-tick JSONL fixture {}", path.display()))?;
            load_jsonl_reader(BufReader::new(file), options)
                .with_context(|| format!("decode raw-tick JSONL fixture {}", path.display()))
        }
    }
}

/// Load a JSONL reader without reordering its rows.
pub fn load_jsonl_reader<R: BufRead>(reader: R, options: TickLoadOptions) -> Result<TickFixture> {
    let mut ticks = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        if reached_limit(ticks.len(), options.max_ticks) {
            break;
        }
        let line = line.with_context(|| format!("read JSONL line {}", line_index + 1))?;
        if line.len() > MAX_RECORD_BYTES {
            bail!(
                "JSONL line {} exceeds {} bytes",
                line_index + 1,
                MAX_RECORD_BYTES
            );
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: JsonTick = serde_json::from_str(trimmed)
            .with_context(|| format!("parse JSONL line {}", line_index + 1))?;
        ticks.push(
            row.into_tick()
                .with_context(|| format!("decode JSONL raw tick line {}", line_index + 1))?,
        );
    }
    finish(ticks, TickFixtureFormat::Jsonl, options)
}

/// Load a CSV reader without reordering its rows.  Required columns are
/// `price` and `size`, plus either `timestamp` or `ts_ns`; all repository
/// metadata columns are optional.
pub fn load_csv_reader<R: Read>(reader: R, options: TickLoadOptions) -> Result<TickFixture> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .flexible(false)
        .from_reader(reader);
    let header = csv_reader
        .headers()
        .context("read raw-tick CSV header")?
        .clone();
    let columns = header
        .iter()
        .map(|value| value.trim_start_matches('\u{feff}').to_ascii_lowercase())
        .collect::<Vec<_>>();
    let find = |name: &str| columns.iter().position(|column| column == name);
    let required = |name: &str| find(name).with_context(|| format!("CSV missing {name} column"));
    let timestamp_index = find("timestamp")
        .or_else(|| find("time"))
        .or_else(|| find("ts"));
    let ts_ns_index = find("ts_ns").or_else(|| find("timestamp_ns"));
    if timestamp_index.is_none() && ts_ns_index.is_none() {
        bail!("CSV requires timestamp or ts_ns column");
    }
    let price_index = required("price")?;
    let size_index = required("size")?;
    let mut ticks = Vec::new();
    for (row_index, record) in csv_reader.records().enumerate() {
        if reached_limit(ticks.len(), options.max_ticks) {
            break;
        }
        let record = record.with_context(|| format!("read CSV row {}", row_index + 2))?;
        let field = |index: usize| {
            record
                .get(index)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .with_context(|| format!("CSV row {} has an empty field", row_index + 2))
        };
        let optional = |name: &str| -> Option<&str> {
            find(name)
                .and_then(|index| record.get(index))
                .map(str::trim)
                .filter(|value| !value.is_empty())
        };
        let timestamp = timestamp_index
            .map(|index| field(index).and_then(parse_timestamp))
            .transpose()?;
        let ts_ns = ts_ns_index
            .map(|index| field(index).and_then(|value| parse_i64(value, "ts_ns")))
            .transpose()?;
        let (timestamp, ts_ns) = timestamp_and_ns(timestamp, ts_ns)?;
        let parse_f64 = |name: &str, index: usize| {
            field(index)?
                .parse::<f64>()
                .with_context(|| format!("parse CSV {name}"))
        };
        let optional_f64 = |name: &str| -> Result<Option<f64>> {
            optional(name)
                .map(|value| {
                    value
                        .parse::<f64>()
                        .with_context(|| format!("parse CSV {name}"))
                })
                .transpose()
        };
        let optional_i64 = |name: &str| -> Result<Option<i64>> {
            optional(name)
                .map(|value| parse_i64(value, name))
                .transpose()
        };
        let optional_i32 = |name: &str| -> Result<Option<i32>> {
            optional(name)
                .map(|value| {
                    value
                        .parse::<i32>()
                        .with_context(|| format!("parse CSV {name}"))
                })
                .transpose()
        };
        ticks.push(RawTick {
            timestamp,
            ts_ns,
            tick_id: optional_i64("tick_id")?,
            price: parse_f64("price", price_index)?,
            size: parse_f64("size", size_index)?,
            bid_price: optional_f64("bid_price")?,
            bid_size: optional_f64("bid_size")?,
            ask_price: optional_f64("ask_price")?,
            ask_size: optional_f64("ask_size")?,
            chart_id: optional_i64("chart_id")?,
            trade_date: optional_i32("trade_date")?,
            packet_source: optional("packet_source").map(str::to_owned),
            packet_base_ts_ms: optional_i64("packet_base_ts_ms")?,
            packet_base_price_ticks: optional_i64("packet_base_price_ticks")?,
        });
    }
    finish(ticks, TickFixtureFormat::Csv, options)
}

fn load_parquet(path: &Path, options: TickLoadOptions) -> Result<TickFixture> {
    let file = File::open(path)
        .with_context(|| format!("open raw-tick Parquet fixture {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open raw-tick Parquet reader {}", path.display()))?
        .with_batch_size(DEFAULT_PARQUET_BATCH_ROWS)
        .build()
        .with_context(|| format!("build raw-tick Parquet reader {}", path.display()))?;
    let mut ticks = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("read Parquet batch {}", path.display()))?;
        append_parquet_batch(&mut ticks, &batch, path, options.max_ticks)?;
        if reached_limit(ticks.len(), options.max_ticks) {
            break;
        }
    }
    finish(ticks, TickFixtureFormat::Parquet, options)
}

fn append_parquet_batch(
    ticks: &mut Vec<RawTick>,
    batch: &RecordBatch,
    _path: &Path,
    max_ticks: usize,
) -> Result<()> {
    let schema = batch.schema();
    let column = |name: &str| -> Result<&dyn Array> {
        let index = schema
            .index_of(name)
            .with_context(|| format!("Parquet fixture missing {name} column"))?;
        Ok(batch.column(index).as_ref())
    };
    let optional_column = |name: &str| {
        schema
            .index_of(name)
            .ok()
            .map(|index| batch.column(index).as_ref())
    };
    let timestamps = string_column(column("timestamp")?, "timestamp")?;
    let ts_ns = int64_column(column("ts_ns")?, "ts_ns")?;
    // Older replay-cache Parquet files have a nullable or absent tick_id.
    // Source order plus the proxy sequence still gives them deterministic
    // replay identity; a present ID is retained for trace correlation.
    let tick_ids = optional_column("tick_id")
        .map(|array| int64_column(array, "tick_id"))
        .transpose()?;
    let prices = f64_column(column("price")?, "price")?;
    let sizes = f64_column(column("size")?, "size")?;
    let bid_prices = optional_column("bid_price")
        .map(|array| f64_column(array, "bid_price"))
        .transpose()?;
    let bid_sizes = optional_column("bid_size")
        .map(|array| f64_column(array, "bid_size"))
        .transpose()?;
    let ask_prices = optional_column("ask_price")
        .map(|array| f64_column(array, "ask_price"))
        .transpose()?;
    let ask_sizes = optional_column("ask_size")
        .map(|array| f64_column(array, "ask_size"))
        .transpose()?;
    let chart_ids = optional_column("chart_id")
        .map(|array| int64_column(array, "chart_id"))
        .transpose()?;
    let trade_dates = optional_column("trade_date")
        .map(|array| int32_column(array, "trade_date"))
        .transpose()?;
    let packet_sources = optional_column("packet_source")
        .map(|array| string_column(array, "packet_source"))
        .transpose()?;
    let packet_base_ts_ms = optional_column("packet_base_ts_ms")
        .map(|array| int64_column(array, "packet_base_ts_ms"))
        .transpose()?;
    let packet_base_price_ticks = optional_column("packet_base_price_ticks")
        .map(|array| int64_column(array, "packet_base_price_ticks"))
        .transpose()?;

    for row_index in 0..batch.num_rows() {
        if reached_limit(ticks.len(), max_ticks) {
            break;
        }
        let timestamp = parse_timestamp(timestamps.value(row_index))
            .with_context(|| format!("parse Parquet timestamp row {row_index}"))?;
        let tick = RawTick {
            timestamp,
            ts_ns: required_i64(ts_ns, row_index, "ts_ns")?,
            tick_id: optional_i64(tick_ids, row_index),
            price: required_f64(prices, row_index, "price")?,
            size: required_f64(sizes, row_index, "size")?,
            bid_price: optional_f64(bid_prices, row_index),
            bid_size: optional_f64(bid_sizes, row_index),
            ask_price: optional_f64(ask_prices, row_index),
            ask_size: optional_f64(ask_sizes, row_index),
            chart_id: optional_i64(chart_ids, row_index),
            trade_date: optional_i32(trade_dates, row_index),
            packet_source: optional_string(packet_sources, row_index),
            packet_base_ts_ms: optional_i64(packet_base_ts_ms, row_index),
            packet_base_price_ticks: optional_i64(packet_base_price_ticks, row_index),
        };
        ticks.push(tick);
    }
    Ok(())
}

fn finish(
    mut ticks: Vec<RawTick>,
    format: TickFixtureFormat,
    options: TickLoadOptions,
) -> Result<TickFixture> {
    if ticks.is_empty() {
        bail!("raw-tick fixture contains no ticks");
    }
    let mut previous_ts_ns = None;
    let mut seen_ids = HashSet::new();
    for (index, tick) in ticks.iter().enumerate() {
        validate_tick(tick).with_context(|| format!("validate raw tick row {index}"))?;
        match options.ordering {
            TickOrdering::PreserveSource => {}
            TickOrdering::NonDecreasingTimestamp => {
                if previous_ts_ns.is_some_and(|previous| tick.ts_ns < previous) {
                    bail!("raw ticks are not ordered by timestamp at row {index}");
                }
            }
            TickOrdering::StrictTimestamp => {
                if previous_ts_ns.is_some_and(|previous| tick.ts_ns <= previous) {
                    bail!("raw ticks are not strictly ordered by timestamp at row {index}");
                }
            }
        }
        previous_ts_ns = Some(tick.ts_ns);
        if options.duplicate_tick_ids == DuplicateTickIds::Reject
            && tick.tick_id.is_some_and(|id| !seen_ids.insert(id))
        {
            bail!("duplicate raw tick_id at row {index}");
        }
    }
    if options.max_ticks > 0 && ticks.len() > options.max_ticks {
        ticks.truncate(options.max_ticks);
    }
    Ok(TickFixture { ticks, format })
}

fn validate_tick(tick: &RawTick) -> Result<()> {
    if tick.ts_ns <= 0 {
        bail!("raw tick timestamp nanoseconds must be positive");
    }
    let timestamp_ns = tick
        .timestamp
        .timestamp_nanos_opt()
        .context("raw tick timestamp is outside supported nanosecond range")?;
    if timestamp_ns != tick.ts_ns {
        bail!("raw tick timestamp disagrees with ts_ns");
    }
    if !(tick.price.is_finite() && tick.price > 0.0) {
        bail!("raw tick price must be finite and positive");
    }
    validate_nonnegative(tick.size, "size")?;
    validate_optional_positive(tick.bid_price, "bid_price")?;
    validate_optional_nonnegative(tick.bid_size, "bid_size")?;
    validate_optional_positive(tick.ask_price, "ask_price")?;
    validate_optional_nonnegative(tick.ask_size, "ask_size")?;
    if let (Some(bid), Some(ask)) = (tick.bid_price, tick.ask_price)
        && bid > ask
    {
        bail!("raw tick quote is crossed: bid {bid} > ask {ask}");
    }
    Ok(())
}

fn validate_optional_positive(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(value) = value
        && !(value.is_finite() && value > 0.0)
    {
        bail!("raw tick {name} must be finite and positive");
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, name: &str) -> Result<()> {
    if let Some(value) = value {
        validate_nonnegative(value, name)?;
    }
    Ok(())
}

fn validate_nonnegative(value: f64, name: &str) -> Result<()> {
    if !(value.is_finite() && value >= 0.0) {
        bail!("raw tick {name} must be finite and non-negative");
    }
    Ok(())
}

fn reached_limit(current: usize, limit: usize) -> bool {
    limit > 0 && current >= limit
}

fn timestamp_and_ns(
    timestamp: Option<DateTime<Utc>>,
    ts_ns: Option<i64>,
) -> Result<(DateTime<Utc>, i64)> {
    let timestamp = timestamp
        .or_else(|| ts_ns.map(DateTime::<Utc>::from_timestamp_nanos))
        .context("raw tick requires timestamp or ts_ns")?;
    let ts_ns = ts_ns
        .or_else(|| timestamp.timestamp_nanos_opt())
        .context("raw tick timestamp is outside nanosecond range")?;
    Ok((timestamp, ts_ns))
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .or_else(|_| {
            DateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f%:z")
                .map(|value| value.with_timezone(&Utc))
        })
        .with_context(|| format!("parse timestamp {value}"))
}

fn parse_i64(value: &str, name: &str) -> Result<i64> {
    value
        .parse::<i64>()
        .with_context(|| format!("parse {name}"))
}

#[derive(Debug, Deserialize)]
struct JsonTick {
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    ts_ns: Option<i64>,
    #[serde(default)]
    tick_id: Option<i64>,
    price: f64,
    size: f64,
    #[serde(default)]
    bid_price: Option<f64>,
    #[serde(default)]
    bid_size: Option<f64>,
    #[serde(default)]
    ask_price: Option<f64>,
    #[serde(default)]
    ask_size: Option<f64>,
    #[serde(default)]
    chart_id: Option<i64>,
    #[serde(default)]
    trade_date: Option<i32>,
    #[serde(default)]
    packet_source: Option<String>,
    #[serde(default)]
    packet_base_ts_ms: Option<i64>,
    #[serde(default)]
    packet_base_price_ticks: Option<i64>,
}

impl JsonTick {
    fn into_tick(self) -> Result<RawTick> {
        let timestamp = self.timestamp.as_deref().map(parse_timestamp).transpose()?;
        let (timestamp, ts_ns) = timestamp_and_ns(timestamp, self.ts_ns)?;
        Ok(RawTick {
            timestamp,
            ts_ns,
            tick_id: self.tick_id,
            price: self.price,
            size: self.size,
            bid_price: self.bid_price,
            bid_size: self.bid_size,
            ask_price: self.ask_price,
            ask_size: self.ask_size,
            chart_id: self.chart_id,
            trade_date: self.trade_date,
            packet_source: self.packet_source,
            packet_base_ts_ms: self.packet_base_ts_ms,
            packet_base_price_ticks: self.packet_base_price_ticks,
        })
    }
}

fn string_column<'a>(array: &'a dyn Array, name: &str) -> Result<&'a StringArray> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("Parquet {name} must be UTF-8"))
}

fn int64_column<'a>(array: &'a dyn Array, name: &str) -> Result<&'a Int64Array> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .with_context(|| format!("Parquet {name} must be INT64"))
}

fn int32_column<'a>(array: &'a dyn Array, name: &str) -> Result<&'a Int32Array> {
    array
        .as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("Parquet {name} must be INT32"))
}

fn f64_column<'a>(array: &'a dyn Array, name: &str) -> Result<&'a Float64Array> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("Parquet {name} must be FLOAT64"))
}

fn required_i64(array: &Int64Array, index: usize, name: &str) -> Result<i64> {
    if array.is_null(index) {
        bail!("Parquet required {name} is null at row {index}");
    }
    Ok(array.value(index))
}

fn required_f64(array: &Float64Array, index: usize, name: &str) -> Result<f64> {
    if array.is_null(index) {
        bail!("Parquet required {name} is null at row {index}");
    }
    Ok(array.value(index))
}

fn optional_i64(array: Option<&Int64Array>, index: usize) -> Option<i64> {
    array
        .filter(|array| !array.is_null(index))
        .map(|array| array.value(index))
}

fn optional_i32(array: Option<&Int32Array>, index: usize) -> Option<i32> {
    array
        .filter(|array| !array.is_null(index))
        .map(|array| array.value(index))
}

fn optional_f64(array: Option<&Float64Array>, index: usize) -> Option<f64> {
    array
        .filter(|array| !array.is_null(index))
        .map(|array| array.value(index))
}

fn optional_string(array: Option<&StringArray>, index: usize) -> Option<String> {
    array
        .filter(|array| !array.is_null(index))
        .map(|array| array.value(index).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const T0: i64 = 1_700_000_000_000_000_000;

    fn json_tick(ts_ns: i64, id: i64, price: f64, size: f64) -> String {
        format!(r#"{{"ts_ns":{ts_ns},"tick_id":{id},"price":{price},"size":{size}}}"#)
    }

    #[test]
    fn jsonl_derives_timestamp_and_preserves_source_order() {
        let input = format!(
            "{}\n{}\n",
            json_tick(T0 + 2, 2, 10.0, 0.0),
            json_tick(T0 + 1, 1, 11.0, 1.5)
        );
        let fixture = load_jsonl_reader(Cursor::new(input), TickLoadOptions::default()).unwrap();
        assert_eq!(fixture.format, TickFixtureFormat::Jsonl);
        assert_eq!(fixture.as_slice()[0].tick_id, Some(2));
        assert_eq!(fixture.as_slice()[1].tick_id, Some(1));
        assert_eq!(
            fixture.as_slice()[0].timestamp.timestamp_nanos_opt(),
            Some(T0 + 2)
        );
    }

    #[test]
    fn ordering_policy_rejects_backwards_timestamp() {
        let input = format!(
            "{}\n{}\n",
            json_tick(T0 + 2, 2, 10.0, 1.0),
            json_tick(T0 + 1, 1, 11.0, 1.0)
        );
        let options = TickLoadOptions {
            ordering: TickOrdering::NonDecreasingTimestamp,
            ..TickLoadOptions::default()
        };
        let error = load_jsonl_reader(Cursor::new(input), options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not ordered by timestamp"));
    }

    #[test]
    fn duplicate_policy_rejects_repeated_tick_id_but_allows_equal_timestamps() {
        let input = format!(
            "{}\n{}\n",
            json_tick(T0, 7, 10.0, 1.0),
            json_tick(T0, 7, 10.1, 2.0)
        );
        let error = load_jsonl_reader(Cursor::new(input), TickLoadOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate raw tick_id"));

        let input = format!(
            "{}\n{}\n",
            json_tick(T0, 7, 10.0, 1.0),
            json_tick(T0, 8, 10.1, 2.0)
        );
        let fixture = load_jsonl_reader(Cursor::new(input), TickLoadOptions::default()).unwrap();
        assert_eq!(fixture.len(), 2);

        let input = format!(
            "{}\n{}\n",
            json_tick(T0, 7, 10.0, 1.0),
            json_tick(T0, 7, 10.1, 2.0)
        );
        let options = TickLoadOptions {
            duplicate_tick_ids: DuplicateTickIds::Allow,
            ordering: TickOrdering::StrictTimestamp,
            ..TickLoadOptions::default()
        };
        let error = load_jsonl_reader(Cursor::new(input), options).unwrap_err();
        assert!(format!("{error:#}").contains("strictly ordered"));
    }

    #[test]
    fn invalid_size_and_timestamp_are_rejected() {
        let input = format!("{}\n", json_tick(0, 1, 10.0, f64::NAN));
        let error = load_jsonl_reader(Cursor::new(input), TickLoadOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("parse JSONL") || error.contains("timestamp"));

        let input = format!("{}\n", json_tick(T0, 1, 10.0, -1.0));
        let error = load_jsonl_reader(Cursor::new(input), TickLoadOptions::default()).unwrap_err();
        assert!(format!("{error:#}").contains("non-negative"));
    }

    #[test]
    fn csv_reads_optional_metadata_and_limit() {
        let input = format!(
            "timestamp,ts_ns,tick_id,price,size,bid_price,bid_size,ask_price,ask_size,packet_source\n2023-11-14T22:13:20Z,{T0},4,10.0,1.0,9.9,2.0,10.1,3.0,fixture\n2023-11-14T22:13:20.000000001Z,{next},5,10.2,0.0,,,,,fixture\n",
            next = T0 + 1
        );
        let options = TickLoadOptions {
            max_ticks: 1,
            ..TickLoadOptions::default()
        };
        let fixture = load_csv_reader(Cursor::new(input), options).unwrap();
        assert_eq!(fixture.format, TickFixtureFormat::Csv);
        assert_eq!(fixture.len(), 1);
        assert_eq!(
            fixture.as_slice()[0].packet_source.as_deref(),
            Some("fixture")
        );
        assert_eq!(fixture.as_slice()[0].bid_size, Some(2.0));
    }

    #[test]
    fn timestamp_and_ts_ns_must_agree() {
        let input = format!(
            "timestamp,ts_ns,price,size\n2023-11-14T22:13:20Z,{},10.0,1.0\n",
            T0 + 1
        );
        let error = load_csv_reader(Cursor::new(input), TickLoadOptions::default()).unwrap_err();
        assert!(format!("{error:#}").contains("disagrees"));
    }
}
