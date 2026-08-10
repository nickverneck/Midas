//! Import exact-contract Databento trades archives into the replay raw-tick cache.
//!
//! The downloader intentionally stores provider ZIPs as durable source material.
//! This module is the separate, offline conversion step: it filters one exact
//! contract and UTC window, normalizes Databento nanosecond timestamps and
//! fixed-point prices, then uses the same Parquet/manifest writer as all other
//! raw-tick replay sources.  The imported cache is marked as a Tradovate
//! compatible replay dataset so the existing replay engine can consume it; the
//! manifest's download_request and notes retain the Databento provenance.

use crate::broker::BrokerKind;
use crate::cli::ImportDatabentoTradesArgs;
use crate::config::TradingEnvironment;
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheInstrument, ReplayCacheRawTickRow, ReplayCacheRawTicksWrite,
    ReplayCacheTickSpecs, write_raw_ticks_parquet_cache,
};
use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, NaiveDate, NaiveDateTime, SecondsFormat, Utc};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use zip::ZipArchive;

const DATABENTO_PACKET_SOURCE: &str = "databento_glbx_mdp3_trades";

#[derive(Debug, Clone, Copy)]
struct ImportWindow {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ParsedTrade {
    ts_ns: i64,
    price: f64,
    size: f64,
    sequence: i64,
    source_order: u64,
}

#[derive(Debug, Clone, Copy)]
struct TradeColumns {
    ts_event: usize,
    price: usize,
    size: usize,
    sequence: usize,
    symbol: usize,
}

#[derive(Debug, Default)]
struct ParseStats {
    files: usize,
    input_rows: u64,
    selected_rows: u64,
    filtered_rows: u64,
    malformed_rows: u64,
    symbols: BTreeSet<String>,
}

/// Convert a Databento trades ZIP into one isolated raw-tick replay dataset.
pub(crate) fn import_databento_trades(args: ImportDatabentoTradesArgs) -> Result<()> {
    ensure!(
        args.input.is_file(),
        "Databento input is not a file: {}",
        args.input.display()
    );

    let request = read_sibling_json(&args.input, "request.json");
    let job = read_sibling_json(&args.input, "job.json");
    let contract = args
        .contract
        .as_deref()
        .or_else(|| {
            request
                .as_ref()
                .and_then(|value| value.get("symbol"))
                .and_then(Value::as_str)
        })
        .map(normalize_contract)
        .filter(|value| !value.is_empty());
    let window = resolve_window(&args, job.as_ref(), request.as_ref())?;

    let mut trades = Vec::new();
    let mut stats = ParseStats::default();
    parse_archive(
        &args.input,
        window,
        contract.as_deref(),
        &mut trades,
        &mut stats,
    )?;
    ensure!(
        stats.files > 0,
        "{} contains no Databento *.trades.csv entries",
        args.input.display()
    );
    ensure!(
        !trades.is_empty(),
        "no trades for {} in [{} , {})",
        contract.as_deref().unwrap_or("the archive"),
        window.start,
        window.end
    );

    let resolved_contract = contract
        .or_else(|| stats.symbols.iter().next().cloned())
        .context("could not infer an exact contract symbol from the archive")?;
    if stats.symbols.len() > 1 {
        bail!(
            "Databento archive contains multiple symbols in the selected window: {}; pass a ZIP with one exact raw symbol",
            stats.symbols.iter().cloned().collect::<Vec<_>>().join(", ")
        );
    }
    if let Some(symbol) = stats.symbols.iter().next()
        && !symbol.eq_ignore_ascii_case(&resolved_contract)
    {
        bail!(
            "archive symbol {} does not match requested contract {}",
            symbol,
            resolved_contract
        );
    }

    trades.sort_by(|left, right| {
        left.ts_ns
            .cmp(&right.ts_ns)
            .then_with(|| left.sequence.cmp(&right.sequence))
            .then_with(|| left.source_order.cmp(&right.source_order))
    });
    let ticks = trades
        .into_iter()
        .enumerate()
        .map(|(index, trade)| {
            let timestamp = DateTime::<Utc>::from_timestamp_nanos(trade.ts_ns);
            let tick_id = i64::try_from(index + 1)
                .context("Databento import exceeds i64 tick-id capacity")?;
            Ok(ReplayCacheRawTickRow {
                timestamp,
                ts_ns: trade.ts_ns,
                tick_id: Some(tick_id),
                price: trade.price,
                size: trade.size,
                bid_price: None,
                bid_size: None,
                ask_price: None,
                ask_size: None,
                chart_id: None,
                trade_date: timestamp.format("%Y%m%d").to_string().parse::<i32>().ok(),
                packet_source: Some(DATABENTO_PACKET_SOURCE.to_string()),
                packet_base_ts_ms: None,
                packet_base_price_ticks: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let instrument = args
        .instrument
        .as_deref()
        .map(normalize_instrument)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| instrument_root(&resolved_contract));
    ensure!(!instrument.is_empty(), "instrument root cannot be empty");
    let (default_tick_size, default_value_per_point) = default_specs(&instrument);
    let tick_size = args.tick_size.unwrap_or(default_tick_size);
    let value_per_point = args.value_per_point.unwrap_or(default_value_per_point);
    ensure!(
        tick_size.is_finite() && tick_size > 0.0,
        "--tick-size must be finite and positive"
    );
    ensure!(
        value_per_point.is_finite() && value_per_point > 0.0,
        "--value-per-point must be finite and positive"
    );

    let request_value = json!({
        "provider": "databento",
        "dataset": request.as_ref().and_then(|v| v.get("dataset")).and_then(Value::as_str).unwrap_or("GLBX.MDP3"),
        "schema": "trades",
        "stype_in": "raw_symbol",
        "symbol": resolved_contract,
        "source_archive": args.input.file_name().and_then(|name| name.to_str()).unwrap_or("databento.zip"),
        "filter_start": window.start.to_rfc3339_opts(SecondsFormat::Nanos, true),
        "filter_end": window.end.to_rfc3339_opts(SecondsFormat::Nanos, true),
        "input_rows": stats.input_rows,
        "selected_rows": stats.selected_rows,
        "filtered_rows": stats.filtered_rows,
        "malformed_rows": stats.malformed_rows,
    });
    let warnings = vec![
        "Imported from an exact-contract Databento GLBX.MDP3 trades archive; bid/ask fields are unavailable in the trades schema.".to_string(),
        "tick_id is a deterministic import ordinal sorted by ts_event, provider sequence, and source row; the provider sequence is retained only for deterministic ordering.".to_string(),
    ];
    let outcome = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
        cache_root: args.cache_dir.clone(),
        target: None,
        // Replay cache identity is broker-compatible so existing Tradovate
        // replay loading can consume the dataset. Provenance is explicit in
        // download_request, warnings, tags, and notes.
        provider: BrokerKind::Tradovate,
        env: TradingEnvironment::Sim,
        instrument: ReplayCacheInstrument {
            symbol: instrument.clone(),
            name: Some(format!("{instrument} (Databento trades)")),
            exchange: Some("CME/COMEX".to_string()),
        },
        contract: ReplayCacheContract {
            symbol: resolved_contract.clone(),
            id: None,
            expiration: None,
        },
        request_start: window.start,
        request_end: window.end,
        download_request: request_value,
        tick_specs: ReplayCacheTickSpecs {
            tick_size,
            value_per_point,
        },
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        ticks,
        warnings,
        display_name: Some(format!(
            "{resolved_contract} Databento trades {} to {}",
            window.start.date_naive(),
            window.end.date_naive()
        )),
        tags: Some(vec![
            "databento".to_string(),
            "trades".to_string(),
            "exact-contract".to_string(),
        ]),
        notes: Some(
            "Offline exact-contract Databento trades import. This is a complete trades tape for the selected archive/window, not Level 2; use the deterministic replay aggregator to derive minute, volume, or range bars.".to_string(),
        ),
    })?;

    println!("Imported Databento trades: {}", resolved_contract);
    println!("  archive: {}", args.input.display());
    println!("  window: [{} , {}) UTC", window.start, window.end);
    println!(
        "  rows: {} selected ({} input, {} filtered, {} malformed)",
        stats.selected_rows, stats.input_rows, stats.filtered_rows, stats.malformed_rows
    );
    println!("  cache: {}", outcome.dataset_dir.display());
    println!("  manifest: {}", outcome.manifest_path.display());
    println!(
        "  parquet: {} ({} rows)",
        outcome.data_path.display(),
        outcome.row_count
    );
    println!(
        "  tick specs: tick_size={} value_per_point={}",
        tick_size, value_per_point
    );
    Ok(())
}

fn parse_archive(
    input: &Path,
    window: ImportWindow,
    requested_contract: Option<&str>,
    trades: &mut Vec<ParsedTrade>,
    stats: &mut ParseStats,
) -> Result<()> {
    let file =
        File::open(input).with_context(|| format!("open Databento ZIP {}", input.display()))?;
    let mut archive =
        ZipArchive::new(file).with_context(|| format!("read Databento ZIP {}", input.display()))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("open Databento ZIP entry {index}"))?;
        if entry.is_dir() || !entry.name().ends_with(".trades.csv") {
            continue;
        }
        stats.files += 1;
        let name = entry.name().to_string();
        parse_trade_csv(&mut entry, &name, window, requested_contract, trades, stats)?;
    }
    Ok(())
}

fn parse_trade_csv<R: Read>(
    reader: R,
    entry_name: &str,
    window: ImportWindow,
    requested_contract: Option<&str>,
    trades: &mut Vec<ParsedTrade>,
    stats: &mut ParseStats,
) -> Result<()> {
    let mut reader = BufReader::new(reader);
    let mut header = String::new();
    reader
        .read_line(&mut header)
        .with_context(|| format!("read header from {entry_name}"))?;
    let columns = TradeColumns::from_header(&header)
        .with_context(|| format!("parse Databento trades header in {entry_name}"))?;
    let start_ns = window
        .start
        .timestamp_nanos_opt()
        .context("import start is outside nanosecond range")?;
    let end_ns = window
        .end
        .timestamp_nanos_opt()
        .context("import end is outside nanosecond range")?;
    let mut line = String::new();
    let mut line_number = 1_u64;
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .with_context(|| format!("read {entry_name}"))?
            == 0
        {
            break;
        }
        line_number += 1;
        let raw = line.trim_end_matches(['\r', '\n']);
        if raw.is_empty() {
            continue;
        }
        stats.input_rows = stats.input_rows.saturating_add(1);
        let fields = raw.split(',').collect::<Vec<_>>();
        let required = [
            columns.ts_event,
            columns.price,
            columns.size,
            columns.sequence,
            columns.symbol,
        ];
        if required.iter().any(|index| *index >= fields.len()) {
            stats.malformed_rows = stats.malformed_rows.saturating_add(1);
            continue;
        }
        let symbol = normalize_contract(fields[columns.symbol]);
        if let Some(expected) = requested_contract
            && !symbol.eq_ignore_ascii_case(expected)
        {
            bail!(
                "Databento entry {entry_name}:{line_number} contains symbol {symbol}, expected {expected}"
            );
        }
        stats.symbols.insert(symbol.clone());
        let ts_ns = match fields[columns.ts_event].trim().parse::<i64>() {
            Ok(value) => value,
            Err(_) => {
                stats.malformed_rows = stats.malformed_rows.saturating_add(1);
                continue;
            }
        };
        if ts_ns < start_ns || ts_ns >= end_ns {
            stats.filtered_rows = stats.filtered_rows.saturating_add(1);
            continue;
        }
        let price = match parse_databento_price(fields[columns.price]) {
            Ok(value) => value,
            Err(_) => {
                stats.malformed_rows = stats.malformed_rows.saturating_add(1);
                continue;
            }
        };
        let size = match fields[columns.size].trim().parse::<f64>() {
            Ok(value) if value.is_finite() && value > 0.0 => value,
            _ => {
                stats.malformed_rows = stats.malformed_rows.saturating_add(1);
                continue;
            }
        };
        let sequence = match fields[columns.sequence].trim().parse::<i64>() {
            Ok(value) => value,
            Err(_) => {
                stats.malformed_rows = stats.malformed_rows.saturating_add(1);
                continue;
            }
        };
        let source_order = trades.len() as u64;
        trades.push(ParsedTrade {
            ts_ns,
            price,
            size,
            sequence,
            source_order,
        });
        stats.selected_rows = stats.selected_rows.saturating_add(1);
    }
    Ok(())
}

impl TradeColumns {
    fn from_header(header: &str) -> Result<Self> {
        let names = header
            .trim_start_matches('\u{feff}')
            .trim_end_matches(['\r', '\n'])
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        let find = |name: &str| {
            names
                .iter()
                .position(|candidate| candidate.eq_ignore_ascii_case(name))
                .with_context(|| format!("Databento trades CSV has no {name} column"))
        };
        Ok(Self {
            ts_event: find("ts_event")?,
            price: find("price")?,
            size: find("size")?,
            sequence: find("sequence")?,
            symbol: find("symbol")?,
        })
    }
}

fn parse_databento_price(raw: &str) -> Result<f64> {
    let raw = raw.trim();
    let value = if raw.contains('.') {
        raw.parse::<f64>()
            .with_context(|| format!("invalid decimal Databento price {raw}"))?
    } else {
        raw.parse::<i64>()? as f64 / 1_000_000_000.0
    };
    ensure!(
        value.is_finite() && value > 0.0,
        "Databento price must be finite and positive"
    );
    Ok(value)
}

fn resolve_window(
    args: &ImportDatabentoTradesArgs,
    job: Option<&Value>,
    request: Option<&Value>,
) -> Result<ImportWindow> {
    let start = args
        .start
        .as_deref()
        .or_else(|| {
            job.and_then(|value| value.get("start"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            request
                .and_then(|value| value.get("start"))
                .and_then(Value::as_str)
        })
        .context("missing import start; pass --start or keep sibling job.json/request.json")?;
    let end = args
        .end
        .as_deref()
        .or_else(|| {
            job.and_then(|value| value.get("end"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            request
                .and_then(|value| value.get("end"))
                .and_then(Value::as_str)
        })
        .context("missing import end; pass --end or keep sibling job.json/request.json")?;
    let start = parse_timestamp(start, "--start")?;
    let end = parse_timestamp(end, "--end")?;
    ensure!(start < end, "import --start must be before --end");
    Ok(ImportWindow { start, end })
}

fn parse_timestamp(raw: &str, label: &str) -> Result<DateTime<Utc>> {
    if let Ok(date) = NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d") {
        return Ok(date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is valid")
            .and_utc());
    }
    if let Ok(value) = DateTime::parse_from_rfc3339(raw.trim()) {
        return Ok(value.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(value) = NaiveDateTime::parse_from_str(raw.trim(), format) {
            return Ok(value.and_utc());
        }
    }
    bail!("parse {label} timestamp `{raw}` as UTC date or RFC3339")
}

fn read_sibling_json(input: &Path, file_name: &str) -> Option<Value> {
    let path = input.parent()?.join(file_name);
    let file = File::open(path).ok()?;
    serde_json::from_reader(file).ok()
}

fn normalize_contract(raw: &str) -> String {
    raw.trim().to_ascii_uppercase()
}

fn normalize_instrument(raw: &str) -> String {
    raw.trim().to_ascii_uppercase()
}

fn instrument_root(contract: &str) -> String {
    let letters = contract
        .chars()
        .take_while(|character| character.is_ascii_alphabetic())
        .collect::<String>();
    if letters.len() > 1
        && letters
            .chars()
            .last()
            .is_some_and(|character| "FGHJKMNQUVXZ".contains(character))
    {
        return letters[..letters.len() - 1].to_string();
    }
    letters
}

fn default_specs(instrument: &str) -> (f64, f64) {
    match instrument.to_ascii_uppercase().as_str() {
        "GC" | "MGC" => (
            0.1,
            if instrument.eq_ignore_ascii_case("MGC") {
                10.0
            } else {
                100.0
            },
        ),
        "ES" => (0.25, 50.0),
        "MES" => (0.25, 5.0),
        "NQ" => (0.25, 20.0),
        "MNQ" => (0.25, 2.0),
        "CL" => (0.01, 1_000.0),
        "MCL" => (0.01, 100.0),
        "YM" => (1.0, 5.0),
        "MYM" => (1.0, 0.5),
        _ => (1.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_fixed_point_databento_price() {
        assert!((parse_databento_price("5198700000000").unwrap() - 5198.7).abs() < 1e-9);
        assert!((parse_databento_price("5198.7").unwrap() - 5198.7).abs() < 1e-9);
    }

    #[test]
    fn parses_and_filters_trade_rows() {
        let csv = "ts_recv,ts_event,rtype,publisher_id,instrument_id,action,side,depth,price,size,flags,ts_in_delta,sequence,symbol\n1,1775433600000000000,0,1,2,T,A,0,5198700000000,1,0,0,3,GCZ6\n2,1775433660000000000,0,1,2,T,A,0,5198800000000,2,0,0,4,GCZ6\n";
        let start = DateTime::parse_from_rfc3339("2026-04-06T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let end = DateTime::parse_from_rfc3339("2026-04-07T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let mut trades = Vec::new();
        let mut stats = ParseStats::default();
        parse_trade_csv(
            Cursor::new(csv.as_bytes()),
            "test.csv",
            ImportWindow { start, end },
            Some("GCZ6"),
            &mut trades,
            &mut stats,
        )
        .unwrap();
        assert_eq!(trades.len(), 2);
        assert_eq!(stats.selected_rows, 2);
        assert_eq!(stats.symbols, BTreeSet::from(["GCZ6".to_string()]));
    }

    #[test]
    fn derives_common_contract_specs() {
        assert_eq!(instrument_root("GCZ6"), "GC");
        assert_eq!(default_specs("GC"), (0.1, 100.0));
        assert_eq!(default_specs("MES"), (0.25, 5.0));
    }
}
