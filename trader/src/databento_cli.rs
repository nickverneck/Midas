//! Small, provider-specific Databento historical downloader.
//!
//! This command deliberately stops at a portable Databento batch archive.  The
//! replay-cache importer is a separate step so a downloaded job can be kept,
//! inspected, and imported again without paying for another API request.

use crate::cli::DatabentoTradesArgs;
use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, NaiveDate, NaiveDateTime, SecondsFormat, Utc};
use futures_util::StreamExt;
use reqwest::{Client, Response, StatusCode};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;

const HISTORICAL_API_BASE: &str = "https://hist.databento.com/v0";
const DOWNLOAD_API_BASE: &str = "https://api.databento.com/v0";

/// Submit one exact raw-symbol trades job per contract and download each
/// completed job as a ZIP archive.
pub(crate) async fn download_databento_trades(args: DatabentoTradesArgs) -> Result<()> {
    dotenvy::dotenv().ok();

    let api_key = std::env::var("DATABENTO_API")
        .or_else(|_| std::env::var("DATABENTO_API_KEY"))
        .context("DATABENTO_API is not set; add it to .env or the environment")?;
    let api_key = api_key.trim().to_string();
    ensure!(!api_key.is_empty(), "DATABENTO_API is empty");

    let contracts = normalize_contracts(&args.contracts)?;
    let (start, start_dt) = normalize_timestamp(&args.start, "--start")?;
    let (end, end_dt) = normalize_timestamp(&args.end, "--end")?;
    ensure!(start_dt < end_dt, "--start must be before --end");
    validate_split_duration(&args.split_duration)?;
    ensure!(args.poll_seconds > 0, "--poll-seconds must be positive");
    ensure!(
        args.timeout_minutes > 0,
        "--timeout-minutes must be positive"
    );
    ensure!(
        !args.dataset.trim().is_empty(),
        "--dataset must not be empty"
    );

    let client = Client::builder()
        .user_agent(format!(
            "midas-trader/{} databento-downloader",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .context("build Databento HTTP client")?;

    println!(
        "Databento trades: dataset={} contracts={} range={} to {} (UTC)",
        args.dataset,
        contracts.join(","),
        start,
        end
    );
    println!("Output: {}", args.output_dir.display());
    println!("Each request uses exact raw-symbol symbology; no contract mixing is performed.");

    for contract in contracts {
        let request = SubmitRequest {
            dataset: args.dataset.trim().to_string(),
            contract: contract.clone(),
            start: start.clone(),
            end: end.clone(),
            split_duration: args.split_duration.clone(),
        };
        if args.estimate_only {
            let estimate = estimate_cost(&client, &api_key, &request).await?;
            println!(
                "{contract}: estimated Databento trades cost ${estimate:.4} for {} to {}",
                request.start, request.end
            );
            continue;
        }
        let submitted = submit_job(&client, &api_key, &request).await?;
        let job_id = submitted
            .get("id")
            .and_then(Value::as_str)
            .context("Databento submit response omitted job id")?
            .to_string();
        println!("{contract}: submitted batch job {job_id}");

        let job_dir = job_output_dir(&args.output_dir, &contract, &start_dt, &end_dt, &job_id);
        tokio::fs::create_dir_all(&job_dir)
            .await
            .with_context(|| format!("create Databento output directory {}", job_dir.display()))?;
        write_json(
            &job_dir.join("request.json"),
            &json!({
                "provider": "databento",
                "dataset": request.dataset,
                "schema": "trades",
                "stype_in": "raw_symbol",
                "stype_out": "instrument_id",
                "symbol": request.contract,
                "start": request.start,
                "end": request.end,
                "encoding": "csv",
                "compression": "none",
                "map_symbols": true,
                "split_duration": request.split_duration,
                "job_id": job_id,
            }),
        )
        .await?;

        if args.submit_only {
            write_json(&job_dir.join("submitted-job.json"), &submitted).await?;
            println!(
                "{contract}: submit-only; job metadata: {}",
                job_dir.display()
            );
            continue;
        }

        let details = wait_for_job(
            &client,
            &api_key,
            &job_id,
            Duration::from_secs(args.poll_seconds),
            Duration::from_secs(args.timeout_minutes.saturating_mul(60)),
        )
        .await?;
        write_json(&job_dir.join("job.json"), &details).await?;

        let user_id = details
            .get("user_id")
            .and_then(Value::as_str)
            .or_else(|| submitted.get("user_id").and_then(Value::as_str))
            .context("Databento job response omitted user_id")?;
        let archive_path = job_dir.join(format!("{job_id}.zip"));
        download_job_archive(&client, &api_key, user_id, &job_id, &archive_path).await?;
        println!(
            "{contract}: downloaded {} ({} bytes)",
            archive_path.display(),
            tokio::fs::metadata(&archive_path)
                .await
                .map(|metadata| metadata.len())
                .unwrap_or(0)
        );
        print_job_summary(&contract, &details);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct SubmitRequest {
    dataset: String,
    contract: String,
    start: String,
    end: String,
    split_duration: String,
}

async fn submit_job(client: &Client, api_key: &str, request: &SubmitRequest) -> Result<Value> {
    let form = [
        ("dataset", request.dataset.as_str()),
        ("symbols", request.contract.as_str()),
        ("schema", "trades"),
        ("start", request.start.as_str()),
        ("end", request.end.as_str()),
        ("encoding", "csv"),
        ("compression", "none"),
        ("stype_in", "raw_symbol"),
        ("stype_out", "instrument_id"),
        ("map_symbols", "true"),
        ("pretty_px", "false"),
        ("pretty_ts", "false"),
        ("split_duration", request.split_duration.as_str()),
        ("delivery", "download"),
    ];
    let response = client
        .post(format!("{HISTORICAL_API_BASE}/batch.submit_job"))
        .basic_auth(api_key, Some(""))
        .form(&form)
        .send()
        .await
        .context("submit Databento batch job")?;
    parse_json_response(response, "submit Databento batch job").await
}

async fn estimate_cost(client: &Client, api_key: &str, request: &SubmitRequest) -> Result<f64> {
    let response = client
        .get(format!("{HISTORICAL_API_BASE}/metadata.get_cost"))
        .basic_auth(api_key, Some(""))
        .query(&[
            ("dataset", request.dataset.as_str()),
            ("symbols", request.contract.as_str()),
            ("schema", "trades"),
            ("start", request.start.as_str()),
            ("end", request.end.as_str()),
            ("stype_in", "raw_symbol"),
        ])
        .send()
        .await
        .context("request Databento cost estimate")?;
    let value = parse_json_response(response, "request Databento cost estimate").await?;
    value
        .as_f64()
        .context("Databento cost estimate response was not numeric")
}

async fn get_job_details(client: &Client, api_key: &str, job_id: &str) -> Result<Value> {
    let response = client
        .get(format!("{HISTORICAL_API_BASE}/batch.get_job_details"))
        .basic_auth(api_key, Some(""))
        .query(&[("job_id", job_id)])
        .send()
        .await
        .with_context(|| format!("get Databento job details for {job_id}"))?;
    parse_json_response(response, "get Databento job details").await
}

async fn wait_for_job(
    client: &Client,
    api_key: &str,
    job_id: &str,
    poll_interval: Duration,
    timeout: Duration,
) -> Result<Value> {
    let started = Instant::now();
    let mut last_status = String::new();
    loop {
        let details = get_job_details(client, api_key, job_id).await?;
        let state = details
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let progress = details
            .get("progress")
            .and_then(Value::as_u64)
            .map(|value| format!(" {value}%"))
            .unwrap_or_default();
        let status = format!("{state}{progress}");
        if status != last_status {
            println!("{job_id}: {status}");
            last_status = status;
        }
        match state {
            "done" => return Ok(details),
            "expired" | "failed" | "error" => {
                bail!("Databento job {job_id} ended in state `{state}`")
            }
            _ => {}
        }
        if started.elapsed() >= timeout {
            bail!(
                "timed out waiting for Databento job {job_id} after {} seconds; re-run with --submit-only or check the Databento download center",
                timeout.as_secs()
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn download_job_archive(
    client: &Client,
    api_key: &str,
    user_id: &str,
    job_id: &str,
    destination: &Path,
) -> Result<()> {
    let url = format!("{DOWNLOAD_API_BASE}/batch/download/{user_id}/{job_id}/{job_id}.zip");
    let response = client
        .get(url)
        .basic_auth(api_key, Some(""))
        .send()
        .await
        .with_context(|| format!("download Databento batch archive {job_id}"))?;
    let response = ensure_success(response, "download Databento batch archive").await?;
    let mut file = tokio::fs::File::create(destination)
        .await
        .with_context(|| format!("create Databento archive {}", destination.display()))?;
    let mut stream = response.bytes_stream();
    let mut bytes_written = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read Databento batch archive response")?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write Databento archive {}", destination.display()))?;
        bytes_written = bytes_written.saturating_add(chunk.len() as u64);
    }
    file.flush()
        .await
        .with_context(|| format!("flush Databento archive {}", destination.display()))?;
    ensure!(bytes_written > 0, "Databento archive {job_id} was empty");
    Ok(())
}

async fn parse_json_response(response: Response, operation: &str) -> Result<Value> {
    let response = ensure_success(response, operation).await?;
    response
        .json::<Value>()
        .await
        .with_context(|| format!("parse {operation} JSON response"))
}

async fn ensure_success(response: Response, operation: &str) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<response body unavailable>".to_string());
    let body = body.trim();
    let excerpt = if body.len() > 1000 {
        format!("{}…", &body[..1000])
    } else {
        body.to_string()
    };
    if status == StatusCode::TOO_MANY_REQUESTS {
        bail!("{operation} was rate-limited (HTTP 429): {excerpt}");
    }
    bail!("{operation} failed (HTTP {status}): {excerpt}");
}

fn normalize_contracts(raw: &[String]) -> Result<Vec<String>> {
    let mut contracts = Vec::new();
    for value in raw {
        for part in value.split(',') {
            let symbol = part.trim().to_ascii_uppercase();
            ensure!(!symbol.is_empty(), "contract symbols must not be empty");
            ensure!(
                !matches!(symbol.as_str(), "ALL_SYMBOLS" | "GC.FUT" | "ES.FUT"),
                "--contract requires an exact raw contract such as GCQ6 or GCZ6; `{symbol}` is a parent/all-symbols selector"
            );
            if !contracts.contains(&symbol) {
                contracts.push(symbol);
            }
        }
    }
    ensure!(
        !contracts.is_empty(),
        "provide at least one --contract, for example --contract GCQ6,GCZ6"
    );
    Ok(contracts)
}

fn normalize_timestamp(raw: &str, label: &str) -> Result<(String, DateTime<Utc>)> {
    let raw = raw.trim();
    ensure!(!raw.is_empty(), "{label} must not be empty");
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        let timestamp = date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is valid")
            .and_utc();
        return Ok((
            timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true),
            timestamp,
        ));
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(raw) {
        let timestamp = timestamp.with_timezone(&Utc);
        return Ok((
            timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true),
            timestamp,
        ));
    }
    for format in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(raw, format) {
            let timestamp = naive.and_utc();
            return Ok((
                timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true),
                timestamp,
            ));
        }
    }
    bail!(
        "invalid {label} `{raw}`; use YYYY-MM-DD or an RFC3339 timestamp such as 2026-08-03T00:00:00Z"
    )
}

fn validate_split_duration(value: &str) -> Result<()> {
    ensure!(
        matches!(value, "day" | "week" | "month" | "year" | "none"),
        "--split-duration must be day, week, month, year, or none"
    );
    Ok(())
}

fn job_output_dir(
    root: &Path,
    contract: &str,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
    job_id: &str,
) -> PathBuf {
    root.join(safe_path_component(contract))
        .join(format!(
            "{}_{}",
            start.format("%Y-%m-%dT%H%M%SZ"),
            end.format("%Y-%m-%dT%H%M%SZ")
        ))
        .join(safe_path_component(job_id))
}

fn safe_path_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

async fn write_json(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize Databento job metadata")?;
    tokio::fs::write(path, bytes)
        .await
        .with_context(|| format!("write Databento metadata {}", path.display()))
}

fn print_job_summary(contract: &str, details: &Value) {
    let cost = details
        .get("cost_usd")
        .and_then(Value::as_f64)
        .map(|value| format!("${value:.4}"))
        .unwrap_or_else(|| "n/a".to_string());
    let records = details
        .get("record_count")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let billed = details
        .get("billed_size")
        .and_then(Value::as_u64)
        .map(|value| format_bytes(value))
        .unwrap_or_else(|| "n/a".to_string());
    println!("{contract}: cost={cost} records={records} billed-size={billed}");
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut index = 0;
    while value >= 1024.0 && index + 1 < UNITS.len() {
        value /= 1024.0;
        index += 1;
    }
    if index == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_date_and_rfc3339_to_utc() {
        let (date, date_value) = normalize_timestamp("2026-08-03", "start").expect("date");
        assert_eq!(date, "2026-08-03T00:00:00.000000000Z");
        assert_eq!(date_value.to_rfc3339(), "2026-08-03T00:00:00+00:00");

        let (timestamp, timestamp_value) =
            normalize_timestamp("2026-08-03T04:00:00-04:00", "start").expect("timestamp");
        assert_eq!(timestamp, "2026-08-03T08:00:00.000000000Z");
        assert_eq!(timestamp_value.to_rfc3339(), "2026-08-03T08:00:00+00:00");
    }

    #[test]
    fn normalizes_and_deduplicates_exact_contracts() {
        assert_eq!(
            normalize_contracts(&["gcq6,GCZ6".to_string(), "GCQ6".to_string()]).expect("symbols"),
            vec!["GCQ6", "GCZ6"]
        );
    }

    #[test]
    fn rejects_parent_contract_selectors() {
        let error = normalize_contracts(&["GC.FUT".to_string()]).expect_err("parent rejected");
        assert!(error.to_string().contains("exact raw contract"));
    }

    #[test]
    fn job_path_keeps_contract_and_job_id_separate() {
        let start = DateTime::parse_from_rfc3339("2026-07-27T00:00:00Z")
            .expect("start")
            .with_timezone(&Utc);
        let end = DateTime::parse_from_rfc3339("2026-08-01T00:00:00Z")
            .expect("end")
            .with_timezone(&Utc);
        let path = job_output_dir(Path::new(".run/data"), "GCQ6", &start, &end, "GLBX-ABC");
        assert!(path.ends_with("GCQ6/2026-07-27T000000Z_2026-08-01T000000Z/GLBX-ABC"));
    }
}
