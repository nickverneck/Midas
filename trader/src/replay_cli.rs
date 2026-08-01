use crate::cli::{
    AnalyzeReplayMarginArgs, RankReplaySweepArgs, ReplayDownloadArgs, RepriceReplayResultArgs,
    RunReplaySweepArgs, ValidateReplaySweepArgs,
};
use crate::config::AppConfig;
#[cfg(feature = "replay")]
use anyhow::Context;
use anyhow::{Result, bail};
#[cfg(feature = "replay")]
use std::path::PathBuf;

pub(crate) async fn download_replay_data(
    config: &AppConfig,
    args: ReplayDownloadArgs,
) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (config, args);
        bail!("replay downloader requires `--features replay`");
    }

    #[cfg(feature = "replay")]
    {
        let plan = build_replay_download_plan(config, args)?;
        if config.broker != crate::broker::BrokerKind::Tradovate {
            bail!("replay data downloads currently support Tradovate only");
        }

        #[cfg(not(feature = "tradovate"))]
        {
            let _ = plan;
            bail!("Tradovate server-bar replay downloads require `--features tradovate,replay`");
        }

        #[cfg(feature = "tradovate")]
        download_tradovate_replay(config, plan).await
    }
}

pub(crate) fn reprice_replay_result(
    config: &AppConfig,
    args: RepriceReplayResultArgs,
) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (config, args);
        bail!("replay result repricing requires `--features replay`");
    }

    #[cfg(all(feature = "replay", not(feature = "tradovate")))]
    {
        let _ = (config, args);
        bail!("replay result repricing requires the Tradovate replay module in this build");
    }

    #[cfg(all(feature = "replay", feature = "tradovate"))]
    {
        let _ = config;
        let schedule = crate::tradovate::ReplayFeeSchedule {
            name: args.name,
            currency: args.currency,
            commission_per_contract: args.commission_per_contract,
            exchange_per_contract: args.exchange_per_contract,
            clearing_per_contract: args.clearing_per_contract,
            regulatory_per_contract: args.regulatory_per_contract,
            misc_per_contract: args.misc_per_contract,
        };
        let outcome = crate::tradovate::reprice_replay_result(&args.result, schedule)?;
        println!("Replay result repriced without replaying market data.");
        println!("Scenario: {}", outcome.scenario_name);
        println!("Fees: {:.8}", outcome.fees);
        println!("Net PnL: {:.8}", outcome.net_pnl);
        println!("Result: {}", outcome.result_path.display());
        Ok(())
    }
}

pub(crate) fn analyze_replay_margin(
    config: &AppConfig,
    args: AnalyzeReplayMarginArgs,
) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (config, args);
        bail!("replay margin analysis requires `--features replay`");
    }

    #[cfg(all(feature = "replay", not(feature = "tradovate")))]
    {
        let _ = (config, args);
        bail!("replay margin analysis requires the Tradovate replay module in this build");
    }

    #[cfg(all(feature = "replay", feature = "tradovate"))]
    {
        let _ = config;
        let margin = crate::tradovate::ReplayMarginConfig {
            model: args.model,
            currency: args.currency,
            margin_per_contract: args.margin_per_contract,
            safety_buffer: args.safety_buffer,
            safety_buffer_percent: args.safety_buffer_percent,
        };
        let outcome = crate::tradovate::analyze_replay_margin(&args.result, margin)?;
        println!("Replay margin analysis saved without replaying market data.");
        println!(
            "Required starting capital: {:.8}",
            outcome.required_starting_capital
        );
        println!(
            "Initial capital sufficient: {}",
            outcome.initial_capital_sufficient
        );
        println!("Result: {}", outcome.result_path.display());
        Ok(())
    }
}

pub(crate) fn validate_replay_sweep(args: ValidateReplaySweepArgs) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = args;
        bail!("replay sweep validation requires `--features replay`");
    }

    #[cfg(all(feature = "replay", not(feature = "tradovate")))]
    {
        let _ = args;
        bail!("replay sweep validation requires the Tradovate replay module in this build");
    }

    #[cfg(all(feature = "replay", feature = "tradovate"))]
    {
        let spec = crate::tradovate::ReplaySweepSpec::load(&args.spec)?;
        let report = spec.guardrail_report(None)?;
        let plan = spec.plan()?;
        println!("Replay sweep specification is valid.");
        println!("Sweep: {} ({})", spec.sweep_id, spec.name);
        println!("Dataset view: {}", spec.base_dataset_view.id);
        println!("Runs: {}", plan.children.len());
        println!("Parallelism: {}", spec.parallelism);
        println!(
            "Estimate: input rows {}, memory {}, output {}, runtime {}.",
            display_estimate_value(report.estimate.estimated_input_rows),
            display_estimate_bytes(report.estimate.estimated_memory_bytes),
            display_estimate_bytes(report.estimate.estimated_output_bytes),
            display_estimate_duration(report.estimate.estimated_runtime_seconds),
        );
        for warning in &report.warnings {
            println!("Warning: {warning}");
        }
        for violation in &report.violations {
            println!("Guardrail: {violation}");
        }
        println!(
            "Output formats: {}",
            spec.output_formats
                .iter()
                .map(|format| format.label())
                .collect::<Vec<_>>()
                .join(", ")
        );
        if let Some(output) = args.output {
            plan.save(&output)?;
            println!("Expanded plan: {}", output.display());
        }
        Ok(())
    }
}

pub(crate) async fn run_replay_sweep(config: &AppConfig, args: RunReplaySweepArgs) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (config, args);
        bail!("replay sweep execution requires `--features replay`");
    }

    #[cfg(all(feature = "replay", not(feature = "tradovate")))]
    {
        let _ = (config, args);
        bail!("replay sweep execution requires the Tradovate replay module in this build");
    }

    #[cfg(all(feature = "replay", feature = "tradovate"))]
    {
        let spec = crate::tradovate::ReplaySweepSpec::load(&args.spec)?;
        let report = spec.guardrail_report(Some(&config.replay_cache_dir))?;
        println!(
            "Replay sweep launch estimate: {} combinations, {} parallel jobs, input rows {}, memory {}, output {}, runtime {}.",
            report.estimate.combinations,
            report.estimate.parallel_jobs,
            display_estimate_value(report.estimate.estimated_input_rows),
            display_estimate_bytes(report.estimate.estimated_memory_bytes),
            display_estimate_bytes(report.estimate.estimated_output_bytes),
            display_estimate_duration(report.estimate.estimated_runtime_seconds),
        );
        for warning in &report.warnings {
            println!("Warning: {warning}");
        }
        for violation in &report.violations {
            println!("Guardrail: {violation}");
        }
        if report.requires_confirmation && !args.allow_large && !args.override_guardrails {
            println!(
                "Large sweep confirmation required: pass --allow-large after reviewing the estimate."
            );
        }
        let summary = crate::tradovate::run_replay_sweep(
            config,
            &args.spec,
            args.no_resume,
            args.allow_large,
            args.override_guardrails,
        )
        .await?;
        println!(
            "Replay sweep {} complete: {} completed, {} failed, {} skipped.",
            summary.sweep_id, summary.completed_count, summary.failed_count, summary.skipped_count
        );
        for warning in &summary.warnings {
            println!("Warning: {warning}");
        }
        Ok(())
    }
}

pub(crate) fn rank_replay_sweep(args: RankReplaySweepArgs) -> Result<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = args;
        bail!("replay sweep ranking requires `--features replay`");
    }

    #[cfg(all(feature = "replay", not(feature = "tradovate")))]
    {
        let _ = args;
        bail!("replay sweep ranking requires the Tradovate replay module in this build");
    }

    #[cfg(all(feature = "replay", feature = "tradovate"))]
    {
        let metric = crate::tradovate::ReplaySweepRankingMetric::parse(&args.metric)?;
        let options = crate::tradovate::ReplaySweepRankingOptions {
            metric,
            fee_scenario: args.fee_scenario,
            limit: args.limit,
            min_closed_trades: args.min_closed_trades,
            max_drawdown_pct: args.max_drawdown_pct,
        };
        let document = crate::tradovate::rank_replay_sweep(
            &args.summary,
            args.plan.as_deref(),
            options,
            args.output.as_deref(),
            args.csv.as_deref(),
        )?;
        println!(
            "Replay sweep {} ranked {} candidates with {} ({} rows).",
            document.sweep_id,
            document.total_completed_candidates,
            document.metric.label(),
            document.rows.len()
        );
        for row in &document.rows {
            println!(
                "#{:>3} {:<24} metric={} net_pnl={} drawdown_pct={} neighborhood={} robustness={}",
                row.rank,
                row.run_id,
                display_estimate_f64(row.metric_value),
                display_estimate_f64(row.net_pnl),
                display_estimate_f64(row.max_drawdown_pct),
                row.neighborhood_count,
                display_estimate_f64(row.robustness_score),
            );
        }
        for warning in &document.warnings {
            println!("Warning: {warning}");
        }
        if let Some(output) = args.output {
            println!("Ranking JSON: {}", output.display());
        }
        if let Some(csv) = args.csv {
            println!("Ranking CSV: {}", csv.display());
        }
        Ok(())
    }
}

#[cfg(all(feature = "replay", feature = "tradovate"))]
fn display_estimate_value(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(all(feature = "replay", feature = "tradovate"))]
fn display_estimate_bytes(value: Option<u64>) -> String {
    let Some(value) = value else {
        return "unknown".to_string();
    };
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut scaled = value as f64;
    let mut unit = 0;
    while scaled >= 1024.0 && unit + 1 < UNITS.len() {
        scaled /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", value, UNITS[unit])
    } else {
        format!("{scaled:.1} {}", UNITS[unit])
    }
}

#[cfg(all(feature = "replay", feature = "tradovate"))]
fn display_estimate_duration(value: Option<u64>) -> String {
    let Some(seconds) = value else {
        return "unknown".to_string();
    };
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {}s", seconds / 60, seconds % 60)
    }
}

#[cfg(all(feature = "replay", feature = "tradovate"))]
fn display_estimate_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}

#[cfg(all(feature = "replay", feature = "tradovate"))]
async fn download_tradovate_replay(config: &AppConfig, plan: ReplayDownloadPlan) -> Result<()> {
    match plan.source_kind {
        crate::replay_cache::ReplayCacheSourceKind::ServerBars => {
            let download = crate::tradovate::download_replay_server_bars(
                config,
                crate::tradovate::TradovateServerBarDownloadRequest {
                    contract: plan.contract.clone(),
                    exact_contract: None,
                    start: plan.start,
                    end: plan.end,
                    bar_type: plan.bar_type,
                },
            )
            .await?;
            let outcome = crate::replay_cache::write_server_bars_parquet_cache(
                crate::replay_cache::ReplayCacheServerBarsWrite {
                    cache_root: plan.cache_root.clone(),
                    target: None,
                    provider: config.broker,
                    env: config.env,
                    instrument: crate::replay_cache::ReplayCacheInstrument {
                        symbol: plan.instrument.clone(),
                        name: None,
                        exchange: None,
                    },
                    contract: crate::replay_cache::ReplayCacheContract {
                        symbol: download.contract.name.clone(),
                        id: Some(download.contract.id),
                        expiration: None,
                    },
                    request_start: plan.start,
                    request_end: plan.end,
                    source_kind: crate::replay_cache::ReplayCacheSourceKind::ServerBars,
                    download_request: download.request_body,
                    bar_type: plan.bar_type,
                    tick_specs: download.tick_specs,
                    contract_metadata: Some(download.contract_metadata.clone()),
                    session_template: download.session_template,
                    bars: download.bars,
                    warnings: download.warnings,
                    display_name: plan.display_name.clone(),
                    tags: (!plan.tags.is_empty()).then_some(plan.tags.clone()),
                    notes: Some(
                        "Downloaded through Tradovate read-only metadata/account REST endpoints and the md/getChart market-data WebSocket; no user sync, account stream, or order path was started."
                            .to_string(),
                    ),
                },
            )?;

            println!("Replay server-bar download complete.");
            println!("No user sync, account stream, or order path was started.");
            println!("Provider: {}", config.broker.label());
            println!("Environment: {}", config.env.label());
            println!("Instrument: {}", plan.instrument);
            println!("Contract: {}", plan.contract);
            println!("Date range: {} to {}", plan.start_date, plan.end_date);
            println!("Source kind: {}", plan.source_kind.label());
            println!("Storage: parquet ({})", "snappy");
            println!(
                "Requested shape: {}",
                plan.bar_type.mode_label(plan.chart_mode)
            );
            println!("Rows: {}", outcome.row_count);
            print_suggested_contract_coverage(&download.contract_metadata);
            println!("Data: {}", outcome.data_path.display());
            println!("Manifest: {}", outcome.manifest_path.display());
            Ok(())
        }
        crate::replay_cache::ReplayCacheSourceKind::RawTicks => {
            let outcome = crate::tradovate::download_replay_raw_ticks_chunked_to_cache(
                config,
                crate::tradovate::TradovateChunkedRawTickCacheRequest {
                    instrument: plan.instrument.clone(),
                    contract: plan.contract.clone(),
                    exact_contract: None,
                    target: None,
                    start: plan.start,
                    end: plan.end,
                    cache_root: plan.cache_root.clone(),
                    chunk_duration: chrono::Duration::minutes(i64::from(
                        plan.raw_chunk_minutes,
                    )),
                    minimum_split: chrono::Duration::minutes(i64::from(
                        plan.raw_minimum_split_minutes,
                    )),
                    display_name: plan.display_name.clone(),
                    tags: (!plan.tags.is_empty()).then_some(plan.tags.clone()),
                    notes: Some(
                        "Downloaded raw ticks through bounded, resumable Tradovate read-only metadata/account REST and md/getChart market-data requests; no user sync, account stream, or order path was started."
                            .to_string(),
                    ),
                },
                None,
                || println!("Authenticated read-only replay download session."),
                |progress| {
                    println!(
                        "Raw ticks [{}/{}]: {}",
                        progress.completed_chunks,
                        progress.total_leaf_chunks,
                        progress.message
                    );
                },
                || Ok(()),
                |_, _| {},
            )
            .await?;
            let manifest =
                crate::replay_cache::ReplayCacheManifest::from_path(&outcome.manifest_path)?;
            let bytes = outcome.data_paths.iter().try_fold(0_u64, |total, path| {
                let len = std::fs::metadata(path)
                    .with_context(|| format!("read cache file metadata {}", path.display()))?
                    .len();
                Ok::<_, anyhow::Error>(total.saturating_add(len))
            })?;

            println!("Replay raw tick download complete.");
            println!("No user sync, account stream, or order path was started.");
            println!("Provider: {}", config.broker.label());
            println!("Environment: {}", config.env.label());
            println!("Instrument: {}", plan.instrument);
            println!("Contract: {}", plan.contract);
            println!("Date range: {} to {}", plan.start_date, plan.end_date);
            println!("Source kind: {}", plan.source_kind.label());
            println!("Storage: parquet ({})", "snappy");
            println!("Rows: {}", outcome.row_count);
            println!("Bytes: {bytes}");
            println!("Chunk files: {}", outcome.data_paths.len());
            if let Some(metadata) = manifest.contract_metadata.as_ref() {
                print_suggested_contract_coverage(metadata);
            }
            for path in &outcome.data_paths {
                println!("Data: {}", path.display());
            }
            println!("Manifest: {}", outcome.manifest_path.display());
            Ok(())
        }
        other => bail!("unsupported replay download source kind: {}", other.label()),
    }
}

#[cfg(feature = "replay")]
fn print_suggested_contract_coverage(metadata: &crate::replay_cache::ReplayCacheContractMetadata) {
    if let Some(coverage) = &metadata.suggested_coverage {
        println!(
            "Suggested broad contract coverage: {} to {} (estimate; requested dates were not changed)",
            coverage.start_date, coverage.end_date
        );
        println!("Coverage basis: {}", coverage.basis);
    } else {
        println!("Suggested broad contract coverage: unavailable from maturity metadata");
    }
    println!(
        "Contract metadata: {} endpoint snapshot(s), {} account context(s)",
        replay_contract_metadata_snapshot_count(metadata),
        metadata.context.accounts.len()
    );
}

#[cfg(feature = "replay")]
fn replay_contract_metadata_snapshot_count(
    metadata: &crate::replay_cache::ReplayCacheContractMetadata,
) -> usize {
    1 + [
        metadata.maturity.as_ref(),
        metadata.maturity_chain.as_ref(),
        metadata.product.as_ref(),
        metadata.product_sessions.as_ref(),
        metadata.product_margins.as_ref(),
        metadata.contract_margins.as_ref(),
        metadata.fee_params.as_ref(),
    ]
    .into_iter()
    .flatten()
    .count()
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayDownloadPlan {
    instrument: String,
    contract: String,
    start_date: chrono::NaiveDate,
    end_date: chrono::NaiveDate,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    source_kind: crate::replay_cache::ReplayCacheSourceKind,
    bar_type: crate::broker::BarType,
    chart_mode: crate::broker::CandleMode,
    cache_root: PathBuf,
    display_name: Option<String>,
    tags: Vec<String>,
    raw_chunk_minutes: u32,
    raw_minimum_split_minutes: u32,
}

#[cfg(feature = "replay")]
fn build_replay_download_plan(
    config: &AppConfig,
    args: ReplayDownloadArgs,
) -> Result<ReplayDownloadPlan> {
    let instrument = args.instrument.trim();
    let contract = args.contract.trim();
    if instrument.is_empty() {
        bail!("--instrument cannot be empty");
    }
    if contract.is_empty() {
        bail!("--contract cannot be empty");
    }
    if args.bar_value == 0 {
        bail!("--bar-value must be > 0");
    }
    if args.raw_chunk_minutes == 0 || args.raw_minimum_split_minutes == 0 {
        bail!("raw-tick chunk and minimum split minutes must be > 0");
    }

    let start_date = chrono::NaiveDate::parse_from_str(&args.start, "%Y-%m-%d")
        .with_context(|| format!("parse --start {}", args.start))?;
    let end_date = chrono::NaiveDate::parse_from_str(&args.end, "%Y-%m-%d")
        .with_context(|| format!("parse --end {}", args.end))?;
    if end_date < start_date {
        bail!("--end must be on or after --start");
    }

    let source_kind = parse_replay_download_source_kind(&args.source_kind)?;
    let bar_type = parse_replay_download_bar_type(&args.bar_kind, args.bar_value)?;
    let chart_mode = parse_replay_download_chart_mode(&args.chart_mode)?;
    let cache_root = args
        .cache_dir
        .unwrap_or_else(|| config.replay_cache_dir.clone());
    let display_name = args
        .name
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    let tags = args
        .tags
        .into_iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
        .collect();
    let start = start_date
        .and_hms_opt(0, 0, 0)
        .context("build replay download start timestamp")?
        .and_utc();
    let end = end_date
        .succ_opt()
        .context("build replay download exclusive end date")?
        .and_hms_opt(0, 0, 0)
        .context("build replay download end timestamp")?
        .and_utc();
    Ok(ReplayDownloadPlan {
        instrument: instrument.to_string(),
        contract: contract.to_string(),
        start_date,
        end_date,
        start,
        end,
        source_kind,
        bar_type,
        chart_mode,
        cache_root,
        display_name,
        tags,
        raw_chunk_minutes: args.raw_chunk_minutes,
        raw_minimum_split_minutes: args.raw_minimum_split_minutes,
    })
}

#[cfg(feature = "replay")]
fn parse_replay_download_source_kind(
    raw: &str,
) -> Result<crate::replay_cache::ReplayCacheSourceKind> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "server-bars" | "bars" => Ok(crate::replay_cache::ReplayCacheSourceKind::ServerBars),
        "raw-ticks" | "ticks" | "tick" => Ok(crate::replay_cache::ReplayCacheSourceKind::RawTicks),
        other => bail!("invalid --source-kind `{other}`; use server-bars or raw-ticks"),
    }
}

#[cfg(feature = "replay")]
fn parse_replay_download_bar_type(raw: &str, value: u32) -> Result<crate::broker::BarType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "minute" | "min" | "m" => Ok(crate::broker::BarType::minute(value)),
        "second" | "sec" | "s" => Ok(crate::broker::BarType::second(value)),
        "tick" | "tick-count" | "ticks" => Ok(crate::broker::BarType::tick(value)),
        "volume" | "vol" => Ok(crate::broker::BarType::volume(value)),
        "range" => Ok(crate::broker::BarType::range(value)),
        other => bail!("invalid --bar-kind `{other}`"),
    }
}

#[cfg(feature = "replay")]
fn parse_replay_download_chart_mode(raw: &str) -> Result<crate::broker::CandleMode> {
    match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "ohlc" | "standard" | "regular" => Ok(crate::broker::CandleMode::Standard),
        "heikin-ashi" | "heikin" | "heiken-ashi" | "heiken" => {
            Ok(crate::broker::CandleMode::HeikinAshi)
        }
        other => bail!("invalid --chart-mode `{other}`; use ohlc or heikin-ashi"),
    }
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;

    fn valid_args() -> ReplayDownloadArgs {
        ReplayDownloadArgs {
            instrument: "MES".to_string(),
            contract: "MESU6".to_string(),
            start: "2026-07-23".to_string(),
            end: "2026-07-24".to_string(),
            source_kind: "server-bars".to_string(),
            bar_kind: "minute".to_string(),
            bar_value: 1,
            chart_mode: "ohlc".to_string(),
            cache_dir: None,
            name: None,
            tags: Vec::new(),
            raw_chunk_minutes: 60,
            raw_minimum_split_minutes: 5,
        }
    }

    #[test]
    fn planner_parses_safe_request_parts() {
        assert!(matches!(
            parse_replay_download_source_kind("raw-ticks").expect("source kind"),
            crate::replay_cache::ReplayCacheSourceKind::RawTicks
        ));
        assert_eq!(
            parse_replay_download_bar_type("volume", 6500).expect("bar type"),
            crate::broker::BarType::volume(6500)
        );
        assert_eq!(
            parse_replay_download_chart_mode("heikin-ashi").expect("chart mode"),
            crate::broker::CandleMode::HeikinAshi
        );
        assert!(parse_replay_download_source_kind("dom").is_err());
    }

    #[test]
    fn planner_rejects_empty_or_zero_request_parts() {
        let config = AppConfig::default();
        let mut blank_contract = valid_args();
        blank_contract.contract = " ".to_string();
        assert!(build_replay_download_plan(&config, blank_contract).is_err());

        let mut zero_bar_value = valid_args();
        zero_bar_value.bar_value = 0;
        assert!(build_replay_download_plan(&config, zero_bar_value).is_err());
    }

    #[test]
    fn plan_uses_inclusive_end_date_and_cache_root() {
        let mut config = AppConfig::default();
        config.replay_cache_dir = PathBuf::from("/tmp/trader-cache-test");
        let mut args = valid_args();
        args.bar_value = 5;
        args.chart_mode = "heikin-ashi".to_string();
        args.name = Some("MES research".to_string());
        args.tags = vec!["baseline".to_string()];
        let plan = build_replay_download_plan(&config, args).expect("download plan");

        assert_eq!(plan.start.to_rfc3339(), "2026-07-23T00:00:00+00:00");
        assert_eq!(plan.end.to_rfc3339(), "2026-07-25T00:00:00+00:00");
        assert_eq!(plan.cache_root, PathBuf::from("/tmp/trader-cache-test"));
        assert_eq!(plan.bar_type, crate::broker::BarType::minute(5));
        assert_eq!(plan.chart_mode, crate::broker::CandleMode::HeikinAshi);
        assert_eq!(plan.display_name.as_deref(), Some("MES research"));
        assert_eq!(plan.tags, vec!["baseline"]);
        assert_eq!(plan.raw_chunk_minutes, 60);
        assert_eq!(plan.raw_minimum_split_minutes, 5);
    }

    #[test]
    fn plan_accepts_raw_tick_source_kind() {
        let config = AppConfig::default();
        let mut args = valid_args();
        args.source_kind = "raw-ticks".to_string();
        let plan = build_replay_download_plan(&config, args).expect("raw tick download plan");
        assert_eq!(
            plan.source_kind,
            crate::replay_cache::ReplayCacheSourceKind::RawTicks
        );
    }

    #[test]
    fn planner_rejects_zero_raw_tick_chunk_controls() {
        let config = AppConfig::default();
        let mut args = valid_args();
        args.raw_chunk_minutes = 0;
        assert!(build_replay_download_plan(&config, args).is_err());

        let mut args = valid_args();
        args.raw_minimum_split_minutes = 0;
        assert!(build_replay_download_plan(&config, args).is_err());
    }
}
