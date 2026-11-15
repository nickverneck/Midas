mod app;
mod broker;
mod cli;
mod config;
mod databento_cli;
#[cfg(feature = "replay")]
mod databento_import;
mod engine_cli;
mod engine_control;
mod engine_registry;
mod engine_runtime;
mod engine_session;
mod ipc;
#[cfg(feature = "ironbeam")]
mod ironbeam;
mod lua_replay;
#[cfg(feature = "replay")]
mod meta_gate_schedule;
mod ninjatrader_ncd;
mod ninjatrader_ncd_cli;
#[cfg(feature = "replay")]
mod replay_cache;
mod replay_cli;
#[cfg(feature = "replay")]
mod replay_download;
mod strategies;
mod strategy;
mod strategy_debug;
#[cfg(feature = "tradovate")]
mod tradovate;
mod tui_runtime;
#[cfg(feature = "web-ui")]
mod web;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Mode};
use config::AppConfig;
use databento_cli::download_databento_trades;
#[cfg(feature = "replay")]
use databento_import::import_databento_trades;
use engine_cli::{configure_attach_mode, kill_all_engines, kill_engine, list_engines};
use ipc::run_engine_server;
use ninjatrader_ncd_cli::export_ninjatrader_ncd;
use replay_cli::{
    analyze_replay_margin, download_replay_data, evaluate_replay_walk_forward,
    import_replay_broker_schedule, plan_replay_walk_forward, probe_replay_acceleration,
    profile_replay_sweep, rank_replay_sweep, reprice_replay_result, run_replay_sweep,
    simulate_replay_liquidation, validate_replay_sweep,
};
use tui_runtime::{run_tui, run_tui_with_ninjatrader_replay};

fn parse_chart_bar_type(raw: &str, value: u32) -> Result<broker::BarType> {
    let value = value.max(1);
    match raw.to_ascii_lowercase().as_str() {
        "minute" | "minutes" => Ok(broker::BarType::minute(value)),
        "second" | "seconds" => Ok(broker::BarType::second(value)),
        "tick" | "ticks" => Ok(broker::BarType::tick(value)),
        "volume" => Ok(broker::BarType::volume(value)),
        "range" => Ok(broker::BarType::range(value)),
        other => anyhow::bail!("invalid chart bar kind `{other}`"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
    #[cfg(feature = "web-ui")]
    if let Some(Mode::Web { listen, assets }) = cli.mode.clone() {
        return web::serve(listen, assets).await;
    }
    #[cfg(not(feature = "web-ui"))]
    if matches!(cli.mode, Some(Mode::Web { .. })) {
        anyhow::bail!("web support requires --features web-ui");
    }
    #[cfg(feature = "replay")]
    if let Some(Mode::PrintMetaGateSchedule(args)) = cli.mode.clone() {
        return meta_gate_schedule::print_schedule(&args);
    }
    #[cfg(feature = "tradovate")]
    if let Some(Mode::SwipeProfile(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())
            .map_err(|_| anyhow::anyhow!("could not load standalone market configuration"))?;
        return tradovate::run_swipe_profile(
            config,
            tradovate::SwipeProfileOptions {
                account_filter: args.account_filter,
                contract_query: args.contract_query,
                contract_exact: args.contract_exact,
                bar_value: args.bar_value,
                require_simulation_proxy: args.require_simulation_proxy || !args.allow_real_sim,
                allow_findings: args.allow_findings,
                delays_ms: args.delays_ms,
                iterations_per_delay: args.iterations,
                take_profit_ticks: args.take_profit_ticks,
                stop_loss_ticks: args.stop_loss_ticks,
                order_qty: args.order_qty,
                settle_timeout_ms: args.settle_timeout_ms,
                output_dir: args.output_dir,
            },
        )
        .await;
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::SwipeProfile(_))) {
        anyhow::bail!("tradovate support is not enabled in this build");
    }

    #[cfg(feature = "tradovate")]
    if let Some(Mode::StandaloneMarket(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())
            .map_err(|_| anyhow::anyhow!("could not load standalone market configuration"))?;
        let bar_type = parse_chart_bar_type(&args.bar_kind, args.bar_value)?;
        if args.bar_value == 0 || args.bar_value > 100_000 {
            anyhow::bail!("bar value must be between 1 and 100000");
        }
        if args.timeout_seconds == 0 || args.timeout_seconds > if args.stream { 900 } else { 60 } {
            anyhow::bail!(
                "timeout-seconds must be between 1 and {}",
                if args.stream { 900 } else { 60 }
            );
        }
        if args.stream {
            return tradovate::standalone_market_stream(
                config,
                &args.contract,
                bar_type,
                std::time::Duration::from_secs(args.timeout_seconds),
            )
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "standalone market stream failed; check provider configuration and connectivity"
                )
            });
        }
        let output = tradovate::standalone_market_snapshot(
            config,
            &args.contract,
            bar_type,
            std::time::Duration::from_secs(args.timeout_seconds),
        )
        .await
        .map_err(|_| anyhow::anyhow!("standalone market request failed; check provider credentials, entitlement, contract, and connectivity"))?;
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::StandaloneMarket(_))) {
        anyhow::bail!("standalone market feed requires the Tradovate feature");
    }

    if let Some(Mode::DownloadReplayData(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return download_replay_data(&config, args).await;
    }
    #[cfg(feature = "replay")]
    if let Some(Mode::ChartCache(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        let cache_root = args.cache_dir.unwrap_or(config.replay_cache_dir);
        if let Some(dataset_id) = args.dataset {
            let bar_type = parse_chart_bar_type(&args.bar_kind, args.bar_value)?;
            let candle_mode = match args.chart_mode.to_ascii_lowercase().as_str() {
                "ohlc" | "standard" => broker::CandleMode::Standard,
                "heikin-ashi" | "heikin_ashi" | "ha" => broker::CandleMode::HeikinAshi,
                other => anyhow::bail!("invalid chart mode `{other}`; use ohlc or heikin-ashi"),
            };
            let input = match args.input.to_ascii_lowercase().as_str() {
                "server-bars" | "server_bars" | "bars" => replay_cache::ChartApiInput::ServerBars,
                "raw-ticks" | "raw_ticks" | "ticks" => replay_cache::ChartApiInput::RawTicks,
                other => {
                    anyhow::bail!("invalid chart input `{other}`; use server-bars or raw-ticks")
                }
            };
            let output = replay_cache::chart_api_read_page(
                cache_root,
                &dataset_id,
                replay_cache::ChartApiSeries {
                    input,
                    bar_type,
                    candle_mode,
                },
                args.start_ns,
                args.end_ns,
                args.offset,
                args.limit,
            )?;
            println!("{}", serde_json::to_string(&output)?);
        } else {
            let output = replay_cache::chart_api_catalog(cache_root);
            println!("{}", serde_json::to_string(&output)?);
        }
        return Ok(());
    }
    #[cfg(not(feature = "replay"))]
    if matches!(cli.mode, Some(Mode::ChartCache(_))) {
        anyhow::bail!("chart-cache requires the replay feature");
    }
    if let Some(Mode::DownloadDatabentoTrades(args)) = cli.mode.clone() {
        return download_databento_trades(args).await;
    }
    #[cfg(feature = "replay")]
    if let Some(Mode::ImportDatabentoTrades(args)) = cli.mode.clone() {
        return import_databento_trades(args);
    }
    if let Some(Mode::ExportNinjaTraderNcd(args)) = cli.mode.clone() {
        return export_ninjatrader_ncd(args);
    }
    #[cfg(not(feature = "replay"))]
    if matches!(cli.mode, Some(Mode::ImportDatabentoTrades(_))) {
        anyhow::bail!("Databento replay import requires the replay feature");
    }
    if let Some(Mode::RepriceReplayResult(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return reprice_replay_result(&config, args);
    }
    if let Some(Mode::AnalyzeReplayMargin(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return analyze_replay_margin(&config, args);
    }
    if let Some(Mode::ImportReplayBrokerSchedule(args)) = cli.mode.clone() {
        return import_replay_broker_schedule(args);
    }
    if let Some(Mode::SimulateReplayLiquidation(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return simulate_replay_liquidation(&config, args);
    }
    if let Some(Mode::ValidateReplaySweep(args)) = cli.mode.clone() {
        return validate_replay_sweep(args);
    }
    if let Some(Mode::RunReplaySweep(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return run_replay_sweep(&config, args).await;
    }
    if let Some(Mode::ProbeReplayAcceleration(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return probe_replay_acceleration(&config, args);
    }
    if let Some(Mode::RankReplaySweep(args)) = cli.mode.clone() {
        return rank_replay_sweep(args);
    }
    if let Some(Mode::PlanReplayWalkForward(args)) = cli.mode.clone() {
        return plan_replay_walk_forward(args);
    }
    if let Some(Mode::EvaluateReplayWalkForward(args)) = cli.mode.clone() {
        return evaluate_replay_walk_forward(args);
    }
    if let Some(Mode::ProfileReplaySweep(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return profile_replay_sweep(&config, args).await;
    }
    #[cfg(feature = "tradovate")]
    if let Some(Mode::CaptureReplayDom(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::capture_replay_dom(
            &config,
            tradovate::DomCaptureOptions {
                contract: args.contract,
                start: args.start,
                end: args.end,
                output: args.output,
                speed: args.speed,
                initial_balance: args.initial_balance,
                overwrite: args.overwrite,
            },
        )
        .await;
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::CaptureReplayDom(_))) {
        anyhow::bail!("historical DOM capture requires the Tradovate feature");
    }
    #[cfg(feature = "tradovate")]
    if let Some(Mode::CaptureLiveDom(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::capture_live_dom(
            &config,
            tradovate::LiveDomCaptureOptions {
                contract: args.contract,
                duration_seconds: args.duration_seconds,
                output: args.output,
                overwrite: args.overwrite,
            },
        )
        .await;
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::CaptureLiveDom(_))) {
        anyhow::bail!("live DOM capture requires the Tradovate feature");
    }
    #[cfg(feature = "tradovate")]
    if let Some(Mode::ValidateNinjaTraderToken(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::validate_ninjatrader_token(&config, args.start.as_deref()).await;
    }
    #[cfg(not(feature = "tradovate"))]
    if matches!(cli.mode, Some(Mode::ValidateNinjaTraderToken(_))) {
        anyhow::bail!("NinjaTrader token validation requires the Tradovate feature");
    }
    #[cfg(all(feature = "tradovate", feature = "replay"))]
    if let Some(Mode::ImportBrowserHar(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::import_browser_har(
            &config,
            tradovate::BrowserHarImportOptions {
                input: args.input,
                cache_dir: args.cache_dir,
                contracts: args.contracts,
                environment: args.environment,
                overwrite: args.overwrite,
            },
        );
    }
    #[cfg(not(all(feature = "tradovate", feature = "replay")))]
    if matches!(cli.mode, Some(Mode::ImportBrowserHar(_))) {
        anyhow::bail!("browser HAR import requires the Tradovate and replay features");
    }
    if matches!(cli.mode, Some(Mode::Engine)) {
        return run_engine_server(&cli.engine_socket).await;
    }
    #[cfg(all(feature = "tradovate", feature = "replay"))]
    if let Some(Mode::NinjaTraderReplay(args)) = cli.mode.clone() {
        use broker::{CandleMode, NinjaTraderReplaySpeed, ServiceCommand};
        use chrono::DateTime;
        use engine_runtime::{spawn_and_connect_engine, unique_engine_socket_path};

        let start = DateTime::parse_from_rfc3339(&args.start)?.with_timezone(&chrono::Utc);
        let end = args
            .end
            .as_deref()
            .map(DateTime::parse_from_rfc3339)
            .transpose()?
            .map(|value| value.with_timezone(&chrono::Utc));
        if end.is_some_and(|end| end <= start) {
            anyhow::bail!("--end must be later than --start");
        }
        let speed = NinjaTraderReplaySpeed::new(args.speed).map_err(anyhow::Error::msg)?;
        let paused_speed = NinjaTraderReplaySpeed::new(0).expect("0 is a valid Replay speed");
        let mut config = AppConfig::load(cli.config.as_deref())?;
        config.autoconnect = false;
        if args.bar_value == 0 || args.bar_value > 100_000 {
            anyhow::bail!("--bar-value must be between 1 and 100000");
        }
        let bar_type = parse_chart_bar_type(&args.bar_kind, args.bar_value)?;
        let contract = tradovate::search_replay_download_contracts(&config, &args.contract, 20)
            .await?
            .into_iter()
            .find(|contract| contract.name.eq_ignore_ascii_case(&args.contract))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "exact contract {} was not returned by provider search; refusing hosted replay",
                    args.contract
                )
            })?;
        let isolated_socket = unique_engine_socket_path(&cli.engine_socket);
        let session = spawn_and_connect_engine(cli.config.as_deref(), &isolated_socket).await?;
        let start_replay = ServiceCommand::EnterNinjaTraderReplay {
            config: config.clone(),
            contract,
            start,
            end,
            speed: paused_speed,
            initial_balance: 50_000.0,
            bar_type,
            candle_mode: CandleMode::Standard,
        };
        cli.engine_socket = isolated_socket;
        cli.no_spawn_engine = true;
        // The TUI adopts this exact IPC session, then starts Replay after its
        // event relay is attached. This avoids a second client racing the
        // initial disconnected-state snapshot against hosted Replay startup.
        return run_tui_with_ninjatrader_replay(&cli, config, session, start_replay, speed).await;
    }
    #[cfg(not(all(feature = "tradovate", feature = "replay")))]
    if matches!(cli.mode, Some(Mode::NinjaTraderReplay(_))) {
        anyhow::bail!("ninjatrader-replay requires the tradovate and replay features");
    }
    if matches!(cli.mode, Some(Mode::List)) {
        return list_engines().await;
    }
    if let Some(Mode::Kill { id, close }) = cli.mode {
        return kill_engine(id, close).await;
    }
    if let Some(Mode::KillAll { close }) = cli.mode {
        return kill_all_engines(close).await;
    }

    let attach_mode = if let Some(Mode::Attach { id }) = cli.mode {
        configure_attach_mode(&mut cli, id)?;
        true
    } else {
        false
    };
    let mut config = AppConfig::load(cli.config.as_deref())?;
    if attach_mode {
        config.autoconnect = false;
    }
    run_tui(&cli, config, attach_mode).await
}
