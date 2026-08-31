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
#[cfg(feature = "replay")]
mod meta_gate_schedule;
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

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Mode};
use config::AppConfig;
use databento_cli::download_databento_trades;
#[cfg(feature = "replay")]
use databento_import::import_databento_trades;
use engine_cli::{configure_attach_mode, kill_all_engines, kill_engine, list_engines};
use ipc::run_engine_server;
use replay_cli::{
    analyze_replay_margin, download_replay_data, evaluate_replay_walk_forward,
    import_replay_broker_schedule, plan_replay_walk_forward, probe_replay_acceleration,
    profile_replay_sweep, rank_replay_sweep, reprice_replay_result, run_replay_sweep,
    simulate_replay_liquidation, validate_replay_sweep,
};
use tui_runtime::run_tui;

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
    #[cfg(feature = "replay")]
    if let Some(Mode::PrintMetaGateSchedule(args)) = cli.mode.clone() {
        return meta_gate_schedule::print_schedule(&args);
    }
    #[cfg(feature = "tradovate")]
    if let Some(Mode::SwipeProfile(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
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

    if let Some(Mode::DownloadReplayData(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return download_replay_data(&config, args).await;
    }
    if let Some(Mode::DownloadDatabentoTrades(args)) = cli.mode.clone() {
        return download_databento_trades(args).await;
    }
    #[cfg(feature = "replay")]
    if let Some(Mode::ImportDatabentoTrades(args)) = cli.mode.clone() {
        return import_databento_trades(args);
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
