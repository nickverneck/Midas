mod app;
mod broker;
mod cli;
mod config;
mod engine_cli;
mod engine_control;
mod engine_registry;
mod engine_runtime;
mod engine_session;
mod ipc;
#[cfg(feature = "ironbeam")]
mod ironbeam;
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
use engine_cli::{configure_attach_mode, kill_all_engines, kill_engine, list_engines};
use ipc::run_engine_server;
use replay_cli::download_replay_data;
use tui_runtime::run_tui;

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
    #[cfg(feature = "tradovate")]
    if let Some(Mode::SwipeProfile(args)) = cli.mode.clone() {
        let config = AppConfig::load(cli.config.as_deref())?;
        return tradovate::run_swipe_profile(
            config,
            tradovate::SwipeProfileOptions {
                account_filter: args.account_filter,
                contract_query: args.contract_query,
                contract_exact: args.contract_exact,
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
    if matches!(cli.mode, Some(Mode::Engine)) {
        return run_engine_server(&cli.engine_socket).await;
    }
    if matches!(cli.mode, Some(Mode::List)) {
        return list_engines();
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
