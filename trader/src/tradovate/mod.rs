#[cfg(feature = "replay")]
use crate::broker::BarKind;
use crate::broker::{
    AccountInfo, AccountSnapshot, Bar, BarType, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, EngineHistoryFill, EngineHistorySnapshot, EngineInspectionSnapshot,
    ExecutionProbeManagedProtection, ExecutionProbeOrder, ExecutionProbeSnapshot,
    InstrumentSessionProfile, InstrumentSessionWindow, LatencySnapshot, ManualOrderAction,
    MarketHistoryUpdate, MarketSnapshot, NinjaTraderReplaySpeed, ReplayDownloadOperationId,
    ReplaySpeed, ReplayWindowSnapshot, SERVICE_COMMAND_QUEUE_CAPACITY,
    SERVICE_EVENT_QUEUE_CAPACITY, ServiceCommand, ServiceCommandReceiver, ServiceEvent,
    ServiceEventSender, SessionKind, TradeMarker, TradeMarkerSide, infer_session_profile,
    service_command_channel, service_event_channel,
};
use crate::broker::{ReplayBarFrame, ReplayDomLevel, ReplayFrameSet, ReplayMarketDom};
#[cfg(any(feature = "replay", test))]
use crate::broker::{ReplayExecutionPrecision, ReplayMarketTick};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::hma_cross::HmaCrossExecutionState;
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, ExecutionStrategyConfig,
    MarketHistoryRequirement, NativeExecutionPath, NativeReversalMode, NativeSignalTiming,
    NativeStrategyKind, StrategyKind,
};
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
#[cfg(test)]
use chrono::TimeZone;
use chrono::{DateTime, Utc};
#[cfg(any(feature = "replay", test))]
use chrono_tz::America::New_York;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::ToSocketAddrs;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

#[cfg(feature = "replay")]
mod browser_har;
#[cfg(feature = "replay")]
mod download;
mod execution;
mod gateway;
mod history;
#[cfg(feature = "replay")]
mod hosted_replay;
mod orders;
mod profiler;
mod protocol;
pub(crate) mod replay;
mod service;

use self::protocol::*;
pub use self::service::service_loop;
#[cfg(feature = "replay")]
pub use browser_har::{BrowserHarImportOptions, import_browser_har};
#[cfg(feature = "replay")]
pub use download::{
    TradovateChunkedRawTickCacheRequest, TradovateChunkedRawTickPhase,
    TradovateChunkedRawTickProgress, TradovateRawTickDownloadRequest,
    TradovateReplayDownloadSession, TradovateServerBarDownload, TradovateServerBarDownloadRequest,
    download_replay_raw_ticks, download_replay_raw_ticks_after_auth,
    download_replay_raw_ticks_chunked_to_cache, download_replay_server_bars,
    download_replay_server_bars_after_auth, inspect_replay_download_contract,
    prepare_replay_download_session, prepare_replay_download_session_after_auth,
    search_replay_download_contracts,
};
use execution::*;
use gateway::*;
use history::*;
#[cfg(feature = "manual-orders")]
use orders::dispatch_manual_order;
use orders::{
    MarketOrderDispatchOutcome, build_market_order_request, cancel_order_by_id,
    cancel_orders_by_id, collect_live_protection_orders,
    dispatch_profile_legacy_order_strategy_target, dispatch_target_position_order,
    enqueue_market_order, interrupt_order_strategy_by_id, native_order_strategy_enabled,
    recover_live_protection_order, refresh_managed_protection_order_ids, request_order_json,
    selected_strategy_key, sync_native_protection, sync_native_protection_target,
};
pub use profiler::{SwipeProfileOptions, run_swipe_profile};
#[cfg(feature = "replay")]
#[allow(unused_imports)]
pub(crate) use replay::{
    ReplayAcceleration, ReplayAccelerationDevice, ReplayAccelerationStatus, ReplayBrokerSchedule,
    ReplayEmaBatchResult, ReplayEmaLastPair, ReplayFeeSchedule, ReplayLiquidationConfig,
    ReplayMarginConfig, ReplaySweepExecutionMode, ReplaySweepPerformanceOptions,
    ReplaySweepPerformanceReport, ReplaySweepRankingMetric, ReplaySweepRankingOptions,
    ReplaySweepSpec, ReplayWalkForwardEvaluationDocument, ReplayWalkForwardEvaluationOptions,
    ReplayWalkForwardOptions, ReplayWalkForwardPhase, ReplayWalkForwardSelectionPolicy,
    ReplayWalkForwardWindow, analyze_replay_margin, ema_last_batch, evaluate_replay_walk_forward,
    plan_replay_walk_forward, probe_acceleration, rank_replay_sweep, replay_sweep_performance_csv,
    replay_sweep_performance_json, reprice_replay_result, run_replay_sweep,
    run_replay_sweep_performance, run_replay_sweep_with_mode, simulate_replay_liquidation_result,
    write_replay_sweep_performance_report,
};

include!("types.rs");
include!("session.rs");
include!("auth.rs");
include!("latency.rs");

/// Fetch one chart snapshot through the production Tradovate auth and market
/// data worker without bringing up the account/user-sync or order services.
pub(crate) async fn standalone_market_snapshot(
    cfg: AppConfig,
    requested_contract: &str,
    bar_type: BarType,
    timeout: Duration,
) -> Result<Value> {
    if cfg.broker != BrokerKind::Tradovate {
        bail!("standalone market feed supports Tradovate configuration only");
    }
    let symbol = validate_standalone_market_request(requested_contract, bar_type, timeout)?;
    standalone_market_with_deadline(
        timeout,
        standalone_market_snapshot_inner(cfg, symbol, bar_type),
    )
    .await
}

/// Continuously emits bounded chart snapshots from a single authenticated
/// market worker. Each output line is one complete JSON value (NDJSON).
pub(crate) async fn standalone_market_stream(
    cfg: AppConfig,
    requested_contract: &str,
    bar_type: BarType,
    timeout: Duration,
) -> Result<()> {
    if cfg.broker != BrokerKind::Tradovate {
        bail!("standalone market feed supports Tradovate configuration only");
    }
    let symbol = validate_standalone_market_request(requested_contract, bar_type, timeout)?;
    standalone_market_stream_inner(cfg, symbol, bar_type, timeout).await
}

async fn standalone_market_stream_inner(
    mut cfg: AppConfig,
    symbol: String,
    bar_type: BarType,
    stream_timeout: Duration,
) -> Result<()> {
    use crate::broker::MarketSnapshot;
    use crate::strategy::ExecutionStrategyConfig;
    use gateway::{INTERNAL_EVENT_QUEUE_CAPACITY, InternalEvent, internal_event_channel};
    use tokio::io::AsyncWriteExt;

    const STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
    cfg.candle_mode = bar_type.effective_candle_mode(crate::broker::CandleMode::Standard);
    let setup = async {
        let client = Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(30))
            .build()?;
        let tokens = authenticate(&client, &cfg)
            .await
            .context("authenticate for standalone market data")?;
        let contracts = search_contracts(
            &client,
            &cfg.broker_rest_url(),
            &tokens.access_token,
            &symbol,
            20,
        )
        .await
        .context("search provider contracts")?;
        let contract = contracts
            .into_iter()
            .find(|contract| contract.name.eq_ignore_ascii_case(&symbol))
            .with_context(|| {
                format!("exact contract {symbol} was not returned by provider search")
            })?;
        let specs = fetch_contract_specs(
            &client,
            &cfg.broker_rest_url(),
            &tokens.access_token,
            &contract,
        )
        .await
        .ok();
        let requirement =
            ExecutionStrategyConfig::default().market_history_requirement(cfg.history_bars);
        cfg.history_bars = requirement.requested_bars;
        Ok::<_, anyhow::Error>((contract, specs, requirement, tokens.md_access_token))
    };
    let (contract, specs, requirement, md_token) =
        standalone_market_startup(STARTUP_TIMEOUT, setup).await?;
    // Install process signal handlers before creating the provider worker so
    // registration failure cannot leave a detached WebSocket task behind.
    #[cfg(unix)]
    let mut terminate_signal =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    #[cfg(unix)]
    let mut interrupt_signal =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    #[cfg(unix)]
    let signal_exit = async {
        tokio::select! {
            _ = terminate_signal.recv() => (),
            _ = interrupt_signal.recv() => (),
        }
        Ok::<(), anyhow::Error>(())
    };
    #[cfg(not(unix))]
    let signal_exit = async {
        tokio::signal::ctrl_c()
            .await
            .context("register Ctrl-C handler")
    };
    tokio::pin!(signal_exit);
    // Poll the listener once before spawning so platform-specific handler
    // installation failures are observed without an active market worker.
    tokio::select! {
        biased;
        result = &mut signal_exit => { result?; return Ok(()); },
        _ = tokio::task::yield_now() => {},
    }
    let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    let candle_mode = cfg.candle_mode;
    let worker_cfg = cfg.clone();
    let worker_tx = internal_tx.clone();
    let (worker_done_tx, mut worker_done_rx) = tokio::sync::oneshot::channel();
    let worker = AbortTaskOnDrop::new(tokio::spawn(async move {
        market_data_worker(
            worker_cfg,
            md_token,
            contract,
            specs,
            bar_type,
            candle_mode,
            requirement,
            worker_tx,
        )
        .await;
        let _ = worker_done_tx.send(());
    }));
    let outcome = async {
        let mut market = MarketSnapshot::default();
        let mut stdout = tokio::io::stdout();
        let first_snapshot_deadline = time::sleep(STARTUP_TIMEOUT);
        tokio::pin!(first_snapshot_deadline);
        let overall_deadline = time::sleep(stream_timeout);
        tokio::pin!(overall_deadline);
        let mut published = false;
        loop {
            let event = tokio::select! {
                biased;
                _ = &mut overall_deadline => break Ok(()),
                _ = &mut first_snapshot_deadline, if !published => bail!("standalone market worker produced no first snapshot before startup deadline"),
                signal = &mut signal_exit => { signal?; break Ok(()); },
                event = standalone_stream_recv(&internal_tx, &mut internal_rx, &mut worker_done_rx) => event?,
            };
            match event {
                Some(InternalEvent::Market(update)) => {
                    apply_market_update(&mut market, update);
                    if market.bars.is_empty() { continue; }
                    published = true;
                    let bars = market.bars.iter().map(|bar| json!({
                        "ts_ns": bar.ts_ns.to_string(), "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close, "volume": bar.volume,
                    })).collect::<Vec<_>>();
                    let snapshot = json!({
                        "source": "tradovate_market_data_worker", "broker_source": "Tradovate market data",
                        "connection_status": "connected", "chart_freshness": "unknown",
                        "fidelity": "Provider chart bars; not a guaranteed complete trade tape.",
                        "observed_at": Utc::now().to_rfc3339(), "contract": market.contract_name,
                        "bar_type": bar_type.chart_description(), "status": market.status,
                        "bars": bars, "latest_bar_ts_ns": market.bars.last().map(|bar| bar.ts_ns.to_string()),
                    });
                    let encoded = serde_json::to_vec(&snapshot)?;
                    if encoded.len() > 2 * 1024 * 1024 { bail!("standalone market snapshot exceeded output limit"); }
                    let write = async {
                        stdout.write_all(&encoded).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await
                    };
                    tokio::select! {
                        _ = &mut overall_deadline => break Ok(()),
                        signal = &mut signal_exit => { signal?; break Ok(()); },
                        result = write => result?,
                    }
                }
                Some(InternalEvent::Error(error)) => bail!("standalone market worker failed: {error}"),
                Some(_) => {}
                None => bail!("standalone market worker stopped before stream ended"),
            }
        }
    }.await;
    worker.abort_and_wait().await;
    outcome
}

async fn standalone_stream_recv(
    sender: &gateway::InternalEventSender,
    receiver: &mut gateway::InternalEventReceiver,
    worker_done: &mut tokio::sync::oneshot::Receiver<()>,
) -> Result<Option<gateway::InternalEvent>> {
    tokio::select! {
        biased;
        _ = sender.overflow_notified() => bail!("standalone market event queue overflow; stopping to avoid publishing an incomplete chart"),
        _ = worker_done => bail!("standalone market worker closed before stream completion"),
        event = receiver.recv() => Ok(event),
    }
}

async fn standalone_market_startup<F, T>(timeout: Duration, operation: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    tokio::time::timeout(timeout, operation)
        .await
        .context("standalone market authentication/startup deadline exceeded")?
}

async fn standalone_market_with_deadline<F, T>(timeout: Duration, operation: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    tokio::time::timeout(timeout, operation)
        .await
        .map_err(|_| anyhow::anyhow!("timed out waiting for standalone market data"))?
}

struct AbortTaskOnDrop(Option<tokio::task::JoinHandle<()>>);

impl AbortTaskOnDrop {
    fn new(task: tokio::task::JoinHandle<()>) -> Self {
        Self(Some(task))
    }

    async fn abort_and_wait(mut self) {
        if let Some(mut task) = self.0.take() {
            task.abort();
            let _ = tokio::time::timeout(Duration::from_secs(2), &mut task).await;
        }
    }
}

impl Drop for AbortTaskOnDrop {
    fn drop(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}

async fn standalone_market_snapshot_inner(
    mut cfg: AppConfig,
    symbol: String,
    bar_type: BarType,
) -> Result<Value> {
    use crate::broker::MarketSnapshot;
    use crate::strategy::ExecutionStrategyConfig;
    use gateway::{INTERNAL_EVENT_QUEUE_CAPACITY, InternalEvent, internal_event_channel};

    cfg.candle_mode = bar_type.effective_candle_mode(crate::broker::CandleMode::Standard);
    let client = Client::builder()
        .tcp_nodelay(true)
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .context("create read-only market client")?;
    let tokens = authenticate(&client, &cfg)
        .await
        .context("authenticate for standalone market data")?;
    let contracts = search_contracts(
        &client,
        &cfg.broker_rest_url(),
        &tokens.access_token,
        &symbol,
        20,
    )
    .await
    .context("search provider contracts")?;
    let contract = contracts
        .into_iter()
        .find(|contract| contract.name.eq_ignore_ascii_case(&symbol))
        .with_context(|| format!("exact contract {symbol} was not returned by provider search"))?;
    let specs = fetch_contract_specs(
        &client,
        &cfg.broker_rest_url(),
        &tokens.access_token,
        &contract,
    )
    .await
    .ok();

    let requirement =
        ExecutionStrategyConfig::default().market_history_requirement(cfg.history_bars);
    cfg.history_bars = requirement.requested_bars;
    let (internal_tx, mut internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    let worker = AbortTaskOnDrop::new(tokio::spawn(market_data_worker(
        cfg.clone(),
        tokens.md_access_token,
        contract.clone(),
        specs,
        bar_type,
        cfg.candle_mode,
        requirement,
        internal_tx,
    )));

    let outcome = async {
        let mut market = MarketSnapshot::default();
        loop {
            match internal_rx.recv().await {
                Some(InternalEvent::Market(update)) => {
                    apply_market_update(&mut market, update);
                    if !market.bars.is_empty() {
                        let observed_at = Utc::now().to_rfc3339();
                        let bars = market
                            .bars
                            .iter()
                            .map(|bar| {
                                json!({
                                    "ts_ns": bar.ts_ns.to_string(),
                                    "open": bar.open,
                                    "high": bar.high,
                                    "low": bar.low,
                                    "close": bar.close,
                                    "volume": bar.volume,
                                })
                            })
                            .collect::<Vec<_>>();
                        return Ok(json!({
                            "source": "tradovate_market_data_worker",
                            "broker_source": "Tradovate market data",
                            "connection_status": "connected",
                            "chart_freshness": "unknown",
                            "fidelity": "Provider chart bars; not a guaranteed complete trade tape.",
                            "observed_at": observed_at,
                            "contract": market.contract_name,
                            "bar_type": bar_type.chart_description(),
                            "status": market.status,
                            "bars": bars,
                            "latest_bar_ts_ns": market.bars.last().map(|bar| bar.ts_ns.to_string()),
                        }));
                    }
                }
                Some(InternalEvent::Error(error)) => {
                    bail!("standalone market worker failed: {error}");
                }
                Some(_) => {}
                None => bail!("standalone market worker stopped before publishing bars"),
            }
        }
    }
    .await;
    worker.abort_and_wait().await;
    outcome
}

fn validate_standalone_market_request(
    requested_contract: &str,
    bar_type: BarType,
    timeout: Duration,
) -> Result<String> {
    let symbol = requested_contract.trim().to_ascii_uppercase();
    if symbol.is_empty()
        || symbol.len() > 16
        || !symbol.bytes().all(|byte| byte.is_ascii_alphanumeric())
    {
        bail!("contract must be an exact symbol containing 1–16 ASCII letters or digits");
    }
    if !(1..=100_000).contains(&bar_type.value()) {
        bail!("bar value must be between 1 and 100000");
    }
    if timeout.is_zero() || timeout > Duration::from_secs(15 * 60) {
        bail!("standalone market timeout must be between 1 and 900 seconds");
    }
    Ok(symbol)
}

#[cfg(test)]
mod standalone_market_tests {
    use super::*;
    use gateway::{INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY, InternalEvent, internal_event_channel};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn standalone_market_validation_accepts_exact_contract_and_supported_bar() {
        assert_eq!(
            validate_standalone_market_request(
                " mesz6 ",
                BarType::minute(1),
                Duration::from_secs(15),
            )
            .unwrap(),
            "MESZ6"
        );
        assert!(
            validate_standalone_market_request(
                "MESZ6;orders",
                BarType::minute(1),
                Duration::from_secs(15),
            )
            .is_err()
        );
        assert!(
            validate_standalone_market_request(
                "MESZ6",
                BarType::tick(100_001),
                Duration::from_secs(15),
            )
            .is_err()
        );
        assert!(
            validate_standalone_market_request("MESZ6", BarType::minute(1), Duration::ZERO,)
                .is_err()
        );
    }

    #[tokio::test]
    async fn standalone_stream_fails_closed_when_bounded_event_queue_overflows() {
        let (tx, mut rx) = internal_event_channel(1);
        for _ in 0..INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY {
            tx.send(InternalEvent::Error("fill critical lane".into()))
                .unwrap();
        }
        assert!(tx.send(InternalEvent::Error("overflow".into())).is_err());
        let (_done_tx, mut done_rx) = tokio::sync::oneshot::channel();
        let outcome = standalone_stream_recv(&tx, &mut rx, &mut done_rx).await;
        assert!(outcome.is_err());
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[tokio::test]
    async fn standalone_stream_reports_worker_close_after_last_snapshot_event() {
        let (tx, mut rx) = internal_event_channel(1);
        let (done_tx, mut done_rx) = tokio::sync::oneshot::channel();
        tx.send(InternalEvent::Error("last update marker".into()))
            .unwrap();
        assert!(matches!(
            standalone_stream_recv(&tx, &mut rx, &mut done_rx).await,
            Ok(Some(InternalEvent::Error(_)))
        ));
        done_tx.send(()).unwrap();
        let closed = standalone_stream_recv(&tx, &mut rx, &mut done_rx).await;
        assert!(matches!(closed, Err(error) if error.to_string().contains("worker closed")));
    }

    #[tokio::test]
    async fn standalone_stream_cancellation_awaits_worker_drop() {
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_in_task = dropped.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let worker = AbortTaskOnDrop::new(tokio::spawn(async move {
            struct MarkDropped(Arc<AtomicBool>);
            impl Drop for MarkDropped {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _mark = MarkDropped(dropped_in_task);
            let _ = ready_tx.send(());
            std::future::pending::<()>().await;
        }));
        ready_rx.await.unwrap();
        worker.abort_and_wait().await;
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn standalone_stream_startup_deadline_fails_hung_auth_without_provider_access() {
        let result = standalone_market_startup(
            Duration::from_millis(5),
            std::future::pending::<Result<()>>(),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("startup deadline"));
    }

    #[tokio::test]
    async fn standalone_market_deadline_covers_pending_preflight_and_cancels_worker() {
        let task_dropped = Arc::new(AtomicBool::new(false));
        let task_dropped_in_worker = task_dropped.clone();
        let operation = async move {
            let _worker = AbortTaskOnDrop::new(tokio::spawn(async move {
                struct MarkDropped(Arc<AtomicBool>);
                impl Drop for MarkDropped {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::SeqCst);
                    }
                }
                let _mark_dropped = MarkDropped(task_dropped_in_worker);
                std::future::pending::<()>().await;
            }));
            // Models a provider preflight request that never resolves.
            std::future::pending::<Result<()>>().await?;
            Ok(())
        };

        let error = standalone_market_with_deadline(Duration::from_millis(10), operation)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"));
        tokio::task::yield_now().await;
        assert!(task_dropped.load(Ordering::SeqCst));
    }
}
include!("market.rs");
include!("dom_capture.rs");
include!("store.rs");

fn tradovate_capabilities() -> BrokerCapabilities {
    BrokerCapabilities {
        replay: cfg!(feature = "replay"),
        manual_orders: cfg!(feature = "manual-orders"),
        automated_orders: true,
        native_protection: true,
    }
}
