use super::account::{
    refresh_account_state, selected_market_entry_price, selected_market_position_qty,
    selected_trade_markers,
};
use super::api::{
    authenticate, create_stream_id, list_accounts, save_token_cache, search_contracts,
    subscribe_time_bars,
};
use super::execution::{
    arm_execution_strategy, disarm_execution_strategy, handle_execution_account_sync,
    maybe_run_execution_strategy,
};
use super::orders::{
    dispatch_manual_order, dispatch_target_position_order, refresh_managed_protection,
    selected_protection_prices, sync_native_protection,
};
use super::state::{InternalEvent, IronbeamSession, IronbeamState, OrderDispatchOutcome};
use super::stream::{apply_stream_payload, run_market_stream};
use super::support::{
    FOLLOWUP_REFRESH_DELAY_MS, contract_session_profile, pick_number, pick_str, schedule_refresh,
};
use crate::broker::{
    BarType, BrokerCapabilities, BrokerKind, LatencySnapshot, MarketSnapshot, ReplaySpeed,
    ServiceCommand, ServiceEvent, SessionKind,
};
use crate::strategy::ExecutionStateSnapshot;
use anyhow::{Context, Result, bail};
use reqwest::Client;
use std::time::Duration;
use tokio::sync::{
    mpsc::{UnboundedReceiver, UnboundedSender},
    watch,
};

pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: watch::Sender<MarketSnapshot>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut state = IronbeamState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        session: None,
        latency: LatencySnapshot::default(),
    };

    while let Some(next) = tokio::select! {
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
    } {
        let result = match next {
            Either::Command(cmd) => {
                handle_command(cmd, &mut state, &event_tx, &market_tx, internal_tx.clone()).await
            }
            Either::Internal(internal) => {
                handle_internal(
                    internal,
                    &mut state,
                    &event_tx,
                    &market_tx,
                    internal_tx.clone(),
                )
                .await
            }
        };
        if let Err(err) = result {
            let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
        }
    }

    shutdown_session(state.session.as_mut(), &market_tx);
}

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
}

async fn handle_command(
    cmd: ServiceCommand,
    state: &mut IronbeamState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            shutdown_session(state.session.as_mut(), market_tx);
            state.session = None;
            state.latency = LatencySnapshot::default();
            let _ = market_tx.send(MarketSnapshot::default());
            let _ = event_tx.send(ServiceEvent::Status(format!(
                "Authenticating against Ironbeam {}...",
                cfg.env.label()
            )));

            let auth = authenticate(&state.client, &cfg, &mut state.latency).await?;
            save_token_cache(&cfg.session_cache_path, &auth)?;
            let accounts =
                list_accounts(&state.client, &cfg, &auth.token, &mut state.latency).await?;
            let selected_account_id = accounts.first().map(|account| account.id);

            let mut session = IronbeamSession {
                cfg: cfg.clone(),
                token: auth.token,
                user_name: auth.user_name.clone(),
                accounts: accounts.clone(),
                selected_account_id,
                selected_contract: None,
                account_state: super::state::AccountState::default(),
                account_snapshots: Vec::new(),
                execution_config: crate::strategy::ExecutionStrategyConfig::default(),
                execution_runtime: super::state::ExecutionRuntimeState::default(),
                managed_protection: std::collections::BTreeMap::new(),
                market: MarketSnapshot::default(),
                market_task: None,
            };
            refresh_account_state(&state.client, &mut session, &mut state.latency).await?;
            emit_account_snapshots(event_tx, &session);
            state.session = Some(session);

            let _ = event_tx.send(ServiceEvent::Connected {
                broker: BrokerKind::Ironbeam,
                env: cfg.env,
                user_name: auth.user_name,
                auth_mode: cfg.auth_mode,
                session_kind: SessionKind::Live,
                capabilities: ironbeam_capabilities(),
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            if let Some(session) = state.session.as_ref() {
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::EnterReplayMode { .. } => {
            let _ = event_tx.send(ServiceEvent::Error(
                "Ironbeam replay is not available in this build.".to_string(),
            ));
        }
        ServiceCommand::ReplayState => {
            let Some(session) = state.session.as_ref() else {
                let _ = event_tx.send(ServiceEvent::Disconnected);
                return Ok(());
            };
            let _ = event_tx.send(ServiceEvent::Connected {
                broker: BrokerKind::Ironbeam,
                env: session.cfg.env,
                user_name: session.user_name.clone(),
                auth_mode: session.cfg.auth_mode,
                session_kind: SessionKind::Live,
                capabilities: ironbeam_capabilities(),
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(session.accounts.clone()));
            emit_account_snapshots(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::SelectAccount { account_id } => {
            let session = require_session_mut(state.session.as_mut())?;
            if !session
                .accounts
                .iter()
                .any(|account| account.id == account_id)
            {
                bail!("unknown Ironbeam account id `{account_id}`");
            }
            session.selected_account_id = Some(account_id);
            handle_execution_account_sync(&state.client, session, &mut state.latency, internal_tx)
                .await?;
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SearchContracts { query, limit } => {
            let session = require_session(state.session.as_ref())?;
            let trimmed = query.trim();
            if trimmed.len() < 3 {
                let _ = event_tx.send(ServiceEvent::Status(
                    "Ironbeam symbol search expects at least 3 characters.".to_string(),
                ));
                let _ = event_tx.send(ServiceEvent::ContractSearchResults {
                    query,
                    results: Vec::new(),
                });
                return Ok(());
            }
            let results = search_contracts(
                &state.client,
                &session.cfg,
                &session.token,
                trimmed,
                limit,
                &mut state.latency,
            )
            .await?;
            let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            let session = require_session_mut(state.session.as_mut())?;
            if bar_type != BarType::Minute1 {
                let _ = event_tx.send(ServiceEvent::Status(
                    "Ironbeam currently supports 1-minute bars only; using 1 Min.".to_string(),
                ));
            }

            shutdown_market_task(session, market_tx);
            let stream_id = create_stream_id(
                &state.client,
                &session.cfg,
                &session.token,
                &mut state.latency,
            )
            .await?;
            subscribe_time_bars(
                &state.client,
                &session.cfg,
                &session.token,
                &stream_id,
                &contract,
                &mut state.latency,
            )
            .await?;

            session.selected_contract = Some(contract.clone());
            session.execution_runtime.clear_pending_target();
            session.execution_runtime.last_closed_bar_ts = None;
            session.execution_runtime.reset_execution();
            session.execution_runtime.last_summary =
                "Selected contract changed; waiting for Ironbeam bars.".to_string();
            session.market = MarketSnapshot {
                contract_id: Some(contract.id),
                contract_name: Some(contract.name.clone()),
                bars: Vec::new(),
                trade_markers: selected_trade_markers(&session.account_state, &contract),
                session_profile: Some(contract_session_profile(&contract)),
                value_per_point: pick_number(
                    &contract.raw,
                    &["pipValue", "pointValue", "pointValueUsd", "valuePerPoint"],
                ),
                tick_size: pick_number(&contract.raw, &["minTick", "pipSize", "tickSize"]),
                history_loaded: 0,
                live_bars: 0,
                status: format!("Subscribing to {} 1-minute bars...", contract.name),
            };
            refresh_managed_protection(session);
            super::account::rebuild_account_snapshots(session);
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = market_tx.send(session.market.clone());
            session.market_task = Some(tokio::spawn(run_market_stream(
                session.cfg.env,
                stream_id,
                session.token.clone(),
                contract,
                internal_tx,
            )));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetReplaySpeed { .. } => {
            let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(ReplaySpeed::default()));
        }
        ServiceCommand::ManualOrder { action } => {
            let session = require_session_mut(state.session.as_mut())?;
            match dispatch_manual_order(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                action,
            )
            .await?
            {
                OrderDispatchOutcome::NoOp { message } => {
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                OrderDispatchOutcome::Queued { target_qty } => {
                    if let Some(target_qty) = target_qty {
                        session
                            .execution_runtime
                            .set_pending_target(Some(target_qty));
                        session.execution_runtime.last_summary =
                            "Manual close requested; waiting for Ironbeam fill.".to_string();
                    }
                    emit_execution_state(event_tx, session);
                }
            }
            emit_account_snapshots(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetTargetPosition {
            target_qty,
            automated,
            reason,
        } => {
            let session = require_session_mut(state.session.as_mut())?;
            match dispatch_target_position_order(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                target_qty,
                automated,
                &reason,
            )
            .await?
            {
                OrderDispatchOutcome::NoOp { message } => {
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                OrderDispatchOutcome::Queued { target_qty } => {
                    session.execution_runtime.set_pending_target(target_qty);
                }
            }
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::ProfileLegacyOrderStrategyTarget { .. } => {
            let _ = event_tx.send(ServiceEvent::Error(
                "legacy order-strategy profiler mode is only available on Tradovate".to_string(),
            ));
        }
        ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
        } => {
            let session = require_session_mut(state.session.as_mut())?;
            sync_native_protection(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                signed_qty,
                take_profit_price,
                stop_price,
                &reason,
            )
            .await?;
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            let session = require_session_mut(state.session.as_mut())?;
            if session.execution_config != config {
                session.execution_config = config;
                if session.execution_runtime.armed {
                    disarm_execution_strategy(
                        session,
                        "Native strategy config changed; press Continue to re-arm.".to_string(),
                    );
                }
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::ArmExecutionStrategy => {
            let session = require_session_mut(state.session.as_mut())?;
            arm_execution_strategy(session);
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            let session = require_session_mut(state.session.as_mut())?;
            disarm_execution_strategy(session, reason);
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::ProbeExecution { .. } => {
            let _ = event_tx.send(ServiceEvent::Error(
                "Execution probe is only available on Tradovate.".to_string(),
            ));
        }
    }

    Ok(())
}

async fn handle_internal(
    internal: InternalEvent,
    state: &mut IronbeamState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::StreamStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::StreamError(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
        InternalEvent::RefreshAccountState { reason } => {
            let Some(session) = state.session.as_mut() else {
                return Ok(());
            };
            if let Some(reason) = reason {
                let _ = event_tx.send(ServiceEvent::Status(reason));
            }
            refresh_account_state(&state.client, session, &mut state.latency).await?;
            handle_execution_account_sync(
                &state.client,
                session,
                &mut state.latency,
                internal_tx.clone(),
            )
            .await?;
            let _ = market_tx.send(session.market.clone());
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::StreamPayload(raw) => {
            let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw) else {
                return Ok(());
            };
            let Some(session) = state.session.as_mut() else {
                return Ok(());
            };
            let stream_effect = apply_stream_payload(session, &parsed);
            if let Some(reset) = parsed.get("r") {
                let message = pick_str(reset, &["message", "m"]).unwrap_or("stream reset");
                let _ = event_tx.send(ServiceEvent::Status(format!(
                    "Ironbeam stream reset: {message}"
                )));
                schedule_refresh(
                    internal_tx.clone(),
                    Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
                    Some("Refreshing Ironbeam account state after reset.".to_string()),
                );
            }

            if stream_effect.account_changed {
                handle_execution_account_sync(
                    &state.client,
                    session,
                    &mut state.latency,
                    internal_tx.clone(),
                )
                .await?;
                emit_account_snapshots(event_tx, session);
                emit_execution_state(event_tx, session);
            } else if stream_effect.market_changed {
                maybe_run_execution_strategy(
                    &state.client,
                    session,
                    &mut state.latency,
                    internal_tx.clone(),
                    event_tx,
                )
                .await?;
                emit_execution_state(event_tx, session);
            }

            if stream_effect.market_changed || stream_effect.account_changed {
                let _ = market_tx.send(session.market.clone());
            }

            if stream_effect.refresh_recommended {
                schedule_refresh(
                    internal_tx,
                    Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
                    None,
                );
            }
        }
    }

    Ok(())
}

fn ironbeam_capabilities() -> BrokerCapabilities {
    BrokerCapabilities {
        replay: false,
        manual_orders: true,
        automated_orders: true,
        native_protection: true,
    }
}

fn require_session(session: Option<&IronbeamSession>) -> Result<&IronbeamSession> {
    session.context("Connect to Ironbeam first.")
}

fn require_session_mut(session: Option<&mut IronbeamSession>) -> Result<&mut IronbeamSession> {
    session.context("Connect to Ironbeam first.")
}

fn shutdown_session(
    session: Option<&mut IronbeamSession>,
    market_tx: &watch::Sender<MarketSnapshot>,
) {
    if let Some(session) = session {
        shutdown_market_task(session, market_tx);
    }
}

fn shutdown_market_task(session: &mut IronbeamSession, market_tx: &watch::Sender<MarketSnapshot>) {
    if let Some(task) = session.market_task.take() {
        task.abort();
    }
    session.market = MarketSnapshot::default();
    let _ = market_tx.send(MarketSnapshot::default());
}

pub(super) fn emit_account_snapshots(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &IronbeamSession,
) {
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(
        session.account_snapshots.clone(),
    ));
}

fn emit_execution_state(event_tx: &UnboundedSender<ServiceEvent>, session: &IronbeamSession) {
    let (take_profit_price, stop_price) = selected_protection_prices(session);
    let snapshot = ExecutionStateSnapshot {
        config: session.execution_config.clone(),
        runtime: session.execution_runtime.snapshot(),
        selected_account_id: session.selected_account_id,
        selected_contract_name: session
            .selected_contract
            .as_ref()
            .map(|contract| contract.name.clone()),
        market_position_qty: selected_market_position_qty(session),
        market_entry_price: selected_market_entry_price(session),
        selected_contract_take_profit_price: take_profit_price,
        selected_contract_stop_price: stop_price,
    };
    let _ = event_tx.send(ServiceEvent::ExecutionState(snapshot));
}
