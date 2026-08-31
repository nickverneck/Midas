use crate::broker::{EngineHistorySummary, EngineInspectionSnapshot, ServiceCommand, ServiceEvent};
use crate::cli::Cli;
use crate::engine_control::{close_and_kill_engine, kill_engine_process};
use crate::engine_registry::{list_running_engines, resolve_engine};
use crate::ipc::connect_client;
use anyhow::{Result, bail};
use futures_util::future::join_all;
use std::time::Duration;
use tokio::time::timeout;

const ENGINE_LIST_QUERY_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) async fn list_engines() -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    let details = join_all(engines.iter().map(query_engine_details)).await;

    println!("ID\tSTATUS\tCONTRACT\tSTRATEGY");
    for (engine, detail) in engines.iter().zip(details.iter()) {
        let status = if engine.socket_is_live {
            "live"
        } else {
            "stale"
        };
        println!(
            "{}\t{}\t{}\t{}",
            engine.id,
            status,
            detail.contract_name.as_deref().unwrap_or("-"),
            detail.strategy_label()
        );
    }

    for (engine, detail) in engines.iter().zip(details) {
        if !detail.saw_inspection && detail.latest_error.is_none() && detail.latest_status.is_none()
        {
            continue;
        }
        println!();
        println!("Engine {} details:", engine.id);
        print_engine_details(&detail);
    }
    Ok(())
}

#[derive(Debug, Default)]
struct EngineListDetails {
    broker_mode: Option<String>,
    account_id: Option<i64>,
    account_name: Option<String>,
    contract_id: Option<i64>,
    contract_name: Option<String>,
    run_id: Option<String>,
    position_qty: Option<i32>,
    entry_price: Option<f64>,
    realized_pnl: Option<f64>,
    unrealized_pnl: Option<f64>,
    fees: Option<f64>,
    fills: Option<usize>,
    wins: Option<usize>,
    losses: Option<usize>,
    strategy: Option<String>,
    strategy_armed: Option<bool>,
    strategy_status: Option<String>,
    latest_status: Option<String>,
    latest_error: Option<String>,
    saw_inspection: bool,
}

impl EngineListDetails {
    fn apply(&mut self, event: ServiceEvent) {
        match event {
            ServiceEvent::StateInspected(snapshot) => self.apply_inspection(snapshot),
            ServiceEvent::Status(message) => self.latest_status = Some(message),
            ServiceEvent::Error(message) => {
                self.latest_error = Some(message.clone());
                self.latest_status = Some(message);
            }
            _ => {}
        }
    }

    fn apply_inspection(&mut self, snapshot: EngineInspectionSnapshot) {
        self.saw_inspection = true;
        self.broker_mode = Some(format!(
            "{} {} {}",
            snapshot.broker.label(),
            snapshot.env.label(),
            snapshot.session_kind.label()
        ));
        self.account_id = snapshot.account_id;
        self.account_name = snapshot.account_name;
        self.contract_id = snapshot.contract_id;
        self.contract_name = snapshot.contract_name;

        let execution = snapshot.execution;
        self.position_qty = Some(execution.market_position_qty);
        self.entry_price = execution.market_entry_price;
        self.strategy = Some(match execution.config.kind {
            crate::strategy::StrategyKind::Native => format!(
                "{} / {}",
                execution.config.kind.label(),
                execution.config.native_strategy.label()
            ),
            kind => kind.label().to_string(),
        });
        self.strategy_armed = Some(execution.runtime.armed);
        self.strategy_status = if execution.runtime.last_summary.trim().is_empty() {
            Some(if execution.runtime.armed {
                "armed".to_string()
            } else {
                "idle".to_string()
            })
        } else {
            Some(execution.runtime.last_summary)
        };

        if let Some(history) = snapshot.history {
            self.apply_history(history);
        }
    }

    fn apply_history(&mut self, history: EngineHistorySummary) {
        self.run_id = Some(history.run_id);
        self.account_id = Some(history.account_id);
        self.account_name = Some(history.account_name);
        self.contract_id = Some(history.contract_id);
        self.contract_name = Some(history.contract_name);
        self.position_qty = Some(history.position_qty);
        self.entry_price = history.average_entry_price;
        self.realized_pnl = Some(history.realized_pnl);
        self.unrealized_pnl = Some(history.unrealized_pnl);
        self.fees = Some(history.fees);
        self.fills = Some(history.fill_count);
        self.wins = Some(history.wins);
        self.losses = Some(history.losses);
    }

    fn account_label(&self) -> String {
        match (&self.account_name, self.account_id) {
            (Some(name), Some(id)) => format!("{name} (#{id})"),
            (Some(name), None) => name.clone(),
            (None, Some(id)) => format!("#{id}"),
            (None, None) => "-".to_string(),
        }
    }

    fn position_label(&self) -> String {
        match (self.position_qty, self.entry_price) {
            (Some(qty), Some(entry)) => format!("{qty} @ {entry:.2}"),
            (Some(qty), None) => qty.to_string(),
            _ => "-".to_string(),
        }
    }

    fn contract_label(&self) -> String {
        match (&self.contract_name, self.contract_id) {
            (Some(name), Some(id)) => format!("{name} (#{id})"),
            (Some(name), None) => name.clone(),
            _ => "-".to_string(),
        }
    }

    fn pnl_label(&self) -> String {
        match (self.realized_pnl, self.unrealized_pnl) {
            (Some(realized), Some(unrealized)) => format!(
                "realized {realized:+.2} | unrealized {unrealized:+.2} | net {:+.2}",
                realized + unrealized
            ),
            _ => "-".to_string(),
        }
    }

    fn activity_label(&self) -> String {
        match (self.fills, self.wins, self.losses) {
            (Some(fills), Some(wins), Some(losses)) => {
                format!("fills {fills} | wins {wins} | losses {losses}")
            }
            _ => "-".to_string(),
        }
    }

    fn strategy_label(&self) -> String {
        let Some(strategy) = &self.strategy else {
            return "-".to_string();
        };
        match self.strategy_armed {
            Some(true) => format!("{strategy} (armed)"),
            Some(false) => format!("{strategy} (idle)"),
            None => strategy.clone(),
        }
    }
}

async fn query_engine_details(engine: &crate::engine_registry::RunningEngine) -> EngineListDetails {
    let mut details = EngineListDetails::default();
    if !engine.socket_is_live {
        return details;
    }

    let result = timeout(ENGINE_LIST_QUERY_TIMEOUT, async {
        let (command_tx, mut event_rx) = connect_client(&engine.socket_path).await?;
        command_tx
            .send(ServiceCommand::InspectState)
            .map_err(|error| anyhow::anyhow!("send state-inspection request: {error}"))?;

        while let Some(event) = event_rx.recv().await {
            let terminal = matches!(
                &event,
                ServiceEvent::Disconnected | ServiceEvent::StateInspected(_)
            );
            details.apply(event);
            if terminal {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    })
    .await;

    match result {
        Err(_) => {
            details.latest_error = Some(format!(
                "engine snapshot timed out after {}s",
                ENGINE_LIST_QUERY_TIMEOUT.as_secs()
            ));
            details.latest_status = details.latest_error.clone();
        }
        Ok(Err(error)) => {
            details.latest_error = Some(error.to_string());
            details.latest_status = details.latest_error.clone();
        }
        Ok(Ok(())) if !details.saw_inspection && details.latest_error.is_none() => {
            details.latest_status = Some("cached engine state unavailable".to_string());
        }
        Ok(Ok(())) => {}
    }

    details
}

fn print_engine_details(details: &EngineListDetails) {
    println!(
        "  Run: {} | Broker/session: {}",
        details.run_id.as_deref().unwrap_or("-"),
        details.broker_mode.as_deref().unwrap_or("-")
    );
    println!(
        "  Account: {} | Contract: {}",
        details.account_label(),
        details.contract_label()
    );
    println!(
        "  Position: {} | PnL: {} | Fees: {}",
        details.position_label(),
        details.pnl_label(),
        details
            .fees
            .map(|fees| format!("{fees:.2}"))
            .as_deref()
            .unwrap_or("-")
    );
    println!("  Activity: {}", details.activity_label());
    println!(
        "  Strategy: {} | Status: {}",
        details.strategy_label(),
        details.strategy_status.as_deref().unwrap_or("-")
    );
    if let Some(error) = &details.latest_error {
        println!("  IPC: error: {error}");
    } else if let Some(status) = &details.latest_status {
        println!("  IPC: {status}");
    }
}

#[cfg(test)]
mod tests {
    use super::EngineListDetails;
    use crate::broker::{
        BrokerKind, EngineHistorySummary, EngineInspectionSnapshot, ServiceEvent, SessionKind,
    };
    use crate::config::TradingEnvironment;
    use crate::strategy::{ExecutionStateSnapshot, NativeStrategyKind};
    use chrono::{TimeZone, Utc};

    #[test]
    fn merges_execution_and_history_snapshots_for_list_details() {
        let mut details = EngineListDetails::default();
        let mut execution = ExecutionStateSnapshot::default();
        execution.config.native_strategy = NativeStrategyKind::EmaCross;
        execution.runtime.armed = true;
        execution.runtime.last_summary = "Buy on closed bar".to_string();
        execution.selected_account_id = Some(42);
        execution.selected_contract_name = Some("GCZ6".to_string());
        execution.market_position_qty = 2;
        execution.market_entry_price = Some(2500.25);
        details.apply(ServiceEvent::StateInspected(EngineInspectionSnapshot {
            broker: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            session_kind: SessionKind::Live,
            account_id: Some(42),
            account_name: Some("SIM42".to_string()),
            contract_id: Some(7),
            contract_name: Some("GCZ6".to_string()),
            execution,
            history: Some(EngineHistorySummary {
                run_id: "run-42".to_string(),
                started_at_utc: Utc.timestamp_opt(0, 0).single().expect("epoch"),
                updated_at_utc: None,
                account_id: 42,
                account_name: "SIM42".to_string(),
                contract_id: 7,
                contract_name: "GCZ6".to_string(),
                position_qty: 2,
                average_entry_price: Some(2500.5),
                realized_pnl: 125.0,
                unrealized_pnl: -25.0,
                fees: 3.0,
                wins: 2,
                losses: 1,
                fill_count: 4,
            }),
        }));

        assert_eq!(
            details.broker_mode.as_deref(),
            Some("Tradovate Simulation Live")
        );
        assert_eq!(details.account_label(), "SIM42 (#42)");
        assert_eq!(details.contract_name.as_deref(), Some("GCZ6"));
        assert_eq!(details.position_label(), "2 @ 2500.50");
        assert_eq!(
            details.pnl_label(),
            "realized +125.00 | unrealized -25.00 | net +100.00"
        );
        assert_eq!(details.activity_label(), "fills 4 | wins 2 | losses 1");
        assert_eq!(
            details.strategy_label(),
            "Native Rust / EMA Crossover (armed)"
        );
        assert_eq!(
            details.strategy_status.as_deref(),
            Some("Buy on closed bar")
        );
        assert_eq!(details.fills, Some(4));
        assert_eq!(details.run_id.as_deref(), Some("run-42"));
    }

    #[test]
    fn unavailable_snapshot_fields_render_as_placeholders() {
        let details = EngineListDetails::default();

        assert_eq!(details.account_label(), "-");
        assert_eq!(details.position_label(), "-");
        assert_eq!(details.pnl_label(), "-");
        assert_eq!(details.activity_label(), "-");
        assert_eq!(details.strategy_label(), "-");
    }
}

pub(crate) async fn kill_engine(id: u32, close: bool) -> Result<()> {
    if close {
        close_and_kill_engine(id).await?;
        println!("Closed the selected market and killed engine {id}.");
    } else {
        kill_engine_process(id).await?;
        println!("Killed engine {id}.");
    }
    Ok(())
}

pub(crate) async fn kill_all_engines(close: bool) -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    let mut failures = Vec::new();
    for engine in engines {
        let result = if close {
            close_and_kill_engine(engine.id).await
        } else {
            kill_engine_process(engine.id).await
        };
        match result {
            Ok(()) if close => println!(
                "Closed the selected market and killed engine {}.",
                engine.id
            ),
            Ok(()) => println!("Killed engine {}.", engine.id),
            Err(err) => failures.push(format!("{}: {err}", engine.id)),
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        bail!("failed to stop some engines: {}", failures.join("; "))
    }
}

pub(crate) fn configure_attach_mode(cli: &mut Cli, id: u32) -> Result<()> {
    let engine = resolve_engine(id)?;
    if !engine.socket_is_live {
        bail!(
            "engine {} is running but its socket {} is unavailable",
            engine.id,
            engine.socket_path.display()
        );
    }

    cli.engine_socket = engine.socket_path;
    cli.no_spawn_engine = true;
    Ok(())
}
