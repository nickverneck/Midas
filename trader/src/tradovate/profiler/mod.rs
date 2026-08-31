use super::*;
use anyhow::bail;
use tokio::sync::watch;

mod analysis;
mod attempts;
mod harness;
mod report;
mod workflow;

use analysis::{profile_output_dir, select_profile_account, select_profile_contract};
use attempts::AttemptBook;
#[cfg(test)]
use harness::merge_probe_order_detail;
use harness::{ProfileHarness, ProfileRestInspector};
pub use report::SwipeProfileOptions;
use report::{
    SwipeProfileReport, SwipeProfileReportOptions, SwipeScenario, SwipeScenarioReport,
    render_text_report,
};
use workflow::{
    build_profile_execution_config, flatten_selected_contract, open_bootstrap_long, run_delay_sweep,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const ACCOUNT_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(60);
const CONTRACT_SEARCH_TIMEOUT: Duration = Duration::from_secs(15);
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(30);
const EXECUTION_CONFIG_TIMEOUT: Duration = Duration::from_secs(10);
const EXECUTION_ARM_TIMEOUT: Duration = Duration::from_secs(10);
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn run_swipe_profile(mut config: AppConfig, options: SwipeProfileOptions) -> Result<()> {
    let allow_findings = options.allow_findings;
    if options.require_simulation_proxy && !config.simulation_proxy.enabled {
        bail!(
            "swipe-profile requires [simulation_proxy].enabled = true when --require-simulation-proxy is set"
        );
    }
    config.broker = BrokerKind::Tradovate;
    config.env = TradingEnvironment::Sim;
    config.auth_mode = AuthMode::TokenFile;
    config.autoconnect = false;
    let bar_type = BarType::range(options.bar_value);

    let report_dir = profile_output_dir(options.output_dir.as_ref())?;
    let raw_log_path = report_dir.join("events.log");
    let json_report_path = report_dir.join("report.json");
    let text_report_path = report_dir.join("report.txt");

    let (cmd_tx, cmd_rx) = service_command_channel(SERVICE_COMMAND_QUEUE_CAPACITY);
    let (event_tx, event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
    let (market_tx, market_rx) = watch::channel(MarketSnapshot::default());
    tokio::spawn(crate::broker::service_loop(cmd_rx, event_tx, market_tx));
    let rest_inspector = Some(ProfileRestInspector::from_config(&config)?);
    let mut harness = ProfileHarness::new(cmd_tx, event_rx, market_rx, rest_inspector);
    let mut attempts = AttemptBook::new();

    let run_result: Result<SwipeProfileReport> = async {
        harness.send(ServiceCommand::Connect(config.clone()))?;
        harness
            .wait_for(CONNECT_TIMEOUT, &mut attempts, |_, event| {
                matches!(event, ServiceEvent::Connected { .. })
            })
            .await
            .context("waiting for Tradovate Connected event")?;
        if harness.accounts().is_empty() {
            harness
                .wait_for(ACCOUNT_BOOTSTRAP_TIMEOUT, &mut attempts, |state, _| {
                    !state.accounts().is_empty()
                })
                .await
                .context("waiting for account bootstrap to complete")?;
        }

        let selected_account = select_profile_account(harness.accounts(), &options.account_filter)
            .cloned()
            .context("no matching sim account found")?;
        harness.send(ServiceCommand::SelectAccount {
            account_id: selected_account.id,
        })?;

        harness.send(ServiceCommand::SearchContracts {
            query: options.contract_query.clone(),
            limit: 12,
        })?;
        harness
            .wait_for(CONTRACT_SEARCH_TIMEOUT, &mut attempts, |state, event| {
                matches!(event, ServiceEvent::ContractSearchResults { .. })
                    && !state.contract_results().is_empty()
            })
            .await
            .context("waiting for ES contract search results")?;
        let selected_contract = select_profile_contract(
            harness.contract_results(),
            &options.contract_query,
            options.contract_exact.as_deref(),
        )
        .context("no matching ES contract found from sim search results")?;

        harness.send(ServiceCommand::SubscribeBars {
            contract: selected_contract.clone(),
            bar_type,
            candle_mode: CandleMode::Standard,
        })?;
        harness
            .wait_for(SUBSCRIBE_TIMEOUT, &mut attempts, |state, _| {
                state.market().contract_id == Some(selected_contract.id)
                    && state.market().tick_size.is_some()
                    && !state.market().bars.is_empty()
            })
            .await
            .context("waiting for ES 1 Range market bootstrap")?;

        let mut scenario_reports = Vec::new();
        for scenario in [
            SwipeScenario::DirectStartOrderStrategy,
            SwipeScenario::DirectMarketThenSync,
            SwipeScenario::FlattenConfirmEnter,
            SwipeScenario::CloseAllEnter,
        ] {
            let execution_config = build_profile_execution_config(&options, scenario);
            harness.send(ServiceCommand::SetExecutionStrategyConfig(
                execution_config.clone(),
            ))?;
            harness
                .wait_for(EXECUTION_CONFIG_TIMEOUT, &mut attempts, |state, _| {
                    state.execution_state().config == execution_config
                })
                .await
                .with_context(|| {
                    format!("waiting for execution config sync for {}", scenario.slug())
                })?;
            harness.send(ServiceCommand::ArmExecutionStrategy)?;
            harness
                .wait_for(EXECUTION_ARM_TIMEOUT, &mut attempts, |state, _| {
                    state.execution_state().runtime.armed
                })
                .await
                .with_context(|| format!("waiting to arm {}", scenario.slug()))?;

            flatten_selected_contract(
                &mut harness,
                &mut attempts,
                options.settle_timeout_ms,
                &format!("{}-preflight", scenario.slug()),
            )
            .await?;

            let bootstrap_tag = format!("swipe-profile:{}:bootstrap", scenario.slug());
            let bootstrap = open_bootstrap_long(
                &mut harness,
                &mut attempts,
                options.order_qty,
                &bootstrap_tag,
                options.settle_timeout_ms,
            )
            .await?;

            let mut delay_reports = Vec::new();
            for delay_ms in &options.delays_ms {
                let delay_report =
                    run_delay_sweep(&mut harness, &mut attempts, scenario, *delay_ms, &options)
                        .await?;
                delay_reports.push(delay_report);
                flatten_selected_contract(
                    &mut harness,
                    &mut attempts,
                    options.settle_timeout_ms,
                    &format!("{}-post-delay-{delay_ms}", scenario.slug()),
                )
                .await?;

                harness.send(ServiceCommand::ArmExecutionStrategy)?;
                harness
                    .wait_for(EXECUTION_ARM_TIMEOUT, &mut attempts, |state, _| {
                        state.execution_state().runtime.armed
                    })
                    .await
                    .with_context(|| {
                        format!(
                            "waiting to re-arm {} after delay {}ms",
                            scenario.slug(),
                            delay_ms
                        )
                    })?;
                let reset_tag = format!(
                    "swipe-profile:{}:delay-{delay_ms}:bootstrap-reset",
                    scenario.slug()
                );
                let _ = open_bootstrap_long(
                    &mut harness,
                    &mut attempts,
                    options.order_qty,
                    &reset_tag,
                    options.settle_timeout_ms,
                )
                .await?;
            }

            scenario_reports.push(SwipeScenarioReport {
                scenario,
                bootstrap,
                delays: delay_reports,
            });
        }

        flatten_selected_contract(
            &mut harness,
            &mut attempts,
            options.settle_timeout_ms,
            "swipe-profile-final-cleanup",
        )
        .await?;

        Ok(SwipeProfileReport {
            generated_at_utc: Utc::now(),
            env: config.env,
            account_id: selected_account.id,
            account_name: selected_account.name.clone(),
            contract_id: selected_contract.id,
            contract_name: selected_contract.name.clone(),
            bar_type,
            options: SwipeProfileReportOptions {
                account_filter: options.account_filter,
                contract_query: options.contract_query,
                contract_exact: options.contract_exact,
                bar_value: options.bar_value,
                require_simulation_proxy: options.require_simulation_proxy,
                allow_findings: options.allow_findings,
                delays_ms: options.delays_ms,
                iterations_per_delay: options.iterations_per_delay,
                take_profit_ticks: options.take_profit_ticks,
                stop_loss_ticks: options.stop_loss_ticks,
                order_qty: options.order_qty,
                settle_timeout_ms: options.settle_timeout_ms,
            },
            scenarios: scenario_reports,
        })
    }
    .await;

    fs::write(&raw_log_path, harness.raw_log_output())
        .with_context(|| format!("write {}", raw_log_path.display()))?;

    let report = run_result?;

    fs::write(&json_report_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", json_report_path.display()))?;
    fs::write(&text_report_path, render_text_report(&report))
        .with_context(|| format!("write {}", text_report_path.display()))?;

    if !allow_findings {
        let mut failures = Vec::new();
        for scenario in &report.scenarios {
            if !scenario.bootstrap.settled || !scenario.bootstrap.findings.is_empty() {
                failures.push(format!("{} bootstrap", scenario.scenario.slug()));
            }
            for delay in &scenario.delays {
                if !delay.final_settled || !delay.final_findings.is_empty() {
                    failures.push(format!(
                        "{} delay {}ms",
                        scenario.scenario.slug(),
                        delay.delay_ms
                    ));
                }
                if delay
                    .attempts
                    .iter()
                    .any(|attempt| !attempt.delay_findings.is_empty())
                {
                    failures.push(format!(
                        "{} delay {}ms attempt findings",
                        scenario.scenario.slug(),
                        delay.delay_ms
                    ));
                }
            }
        }
        if !failures.is_empty() {
            bail!(
                "swipe-profile validation failed for {}; reports were written to {}",
                failures.join(", "),
                report_dir.display()
            );
        }
    }

    println!("Swipe profile complete.");
    println!("Account: {} ({})", report.account_name, report.account_id);
    println!(
        "Contract: {} ({})",
        report.contract_name, report.contract_id
    );
    println!("Event log: {}", raw_log_path.display());
    println!("JSON report: {}", json_report_path.display());
    println!("Text report: {}", text_report_path.display());

    Ok(())
}

#[cfg(test)]
mod tests;
