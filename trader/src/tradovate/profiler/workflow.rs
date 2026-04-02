use super::analysis::{
    has_any_visible_broker_path, probe_findings, probe_is_settled, summarize_delay_reports,
};
use super::attempts::{AttemptBook, AttemptDispatchProgress};
use super::harness::ProfileHarness;
use super::report::{SwipeBootstrapReport, SwipeDelayReport, SwipeProfileOptions, SwipeScenario};
use super::*;
use std::time::Instant;

pub(super) async fn open_bootstrap_long(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    order_qty: i32,
    reason_tag: &str,
    settle_timeout_ms: u64,
) -> Result<SwipeBootstrapReport> {
    let bootstrap_index = attempts.start_attempt(0, order_qty, reason_tag.to_string());
    harness.send(ServiceCommand::SetTargetPosition {
        target_qty: order_qty,
        automated: true,
        reason: reason_tag.to_string(),
    })?;
    let mut findings = Vec::new();
    match wait_for_submit_observation(harness, attempts, reason_tag, Duration::from_secs(5)).await?
    {
        AttemptDispatchProgress::Submitted => {}
        AttemptDispatchProgress::DispatchNote(note) => findings.push(note),
        AttemptDispatchProgress::Timeout => findings.push(format!(
            "timed out waiting for submit observation for {reason_tag}"
        )),
    }

    let (settled, settle_ms, final_probe, settle_findings) = wait_for_settled_state(
        harness,
        attempts,
        order_qty,
        order_qty.abs(),
        Duration::from_millis(settle_timeout_ms),
        reason_tag,
    )
    .await?;
    findings.extend(settle_findings);

    Ok(SwipeBootstrapReport {
        reason_tag: reason_tag.to_string(),
        submit: attempts.submit_for(bootstrap_index),
        dispatch_notes: attempts.dispatch_notes_for(bootstrap_index),
        settled,
        settle_ms,
        final_probe,
        findings,
    })
}

pub(super) async fn run_delay_sweep(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    scenario: SwipeScenario,
    delay_ms: u64,
    options: &SwipeProfileOptions,
) -> Result<SwipeDelayReport> {
    let mut attempt_indexes = Vec::new();
    let mut target_qty = -options.order_qty;

    for iteration in 0..options.iterations_per_delay {
        let reason_tag = format!(
            "swipe-profile:{}:delay-{}ms:iter-{}:target-{}",
            scenario.slug(),
            delay_ms,
            iteration + 1,
            target_qty
        );
        let attempt_index = attempts.start_attempt(iteration + 1, target_qty, reason_tag.clone());
        attempt_indexes.push(attempt_index);
        if scenario.uses_legacy_order_strategy_reversal() {
            harness.send(ServiceCommand::ProfileLegacyOrderStrategyTarget {
                target_qty,
                reason: reason_tag.clone(),
            })?;
        } else {
            harness.send(ServiceCommand::SetTargetPosition {
                target_qty,
                automated: true,
                reason: reason_tag.clone(),
            })?;
        }
        let mut delay_findings = Vec::new();
        match wait_for_submit_observation(harness, attempts, &reason_tag, Duration::from_secs(5))
            .await?
        {
            AttemptDispatchProgress::Submitted => {}
            AttemptDispatchProgress::DispatchNote(note) => delay_findings.push(note),
            AttemptDispatchProgress::Timeout => delay_findings.push(format!(
                "timed out waiting for submit observation for {reason_tag}"
            )),
        }
        harness
            .pump_for(Duration::from_millis(delay_ms), attempts)
            .await?;
        let probe = harness
            .request_probe(
                &format!(
                    "swipe-profile:{}:delay-{}ms:iter-{}:probe",
                    scenario.slug(),
                    delay_ms,
                    iteration + 1
                ),
                attempts,
            )
            .await?;
        delay_findings.extend(probe_findings(&probe, options.order_qty.abs()));
        attempts.set_delay_probe(attempt_index, probe, delay_findings);
        target_qty = -target_qty;
    }

    let final_expected_qty = -target_qty;
    let (final_settled, final_settle_ms, final_probe, final_findings) = wait_for_settled_state(
        harness,
        attempts,
        final_expected_qty,
        options.order_qty.abs(),
        Duration::from_millis(options.settle_timeout_ms),
        &format!(
            "swipe-profile:{}:delay-{}ms:final",
            scenario.slug(),
            delay_ms
        ),
    )
    .await?;

    let reports = attempt_indexes
        .into_iter()
        .filter_map(|attempt_index| attempts.report_clone(attempt_index))
        .collect::<Vec<_>>();

    let summary = summarize_delay_reports(&reports);
    Ok(SwipeDelayReport {
        delay_ms,
        attempts: reports,
        final_settled,
        final_settle_ms,
        final_probe,
        final_findings,
        summary,
    })
}

pub(super) async fn flatten_selected_contract(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    settle_timeout_ms: u64,
    reason_tag: &str,
) -> Result<()> {
    let pre_probe = harness
        .request_probe(&format!("{reason_tag}:pre"), attempts)
        .await?;
    if pre_probe.execution_state.market_position_qty == 0
        && !has_any_visible_broker_path(&pre_probe)
    {
        return Ok(());
    }

    harness.send(ServiceCommand::ManualOrder {
        action: ManualOrderAction::Close,
    })?;
    let _ = wait_for_settled_state(
        harness,
        attempts,
        0,
        0,
        Duration::from_millis(settle_timeout_ms),
        reason_tag,
    )
    .await?;
    Ok(())
}

async fn wait_for_submit_observation(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    reason_tag: &str,
    timeout: Duration,
) -> Result<AttemptDispatchProgress> {
    let started = Instant::now();
    let mut last_note: Option<String> = None;
    loop {
        if attempts.has_submit_for_reason(reason_tag) {
            return Ok(AttemptDispatchProgress::Submitted);
        }
        last_note = attempts
            .last_dispatch_note_for_reason(reason_tag)
            .or(last_note);
        let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
            return Ok(last_note
                .map(AttemptDispatchProgress::DispatchNote)
                .unwrap_or(AttemptDispatchProgress::Timeout));
        };
        let Some(_) = harness.next_event(remaining, attempts).await? else {
            return Ok(last_note
                .map(AttemptDispatchProgress::DispatchNote)
                .unwrap_or(AttemptDispatchProgress::Timeout));
        };
    }
}

async fn wait_for_settled_state(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    expected_qty: i32,
    expected_abs_qty: i32,
    timeout: Duration,
    probe_prefix: &str,
) -> Result<(
    bool,
    Option<u64>,
    Option<ExecutionProbeSnapshot>,
    Vec<String>,
)> {
    let started = Instant::now();
    let mut probe_counter = 0usize;
    let mut last_probe: Option<ExecutionProbeSnapshot>;
    let mut last_findings: Vec<String>;

    loop {
        let probe = harness
            .request_probe(&format!("{probe_prefix}:settle-{probe_counter}"), attempts)
            .await?;
        let findings = probe_findings(&probe, expected_abs_qty);
        let settled = probe_is_settled(&probe, expected_qty, expected_abs_qty);
        last_findings = findings.clone();
        last_probe = Some(probe.clone());
        if settled {
            return Ok((
                true,
                Some(started.elapsed().as_millis() as u64),
                Some(probe),
                findings,
            ));
        }
        if started.elapsed() >= timeout {
            last_findings.push(format!(
                "settle timeout after {}ms waiting for target {}",
                timeout.as_millis(),
                expected_qty
            ));
            return Ok((false, None, last_probe, last_findings));
        }
        probe_counter = probe_counter.saturating_add(1);
        harness
            .pump_for(Duration::from_millis(100), attempts)
            .await?;
    }
}

pub(super) fn build_profile_execution_config(
    options: &SwipeProfileOptions,
    scenario: SwipeScenario,
) -> ExecutionStrategyConfig {
    let mut config = ExecutionStrategyConfig::default();
    config.kind = StrategyKind::Native;
    config.native_strategy = NativeStrategyKind::HmaAngle;
    config.native_signal_timing = NativeSignalTiming::ClosedBar;
    config.native_reversal_mode = scenario.reversal_mode();
    config.order_qty = options.order_qty.max(1);
    config.native_hma.bars_required_to_trade = 1_000_000;
    config.native_hma.take_profit_ticks = options.take_profit_ticks.max(0.0);
    config.native_hma.stop_loss_ticks = options.stop_loss_ticks.max(0.0);
    config.native_hma.use_trailing_stop = false;
    config
}
