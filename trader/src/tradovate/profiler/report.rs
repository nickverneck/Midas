use super::*;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SwipeProfileOptions {
    pub account_filter: String,
    pub contract_query: String,
    pub contract_exact: Option<String>,
    pub delays_ms: Vec<u64>,
    pub iterations_per_delay: usize,
    pub take_profit_ticks: f64,
    pub stop_loss_ticks: f64,
    pub order_qty: i32,
    pub settle_timeout_ms: u64,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SwipeScenario {
    DirectStartOrderStrategy,
    DirectMarketThenSync,
    FlattenConfirmEnter,
    CloseAllEnter,
}

impl SwipeScenario {
    pub(super) fn slug(self) -> &'static str {
        match self {
            Self::DirectStartOrderStrategy => "direct_startorderstrategy",
            Self::DirectMarketThenSync => "direct_market_then_sync",
            Self::FlattenConfirmEnter => "flatten_confirm_enter",
            Self::CloseAllEnter => "close_all_enter",
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::DirectStartOrderStrategy => "Direct / startorderstrategy Reversal",
            Self::DirectMarketThenSync => "Direct / Market Reversal + Native Sync",
            Self::FlattenConfirmEnter => "Flatten > Confirm > Enter",
            Self::CloseAllEnter => "CloseAll > Enter",
        }
    }

    pub(super) fn reversal_mode(self) -> NativeReversalMode {
        match self {
            Self::DirectStartOrderStrategy => NativeReversalMode::Direct,
            Self::DirectMarketThenSync => NativeReversalMode::Direct,
            Self::FlattenConfirmEnter => NativeReversalMode::FlattenConfirmEnter,
            Self::CloseAllEnter => NativeReversalMode::CloseAllEnter,
        }
    }

    pub(super) fn uses_legacy_order_strategy_reversal(self) -> bool {
        matches!(self, Self::DirectStartOrderStrategy)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeProfileReport {
    pub(super) generated_at_utc: DateTime<Utc>,
    pub(super) env: TradingEnvironment,
    pub(super) account_id: i64,
    pub(super) account_name: String,
    pub(super) contract_id: i64,
    pub(super) contract_name: String,
    pub(super) bar_type: BarType,
    pub(super) options: SwipeProfileReportOptions,
    pub(super) scenarios: Vec<SwipeScenarioReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeProfileReportOptions {
    pub(super) account_filter: String,
    pub(super) contract_query: String,
    pub(super) contract_exact: Option<String>,
    pub(super) delays_ms: Vec<u64>,
    pub(super) iterations_per_delay: usize,
    pub(super) take_profit_ticks: f64,
    pub(super) stop_loss_ticks: f64,
    pub(super) order_qty: i32,
    pub(super) settle_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeScenarioReport {
    pub(super) scenario: SwipeScenario,
    pub(super) bootstrap: SwipeBootstrapReport,
    pub(super) delays: Vec<SwipeDelayReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeBootstrapReport {
    pub(super) reason_tag: String,
    pub(super) submit: Option<SwipeSubmitObservation>,
    pub(super) dispatch_notes: Vec<String>,
    pub(super) settled: bool,
    pub(super) settle_ms: Option<u64>,
    pub(super) final_probe: Option<ExecutionProbeSnapshot>,
    pub(super) findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeDelayReport {
    pub(super) delay_ms: u64,
    pub(super) attempts: Vec<SwipeAttemptReport>,
    pub(super) final_settled: bool,
    pub(super) final_settle_ms: Option<u64>,
    pub(super) final_probe: Option<ExecutionProbeSnapshot>,
    pub(super) final_findings: Vec<String>,
    pub(super) summary: SwipeDelaySummary,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeDelaySummary {
    pub(super) attempts_total: usize,
    pub(super) submit_observed: usize,
    pub(super) fill_observed: usize,
    pub(super) delay_snapshots_with_mismatched_leg_qty: usize,
    pub(super) delay_snapshots_with_no_visible_protection: usize,
    pub(super) delay_snapshots_with_oversized_position: usize,
    pub(super) avg_submit_rtt_ms: Option<u64>,
    pub(super) avg_seen_ms: Option<u64>,
    pub(super) avg_exec_report_ms: Option<u64>,
    pub(super) avg_fill_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeAttemptReport {
    pub(super) iteration: usize,
    pub(super) target_qty: i32,
    pub(super) reason_tag: String,
    pub(super) sent_at_utc: DateTime<Utc>,
    pub(super) send_to_submit_event_ms: Option<u64>,
    pub(super) submit: Option<SwipeSubmitObservation>,
    pub(super) dispatch_notes: Vec<String>,
    pub(super) delay_probe: Option<ExecutionProbeSnapshot>,
    pub(super) delay_findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SwipeSubmitObservation {
    pub(super) received_at_utc: DateTime<Utc>,
    pub(super) submit_message: String,
    pub(super) broker_submit_ms: Option<u64>,
    pub(super) request_id: Option<String>,
    pub(super) seen_ms: Option<u64>,
    pub(super) exec_report_ms: Option<u64>,
    pub(super) fill_ms: Option<u64>,
}

pub(super) fn render_text_report(report: &SwipeProfileReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Swipe Profile Report\nGenerated: {}\nEnv: {}\nAccount: {} ({})\nContract: {} ({})\nBar Type: {}\n\n",
        report.generated_at_utc.to_rfc3339(),
        report.env.label(),
        report.account_name,
        report.account_id,
        report.contract_name,
        report.contract_id,
        report.bar_type.label(),
    ));
    out.push_str(&format!(
        "Options: delays={:?} iterations_per_delay={} tp_ticks={} sl_ticks={} qty={} settle_timeout_ms={}\n\n",
        report.options.delays_ms,
        report.options.iterations_per_delay,
        report.options.take_profit_ticks,
        report.options.stop_loss_ticks,
        report.options.order_qty,
        report.options.settle_timeout_ms,
    ));

    for scenario in &report.scenarios {
        out.push_str(&format!("Scenario: {}\n", scenario.scenario.label()));
        out.push_str(&format!(
            "Bootstrap: settled={} settle_ms={:?} dispatch_notes={} findings={}\n",
            scenario.bootstrap.settled,
            scenario.bootstrap.settle_ms,
            if scenario.bootstrap.dispatch_notes.is_empty() {
                "none".to_string()
            } else {
                scenario.bootstrap.dispatch_notes.join(" | ")
            },
            if scenario.bootstrap.findings.is_empty() {
                "none".to_string()
            } else {
                scenario.bootstrap.findings.join(" | ")
            }
        ));
        for delay in &scenario.delays {
            out.push_str(&format!(
                "  Delay {}ms: attempts={} submit={} fill={} avg_submit={:?} avg_fill={:?} mismatched_leg_qty={} no_visible_protection={} oversized_position={} final_settled={} final_findings={}\n",
                delay.delay_ms,
                delay.summary.attempts_total,
                delay.summary.submit_observed,
                delay.summary.fill_observed,
                delay.summary.avg_submit_rtt_ms,
                delay.summary.avg_fill_ms,
                delay.summary.delay_snapshots_with_mismatched_leg_qty,
                delay.summary.delay_snapshots_with_no_visible_protection,
                delay.summary.delay_snapshots_with_oversized_position,
                delay.final_settled,
                if delay.final_findings.is_empty() {
                    "none".to_string()
                } else {
                    delay.final_findings.join(" | ")
                }
            ));
            for attempt in &delay.attempts {
                if attempt.delay_findings.is_empty() && attempt.dispatch_notes.is_empty() {
                    continue;
                }
                out.push_str(&format!(
                    "    Iter {} target {} dispatch={} findings: {}\n",
                    attempt.iteration,
                    attempt.target_qty,
                    if attempt.dispatch_notes.is_empty() {
                        "none".to_string()
                    } else {
                        attempt.dispatch_notes.join(" | ")
                    },
                    attempt.delay_findings.join(" | ")
                ));
            }
        }
        out.push('\n');
    }

    out
}
