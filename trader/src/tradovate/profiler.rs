use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const ACCOUNT_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(60);
const CONTRACT_SEARCH_TIMEOUT: Duration = Duration::from_secs(15);
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(30);
const EXECUTION_CONFIG_TIMEOUT: Duration = Duration::from_secs(10);
const EXECUTION_ARM_TIMEOUT: Duration = Duration::from_secs(10);
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

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
enum SwipeScenario {
    DirectStartOrderStrategy,
    DirectMarketThenSync,
    FlattenConfirmEnter,
    CloseAllEnter,
}

impl SwipeScenario {
    fn slug(self) -> &'static str {
        match self {
            Self::DirectStartOrderStrategy => "direct_startorderstrategy",
            Self::DirectMarketThenSync => "direct_market_then_sync",
            Self::FlattenConfirmEnter => "flatten_confirm_enter",
            Self::CloseAllEnter => "close_all_enter",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DirectStartOrderStrategy => "Direct / startorderstrategy Reversal",
            Self::DirectMarketThenSync => "Direct / Market Reversal + Native Sync",
            Self::FlattenConfirmEnter => "Flatten > Confirm > Enter",
            Self::CloseAllEnter => "CloseAll > Enter",
        }
    }

    fn reversal_mode(self) -> NativeReversalMode {
        match self {
            Self::DirectStartOrderStrategy => NativeReversalMode::Direct,
            Self::DirectMarketThenSync => NativeReversalMode::Direct,
            Self::FlattenConfirmEnter => NativeReversalMode::FlattenConfirmEnter,
            Self::CloseAllEnter => NativeReversalMode::CloseAllEnter,
        }
    }

    fn uses_legacy_order_strategy_reversal(self) -> bool {
        matches!(self, Self::DirectStartOrderStrategy)
    }
}

#[derive(Debug, Clone, Serialize)]
struct SwipeProfileReport {
    generated_at_utc: DateTime<Utc>,
    env: TradingEnvironment,
    account_id: i64,
    account_name: String,
    contract_id: i64,
    contract_name: String,
    bar_type: BarType,
    options: SwipeProfileReportOptions,
    scenarios: Vec<SwipeScenarioReport>,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeProfileReportOptions {
    account_filter: String,
    contract_query: String,
    contract_exact: Option<String>,
    delays_ms: Vec<u64>,
    iterations_per_delay: usize,
    take_profit_ticks: f64,
    stop_loss_ticks: f64,
    order_qty: i32,
    settle_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeScenarioReport {
    scenario: SwipeScenario,
    bootstrap: SwipeBootstrapReport,
    delays: Vec<SwipeDelayReport>,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeBootstrapReport {
    reason_tag: String,
    submit: Option<SwipeSubmitObservation>,
    dispatch_notes: Vec<String>,
    settled: bool,
    settle_ms: Option<u64>,
    final_probe: Option<ExecutionProbeSnapshot>,
    findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeDelayReport {
    delay_ms: u64,
    attempts: Vec<SwipeAttemptReport>,
    final_settled: bool,
    final_settle_ms: Option<u64>,
    final_probe: Option<ExecutionProbeSnapshot>,
    final_findings: Vec<String>,
    summary: SwipeDelaySummary,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeDelaySummary {
    attempts_total: usize,
    submit_observed: usize,
    fill_observed: usize,
    delay_snapshots_with_mismatched_leg_qty: usize,
    delay_snapshots_with_no_visible_protection: usize,
    delay_snapshots_with_oversized_position: usize,
    avg_submit_rtt_ms: Option<u64>,
    avg_seen_ms: Option<u64>,
    avg_exec_report_ms: Option<u64>,
    avg_fill_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeAttemptReport {
    iteration: usize,
    target_qty: i32,
    reason_tag: String,
    sent_at_utc: DateTime<Utc>,
    send_to_submit_event_ms: Option<u64>,
    submit: Option<SwipeSubmitObservation>,
    dispatch_notes: Vec<String>,
    delay_probe: Option<ExecutionProbeSnapshot>,
    delay_findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SwipeSubmitObservation {
    received_at_utc: DateTime<Utc>,
    submit_message: String,
    broker_submit_ms: Option<u64>,
    request_id: Option<String>,
    seen_ms: Option<u64>,
    exec_report_ms: Option<u64>,
    fill_ms: Option<u64>,
}

enum AttemptDispatchProgress {
    Submitted,
    DispatchNote(String),
    Timeout,
}

struct LiveAttemptState {
    sent_at: Instant,
    report: SwipeAttemptReport,
}

struct AttemptBook {
    attempts: Vec<LiveAttemptState>,
    request_to_index: BTreeMap<String, usize>,
}

impl AttemptBook {
    fn new() -> Self {
        Self {
            attempts: Vec::new(),
            request_to_index: BTreeMap::new(),
        }
    }

    fn start_attempt(
        &mut self,
        iteration: usize,
        target_qty: i32,
        reason_tag: String,
    ) -> usize {
        let index = self.attempts.len();
        self.attempts.push(LiveAttemptState {
            sent_at: Instant::now(),
            report: SwipeAttemptReport {
                iteration,
                target_qty,
                reason_tag,
                sent_at_utc: Utc::now(),
                send_to_submit_event_ms: None,
                submit: None,
                dispatch_notes: Vec::new(),
                delay_probe: None,
                delay_findings: Vec::new(),
            },
        });
        index
    }

    fn observe(&mut self, at_utc: DateTime<Utc>, event: &ServiceEvent) {
        let Some(message) = event_message(event) else {
            return;
        };

        for (index, attempt) in self.attempts.iter_mut().enumerate() {
            if !message.contains(&attempt.report.reason_tag) {
                continue;
            }
            if !message.contains("submitted:") {
                if matches!(event, ServiceEvent::Status(_))
                    && !attempt
                        .report
                        .dispatch_notes
                        .iter()
                        .any(|note| note == message)
                {
                    attempt.report.dispatch_notes.push(message.to_string());
                }
                continue;
            }
            if attempt.report.submit.is_none() {
                attempt.report.send_to_submit_event_ms =
                    Some(attempt.sent_at.elapsed().as_millis() as u64);
                attempt.report.submit = Some(SwipeSubmitObservation {
                    received_at_utc: at_utc,
                    submit_message: message.to_string(),
                    broker_submit_ms: None,
                    request_id: parse_submit_request_id(message),
                    seen_ms: None,
                    exec_report_ms: None,
                    fill_ms: None,
                });
                if let Some(request_id) = attempt
                    .report
                    .submit
                    .as_ref()
                    .and_then(|submit| submit.request_id.clone())
                {
                    self.request_to_index.insert(request_id, index);
                }
            }
            if let (ServiceEvent::DebugLog(debug_message), Some(submit)) =
                (event, attempt.report.submit.as_mut())
            {
                if submit.broker_submit_ms.is_none() {
                    submit.broker_submit_ms = parse_debug_stage_ms(debug_message, "submit");
                }
                if submit.request_id.is_none() {
                    submit.request_id = parse_submit_request_id(debug_message);
                    if let Some(request_id) = submit.request_id.clone() {
                        self.request_to_index.insert(request_id, index);
                    }
                }
            }
        }

        if let ServiceEvent::DebugLog(debug_message) = event {
            let Some(request_id) = parse_stage_request_id(debug_message) else {
                return;
            };
            let Some(index) = self.request_to_index.get(&request_id).copied() else {
                return;
            };
            let Some(submit) = self.attempts[index].report.submit.as_mut() else {
                return;
            };
            if submit.seen_ms.is_none() {
                submit.seen_ms = parse_debug_stage_ms(debug_message, "seen");
            }
            if submit.exec_report_ms.is_none() {
                submit.exec_report_ms = parse_debug_stage_ms(debug_message, "ack");
            }
            if submit.fill_ms.is_none() {
                submit.fill_ms = parse_debug_stage_ms(debug_message, "fill");
            }
        }
    }

    fn set_delay_probe(
        &mut self,
        index: usize,
        probe: ExecutionProbeSnapshot,
        findings: Vec<String>,
    ) {
        if let Some(attempt) = self.attempts.get_mut(index) {
            attempt.report.delay_probe = Some(probe);
            attempt.report.delay_findings = findings;
        }
    }

}

#[derive(Default)]
struct ProfileHarnessState {
    accounts: Vec<AccountInfo>,
    contract_results: Vec<ContractSuggestion>,
    market: MarketSnapshot,
    latency: LatencySnapshot,
    execution_state: ExecutionStateSnapshot,
}

#[derive(Clone)]
struct ProfileRestInspector {
    client: Client,
    env: TradingEnvironment,
    access_token: String,
}

impl ProfileRestInspector {
    fn from_config(config: &AppConfig) -> Result<Self> {
        let access_token = if let Some(token) = empty_as_none(&config.token_override) {
            token.to_string()
        } else {
            load_token_file(&config.token_path)
                .or_else(|_| load_token_file(&config.session_cache_path))
                .with_context(|| {
                    format!(
                        "load token from {} or {}",
                        config.token_path.display(),
                        config.session_cache_path.display()
                    )
                })?
                .access_token
        };
        Ok(Self {
            client: Client::builder()
                .tcp_nodelay(true)
                .pool_idle_timeout(Duration::from_secs(300))
                .pool_max_idle_per_host(4)
                .tcp_keepalive(Duration::from_secs(30))
                .build()
                .context("build profiler REST client")?,
            env: config.env,
            access_token,
        })
    }

    async fn enrich_probe(&self, probe: &mut ExecutionProbeSnapshot) {
        let mut order_ids = probe
            .selected_working_orders
            .iter()
            .chain(probe.linked_active_orders.iter())
            .filter_map(|order| order.order_id)
            .collect::<BTreeSet<_>>();

        if let Some(order_strategy_id) = probe.broker_order_strategy_id {
            if let Ok(links) = self
                .fetch_entity_deps("orderStrategyLink", order_strategy_id)
                .await
            {
                for link in links {
                    if let Some(order_id) = json_i64(&link, "orderId") {
                        order_ids.insert(order_id);
                    }
                }
            }
        }

        let details = futures_util::future::join_all(
            order_ids
                .into_iter()
                .map(|order_id| self.fetch_order_probe_details(order_id)),
        )
        .await;

        for detail in details.into_iter().flatten() {
            merge_probe_order_detail(probe, detail);
        }
    }

    async fn fetch_order_probe_details(&self, order_id: i64) -> Option<ExecutionProbeOrder> {
        let order = self.fetch_entity_item("order", order_id).await.ok()?;
        let mut detail = execution_probe_order_snapshot(&order);
        let latest_version = self
            .fetch_entity_deps("orderVersion", order_id)
            .await
            .ok()
            .and_then(|versions| {
                versions
                    .into_iter()
                    .max_by_key(|version| extract_entity_id(version).unwrap_or_default())
            });

        if let Some(version) = latest_version.as_ref() {
            if detail.order_qty.is_none() {
                detail.order_qty = pick_number(version, &["orderQty", "qty", "quantity"])
                    .map(|value| value.abs().round() as i32);
            }
            if detail.order_type.is_none() {
                detail.order_type = version
                    .get("orderType")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
            }
            if detail.price.is_none() {
                detail.price = pick_number(version, &["price"]);
            }
            if detail.stop_price.is_none() {
                detail.stop_price = pick_number(version, &["stopPrice"]);
            }
        }

        Some(detail)
    }

    async fn fetch_entity_item(&self, entity: &str, id: i64) -> Result<Value> {
        let url = format!("{}/{entity}/item", self.env.rest_url());
        let response = self
            .client
            .get(url)
            .bearer_auth(&self.access_token)
            .query(&[("id", id.to_string())])
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{entity}/item failed ({status}): {body}");
        }
        Ok(serde_json::from_str(&body)?)
    }

    async fn fetch_entity_deps(&self, entity: &str, master_id: i64) -> Result<Vec<Value>> {
        let url = format!("{}/{entity}/deps", self.env.rest_url());
        let response = self
            .client
            .get(url)
            .bearer_auth(&self.access_token)
            .query(&[("masterid", master_id.to_string())])
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{entity}/deps failed ({status}): {body}");
        }
        let parsed: Value = serde_json::from_str(&body)?;
        Ok(match parsed {
            Value::Array(items) => items,
            Value::Object(_) => vec![parsed],
            _ => Vec::new(),
        })
    }
}

fn merge_probe_order_detail(probe: &mut ExecutionProbeSnapshot, detail: ExecutionProbeOrder) {
    let mut merged = false;
    for existing in probe
        .selected_working_orders
        .iter_mut()
        .chain(probe.linked_active_orders.iter_mut())
    {
        if existing.order_id != detail.order_id {
            continue;
        }
        if existing.order_strategy_id.is_none() {
            existing.order_strategy_id = detail.order_strategy_id;
        }
        if existing.cl_ord_id.is_none() {
            existing.cl_ord_id = detail.cl_ord_id.clone();
        }
        if existing.order_type.is_none() {
            existing.order_type = detail.order_type.clone();
        }
        if existing.action.is_none() {
            existing.action = detail.action.clone();
        }
        if existing.order_qty.is_none() {
            existing.order_qty = detail.order_qty;
        }
        if existing.price.is_none() {
            existing.price = detail.price;
        }
        if existing.stop_price.is_none() {
            existing.stop_price = detail.stop_price;
        }
        if existing.status.is_none() {
            existing.status = detail.status.clone();
        }
        merged = true;
    }

    if !merged {
        probe.linked_active_orders.push(detail);
    }
}

struct ProcessedEvent {
    at_utc: DateTime<Utc>,
    event: ServiceEvent,
}

struct ProfileHarness {
    cmd_tx: UnboundedSender<ServiceCommand>,
    event_rx: UnboundedReceiver<ServiceEvent>,
    market_rx: watch::Receiver<MarketSnapshot>,
    rest_inspector: Option<ProfileRestInspector>,
    state: ProfileHarnessState,
    raw_log_lines: Vec<String>,
}

impl ProfileHarness {
    fn new(
        cmd_tx: UnboundedSender<ServiceCommand>,
        event_rx: UnboundedReceiver<ServiceEvent>,
        market_rx: watch::Receiver<MarketSnapshot>,
        rest_inspector: Option<ProfileRestInspector>,
    ) -> Self {
        Self {
            cmd_tx,
            event_rx,
            market_rx,
            rest_inspector,
            state: ProfileHarnessState::default(),
            raw_log_lines: Vec::new(),
        }
    }

    fn send(&self, command: ServiceCommand) -> Result<()> {
        self.cmd_tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("service command channel is closed"))
    }

    async fn recv_event(&mut self, timeout: Duration) -> Result<Option<ProcessedEvent>> {
        let maybe_event = tokio::time::timeout(timeout, async {
            tokio::select! {
                maybe_event = self.event_rx.recv() => maybe_event,
                changed = self.market_rx.changed() => {
                    if changed.is_err() {
                        None
                    } else {
                        Some(ServiceEvent::MarketSnapshot(self.market_rx.borrow().clone()))
                    }
                }
            }
        })
        .await
        .ok()
        .flatten();
        let Some(event) = maybe_event else {
            return Ok(None);
        };
        let processed = ProcessedEvent {
            at_utc: Utc::now(),
            event,
        };
        self.record(&processed);
        Ok(Some(processed))
    }

    fn record(&mut self, processed: &ProcessedEvent) {
        match &processed.event {
            ServiceEvent::AccountsLoaded(accounts) => {
                self.state.accounts = accounts.clone();
            }
            ServiceEvent::ContractSearchResults { results, .. } => {
                self.state.contract_results = results.clone();
            }
            ServiceEvent::MarketSnapshot(snapshot) => {
                self.state.market = snapshot.clone();
            }
            ServiceEvent::Latency(latency) => {
                self.state.latency = *latency;
            }
            ServiceEvent::ExecutionState(snapshot) => {
                self.state.execution_state = snapshot.clone();
            }
            _ => {}
        }

        if let Some(line) = format_log_line(processed) {
            self.raw_log_lines.push(line);
        }
    }

    async fn wait_for<F>(
        &mut self,
        timeout: Duration,
        attempts: &mut AttemptBook,
        mut predicate: F,
    ) -> Result<ProcessedEvent>
    where
        F: FnMut(&ProfileHarnessState, &ProcessedEvent) -> bool,
    {
        let started = Instant::now();
        loop {
            let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
                bail!("timed out after {}ms waiting for broker event", timeout.as_millis());
            };
            let Some(processed) = self.recv_event(remaining).await? else {
                bail!("timed out after {}ms waiting for broker event", timeout.as_millis());
            };
            attempts.observe(processed.at_utc, &processed.event);
            if let ServiceEvent::Error(message) = &processed.event {
                bail!("{message}");
            }
            if predicate(&self.state, &processed) {
                return Ok(processed);
            }
        }
    }

    async fn pump_for(&mut self, duration: Duration, attempts: &mut AttemptBook) -> Result<()> {
        let started = Instant::now();
        loop {
            let Some(remaining) = duration.checked_sub(started.elapsed()) else {
                return Ok(());
            };
            let Some(processed) = self.recv_event(remaining).await? else {
                return Ok(());
            };
            attempts.observe(processed.at_utc, &processed.event);
            if let ServiceEvent::Error(message) = processed.event {
                bail!("{message}");
            }
        }
    }

    async fn request_probe(
        &mut self,
        tag: &str,
        attempts: &mut AttemptBook,
    ) -> Result<ExecutionProbeSnapshot> {
        self.send(ServiceCommand::ProbeExecution {
            tag: tag.to_string(),
        })?;
        let processed = self
            .wait_for(PROBE_TIMEOUT, attempts, |_, processed| {
                matches!(
                    &processed.event,
                    ServiceEvent::ExecutionProbe(snapshot) if snapshot.tag == tag
                )
            })
            .await?;
        match processed.event {
            ServiceEvent::ExecutionProbe(mut snapshot) => {
                if let Some(inspector) = self.rest_inspector.as_ref() {
                    inspector.enrich_probe(&mut snapshot).await;
                }
                Ok(snapshot)
            }
            _ => bail!("expected execution probe response for {tag}"),
        }
    }
}

pub async fn run_swipe_profile(mut config: AppConfig, options: SwipeProfileOptions) -> Result<()> {
    config.broker = BrokerKind::Tradovate;
    config.env = TradingEnvironment::Sim;
    config.auth_mode = AuthMode::TokenFile;
    config.autoconnect = false;

    let report_dir = profile_output_dir(options.output_dir.as_ref())?;
    let raw_log_path = report_dir.join("events.log");
    let json_report_path = report_dir.join("report.json");
    let text_report_path = report_dir.join("report.txt");

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (market_tx, market_rx) = watch::channel(MarketSnapshot::default());
    tokio::spawn(crate::broker::service_loop(cmd_rx, event_tx, market_tx));
    let rest_inspector = Some(ProfileRestInspector::from_config(&config)?);
    let mut harness = ProfileHarness::new(cmd_tx, event_rx, market_rx, rest_inspector);
    let mut attempts = AttemptBook::new();

    let run_result: Result<SwipeProfileReport> = async {
        harness.send(ServiceCommand::Connect(config.clone()))?;
        harness
            .wait_for(CONNECT_TIMEOUT, &mut attempts, |_, processed| {
                matches!(processed.event, ServiceEvent::Connected { .. })
            })
            .await
            .context("waiting for Tradovate Connected event")?;
        if harness.state.accounts.is_empty() {
            harness
                .wait_for(ACCOUNT_BOOTSTRAP_TIMEOUT, &mut attempts, |state, _| {
                    !state.accounts.is_empty()
                })
                .await
                .context("waiting for account bootstrap to complete")?;
        }

        let selected_account = select_profile_account(&harness.state.accounts, &options.account_filter)
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
            .wait_for(CONTRACT_SEARCH_TIMEOUT, &mut attempts, |state, processed| {
                matches!(processed.event, ServiceEvent::ContractSearchResults { .. })
                    && !state.contract_results.is_empty()
            })
            .await
            .context("waiting for ES contract search results")?;
        let selected_contract = select_profile_contract(
            &harness.state.contract_results,
            &options.contract_query,
            options.contract_exact.as_deref(),
        )
        .context("no matching ES contract found from sim search results")?;

        harness.send(ServiceCommand::SubscribeBars {
            contract: selected_contract.clone(),
            bar_type: BarType::Range1,
        })?;
        harness
            .wait_for(SUBSCRIBE_TIMEOUT, &mut attempts, |state, _| {
                state.market.contract_id == Some(selected_contract.id)
                    && state.market.tick_size.is_some()
                    && !state.market.bars.is_empty()
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
            harness.send(ServiceCommand::SetExecutionStrategyConfig(execution_config.clone()))?;
            harness
                .wait_for(EXECUTION_CONFIG_TIMEOUT, &mut attempts, |state, _| {
                    state.execution_state.config == execution_config
                })
                .await
                .with_context(|| format!("waiting for execution config sync for {}", scenario.slug()))?;
            harness.send(ServiceCommand::ArmExecutionStrategy)?;
            harness
                .wait_for(EXECUTION_ARM_TIMEOUT, &mut attempts, |state, _| {
                    state.execution_state.runtime.armed
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
                let delay_report = run_delay_sweep(
                    &mut harness,
                    &mut attempts,
                    scenario,
                    *delay_ms,
                    &options,
                )
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
                        state.execution_state.runtime.armed
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
            bar_type: BarType::Range1,
            options: SwipeProfileReportOptions {
                account_filter: options.account_filter,
                contract_query: options.contract_query,
                contract_exact: options.contract_exact,
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

    fs::write(&raw_log_path, harness.raw_log_lines.join("\n"))
        .with_context(|| format!("write {}", raw_log_path.display()))?;

    let report = run_result?;

    fs::write(&json_report_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", json_report_path.display()))?;
    fs::write(&text_report_path, render_text_report(&report))
        .with_context(|| format!("write {}", text_report_path.display()))?;

    println!("Swipe profile complete.");
    println!("Account: {} ({})", report.account_name, report.account_id);
    println!("Contract: {} ({})", report.contract_name, report.contract_id);
    println!("Event log: {}", raw_log_path.display());
    println!("JSON report: {}", json_report_path.display());
    println!("Text report: {}", text_report_path.display());

    Ok(())
}

async fn open_bootstrap_long(
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
        submit: attempts
            .attempts
            .get(bootstrap_index)
            .and_then(|attempt| attempt.report.submit.clone()),
        dispatch_notes: attempts
            .attempts
            .get(bootstrap_index)
            .map(|attempt| attempt.report.dispatch_notes.clone())
            .unwrap_or_default(),
        settled,
        settle_ms,
        final_probe,
        findings,
    })
}

async fn run_delay_sweep(
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
        &format!("swipe-profile:{}:delay-{}ms:final", scenario.slug(), delay_ms),
    )
    .await?;

    let mut reports = Vec::new();
    for attempt_index in attempt_indexes {
        if let Some(report) = attempts
            .attempts
            .get(attempt_index)
            .map(|attempt| attempt.report.clone())
        {
            reports.push(report);
        }
    }

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

async fn flatten_selected_contract(
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
        if attempts
            .attempts
            .iter()
            .find(|attempt| attempt.report.reason_tag == reason_tag)
            .and_then(|attempt| attempt.report.submit.clone())
            .is_some()
        {
            return Ok(AttemptDispatchProgress::Submitted);
        }
        last_note = attempts
            .attempts
            .iter()
            .find(|attempt| attempt.report.reason_tag == reason_tag)
            .and_then(|attempt| attempt.report.dispatch_notes.last().cloned())
            .or(last_note);
        let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
            return Ok(last_note
                .map(AttemptDispatchProgress::DispatchNote)
                .unwrap_or(AttemptDispatchProgress::Timeout));
        };
        let Some(processed) = harness.recv_event(remaining).await? else {
            return Ok(last_note
                .map(AttemptDispatchProgress::DispatchNote)
                .unwrap_or(AttemptDispatchProgress::Timeout));
        };
        attempts.observe(processed.at_utc, &processed.event);
        if let ServiceEvent::Error(message) = processed.event {
            bail!("{message}");
        }
    }
}

async fn wait_for_settled_state(
    harness: &mut ProfileHarness,
    attempts: &mut AttemptBook,
    expected_qty: i32,
    expected_abs_qty: i32,
    timeout: Duration,
    probe_prefix: &str,
) -> Result<(bool, Option<u64>, Option<ExecutionProbeSnapshot>, Vec<String>)> {
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

fn build_profile_execution_config(
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

fn select_profile_account<'a>(
    accounts: &'a [AccountInfo],
    filter: &str,
) -> Option<&'a AccountInfo> {
    let needle = filter.trim().to_ascii_lowercase();
    accounts
        .iter()
        .find(|account| account.name.to_ascii_lowercase().contains(&needle))
        .or_else(|| {
            accounts
                .iter()
                .find(|account| account.name.to_ascii_lowercase().contains("demo"))
        })
        .or_else(|| accounts.first())
}

fn select_profile_contract(
    contracts: &[ContractSuggestion],
    query: &str,
    exact: Option<&str>,
) -> Option<ContractSuggestion> {
    let exact = exact.map(|value| value.trim().to_ascii_uppercase());
    if let Some(exact) = exact.as_deref() {
        if let Some(contract) = contracts
            .iter()
            .find(|contract| contract.name.eq_ignore_ascii_case(exact))
        {
            return Some(contract.clone());
        }
    }

    let query = query.trim().to_ascii_uppercase();
    contracts
        .iter()
        .find(|contract| {
            let name = contract.name.to_ascii_uppercase();
            let description = contract.description.to_ascii_uppercase();
            name.starts_with(&query)
                && !name.starts_with("MES")
                && description.contains("E-MINI S&P")
        })
        .or_else(|| {
            contracts
                .iter()
                .find(|contract| contract.name.to_ascii_uppercase().starts_with(&query))
        })
        .cloned()
}

fn profile_output_dir(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
        return Ok(path.clone());
    }
    let dir = PathBuf::from(".run")
        .join("swipe-profiles")
        .join(Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

fn protective_orders(probe: &ExecutionProbeSnapshot) -> Vec<&ExecutionProbeOrder> {
    probe.selected_working_orders
        .iter()
        .filter(|order| {
            order.order_type
                .as_deref()
                .map(|order_type| {
                    matches!(
                        order_type.to_ascii_lowercase().as_str(),
                        "limit"
                            | "mit"
                            | "stop"
                            | "stoplimit"
                            | "trailingstop"
                            | "trailingstoplimit"
                    )
                })
                .unwrap_or(false)
        })
        .collect()
}

fn has_matching_strategy_owned_protection(
    probe: &ExecutionProbeSnapshot,
    expected_abs_qty: i32,
) -> bool {
    let bracket_qtys_match = !probe.broker_strategy_bracket_qtys.is_empty()
        && probe
            .broker_strategy_bracket_qtys
            .iter()
            .all(|qty| *qty == expected_abs_qty);
    bracket_qtys_match && (protective_orders(probe).len() >= 2 || probe.linked_active_orders.len() >= 2)
}

fn has_any_visible_broker_path(probe: &ExecutionProbeSnapshot) -> bool {
    probe.order_submit_in_flight
        || probe.protection_sync_in_flight
        || probe.tracker_order_is_active
        || probe.tracker_strategy_has_live_orders
        || probe.tracker_within_strategy_grace
        || !probe.selected_working_orders.is_empty()
        || !probe.linked_active_orders.is_empty()
        || probe.managed_protection.is_some()
}

fn probe_is_settled(
    probe: &ExecutionProbeSnapshot,
    expected_qty: i32,
    expected_abs_qty: i32,
) -> bool {
    if probe.execution_state.market_position_qty != expected_qty {
        return false;
    }
    if expected_qty == 0 {
        return !has_any_visible_broker_path(probe);
    }

    let protective_orders = protective_orders(probe);
    let has_matching_visible_protection = probe
        .managed_protection
        .as_ref()
        .is_some_and(|protection| {
            protection.signed_qty.abs() == expected_abs_qty
                && protection.take_profit_price.is_some()
                && protection.stop_price.is_some()
        })
        || (protective_orders.len() >= 2
            && protective_orders
                .iter()
                .all(|order| order.order_qty == Some(expected_abs_qty)))
        || has_matching_strategy_owned_protection(probe, expected_abs_qty);

    has_matching_visible_protection
}

fn probe_findings(probe: &ExecutionProbeSnapshot, expected_abs_qty: i32) -> Vec<String> {
    let mut findings = Vec::new();
    let actual_qty = probe.execution_state.market_position_qty;
    if actual_qty.abs() > expected_abs_qty {
        findings.push(format!(
            "position {} exceeds configured qty {}",
            actual_qty, expected_abs_qty
        ));
    }

    let protective_orders = protective_orders(probe);
    let mismatched_leg_qtys = protective_orders
        .iter()
        .filter_map(|order| order.order_qty)
        .filter(|qty| *qty != expected_abs_qty)
        .collect::<Vec<_>>();
    if !mismatched_leg_qtys.is_empty() {
        findings.push(format!(
            "visible protection leg qty mismatch: {:?} (expected {})",
            mismatched_leg_qtys, expected_abs_qty
        ));
    }
    let mismatched_strategy_bracket_qtys = probe
        .broker_strategy_bracket_qtys
        .iter()
        .copied()
        .filter(|qty| *qty != expected_abs_qty)
        .collect::<Vec<_>>();
    if !mismatched_strategy_bracket_qtys.is_empty() {
        findings.push(format!(
            "broker strategy bracket qty mismatch: {:?} (expected {})",
            mismatched_strategy_bracket_qtys, expected_abs_qty
        ));
    }
    if protective_orders.len() > 2 {
        findings.push(format!(
            "visible protective order count is {} (expected at most 2)",
            protective_orders.len()
        ));
    }
    if probe.linked_active_orders.len() > 2 {
        findings.push(format!(
            "linked strategy order count is {} (expected at most 2)",
            probe.linked_active_orders.len()
        ));
    }
    if actual_qty != 0
        && probe
            .managed_protection
            .as_ref()
            .is_none_or(|protection| protection.signed_qty.abs() != expected_abs_qty)
        && protective_orders.is_empty()
        && probe.linked_active_orders.is_empty()
        && !has_any_visible_broker_path(probe)
    {
        findings.push("no visible protection on selected contract".to_string());
    }

    findings
}

fn summarize_delay_reports(attempts: &[SwipeAttemptReport]) -> SwipeDelaySummary {
    SwipeDelaySummary {
        attempts_total: attempts.len(),
        submit_observed: attempts.iter().filter(|attempt| attempt.submit.is_some()).count(),
        fill_observed: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .submit
                    .as_ref()
                    .and_then(|submit| submit.fill_ms)
                    .is_some()
            })
            .count(),
        delay_snapshots_with_mismatched_leg_qty: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("leg qty mismatch"))
            })
            .count(),
        delay_snapshots_with_no_visible_protection: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("no visible protection"))
            })
            .count(),
        delay_snapshots_with_oversized_position: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("exceeds configured qty"))
            })
            .count(),
        avg_submit_rtt_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| attempt.submit.as_ref().and_then(|submit| submit.broker_submit_ms)),
        ),
        avg_seen_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| attempt.submit.as_ref().and_then(|submit| submit.seen_ms)),
        ),
        avg_exec_report_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| {
                    attempt
                        .submit
                        .as_ref()
                        .and_then(|submit| submit.exec_report_ms)
                }),
        ),
        avg_fill_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| attempt.submit.as_ref().and_then(|submit| submit.fill_ms)),
        ),
    }
}

fn avg_option_u64(values: impl Iterator<Item = u64>) -> Option<u64> {
    let mut count = 0u64;
    let mut total = 0u64;
    for value in values {
        total = total.saturating_add(value);
        count = count.saturating_add(1);
    }
    if count == 0 {
        None
    } else {
        Some(total / count)
    }
}

fn parse_submit_request_id(message: &str) -> Option<String> {
    bracketed_value(message, "uuid ")
        .or_else(|| bracketed_value(message, "clOrdId "))
}

fn parse_stage_request_id(message: &str) -> Option<String> {
    bracketed_value(message, "request ")
}

fn bracketed_value(message: &str, prefix: &str) -> Option<String> {
    let needle = format!("[{prefix}");
    let start = message.find(&needle)? + needle.len();
    let tail = &message[start..];
    let end = tail.find(']')?;
    Some(tail[..end].to_string())
}

fn parse_debug_stage_ms(message: &str, stage: &str) -> Option<u64> {
    let prefix = format!("{stage} ");
    let tail = message.strip_prefix(&prefix)?;
    let token = tail.split_whitespace().next()?;
    parse_duration_token_ms(token)
}

fn parse_duration_token_ms(token: &str) -> Option<u64> {
    if let Some(raw) = token.strip_suffix("ms") {
        return raw.parse::<f64>().ok().map(|value| value.round() as u64);
    }
    if let Some(raw) = token.strip_suffix('s') {
        return raw
            .parse::<f64>()
            .ok()
            .map(|value| (value * 1000.0).round() as u64);
    }
    if let Some(raw) = token.strip_suffix('m') {
        return raw
            .parse::<f64>()
            .ok()
            .map(|value| (value * 60_000.0).round() as u64);
    }
    None
}

fn event_message(event: &ServiceEvent) -> Option<&str> {
    match event {
        ServiceEvent::Status(message)
        | ServiceEvent::DebugLog(message)
        | ServiceEvent::Error(message) => Some(message),
        _ => None,
    }
}

fn format_log_line(processed: &ProcessedEvent) -> Option<String> {
    let timestamp = processed.at_utc.format("%H:%M:%S%.3f");
    let message = match &processed.event {
        ServiceEvent::Status(message) => format!("STATUS {message}"),
        ServiceEvent::DebugLog(message) => format!("DEBUG {message}"),
        ServiceEvent::Error(message) => format!("ERROR {message}"),
        ServiceEvent::Connected {
            broker,
            env,
            user_name,
            session_kind,
            ..
        } => format!(
            "CONNECTED broker={} env={} session={} user={}",
            broker.label(),
            env.label(),
            session_kind.label(),
            user_name.as_deref().unwrap_or("n/a")
        ),
        ServiceEvent::AccountsLoaded(accounts) => {
            let names = accounts
                .iter()
                .map(|account| account.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("ACCOUNTS count={} [{}]", accounts.len(), names)
        }
        ServiceEvent::ContractSearchResults { query, results } => format!(
            "CONTRACTS query={} count={} [{}]",
            query,
            results.len(),
            results
                .iter()
                .map(|contract| contract.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ServiceEvent::ExecutionState(snapshot) => format!(
            "EXECUTION armed={} pending={:?} qty={} summary={}",
            snapshot.runtime.armed,
            snapshot.runtime.pending_target_qty,
            snapshot.market_position_qty,
            snapshot.runtime.last_summary
        ),
        ServiceEvent::ExecutionProbe(snapshot) => format!(
            "PROBE tag={} qty={} strategy={:?}/{:?} strategy_params={:?}/{:?} tracker={:?}/active={} strategy_live={} grace={} submit_in_flight={} protection_in_flight={} visible_orders={} linked_orders={} visible_qtys={:?} linked_qtys={:?} managed={}",
            snapshot.tag,
            snapshot.execution_state.market_position_qty,
            snapshot.tracked_order_strategy_id,
            snapshot.broker_order_strategy_id,
            snapshot.broker_strategy_entry_order_qty,
            snapshot.broker_strategy_bracket_qtys,
            snapshot.tracker_order_strategy_id,
            snapshot.tracker_order_is_active,
            snapshot.tracker_strategy_has_live_orders,
            snapshot.tracker_within_strategy_grace,
            snapshot.order_submit_in_flight,
            snapshot.protection_sync_in_flight,
            snapshot.selected_working_orders.len(),
            snapshot.linked_active_orders.len(),
            snapshot
                .selected_working_orders
                .iter()
                .filter_map(|order| order.order_qty)
                .collect::<Vec<_>>(),
            snapshot
                .linked_active_orders
                .iter()
                .filter_map(|order| order.order_qty)
                .collect::<Vec<_>>(),
            snapshot
                .managed_protection
                .as_ref()
                .map(|protection| format!(
                    "qty={} tp={:?} sl={:?}",
                    protection.signed_qty, protection.take_profit_price, protection.stop_price
                ))
                .unwrap_or_else(|| "none".to_string())
        ),
        _ => return None,
    };
    Some(format!("[{timestamp}] {message}"))
}

fn render_text_report(report: &SwipeProfileReport) -> String {
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

#[cfg(test)]
mod profiler_tests {
    use super::*;
    use crate::broker::{ExecutionProbeOrder, LatencySnapshot};
    use crate::strategy::ExecutionStateSnapshot;

    fn base_probe() -> ExecutionProbeSnapshot {
        let mut execution_state = ExecutionStateSnapshot::default();
        execution_state.market_position_qty = 1;
        ExecutionProbeSnapshot {
            tag: "test".to_string(),
            captured_at_utc: Utc::now(),
            execution_state,
            latency: LatencySnapshot::default(),
            order_submit_in_flight: false,
            protection_sync_in_flight: false,
            tracker_order_id: None,
            tracker_order_is_active: false,
            tracker_order_strategy_id: None,
            tracker_strategy_has_live_orders: false,
            tracker_within_strategy_grace: false,
            tracked_order_strategy_id: None,
            broker_order_strategy_id: Some(77),
            broker_order_strategy_status: Some("Working".to_string()),
            broker_strategy_entry_order_qty: Some(1),
            broker_strategy_bracket_qtys: vec![1],
            selected_working_orders: Vec::new(),
            linked_active_orders: Vec::new(),
            managed_protection: None,
        }
    }

    #[test]
    fn strategy_owned_bracket_with_matching_params_counts_as_settled() {
        let mut probe = base_probe();
        probe.linked_active_orders = vec![
            ExecutionProbeOrder {
                order_id: Some(1001),
                order_strategy_id: Some(77),
                cl_ord_id: Some("midas-tp".to_string()),
                order_type: Some("Limit".to_string()),
                action: Some("Sell".to_string()),
                order_qty: None,
                price: Some(5030.0),
                stop_price: None,
                status: Some("Working".to_string()),
            },
            ExecutionProbeOrder {
                order_id: Some(1002),
                order_strategy_id: Some(77),
                cl_ord_id: Some("midas-sl".to_string()),
                order_type: Some("Stop".to_string()),
                action: Some("Sell".to_string()),
                order_qty: None,
                price: None,
                stop_price: Some(4970.0),
                status: Some("Working".to_string()),
            },
        ];

        assert!(probe_is_settled(&probe, 1, 1));
    }

    #[test]
    fn merge_probe_order_detail_backfills_qty_from_rest_version() {
        let mut probe = base_probe();
        probe.selected_working_orders = vec![ExecutionProbeOrder {
            order_id: Some(1001),
            order_strategy_id: Some(77),
            cl_ord_id: Some("midas-sl".to_string()),
            order_type: Some("Stop".to_string()),
            action: Some("Sell".to_string()),
            order_qty: None,
            price: None,
            stop_price: Some(4970.0),
            status: Some("Working".to_string()),
        }];

        merge_probe_order_detail(
            &mut probe,
            ExecutionProbeOrder {
                order_id: Some(1001),
                order_strategy_id: Some(77),
                cl_ord_id: Some("midas-sl".to_string()),
                order_type: Some("Stop".to_string()),
                action: Some("Sell".to_string()),
                order_qty: Some(1),
                price: None,
                stop_price: Some(4970.0),
                status: Some("Working".to_string()),
            },
        );

        assert_eq!(probe.selected_working_orders[0].order_qty, Some(1));
    }
}
