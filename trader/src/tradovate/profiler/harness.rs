use super::attempts::AttemptBook;
use super::*;
use std::collections::BTreeSet;
use std::time::Instant;
use tokio::sync::watch;

#[derive(Default)]
pub(super) struct ProfileHarnessState {
    accounts: Vec<AccountInfo>,
    contract_results: Vec<ContractSuggestion>,
    market: MarketSnapshot,
    latency: LatencySnapshot,
    execution_state: ExecutionStateSnapshot,
}

impl ProfileHarnessState {
    pub(super) fn accounts(&self) -> &[AccountInfo] {
        &self.accounts
    }

    pub(super) fn contract_results(&self) -> &[ContractSuggestion] {
        &self.contract_results
    }

    pub(super) fn market(&self) -> &MarketSnapshot {
        &self.market
    }

    pub(super) fn execution_state(&self) -> &ExecutionStateSnapshot {
        &self.execution_state
    }
}

#[derive(Clone)]
pub(super) struct ProfileRestInspector {
    client: Client,
    env: TradingEnvironment,
    access_token: String,
}

impl ProfileRestInspector {
    pub(super) fn from_config(config: &AppConfig) -> Result<Self> {
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

    pub(super) async fn enrich_probe(&self, probe: &mut ExecutionProbeSnapshot) {
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

pub(super) fn merge_probe_order_detail(
    probe: &mut ExecutionProbeSnapshot,
    detail: ExecutionProbeOrder,
) {
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

pub(super) struct ProfileHarness {
    cmd_tx: UnboundedSender<ServiceCommand>,
    event_rx: UnboundedReceiver<ServiceEvent>,
    market_rx: watch::Receiver<MarketSnapshot>,
    rest_inspector: Option<ProfileRestInspector>,
    state: ProfileHarnessState,
    raw_log_lines: Vec<String>,
}

impl ProfileHarness {
    pub(super) fn new(
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

    pub(super) fn send(&self, command: ServiceCommand) -> Result<()> {
        self.cmd_tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("service command channel is closed"))
    }

    pub(super) fn accounts(&self) -> &[AccountInfo] {
        self.state.accounts()
    }

    pub(super) fn contract_results(&self) -> &[ContractSuggestion] {
        self.state.contract_results()
    }

    pub(super) fn raw_log_output(&self) -> String {
        self.raw_log_lines.join("\n")
    }

    pub(super) async fn next_event(
        &mut self,
        timeout: Duration,
        attempts: &mut AttemptBook,
    ) -> Result<Option<ServiceEvent>> {
        let Some(processed) = self.recv_event(timeout).await? else {
            return Ok(None);
        };
        attempts.observe(processed.at_utc, &processed.event);
        if let ServiceEvent::Error(message) = &processed.event {
            bail!("{message}");
        }
        Ok(Some(processed.event))
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

    pub(super) async fn wait_for<F>(
        &mut self,
        timeout: Duration,
        attempts: &mut AttemptBook,
        mut predicate: F,
    ) -> Result<ServiceEvent>
    where
        F: FnMut(&ProfileHarnessState, &ServiceEvent) -> bool,
    {
        let started = Instant::now();
        loop {
            let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
                bail!(
                    "timed out after {}ms waiting for broker event",
                    timeout.as_millis()
                );
            };
            let Some(event) = self.next_event(remaining, attempts).await? else {
                bail!(
                    "timed out after {}ms waiting for broker event",
                    timeout.as_millis()
                );
            };
            if predicate(&self.state, &event) {
                return Ok(event);
            }
        }
    }

    pub(super) async fn pump_for(
        &mut self,
        duration: Duration,
        attempts: &mut AttemptBook,
    ) -> Result<()> {
        let started = Instant::now();
        loop {
            let Some(remaining) = duration.checked_sub(started.elapsed()) else {
                return Ok(());
            };
            let Some(_) = self.next_event(remaining, attempts).await? else {
                return Ok(());
            };
        }
    }

    pub(super) async fn request_probe(
        &mut self,
        tag: &str,
        attempts: &mut AttemptBook,
    ) -> Result<ExecutionProbeSnapshot> {
        self.send(ServiceCommand::ProbeExecution {
            tag: tag.to_string(),
        })?;
        let event = self
            .wait_for(PROBE_TIMEOUT, attempts, |_, event| {
                matches!(
                    event,
                    ServiceEvent::ExecutionProbe(snapshot) if snapshot.tag == tag
                )
            })
            .await?;
        match event {
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
