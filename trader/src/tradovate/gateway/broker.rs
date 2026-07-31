use super::*;

#[cfg(feature = "replay")]
use crate::broker::{
    ReplayBarProtectionPolicy, ReplayEngineMode, ReplayLatencyConfig, ReplayLatencyModel,
};
#[cfg(feature = "replay")]
use crate::tradovate::replay::virtual_time::{ReplayVirtualEventKind, ReplayVirtualEventQueue};
#[cfg(feature = "replay")]
use std::collections::VecDeque;

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
struct ReplayLatencySampler {
    config: ReplayLatencyConfig,
    draw_index: u64,
}

#[cfg(feature = "replay")]
impl Default for ReplayLatencySampler {
    fn default() -> Self {
        Self {
            config: ReplayLatencyConfig::default(),
            draw_index: 0,
        }
    }
}

#[cfg(feature = "replay")]
impl ReplayLatencySampler {
    fn reset(&mut self, config: ReplayLatencyConfig) {
        self.config = config;
        self.draw_index = 0;
    }

    fn next_latency_ms(&mut self) -> u64 {
        match self.config.model {
            ReplayLatencyModel::IgnoredLegacy | ReplayLatencyModel::Fixed => {
                self.config.fixed_latency_ms
            }
            ReplayLatencyModel::ObservedMean => observed_mean(&self.config.observed_samples_ms),
            ReplayLatencyModel::ObservedP95 => {
                observed_percentile(&self.config.observed_samples_ms, 95)
            }
            ReplayLatencyModel::ObservedP99 => {
                observed_percentile(&self.config.observed_samples_ms, 99)
            }
            ReplayLatencyModel::SeededObserved => {
                let samples = &self.config.observed_samples_ms;
                if samples.is_empty() {
                    return self.config.fixed_latency_ms;
                }
                let random = splitmix64(self.config.seed.wrapping_add(self.draw_index));
                self.draw_index = self.draw_index.saturating_add(1);
                samples[(random % samples.len() as u64) as usize]
            }
        }
    }
}

#[cfg(feature = "replay")]
fn observed_mean(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let total = samples
        .iter()
        .fold(0_u128, |sum, sample| sum.saturating_add(*sample as u128));
    ((total + (samples.len() as u128 / 2)) / samples.len() as u128) as u64
}

#[cfg(feature = "replay")]
fn observed_percentile(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (percentile.saturating_mul(sorted.len()).saturating_add(99) / 100)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

#[cfg(feature = "replay")]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

#[cfg(feature = "replay")]
enum DeferredReplayCommandKind {
    MarketOrder(PendingMarketOrder),
    LiquidatePosition(PendingLiquidation),
    OrderStrategy(PendingOrderStrategyTransition),
    LiquidateThenOrderStrategy(PendingLiquidation, PendingOrderStrategyTransition),
}

#[cfg(feature = "replay")]
impl DeferredReplayCommandKind {
    fn reference_ts_ns(&self) -> Option<i64> {
        match self {
            Self::MarketOrder(order) => order.reference_ts_ns,
            Self::LiquidatePosition(liquidation) => liquidation.reference_ts_ns,
            Self::OrderStrategy(strategy) => strategy.reference_ts_ns,
            Self::LiquidateThenOrderStrategy(liquidation, strategy) => {
                liquidation.reference_ts_ns.or(strategy.reference_ts_ns)
            }
        }
    }
}

#[cfg(feature = "replay")]
struct DeferredReplayCommand {
    lifecycle_sequence: u64,
    latency_ms: u64,
    signal_ts_ns: Option<i64>,
    exchange_arrival_ts_ns: Option<i64>,
    scheduled: bool,
    arrived: bool,
    kind: DeferredReplayCommandKind,
}

#[cfg(feature = "replay")]
impl DeferredReplayCommand {
    fn new(
        lifecycle_sequence: u64,
        fixed_latency_ms: u64,
        kind: DeferredReplayCommandKind,
    ) -> Self {
        let signal_ts_ns = kind.reference_ts_ns();
        let fixed_latency_ns =
            i64::try_from(fixed_latency_ms.saturating_mul(1_000_000)).unwrap_or(i64::MAX);
        let exchange_arrival_ts_ns =
            signal_ts_ns.map(|timestamp| timestamp.saturating_add(fixed_latency_ns));
        Self {
            lifecycle_sequence,
            latency_ms: fixed_latency_ms,
            signal_ts_ns,
            exchange_arrival_ts_ns,
            scheduled: false,
            arrived: false,
            kind,
        }
    }

    fn execute(self, replay_state: &mut ReplayBrokerState, bar: &Bar) -> Vec<InternalEvent> {
        let signal_ts_ns = self.signal_ts_ns.unwrap_or(bar.ts_ns);
        let exchange_arrival_ts_ns = self.exchange_arrival_ts_ns.unwrap_or(signal_ts_ns);
        let lifecycle_sequence = self.lifecycle_sequence;
        let latency_ms = self.latency_ms;
        let mut events = match self.kind {
            DeferredReplayCommandKind::MarketOrder(mut order) => {
                order.reference_ts_ns = Some(bar.ts_ns);
                order.reference_price = Some(bar.open);
                replay_state
                    .simulate_market_order(order)
                    .unwrap_or_else(|failure| vec![InternalEvent::BrokerOrderFailed(failure)])
            }
            DeferredReplayCommandKind::LiquidatePosition(mut liquidation) => {
                liquidation.reference_ts_ns = Some(bar.ts_ns);
                liquidation.reference_price = Some(bar.open);
                replay_state
                    .simulate_liquidation(liquidation)
                    .unwrap_or_else(|failure| vec![InternalEvent::BrokerOrderFailed(failure)])
            }
            DeferredReplayCommandKind::OrderStrategy(mut strategy) => {
                reprice_strategy_for_bar_open(&mut strategy, bar);
                replay_state
                    .simulate_order_strategy(strategy)
                    .unwrap_or_else(|failure| vec![InternalEvent::OrderStrategyFailed(failure)])
            }
            DeferredReplayCommandKind::LiquidateThenOrderStrategy(
                mut liquidation,
                mut strategy,
            ) => {
                liquidation.reference_ts_ns = Some(bar.ts_ns);
                liquidation.reference_price = Some(bar.open);
                reprice_strategy_for_bar_open(&mut strategy, bar);
                replay_state
                    .simulate_liquidation_then_order_strategy(liquidation, strategy)
                    .unwrap_or_else(|failure| vec![InternalEvent::OrderStrategyFailed(failure)])
            }
        };
        apply_replay_latency(&mut events, latency_ms);
        annotate_replay_fill_events(
            &mut events,
            ReplayEngineMode::Deterministic,
            "raw_bar_open",
            Some(lifecycle_sequence),
            Some(signal_ts_ns),
            Some(signal_ts_ns),
            Some(exchange_arrival_ts_ns),
            Some(bar.ts_ns),
            bar.ts_ns,
            latency_ms,
        );
        events
    }
}

#[cfg(feature = "replay")]
#[derive(Default)]
struct ReplayLifecycleDispatchQueue {
    virtual_events: ReplayVirtualEventQueue,
    dispatch_events: HashMap<u64, InternalEvent>,
}

#[cfg(feature = "replay")]
impl ReplayLifecycleDispatchQueue {
    fn schedule(
        &mut self,
        market_ts_ns: i64,
        logical_step: u64,
        kind: ReplayVirtualEventKind,
        dispatch_event: Option<InternalEvent>,
    ) -> Result<()> {
        let sequence = self
            .virtual_events
            .schedule_at_step(market_ts_ns, logical_step, kind)?;
        if let Some(event) = dispatch_event {
            self.dispatch_events.insert(sequence, event);
        }
        Ok(())
    }

    fn pop_next_through(
        &mut self,
        market_ts_ns: i64,
        logical_step: u64,
    ) -> Option<(ReplayVirtualEventKind, Option<InternalEvent>)> {
        let event = self
            .virtual_events
            .pop_next_through(market_ts_ns, logical_step, u8::MAX)?;
        let dispatch = self.dispatch_events.remove(&event.sequence);
        Some((event.kind, dispatch))
    }
}

#[cfg(feature = "replay")]
fn replay_bar_open_step(bar_index: u64) -> u64 {
    bar_index.saturating_mul(2).saturating_add(1)
}

#[cfg(feature = "replay")]
fn replay_evaluation_step(evaluation_id: u64) -> u64 {
    evaluation_id.saturating_mul(2).saturating_add(2)
}

#[cfg(feature = "replay")]
fn schedule_pending_replay_commands(
    commands: &mut VecDeque<DeferredReplayCommand>,
    lifecycle: &mut ReplayLifecycleDispatchQueue,
    logical_step: u64,
    fallback_ts_ns: Option<i64>,
) -> Result<Option<i64>> {
    let mut latest_submission_ts_ns = None;
    for command in commands.iter_mut().filter(|command| !command.scheduled) {
        let signal_ts_ns = command.signal_ts_ns.or(fallback_ts_ns).context(
            "deterministic replay command has no signal timestamp or replay-bar fallback",
        )?;
        let arrival_ts_ns = command.exchange_arrival_ts_ns.unwrap_or(signal_ts_ns);
        lifecycle.schedule(
            signal_ts_ns,
            logical_step,
            ReplayVirtualEventKind::OrderSubmitted {
                order_id: command.lifecycle_sequence,
            },
            None,
        )?;
        lifecycle.schedule(
            arrival_ts_ns,
            logical_step,
            ReplayVirtualEventKind::OrderArrivesAtExchange {
                order_id: command.lifecycle_sequence,
            },
            None,
        )?;
        command.signal_ts_ns = Some(signal_ts_ns);
        command.exchange_arrival_ts_ns = Some(arrival_ts_ns);
        command.scheduled = true;
        latest_submission_ts_ns = Some(
            latest_submission_ts_ns.map_or(signal_ts_ns, |latest: i64| latest.max(signal_ts_ns)),
        );
    }
    Ok(latest_submission_ts_ns)
}

#[cfg(feature = "replay")]
fn drain_replay_lifecycle_through(
    commands: &mut VecDeque<DeferredReplayCommand>,
    lifecycle: &mut ReplayLifecycleDispatchQueue,
    market_ts_ns: i64,
    logical_step: u64,
    internal_tx: &UnboundedSender<InternalEvent>,
) {
    while let Some((kind, dispatch)) = lifecycle.pop_next_through(market_ts_ns, logical_step) {
        if let ReplayVirtualEventKind::OrderArrivesAtExchange { order_id } = kind {
            if let Some(command) = commands
                .iter_mut()
                .find(|command| command.lifecycle_sequence == order_id)
            {
                command.arrived = true;
            }
        }
        if let Some(event) = dispatch {
            let _ = internal_tx.send(event);
        }
    }
}

#[cfg(feature = "replay")]
fn replay_dispatch_kind(event: &InternalEvent, lifecycle_sequence: u64) -> ReplayVirtualEventKind {
    match event {
        InternalEvent::BrokerOrderAck(_)
        | InternalEvent::BrokerOrderFailed(_)
        | InternalEvent::OrderStrategyAck(_)
        | InternalEvent::OrderStrategyFailed(_) => ReplayVirtualEventKind::OrderAck {
            order_id: lifecycle_sequence,
        },
        InternalEvent::UserEntities(entities)
            if entities.iter().any(|entity| {
                !entity.deleted && entity.entity_type.eq_ignore_ascii_case("fill")
            }) =>
        {
            ReplayVirtualEventKind::Fill {
                fill_id: lifecycle_sequence,
            }
        }
        _ => ReplayVirtualEventKind::ProtectionUpdate {
            order_id: lifecycle_sequence,
        },
    }
}

#[cfg(feature = "replay")]
fn enqueue_deferred_replay_command(
    queue: &mut VecDeque<DeferredReplayCommand>,
    next_lifecycle_sequence: &mut u64,
    fixed_latency_ms: u64,
    kind: DeferredReplayCommandKind,
) {
    let sequence = *next_lifecycle_sequence;
    *next_lifecycle_sequence = next_lifecycle_sequence.saturating_add(1);
    queue.push_back(DeferredReplayCommand::new(sequence, fixed_latency_ms, kind));
}

#[cfg(feature = "replay")]
fn reprice_strategy_for_bar_open(strategy: &mut PendingOrderStrategyTransition, bar: &Bar) {
    let price_delta = strategy
        .reference_price
        .map(|reference_price| bar.open - reference_price)
        .unwrap_or_default();
    strategy.take_profit_price = strategy.take_profit_price.map(|price| price + price_delta);
    strategy.stop_price = strategy.stop_price.map(|price| price + price_delta);
    strategy.reference_ts_ns = Some(bar.ts_ns);
    strategy.reference_price = Some(bar.open);
}

#[cfg(feature = "replay")]
fn apply_replay_latency(events: &mut [InternalEvent], latency_ms: u64) {
    for event in events {
        match event {
            InternalEvent::BrokerOrderAck(ack) => ack.submit_rtt_ms = latency_ms,
            InternalEvent::OrderStrategyAck(ack) => ack.submit_rtt_ms = latency_ms,
            _ => {}
        }
    }
}

#[cfg(feature = "replay")]
#[allow(clippy::too_many_arguments)]
fn annotate_replay_fill_events(
    events: &mut [InternalEvent],
    engine_mode: ReplayEngineMode,
    fill_source: &str,
    lifecycle_sequence: Option<u64>,
    signal_ts_ns: Option<i64>,
    submission_ts_ns: Option<i64>,
    exchange_arrival_ts_ns: Option<i64>,
    acknowledgement_ts_ns: Option<i64>,
    fill_ts_ns: i64,
    latency_ms: u64,
) {
    for event in events {
        let InternalEvent::UserEntities(entities) = event else {
            continue;
        };
        for envelope in entities {
            if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                continue;
            }
            let Some(fill) = envelope.entity.as_object_mut() else {
                continue;
            };
            fill.insert(
                "replayEngineMode".to_string(),
                json!(match engine_mode {
                    ReplayEngineMode::Legacy => "legacy",
                    ReplayEngineMode::Deterministic => "deterministic",
                }),
            );
            fill.insert("replayFillSource".to_string(), json!(fill_source));
            fill.insert("replayFillTimestampNs".to_string(), json!(fill_ts_ns));
            fill.insert("replayLatencyMs".to_string(), json!(latency_ms));
            if let Some(sequence) = lifecycle_sequence {
                fill.insert("replayLifecycleSequence".to_string(), json!(sequence));
            }
            if let Some(timestamp) = signal_ts_ns {
                fill.insert("replaySignalTimestampNs".to_string(), json!(timestamp));
            }
            if let Some(timestamp) = submission_ts_ns {
                fill.insert("replaySubmissionTimestampNs".to_string(), json!(timestamp));
            }
            if let Some(timestamp) = exchange_arrival_ts_ns {
                fill.insert(
                    "replayExchangeArrivalTimestampNs".to_string(),
                    json!(timestamp),
                );
            }
            if let Some(timestamp) = acknowledgement_ts_ns {
                fill.insert(
                    "replayAcknowledgementTimestampNs".to_string(),
                    json!(timestamp),
                );
            }
        }
    }
}

pub(crate) fn spawn_broker_gateway_task(
    request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(broker_gateway_worker(request_rx, internal_tx))
}

async fn broker_gateway_worker(
    mut request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    let mut replay_state = ReplayBrokerState::default();
    #[cfg(feature = "replay")]
    let mut replay_engine_mode = ReplayEngineMode::Legacy;
    #[cfg(feature = "replay")]
    let mut replay_latency_sampler = ReplayLatencySampler::default();
    #[cfg(feature = "replay")]
    let mut replay_bar_protection_policy = ReplayBarProtectionPolicy::NearestOpen;
    #[cfg(feature = "replay")]
    let mut next_replay_lifecycle_sequence = 1_u64;
    #[cfg(feature = "replay")]
    let mut deferred_replay_commands = VecDeque::<DeferredReplayCommand>::new();
    #[cfg(feature = "replay")]
    let mut replay_lifecycle = ReplayLifecycleDispatchQueue::default();
    while let Some(command) = request_rx.recv().await {
        match command {
            BrokerCommand::MarketOrder { request_tx, order } => {
                if order.simulate {
                    #[cfg(feature = "replay")]
                    if replay_engine_mode == ReplayEngineMode::Deterministic {
                        let latency_ms = replay_latency_sampler.next_latency_ms();
                        enqueue_deferred_replay_command(
                            &mut deferred_replay_commands,
                            &mut next_replay_lifecycle_sequence,
                            latency_ms,
                            DeferredReplayCommandKind::MarketOrder(order),
                        );
                        continue;
                    }
                    match replay_state.simulate_market_order(order) {
                        Ok(events) => {
                            for event in events {
                                let _ = internal_tx.send(event);
                            }
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
                        }
                    }
                } else {
                    match submit_market_order_via_gateway(&request_tx, order).await {
                        Ok(ack) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderAck(ack));
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
                        }
                    }
                }
            }
            BrokerCommand::LiquidatePosition {
                request_tx,
                liquidation,
            } => {
                if liquidation.simulate {
                    #[cfg(feature = "replay")]
                    if replay_engine_mode == ReplayEngineMode::Deterministic {
                        let latency_ms = replay_latency_sampler.next_latency_ms();
                        enqueue_deferred_replay_command(
                            &mut deferred_replay_commands,
                            &mut next_replay_lifecycle_sequence,
                            latency_ms,
                            DeferredReplayCommandKind::LiquidatePosition(liquidation),
                        );
                        continue;
                    }
                    match replay_state.simulate_liquidation(liquidation) {
                        Ok(events) => {
                            for event in events {
                                let _ = internal_tx.send(event);
                            }
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
                        }
                    }
                } else {
                    match submit_liquidation_via_gateway(&request_tx, liquidation).await {
                        Ok(ack) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderAck(ack));
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
                        }
                    }
                }
            }
            BrokerCommand::OrderStrategy {
                request_tx,
                strategy,
            } => {
                if strategy.simulate {
                    #[cfg(feature = "replay")]
                    if replay_engine_mode == ReplayEngineMode::Deterministic {
                        let latency_ms = replay_latency_sampler.next_latency_ms();
                        enqueue_deferred_replay_command(
                            &mut deferred_replay_commands,
                            &mut next_replay_lifecycle_sequence,
                            latency_ms,
                            DeferredReplayCommandKind::OrderStrategy(strategy),
                        );
                        continue;
                    }
                    match replay_state.simulate_order_strategy(strategy) {
                        Ok(events) => {
                            for event in events {
                                let _ = internal_tx.send(event);
                            }
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyFailed(failure));
                        }
                    }
                } else {
                    match submit_order_strategy_via_gateway(&request_tx, strategy).await {
                        Ok(ack) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyAck(ack));
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyFailed(failure));
                        }
                    }
                }
            }
            BrokerCommand::LiquidateThenOrderStrategy {
                request_tx,
                liquidation,
                strategy,
            } => {
                if liquidation.simulate {
                    #[cfg(feature = "replay")]
                    if replay_engine_mode == ReplayEngineMode::Deterministic {
                        let latency_ms = replay_latency_sampler.next_latency_ms();
                        enqueue_deferred_replay_command(
                            &mut deferred_replay_commands,
                            &mut next_replay_lifecycle_sequence,
                            latency_ms,
                            DeferredReplayCommandKind::LiquidateThenOrderStrategy(
                                liquidation,
                                strategy,
                            ),
                        );
                        continue;
                    }
                    match replay_state
                        .simulate_liquidation_then_order_strategy(liquidation, strategy)
                    {
                        Ok(events) => {
                            for event in events {
                                let _ = internal_tx.send(event);
                            }
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyFailed(failure));
                        }
                    }
                } else {
                    match submit_liquidation_then_order_strategy_via_gateway(
                        &request_tx,
                        liquidation,
                        strategy,
                    )
                    .await
                    {
                        Ok(ack) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyAck(ack));
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::OrderStrategyFailed(failure));
                        }
                    }
                }
            }
            BrokerCommand::NativeProtection { request_tx, sync } => {
                if sync.simulate {
                    match replay_state.simulate_native_protection(sync) {
                        Ok(events) => {
                            for event in events {
                                let _ = internal_tx.send(event);
                            }
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::ProtectionSyncFailed(failure));
                        }
                    }
                } else {
                    match submit_native_protection_via_gateway(&request_tx, sync).await {
                        Ok(ack) => {
                            let _ = internal_tx.send(InternalEvent::ProtectionSyncApplied(ack));
                        }
                        Err(failure) => {
                            let _ = internal_tx.send(InternalEvent::ProtectionSyncFailed(failure));
                        }
                    }
                }
            }
            #[cfg(feature = "replay")]
            BrokerCommand::ReplayBar {
                bar,
                bar_index,
                response_tx,
            } => {
                let bar_step = replay_bar_open_step(bar_index);
                if deferred_replay_commands
                    .iter()
                    .any(|command| !command.scheduled)
                {
                    let submission_step = bar_step.saturating_sub(1);
                    if let Err(error) = schedule_pending_replay_commands(
                        &mut deferred_replay_commands,
                        &mut replay_lifecycle,
                        submission_step,
                        Some(bar.ts_ns),
                    ) {
                        let _ = internal_tx.send(InternalEvent::Error(format!(
                            "deterministic replay lifecycle: {error}"
                        )));
                    }
                }
                if let Err(error) = replay_lifecycle.schedule(
                    bar.ts_ns,
                    bar_step,
                    ReplayVirtualEventKind::RawBarOpen {
                        bar_sequence: bar_step,
                    },
                    None,
                ) {
                    let _ = internal_tx.send(InternalEvent::Error(format!(
                        "deterministic replay lifecycle: {error}"
                    )));
                }
                drain_replay_lifecycle_through(
                    &mut deferred_replay_commands,
                    &mut replay_lifecycle,
                    bar.ts_ns,
                    bar_step,
                    &internal_tx,
                );

                let mut waiting_commands = VecDeque::new();
                while let Some(command) = deferred_replay_commands.pop_front() {
                    if !command.arrived {
                        waiting_commands.push_back(command);
                        continue;
                    }
                    let lifecycle_sequence = command.lifecycle_sequence;
                    for event in command.execute(&mut replay_state, &bar) {
                        let kind = replay_dispatch_kind(&event, lifecycle_sequence);
                        if let Err(error) =
                            replay_lifecycle.schedule(bar.ts_ns, bar_step, kind, Some(event))
                        {
                            let _ = internal_tx.send(InternalEvent::Error(format!(
                                "deterministic replay lifecycle: {error}"
                            )));
                        }
                    }
                }
                deferred_replay_commands = waiting_commands;

                let effective_protection_policy = match replay_engine_mode {
                    ReplayEngineMode::Legacy => ReplayBarProtectionPolicy::NearestOpen,
                    ReplayEngineMode::Deterministic => replay_bar_protection_policy,
                };
                let mut protection_events =
                    replay_state.simulate_replay_bar(&bar, effective_protection_policy);
                let protection_sequence = (!protection_events.is_empty()).then(|| {
                    let sequence = next_replay_lifecycle_sequence;
                    next_replay_lifecycle_sequence =
                        next_replay_lifecycle_sequence.saturating_add(1);
                    sequence
                });
                annotate_replay_fill_events(
                    &mut protection_events,
                    replay_engine_mode,
                    "raw_bar_ohlc",
                    protection_sequence,
                    None,
                    None,
                    None,
                    None,
                    bar.ts_ns,
                    0,
                );
                for event in protection_events {
                    let lifecycle_sequence = protection_sequence.unwrap_or_default();
                    let is_fill = matches!(
                        replay_dispatch_kind(&event, lifecycle_sequence),
                        ReplayVirtualEventKind::Fill { .. }
                    );
                    let kind = replay_dispatch_kind(&event, lifecycle_sequence);
                    if let Err(error) =
                        replay_lifecycle.schedule(bar.ts_ns, bar_step, kind, Some(event))
                    {
                        let _ = internal_tx.send(InternalEvent::Error(format!(
                            "deterministic replay lifecycle: {error}"
                        )));
                    }
                    if is_fill {
                        if let Err(error) = replay_lifecycle.schedule(
                            bar.ts_ns,
                            bar_step,
                            ReplayVirtualEventKind::ProtectionUpdate {
                                order_id: lifecycle_sequence,
                            },
                            None,
                        ) {
                            let _ = internal_tx.send(InternalEvent::Error(format!(
                                "deterministic replay lifecycle: {error}"
                            )));
                        }
                    }
                }
                drain_replay_lifecycle_through(
                    &mut deferred_replay_commands,
                    &mut replay_lifecycle,
                    bar.ts_ns,
                    bar_step,
                    &internal_tx,
                );
                let _ = internal_tx.send(InternalEvent::ReplayBarrier(response_tx));
            }
            #[cfg(feature = "replay")]
            BrokerCommand::ConfigureReplay {
                mode,
                latency,
                bar_protection_policy,
                response_tx,
            } => {
                replay_state = ReplayBrokerState::default();
                deferred_replay_commands.clear();
                replay_lifecycle = ReplayLifecycleDispatchQueue::default();
                next_replay_lifecycle_sequence = 1;
                replay_engine_mode = mode;
                replay_latency_sampler.reset(latency);
                replay_bar_protection_policy = bar_protection_policy;
                let _ = response_tx.send(());
            }
            #[cfg(feature = "replay")]
            BrokerCommand::ReplayDrain {
                market_ts_ns,
                evaluation_id,
                response_tx,
            } => {
                let coordinator_step = evaluation_id.map(replay_evaluation_step);
                if let (Some(market_ts_ns), Some(evaluation_id), Some(coordinator_step)) =
                    (market_ts_ns, evaluation_id, coordinator_step)
                {
                    for kind in [
                        ReplayVirtualEventKind::BarClose {
                            bar_index: evaluation_id as usize,
                        },
                        ReplayVirtualEventKind::StrategyEvaluation { evaluation_id },
                    ] {
                        if let Err(error) =
                            replay_lifecycle.schedule(market_ts_ns, coordinator_step, kind, None)
                        {
                            let _ = internal_tx.send(InternalEvent::Error(format!(
                                "deterministic replay coordinator: {error}"
                            )));
                        }
                    }
                }
                if deferred_replay_commands
                    .iter()
                    .any(|command| !command.scheduled)
                {
                    let submission_step = coordinator_step.unwrap_or_default();
                    match schedule_pending_replay_commands(
                        &mut deferred_replay_commands,
                        &mut replay_lifecycle,
                        submission_step,
                        None,
                    ) {
                        Ok(Some(submission_ts_ns)) => drain_replay_lifecycle_through(
                            &mut deferred_replay_commands,
                            &mut replay_lifecycle,
                            submission_ts_ns,
                            submission_step,
                            &internal_tx,
                        ),
                        Ok(None) => {}
                        Err(error) => {
                            let _ = internal_tx.send(InternalEvent::Error(format!(
                                "deterministic replay lifecycle: {error}"
                            )));
                        }
                    }
                }
                if let (Some(market_ts_ns), Some(coordinator_step)) =
                    (market_ts_ns, coordinator_step)
                {
                    drain_replay_lifecycle_through(
                        &mut deferred_replay_commands,
                        &mut replay_lifecycle,
                        market_ts_ns,
                        coordinator_step,
                        &internal_tx,
                    );
                }
                let _ = internal_tx.send(InternalEvent::ReplayBarrier(response_tx));
            }
        }
    }
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;

    fn replay_bar(ts_ns: i64, open: f64) -> Bar {
        Bar {
            ts_ns,
            open,
            high: open + 1.0,
            low: open - 1.0,
            close: open + 0.25,
            volume: Some(10.0),
        }
    }

    fn market_order(reference_ts_ns: i64, reference_price: f64) -> PendingMarketOrder {
        PendingMarketOrder {
            simulate: true,
            cl_ord_id: "midas-replay-entry-1".to_string(),
            payload: json!({}),
            account_id: 7,
            contract_id: 11,
            interrupt_order_strategy_id: None,
            cancel_order_ids: Vec::new(),
            action_label: "Strategy".to_string(),
            order_action: "Buy".to_string(),
            order_qty: 1,
            contract_name: "MESU6".to_string(),
            account_name: "SIM".to_string(),
            reference_ts_ns: Some(reference_ts_ns),
            reference_price: Some(reference_price),
            simulated_next_qty: 1,
            reason_suffix: None,
            target_qty: Some(1),
        }
    }

    fn order_strategy(
        reference_ts_ns: i64,
        reference_price: f64,
    ) -> PendingOrderStrategyTransition {
        PendingOrderStrategyTransition {
            simulate: true,
            uuid: "midas-replay-strategy-1".to_string(),
            payload: json!({}),
            interrupt_order_strategy_id: None,
            cancel_order_ids: Vec::new(),
            order_action: "Buy".to_string(),
            entry_order_qty: 1,
            target_qty: 1,
            contract_name: "MESU6".to_string(),
            account_name: "SIM".to_string(),
            reference_ts_ns: Some(reference_ts_ns),
            reference_price: Some(reference_price),
            take_profit_price: Some(reference_price + 10.0),
            stop_price: Some(reference_price - 10.0),
            replay_auto_trail: None,
            reason_suffix: None,
            key: StrategyProtectionKey {
                account_id: 7,
                contract_id: 11,
            },
        }
    }

    async fn configure(
        broker_tx: &UnboundedSender<BrokerCommand>,
        mode: ReplayEngineMode,
        fixed_latency_ms: u64,
    ) {
        configure_with(
            broker_tx,
            mode,
            ReplayLatencyConfig {
                fixed_latency_ms,
                ..ReplayLatencyConfig::default()
            },
            ReplayBarProtectionPolicy::Conservative,
        )
        .await;
    }

    async fn configure_with(
        broker_tx: &UnboundedSender<BrokerCommand>,
        mode: ReplayEngineMode,
        latency: ReplayLatencyConfig,
        bar_protection_policy: ReplayBarProtectionPolicy,
    ) {
        let (response_tx, response_rx) = oneshot::channel();
        broker_tx
            .send(BrokerCommand::ConfigureReplay {
                mode,
                latency,
                bar_protection_policy,
                response_tx,
            })
            .unwrap();
        response_rx.await.unwrap();
    }

    async fn process_bar(
        broker_tx: &UnboundedSender<BrokerCommand>,
        internal_rx: &mut UnboundedReceiver<InternalEvent>,
        bar: Bar,
        bar_index: u64,
    ) -> Vec<InternalEvent> {
        let (response_tx, response_rx) = oneshot::channel();
        broker_tx
            .send(BrokerCommand::ReplayBar {
                bar,
                bar_index,
                response_tx,
            })
            .unwrap();

        let mut events = Vec::new();
        loop {
            match internal_rx.recv().await.expect("gateway event") {
                InternalEvent::ReplayBarrier(response_tx) => {
                    let _ = response_tx.send(());
                    break;
                }
                event => events.push(event),
            }
        }
        response_rx.await.unwrap();
        events
    }

    async fn drain_broker(
        broker_tx: &UnboundedSender<BrokerCommand>,
        internal_rx: &mut UnboundedReceiver<InternalEvent>,
        market_ts_ns: Option<i64>,
        evaluation_id: Option<u64>,
    ) -> Vec<InternalEvent> {
        let (response_tx, response_rx) = oneshot::channel();
        broker_tx
            .send(BrokerCommand::ReplayDrain {
                market_ts_ns,
                evaluation_id,
                response_tx,
            })
            .unwrap();

        let mut events = Vec::new();
        loop {
            match internal_rx.recv().await.expect("gateway event") {
                InternalEvent::ReplayBarrier(response_tx) => {
                    let _ = response_tx.send(());
                    break;
                }
                event => events.push(event),
            }
        }
        response_rx.await.unwrap();
        events
    }

    fn fill_from_events(events: &[InternalEvent]) -> Option<&Value> {
        events.iter().find_map(|event| match event {
            InternalEvent::UserEntities(entities) => entities
                .iter()
                .find(|entity| entity.entity_type == "fill")
                .map(|entity| &entity.entity),
            _ => None,
        })
    }

    fn fills_from_events(events: &[InternalEvent]) -> Vec<&Value> {
        events
            .iter()
            .filter_map(|event| match event {
                InternalEvent::UserEntities(entities) => Some(entities),
                _ => None,
            })
            .flatten()
            .filter(|entity| entity.entity_type.eq_ignore_ascii_case("fill"))
            .map(|entity| &entity.entity)
            .collect()
    }

    #[test]
    fn observed_and_seeded_latency_models_are_reproducible() {
        assert_eq!(observed_mean(&[10, 20, 40]), 23);
        assert_eq!(observed_percentile(&(1..=100).collect::<Vec<_>>(), 95), 95);
        assert_eq!(observed_percentile(&(1..=100).collect::<Vec<_>>(), 99), 99);

        let config = ReplayLatencyConfig {
            model: ReplayLatencyModel::SeededObserved,
            observed_samples_ms: vec![15, 30, 60, 120],
            seed: 42,
            ..ReplayLatencyConfig::default()
        };
        let mut first = ReplayLatencySampler::default();
        first.reset(config.clone());
        let mut second = ReplayLatencySampler::default();
        second.reset(config);
        let first_trace = (0..12).map(|_| first.next_latency_ms()).collect::<Vec<_>>();
        let second_trace = (0..12)
            .map(|_| second.next_latency_ms())
            .collect::<Vec<_>>();
        assert_eq!(first_trace, second_trace);
        assert!(
            first_trace
                .iter()
                .all(|sample| [15, 30, 60, 120].contains(sample))
        );
    }

    #[tokio::test]
    async fn deterministic_market_order_fills_at_first_bar_open_after_fixed_latency() {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure(&broker_tx, ReplayEngineMode::Deterministic, 50).await;

        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::MarketOrder {
                request_tx,
                order: market_order(1_000_000_000, 10.0),
            })
            .unwrap();

        let before_arrival = process_bar(
            &broker_tx,
            &mut internal_rx,
            replay_bar(1_049_999_999, 11.0),
            0,
        )
        .await;
        assert!(fill_from_events(&before_arrival).is_none());

        let at_arrival = process_bar(
            &broker_tx,
            &mut internal_rx,
            replay_bar(1_050_000_000, 12.5),
            1,
        )
        .await;
        let fill = fill_from_events(&at_arrival).expect("next-open fill");
        assert_eq!(fill.get("price").and_then(Value::as_f64), Some(12.5));
        assert_eq!(
            fill.get("timestamp").and_then(Value::as_i64),
            Some(1_050_000_000)
        );
        assert_eq!(
            fill.get("replayFillSource").and_then(Value::as_str),
            Some("raw_bar_open")
        );
        assert_eq!(
            fill.get("replaySignalTimestampNs").and_then(Value::as_i64),
            Some(1_000_000_000)
        );
        assert_eq!(
            fill.get("replayExchangeArrivalTimestampNs")
                .and_then(Value::as_i64),
            Some(1_050_000_000)
        );
        assert_eq!(
            fill.get("replayAcknowledgementTimestampNs")
                .and_then(Value::as_i64),
            Some(1_050_000_000)
        );
        assert_eq!(
            fill.get("replayLifecycleSequence").and_then(Value::as_u64),
            Some(1)
        );
        assert!(at_arrival.iter().any(|event| matches!(
            event,
            InternalEvent::BrokerOrderAck(BrokerOrderAck {
                submit_rtt_ms: 50,
                ..
            })
        )));

        drop(broker_tx);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn legacy_market_order_keeps_immediate_reference_fill() {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure(&broker_tx, ReplayEngineMode::Legacy, 250).await;

        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::MarketOrder {
                request_tx,
                order: market_order(2_000_000_000, 10.25),
            })
            .unwrap();

        let first = internal_rx.recv().await.unwrap();
        let second = internal_rx.recv().await.unwrap();
        let events = vec![first, second];
        let fill = fill_from_events(&events).expect("legacy fill");
        assert_eq!(fill.get("price").and_then(Value::as_f64), Some(10.25));
        assert_eq!(
            fill.get("timestamp").and_then(Value::as_i64),
            Some(2_000_000_000)
        );
        assert_eq!(
            fill.get("replayFillSource").and_then(Value::as_str),
            Some("legacy_reference_price")
        );
        assert!(events.iter().any(|event| matches!(
            event,
            InternalEvent::BrokerOrderAck(BrokerOrderAck {
                submit_rtt_ms: 0,
                ..
            })
        )));

        drop(broker_tx);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn deterministic_order_strategy_uses_raw_open_and_rebased_brackets() {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure(&broker_tx, ReplayEngineMode::Deterministic, 0).await;

        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::OrderStrategy {
                request_tx,
                strategy: order_strategy(3_000_000_000, 100.0),
            })
            .unwrap();

        let events = process_bar(
            &broker_tx,
            &mut internal_rx,
            replay_bar(4_000_000_000, 105.0),
            0,
        )
        .await;
        let fill = fill_from_events(&events).expect("strategy entry fill");
        assert_eq!(fill.get("price").and_then(Value::as_f64), Some(105.0));
        assert_eq!(
            fill.get("timestamp").and_then(Value::as_i64),
            Some(4_000_000_000)
        );

        let orders = events
            .iter()
            .filter_map(|event| match event {
                InternalEvent::UserEntities(entities) => Some(entities),
                _ => None,
            })
            .flatten()
            .filter(|entity| entity.entity_type == "order")
            .map(|entity| &entity.entity)
            .collect::<Vec<_>>();
        let take_profit = orders
            .iter()
            .find(|order| order.get("orderType").and_then(Value::as_str) == Some("Limit"))
            .expect("take-profit order");
        let stop = orders
            .iter()
            .find(|order| order.get("orderType").and_then(Value::as_str) == Some("Stop"))
            .expect("stop order");
        assert_eq!(
            take_profit.get("price").and_then(Value::as_f64),
            Some(115.0)
        );
        assert_eq!(stop.get("stopPrice").and_then(Value::as_f64), Some(95.0));

        let protection_events = process_bar(
            &broker_tx,
            &mut internal_rx,
            replay_bar(5_000_000_000, 114.5),
            1,
        )
        .await;
        let protection_fill =
            fill_from_events(&protection_events).expect("take-profit fill from raw OHLC");
        assert_eq!(
            protection_fill
                .get("replayFillSource")
                .and_then(Value::as_str),
            Some("raw_bar_ohlc")
        );
        assert_eq!(
            protection_fill
                .get("replayExitReason")
                .and_then(Value::as_str),
            Some("take_profit")
        );
        assert_eq!(
            protection_fill
                .get("replayProtectionOrderId")
                .and_then(Value::as_i64),
            protection_fill.get("orderId").and_then(Value::as_i64)
        );

        drop(broker_tx);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn deterministic_lifecycle_handles_signal_and_next_bar_at_same_timestamp() {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure(&broker_tx, ReplayEngineMode::Deterministic, 0).await;

        let first_bar_events =
            process_bar(&broker_tx, &mut internal_rx, replay_bar(10, 100.0), 0).await;
        assert!(first_bar_events.is_empty());

        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::MarketOrder {
                request_tx,
                order: market_order(10, 100.25),
            })
            .unwrap();
        let submission_events = drain_broker(&broker_tx, &mut internal_rx, Some(10), Some(0)).await;
        assert!(submission_events.is_empty());

        let duplicate_bar_events =
            process_bar(&broker_tx, &mut internal_rx, replay_bar(10, 101.5), 1).await;
        assert!(
            !duplicate_bar_events
                .iter()
                .any(|event| matches!(event, InternalEvent::Error(_)))
        );
        let fill = fill_from_events(&duplicate_bar_events).expect("duplicate-timestamp fill");
        assert_eq!(fill.get("price").and_then(Value::as_f64), Some(101.5));
        assert_eq!(
            fill.get("replayLifecycleSequence").and_then(Value::as_u64),
            Some(1)
        );

        drop(broker_tx);
        task.await.unwrap();
    }

    async fn ambiguous_bar_exit_reason(policy: ReplayBarProtectionPolicy) -> String {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure_with(
            &broker_tx,
            ReplayEngineMode::Deterministic,
            ReplayLatencyConfig::default(),
            policy,
        )
        .await;

        let mut strategy = order_strategy(1, 100.0);
        strategy.take_profit_price = Some(101.0);
        strategy.stop_price = Some(99.0);
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::OrderStrategy {
                request_tx,
                strategy,
            })
            .unwrap();
        let events = process_bar(&broker_tx, &mut internal_rx, replay_bar(2, 100.0), 0).await;
        let reason = fills_from_events(&events)
            .into_iter()
            .find_map(|fill| fill.get("replayExitReason").and_then(Value::as_str))
            .expect("protection exit")
            .to_string();
        let ambiguous = fills_from_events(&events)
            .into_iter()
            .find(|fill| fill.get("replayExitReason").is_some())
            .and_then(|fill| fill.get("replayAmbiguousBar"))
            .and_then(Value::as_bool);
        assert_eq!(ambiguous, Some(true));

        drop(broker_tx);
        task.await.unwrap();
        reason
    }

    #[tokio::test]
    async fn bar_protection_policy_resolves_both_reachable_brackets_explicitly() {
        assert_eq!(
            ambiguous_bar_exit_reason(ReplayBarProtectionPolicy::Conservative).await,
            "stop_loss"
        );
        assert_eq!(
            ambiguous_bar_exit_reason(ReplayBarProtectionPolicy::Optimistic).await,
            "take_profit"
        );
    }

    #[tokio::test]
    async fn bar_trailing_stop_tightens_after_close_and_fills_on_next_bar() {
        let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = spawn_broker_gateway_task(broker_rx, internal_tx);
        configure(&broker_tx, ReplayEngineMode::Deterministic, 0).await;

        let mut strategy = order_strategy(1, 100.0);
        strategy.take_profit_price = None;
        strategy.stop_price = Some(95.0);
        strategy.replay_auto_trail = Some(ReplayAutoTrail {
            trigger_offset: 2.0,
            stop_offset: 1.0,
            frequency: 1.0,
        });
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        broker_tx
            .send(BrokerCommand::OrderStrategy {
                request_tx,
                strategy,
            })
            .unwrap();
        let entry = process_bar(&broker_tx, &mut internal_rx, replay_bar(2, 100.0), 0).await;
        assert!(
            fills_from_events(&entry)
                .iter()
                .all(|fill| fill.get("replayExitReason").is_none())
        );

        let tightened = process_bar(&broker_tx, &mut internal_rx, replay_bar(3, 102.0), 1).await;
        let trailing_order = tightened
            .iter()
            .filter_map(|event| match event {
                InternalEvent::UserEntities(entities) => Some(entities),
                _ => None,
            })
            .flatten()
            .find(|entity| {
                entity.entity_type == "order"
                    && entity
                        .entity
                        .get("replayTrailingActive")
                        .and_then(Value::as_bool)
                        == Some(true)
            })
            .expect("trailing update");
        assert_eq!(
            trailing_order
                .entity
                .get("stopPrice")
                .and_then(Value::as_f64),
            Some(102.0)
        );
        assert!(fills_from_events(&tightened).is_empty());

        let exit = process_bar(&broker_tx, &mut internal_rx, replay_bar(4, 103.0), 2).await;
        let trailing_fill = fills_from_events(&exit)
            .into_iter()
            .find(|fill| fill.get("replayExitReason").is_some())
            .expect("trailing fill");
        assert_eq!(
            trailing_fill
                .get("replayExitReason")
                .and_then(Value::as_str),
            Some("trailing_stop")
        );

        drop(broker_tx);
        task.await.unwrap();
    }

    #[test]
    fn deterministic_strategy_offsets_are_rebased_to_raw_entry_open() {
        let mut strategy = PendingOrderStrategyTransition {
            simulate: true,
            uuid: "strategy-1".to_string(),
            payload: json!({}),
            interrupt_order_strategy_id: None,
            cancel_order_ids: Vec::new(),
            order_action: "Buy".to_string(),
            entry_order_qty: 1,
            target_qty: 1,
            contract_name: "MESU6".to_string(),
            account_name: "SIM".to_string(),
            reference_ts_ns: Some(1),
            reference_price: Some(100.0),
            take_profit_price: Some(102.0),
            stop_price: Some(99.0),
            replay_auto_trail: None,
            reason_suffix: None,
            key: StrategyProtectionKey {
                account_id: 7,
                contract_id: 11,
            },
        };

        reprice_strategy_for_bar_open(&mut strategy, &replay_bar(2, 105.0));

        assert_eq!(strategy.reference_ts_ns, Some(2));
        assert_eq!(strategy.reference_price, Some(105.0));
        assert_eq!(strategy.take_profit_price, Some(107.0));
        assert_eq!(strategy.stop_price, Some(104.0));
    }
}
