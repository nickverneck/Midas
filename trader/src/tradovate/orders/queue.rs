use super::*;

pub(super) fn enqueue_market_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    order: PendingMarketOrder,
) -> Result<()> {
    let pending_signal = session.pending_signal_context.take();
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: pending_signal.as_ref().map(|signal| signal.started_at),
        signal_context: pending_signal.map(|signal| signal.description),
        cl_ord_id: order.cl_ord_id.clone(),
        strategy_owned_protection: false,
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::MarketOrder { request_tx, order })
        .is_err()
    {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}

pub(super) fn enqueue_liquidation(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    liquidation: PendingLiquidation,
) -> Result<()> {
    session.order_submit_in_flight = true;
    session.order_latency_tracker = None;
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::LiquidatePosition {
            request_tx,
            liquidation,
        })
        .is_err()
    {
        session.order_submit_in_flight = false;
        bail!("broker gateway is closed");
    }
    Ok(())
}

pub(super) fn enqueue_liquidation_then_order_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    liquidation: PendingLiquidation,
    strategy: PendingOrderStrategyTransition,
) -> Result<()> {
    let pending_signal = session.pending_signal_context.take();
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: pending_signal.as_ref().map(|signal| signal.started_at),
        signal_context: pending_signal.map(|signal| signal.description),
        cl_ord_id: strategy.uuid.clone(),
        strategy_owned_protection: true,
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::LiquidateThenOrderStrategy {
            request_tx,
            liquidation,
            strategy,
        })
        .is_err()
    {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}

pub(super) fn enqueue_order_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    strategy: PendingOrderStrategyTransition,
) -> Result<()> {
    let pending_signal = session.pending_signal_context.take();
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        signal_started_at: pending_signal.as_ref().map(|signal| signal.started_at),
        signal_context: pending_signal.map(|signal| signal.description),
        cl_ord_id: strategy.uuid.clone(),
        strategy_owned_protection: true,
        order_id: None,
        order_strategy_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx
        .send(BrokerCommand::OrderStrategy {
            request_tx,
            strategy,
        })
        .is_err()
    {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}
