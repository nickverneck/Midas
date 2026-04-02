use super::*;

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
    while let Some(command) = request_rx.recv().await {
        match command {
            BrokerCommand::MarketOrder { request_tx, order } => {
                if order.simulate {
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
            BrokerCommand::LiquidateThenOrderStrategy {
                request_tx,
                liquidation,
                strategy,
            } => {
                if liquidation.simulate {
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
            BrokerCommand::OrderStrategy {
                request_tx,
                strategy,
            } => {
                if strategy.simulate {
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
            BrokerCommand::ReplayBar { bar, response_tx } => {
                for event in replay_state.simulate_replay_bar(&bar) {
                    let _ = internal_tx.send(event);
                }
                let _ = response_tx.send(());
            }
        }
    }
}
