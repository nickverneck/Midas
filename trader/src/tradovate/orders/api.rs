use super::*;

pub(crate) async fn cancel_orders_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_ids: &[i64],
) -> Result<bool> {
    let mut cancelled = false;
    for order_id in order_ids {
        if cancel_order_by_id(request_tx, *order_id).await? {
            cancelled = true;
        }
    }
    Ok(cancelled)
}

pub(crate) async fn interrupt_order_strategy_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_strategy_id: i64,
) -> Result<()> {
    let payload = json!({
        "orderStrategyId": order_strategy_id,
    });
    let _ =
        request_order_json(request_tx, "orderStrategy/interruptorderstrategy", &payload).await?;
    Ok(())
}

pub(crate) async fn cancel_order_by_id(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order_id: i64,
) -> Result<bool> {
    let payload = json!({
        "orderId": order_id,
        "isAutomated": true,
    });
    match request_order_json(request_tx, "order/cancelorder", &payload).await {
        Ok(_) => Ok(true),
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("TooLate") || msg.contains("Already cancelled") {
                Ok(false)
            } else {
                Err(err)
            }
        }
    }
}

pub(crate) async fn request_order_json(
    request_tx: &UnboundedSender<UserSocketCommand>,
    endpoint: &str,
    payload: &Value,
) -> Result<Value> {
    let (response_tx, response_rx) = oneshot::channel();
    request_tx
        .send(UserSocketCommand {
            endpoint: endpoint.to_string(),
            query: None,
            body: Some(payload.clone()),
            response_tx,
        })
        .map_err(|_| anyhow::anyhow!("user websocket request channel is closed"))?;

    let parsed = response_rx
        .await
        .map_err(|_| anyhow::anyhow!("user websocket response channel was dropped"))?
        .map_err(anyhow::Error::msg)?;
    if let Some(failure) = parsed.get("failureReason").and_then(Value::as_str) {
        if !failure.trim().is_empty() {
            bail!("{endpoint} rejected: {failure}");
        }
    }
    if let Some(err_text) = parsed.get("errorText").and_then(Value::as_str) {
        if !err_text.trim().is_empty() {
            bail!("{endpoint} errorText: {err_text}");
        }
    }
    Ok(parsed)
}
