use super::state::{AccountRefresh, IronbeamSession};
use super::support::{
    empty_as_none, ironbeam_duration_code, ironbeam_rest_url, pick_str, request_json, stable_id,
};
use crate::broker::{AccountInfo, ContractSuggestion, LatencySnapshot};
use crate::config::{AppConfig, AuthMode};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IronbeamTokenCache {
    token: String,
    #[serde(rename = "accessToken", skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct AuthResult {
    pub(super) token: String,
    pub(super) user_name: Option<String>,
}

pub(super) async fn authenticate(
    client: &Client,
    cfg: &AppConfig,
    latency: &mut LatencySnapshot,
) -> Result<AuthResult> {
    if let Some(token) = empty_as_none(&cfg.token_override) {
        latency.rest_rtt_ms = Some(0);
        return Ok(AuthResult {
            token: token.to_string(),
            user_name: empty_as_none(&cfg.username).map(ToString::to_string),
        });
    }

    match cfg.auth_mode {
        AuthMode::TokenFile => load_token_file(&cfg.token_path)
            .or_else(|_| load_token_file(&cfg.session_cache_path))
            .with_context(|| {
                format!(
                    "load token from {} or {}",
                    cfg.token_path.display(),
                    cfg.session_cache_path.display()
                )
            }),
        AuthMode::Credentials => request_access_token(client, cfg, latency).await,
    }
}

fn load_token_file(path: &Path) -> Result<AuthResult> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse token JSON {}", path.display()))?;
    let token = pick_str(&parsed, &["token", "accessToken"])
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("token JSON missing token/accessToken")?;
    Ok(AuthResult {
        token,
        user_name: pick_str(&parsed, &["name"]).map(ToString::to_string),
    })
}

pub(super) fn save_token_cache(path: &Path, auth: &AuthResult) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = IronbeamTokenCache {
        token: auth.token.clone(),
        access_token: Some(auth.token.clone()),
        name: auth.user_name.clone(),
    };
    fs::write(path, serde_json::to_string_pretty(&body)?)
        .with_context(|| format!("write token cache {}", path.display()))?;
    Ok(())
}

async fn request_access_token(
    client: &Client,
    cfg: &AppConfig,
    latency: &mut LatencySnapshot,
) -> Result<AuthResult> {
    let payload = json!({
        "username": cfg.username,
        "password": cfg.password,
        "apikey": empty_as_none(&cfg.api_key),
    });
    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!("{}/auth", ironbeam_rest_url(cfg.env)))
            .json(&payload),
        "Ironbeam auth",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    let token = pick_str(&parsed, &["token", "accessToken"])
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("missing token in Ironbeam auth response")?;
    Ok(AuthResult {
        token,
        user_name: empty_as_none(&cfg.username).map(ToString::to_string),
    })
}

pub(super) async fn list_accounts(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    latency: &mut LatencySnapshot,
) -> Result<Vec<AccountInfo>> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllAccounts",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token),
        "Ironbeam account list",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(parsed
        .get("accounts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|account_id| AccountInfo {
            id: stable_id(account_id),
            name: account_id.to_string(),
            raw: json!({ "accountId": account_id }),
        })
        .collect())
}

pub(super) async fn search_contracts(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    query: &str,
    limit: usize,
    latency: &mut LatencySnapshot,
) -> Result<Vec<ContractSuggestion>> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!("{}/info/symbols", ironbeam_rest_url(cfg.env)))
            .bearer_auth(token)
            .query(&[
                ("text", query.to_string()),
                ("limit", limit.min(1000).max(1).to_string()),
                ("preferActive", "true".to_string()),
            ]),
        "Ironbeam symbol search",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(parsed
        .get("symbols")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let symbol = pick_str(item, &["symbol", "exchSym"])?.trim();
            if symbol.is_empty() {
                return None;
            }
            let description =
                pick_str(item, &["description", "symbolType", "contractText"]).unwrap_or(symbol);
            Some(ContractSuggestion {
                id: stable_id(symbol),
                name: symbol.to_string(),
                description: description.to_string(),
                raw: item.clone(),
            })
        })
        .collect())
}

pub(super) async fn fetch_account_refresh(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
) -> Result<AccountRefresh> {
    let (balances, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllBalances",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token)
            .query(&[("balanceType", "CURRENT_OPEN")]),
        "Ironbeam balances",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let (positions, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllPositions",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token),
        "Ironbeam positions",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let (risks, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllRiskInfo",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token),
        "Ironbeam risk info",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let mut orders = BTreeMap::new();
    let mut fills = BTreeMap::new();
    for account in &session.accounts {
        let (parsed_orders, elapsed_ms) = request_json(
            client
                .get(format!(
                    "{}/order/{}/ANY",
                    ironbeam_rest_url(session.cfg.env),
                    account.name
                ))
                .bearer_auth(&session.token),
            &format!("Ironbeam orders for {}", account.name),
        )
        .await?;
        latency.rest_rtt_ms = Some(elapsed_ms);
        orders.insert(
            account.name.clone(),
            parsed_orders
                .get("orders")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        );

        let (parsed_fills, elapsed_ms) = request_json(
            client
                .get(format!(
                    "{}/order/{}/fills",
                    ironbeam_rest_url(session.cfg.env),
                    account.name
                ))
                .bearer_auth(&session.token),
            &format!("Ironbeam fills for {}", account.name),
        )
        .await?;
        latency.rest_rtt_ms = Some(elapsed_ms);
        fills.insert(
            account.name.clone(),
            parsed_fills
                .get("fills")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        );
    }

    Ok(AccountRefresh {
        balances: balances
            .get("balances")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        positions: positions
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        risks: risks
            .get("risks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        orders,
        fills,
    })
}

pub(super) async fn create_stream_id(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    latency: &mut LatencySnapshot,
) -> Result<String> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!("{}/stream/create", ironbeam_rest_url(cfg.env)))
            .bearer_auth(token),
        "Ironbeam stream creation",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    pick_str(&parsed, &["streamId"])
        .map(ToString::to_string)
        .filter(|stream_id| !stream_id.trim().is_empty())
        .context("missing streamId in Ironbeam stream creation response")
}

pub(super) async fn subscribe_time_bars(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    stream_id: &str,
    contract: &ContractSuggestion,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let payload = json!({
        "symbol": super::support::contract_symbol(contract),
        "period": 1,
        "barType": "MINUTE",
        "loadSize": cfg.history_bars.max(1),
    });
    let (_, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/indicator/{stream_id}/timeBars/subscribe",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam time bars subscription",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

pub(super) async fn submit_market_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    symbol: &str,
    side: &str,
    quantity: i32,
    latency: &mut LatencySnapshot,
) -> Result<Option<String>> {
    let payload = json!({
        "accountId": account_name,
        "exchSym": symbol,
        "side": side,
        "quantity": quantity,
        "orderType": "MARKET",
        "duration": ironbeam_duration_code(&cfg.time_in_force),
        "waitForOrderId": true,
    });
    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/order/{account_name}/place",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam market order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(pick_str(&parsed, &["orderId"]).map(ToString::to_string))
}

pub(super) async fn place_protection_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    symbol: &str,
    side: &str,
    quantity: i32,
    order_type: &str,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    latency: &mut LatencySnapshot,
) -> Result<String> {
    let mut payload = Map::new();
    payload.insert("accountId".to_string(), json!(account_name));
    payload.insert("exchSym".to_string(), json!(symbol));
    payload.insert("side".to_string(), json!(side));
    payload.insert("quantity".to_string(), json!(quantity));
    payload.insert("orderType".to_string(), json!(order_type));
    payload.insert(
        "duration".to_string(),
        json!(ironbeam_duration_code(&cfg.time_in_force)),
    );
    payload.insert("waitForOrderId".to_string(), json!(true));
    if let Some(limit_price) = limit_price {
        payload.insert("limitPrice".to_string(), json!(limit_price));
    }
    if let Some(stop_price) = stop_price {
        payload.insert("stopPrice".to_string(), json!(stop_price));
    }

    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/order/{account_name}/place",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&Value::Object(payload)),
        "Ironbeam protection order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    pick_str(&parsed, &["orderId"])
        .map(ToString::to_string)
        .context("missing orderId in Ironbeam protection order response")
}

pub(super) async fn update_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_id: &str,
    quantity: i32,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let mut payload = Map::new();
    payload.insert("orderId".to_string(), json!(order_id));
    payload.insert("quantity".to_string(), json!(quantity));
    if let Some(limit_price) = limit_price {
        payload.insert("limitPrice".to_string(), json!(limit_price));
    }
    if let Some(stop_price) = stop_price {
        payload.insert("stopPrice".to_string(), json!(stop_price));
    }

    let (_, elapsed_ms) = request_json(
        client
            .put(format!(
                "{}/order/{account_name}/update/{order_id}",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&Value::Object(payload)),
        "Ironbeam update order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

pub(super) async fn cancel_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_id: &str,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let (_, elapsed_ms) = request_json(
        client
            .delete(format!(
                "{}/order/{account_name}/cancel/{order_id}",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token),
        "Ironbeam cancel order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

pub(super) async fn cancel_multiple_orders(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_ids: &[String],
    latency: &mut LatencySnapshot,
) -> Result<()> {
    if order_ids.is_empty() {
        return Ok(());
    }
    let payload = json!({
        "accountId": account_name,
        "orderIds": order_ids,
    });
    let (_, elapsed_ms) = request_json(
        client
            .delete(format!(
                "{}/order/{account_name}/cancelMultiple",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam cancel multiple orders",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}
