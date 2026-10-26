async fn authenticate(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    if let Some(token) = empty_as_none(&cfg.token_override) {
        let rest_url = cfg.broker_rest_url();
        let user_name = fetch_auth_me(client, &rest_url, token)
            .await
            .ok()
            .and_then(|value| {
                value
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            });
        return Ok(TokenBundle {
            access_token: token.to_string(),
            md_access_token: token.to_string(),
            expiration_time: None,
            user_id: None,
            user_name,
            api_hosts: load_token_file(&cfg.token_path)
                .ok()
                .and_then(|tokens| tokens.api_hosts),
        });
    }

    match cfg.auth_mode {
        AuthMode::TokenFile => {
            let tokens = load_runtime_token_bundle(cfg)?.tokens;
            let rest_url = cfg.broker_rest_url();
            let user_name = fetch_auth_me(client, &rest_url, &tokens.access_token)
                .await
                .ok()
                .and_then(|value| {
                    value
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .or(tokens.user_name.clone());
            Ok(TokenBundle {
                user_name,
                ..tokens
            })
        }
        AuthMode::Credentials => request_access_token(client, cfg).await,
    }
}

/// Resolve the hosted Market Replay WebSocket only from NinjaTrader's
/// authenticated apiHosts response/token bundle. Never derive it from the
/// selected Live/Demo environment and never fall back to those hosts.
fn ninja_replay_ws_url(tokens: &TokenBundle) -> Result<String> {
    let host = tokens
        .api_hosts
        .as_ref()
        .and_then(|hosts| hosts.replay.as_deref())
        .map(str::trim)
        .filter(|host| {
            !host.is_empty()
                && host.chars().all(|character| {
                    character.is_ascii_alphanumeric() || ".-".contains(character)
                })
                && !host.starts_with('.')
                && !host.ends_with('.')
        })
        .context("NinjaTrader authentication did not provide a valid apiHosts.replay host")?;
    Ok(format!("wss://{host}/v1/websocket"))
}

fn load_token_file(path: &Path) -> Result<TokenBundle> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse token JSON {}", path.display()))?;

    let access_token = parsed
        .get("token")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("accessToken").and_then(Value::as_str))
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("token JSON missing token/accessToken")?;
    let md_access_token = parsed
        .get("mdAccessToken")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| access_token.clone());

    Ok(TokenBundle {
        access_token,
        md_access_token,
        expiration_time: parsed
            .get("expirationTime")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        user_id: json_i64(&parsed, "userId"),
        user_name: parsed
            .get("name")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        api_hosts: parsed
            .get("apiHosts")
            .cloned()
            .and_then(|hosts| serde_json::from_value(hosts).ok()),
    })
}

fn load_token_file_with_snapshot(path: &Path) -> Result<RuntimeTokenBundle> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse token JSON {}", path.display()))?;
    let access_token = parsed
        .get("token")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("accessToken").and_then(Value::as_str))
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("token JSON missing token/accessToken")?;
    let md_access_token = parsed
        .get("mdAccessToken")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| access_token.clone());
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let mut hasher = DefaultHasher::new();
    raw.hash(&mut hasher);

    Ok(RuntimeTokenBundle {
        tokens: TokenBundle {
            access_token,
            md_access_token,
            expiration_time: parsed
                .get("expirationTime")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            user_id: json_i64(&parsed, "userId"),
            user_name: parsed
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            api_hosts: parsed
                .get("apiHosts")
                .cloned()
                .and_then(|hosts| serde_json::from_value(hosts).ok()),
        },
        file_snapshot: Some(TokenFileSnapshot {
            path: path.to_path_buf(),
            modified: metadata.modified().ok(),
            len: metadata.len(),
            content_hash: hasher.finish(),
        }),
    })
}

fn save_token_cache(path: &Path, tokens: &TokenBundle) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }

    let body = TokenCacheFile {
        token: tokens.access_token.clone(),
        access_token: Some(tokens.access_token.clone()),
        md_access_token: Some(tokens.md_access_token.clone()),
        expiration_time: tokens.expiration_time.clone(),
        user_id: tokens.user_id,
        name: tokens.user_name.clone(),
        has_live: None,
        api_hosts: tokens.api_hosts.clone(),
    };
    fs::write(path, serde_json::to_string_pretty(&body)?)
        .with_context(|| format!("write token cache {}", path.display()))?;
    Ok(())
}

async fn request_access_token(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    let rest_url = cfg.broker_rest_url();
    let url = format!("{rest_url}/auth/accesstokenrequest");
    let payload = json!({
        "name": cfg.username,
        "password": cfg.password,
        "appId": empty_as_none(&cfg.app_id),
        "appVersion": empty_as_none(&cfg.app_version),
        "cid": empty_as_none(&cfg.cid),
        "sec": empty_as_none(&cfg.secret),
    });

    let response = client.post(url).json(&payload).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("auth request failed ({status}): {body}");
    }

    let parsed: AccessTokenResponse =
        serde_json::from_str(&body).context("parse access token response")?;
    if let Some(error_text) = parsed.error_text.as_deref() {
        if !error_text.trim().is_empty() {
            bail!("access token request rejected: {error_text}");
        }
    }

    let access_token = parsed
        .access_token
        .filter(|token| !token.trim().is_empty())
        .context("missing accessToken in auth response")?;
    let md_access_token = parsed
        .md_access_token
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| access_token.clone());

    Ok(TokenBundle {
        access_token,
        md_access_token,
        expiration_time: parsed.expiration_time,
        user_id: parsed.user_id,
        user_name: parsed.name,
        api_hosts: parsed.api_hosts,
    })
}

async fn renew_access_token(
    client: &Client,
    rest_url: &str,
    current: &TokenBundle,
) -> Result<TokenBundle> {
    let url = format!("{rest_url}/auth/renewAccessToken");
    let response = client
        .get(url)
        .bearer_auth(&current.access_token)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("renew access token failed ({status}): {body}");
    }

    let parsed: AccessTokenResponse =
        serde_json::from_str(&body).context("parse renew access token response")?;
    if let Some(error_text) = parsed.error_text.as_deref() {
        if !error_text.trim().is_empty() {
            bail!("renew access token rejected: {error_text}");
        }
    }

    let access_token = parsed
        .access_token
        .filter(|token| !token.trim().is_empty())
        .context("missing accessToken in renew response")?;
    let md_access_token = parsed
        .md_access_token
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| current.md_access_token.clone());

    Ok(TokenBundle {
        access_token,
        md_access_token,
        expiration_time: parsed.expiration_time,
        user_id: parsed.user_id.or(current.user_id),
        user_name: parsed.name.or_else(|| current.user_name.clone()),
        api_hosts: parsed.api_hosts.or_else(|| current.api_hosts.clone()),
    })
}

async fn fetch_auth_me(client: &Client, rest_url: &str, token: &str) -> Result<Value> {
    let url = format!("{rest_url}/auth/me");
    let response = client.get(url).bearer_auth(token).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("auth/me failed ({status}): {body}");
    }
    Ok(serde_json::from_str(&body)?)
}

async fn measure_rest_rtt_ms(
    client: &Client,
    rest_url: &str,
    token: &str,
) -> Result<u64> {
    let started = time::Instant::now();
    let _ = fetch_auth_me(client, rest_url, token).await?;
    Ok(started.elapsed().as_millis() as u64)
}

async fn list_accounts(
    client: &Client,
    rest_url: &str,
    token: &str,
) -> Result<Vec<AccountInfo>> {
    let payload = fetch_entity_list(client, rest_url, token, "account").await?;
    Ok(payload
        .into_iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_i64)?;
            let name = item.get("name").and_then(Value::as_str)?.to_string();
            Some(AccountInfo {
                id,
                name,
                raw: item,
            })
        })
        .collect())
}

async fn search_contracts(
    client: &Client,
    rest_url: &str,
    token: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<ContractSuggestion>> {
    let url = format!("{rest_url}/contract/suggest");
    let response = client
        .get(url)
        .bearer_auth(token)
        .query(&[("t", query), ("l", &limit.to_string())])
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("contract/suggest failed ({status}): {body}");
    }
    let value: Value = serde_json::from_str(&body)?;
    let mut seen = HashMap::<i64, ()>::new();
    let mut out = Vec::new();
    if let Some(items) = value.as_array() {
        for item in items {
            let Some(id) = item.get("id").and_then(Value::as_i64) else {
                continue;
            };
            if seen.insert(id, ()).is_some() {
                continue;
            }
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("UNKNOWN")
                .to_string();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "contractMaturityId={}",
                        json_i64(item, "contractMaturityId").unwrap_or_default()
                    )
                });
            out.push(ContractSuggestion {
                id,
                name,
                description,
                raw: item.clone(),
            });
        }
    }
    enrich_contract_maturities(client, rest_url, token, &mut out).await;
    Ok(out)
}

async fn enrich_contract_maturities(
    client: &Client,
    rest_url: &str,
    token: &str,
    contracts: &mut [ContractSuggestion],
) {
    let maturity_ids = contracts
        .iter()
        .filter_map(|contract| json_i64(&contract.raw, "contractMaturityId"))
        .collect::<Vec<_>>();
    if maturity_ids.is_empty() {
        return;
    }

    let url = format!("{rest_url}/contractMaturity/items");
    let response = client
        .get(url)
        .bearer_auth(token)
        .query(&[(
            "ids",
            maturity_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )])
        .send()
        .await;
    let Ok(response) = response else {
        return;
    };
    if !response.status().is_success() {
        return;
    }
    let Ok(maturities) = response.json::<Vec<Value>>().await else {
        return;
    };
    let maturities = maturities
        .into_iter()
        .filter_map(|maturity| json_i64(&maturity, "id").map(|id| (id, maturity)))
        .collect::<HashMap<_, _>>();

    for contract in contracts {
        let Some(maturity_id) = json_i64(&contract.raw, "contractMaturityId") else {
            continue;
        };
        let Some(maturity) = maturities.get(&maturity_id) else {
            continue;
        };
        if let Some(raw) = contract.raw.as_object_mut() {
            raw.insert("_midasContractMaturity".to_string(), maturity.clone());
        }
    }
}

#[cfg(test)]
mod token_file_tests {
    use super::*;

    #[test]
    fn token_file_user_id_accepts_browser_writer_numeric_string() {
        let parsed: Value = serde_json::from_str(r#"{"userId":"12345"}"#).unwrap();
        assert_eq!(json_i64(&parsed, "userId"), Some(12345));

        let numeric: Value = serde_json::from_str(r#"{"userId":12345}"#).unwrap();
        assert_eq!(json_i64(&numeric, "userId"), Some(12345));

        let invalid: Value = serde_json::from_str(r#"{"userId":"unknown"}"#).unwrap();
        assert_eq!(json_i64(&invalid, "userId"), None);
    }
}
