async fn authenticate(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    if let Some(token) = empty_as_none(&cfg.token_override) {
        let user_name = fetch_auth_me(client, &cfg.env, token)
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
        });
    }

    match cfg.auth_mode {
        AuthMode::TokenFile => {
            let tokens = load_token_file(&cfg.token_path)
                .or_else(|_| load_token_file(&cfg.session_cache_path))
                .with_context(|| {
                    format!(
                        "load token from {} or {}",
                        cfg.token_path.display(),
                        cfg.session_cache_path.display()
                    )
                })?;
            let user_name = fetch_auth_me(client, &cfg.env, &tokens.access_token)
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
        user_id: parsed.get("userId").and_then(Value::as_i64),
        user_name: parsed
            .get("name")
            .and_then(Value::as_str)
            .map(ToString::to_string),
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
    };
    fs::write(path, serde_json::to_string_pretty(&body)?)
        .with_context(|| format!("write token cache {}", path.display()))?;
    Ok(())
}

async fn request_access_token(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    let url = format!("{}/auth/accesstokenrequest", cfg.env.rest_url());
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
    })
}

async fn fetch_auth_me(client: &Client, env: &TradingEnvironment, token: &str) -> Result<Value> {
    let url = format!("{}/auth/me", env.rest_url());
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
    env: &TradingEnvironment,
    token: &str,
) -> Result<u64> {
    let started = time::Instant::now();
    let _ = fetch_auth_me(client, env, token).await?;
    Ok(started.elapsed().as_millis() as u64)
}

async fn list_accounts(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
) -> Result<Vec<AccountInfo>> {
    let payload = fetch_entity_list(client, env, token, "account").await?;
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
    env: &TradingEnvironment,
    token: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<ContractSuggestion>> {
    let url = format!("{}/contract/suggest", env.rest_url());
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
    Ok(out)
}
