use super::*;

pub async fn search_replay_download_contracts(
    cfg: &AppConfig,
    query: &str,
    limit: usize,
) -> Result<Vec<ContractSuggestion>> {
    let query = query.trim();
    if query.is_empty() {
        bail!("replay contract search query cannot be empty");
    }
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let snapshot = fetch_metadata_snapshot(
        &client,
        &cfg.env,
        &tokens.access_token,
        "contract/suggest",
        &[("t", query.to_string()), ("l", limit.max(1).to_string())],
    )
    .await?;
    let mut seen = std::collections::BTreeSet::new();
    Ok(snapshot
        .payload
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = item.get("id")?.as_i64()?;
            if !seen.insert(id) {
                return None;
            }
            let name = item.get("name")?.as_str()?.to_string();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "contractMaturityId={}",
                        item.get("contractMaturityId")
                            .and_then(Value::as_i64)
                            .unwrap_or_default()
                    )
                });
            Some(ContractSuggestion {
                id,
                name,
                description,
                raw: item.clone(),
            })
        })
        .collect())
}

pub async fn inspect_replay_download_contract(
    cfg: &AppConfig,
    contract: ContractSuggestion,
) -> Result<TradovateReplayContractInspection> {
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let (metadata, _) = fetch_download_contract_metadata(&client, cfg, &tokens, &contract).await;
    validate_contract_identity(&contract, &metadata.contract.payload)?;
    Ok(TradovateReplayContractInspection {
        contract,
        suggested_coverage: metadata.suggested_coverage,
    })
}

pub(super) async fn fetch_download_contract_metadata(
    client: &Client,
    cfg: &AppConfig,
    tokens: &TokenBundle,
    contract: &ContractSuggestion,
) -> (ReplayCacheContractMetadata, Vec<String>) {
    let fetched_at = Utc::now();
    let mut warnings = Vec::new();
    let (accounts, accounts_fetched_at) =
        match fetch_replay_metadata_accounts(client, &cfg.env, &tokens.access_token).await {
            Ok(accounts) => (accounts, Some(Utc::now())),
            Err(err) => {
                warnings.push(optional_metadata_warning("account/list", &err));
                (Vec::new(), None)
            }
        };
    let context = ReplayCacheMetadataContext {
        provider: cfg.broker,
        env: cfg.env,
        user_id: tokens.user_id,
        user_name: tokens.user_name.clone(),
        accounts,
        accounts_endpoint: Some("account/list".to_string()),
        accounts_fetched_at,
    };
    let mut contract_snapshot = ReplayCacheMetadataSnapshot {
        endpoint: "contract/suggest".to_string(),
        fetched_at,
        source_timestamp: metadata_source_timestamp(&contract.raw),
        payload: contract.raw.clone(),
    };

    match fetch_metadata_snapshot(
        client,
        &cfg.env,
        &tokens.access_token,
        "contract/item",
        &[("id", contract.id.to_string())],
    )
    .await
    {
        Ok(snapshot) => contract_snapshot = snapshot,
        Err(err) => warnings.push(optional_metadata_warning("contract/item", &err)),
    }

    let contract_payload = &contract_snapshot.payload;
    let maturity_id = json_i64(contract_payload, "contractMaturityId")
        .or_else(|| json_i64(&contract.raw, "contractMaturityId"));
    let maturity = if let Some(maturity_id) = maturity_id {
        fetch_optional_metadata(
            client,
            &cfg.env,
            &tokens.access_token,
            "contractMaturity/item",
            &[("id", maturity_id.to_string())],
            &mut warnings,
        )
        .await
    } else {
        warnings.push(
            "Optional replay metadata contractMaturity/item was skipped: contractMaturityId was unavailable."
                .to_string(),
        );
        None
    };
    let product_id = maturity
        .as_ref()
        .and_then(|snapshot| json_i64(&snapshot.payload, "productId"));

    let (maturity_chain, product, product_sessions, product_margins, fee_params) =
        if let Some(product_id) = product_id {
            let maturity_chain = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "contractMaturity/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "product/item",
                &[("id", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product_sessions = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "productSession/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product_margins = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "productMargin/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let fee_params = match fetch_fee_metadata_snapshot(
                client,
                &cfg.env,
                &tokens.access_token,
                product_id,
            )
            .await
            {
                Ok(snapshot) => Some(snapshot),
                Err(err) => {
                    warnings.push(optional_metadata_warning(
                        "contract/getproductfeeparams",
                        &err,
                    ));
                    None
                }
            };
            (
                maturity_chain,
                product,
                product_sessions,
                product_margins,
                fee_params,
            )
        } else {
            warnings.push(
                "Optional replay product metadata was skipped: productId was unavailable."
                    .to_string(),
            );
            (None, None, None, None, None)
        };
    let contract_margins = fetch_optional_metadata(
        client,
        &cfg.env,
        &tokens.access_token,
        "contractMargin/deps",
        &[("masterid", contract.id.to_string())],
        &mut warnings,
    )
    .await;
    let suggested_coverage = suggested_contract_coverage(
        maturity.as_ref().map(|snapshot| &snapshot.payload),
        maturity_chain.as_ref().map(|snapshot| &snapshot.payload),
    );
    if suggested_coverage.is_none() {
        warnings.push(
            "A suggested broad contract coverage range could not be derived from maturity metadata."
                .to_string(),
        );
    }

    (
        ReplayCacheContractMetadata {
            context,
            contract: contract_snapshot,
            maturity,
            maturity_chain,
            product,
            product_sessions,
            product_margins,
            contract_margins,
            fee_params,
            suggested_coverage,
        },
        warnings,
    )
}

pub(super) async fn fetch_replay_metadata_accounts(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
) -> Result<Vec<ReplayCacheMetadataAccount>> {
    let endpoint = "account/list";
    let response = client
        .get(format!("{}/{endpoint}", env.rest_url()))
        .bearer_auth(token)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    let accounts = match payload {
        Value::Array(accounts) => accounts,
        Value::Object(_) => vec![payload],
        _ => Vec::new(),
    };
    Ok(accounts
        .into_iter()
        .filter_map(|account| {
            Some(ReplayCacheMetadataAccount {
                id: account.get("id")?.as_i64()?,
                name: account.get("name")?.as_str()?.to_string(),
            })
        })
        .collect())
}

pub(super) async fn fetch_optional_metadata(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    endpoint: &str,
    query: &[(&'static str, String)],
    warnings: &mut Vec<String>,
) -> Option<ReplayCacheMetadataSnapshot> {
    match fetch_metadata_snapshot(client, env, token, endpoint, query).await {
        Ok(snapshot) => Some(snapshot),
        Err(err) => {
            warnings.push(optional_metadata_warning(endpoint, &err));
            None
        }
    }
}

pub(super) async fn fetch_metadata_snapshot(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    endpoint: &str,
    query: &[(&'static str, String)],
) -> Result<ReplayCacheMetadataSnapshot> {
    let url = format!("{}/{}", env.rest_url(), endpoint);
    let response = client
        .get(url)
        .bearer_auth(token)
        .query(query)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    Ok(ReplayCacheMetadataSnapshot {
        endpoint: endpoint.to_string(),
        fetched_at: Utc::now(),
        source_timestamp: metadata_source_timestamp(&payload),
        payload,
    })
}

pub(super) async fn fetch_fee_metadata_snapshot(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    product_id: i64,
) -> Result<ReplayCacheMetadataSnapshot> {
    let endpoint = "contract/getproductfeeparams";
    let url = format!("{}/{}", env.rest_url(), endpoint);
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(&fee_metadata_request_body(product_id))
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    Ok(ReplayCacheMetadataSnapshot {
        endpoint: endpoint.to_string(),
        fetched_at: Utc::now(),
        source_timestamp: metadata_source_timestamp(&payload),
        payload,
    })
}

pub(super) fn fee_metadata_request_body(product_id: i64) -> Value {
    json!({ "productIds": [product_id] })
}

pub(super) fn optional_metadata_warning(endpoint: &str, err: &anyhow::Error) -> String {
    format!("Optional replay metadata {endpoint} failed: {err}")
}

pub(super) fn sanitized_metadata_http_error(
    endpoint: &str,
    status: reqwest::StatusCode,
    _provider_body: &str,
) -> anyhow::Error {
    let classification = match status.as_u16() {
        401 => "authentication rejected",
        403 => "authorization rejected",
        404 => "endpoint or entity not found",
        408 => "request timed out",
        429 => "provider rate limit",
        400..=499 => "provider rejected the request",
        500..=599 => "provider server error",
        _ => "unexpected provider response",
    };
    anyhow::anyhow!(
        "{endpoint} failed with HTTP {} ({classification}); provider response body omitted",
        status.as_u16()
    )
}

pub(super) fn metadata_source_timestamp(payload: &Value) -> Option<DateTime<Utc>> {
    match payload {
        Value::Array(items) => items.iter().find_map(metadata_source_timestamp),
        Value::Object(fields) => {
            for key in [
                "timestamp",
                "updatedAt",
                "lastModified",
                "definitionTimestamp",
            ] {
                if let Some(timestamp) = fields
                    .get(key)
                    .and_then(Value::as_str)
                    .and_then(parse_metadata_timestamp)
                {
                    return Some(timestamp);
                }
            }
            fields.values().find_map(metadata_source_timestamp)
        }
        _ => None,
    }
}

pub(super) fn parse_metadata_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

pub(super) fn metadata_expiration_date(payload: &Value) -> Option<chrono::NaiveDate> {
    let raw = payload.get("expirationDate")?.as_str()?;
    parse_metadata_timestamp(raw)
        .map(|timestamp| timestamp.date_naive())
        .or_else(|| chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok())
        .or_else(|| {
            raw.get(..10)
                .and_then(|date| chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
        })
}

pub(super) fn suggested_contract_coverage(
    selected_maturity: Option<&Value>,
    maturity_chain: Option<&Value>,
) -> Option<ReplayCacheSuggestedCoverage> {
    let selected_maturity = selected_maturity?;
    let selected_expiration = metadata_expiration_date(selected_maturity)?;
    let selected_id = json_i64(selected_maturity, "id");
    let chain = maturity_chain?.as_array()?;
    let previous_expiration = chain
        .iter()
        .filter(|maturity| {
            selected_id.is_none_or(|selected_id| json_i64(maturity, "id") != Some(selected_id))
        })
        .filter_map(metadata_expiration_date)
        .filter(|expiration| *expiration < selected_expiration)
        .max()?;
    Some(ReplayCacheSuggestedCoverage {
        start_date: previous_expiration,
        end_date: selected_expiration,
        basis: "previous maturity expiration through selected maturity expiration; actual liquidity and provider retention may differ"
            .to_string(),
        estimated: true,
    })
}

pub(super) fn market_specs_from_metadata(
    metadata: &ReplayCacheContractMetadata,
) -> Option<MarketSpecs> {
    let product = metadata.product.as_ref().map(|snapshot| &snapshot.payload);
    let contract = &metadata.contract.payload;
    let tick_size = product
        .and_then(|product| {
            json_number(product, "tickSize").or_else(|| json_number(product, "minTick"))
        })
        .or_else(|| json_number(contract, "providerTickSize"))
        .or_else(|| json_number(contract, "tickSize"));
    let value_per_point = product.and_then(|product| json_number(product, "valuePerPoint"));
    if product.is_none() && tick_size.is_none() && value_per_point.is_none() {
        return None;
    }
    Some(MarketSpecs {
        session_profile: product.map(infer_session_profile),
        value_per_point,
        tick_size,
    })
}

pub(super) async fn resolve_download_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract_symbol: &str,
    limit: usize,
) -> Result<ContractSuggestion> {
    let contracts = search_contracts(client, env, token, contract_symbol, limit).await?;
    contracts
        .iter()
        .find(|contract| contract.name.eq_ignore_ascii_case(contract_symbol))
        .cloned()
        .with_context(|| {
            let available = contracts
                .iter()
                .map(|contract| contract.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if available.is_empty() {
                format!("contract search returned no results for {contract_symbol}")
            } else {
                format!(
                    "contract search did not return exact symbol {contract_symbol}; suggestions: {available}"
                )
            }
        })
}

pub(super) async fn resolve_requested_download_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract_symbol: &str,
    exact_contract: Option<&ContractSuggestion>,
    limit: usize,
) -> Result<ContractSuggestion> {
    if let Some(contract) = exact_contract {
        if !contract.name.eq_ignore_ascii_case(contract_symbol) {
            bail!(
                "selected contract name {} does not match requested symbol {contract_symbol}",
                contract.name
            );
        }
        return Ok(contract.clone());
    }
    resolve_download_contract(client, env, token, contract_symbol, limit).await
}

pub(super) fn validate_contract_identity(
    contract: &ContractSuggestion,
    payload: &Value,
) -> Result<()> {
    if let Some(id) = payload.get("id").and_then(Value::as_i64)
        && id != contract.id
    {
        bail!(
            "provider contract id {id} does not match selected contract id {}",
            contract.id
        );
    }
    if let Some(name) = payload.get("name").and_then(Value::as_str)
        && !name.eq_ignore_ascii_case(&contract.name)
    {
        bail!(
            "provider contract name {name} does not match selected contract name {}",
            contract.name
        );
    }
    Ok(())
}
