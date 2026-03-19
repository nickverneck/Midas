#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenMaintenanceAction {
    RefreshCredentials,
    ReloadTokenFile,
}

fn next_token_maintenance_action(
    cfg: &AppConfig,
    tokens: &TokenBundle,
) -> Result<Option<TokenMaintenanceAction>> {
    if empty_as_none(&cfg.token_override).is_some() {
        return Ok(None);
    }

    match cfg.auth_mode {
        AuthMode::Credentials => {
            if token_refresh_due(tokens, Utc::now()) {
                Ok(Some(TokenMaintenanceAction::RefreshCredentials))
            } else {
                Ok(None)
            }
        }
        AuthMode::TokenFile => {
            let loaded = load_runtime_token_bundle(cfg)?;
            if token_bundle_changed(tokens, &loaded) {
                Ok(Some(TokenMaintenanceAction::ReloadTokenFile))
            } else {
                Ok(None)
            }
        }
    }
}

fn load_runtime_token_bundle(cfg: &AppConfig) -> Result<TokenBundle> {
    load_token_file(&cfg.token_path)
        .or_else(|_| load_token_file(&cfg.session_cache_path))
        .with_context(|| {
            format!(
                "load token from {} or {}",
                cfg.token_path.display(),
                cfg.session_cache_path.display()
            )
        })
}

fn token_refresh_due(tokens: &TokenBundle, now: DateTime<Utc>) -> bool {
    let Some(expires_at) = token_expires_at(tokens) else {
        return false;
    };
    expires_at <= now + chrono::Duration::seconds(TOKEN_REFRESH_LEAD_SECS)
}

fn token_expires_at(tokens: &TokenBundle) -> Option<DateTime<Utc>> {
    tokens
        .expiration_time
        .as_deref()
        .and_then(parse_expiration_time)
        .or_else(|| jwt_expiration_time(&tokens.access_token))
}

fn parse_expiration_time(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }

    let numeric = raw.trim().parse::<i64>().ok()?;
    let seconds = if numeric > 10_000_000_000 {
        numeric / 1000
    } else {
        numeric
    };
    DateTime::<Utc>::from_timestamp(seconds, 0)
}

fn jwt_expiration_time(token: &str) -> Option<DateTime<Utc>> {
    let claims = token.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD
        .decode(claims)
        .or_else(|_| URL_SAFE.decode(claims))
        .ok()?;
    let parsed: Value = serde_json::from_slice(&payload).ok()?;
    let seconds = parsed.get("exp").and_then(Value::as_i64)?;
    DateTime::<Utc>::from_timestamp(seconds, 0)
}

fn token_bundle_changed(current: &TokenBundle, next: &TokenBundle) -> bool {
    current.access_token != next.access_token
        || current.md_access_token != next.md_access_token
        || current.expiration_time != next.expiration_time
        || current.user_id != next.user_id
        || current.user_name != next.user_name
}

#[derive(Debug, Clone, Copy, Default)]
struct TaskRestartState {
    user_restarted: bool,
    market_restarted: bool,
    rest_probe_restarted: bool,
}

async fn ensure_background_tasks(
    state: &mut ServiceState,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<TaskRestartState> {
    let Some(session) = state.session.as_ref() else {
        return Ok(TaskRestartState::default());
    };

    let user_needed = state
        .user_task
        .as_ref()
        .is_none_or(tokio::task::JoinHandle::is_finished);
    let market_needed = session.selected_contract.is_some()
        && state
            .market_task
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished);
    let rest_probe_needed = state
        .rest_probe_task
        .as_ref()
        .is_none_or(tokio::task::JoinHandle::is_finished);

    let user_spawn = if user_needed {
        Some((
            session.cfg.clone(),
            session.tokens.clone(),
            session
                .accounts
                .iter()
                .map(|account| account.id)
                .collect::<Vec<_>>(),
        ))
    } else {
        None
    };
    let market_spawn = if market_needed {
        session.selected_contract.as_ref().map(|contract| {
            (
                session.cfg.clone(),
                session.tokens.access_token.clone(),
                session.tokens.md_access_token.clone(),
                contract.clone(),
            )
        })
    } else {
        None
    };
    let rest_probe_spawn = if rest_probe_needed {
        Some((
            state.client.clone(),
            session.cfg.clone(),
            session.tokens.access_token.clone(),
        ))
    } else {
        None
    };

    if user_needed {
        if let Some(task) = state.user_task.take() {
            task.abort();
        }
        if let Some((cfg, tokens, account_ids)) = user_spawn {
            let (request_tx, user_task) =
                spawn_user_sync_task(cfg, tokens, account_ids, internal_tx.clone());
            if let Some(session) = state.session.as_mut() {
                session.request_tx = request_tx;
            }
            state.user_task = Some(user_task);
        }
    }

    if market_needed {
        if let Some(task) = state.market_task.take() {
            task.abort();
        }
        if let Some((cfg, access_token, md_access_token, contract)) = market_spawn {
            let market_specs =
                fetch_contract_specs(&state.client, &cfg.env, &access_token, &contract)
                    .await
                    .ok();
            let bar_type = state
                .session
                .as_ref()
                .map(|s| s.bar_type)
                .unwrap_or_default();
            state.market_task = Some(tokio::spawn(market_data_worker(
                cfg,
                md_access_token,
                contract,
                market_specs,
                bar_type,
                internal_tx.clone(),
            )));
        }
    }

    if rest_probe_needed {
        if let Some(task) = state.rest_probe_task.take() {
            task.abort();
        }
        if let Some((client, cfg, access_token)) = rest_probe_spawn {
            state.rest_probe_task = Some(spawn_rest_probe_task(
                client,
                cfg,
                access_token,
                internal_tx.clone(),
            ));
        }
    }

    Ok(TaskRestartState {
        user_restarted: user_needed,
        market_restarted: market_needed,
        rest_probe_restarted: rest_probe_needed,
    })
}

fn shutdown_state(state: &mut ServiceState, event_tx: &UnboundedSender<ServiceEvent>) {
    shutdown_tasks(state);
    state.session = None;
    let _ = event_tx.send(ServiceEvent::Disconnected);
}

fn shutdown_tasks(state: &mut ServiceState) {
    if let Some(task) = state.user_task.take() {
        task.abort();
    }
    if let Some(task) = state.market_task.take() {
        task.abort();
    }
    if let Some(task) = state.rest_probe_task.take() {
        task.abort();
    }
}
