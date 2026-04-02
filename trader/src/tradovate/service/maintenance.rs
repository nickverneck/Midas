use super::*;

pub(super) async fn maintain_session(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        return Ok(());
    };
    if session.replay_enabled {
        return Ok(());
    }

    let refresh_action = next_token_maintenance_action(&session.cfg, &session.tokens)?;
    let mut forced_restart = false;
    let mut status_message = None;

    if let Some(action) = refresh_action {
        let next_tokens = match action {
            TokenMaintenanceAction::RefreshCredentials => {
                request_access_token(&state.client, &session.cfg).await?
            }
            TokenMaintenanceAction::ReloadTokenFile => load_runtime_token_bundle(&session.cfg)?,
        };

        if let Some(session) = state.session.as_mut() {
            if token_bundle_changed(&session.tokens, &next_tokens) {
                session.tokens = next_tokens;
                save_token_cache(&session.cfg.session_cache_path, &session.tokens)?;
                refresh_session_state(&state.client, session, event_tx).await?;
                if let Some(task) = state.user_task.take() {
                    task.abort();
                }
                if let Some(task) = state.market_task.take() {
                    task.abort();
                }
                if let Some(task) = state.rest_probe_task.take() {
                    task.abort();
                }
                forced_restart = true;
                status_message = Some(match action {
                    TokenMaintenanceAction::RefreshCredentials => {
                        "Session token refreshed; reconnecting background streams.".to_string()
                    }
                    TokenMaintenanceAction::ReloadTokenFile => {
                        "Session token reloaded from file; reconnecting background streams."
                            .to_string()
                    }
                });
            }
        }
    }

    let restart = ensure_background_tasks(state, internal_tx).await?;

    if let Some(message) = status_message {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }
    if restart.user_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "User sync stream restarted.".to_string(),
        ));
    }
    if restart.market_restarted && !forced_restart {
        let contract_name = state
            .session
            .as_ref()
            .and_then(|session| session.selected_contract.as_ref())
            .map(|contract| contract.name.clone())
            .unwrap_or_else(|| "selected contract".to_string());
        let _ = event_tx.send(ServiceEvent::Status(format!(
            "Market data stream restarted for {contract_name}."
        )));
    }
    if restart.rest_probe_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "REST latency probe restarted.".to_string(),
        ));
    }

    Ok(())
}

async fn refresh_session_state(
    client: &Client,
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let accounts = list_accounts(client, &session.cfg.env, &session.tokens.access_token).await?;
    let mut user_store = UserSyncStore::default();
    seed_user_store(
        client,
        &session.cfg.env,
        &session.tokens.access_token,
        &mut user_store,
    )
    .await;

    session.accounts = accounts.clone();
    if let Some(selected_account_id) = session.selected_account_id {
        if !session
            .accounts
            .iter()
            .any(|account| account.id == selected_account_id)
        {
            session.selected_account_id = session.accounts.first().map(|account| account.id);
        }
    } else {
        session.selected_account_id = session.accounts.first().map(|account| account.id);
    }
    session.user_store = user_store;

    let snapshots = session.user_store.build_snapshots(
        &session.accounts,
        Some(&session.market),
        &session.managed_protection,
    );
    let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts));
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
    Ok(())
}
