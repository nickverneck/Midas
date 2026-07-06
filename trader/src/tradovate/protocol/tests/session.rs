use super::*;

#[test]
fn jwt_expiration_time_reads_exp_claim() {
    let token = "eyJhbGciOiJub25lIn0.eyJleHAiOjE4OTM0NTYwMDB9.sig";
    let expires_at = jwt_expiration_time(token).expect("jwt exp should parse");

    let expected = DateTime::<Utc>::from_timestamp(1_893_456_000, 0).unwrap();
    assert_eq!(expires_at, expected);
}

#[test]
fn token_refresh_due_uses_jwt_exp_when_expiration_time_missing() {
    let tokens = TokenBundle {
        access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
        md_access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
        expiration_time: None,
        user_id: Some(42),
        user_name: Some("demo".to_string()),
    };

    let now = DateTime::<Utc>::from_timestamp(1_773_436_044 - 60, 0).unwrap();
    assert!(token_refresh_due(&tokens, now));
}

#[test]
fn parse_expiration_time_accepts_rfc3339() {
    let parsed = parse_expiration_time("2026-03-13T15:04:05Z").expect("timestamp should parse");
    let expected = DateTime::parse_from_rfc3339("2026-03-13T15:04:05Z")
        .unwrap()
        .with_timezone(&Utc);
    assert_eq!(parsed, expected);
}

#[test]
fn token_file_maintenance_reloads_on_external_rewrite() {
    let unique = format!(
        "trader-token-rewrite-{}-{}.json",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap()
    );
    let token_path = std::env::temp_dir().join(unique);
    let cache_path = token_path.with_extension("cache.json");

    std::fs::write(
        &token_path,
        r#"{"token":"same-token","accessToken":"same-token","mdAccessToken":"same-md","expirationTime":"2026-07-06T15:45:46Z"}"#,
    )
    .expect("write initial token file");

    let mut cfg = AppConfig::default();
    cfg.auth_mode = AuthMode::TokenFile;
    cfg.token_path = token_path.clone();
    cfg.session_cache_path = cache_path;

    let loaded = load_runtime_token_bundle(&cfg).expect("load initial token file");
    assert!(
        next_token_maintenance_action(&cfg, &loaded.tokens, loaded.file_snapshot.as_ref())
            .expect("check unchanged token file")
            .is_none()
    );

    std::fs::write(
        &token_path,
        r#"{
  "token": "same-token",
  "accessToken": "same-token",
  "mdAccessToken": "same-md",
  "expirationTime": "2026-07-06T15:45:46Z"
}"#,
    )
    .expect("rewrite token file");

    let action = next_token_maintenance_action(&cfg, &loaded.tokens, loaded.file_snapshot.as_ref())
        .expect("check rewritten token file");
    assert!(matches!(
        action,
        Some(TokenMaintenanceAction::ReloadTokenFile(_))
    ));

    let _ = std::fs::remove_file(&token_path);
}
