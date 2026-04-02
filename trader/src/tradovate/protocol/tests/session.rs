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
