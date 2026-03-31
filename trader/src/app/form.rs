impl FormState {
    fn from_config(config: &AppConfig) -> Self {
        Self {
            env: config.env,
            auth_mode: config.auth_mode,
            log_mode: config.log_mode,
            token_override: config.token_override.clone(),
            username: config.username.clone(),
            password: config.password.clone(),
            api_key: config.api_key.clone(),
            app_id: config.app_id.clone(),
            app_version: config.app_version.clone(),
            cid: config.cid.clone(),
            secret: config.secret.clone(),
            token_path: config.token_path.display().to_string(),
        }
    }
}
