//! Standalone HTTP server for the research workspace.
mod chart;
mod standalone;
mod strategy;
mod strategy_artifacts;
mod strategy_editor;

use anyhow::{Context, Result, ensure};
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get_service,
};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
};
use tower_http::services::{ServeDir, ServeFile};

fn allowed_host(host: &str, configured: Option<&str>) -> bool {
    let Ok(authority) = host.parse::<axum::http::uri::Authority>() else {
        return false;
    };
    if authority.port_u16().is_none() && host.rsplit_once(':').is_some() && !host.starts_with('[') {
        return false;
    }
    let name = authority.host().trim_matches(['[', ']']);
    if name.eq_ignore_ascii_case("localhost") || name == "127.0.0.1" || name == "::1" {
        return true;
    }
    let Ok(IpAddr::V4(addr)) = name.parse::<IpAddr>() else {
        return false;
    };
    configured.and_then(|s| s.parse::<Ipv4Addr>().ok()) == Some(addr)
}

fn allowed_headers(headers: &HeaderMap, lan: Option<&str>) -> bool {
    let Some(host) = headers
        .get(axum::http::header::HOST)
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };
    if !allowed_host(host, lan) {
        return false;
    }
    if let Some(origin) = headers.get(axum::http::header::ORIGIN) {
        let Ok(origin) = origin.to_str() else {
            return false;
        };
        if !origin.eq_ignore_ascii_case(&format!("http://{host}"))
            && !origin.eq_ignore_ascii_case(&format!("https://{host}"))
        {
            return false;
        }
    }
    true
}

async fn local_request_guard(request: Request, next: Next) -> Response {
    let configured = std::env::var("MIDAS_LAN_HOST").ok();
    if !allowed_headers(request.headers(), configured.as_deref()) {
        return (
            StatusCode::FORBIDDEN,
            "Request host or origin is not allowed",
        )
            .into_response();
    }
    next.run(request).await
}

pub(crate) async fn serve(listen: SocketAddr, assets: Option<PathBuf>) -> Result<()> {
    let assets = assets.unwrap_or_else(|| {
        let local = PathBuf::from("webui/dist-rust");
        if local.is_dir() {
            local
        } else {
            PathBuf::from("../webui/dist-rust")
        }
    });
    ensure!(
        assets.join("rust.html").is_file(),
        "research workspace assets missing at {}; run `bun run build:rust` in webui",
        assets.display()
    );
    let index = assets.join("rust.html");
    let app = Router::new()
        .merge(chart::router())
        .merge(standalone::router())
        .merge(strategy::router())
        .merge(strategy_editor::router())
        .route("/", get_service(ServeFile::new(index)))
        .fallback_service(ServeDir::new(assets))
        .layer(DefaultBodyLimit::max(32 * 1024))
        .layer(middleware::from_fn(local_request_guard));
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind web server to {listen}"))?;
    eprintln!("Research workspace listening on http://{listen}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    #[test]
    fn local_host_and_origin_gate() {
        let mut h = HeaderMap::new();
        h.insert("host", HeaderValue::from_static("192.168.1.57:9757"));
        h.insert(
            "origin",
            HeaderValue::from_static("http://192.168.1.57:9757"),
        );
        assert!(allowed_headers(&h, Some("192.168.1.57")));
        assert!(!allowed_headers(&h, None));
        h.insert("origin", HeaderValue::from_static("http://example.com"));
        assert!(!allowed_headers(&h, Some("192.168.1.57")));
        h.insert("host", HeaderValue::from_static("localhost:9757"));
        h.remove("origin");
        assert!(allowed_headers(&h, None));
        h.insert("host", HeaderValue::from_static("localhost.evil:9757"));
        assert!(!allowed_headers(&h, None));
    }
}
