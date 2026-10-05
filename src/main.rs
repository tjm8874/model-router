use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use futures_util::TryStreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::IpAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(version, about = "Named-backend OpenAI-compatible model router")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Override server.port from the configuration.
    #[arg(long)]
    port: Option<u16>,
    #[arg(long)]
    check_config: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    server: Server,
    backends: BTreeMap<String, Backend>,
}

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Server {
    host: IpAddr,
    port: u16,
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
    catalog_timeout_secs: u64,
    max_body_bytes: usize,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".parse().unwrap(),
            port: 18080,
            connect_timeout_secs: 10,
            read_timeout_secs: 300,
            catalog_timeout_secs: 15,
            max_body_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Backend {
    base_url: String,
    /// Allowed POST paths relative to base_url, e.g. chat/completions.
    endpoints: Vec<String>,
    /// Static upstream IDs for servers without a models endpoint.
    #[serde(default)]
    models: Vec<String>,
    /// Environment-variable name, never a literal credential.
    api_key_env: Option<String>,
}

impl Config {
    fn validate(&mut self) -> Result<()> {
        ensure!(
            self.server.host.is_loopback(),
            "server.host must be loopback"
        );
        ensure!(self.server.port > 0, "port must be 1-65535");
        ensure!(
            !self.backends.is_empty(),
            "at least one backend is required"
        );
        ensure!(
            self.server.connect_timeout_secs > 0
                && self.server.read_timeout_secs > 0
                && self.server.catalog_timeout_secs > 0
                && self.server.max_body_bytes > 0,
            "timeouts and body limit must be positive"
        );
        for (name, b) in &mut self.backends {
            ensure!(
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)),
                "invalid backend name {name}"
            );
            let url = reqwest::Url::parse(&b.base_url).context("invalid backend base_url")?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none(),
                "invalid backend URL for {name}"
            );
            ensure!(
                !b.endpoints.is_empty()
                    && b.endpoints.iter().all(|s| matches!(
                        s.as_str(),
                        "chat/completions" | "embeddings" | "decide"
                    )),
                "unsupported endpoints for {name}"
            );
            ensure!(
                b.models.iter().all(|m| !m.trim().is_empty()),
                "empty model ID for {name}"
            );
            if let Some(var) = &b.api_key_env {
                let key = std::env::var(var)
                    .with_context(|| format!("missing credential environment variable {var}"))?;
                ensure!(
                    !key.trim().is_empty()
                        && reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")).is_ok(),
                    "invalid credential for {name}"
                );
            }
            b.base_url = b.base_url.trim_end_matches('/').to_string();
        }
        Ok(())
    }
}

#[derive(Clone)]
struct App {
    config: Arc<Config>,
    client: reqwest::Client,
}

fn transport_cause(error: &reqwest::Error) -> String {
    use std::error::Error;
    let mut cause = error.source();
    while let Some(current) = cause {
        if let Some(io) = current.downcast_ref::<std::io::Error>() {
            return io.to_string();
        }
        cause = current.source();
    }
    if error.is_timeout() {
        "timeout".into()
    } else {
        "transport failure".into()
    }
}

fn app(config: Config) -> Result<Router> {
    let limit = config.server.max_body_bytes;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(config.server.connect_timeout_secs))
        .read_timeout(Duration::from_secs(config.server.read_timeout_secs))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()?;
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/decide", post(decide))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(App {
            config: Arc::new(config),
            client,
        }))
}

fn failure(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"type": "router_error", "code": code, "message": message}})),
    )
        .into_response()
}

async fn health(State(app): State<App>) -> Json<Value> {
    // Liveness only: upstream readiness is reported by /v1/models.
    Json(
        json!({"status": "ok", "version": env!("CARGO_PKG_VERSION"), "backends": app.config.backends.keys().collect::<Vec<_>>() }),
    )
}

fn request(
    app: &App,
    name: &str,
    method: reqwest::Method,
    endpoint: &str,
) -> reqwest::RequestBuilder {
    let b = &app.config.backends[name];
    let mut r = app
        .client
        .request(method, format!("{}/{endpoint}", b.base_url));
    if let Some(var) = &b.api_key_env {
        // Validated at startup; incoming Core Authorization is never forwarded.
        r = r.bearer_auth(std::env::var(var).expect("validated credential"));
    }
    r
}

async fn models(State(app): State<App>) -> Response {
    let mut jobs = tokio::task::JoinSet::new();
    for (name, b) in &app.config.backends {
        let name = name.clone();
        let b = b.clone();
        let app = app.clone();
        jobs.spawn(async move {
            let result = async {
                if !b.models.is_empty() {
                    return Ok::<Vec<Value>, anyhow::Error>(
                        b.models
                            .iter()
                            .map(|id| json!({"id": id, "object": "model", "owned_by": name}))
                            .collect(),
                    );
                }
                let value: Value = request(&app, &name, reqwest::Method::GET, "models")
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                let data = value
                    .get("data")
                    .and_then(Value::as_array)
                    .context("invalid models response")?;
                ensure!(
                    data.iter().all(|m| m
                        .get("id")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())),
                    "invalid model entry"
                );
                Ok(data.clone())
            };
            let result = tokio::time::timeout(
                Duration::from_secs(app.config.server.catalog_timeout_secs),
                result,
            )
            .await;
            (name, result)
        });
    }
    let mut data = Vec::new();
    let mut errors = Vec::new();
    while let Some(result) = jobs.join_next().await {
        match result {
            Ok((name, Ok(Ok(entries)))) => {
                for mut entry in entries {
                    entry["id"] = json!(format!("{name}:{}", entry["id"].as_str().unwrap()));
                    data.push(entry);
                }
            }
            Ok((name, _)) => errors.push(json!({"backend": name, "code": "catalog_unavailable"})),
            Err(_) => errors.push(json!({"code": "catalog_task_failed"})),
        }
    }
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let status = if data.is_empty() && !errors.is_empty() {
        StatusCode::BAD_GATEWAY
    } else {
        StatusCode::OK
    };
    (
        status,
        Json(json!({"object":"list", "data":data, "router_errors":errors})),
    )
        .into_response()
}

async fn chat(State(app): State<App>, headers: HeaderMap, body: Bytes) -> Response {
    forward(app, "chat/completions", headers, body).await
}
async fn embeddings(State(app): State<App>, headers: HeaderMap, body: Bytes) -> Response {
    forward(app, "embeddings", headers, body).await
}
async fn decide(State(app): State<App>, headers: HeaderMap, body: Bytes) -> Response {
    forward(app, "decide", headers, body).await
}

fn split_model(value: &str) -> Option<(&str, &str)> {
    let (name, model) = value.split_once(':')?;
    (!name.is_empty() && !model.trim().is_empty()).then_some((name, model))
}

async fn forward(app: App, endpoint: &'static str, headers: HeaderMap, body: Bytes) -> Response {
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(Value::Object(o)) => Value::Object(o),
        _ => {
            return failure(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                "Expected a JSON object",
            );
        }
    };
    let Some((name, model)) = payload
        .get("model")
        .and_then(Value::as_str)
        .and_then(split_model)
    else {
        return failure(
            StatusCode::BAD_REQUEST,
            "invalid_model",
            "model must be backend:model-id",
        );
    };
    let (name, model) = (name.to_string(), model.to_string());
    let Some(backend) = app.config.backends.get(&name) else {
        return failure(
            StatusCode::BAD_REQUEST,
            "unknown_backend",
            "Backend is not configured",
        );
    };
    if !backend.endpoints.iter().any(|s| s == endpoint) {
        return failure(
            StatusCode::BAD_REQUEST,
            "unsupported_endpoint",
            "Backend does not support this endpoint",
        );
    }
    if !backend.models.is_empty() && !backend.models.contains(&model) {
        return failure(
            StatusCode::BAD_REQUEST,
            "unknown_model",
            "Model is not declared for this backend",
        );
    }
    payload["model"] = json!(model);
    let mut upstream = request(&app, &name, reqwest::Method::POST, endpoint).json(&payload);
    if let Some(accept) = headers.get(header::ACCEPT) {
        upstream = upstream.header(header::ACCEPT, accept);
    }
    let start = Instant::now();
    let upstream = match upstream.send().await {
        Ok(r) => r,
        Err(e) => {
            let (status, code) = if e.is_timeout() {
                (StatusCode::GATEWAY_TIMEOUT, "upstream_timeout")
            } else {
                (StatusCode::BAD_GATEWAY, "upstream_unavailable")
            };
            tracing::warn!(
                backend = name,
                model,
                endpoint,
                elapsed_ms = start.elapsed().as_millis() as u64,
                code,
                cause = transport_cause(&e),
                "upstream failed"
            );
            return failure(
                status,
                code,
                "Upstream request failed; check router and backend logs",
            );
        }
    };
    let status = upstream.status();
    let mut response = Response::builder().status(status);
    // Never forward hop-by-hop, credentials, Set-Cookie or server-internal headers.
    for key in [
        header::CONTENT_TYPE,
        header::CACHE_CONTROL,
        header::RETRY_AFTER,
    ] {
        if let Some(value) = upstream.headers().get(&key) {
            response = response.header(key, value);
        }
    }
    response = response.header("x-model-router-backend", &name);
    tracing::info!(
        backend = name,
        model,
        endpoint,
        status = status.as_u16(),
        header_ms = start.elapsed().as_millis() as u64,
        "routed"
    );
    // No spawned relay task: downstream disconnect drops this stream and the upstream body.
    // Preserve JSON and SSE bytes, tool calls, usage and backend error bodies verbatim.
    let stream = upstream.bytes_stream().inspect_err(move |_| {
        tracing::warn!(backend = name, endpoint, "upstream body interrupted");
    });
    response.body(Body::from_stream(stream)).unwrap()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "model_router=info".into()),
        )
        .init();
    let args = Args::parse();
    let mut config: Config =
        toml::from_str(&std::fs::read_to_string(&args.config).context("read config")?)
            .context("parse config")?;
    if let Some(port) = args.port {
        config.server.port = port;
    }
    config.validate()?;
    if args.check_config {
        println!("Configuration valid ({} backends)", config.backends.len());
        return Ok(());
    }
    let address = (config.server.host, config.server.port);
    let router = app(config)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(address = %listener.local_addr()?, "model router listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .unwrap();
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests;
