use super::*;
use axum::{extract::Request, routing::any};
use futures_util::StreamExt;
use std::sync::atomic::{AtomicBool, Ordering};
use tower::ServiceExt;

async fn server(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, task)
}

fn config(url: &str) -> Config {
    Config {
        server: Server::default(),
        backends: BTreeMap::from([(
            "TestBackend".into(),
            Backend {
                base_url: url.into(),
                endpoints: vec![
                    "chat/completions".into(),
                    "embeddings".into(),
                    "decide".into(),
                ],
                models: vec![],
                api_key_env: None,
            },
        )]),
    }
}

async fn call(router: Router, path: &str, body: Value) -> Response {
    router
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer core-secret")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn value(response: Response) -> Value {
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn model_split_preserves_tags_and_slashes() {
    assert_eq!(
        split_model("TestBackend:org/model:8bit"),
        Some(("TestBackend", "org/model:8bit"))
    );
    for invalid in ["model", ":model", "backend:", "backend: "] {
        assert!(split_model(invalid).is_none());
    }
}

#[test]
fn invalid_config_is_rejected() {
    let mut c = config("http://127.0.0.1:8000/v1/");
    c.validate().unwrap();
    assert_eq!(
        c.backends["TestBackend"].base_url,
        "http://127.0.0.1:8000/v1"
    );
    for url in [
        "file:///tmp/model",
        "http://user:password@host/v1",
        "http://host/v1?token=x",
    ] {
        let mut c = config(url);
        assert!(c.validate().is_err());
    }
    let mut c = config("http://host/v1");
    c.server.host = "0.0.0.0".parse().unwrap();
    assert!(c.validate().is_err());
    let mut c = config("http://host/v1");
    c.server.port = 0;
    assert!(c.validate().is_err());
    let mut c = config("http://host/v1");
    c.backends.get_mut("TestBackend").unwrap().endpoints = vec!["admin".into()];
    assert!(c.validate().is_err());
}

#[tokio::test]
async fn model_rewrite_preserves_payload_and_does_not_leak_core_auth() {
    let mock = Router::new().route(
        "/v1/chat/completions",
        post(
            |headers: HeaderMap, Json(payload): Json<Value>| async move {
                assert!(!headers.contains_key(header::AUTHORIZATION));
                Json(payload)
            },
        ),
    );
    let (url, task) = server(mock).await;
    let original = json!({"model":"TestBackend:org/model:8bit", "messages":[{"role":"user","content":"日本語"}],
        "tools":[{"type":"function","function":{"name":"read_file"}}], "stream":false, "extra_option":42});
    let r = call(
        app(config(&url)).unwrap(),
        "/v1/chat/completions",
        original.clone(),
    )
    .await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(r.headers()["x-model-router-backend"], "TestBackend");
    let mut expected = original;
    expected["model"] = json!("org/model:8bit");
    assert_eq!(value(r).await, expected);
    task.abort();
}

#[tokio::test]
async fn embeddings_and_decisions_use_their_own_paths() {
    let mock = Router::new().fallback(any(|request: Request| async move {
        let path = request.uri().path().to_string();
        let payload: Value = serde_json::from_slice(
            &axum::body::to_bytes(request.into_body(), 65536)
                .await
                .unwrap(),
        )
        .unwrap();
        Json(json!({"path":path,"payload":payload}))
    }));
    let (url, task) = server(mock).await;
    for endpoint in ["embeddings", "decide"] {
        let r = call(
            app(config(&url)).unwrap(),
            &format!("/v1/{endpoint}"),
            json!({"model":"TestBackend:model", "input":["日本語"],"kind":"choice"}),
        )
        .await;
        let data = value(r).await;
        assert_eq!(data["path"], format!("/v1/{endpoint}"));
        assert_eq!(data["payload"]["model"], "model");
    }
    task.abort();
}

#[tokio::test]
async fn malformed_unknown_and_unsupported_routes_fail_before_network() {
    let mut c = config("http://127.0.0.1:1/v1");
    c.backends.get_mut("TestBackend").unwrap().endpoints = vec!["embeddings".into()];
    for (payload, code) in [
        (json!({}), "invalid_model"),
        (json!({"model":"no:model"}), "unknown_backend"),
        (json!({"model":"TestBackend:model"}), "unsupported_endpoint"),
        (json!([]), "invalid_json"),
    ] {
        let r = call(app(c.clone()).unwrap(), "/v1/chat/completions", payload).await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(value(r).await["error"]["code"], code);
    }
}

#[tokio::test]
async fn upstream_errors_are_preserved() {
    let mock = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "7"), ("set-cookie", "secret=x")],
                Json(json!({"error":{"code":"busy"}})),
            )
        }),
    );
    let (url, task) = server(mock).await;
    let r = call(
        app(config(&url)).unwrap(),
        "/v1/chat/completions",
        json!({"model":"TestBackend:model"}),
    )
    .await;
    assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(r.headers()["retry-after"], "7");
    assert!(!r.headers().contains_key("set-cookie"));
    assert_eq!(value(r).await["error"]["code"], "busy");
    task.abort();
}

#[tokio::test]
async fn unavailable_and_stalled_upstreams_are_distinct() {
    let r = call(
        app(config("http://127.0.0.1:1/v1")).unwrap(),
        "/v1/chat/completions",
        json!({"model":"TestBackend:model"}),
    )
    .await;
    assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
    let (url, task) = server(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            tokio::time::sleep(Duration::from_secs(3)).await;
            Json(json!({}))
        }),
    ))
    .await;
    let mut c = config(&url);
    c.server.read_timeout_secs = 1;
    let r = call(
        app(c).unwrap(),
        "/v1/chat/completions",
        json!({"model":"TestBackend:model"}),
    )
    .await;
    assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
    task.abort();
}

#[tokio::test]
async fn model_catalog_prefixes_names_and_reports_partial_failure() {
    let (url, task) = server(Router::new().route(
        "/v1/models",
        get(|| async {
            Json(json!({"data":[{"id":"org/model:8bit","object":"model","context_length":8192}]}))
        }),
    ))
    .await;
    let mut c = config(&url);
    let mut broken = c.backends["TestBackend"].clone();
    broken.base_url = "http://127.0.0.1:1/v1".into();
    c.backends.insert("Offline".into(), broken);
    let mut static_backend = c.backends["TestBackend"].clone();
    static_backend.models = vec!["JEV-27B-VL".into()];
    c.backends.insert("M3U_JEV".into(), static_backend);
    let r = app(c)
        .unwrap()
        .oneshot(
            axum::http::Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let v = value(r).await;
    assert_eq!(v["data"][0]["id"], "M3U_JEV:JEV-27B-VL");
    assert_eq!(v["data"][1]["id"], "TestBackend:org/model:8bit");
    assert_eq!(v["data"][1]["context_length"], 8192);
    assert_eq!(v["router_errors"][0]["backend"], "Offline");
    task.abort();
}

struct DropMarker(Arc<AtomicBool>);
impl Drop for DropMarker {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn sse_is_incremental_and_downstream_drop_cancels_upstream_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let capture = dropped.clone();
    let mock = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let capture = capture.clone();
            async move {
                let stream = futures_util::stream::unfold(
                    (0, DropMarker(capture)),
                    |(i, marker)| async move {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Some((
                            Ok::<_, std::io::Error>(Bytes::from(format!(
                                "data: {{\"index\":{i}}}\n\n"
                            ))),
                            (i + 1, marker),
                        ))
                    },
                );
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let (url, task) = server(mock).await;
    let r = call(
        app(config(&url)).unwrap(),
        "/v1/chat/completions",
        json!({"model":"TestBackend:model","stream":true}),
    )
    .await;
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let mut stream = r.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first, "data: {\"index\":0}\n\n");
    drop(stream);
    tokio::time::timeout(Duration::from_secs(2), async {
        while !dropped.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
}

#[tokio::test]
async fn oversized_requests_are_rejected() {
    let mut c = config("http://127.0.0.1:1/v1");
    c.server.max_body_bytes = 8;
    let r = call(
        app(c).unwrap(),
        "/v1/chat/completions",
        json!({"model":"TestBackend:model"}),
    )
    .await;
    assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
