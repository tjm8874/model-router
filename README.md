# Model Router

A small Rust gateway for people who run **multiple local LLM servers**.
One OpenAI-compatible URL in your client; named backends in each model ID.

For example, keep OpenHuman Core and a small model on a **Mac mini with 64 GB RAM**,
and serve a larger model on a **Linux GPU PC**. Route everyday chat to the Mac mini
and reasoning or agent work to the Linux PC, without patching OpenHuman.
This is an independent community project, not an official OpenHuman component.

```text
OpenHuman Core on Mac mini
  └─ http://127.0.0.1:18080/v1  (Model Router)
       ├─ MacMini:your-small-model-id → local server :8080/v1
       └─ LinuxGPU:your-large-model-id → SSH tunnel → Linux PC :8000/v1
```

The gateway does not host or download models. The hardware example describes a
topology, not a promise that a particular model or context length fits in memory.
Choose and quantize models for your machines and leave room for Core, KV caches,
and concurrent requests.

## Build and run

Install the current stable [Rust toolchain](https://www.rust-lang.org/tools/install).
On macOS, install the Xcode command line tools; on Linux, install your distribution's
C compiler and linker (for example, `build-essential` on Ubuntu).

```sh
git clone https://github.com/tjm8874/model-router.git
cd model-router
cargo build --release --locked
cp examples/mac-mini-linux.toml router.toml
# Edit the backend URLs to match your running servers.
./target/release/model-router --config router.toml --check-config
./target/release/model-router --config router.toml
```

Use `--port 18081` to override the configured listen port. Configuration is read at
startup; validate it and restart the process after editing it. `--check-config`
checks syntax and settings, not upstream connectivity or model readiness.

## Mac mini + Linux PC example

1. Run a small model on the Mac mini with an OpenAI-compatible server such as
   MLX or LM Studio, listening at `http://127.0.0.1:8080/v1`.
2. Run a larger model with an OpenAI-compatible server such as vLLM on the Linux
   PC, listening on that PC's loopback port `8000`.
3. On the Mac mini, forward that port over SSH. Replace the example SSH host
   with your own; keep this tunnel running alongside the router:

   ```sh
   ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
     -L 127.0.0.1:18000:127.0.0.1:8000 user@linux-gpu.example
   ```

4. Use [the example router configuration](examples/mac-mini-linux.toml).
   Query the combined catalogue and copy the exact IDs it returns:

   ```sh
   curl http://127.0.0.1:18080/v1/models
   ```

   A backend ID `org/model:4bit` becomes `MacMini:org/model:4bit` in the catalogue.
   Only the first colon separates the backend; remaining colons and slashes
   are preserved. Missing backends are reported in `router_errors` alongside
   the available models. A partial catalogue is not proof that all servers work.

5. Test each route before assigning it in your client:

   ```sh
   curl http://127.0.0.1:18080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"MacMini:your-small-model-id","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
   ```

   Replace the placeholder ID. Repeat with `LinuxGPU:your-large-model-id`;
   add `"stream":true` to test streaming.

The same design works with two Linux hosts or several servers on one machine.
Backend names are configuration labels, not hostnames or URLs embedded in requests.

## OpenHuman integration

[OpenHuman](https://github.com/tinyhumansai/openhuman) accepts local OpenAI-compatible
providers. Point Core's `[local_ai] base_url` at `http://127.0.0.1:18080/v1` and
use its `local-openai:` provider prefix followed by the router's model ID:

```toml
# Top-level workload keys; put these before table headers.
chat_provider = "local-openai:MacMini:your-small-model-id"
reasoning_provider = "local-openai:LinuxGPU:your-large-model-id"
agentic_provider = "local-openai:LinuxGPU:your-large-model-id"

[local_ai]
runtime_enabled = true
opt_in_confirmed = true
provider = "lm_studio"
base_url = "http://127.0.0.1:18080/v1"
chat_model_id = "MacMini:your-small-model-id"
vision_model_id = ""
```

`lm_studio` is OpenHuman's existing local OpenAI-compatible settings identifier;
the actual server behind this URL is Model Router, not necessarily LM Studio.
See [the expanded workload example](examples/openhuman.toml). Merge these settings
into the **active Core user's** configuration, preserving unrelated settings.
If the UI runs on another computer, the URL above is still relative to the Core
machine. Workload assignments do not move the Core's memory or project storage.

`OPENHUMAN_LOCAL_INFERENCE_URL` and `LOCAL_OPENAI_URL`, if set in Core's environment,
can override `local_ai.base_url`; update or unset them when switching endpoints.
Configure image routes only for a vision-capable upstream model and register its
capabilities/context window in OpenHuman. Text-only models are not vision models.
Embedding and background-memory integration depends on the OpenHuman version
and its memory engine; this text-generation example does not move those engines
to the router. The gateway's `/v1/embeddings` endpoint can be used by clients that
support an OpenAI-compatible embedding service explicitly.

## API and configuration

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Router liveness only; does not check upstream readiness. |
| `GET /v1/models` | Combine upstream IDs with `backend:` prefixes; report partial failures. |
| `POST /v1/chat/completions` | Remove the backend prefix and forward JSON or SSE. |
| `POST /v1/embeddings` | Forward to a configured OpenAI-compatible embedding service. |
| `POST /v1/decide` | Forward to an existing Decisions API service; no API conversion. |

Each backend has a `base_url` (including `/v1`) and an `endpoints` allowlist.
Optional `models = ["exact-id"]` provides a static catalogue and model allowlist
for services without `GET /v1/models`. Listing a static model does not prove that
it is loaded. Optional `api_key_env = "VARIABLE_NAME"` reads an upstream bearer key
from the router's environment; do not put credentials in tracked files.

Unknown backends, empty model IDs, and unsupported endpoints return 400.
Connection failures return 502 and timeouts return 504 before response headers
are sent. Once streaming starts, upstream disconnects/timeouts interrupt the
stream instead of changing the HTTP status. Upstream HTTP error bodies/statuses
are preserved. JSON/SSE response bytes, tool calls, usage, and upstream model IDs
are passed through. There are no automatic retries or fallback models.

Incoming Authorization is not forwarded. Upstream authentication is configured
separately per backend. The router binds only to loopback and has no inbound
authentication; run it beside Core. Use an SSH tunnel or HTTPS for authenticated
remote upstreams. Routing a request to another machine transfers its prompt to
that machine, even though Core sees a loopback endpoint.

Logs contain backend/model/API names, HTTP status, and timing, not prompts,
answers, or bearer keys. The default 32 MiB request-body limit and timeouts are
configurable in `[server]`. `read_timeout_secs` is an idle-read timeout, not a
maximum total inference duration. HTTP proxy environment variables are ignored;
requests go directly to the configured backends. A downstream disconnect drops
the upstream response stream.

## Run continuously

Use your normal process supervisor to run the binary with an absolute `--config`
path and restart it on failure: a per-user LaunchAgent on macOS or a systemd user
service on Linux. Set credential environment variables in that service's
environment, and supervise SSH tunnels separately. Model Router does not manage
upstream processes or reconnect tunnels. Restart after configuration changes.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --release --locked
```

CI runs these checks and validates the example configuration on macOS and Linux.
Tests use local mock HTTP servers and cover request rewriting, authentication
separation, route validation, embedding/Decisions paths, partial catalogues,
upstream errors/timeouts, streaming/cancellation, and request-body limits.
No model downloads or external inference services are required for the tests.

## License

[MIT](LICENSE): free to use, modify, distribute, and use commercially, subject to
retaining the copyright and license notice. Dependencies retain their own licenses.
