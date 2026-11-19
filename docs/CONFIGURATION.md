# Configuration

A.I providers, Manga Cloud and the environment variables the app reads.

## AI setup

Provider keys are set in the settings UI or through environment variables:

~~~text
GEMINI_API_KEY
GOOGLE_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
EXPLABS_API_KEY
~~~

The settings UI discovers a provider's models through `/models`, caches listings cleanly per provider, selects exact model IDs, and registers custom OpenAI-compatible providers. Custom endpoints must be public HTTPS URLs using `protocol="openai"`, and credentials are never accepted inside the API URL.

### Built-in API Rescue Proxy (`/api/proxy/v1`)

Internal A.I calls (translation, A.I mode, Chaos agent) and local OpenAI-compatible tools can route through `http://127.0.0.1:8000/api/proxy/v1/chat/completions`. The proxy automatically classifies and recovers from transient 5xx errors, 413 oversized payloads, excessive tool/image counts, rejected optional fields, unsupported `stream`/`system` fields, context overflows, malformed JSON responses, and alternate `/chat/completions` paths, skipping 429/quota errors cleanly and learning auto-wrap rules only after a 2xx response succeeds.

- **Security boundaries**: `/api/proxy/*` accepts loopback same-origin requests only (`Sec-Fetch-Site`, `Origin`/`Referer`, and `Content-Type: application/json` enforced). Passing an explicit `X-Target-Base-URL`, `base_url`, or `target_base_url` strictly requires the caller's own `Authorization: Bearer <key>` and never reads keys from the app's secret store. Non-local traffic pins the validated public IP at socket connect time, and API secrets/query tokens are redacted from stored logs and error details.
- **Monitoring**: view recent rescue events and manage auto-wrapped `(api_base, model)` rules in **Settings → Proxy API** (or `GET /api/proxy/status`, `POST /api/proxy/unwrap`, `POST /api/proxy/logs/clear`).

### Manga Cloud (experimental, off by default)

Every feature, A.I mode included, is free with your own A.I key. Manga Cloud is for people without one: with `MANGA_TIERS=1` the app offers a Manga Cloud provider served by the gateway at `MANGA_CLOUD_URL`. Users sign in from the A.I mode panel with their email and a 6-digit code and top up a prepaid balance. Each A.I call of a chapter is charged at its real cost plus a 5 % fee (`GATEWAY_FEE_PERCENT`). A long webtoon chapter costs about $0.30, so a chapter needs about $0.32 of balance to start. Payment-provider fees are added to what the user pays, so the balance always gets the full top-up. The gateway stops a chapter at what the balance can pay for, and at $2 whatever the balance.

The gateway lives in `gateway/` and runs with `python -m gateway`. [DEPLOY.md](DEPLOY.md) puts it online behind Caddy with HTTPS and lists its limits and admin commands.

| Area | Variables |
| --- | --- |
| A.I upstream | `GATEWAY_UPSTREAM_BASE`, `GATEWAY_UPSTREAM_KEY`, `GATEWAY_UPSTREAM_MODEL`, `GATEWAY_PRICE_INPUT_PER_M`, `GATEWAY_PRICE_OUTPUT_PER_M`; for Azure OpenAI also `GATEWAY_UPSTREAM_AUTH=api-key` and `GATEWAY_UPSTREAM_MAX_TOKENS_FIELD=max_completion_tokens` |
| Login email | `GATEWAY_RESEND_API_KEY`, `GATEWAY_MAIL_FROM`, `GATEWAY_DEV_LOGIN=1` (shows the code instead of mailing it, local testing only) |
| Wallet | `GATEWAY_FEE_PERCENT` (5), `GATEWAY_CHAPTER_ESTIMATE_USD` (0.30), `GATEWAY_VND_PER_USD` (26000), `GATEWAY_TOPUPS_VND`, `GATEWAY_TOPUPS_USD` |
| payOS (VietQR) | `GATEWAY_PAYOS_CLIENT_ID`, `GATEWAY_PAYOS_API_KEY`, `GATEWAY_PAYOS_CHECKSUM_KEY`, `GATEWAY_PAYOS_FEE_PERCENT` (0); webhook `/v1/billing/payos/webhook` |
| Lemon Squeezy (cards) | `GATEWAY_LS_API_KEY`, `GATEWAY_LS_STORE_ID`, `GATEWAY_LS_VARIANT`, `GATEWAY_LS_WEBHOOK_SECRET`, `GATEWAY_LS_FEE_PERCENT` (5), `GATEWAY_LS_FEE_FIXED_USD` (0.50); webhook `/v1/billing/lemonsqueezy/webhook` |
| Other | `GATEWAY_DB`, `GATEWAY_ADMIN_KEY`, `GATEWAY_RETURN_URL`, `GATEWAY_HOST`, `GATEWAY_PORT`, `GATEWAY_TRUSTED_PROXIES` |

Signed-in users see their balance and every top-up and chapter charge, and can sign out on every device or delete their account from the A.I mode panel. A payOS top-up is credited once, when its webhook reports the exact amount. A Lemon Squeezy top-up is a one-time order at a custom price on one product variant; a refund takes the credit back.

## Runtime settings

Detection, inpainting, OCR and rendering thresholds are fixed constants in `app/parameters.py`. Only operational settings read environment variables:

| Area | Variables |
| --- | --- |
| Models & Processing | `MANGA_KIUYHA_MODEL` (default `kiuyha-tuned-1023.onnx`; also supports `kiuyha-tuned-s-1023.onnx` or `kiuyha-tuned-transient-1024.onnx`), `MANGA_PIPELINE_DEFAULT_WORKERS`, `MANGA_PIPELINE_SLICE_WORKER_LIMIT`, `MANGA_USE_DYNAMIC_LAMA`, `MANGA_INPAINT`, `MANGA_INPAINT_PRELOAD`, `MANGA_FIXED_LAMA_*` |
| ONNX Runtime | `MANGA_ORT_PROVIDER`, `MANGA_ORT_REQUIRE_PROVIDER`, `MANGA_ORT_INTRA_OP_THREADS`, `MANGA_ORT_OPENVINO_*`, `MANGA_ORT_CPU_MEM_ARENA`, `MANGA_ORT_MEM_PATTERN`, `MANGA_ORT_SERIALIZE_INFERENCE` |
| OCR | `MANGA_PPOCRV6_TIER`, `MANGA_PPOCRV6_TEXTLINE_ORIENTATION`, `MANGA_OCR_TARGET_SELECTION`, `MANGA_OCR_IMAGE_CACHE_MB`, `MANGA_OCR_JOB_CONCURRENCY_LIMIT`, `MANGA_OCR_JOB_ACTIVE_LIMIT` |
| AI Models & Timeouts | `GEMINI_VISUAL_QC_MODEL`, `GEMINI_VISUAL_QC_TIMEOUT_SECONDS`, `DEEPSEEK_VISUAL_QC_MODEL`, `MANGA_DEEPSEEK_TRANSLATION_MODEL`, `MANGA_MODEL_PRICES_JSON`, `MANGA_TRANSLATION_CONNECT_TIMEOUT_SECONDS`, `MANGA_TRANSLATION_READ_TIMEOUT_SECONDS` |
| Chaos Agent | `MANGA_AGENT_MODE` (set `0`/`off` to disable), `MANGA_AGENT_SANDBOX`, `AGENT_STATE` |
| Access & Server | `HOST`, `PORT`, `WORKERS`, `RELOAD`, `MANGA_ALLOWED_HOSTS` (extra host names, comma separated; loopback always works) |
| Network & Downloads | `MANGA_DOWNLOAD_WORKERS`, `MANGA_DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS`, `MANGA_REMOTE_CONNECT_TIMEOUT_SECONDS` |
