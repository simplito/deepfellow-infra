# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- New native-Kubernetes Helm chart at `deploy/helm/deepfellow` that deploys the whole suite — Infra, Server, Workspace, their MongoDBs, and optionally a vector database (Qdrant or the Milvus stack) and the FalkorDB knowledge graph — without Docker-in-Docker. Model backends (llama.cpp) run as ordinary Kubernetes Deployments and are registered into Infra as external `openai` service instances, one instance per backend, so Infra never needs access to a Docker socket. A provisioning Job reconciles the rest: it creates the admin user, organization and project, mints the project API key into a Kubernetes Secret (the key is returned only once, so the Secret is written before anything else and preserved on uninstall), registers each backend's service and custom model, waits for the model to become available, and only then grants the project access to it. The chart is deliberately fail-closed: it refuses to render until image digests (or an explicit mutable-tag acknowledgement) and credential sources are supplied, rather than silently deploying with defaults.
- New `DF_EXTERNAL_ONLY` mode for running Infra where no Docker daemon exists (e.g. as a plain Kubernetes pod). When enabled, startup performs no Docker work at all — no `docker` CLI lookup, no `docker compose version`, no GPU probe, no rootless check and no network subnet inspection — and only service types that talk to an already-running backend over HTTP are registered (currently `openai`). Docker-backed service and model operations return a deterministic `409` stating they are disabled by `DF_EXTERNAL_ONLY`, instead of failing later with an obscure error. Normal mode is unchanged.

## [0.35.0] - 2026-10-15

### Added

- The speaches-ai install form now has a `keep_alive` field: how many seconds an idle speech-to-text or text-to-speech model stays loaded in memory (`0` unloads it right after use, `-1` never unloads it). Left empty, speaches keeps its own default of 300 seconds. Keeping models loaded avoids the slower first request after a pause. An invalid value is rejected with a `400` naming the field, and on Edit before the running container is touched.
- The Infra Web Panel has an **Update** action in the actions menu of an installed model or service that has a Docker image version field. It asks for confirmation and applies the newest version through the existing edit endpoints, keeping all other settings. The newest version is chosen by Infra and always keeps the variant of the running tag (`-rocm`, `-cu129`, `server-vulkan-b<build>`, never an `-rc` or `-volta` build). A CUDA build (vLLM, SGLang) may move up to the CUDA version the host's NVIDIA driver supports, and keeps its CUDA major version when that cannot be read. A model or service already on the newest version only gets a message.
- `GET /admin/services/{id}/docker-tags` returns `newest`, the newest tag of the same variant as the running one, and accepts a `current` query parameter with the installed tag (the default tag when omitted). When an even newer CUDA build exists but needs a newer CUDA than the host's NVIDIA driver supports, `newest_needs_cuda` (`tag`, `required_cuda`, `host_cuda`) says so, and the WebUI warns that the NVIDIA driver has to be updated to use it.
- Optional Sentry error tracking for the backend, off by default: unhandled exceptions (the `500` responses carrying an `X-Error-Id`) are reported to a Sentry project, SaaS or self-hosted, marked as unhandled. It is configured only through `PUT /admin/config` or the WebUI Configuration tab, with the new `sentry_enabled`, `sentry_dsn` (masked like other secrets) and `sentry_environment` (empty means `production`) settings. Unlike other dynamic settings, they are never read from `DF_SENTRY_*` environment variables. Changes apply without a restart, and a change that failed to apply is retried on the next config save. A malformed DSN is rejected with `400`, also when enabling Sentry over one already stored. A malformed DSN already in `config.json` turns Sentry off with a warning instead of blocking startup, and doesn't block turning Sentry off or changing other settings. Each event is tagged with `error_id` (matching the response and the log line), the OTel `trace_id`/`span_id` when a traced endpoint failed, and `infra_name`, and carries the infra version as release and the instance `name` as server name. Events carry no stack-frame local variables, request data or PII, and the SDK's own `SENTRY_*` environment variables are ignored. Every secret value from the configuration (API and mesh keys, tokens, passwords, the Docker Hub token and the DSN) is replaced with `[Filtered]` before sending. Exception messages are still included and may quote request content, e.g. a provider error echoing a prompt. Sentry delivery problems (rejected key, rate limit, network errors) are logged, and enabling Sentry logs the DSN's host and project.
- The doc-chunker install form now has the settings for async conversion jobs: how the workers are split between async and sync jobs (`async_priority_workers`, `sync_only_workers`, next to `max_concurrent`), `max_queued_jobs` and `job_result_ttl`. A worker split doc-chunker would refuse to start with is rejected at install with a `400` naming the field; a field left empty falls back to doc-chunker's own default. doc-chunker's data directory (`/var/lib/doc-chunker`: registered tokenizers, async job uploads and results) is now kept on a volume, so it survives container recreation. `picture_description_prompt` is now required, as an empty value was sent to the vision model as an empty prompt. The new settings take effect only with a doc-chunker image that supports async jobs.
- `GET /v1/models` and `GET /v1/models/{id}` with `additional_data=true` now report `usage` (requests in flight: live for local models, near-live for models from child nodes, but only a snapshot from registration time for models a node receives from its parent/ancestors, since usage updates travel upward only) and `capacity` (total concurrency limit, `null` when any registration is unbounded or unknown) per model, summed across its registrations.
- Unhandled exceptions now return a `500` in the OpenAI error format (`type: "api_error"`) instead of a plain-text `Internal Server Error`, with an error ID in the message and in a new `X-Error-Id` header. The same ID is logged with the exception type, method and path, plus the OTel `trace_id`/`span_id` when a traced endpoint failed. If a streaming response had already started, only the log line is written.
- `GET /info` now reports `commit`, the commit the running image was built from (`null` when it is not baked in, e.g. a local run).
- Each instance can now set a container-name suffix (WebUI settings, or `PUT /admin/config`; lowercase letters, digits, `_` and `-`, at most 32 characters) so two DF Infra instances sharing a Docker host no longer collide on the same auto-derived container name. A brand-new install with no prior configuration gets a short random suffix automatically (`_` plus 6 hex characters), so two fresh instances don't collide even without manual setup; an existing install keeps a blank suffix, so nothing is renamed on upgrade. Container names are pinned in `services.json`, so changing the suffix only affects new installs and explicit updates/reinstalls - already-running services keep their current container names. The suffix now also covers the built-in `deepfellow-bge-m3`, `lemmatizer`, `gliner` and `doc_chunker` custom models, which previously always used a fixed container name. While the legacy `DF_CONTAINER_NAME_PREFIX` is set, it takes priority: setting a suffix is rejected, and those four models keep their fixed names. To drop the legacy prefix, start this version once with it still set (that pins the current names), then remove it and restart. With `DF_DOCKER_SUBNET` set, a suffixed container is reached by its container name, which its compose service is named after too, so instances sharing a subnet don't resolve each other's containers (unsuffixed and legacy-prefixed containers keep their hostname). A container whose hostname would exceed the 63-character DNS label limit is refused at install or startup with a clear error instead of every request to it failing later.
- `just check`/CI now verify that `example.env` documents every bootstrap-only `AppSettings` setting (a field not managed via `config.json`/`PUT /admin/config`), via a new `example-env-check` recipe and CI job. Fixed the drift it caught: `DF_DOCKER_SUBNET`, `DF_STORAGE_DIR`, `DF_STORAGE_SERVICES_DIR`, `DF_COMPOSE_PREFIX`, `DF_DOCKER_HUB_TOKEN`, and the three `OLLAMA_*` settings were missing from `example.env` entirely, and `DF_STANDARD_PROXY_TIMEOUT_SECONDS` was documented as a normal override even though it's dynamic config — setting it via the environment only has any effect once, before `config.json` first exists. The deleted comment's stated default (300) was also already stale — the real default has been 600 since `0.33.0`.
- Adding or editing a custom vLLM model now has a `quantization` field (e.g. `awq`, `gptq_marlin`, `bitsandbytes`), matching the field already available on built-in catalog models. Both the built-in and custom-model quantization fields are now a dropdown of this image's actual supported values instead of free text.
- New Finetune service (`finetune`) running the `deepfellow-finetune:0.0.18` image, an LLM LoRA fine-tuning framework with an OpenAI-compatible fine-tuning job queue and adapter registry. It replaces the `deepfellow-finetune` model of the Custom service (see Removed below). Installing the service starts its container in one step; there are no models to install separately. Each instance runs its own container with its own uploaded training files, configs and job database. The job database is now persisted on a volume, so job history survives container recreation. The HuggingFace base-model cache and the trained adapters directory are shared by all instances, so an adapter trained (or deleted) through one instance is visible to (or gone from) every instance. From `0.0.18` on, df-finetune never retrains an adapter that is already published when a job is requeued after a worker restart, so an adapter id keeps pointing at the same weights.
- `GET /admin/services/{service_id}/adapters` (admin-only) lists the fine-tuned LoRA adapters of a Finetune service instance, read from the running container's own `/v1/adapters`. The Finetune service page in the WebUI shows this list.
- vLLM models can now be served as a df-finetune LoRA adapter on top of their base model. Set the new `adapter_id` field to an adapter's id instead of `hf_id`: the base model and LoRA rank are resolved from the installed Finetune service's adapter registry, the adapter's safetensors files are downloaded, and vLLM is started with `--enable-lora`, `--lora-modules <model>=<adapter>` and `--max-lora-rank` (the adapter's rank rounded up to a value vLLM accepts). Requires the Finetune service to be installed. Downloaded adapter files are reused across installs while the registry still reports the same `created` timestamp for the adapter; if it changed (the adapter was rewritten under the same id), the files are downloaded again and the rank refreshed. If the registry is unreachable, e.g. at startup before the Finetune service has loaded, an already-downloaded copy is used as-is.
- WebUI unit tests now cover how install progress values are normalized for the progress bar: missing, `NaN`, infinite and non-numeric values are dropped, in-range values pass through unchanged, and out-of-range values are clamped to 0–1.
- The model list now reports whether each model supports vision (image) and audio input, resolved automatically whenever the model list refreshes: from Ollama's own `/api/show` capability report for Ollama models, and from the Hugging Face architecture (matched against a known, incomplete allowlist) for vLLM/SGLang models. For llama.cpp, the catalog only records whether a repo ships a companion vision-projector (`mmproj`) file - a hint that the base model *could* support vision, not confirmation that it does, since llama.cpp installs don't currently download or wire up that file - so llama.cpp models report supports_vision as unknown. Shown as unknown whenever it can't be determined (e.g. remote/OpenAI-compatible providers, an architecture not on the allowlist, or llama.cpp as above). For vLLM and SGLang, an already-installed model's reported values now stay in sync with a later catalog refresh instead of freezing at whatever was known when it was installed. The static `vllm-min.json`/`sglang-min.json`/`llamacpp-min.json` catalogs bundled with this release predate these fields and need a `just get-vllm-models`/`just get-sglang-models`/`just get-llamacpp-models` (or an in-app catalog refresh) re-crawl to actually carry them; until then, existing catalog entries report `supports_vision`/`supports_audio` as unknown.

### Removed

- The temporary DFFL-to-MIT migration step of the license header check (`scripts/check_license_header.py`): `--fix` no longer strips the old DeepFellow Free License prose header from `.py` files, as the transition period that began on 2026-08-14 has ended.
- The Coqui TTS service (`CoquiService`) and its model registry. It was unmaintained and unused. Installed Coqui instances are no longer managed by DF Infra; remove their containers manually.
- The `adapter_registry_url` setting (`DF_ADAPTER_REGISTRY_URL`, `PUT /admin/config`). The df-finetune adapter registry's address is now resolved live from the installed Finetune service, since the manually configured URL drifted from the container's dynamically assigned host port. An existing `config.json` is migrated automatically (schema version 2 → 3) and the key dropped. `adapter_registry_secret` is unchanged.
- The `deepfellow-finetune` model of the Custom service, superseded by the Finetune service. On the first startup after upgrading, an installed one is dropped from the Custom service's config and its container is removed. Its data under the Custom service's working directory is left in place but not migrated; install the Finetune service instead.

### Changed

- New installs of these services now use newer images by default: llama.cpp `b11641` (`server-cuda12-`, `server-vulkan-` and `server-` builds, was `b10902`), vLLM `v0.31.0` (`vllm/vllm-openai:v0.31.0-cu129-ubuntu2404` and `vllm/vllm-openai-cpu:v0.31.0`, was `v0.29.0`) and SGLang `v0.5.21-cu130` (was `v0.5.19-cu129`). Already installed models keep their image until updated with the **Update** action or the image version field.
  - **SGLang no longer has a CUDA 12 build**, since upstream publishes CUDA 13 images only from `v0.5.20` on. The new default needs an NVIDIA driver supporting CUDA 13.0 (R580 or newer) and doesn't run on Volta (e.g. V100) or older GPUs. On a host with an older driver, keep SGLang on `v0.5.19-cu129` through the image version field.
  - vLLM `v0.30.0` removed GPTQ act-order support: a GPTQ checkpoint quantized with `desc_act=True` (or a compressed-tensors one with `actorder=group`) now fails to start. None of the catalog's GPTQ models use it.
  - Some `extra_args` values accepted by the old SGLang image are now rejected: deprecated flags removed upstream (e.g. `--cuda-graph-bs`, `--cuda-graph-max-bs`, `--stream-output`) and `marlin` as a `--quantization` value.
- The generated model catalogs (`static/llamacpp-min.json`, `static/ollama-min.json`, `static/sglang-min.json`, `static/vllm-min.json`) are refreshed from their upstream sources.
- The vLLM and SGLang model catalogs now cover the top 200 models by likes (was 100), so popular models like `Qwen/Qwen-AgentWorld-35B-A3B` no longer drop out of a refresh, and no longer list CI test fixtures (`trl-internal-testing/*`, `peft-internal-testing/*`, `*tiny-random*`) or MLX builds (Apple silicon only), which cannot be served. Applies to `just get-vllm-models`/`just get-sglang-models` and to the in-app catalog refresh; the bundled `vllm-min.json` and `sglang-min.json` are regenerated.

- vLLM and SGLang models are now sized with `gpu_priority` (0.01 - 1.00) instead of an absolute `gpu_memory_utilization` fraction: the share of currently-free GPU memory to reserve for the model (free VRAM minus a 10% safety margin, capped at 90% of total), set with a slider in the install/edit form. It is not a weight normalized against other models' values, so installing one model no longer requires recomputing every other model's value. An explicit share takes precedence over a model definition's own default utilization, so 1.00 matches the default sizing only for models that don't define their own.
- The model list and the install/edit form of vLLM and SGLang models show a GPU allocation bar that previews the result. The form also warns about a share too small to reserve at least 1% of VRAM, but only when GPU stats are available; the backend rejects such a share regardless.
- A vLLM/SGLang model keeps its allocation once started: the share is applied when the model is installed, restarts reuse the saved allocation, and editing other options without moving the slider keeps the same amount of memory. Running models are never re-sized, as vLLM/SGLang fix `--gpu-memory-utilization`/`--mem-fraction-static` at process launch. The combined-utilization limit (422) still applies.
- Models installed with the old absolute `gpu_memory_utilization` key keep it, with its absolute meaning, including across later edits of their other settings, until a share is set for them; the edit form shows that legacy value. A stored `gpu_memory_utilization` of 0 still means "unset", and values outside 0-1 are now rejected.
- LoRA models no longer show a "Test" action in the WebUI model list, since a LoRA adapter can't be tested on its own (it needs a text-to-image model).
- The doc-chunker custom service now installs the `v2.4.0` image by default (GPU and CPU, was `v1.0.3`), the version df-server expects. A fresh install with default settings now provides the document-conversion endpoint df-server calls. The `v2.4.0` image is multi-architecture (`amd64` and `arm64`), so doc-chunker also runs on ARM hosts without picking a version manually.
- Every model-serving service (except Custom and MCP) has a new per-model `alias` option that registers an additional name for the model - the original name keeps working, and both names share the same usage and capacity. Both names appear in `/v1/models`. Aliases are not propagated through the mesh; peers see only the original name. **Note:** `alias` used to be an alternate spelling of `rename`, which this release drops - an `alias` value saved by an older version now adds a second name instead of replacing the original; use `rename` for that.
- The per-model `alias` option (available on every model-serving service except Custom and MCP) is now called `rename`, better reflecting that it fully replaces the model's registered name rather than adding a second one alongside it. `alias` is no longer accepted as a spelling of `rename`; it is now a separate option (see above).
- The Finetune service now runs the `deepfellow-finetune:0.0.20` image (was `0.0.18`). Existing installs switch to it on the next DF Infra restart, with no reinstall needed. That image requires a bearer token, so the worker container now gets `adapter_registry_secret` as its `DFFT_TOKEN`, and requests proxied through `/custom/<prefix>` are now authenticated to it with that secret. With an empty `adapter_registry_secret`, the worker runs unauthenticated as before (`DFFT_ALLOW_NOAUTH=1`), since otherwise it would refuse to start. After changing `adapter_registry_secret`, restart DF Infra so the worker container and the proxy pick up the new value. The container now runs as DF Infra's own user (`0:0` on rootless Docker) instead of the image's default uid 1000, and the per-instance `configs`, `data` and `db` directories are created by DF Infra before the container starts. Together these keep the bind-mounted directories writable by the container and removable by DF Infra. Other changes since `0.0.18` that clients may notice:
  - training files over 1 GB are rejected (HTTP 413);
  - a job's `training_file` must be an uploaded file ID, not a server path;
  - the job's `suffix` is sanitized into the adapter name;
  - a crashed worker's leftover training process is killed on restart;
  - adapters are published atomically;
  - jobs accept OpenAI-style `metadata`, which the adapter registry (`GET /admin/services/{service_id}/adapters`) also returns;
  - a job with an `eval_file` answers its prompts with the trained adapter after training and lists the answers in `result_files`, downloadable with `GET /v1/files/{id}/content`. DeepFellow Server's fine-tuning pipelines use this to score new adapters.
- **vLLM and SGLang models installed before this version may fail to start after upgrading**, depending on their `extra_args`. Each `extra_args` key and value now reaches vLLM/SGLang exactly as entered, as one argument each. Before, Docker Compose split them again on spaces and stripped quotes, and some `extra_args` only worked because of that. After upgrading, check the `extra_args` of each vLLM and SGLang model for these three forms and fix them in the model's edit form:
  - a flag and its value written together in the key, with an empty value (e.g. `"--tensor-parallel-size 2": ""`): move the value into the value field (`"--tensor-parallel-size": "2"`);
  - a JSON value wrapped in single quotes to work around the JSON bug fixed in this version (e.g. `'{"method":"mtp"}'`): remove the single quotes;
  - several values for one flag, separated by spaces (e.g. `"--cuda-graph-sizes": "1 2 4 8"`): these now reach vLLM/SGLang as one argument, which they reject, and `extra_args` has no other way to give a flag several values. Remove the flag to get the model running again.

### Fixed

- Two vLLM or SGLang installs started close together can no longer be given more GPU memory than is left. Downloads still run in parallel, but sizing a model's share and starting its container (until it is healthy) now run one install at a time, so each model is sized from a measurement that already includes the previous one and the second install is rejected with a `422` instead of failing when its container starts. A model installed without an explicit GPU share is also rejected with a `422` when almost no GPU memory is left, instead of being started with next to none.
- A doc-chunker installed with default settings (or via the API with an empty spec) can now describe images and transcribe audio (DFINFRA-392): an empty gateway address defaults to `http://infra:8086`, and an empty API key means this Infra's own `infra_api_key`, filled in only when the address is this Infra's (`http://infra:8086` or `infra_url`) and never stored in the spec or `services.json`. Existing installs pick it up on the next Infra restart, and so does a rotated key; a duplicated doc-chunker does not get the key automatically.
- The Docker image version list of llama.cpp (and any other image on GHCR or another OCI registry) now includes the newest builds. GHCR lists tags oldest first and `ggml-org/llama.cpp` has over 12 000, but only the first 2 000 were read, so the list stopped at build b5985 while the pinned default was b10902. Infra now reads up to 50 000 tags in pages of 1 000, keeps the 500 newest, and orders `-b<build>` tags by build number (`cuda13` builds no longer outrank newer `cuda12` ones). The first request takes a few seconds longer; the result is cached.
- Three background failures that used to leave no trace in the logs are now logged: the startup connection to the parent mesh node, the Ollama External model sync loop, and the MCP background tools fetch.
- An MCP installed on a mesh child node is now visible and usable through its parent. The child's health status is passed over the mesh and applied at the parent (the parent lists only healthy MCPs), and the child notifies the parent after each health check that changes the MCP's tools, transport or health. The parent's MCP proxy now forwards the full request path (it sent the generic MCP list URL instead of the specific server's), passes the `Accept`, `Mcp-Session-Id` and `WWW-Authenticate` headers, and uses an SSE proxy for MCPs running over SSE instead of always streamable HTTP. A child that predates this fix still shows its MCPs as unhealthy at the parent, so both nodes need updating.
- An MCP whose server stops working is no longer reported as healthy forever. Infra now re-checks every installed MCP every 30 seconds and hides it (here and on mesh parents) after 3 consecutive failed checks; it reappears on the first successful check. A crashed check or a pending OAuth authorization never marks an MCP unhealthy.
- vLLM and SGLang models installed with an `extra_args` value containing JSON (e.g. `--speculative-config {"method":"mtp",...}`) now start correctly: the container command is passed to Docker Compose as a list instead of a joined string, which Compose re-split with shell rules and stripped the JSON's double quotes. After upgrading, only containers whose resulting command actually changes are recreated; the rest keep running. Some `extra_args` that relied on the old splitting need changing; see the vLLM/SGLang entry under Changed.
- API keys and other auth secrets (the Server key, the admin key, the metrics username and password, the mesh key and the one-time sub-infra verification key) are now compared in constant time, so response timing no longer reveals how much of a guessed key was correct (DFINFRA-370). Accepted and rejected keys behave exactly as before.
- A sub-infra can no longer join the mesh when this instance has no mesh key configured (DFINFRA-370). An empty mesh key used to accept an empty `auth`, and the follow-up verification call goes to a URL the connecting client supplies, so any websocket client could register as a sub-infra and inject models into the registry. To accept sub-infras, set a mesh key (WebUI settings or `PUT /admin/config`).
- DF Infra no longer hangs (stuck at high CPU, not answering any request until restarted) after proxying a second upstream response without a body, such as a `204` from `DELETE /custom/doc_chunker/convert/jobs/{id}` (DFINFRA-396). The first such response worked; every later one made the proxy loop forever on an empty body.
- Models of remote services (OpenAI, Claude, Google AI, OpenAI-compatible custom endpoints and other cloud providers) now report the endpoints they actually accept in `props.endpoints` of `GET /v1/models` (DFINFRA-380). Embedding, speech-to-text, text-to-speech and image-generation models used to advertise the four chat endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`) instead of their own (`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/images/generations`), so clients selecting models by `endpoints` could not find an embeddings-capable model. Requests to those models already worked. Installed models pick up the corrected list on the next Infra restart.
- vLLM is no longer reported as unsupported on ARM hosts (including Apple Silicon) without a GPU. The AVX-512 requirement now applies only to x86 hosts.
- Doc-chunker conversions longer than 10 minutes are no longer cut off by the proxy (DFINFRA-372). doc-chunker sends its result only once the whole document is converted, so the proxy's idle-read window (`standard_proxy_timeout_seconds`, 600s) capped every conversion. A custom service can now declare its own default proxy timeout, used when an install leaves `proxy_timeout_seconds` empty (a value set at install still wins, and services without one keep using the global default). doc_chunker declares 7200s, matching df-server's `doc_chunker_convert_timeout_seconds` default (DFSERVER-184). Its per-document `document_timeout` default rises from 600s to 3600s. Existing doc_chunker installs that left `proxy_timeout_seconds` empty pick up 7200s on the next Infra restart; a `document_timeout` saved with the install (e.g. the form's old `600` default) stays until it is changed or the model is reinstalled.
- The brave-search MCP model catalog entry now sets `BRAVE_MCP_TRANSPORT=http`, fixing installation always failing: `brave-search-mcp-server` defaults to `stdio` transport, so the container exited immediately (exit code 0, no logs) instead of serving the `streamable_http` endpoint DF Infra expects on port 8080.
- Remote/OpenAI-compatible service instances (OpenAI, Claude, Google AI and OpenAI-compatible custom endpoints) now build correct request URLs regardless of whether the configured API URL includes a trailing slash or an explicit `/v1` (or provider-specific) version suffix. Previously a trailing slash on the API URL could silently double up the version segment (e.g. `/v1/v1/embeddings`), producing failed requests to the provider.
- Stable Diffusion Next's "not supported on macOS" restriction is now shown upfront in the services list (as a disabled reason), instead of only after clicking Install and getting a raw error.
- The web-search MCP model catalog entry now sets `proxy_transport="sse"`, fixing `tools/list` returning an empty tool list through the toolbox proxy: the entry never set `proxy_transport`, so it defaulted to `streamable_http` and DF Infra registered the proxy against `base_url + "/mcp"` — a path the container (which speaks SSE only, via `supergateway`) doesn't expose.
- Ollama VRAM estimates now use the `kv_cache_type` actually configured for the specific model instance instead of always assuming the gateway's default (`f16`), fixing inaccurate estimates for instances configured with a quantized KV cache (e.g. `q8_0`). The undocumented `OLLAMA_NUM_PARALLEL`/`OLLAMA_KV_CACHE_TYPE` environment variables, which previously could override the gateway-wide fallback used only when an instance didn't set its own value, have been removed — they were never real per-instance Ollama settings, just internal estimation defaults, and are now fixed fallback constants instead (`f16` for the cache type, matching Ollama's own default; `1` for parallelism, a deliberately conservative stand-in for Ollama's real "auto" behavior, not a reproduction of it).
- The DuckDuckGo MCP model catalog entry now pins `duckduckgo-mcp-server:0.6.1` (was `0.4.0`) and sets `DDG_ALLOWED_HOSTS` to the host DF Infra connects to it by, on any port (e.g. `duckduckgo:*`, or `localhost:*` without `DF_DOCKER_SUBNET`), fixing `tools/list` returning an empty tool list: the pinned version predated the MCP Python SDK's DNS-rebinding protection allow-list, so it rejected every request proxied over the internal Docker network (`Host: duckduckgo:8000`) with a silently-swallowed `421 Misdirected Request`.
- All app logging now goes through a single `logger` defined once in `server/utils/logger.py` and imported everywhere else, instead of ~45 modules each independently calling `logging.getLogger("uvicorn")` (and `tracing.py` doing it slightly differently from the rest) - both approaches reached the same handler, but only by relying on the same string literal being repeated correctly in every file.
- The release tag pipeline no longer fails when a tag is pushed right after merging to `main`: `tag_docker_image` now retries pulling the release image for up to 20 minutes instead of failing immediately, so it doesn't race ahead of `main`'s own `build_and_publish_docker_image` job.
- Retrying `PUT /admin/config` after a partial failure to reconnect the mesh, reconfigure OTLP log export, or disable tracing now actually retries what failed, instead of silently reporting success without reapplying it.
- Commands run inside a service's container (e.g. creating a custom Ollama model) now pass each value as a single argument instead of splitting on spaces, so a model name can no longer smuggle extra flags into `ollama create`, and file paths containing spaces work.
- Installing a custom Ollama model with a quantization option set on Ollama 0.34.1 or newer, where `ollama create -q` only works for safetensors imports, now fails immediately with a clear error when the `FROM` source is a GGUF file or a model referenced by name — before anything is downloaded — instead of downloading the source and then failing with a generic "Cannot install modelfile". An `ollama create` failure that does reach the command now translates Ollama's quantization errors (both the pre-0.34.1 "only supported for F16 and F32 models" and the newer "only supported for safetensors imports" / unsupported quantize type errors) into user-facing messages.

## [0.34.0] - 2026-09-16

### Fixed
- Reconfiguring OTLP log export via `/admin/config` no longer blocks the API for up to ~30s when the OTLP collector is unreachable, and a failed reconfigure no longer permanently disables further log-export retries.
- OTLP log export is now properly shut down when the app exits, giving buffered log records a chance to flush.
- Cancelling a service instance update no longer marks the instance as "not installing" while the automatic rollback to its previous configuration (see `0.32.0` below) is still running in the background. Previously the cancel request cleared the instance's tracking state as soon as the update's own reinstall task stopped, without waiting for the separate rollback it had just triggered - so the instance briefly looked fully uninstalled, and a concurrent install/update request could slip past the "already installing" guard and collide with the still-running rollback. The cancel request (and `POST /admin/services/{service_id}/cancel`) now only completes once the rollback (or the determination that reverting also failed) has actually settled, bounded by the same timeout already used elsewhere for this operation.
- The services list no longer lets a service be clicked into (or its Cancel button clicked again) while a cancel request - including any automatic rollback it triggered - is still being processed on the backend. It stays in the same non-clickable "installing" state already used for a plain install, instead of looking done the instant Cancel is clicked (or, for a service mid-rollback, momentarily flashing as "Installed" due to the services list's own stale cached data), and shows "Rolling back" instead of a stale "Downloading"/"Installing" label for that whole wait, when cancelling an update specifically triggers the rollback. A toast now also reports whether that rollback actually reverted the service or, having also failed, left it fully uninstalled - instead of only an optimistic "cancelled" message.

## [0.33.0] - 2026-09-11

### Added
- MCP server health checks and OAuth token refresh now emit OpenTelemetry metrics (`mcp.healthcheck.count`, `mcp.healthcheck.duration` in ms, `mcp.oauth.refresh.count`) and spans, for observability into MCP integration failures. Enabled by the existing `otel_tracing_enabled` flag — no separate metrics flag.
- Adding a custom vLLM model by HuggingFace id now checks upfront whether the repo can actually be served (GGUF/quantized repos, LLM repos without a usable chat template, and reranker repos with an unsupported architecture), returning a clear error immediately instead of only failing later when the model's container tries to start.
- vLLM, llama.cpp, and SGLang catalogs can now be refreshed at runtime instead of only via the offline `just get-vllm-models`/`just get-llamacpp-models`/`just get-sglang-models` release recipes: an admin can trigger `POST /admin/services/{service_id}/catalog/refresh` (or the model list's "Refresh catalog" WebUI action) to re-crawl HuggingFace for the current top trending/downloaded/liked models and reload the service's default model list, without a redeploy. Runtime refresh is only available when `static/` is writable; it shares a single in-flight HuggingFace rate-limit budget across all three services, and refuses to write a catalog that shrank suspiciously compared to the one it would replace.
- An in-progress service install or update can now be cancelled instead of only waited out: a new "Cancel" button in the WebUI (and `POST /admin/services/{service_id}/cancel`) stops the underlying Docker work and returns the instance to a clean, retryable state, matching the cancel capability model installs already had. Not offered for cloud services, which have nothing to cancel.

### Changed
- Default `standard_proxy_timeout_seconds`/`DF_STANDARD_PROXY_TIMEOUT_SECONDS` is now 600 seconds instead of 300, matching the default used by the OpenAI/Azure SDKs and LiteLLM, so long-running chat completions and other proxied responses aren't cut off prematurely.

### Fixed
- Setting or changing the Civitai, Hugging Face, or adapter registry token via `PUT /admin/config` (or the WebUI settings page) now takes effect immediately for model downloads. Previously the token was persisted correctly but downloads kept using the value captured at process startup until the backend was restarted.
- A custom endpoint (e.g. a document-processing service) installed only on a child node in a node-mesh now works correctly when accessed through a parent node's proxy. Previously the parent forwarded the request to the child with the model's prefix stripped from the URL, so the child could never route it and the request always failed — even though the mesh connection itself was healthy.
- An installed MCP server whose container only works over one transport (e.g. SSE) while DeepFellow had registered it on the other (streamable HTTP, the default) no longer silently returns empty tools/results. The periodic healthcheck already detected the working transport as a fallback probe, but only updated the transport shown in the UI — the live proxy stayed pointed at the broken one. It now re-registers the live proxy on the transport it actually found working.
- Cancelling a docker-compose-based install no longer leaves the underlying `docker compose` subprocess (and any container it already created) running orphaned in the background. `Utils.run_command` now kills the subprocess and waits for it to exit when the task awaiting it is cancelled, matching the pattern `DockerService.build_image` already used — this affects every command run through it, not just `docker compose`.
- Cancelling or otherwise failing an MCP model install after its Docker container has started now tears that container down and removes the in-progress install bookkeeping, matching the rollback already in place for vLLM, SGLang, and llama.cpp model installs. Previously MCP left both the container and the bookkeeping behind.
- A model install that adopts a pre-existing container and then fails at a later step (registration, capacity detection, etc.) no longer tears down that adopted container on rollback. The previous fix to make stop/restart/uninstall act on compose-file-less adopted containers (directly above) had the side effect of making install rollback destroy an adopted container too, since it now looked no different from an intentionally-installed compose-file-less service; rollback now only tears down what the failed install attempt itself created.
- Stopping, restarting, or uninstalling a service whose container was adopted rather than freshly installed (see the adopted-container fix in `0.32.0` below) now actually acts on that container instead of silently doing nothing or failing, since these operations no longer assume a compose file always exists.
- A service instance's "already installing" guard could be bypassed by a concurrent second install/update request: right after the guard check passed, the tracking field was reset to `None` for the entire duration of the slow pre-promise setup work, so a request arriving in that window slipped past the guard and clobbered the first install's tracking state. Affected 6 of 9 backing services (Ollama, llama.cpp, vLLM, SGLang, Coqui, Speaches AI) whose setup does real Docker image/registry checks before anything else. The same window also made the service-install progress endpoint spuriously 404 and made service listings report the instance as neither installed nor installing. The tracking object now has a two-phase lifecycle: reserved immediately (before any slow work begins) and resolved once the real promise exists. The install-progress endpoint now returns a `504` instead of hanging indefinitely if that pre-promise setup work is itself stuck.
- Editing a catalog model's install-time options (e.g. `gpu_memory_utilization`) or an installed custom model's definition, for services like vLLM whose model type carries no `default_prefix` (unlike MCP/Custom models), no longer crashes with a 500 `AttributeError`.
- Reranker/embedding registries (`static/vllm-min.json`, `static/sglang-min.json`) are no longer discovered by name search alone (`search=rerank`/`search=embed`), which missed any real reranker or embedding model whose repo name didn't spell out that word — e.g. `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `cross-encoder/ms-marco-MiniLM-L6-v2`. Candidates are now also discovered via HuggingFace's `pipeline_tag`, kept alongside the old name search; candidates found this way are filtered to exclude cross-encoders trained for a different task than reranking (e.g. `cross-encoder/stsb-*`, `cross-encoder/qnli-*`, `cross-encoder/quora-*`), which share the same architecture and tag as real rerankers but return meaningless rankings if served as one. A candidate whose HF detail lookup ultimately fails is now dropped instead of kept with fabricated data (`size: "N/A"`, incorrect `is_generative: true`). `just get-sglang-models` is also fixed — it always failed with `ModuleNotFoundError`.
- Retrying a model install while the original install is still running (e.g. after the client disconnects and reconnects) no longer reports instant fake success: the retry now waits for and reflects the real outcome of the in-progress install instead of a fabricated "already installing" response.
- Cancelling a service instance install/update (once a cancel path exists) would only have stopped a thin post-install bookkeeping wrapper, not the real work `_install_instance()` started (e.g. a `docker compose up` still in progress) — the reservation tracked the chained post-install promise instead of the real, unchained one, and cancellation only cascades down a chain, never up it. `InstallingInstance` now tracks both the real promise and the chained one separately, mirroring `InstallingModel`.
- A service instance update whose reinstall step failed (or was cancelled) used to leave the service permanently uninstalled, since its previous installation was already torn down and nothing restored it. Editing a service's settings now automatically reverts to the previous working configuration on failure, reloading its previously-installed models; the client isn't told the update failed until that revert has also concluded, and if the revert itself fails too, is told explicitly that the service is now fully uninstalled and needs a fresh install rather than a retry.

## [0.32.0] - 2026-09-01

### Added
- New `just refresh-generated-model-lists` recipe refreshes the four generated model registries (ollama, vLLM, llama.cpp, SGLang) in one run, in the style of `just check`. The five hand-curated lists in `static/` are deliberately left alone.
- Custom models added to vLLM and SGLang can now target a specific HuggingFace revision (branch, tag, or commit SHA) instead of only the repository's default branch.
- New `just get-llamacpp-models` recipe generates `static/llamacpp-min.json`, a registry of GGUF models discovered on HuggingFace for the llama.cpp backend, mirroring the existing `get-vllm-models`/`get-ollama-models` tooling. Every quant a repo ships is listed as a separate entry (not just one picked tier); multimodal projector files, split (multi-part) GGUF shards, multi-component pipeline files, non-chat repos (image/video diffusion, TTS, ASR, OCR), and embedding-only repos (including the `bge-`/`gte-`/`e5-` family prefixes, not just repos with "embed" in the name) are excluded since llama.cpp can't serve them as standalone chat models.
- New **Warnings** page in the WebUI (and `GET`/`DELETE /admin/warnings` API) surfaces failures that used to be visible only in the server logs: a service or model that fails to load, or a service present in `config.json` but no longer recognized in this build (e.g. after a downgrade), is now recorded as a dismissible warning. Warnings persist across restarts and are automatically cleared once the service or model successfully (re)loads, so they don't need to be manually dismissed once resolved.
- The sidebar's **Warnings** entry now shows a badge with the current warning count, so an unresolved failure is visible without opening the page.
- A service instance that fails to install or update is now also recorded as a warning, matching the existing behavior for a model that fails to install.
- New `just get-llamacpp-models` recipe generates `static/llamacpp-min.json`, a registry of GGUF models discovered on HuggingFace for the llama.cpp backend, mirroring the existing `get-vllm-models`/`get-ollama-models` tooling. Every quant a repo ships is listed as a separate entry (not just one picked tier); multimodal projector files, split (multi-part) GGUF shards, multi-component pipeline files, non-chat repos (image/video diffusion, TTS, ASR, OCR), and embedding-only repos (including the `bge-`/`gte-`/`e5-` family prefixes, not just repos with "embed" in the name) are excluded since llama.cpp can't serve them as standalone chat models.
- Claude and Google (Gemini) services can now refresh their model catalog from the provider's live model listing, matching existing OpenAI/Ollama behavior.
- Custom models added to vLLM and SGLang can now target a specific HuggingFace revision (branch, tag, or commit SHA) instead of only the repository's default branch.
- A subinfra connected to the mesh can now see and route to its parent's (and further ancestors') models, not just the other way around — both when it first connects and live as those models change — gated by a new `share_models_downstream` config option (on by default).
- A service instance that fails to install or update is now also recorded as a warning, matching the existing behavior for a model that fails to install.
- Installed MCP and custom services can now be edited in place via a new "Edit Settings" action, without manually uninstalling first — the backend transparently uninstalls, applies the new settings, and reinstalls. This also closes two previous edit gaps: catalog MCP models (e.g. `open-websearch`) can now have their prefix/envs/headers edited, and plain docker-image MCP servers (not `user`/`proxy` kind) can now be edited at all.
- Added a "Duplicate" option to the Edit dialog for MCP and custom service models, letting an admin create an independent copy of a model's settings under a new id and prefix for side-by-side comparison, without affecting the original.
- New `standard_proxy_timeout_seconds`/`DF_STANDARD_PROXY_TIMEOUT_SECONDS` setting (default 300) controls the timeout for outbound proxy HTTP requests to backend model/tool services, previously hardcoded to aiohttp's unconfigurable 300-second default. It's used as a total-duration cap for chat completions, embeddings, audio, images, rerank, and MCP proxying, and as a per-chunk idle-read window (no total cap) for custom-service proxying (doc-chunker, GLiNER, etc.). Live-editable via `/admin/config` and the WebUI settings page, not just `.env`.
- GLiNER named-entity-recognition is now installable as a custom service (CPU/GPU), exposed at `/custom/{prefix}/extract_entities` for knowledge-graph entity extraction.
- Built-in custom-service and MCP models (doc-chunker, bge-m3, lemmatizer, finetune, open-websearch, brave-search, etc.) now offer a Docker image version selector in their install form, letting an admin pick a specific tag instead of always getting the pinned default. Also fixes tag listing for images hosted on `hub.simplito.com`, which previously fell through to the wrong (Docker Hub) API.
- Custom-service installs can now set their own proxy read timeout via a new `proxy_timeout_seconds` field on the install form, overriding the global `standard_proxy_timeout_seconds` default for that specific service.
- New `webui2/` directory installs the DeepFellow Dashboard as an alternative web UI, as a proof of concept. `webui/` still builds the UI the image serves, and nothing about the default image content changes: the `Dockerfile` gained a `UI_VARIANT` build argument that defaults to `legacy`, and `DOCKER_BUILDKIT=1 docker build --build-arg UI_VARIANT=dashboard --secret id=gitlab_npm_token,src=<a file holding a registry token> .` selects the Dashboard instead. BuildKit is required for both builds now. New `just ui2-rebuild`, `just ui2-clean`, `just ui2-check-ignores`, `just ui-restore` and `just dashboard` recipes drive it. A `build_dashboard_variant_image` CI job builds the Dashboard variant on every pipeline and reads what came out of it, and a `check_dashboard_ui_ignores` job verifies that `.gitignore` still covers every file name that build emits. Both drop themselves while the `GITLAB_NPM_TOKEN` CI/CD variable is unset. See `webui2/README.md`, and `webui2/MIGRATION.md` for what adopting it would take.

### Changed
- The doc-chunker custom service install form now matches the reworked doc-chunker: image description and audio transcription share a single "inference gateway" address and optional API key instead of separate URL/key/model fields for each, new **LibreOffice conversion timeout** and **parallel image descriptions** settings were added, and the picture-description mode selector along with the preset/local HuggingFace options (and the HuggingFace token field) were removed since the service no longer supports them. Services installed with the old form keep their old settings until reinstalled or edited.
- `.gitignore` and `.dockerignore` now also list the file names the DeepFellow Dashboard build emits, beside the ones the legacy web UI emits, so neither UI's output can be committed or reach a build context by accident. Everything else under `static/` stays tracked, exactly as before.
- Model request routing now prefers already-warm, higher-capacity instances and packs traffic onto the fewest of them before spilling over, instead of always spreading load evenly across every registered instance regardless of how much concurrent load each can actually take.
- Model registries for llama.cpp, Coqui, rerank, Speaches, and Stable Diffusion are now loaded from `static/*-min.json` files instead of hardcoded in the service code
- Previously installed models keep working after the maintained registry list is refreshed or pruned.
- llama.cpp's default model list now loads from `static/llamacpp-min.json` at startup instead of being hardcoded in source, matching how vLLM and Ollama already manage their default model lists.
- vLLM service's hardware-support guard and GPU-utilization release logic are each backed by a single shared implementation instead of duplicated copies, removing the risk of the copies drifting apart on future changes.
- `.gitignore` and `.dockerignore` now also list the file names the DeepFellow Dashboard build emits, beside the ones the legacy web UI emits, so neither UI's output can be committed or reach a build context by accident. Everything else under `static/` stays tracked, exactly as before.
- Model request routing now prefers already-warm, higher-capacity instances and packs traffic onto the fewest of them before spilling over, instead of always spreading load evenly across every registered instance regardless of how much concurrent load each can actually take.
- SGLang instances now report their KV-cache-sized `max_running_requests` from startup logs as routing capacity, same as vLLM, so capacity- and warmth-aware routing packs traffic onto them instead of treating them as unbounded.
- Ollama's "Maximum parallel requests" field default in the WebUI is now `1` instead of `3`, matching the actual fallback (`OLLAMA_NUM_PARALLEL` unset defaults to 1) - a newly created Ollama instance that doesn't override this field now gets `OLLAMA_NUM_PARALLEL=1` instead of `3`.
- Warmth-aware routing now correctly tracks a backend's loaded/evicted model state from right after instance install, treats a backend with no reported concurrency number as unknown capacity (ranked last, never counted as saturated) rather than unbounded, treats Ollama's `OLLAMA_NUM_PARALLEL=0` ("auto") as bounded at DeepFellow's configured `OLLAMA_NUM_PARALLEL` default (conservatively `1`, not Ollama's own real auto-scaling behavior of up to 4 concurrent requests depending on available VRAM) rather than zero capacity, no longer registers a freshly-installed Ollama model as warm before it has actually loaded, no longer downgrades a mesh peer's genuinely unbounded backend (e.g. a cloud/proxy model with no concurrency cap) to "unknown capacity" on receipt, rejects a llama.cpp instance's invalid `num_parallel` (0 or negative) at both setup and update time instead of failing later during model registration and leaving an orphaned Docker container behind, no longer orphans a llama.cpp model's Docker container when a failure occurs between starting it and registering it, and no longer lets a misconfigured `OLLAMA_NUM_PARALLEL=0` in DeepFellow's own environment (as opposed to a per-instance override, which was already handled) silently fail every Ollama model install on that instance.

### Fixed
- `GET /v1/models` now reports `props.context_window` and `props.max_context_window` as `null` for a model whose registrations declare no context window, instead of `0`. Consumers use these as a ceiling for their own request sizing, and a reported `0` made them size against an impossible window — it broke DeepFellow Server's vector store chunking outright.
- `deepfellow-bge-m3` now declares its 8192-token context window.
- A zero or negative context window arriving from gguf metadata, a model directory's `config.json`, remote-model admin fields or Ollama form fields is now treated as "unknown" rather than published as a real window.
- Custom-service proxy requests (doc-chunker and other custom services, plus the mesh and Stable Diffusion custom-endpoint proxies) no longer have their total request duration capped at a flat 5 minutes; a backend that streams partial output now resets a per-chunk 5-minute window each time it sends data (15 minutes for doc-chunker and other custom services), so a slow-but-still-responding backend is no longer cut off just for taking a long time overall.
- Chat completions requesting JSON mode via `response_format={"type": "json_object"}` (without an explicit `json_schema`) are no longer rejected with a 422 error.
- vLLM service now logs a debug message when it can't resolve a custom model's size or can't determine its VRAM usage from container logs, instead of silently returning `None`.
- vLLM model install no longer silently swallows a failed Docker container stop during cleanup — the failure is now logged so an orphaned container can be found and removed.
- vLLM model installation no longer gets stuck permanently in "installing" state if the request is cancelled during a graceful shutdown while GPU/quantization checks are still running.
- vLLM's KV-cache-overflow retry logic no longer intercepts unrelated Docker startup failures; only the specific "estimated maximum model length" error now triggers a retry with an adjusted `--max-model-len`.
- Uninstalling an Ollama model (without purging) now unloads it from VRAM; previously it kept occupying VRAM until the whole Ollama container was restarted.
- Persisted model definitions no longer get wiped when a model drops out of the live registry (e.g. after a catalog refresh): the config snapshot now falls back to the last-persisted definition instead of overwriting it with `null`.
- Backfilling missing `definition` fields into `config.json` on startup no longer drops a model that failed to come up on that boot (transient error): the regenerated config now preserves that model's last-persisted entry instead of omitting it, so it keeps getting retried on future loads.
- A model definition snapshot from an older schema no longer aborts loading the rest of an instance's models: restoring it is now covered by the same error handling as the model install itself.
- A model that fails to (re)load is no longer permanently dropped from `config.json` the next time any unrelated action (installing another model, uninstalling a model, editing a custom model, etc.) triggers a config save — it stays persisted so it keeps getting a retry on every future load, until it's explicitly uninstalled.
- Refreshing the Ollama catalog no longer makes an already-installed model disappear from `list_models`/`get_model` at runtime when it has dropped out of the static and dynamic catalog: the persisted definition is now re-applied after the catalog rebuild, matching the existing restore-on-load behavior.
- Fixed the Ollama install progress bar getting stuck at 100% forever when a service install's post-processing step (e.g. saving config) failed after an install/uninstall/reinstall cycle.
- Rotating `infra_api_key` no longer leaves already-connected subinfras calling ancestor-proxied models with the old key: a proxy registration is now refreshed whenever the reporting peer's API key changes, not only when the model id itself is new.
- Fixed a `pyright` type-check failure on `main` caused by Docker Model Runner's `_generate_instance_config` override not matching its base class signature.
- Installing a service no longer fails outright when a Docker container with the same name already exists as an orphaned leftover from a previous, no-longer-tracked Infra instance (e.g. after wiping local state without running `infra uninstall`): if the leftover container is confidently the same healthy service (matching image, health, port, env vars, and volumes), it's adopted instead of blocking the install. An adopted vLLM/SGLang model, which has no persisted capacity to reuse (that prior instance's state is gone) and isn't actually restarted during adoption, now falls back to reading its still-running container's startup logs to recover the real concurrency capacity instead of leaving it permanently unknown.
- A HuggingFace model download with an invalid or unreachable repository id no longer silently reports install success with an empty model directory; it now fails fast with a "model repository not found" error instead of a confusing container-startup failure minutes later.
- Installing a rerank model on a rootful Docker setup no longer fails with `PermissionError`: the HuggingFace cache directory bind-mounted into the rerank container is now created by the server before the container starts (previously the Docker daemon created it as `root:root`, so the host-side model download couldn't write into it), and the container itself now runs as the server's user instead of root, so it no longer leaves root-owned files in that shared cache.
- `POST /v1/rerank` no longer returns 500 now that the rerank container runs as the server's user. The container's HuggingFace cache moved from `/root/.cache/huggingface` (unreachable for a non-root user, since `/root` is only traversable by root) to `/mnt/hf`, set via `HF_HOME`, and `USER` is now set in the container because torch's inductor cache derives its path from `getpass.getuser()`, which raised `KeyError: getpwuid(): uid not found` for a uid with no `/etc/passwd` entry in the image. The failure surfaced only at model-load time, so the container's healthcheck kept reporting healthy while every rerank request failed.
- Docker image builds now honor the `exclude-newer` dependency-resolution pin again. The build stage pinned uv 0.8.12, which cannot parse the `"1 week"` duration syntax in `[tool.uv]`, so it warned "Failed to parse pyproject.toml during settings discovery" and silently ignored the entire table — including the `required-version` floor meant to catch this exact mismatch. The pin is now uv 0.11.8, matching the version the lint and test jobs already use.
- A custom-service model could occasionally start two concurrent install attempts if requested twice in close succession; the model is now marked as installing before its Docker image is verified, closing that window.

## [0.31.0] - 2026-08-06

### Added
- New `POST /admin/mcp/convert-config` endpoint converts a standard MCP client JSON config (`{"mcpServers": {...}}`) into DeepFellow's custom-model parameters, so the CLI and other clients can reuse the same conversion the WebUI's "Auto-Import" tab already performed client-side.
- New `just get-vllm-models` recipe generates `static/vllm-min.json`, a curated registry of top HuggingFace LLM, reranker, and embedding models for the vLLM backend, mirroring the existing `get-ollama-models` tooling. Retries with backoff when the HuggingFace API rate-limits requests. Reranker candidates are filtered to models with a `*ForSequenceClassification` architecture, the only kind vLLM can serve as a cross-encoder — models merely named "reranker" but built as generative or custom-ranking architectures (e.g. Qwen3-Reranker, jina-reranker-v3) are excluded since they fail to load in vLLM.
- Read-only smoke test suite (`tests/bruno/`) that checks availability, docs, reported version, the admin and OpenAI-compatible read endpoints, and auth rejections over real HTTP against a running instance. It runs after every deploy to dev (main, hotfix branches and release tags); on a release tag it is a blocking gate, so a tagged build that fails it is never published to GitHub. Run it locally with `tests/bruno/run-local.sh`.
- New `GET /info` endpoint, authorized with the server key (`DF_INFRA_API_KEY`), returns the running Infra version (`{"version": "..."}`) read from `pyproject.toml`.
- Ollama service now has a "↻ Refresh catalog" button that fetches trending models from the Ollama library and merges them into the model list as a dynamic overlay, so newly released models show up without waiting for an app update. Each click always fetches fresh results; the 6-hour cache only applies to callers that don't pass `force=true`.
- Ollama Cloud support with cached model list.
- Most infra settings (mesh connection, API keys, MCP session limits, OTEL tracing/logging, etc.) are now stored in `config.json` and can be changed from the Configuration page (or the `/admin/config` API) without restarting the server. Only bootstrap settings still require `.env` and a restart.
- Ollama, llama.cpp, and vLLM service install and edit dialogs now include a searchable **Docker image version** selector populated from the container registry, filtered by the selected hardware variant. Leave the field empty to use the default bundled version.
- New `GET /admin/services/{id}/docker-tags` API endpoint returns available Docker image tags for a service, with server-side caching (2 h TTL) and optional `?hardware=` filtering.
- Optional `DOCKER_HUB_TOKEN` environment variable for authenticated Docker Hub access to raise rate limits when fetching image tags.
- Model download progress is now logged server-side (every ~10%), covering both the generic model downloader (HuggingFace, Civitai, adapter registry, direct URLs) and Ollama's native `/api/pull`, so real download progress is visible in the console instead of only the UI.
- New Claude models: Opus 4.7, Opus 4.8, Sonnet 5 and Fable 5.
- New OpenAI models: GPT-5.4 Pro, GPT-5.5, GPT-5.5 Pro, and the GPT-5.6 family (Sol, Terra, Luna).
- New DeepSeek service (`deepseek-v4-flash`, `deepseek-v4-pro`).
- New Kimi service (`kimi-k2.6`, `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`).
- DeepSeek V4 (Flash, Pro) and Kimi (K2.6, K2.7 Code) added to the self-hosted vLLM model registry.
- New Google AI (Gemini) chat models: Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.5 Flash-Lite, Gemini 3.1 Flash-Lite, Gemini 3 Flash (preview), and Gemma 4 (26B-A4B, 31B).
- New Google AI embedding and image models: Gemini Embedding 2, the Nano Banana 2/2-Lite/Pro image models, and GA Imagen 4.0 Standard/Ultra.
- Remote (proxy) MCP servers now support OAuth 2.1 authorization (PKCE): DeepFellow auto-detects when a server requires OAuth, walks discovery/registration per the MCP Authorization spec, and exposes an "Authorize" action in the WebUI; tokens are refreshed automatically in the background.
- New SGLang service (modeled on the vLLM service) for serving LLM, embedding, and reranker models. Rerank responses are normalized from SGLang's native array shape into the Cohere-style `{"results": [...]}` shape the rest of the API expects.
- New Docker Model Runner service (modeled on the ollama-external proxy pattern) for serving LLMs via Docker Desktop's built-in Model Runner (`vllm` / `vllm-metal` backend), including GPU-accelerated inference on Apple Silicon via Metal without Ollama. Configurable base URL (default `http://localhost:12434`); models install from a curated built-in catalogue (`static/docker-model-runner-min.json`) and proxy chat/embeddings like other services. Requires Docker Desktop 4.40+ with Docker Model Runner enabled; the `vllm-metal` backend requires macOS on Apple Silicon.

### Changed
- Direct (non-dev) dependencies still in the `0.x` series are now pinned with `~=` instead of `>=` in `pyproject.toml`, so `uv lock` can only pick up patch-level updates for them, not minor bumps that may break compatibility.
- The project now licenses under the MIT License. Python files require an `SPDX-License-Identifier: MIT` header instead of the previous DeepFellow Free License copyright block; `just license-check --fix` migrates old headers automatically.
- `OllamaChatMessage.role` is now a free-form string instead of a `Literal`, matching Ollama's native API, which does not validate role names. Required by models that define their own role vocabulary — e.g. Granite Guardian 3.3 passes RAG documents as `document` (and `document <id>` for multiple documents) when checking groundedness and context relevance.
- Container healthcheck (`scripts/healthcheck.py`) now probes `/health` instead of `/docs`, consistent with every other health probe.
- Bumped bundled llama.cpp Docker image from `b9894` to `b10068` (latest published image build). Known risk: two upstream vulnerabilities remain unpatched as of this release — CVE-2026-2069 (GBNF grammar stack overflow) and an unpatched GGUF `general.alignment` integer overflow — both were already present in the previous `b9894` pin and are unrelated to this bump; mitigate by not accepting untrusted GGUF files or exposing GBNF grammar sampling to untrusted input until upstream ships a fix.
- Bumped bundled Ollama Docker image from `0.31.1` to `0.32.1` and vLLM from `v0.24.0` to `v0.25.1` to stay current with upstream releases. No known outstanding security advisories affected the previous pins.
- Bumped bundled open-websearch MCP Docker image from `v1.2.0` to `v2.1.9` to stay current with upstream releases.
- Ollama VRAM usage for loaded models is now read directly from Ollama's `/api/ps` response instead of being parsed from container logs.
- Python files under `ee/` now require an `SPDX-License-Identifier: LicenseRef-DeepFellow-Free` header instead of MIT; `just license-check` rejects a stray MIT header in `ee/`, and `--fix` migrates it automatically.
- Coqui TTS now runs from `idiap/coqui-ai-TTS`, an actively maintained community fork, instead of the original `coqui-ai/TTS` repo, which has had no upstream activity since August 2024. Also fixes the bundled CPU/GPU Docker images being swapped (the "CPU" option was pulling the GPU-capable image and vice versa).

### Fixed
- Custom model installation for SGLang and VLLM now correctly uses model's hf_id instead of model_id.
- speaches-ai model installs no longer fail with a `PermissionError` on a fresh install when running the backend as a non-root user (e.g. via `just dev`). The container now runs as the current host user instead of always as root, so the cache directory Docker creates on first start is writable by the backend.
- `POST /v1/embeddings` Request schema now also accepts token-array `input` and restricts `encoding_format` to `base64`/`float`.
- `/v1/responses` no longer fails a pyright type check by passing an unawaited coroutine as request headers.
- vLLM model install no longer fails outright when a model's default context length needs more KV cache than is actually free (e.g. `ValueError: ... estimated maximum model length is 110256`). If the user didn't request a specific `max_model_length`, install now retries once with the length vLLM itself suggests instead of leaving the model unusable.
- Infra not resolving `:latest` tags from /api/ps Ollama endpoint for VRAM calculations.
- `scripts/get_ollama_models.py` no longer captures raw scraped HTML markup as a model's `size` when the Ollama library page renders a "usage slot" widget instead of plain size text (seen for at least one cloud-hosted model, `gemini-3-flash-preview`); the captured text is now validated against a size-string pattern and discarded otherwise. Also corrected the already-affected entry in `static/ollama-min.json`.
- vLLM now detects when an already-installed model's Docker container stops running for good (crash, OOM, manual `docker stop`, hung process reported `unhealthy`) and releases its reserved GPU memory automatically, without requiring a backend restart. Previously the GPU-utilization counter stayed occupied forever for a dead container, blocking further model loads.
- llama.cpp now detects a crashed/killed model container (crash, OOM, `docker stop`, `unhealthy`) and releases its VRAM automatically, without a backend restart — matching the vLLM fix. Previously the model stayed reported as loaded, blocking further installs.
- `config.json` is now written to the storage directory instead of the app directory, so dynamic settings (mesh key, API keys, etc.) survive container recreation instead of resetting on every redeploy.
- OTEL trace/log exporter endpoint changes now take effect immediately instead of silently keeping the old endpoint until a restart.
- vLLM now releases reserved GPU memory and stops the model container on every model-install error path (option parsing, download, container start, post-start setup, and endpoint registration), not just on a Docker `RuntimeError`. Previously a failure after the container started left it running and kept the GPU-utilization budget occupied, blocking further model loads until a full service restart.
- VRAM estimation no longer crashes the entire model listing endpoint when a model (e.g. Qwen3.5, Gemma4) reports `null` for `num_key_value_heads` in its Ollama architecture metadata. Incomplete architecture metadata is now handled in the calculator itself: a missing or zero KV-head count falls back to standard MHA (`head_count_kv = head_count`, matching llama.cpp's own default) and a missing embedding length or layer count reports the estimate as unavailable, instead of raising a `TypeError` that surfaced as HTTP 500.
- VRAM estimates for models whose metadata omits `num_key_value_heads` are no longer understated. The previous fallback assumed a single KV head, which shrank the computed KV cache by the model's full GQA ratio (up to 32× for a 32-head model); the MHA fallback now reports the real cache size.
- llama.cpp models whose GGUF metadata omits `attention.head_count_kv` now get a VRAM estimate instead of none at all — the field is no longer treated as mandatory when parsing GGUF architecture parameters.
- The WebUI model list now shows an error message with a Retry button when the models endpoint fails, instead of silently rendering an empty "No models found" table that was indistinguishable from a service with no models. Failed VRAM estimates are also logged (at debug level) rather than being suppressed without a trace.
- A transient failure to reach Ollama's `/api/ps` no longer reports a still-loaded model as unloaded with a lower-fidelity VRAM estimate.
- Installing an MCP server without a required API key/header (e.g. brave-search without `BRAVE_API_KEY`, ollama-websearch without `OLLAMA_API_KEY`) no longer shows a fake download that runs to completion without installing anything. The missing field is now flagged inline in the install form before submit, and the backend rejects the request with a clear error listing the missing keys.
- The Cancel button for an in-progress model installation in the WebUI now appears in the model's table row, so it stays available after refreshing the page. Previously cancelling was only possible from the bottom progress toast, which disappeared on refresh and left the installation with no way to cancel.
- API errors now return an OpenAI-style `invalid_request_error` body with `message` and `param` identifying the offending field, instead of a bare `{"detail":"Bad Request"}` with no diagnostic detail.
- `/v1/models` now reports each model's real registration timestamp and owning service in `created`/`owned_by` instead of always `0`/`"unknown"`.
- `/v1/responses` now resolves `item_reference` inputs pointing at a stateful item (e.g. a reasoning item) this gateway emitted earlier in the conversation, instead of rejecting it (Ollama-backed models) or crashing with a 500 (OpenAI-backed models).
- `/v1/responses` no longer rejects function tools whose parameter schema has an optional (not-`required`) property — `strict` now defaults to `false` instead of `true`.
- `/v1/responses` no longer rejects `function_call`/`function_call_output` history items on the OpenAI-backed path: a duplicated `type` discriminator on `CustomToolCall` was colliding with `FunctionToolCall`'s, making the union ambiguous for every `function_call` item. `id`/`call_id` also no longer share one value generated once per process, so tool-call history replays byte-for-byte.
- `/v1/models?additional_data=true` now reports the correct endpoint list for Ollama, llama.cpp, and vLLM models; a malformed `LLM_ENDPOINTS` entry had merged two endpoint paths into one string and dropped the leading `/` used by every other endpoint.
- The vLLM and SGLang model registries (`just get-vllm-models` / `just get-sglang-models`) no longer omit every multimodal model. HuggingFace tags vision-language chat models as `image-text-to-text` rather than `text-generation`, so the scraper never even requested them, while community re-uploads of the same weights that happened to be tagged `text-generation` were present. Single-purpose OCR models pulled in by the new tag are excluded, since they are text extractors rather than chat models.
- The scraper's non-LLM name filter no longer discards valid models over a substring match: `ner` matched inside `ForConditionalGeneration` (every multimodal repo) and inside author names, so those models were dropped before any other check ran. Task words are now matched on word boundaries.

### Changed
- Stable Diffusion service to Stable Diffusion Next
- Chnage errors with Stable Diffusion to contain Stable Diffusion Next

## [0.30.0] - 2026-06-26

### Added
- MCP and custom service models now display a short description and an optional repository link below their name in the WebUI.
- All built-in MCP servers and custom services now have descriptions and repository URLs populated in their definitions.
- User-defined MCP servers (Command and Remote URL) support optional Description and Repository URL fields when added through the WebUI.

### Fixed

- Fixed VRAM estimation to use the correct bits-per-weight for each specific GGUF quantization variant (e.g. Q4_0 vs Q4_K_M) instead of a single per-model-family value, and applies a small per-family runtime-overhead multiplier (Q4 → ×1.02, Q8 → ×1.0, etc.) calibrated against real GPU measurements rather than guessed.

## [0.29.0] - 2026-06-19

### Added
- MCP servers now auto-detect their transport type (`streamable_http` or `sse`) and tool list on installation; they appear in the model list only after a successful health check.
- Toast notifications in the WebUI now have a close (X) button in the top-right corner, so pop-ups (e.g. installation messages) can be dismissed manually. In-progress download/install toasts keep their "Cancel" action and gain the X once they finish.
- All service types in the Infra WebUI now show a "↺ Refresh" button on their Models page; previously the button was only available for `ollama-external` services.
- Readable, read-only view of an installed service's configuration in the WebUI: a "Settings" button on each installed service opens a dialog showing every option with its proper field label (reusing the install-form spec metadata), with password fields masked and revealable via an eye icon. This replaces the raw key/value config dump previously shown inline in the services list.
- Installed services now show an **Edit** button alongside the Settings view, allowing in-place reconfiguration of a service without uninstalling it.
- Admin-only **Configuration** page in the WebUI: displays all infra environment variables with copy-to-clipboard buttons; secret values (API keys, tokens) are masked by default and revealed on demand via an eye icon; filled values sort to the top of the table.
- Cancel button for an in-progress model installation in the WebUI. Cancelling stops the underlying Docker image pull, tears down the whole install promise chain (no orphaned background tasks), closes the progress stream cleanly, and leaves the model uninstalled.
- MCP servers can now be registered from the Infra Web Panel in three ways: running a stdio-based server in Docker via a built-in bridge (Command), proxying a remote endpoint (Remote URL), or using a pre-built Docker image (Custom Image).
- Custom MCP proxy models (Remote URL) now support SSE transport in addition to the default Streamable HTTP — a `proxy_transport` field (`streamable_http` | `sse`) can be selected when installing a Remote URL MCP server.
- Added preferred default hardware when installing Service through UI.
- The **default model prefix** field in MCP server forms (Command and Remote URL) is now auto-proposed based on the server name/ID as the user types; the suggestion is overridden once the user manually edits the prefix field.

### Fixed

- Fixed models appearing twice in Infra UI Mesh when installed twice in a short timespan.
- Fixed inflated Ollama VRAM/RAM estimates for not-yet-loaded models: the estimate now uses the context Ollama actually runs with instead of the model's full native window.
- User-defined (Command/stdio) MCP servers now show their size in the WebUI instead of "N/A": it is resolved from the locally built Docker image on install and persisted so it survives restarts.
- Fixed the WebUI model-install progress percentage drifting out of sync between the table row and the bottom-right toast (most visible on slow connections): the toast now reads its `%` from the install-progress store (the same source as the row) instead of computing it from a raw value, so it respects the store's monotonicity rules and no longer freezes during the final completion animation.
- Installing a service/model whose Docker container name is already taken now fails with a clear `409 Conflict` ("A container named '<name>' already exists — remove it or choose a different name") instead of a generic "unknown error" that swallowed the underlying docker "is already in use by container" message.
- Fixed metrics password authentication always rejecting valid credentials by comparing a `SecretStr` object directly instead of calling `.get_secret_value()`.
- Fixed docker build subprocess not being killed on cancellation or exception during log streaming, leaving orphaned processes.
- Fixed `KeyError` crash when the install-progress error callback fired after the progress entry was already removed.
- Fixed MCP service teardown silently swallowing errors from model uninstall; errors are now logged.
- Fixed VLLM GPU memory utilization not being released when model installation is cancelled via `asyncio.CancelledError`.
- Installing a GPU-dependent service (e.g. Stable Diffusion Next) on a host without the NVIDIA container toolkit now surfaces a clear, actionable error instead of silently appearing to succeed and then failing health checks repeatedly.
- `push_to_github` release job no longer fails on a shallow clone; release tags are now verified to originate from `main`.
- CI no longer fails `pyright` non-deterministically on merge pipelines: the CI uv version is bumped to `0.11.8`, which understands the relative `exclude-newer` cooldown, so `uv.lock` is honored instead of being silently ignored and re-resolved. A `required-version = ">=0.11.8"` floor prevents an older uv from reintroducing the issue.
- Requesting a model that is not installed now returns `404 Model not found` instead of the misleading `400 Given model is not supported`. The `400` response is now reserved for the case where the model exists but does not support the requested endpoint (e.g. calling `/v1/embeddings` with a chat model).
- Remote service handling of models with aliases.
- Fixed MCP model types (`mcp`) being reported as untestable — the **Test** button now works for MCP servers and sends a `tools/list` request to validate the connection.
- Fixed DuckDuckGo MCP server (and other SSE-based Remote URL MCP servers) not working due to being hardcoded to Streamable HTTP transport.
- Fixed auto-import of MCP JSON config not detecting SSE transport when the server URL ends with `/sse`; also added a confirmation dialog before overwriting existing import.
- Fixed WebSocket client crash on mesh connect when services with custom endpoints are registered — `custom` was missing from the valid model type set.
- Model download errors during installation now return `504 Gateway Timeout` on network timeout and `507 Insufficient Storage` when disk space runs out, instead of a generic error; the installation progress stream now includes the actual error text in all cases.

### Changed
- New Python dependencies must be at least 1 week old (`exclude-newer = "1 week"` in `pyproject.toml`); new npm packages must be at least 7 days old (`min-release-age=7` in `.npmrc`) before they can be added to the project.

## [0.28.0] - 2026-06-03

### Changed
- Concurrent model uninstallation in all services.
- `openspec/specs` directory is now tracked in git so OpenSpec can read project specs between tasks

### Added
- Ollama External service models now report `context_window` and `max_context_window` in `GET /v1/models?additional_data=true`, populated from Ollama's `/api/show` endpoint.
- Added opt-in OTLP log export (`otel_logging_enabled = true`): Python application logs and uvicorn access/error logs are forwarded to the configured OTLP endpoint alongside traces.

### Fixed
- Removed "Required for OpenAI" flag for API_KEY when installing cloud service through Infra UI
- Improved error messages when pulling a HuggingFace model fails in Ollama and Ollama External: missing `hf.co/` prefix, non-GGUF repository, and gated model now show actionable messages instead of a generic "Model not available". Unknown errors now include the raw Ollama error text.
- Fixed GPU realtime stats raising an uncaught `OSError` when `nvidia-smi`/`rocm-smi` cannot be launched by the server process — a missing or non-launchable binary now correctly results in no GPU stats instead of an error.
- Fixed test result modal leaving the test running and stuck in "pending" when dismissed via Esc, backdrop click, or the X button — every dismissal method now cancels the in-flight test, same as the Cancel button.
- Fixed WebUI installation progress bar resetting to 0% when the page is reloaded mid-install.
- Fixed progress bar jumping backwards during active installation on each 10-second background refetch.
- Fixed progress bar regressing from 100% back to ~90% and replaying the completion animation a second time.
- Fixed orphaned `setInterval` timers when the polling effect and an active install mutation raced to track the same service/model.
- Fixed Ollama and Ollama External download progress overcounting bytes when multiple model layers are downloaded in parallel — per-digest tracking replaces the previous single `(last_digest, last_value)` pair.
- Added Vitest unit tests for `progress-simulation.ts` and `install-progress-store.ts` covering all regression scenarios.
- Added backend regression tests for Ollama per-digest progress tracking in `test_ollama_service.py` and `test_ollama_external_service.py`.
- Fixed command injection vulnerability in `Utils.run_command` — replaced `create_subprocess_shell` with `create_subprocess_exec` and changed the signature to `list[str]`, eliminating shell interpretation of subprocess arguments.
- Fixed model install falsely reporting success after a failed download: a stale download-progress entry could make a retry skip the actual download and leave the model "not installed". The entry is now always cleaned up on failure across all model services (llamacpp, ollama, ollama-external, rerank, speaches-ai, stable-diffusion, vllm).
- Added backend regression tests for download-progress cleanup on failure across all model services.

### Changed
- `check_key` is now a required field in the WebSocket `InitRequest`; subinfra nodes that omit it are rejected at parse time. The optional-field migration workaround has been removed.
- `healthcheck_start_period` in `SrvMcpCustomModel` and `SrvCustomCustomModel` now validates against `^\d+[smh]$` (e.g. `30s`, `5m`, `1h`), rejecting invalid Docker duration strings at parse time.

## [0.27.0] - 2026-05-27

### Added
- Actual VRAM/RAM usage.
- Estimated VRAM/RAM usage per model in ollama, llamacpp and vllm services.
- Infra mesh topology: each node connects to a single parent and learns the full ancestor chain up the graph.
- Propagation of topology changes via RPC `topology_update` (join/leave).
- Every node has knowledge about other nodes (upwards and downwards).
- New endpoint `GET /admin/mesh/topology` that returns the mesh topology tree from the current node's perspective.

### Changed
- Removed unused Pydantic models, orphaned config fields, dead functions, and stale commented-out code across multiple services.
- Replaced commented-out `print()` / `logger.info()` debug statements in `docker.py` and `core.py` with proper `logger.debug()` calls.
- Added `vulture` to dev dependencies for dead code detection.

### Fixed
- Fixed typo `uliumits` → `ulimits` in `DockerOptions` (`docker.py`).
- Fixed Stable Diffusion Next `n_iter` value defaulting to `None`/`0` when `body.n` is falsy — now correctly defaults to `1`.

## [0.26.3] - 26.05.2026

### Added
- Actual VRAM/RAM usage 
- Estimated VRAM/RAM usage per model in ollama, llamacpp and vllm services

### Fixed
- Fixed "Test" action for llama.cpp models in the admin UI — models with type `llm-v1-v2-v3-ant` were incorrectly reported as untestable.
- Download progress counter no longer jumps back during model installation.
- Removed broken PLLuM entries from `scripts/custom_models.json`: `tensorblock/Llama-PLLuM-8B-chat-GGUF` (account deleted, all URLs 404) and `CYFRAGOVPL/Llama-PLLuM-8B-chat` (modelfile error, cannot be installed). Refreshed `static/ollama-min.json` via `just ollama-get-models`.

## [0.26.0] - 2026-05-22

### Added
- Firecrawl MCP server support, providing web scraping and crawling capabilities via `/mcp/firecrawl/mcp`.
- DuckDuckGo MCP server for web search and web fetch capabilities.
- VLLM request priority support.
- Lemmatizer service integration.
- `df-finetune` support in custom services.
- Exposed Ollama `/api/chat` endpoint.
- BGE model support in custom services.
- Docling document chunker service.
- Fake progress bar for long-running installation steps.
- Cloud support flag.
- Parameter value validation for service configuration.
- Issue and MR templates for the repository.

### Fixed
- Custom endpoint query parameters are now correctly forwarded when proxying requests.
- Fixed slow VLLM startup.
- Fixed SHA parsing error with `@@` characters in output.
- CUDA version check for Speeches GPU — user now gets an error when CUDA version is insufficient.
- User no longer required to provide `size` value in custom models and services.

### Changed
- Refactored Ollama Modelfile creation.

## [0.25.0] - 2026-04-10

### Added
- Updated Ollama version.
- Added Scrapling MCP server.
- Added adapter registry integration.

### Changed
- Sorted Ollama model index and model entries.

## [0.24.2] - 2026-04-02

### Added
- Ollama websearch support.
- Streaming support for `/v1/responses` endpoint, including missing streaming response features.
- Endpoint to list custom services and custom MCP servers.
- GPT-5.3 and GPT-5.4 model support.
- Open-source infrastructure release.

### Fixed
- Fixed rerank service not starting on GPU.

## [0.24.1] - 2026-03-20

### Fixed
- Fixed function tool `parameters` field expected as dict instead of string.
- Updated rerank model configuration.

## [0.24.0] - 2026-03-10

### Added
- Added Claude service (Anthropic API proxy).
- Added reranking model support.
- Added Intel GPU support for LlamaCpp.
- Added Ollama custom context length option.
- Added LlamaCpp context window option exposed to users.
- Added model context length metadata for Ollama models.
- Added Ollama model alias info.
- Added model properties support.
- Added required headers definition per service.
- Added MCP websearch Docker fixes.

### Fixed
- Fixed `/v1/messages` endpoint.
- Fixed unclosed connection issue.
- Fixed LlamaCpp error during model installation.
- Fixed MCP and custom endpoint API bypass checks.

## [0.23.1] - 2026-02-19

### Fixed
- Fixed `json_schema` handling in `/v1/chat/completions`.

## [0.23.0] - 2026-02-18

### Added
- Option to create many service instances with independent options, models, and custom models; instances share downloaded models.
- New services.json scheme v2 with automatic conversion from v1; v2 is not backward compatible with v1.
- MCP Service support: Docker streamable HTTP and SSE MCP servers, websearch MCP servers, headers support, and required envs/headers per model.
- Added `all-MiniLM-L6-v2` embedding model.

### Fixed
- Volumes and environment variables in custom services are no longer required.

## [0.22.0] - 2026-02-13

- Updated ollama version.
- Added support for text to image in ollama.

## [0.21.5] - 2026-02-05

- Fixed model framing issues in UI.

## [0.21.4] - 2026-02-05

- Added `verbose_json` option to `/v1/audio/transcription` endpoint.
- Fixed `/v1/models` to be compatible with openai standard.

## [0.21.3] - 2026-02-04

- Added test model framing issues in UI.

## [0.21.2] - 2026-02-04

- Added timestamps in logs.

## [0.21.1] - 2026-02-04

- Extended metrics for prometheus with infra count in mesh.
- Added new debug logs.
- Added to base ollama models MythoMax L2.
- UI hotfix: service failed to be installed.

## [0.21.0] - 2026-02-02

- Hotfixes.

## [0.20.0] - 2026-02-02

- Added Nvidia Spark support.
- Implemented New UI.

## [0.19.4] - 2026-01-29

- Hotfixes.

## [0.19.3] - 2026-01-29

- Added healthcheck script.

## [0.19.2] - 2026-01-28

- Hotfixes.

## [0.19.1] - 2026-01-28

- Hotfixes.

## [0.19.0] - 2026-01-28

- Added two sided verification in Deepfellow Mesh.
- Extend options to purge service/model (remove it will all it's files and docker images).
- Extend options to see if service/model is downloaded (files downloaded but not installed).
- Added api healthcheck (`/health` endpoint).
- Implemented metrics for prometheus (`/metrics` endpoint).
- Added Bielik 11b 3.0 instruct to basic models in LlamaCpp.
- Fix image is not compatible with your system.
- Add native openai compatible `v1/responses` endpoint.
- Add native claude compatible `v1/messages` endpoint.
- Fix GPU detection.
- Fix speaches service remove dir on purge.

## [0.18.0] - 2026-01-19 03:05

- Extend option to choose gpu or cpu in service to use specified GPU or all GPUs.
- Hardware selection show in dependence od available hardware.
- Added authentication checks for static and runtime code.

## [0.17.0] - 2026-01-16 04:16

- Changelog started.
- Added OpenAI-compatible API server.
- Added service management for local and remote AI models.
- Added support for multiple AI backends (OpenAI, Ollama, LLaMA.cpp, vLLM, etc.).
- Implemented model registry and load balancing.
- Added WebSocket infrastructure for real-time communication.
- Added proxy support for external AI services.
- Implemented Docker integration with model management.
- Added model testing and validation capabilities.
- Included support for text-to-speech, speech-to-text, and image generation.
- Added lifecycle management for services and models.
