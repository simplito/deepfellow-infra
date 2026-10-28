# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Ollama `mode` option: in `offline` mode Infra downloads Ollama registry models itself, so the container does not need to reach the registry.
- `POST /v1/systemone` serves Ollama decision models (new model type `decision`); needs Ollama 0.35.0 or newer.
- `POST /v1/systemone` also serves Ollama Cloud decision models (e.g. `clef:27b`); custom models of remote services such as OpenAI can have type `decision`.
- Chat completions accept `thinking_token_budget`, `chat_template_kwargs`, `repetition_detection`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logprobs` and `parallel_tool_calls`, passed to self-hosted backends and mesh peers only.
- Legacy completions (`/v1/completions`) accept `thinking_token_budget`, `repetition_detection`, `top_k`, `min_p` and `repetition_penalty`, passed to self-hosted backends and mesh peers only.
- `/admin/config` limits `max_generated_tokens`, `max_context_tokens`, `max_choices`, `max_top_k`.
- vLLM speech-to-text (`stt`) models such as Whisper, served on `/v1/audio/transcriptions`.
- vLLM catalog now lists 50 speech-to-text models, and the catalog refresh fetches them from HuggingFace.
- `just test-unit`, `just test-e2e` and `just test-smoke` commands.

### Changed

- Installed Ollama decision models move from the chat endpoints to `/v1/systemone` on restart.
- Ollama Cloud lists decision models as `decision` instead of `llm`; an installed one moves to `/v1/systemone` on restart.
- Mesh: upgrade every node before any of them has a decision model; older nodes reject peers that report one.
- Default Ollama image is now `0.40.8`; unpinned instances switch on next start and keep their models.
- LLM requests above the `/admin/config` generation limits get a 422 instead of reaching the model.
- `n` above `max_choices` (default 16) gets a 422; the previous fixed cap was 128.
- `max_tokens`, `max_completion_tokens` and `max_output_tokens` of 0 get a 400 (except `/v1/completions`).
- **Breaking:** Ollama `/api/chat` ignores `options` such as `num_gpu` instead of forwarding them.
- Ollama `/api/chat` `options` now have bounds, e.g. `num_ctx` < 1 or `temperature` > 2 gets a 400.
- Ollama `/api/chat` without a positive `num_predict` stops at `max_generated_tokens`.
- `just test-live` renamed to `just test-integration`; `DF_TEST_LIVE_REGISTRIES` to `DF_TEST_INTEGRATION_REGISTRIES`.
- Bruno smoke tests moved to `tests/smoke`.

### Removed

- Adding custom models to Ollama Cloud; they never worked there.

### Fixed

- `$` in a service's command, entrypoint and healthcheck now reaches the container as typed.
- Ollama catalog refresh fills in sizes ollama.com shows as a range (e.g. decision models, `gemma4`).
- Installing an Ollama instance on rootful Docker no longer fails with `Permission denied` when its model store was created as root by an earlier install.
- 5xx responses from local models and streams broken off by the model count as errors in `deepfellow_requests_total` and `deepfellow_request_errors_total` (5xx as `error_type="http_5xx"`), so the error rate on dashboards goes up.
- Custom vLLM models whose HuggingFace repo also ships a GGUF variant (e.g. `mixedbread-ai/mxbai-embed-large-v1`) are no longer rejected as GGUF, and are no longer dropped from the model catalog.
- vLLM speech-to-text catalog now includes models whose HuggingFace repo has GGUF files next to the regular weights.

## [0.36.0] - 2026-10-21

### Added

- `changelog-check` CI job needs an `[Unreleased]` entry (skip: `[skip changelog-check]` in MR title).
- Security headers (`Content-Security-Policy`, `Strict-Transport-Security`, etc.) on every HTTP response.
- `Cache-Control: no-store` on API responses and `no-cache` on WebUI files.
- `props.alias_of` in `GET /v1/models` with `additional_data=true` shows which models an alias points to.

### Changed

- Models installed or edited through the API with partial settings are stored with the WebUI form defaults (e.g. `--trust-remote-code` for embedding/rerank models); existing models are unchanged until edited.
- doc-chunker's `infra_api_key` is now a password field, masked in the WebUI.
- Adding a custom vLLM model now rejects a HuggingFace repo that doesn't exist or can't be accessed, with a clear error.
- `docker build` no longer runs the test suite, and the image no longer contains `tests/`.
- New installs default to vLLM `v0.32.0`, SGLang `v0.5.22` and llama.cpp `b11783`; model catalogs refreshed.

### Fixed

- **Test** button in the WebUI now works for custom models: it checks `GET /health` (or `/healthz`) and the optional `GET /test_model` self-test route.
- vLLM GPU image version list now also offers plain release tags such as `v0.31.0`.
- `just dev` no longer keeps a CPU core busy watching for file changes.
- `$` in container environment values is now escaped, so `pa$word` reaches the container intact; `${VAR}` is no longer expanded. Existing containers pick this up on next start; bge-m3, lemmatizer and doc-chunker containers are recreated once.
- Private HuggingFace repos now install with the configured token instead of failing with "not found".
- `POST /v1/audio/transcriptions` with `response_format=text` now returns plain text (`text/plain`) for every backend, also when the backend answers with JSON (as vLLM does). Backend errors and the other formats (`json`, `verbose_json`, `srt`, `vtt`) are passed through unchanged.

## [0.35.0] - 2026-10-15

### Added

- `keep_alive` setting in the speaches-ai install form
- **Update** action in the WebUI for models and services with an image version
- `GET /admin/services/{id}/docker-tags` reports the newest image tag
- Optional Sentry error tracking (off by default)
- Async job settings in the doc-chunker install form
- doc-chunker data now kept on a volume; `picture_description_prompt` is now required
- Per-model `usage` and `capacity` in `GET /v1/models` with `additional_data=true`
- Unhandled exceptions return an OpenAI-format `500` with an `X-Error-Id` header
- `commit` field in `GET /info`
- Container-name suffix setting so two Infra instances can share a Docker host
- `quantization` dropdown for vLLM models
- Finetune service, replacing the Custom `deepfellow-finetune` model
- `GET /admin/services/{service_id}/adapters` lists Finetune LoRA adapters, also shown in the WebUI
- vLLM models can be served as a Finetune LoRA adapter (`adapter_id`)
- Models report vision and audio input support

### Changed

- **Breaking:** vLLM/SGLang `extra_args` now reach the container as-is; check space-joined or quoted values
- **Breaking:** SGLang defaults to CUDA 13; older drivers must pin `v0.5.19-cu129`
- **Breaking:** `alias` now adds a second model name; the old meaning is now `rename`
- **Breaking:** vLLM `v0.30.0`+ rejects GPTQ `desc_act=True` checkpoints; SGLang rejects removed flags
- **Breaking:** Finetune service now authenticates with `adapter_registry_secret`
- New installs default to newer llama.cpp, vLLM and SGLang versions
- vLLM/SGLang models are sized with `gpu_priority` instead of `gpu_memory_utilization`; existing models keep the old value
- GPU allocation bar in the WebUI; allocation survives restarts and edits
- doc-chunker installs the `v2.4.0` image by default
- Model catalogs refreshed; vLLM/SGLang now cover the top 200 models by likes
- LoRA models no longer show a "Test" action

### Fixed

- Security: API keys and mesh key are compared in constant time
- Security: a sub-infra can no longer join the mesh when no mesh key is set
- Concurrent vLLM/SGLang installs can no longer over-allocate GPU memory
- A failing shutdown step no longer skips the remaining steps
- doc-chunker with default settings can describe images and transcribe audio
- Docker image version list includes the newest llama.cpp builds
- Startup mesh, Ollama External and MCP fetch failures are now logged
- MCP on a mesh child is usable through its parent; update both nodes
- Failed MCP servers are hidden after 3 failed health checks
- vLLM/SGLang `extra_args` containing JSON now start correctly
- Infra no longer hangs after proxying a body-less upstream response such as `204`
- Remote models report their real endpoints in `GET /v1/models`
- vLLM is no longer reported as unsupported on ARM hosts without a GPU
- doc-chunker conversions over 10 minutes are no longer cut off by the proxy
- brave-search, web-search and DuckDuckGo MCP entries install correctly
- Remote service URLs work with or without a trailing slash or `/v1`
- Stable Diffusion Next's macOS restriction is shown upfront
- Ollama VRAM estimates use the instance's `kv_cache_type`
- Retrying `PUT /admin/config` after a partial failure reapplies all changes
- Model names can no longer inject flags into container commands
- Custom Ollama install with quantization on GGUF or named sources fails early with a clear error

### Removed

- Coqui TTS service; remove installed Coqui containers manually
- `adapter_registry_url` setting; `config.json` is migrated automatically
- `deepfellow-finetune` Custom model; install the Finetune service instead

## [0.34.0] - 2026-09-16

### Fixed

- Reconfiguring OTLP log export no longer blocks the API when the collector is down or disables retries after a failure
- OTLP log export flushes buffered records on exit
- `POST /admin/services/{service_id}/cancel` waits for the automatic update rollback to finish
- Services list shows "Rolling back" and toasts the outcome

## [0.33.0] - 2026-09-11

### Added

- OpenTelemetry metrics and spans for MCP health checks and OAuth refresh
- Upfront check that a custom vLLM HuggingFace model can be served
- `POST /admin/services/{service_id}/catalog/refresh` refreshes vLLM, llama.cpp and SGLang catalogs
- Cancel for in-progress service installs and updates (WebUI button and `POST /admin/services/{service_id}/cancel`)

### Changed

- Default `standard_proxy_timeout_seconds` raised from 300 to 600

### Fixed

- Civitai, Hugging Face and adapter registry token changes apply without a restart
- Custom endpoints installed only on a child mesh node work through the parent's proxy
- MCP servers working only over SSE no longer return empty results
- Cancelled or failed installs no longer leave orphaned containers or processes
- Failed install adopting an existing container no longer tears it down on rollback
- Stop, restart and uninstall work on adopted containers
- Concurrent install/update requests can no longer bypass the "already installing" guard
- Install-progress endpoint returns `504` instead of hanging when setup is stuck
- Editing install options or custom models on vLLM-like services no longer returns 500
- Reranker and embedding models are also discovered via HuggingFace `pipeline_tag`
- Retrying a model install during a running install reflects the real outcome
- Failed service update reverts to the previous working configuration

## [0.32.0] - 2026-09-01

### Added

- Custom vLLM and SGLang models can target a specific HuggingFace revision
- **Warnings** page in the WebUI and `GET`/`DELETE /admin/warnings` for load and install failures
- Claude and Google (Gemini) services refresh their model catalog from the provider
- Subinfras see and route to parent models (`share_models_downstream`, on by default)
- "Edit Settings" and "Duplicate" actions for installed MCP and custom services
- `standard_proxy_timeout_seconds` setting (default 300), editable via `/admin/config`
- GLiNER installable as a custom service at `/custom/{prefix}/extract_entities`
- Docker image version selector for built-in custom-service and MCP models, including `hub.simplito.com` images
- `proxy_timeout_seconds` install field for custom services
- Proof-of-concept Dashboard UI (`UI_VARIANT=dashboard`); BuildKit is now required

### Changed

- Reworked doc-chunker install form; old installs keep their settings
- Model routing prefers warm, higher-capacity instances
- SGLang reports `max_running_requests` as routing capacity, like vLLM
- Ollama "Maximum parallel requests" default is now `1` instead of `3`
- Invalid llama.cpp `num_parallel` (0 or negative) is rejected

### Fixed

- `GET /v1/models` reports `null` instead of `0` for unknown context windows
- `deepfellow-bge-m3` declares its 8192-token context window
- Custom-service proxy requests are no longer capped at 5 minutes total
- Chat completions with `response_format={"type": "json_object"}` no longer fail with 422
- vLLM install no longer sticks in "installing" when cancelled during shutdown
- Uninstalling an Ollama model unloads it from VRAM
- Model definitions are preserved when a model leaves the registry or fails to load
- Older-schema model snapshots no longer abort loading other models
- Refreshing the Ollama catalog no longer hides installed models
- Ollama install progress bar no longer sticks at 100% after a failure
- Rotating `infra_api_key` refreshes proxy registrations on connected subinfras
- Installing a service adopts a matching orphaned container instead of failing
- Invalid HuggingFace repository ids fail fast with "model repository not found"
- Rerank install on rootful Docker and `POST /v1/rerank` with the non-root container no longer fail
- Custom-service models no longer start two concurrent installs when requested twice quickly

## [0.31.0] - 2026-08-06

### Added

- `POST /admin/mcp/convert-config` converts an MCP client JSON config into custom-model parameters
- `GET /info` returns the running Infra version
- Curated vLLM model registry
- "Refresh catalog" button in the Ollama service
- Ollama Cloud support
- Most settings are now stored in `config.json`, editable via the Configuration page or `/admin/config`
- Docker image version selector for Ollama, llama.cpp and vLLM
- `GET /admin/services/{id}/docker-tags` returns available image tags
- `DOCKER_HUB_TOKEN` environment variable for authenticated Docker Hub tag fetching
- Server-side logging of model download progress
- New Claude, OpenAI and Google AI models, including Gemini Embedding 2 and Imagen 4
- DeepSeek and Kimi services, and their models in the vLLM registry
- OAuth 2.1 (PKCE) for remote MCP servers, with "Authorize" in the WebUI
- SGLang service for LLM, embedding and reranker models
- Docker Model Runner service (Docker Desktop 4.40+; `vllm-metal` needs Apple Silicon macOS)

### Changed

- Project is now MIT licensed
- Ollama chat messages accept any `role` string
- Container healthcheck probes `/health` instead of `/docs`
- Bundled images updated: llama.cpp, Ollama, vLLM, open-websearch
- llama.cpp has unpatched CVE-2026-2069 and GGUF alignment overflow; avoid untrusted GGUF files
- Ollama VRAM usage is read from `/api/ps`
- Coqui TTS runs from the maintained `idiap/coqui-ai-TTS` fork
- Stable Diffusion service renamed to Stable Diffusion Next

### Fixed

- Custom SGLang and vLLM installs use the model's `hf_id`
- speaches-ai installs no longer fail with `PermissionError` as non-root
- `POST /v1/embeddings` accepts token-array `input` and restricts `encoding_format`
- vLLM install retries with a suggested `max_model_length` when context exceeds free KV cache
- Ollama `:latest` tags resolve for VRAM calculation
- vLLM and llama.cpp release GPU memory when a model container dies or an install fails
- `config.json` lives in the storage directory and survives container recreation
- OTEL exporter endpoint changes apply immediately
- VRAM estimation no longer returns HTTP 500 or understated sizes for missing model metadata
- WebUI model list shows an error with Retry when the models endpoint fails
- Transient `/api/ps` failure no longer marks a loaded Ollama model as unloaded
- MCP install without a required API key/header is flagged in the form and rejected by the backend
- Model install Cancel button stays in the table row after page refresh
- API errors return an OpenAI-style `invalid_request_error` body
- `/v1/models` reports real `created` and `owned_by` values
- `/v1/responses` resolves `item_reference` inputs and accepts optional-property function tools and function-call history items
- `/v1/models?additional_data=true` reports correct endpoint lists for Ollama, llama.cpp and vLLM
- Coqui TTS CPU and GPU Docker images were swapped

## [0.30.0] - 2026-06-26

### Added

- MCP and custom service models show a description and repository link in the WebUI
- Built-in MCP servers and custom services now have descriptions and repository URLs
- Optional Description and Repository URL fields for user-defined MCP servers in the WebUI

### Fixed

- VRAM estimation uses per-quantization bits-per-weight (e.g. Q4_0 vs Q4_K_M) and runtime overhead

## [0.29.0] - 2026-06-19

### Added

- MCP servers auto-detect transport (`streamable_http` or `sse`) and tools on install
- Close (X) button on WebUI toast notifications
- "Refresh" button on the Models page of all service types
- Read-only "Settings" dialog for installed services, with masked, revealable passwords
- "Edit" button for in-place reconfiguration of installed services
- Admin-only Configuration page in the WebUI showing environment variables with masked secrets
- Cancel button for in-progress model installation in the WebUI
- MCP servers can be registered from the WebUI as Command, Remote URL or Custom Image
- `proxy_transport` (`streamable_http` | `sse`) for Remote URL MCP servers
- Preferred default hardware when installing a service through the UI
- Default model prefix in MCP forms is auto-proposed from the server name

### Fixed

- Models no longer appear twice in Infra UI Mesh when installed twice quickly
- Ollama VRAM/RAM estimates for unloaded models use the actual context instead of the native window
- User-defined stdio MCP servers show their size in the WebUI instead of "N/A"
- WebUI install progress stays in sync between table row and toast
- Taken Docker container name now returns a clear `409 Conflict`
- Metrics password authentication no longer rejects valid credentials
- Docker build subprocess is killed on cancellation or exception
- MCP service teardown logs model uninstall errors instead of swallowing them
- vLLM releases GPU memory when model installation is cancelled
- GPU-dependent services on hosts without NVIDIA container toolkit now show a clear error
- Uninstalled model now returns `404 Model not found`; `400` is reserved for unsupported endpoints
- Remote service handling of models with aliases
- **Test** button works for MCP servers via `tools/list`
- DuckDuckGo and other SSE-based Remote URL MCP servers now work
- MCP JSON auto-import detects SSE for `/sse` URLs; confirms before overwriting an import
- WebSocket client no longer crashes on mesh connect with custom-endpoint services
- Model download errors return `504` on timeout and `507` when out of disk space, with error text

## [0.28.0] - 2026-06-03

### Added

- `context_window` and `max_context_window` for Ollama External models in `GET /v1/models?additional_data=true`
- Opt-in OTLP log export (`otel_logging_enabled = true`) for application and uvicorn logs

### Changed

- Models are uninstalled concurrently in all services
- **Breaking:** `check_key` is required in WebSocket `InitRequest`; subinfra nodes omitting it are rejected
- `healthcheck_start_period` in `SrvMcpCustomModel` and `SrvCustomCustomModel` must match `^\d+[smh]$`

### Fixed

- Removed "Required for OpenAI" flag for `API_KEY` in cloud service install
- Clearer errors for failed HuggingFace pulls in Ollama and Ollama External
- GPU realtime stats no longer raise `OSError` when `nvidia-smi`/`rocm-smi` cannot launch
- Test result modal cancels the running test when dismissed
- WebUI install progress bar no longer resets on reload, jumps backwards, or regresses from 100%
- Ollama download progress no longer overcounts bytes for parallel layers
- Security: command injection in `Utils.run_command` removed
- Failed model download no longer leaves a stale progress entry that makes retries report false success

## [0.27.0] - 2026-05-27

### Added

- Actual and estimated per-model VRAM/RAM usage in `ollama`, `llamacpp` and `vllm` services
- Mesh topology: each node connects to one parent and knows its ancestors and descendants
- `GET /admin/mesh/topology` endpoint returning the mesh topology tree from the current node

### Fixed

- Typo `uliumits` in Docker options, now `ulimits`
- Stable Diffusion Next `n_iter` now defaults to `1` when `n` is not set

## [0.26.3] - 2026-05-26

### Added

- Actual and estimated per-model VRAM/RAM usage in `ollama`, `llamacpp` and `vllm` services

### Fixed

- "Test" action for `llm-v1-v2-v3-ant` llama.cpp models in the admin UI
- Download progress counter jumping back during model installation
- Removed broken PLLuM entries from custom models and refreshed the Ollama model list

## [0.26.0] - 2026-05-22

### Added

- Firecrawl MCP server at `/mcp/firecrawl/mcp` and DuckDuckGo MCP server (web search and fetch)
- `vllm` request priority support
- Lemmatizer service and Docling document chunker service
- `df-finetune` and BGE model support in custom services
- Exposed Ollama `/api/chat` endpoint
- Fake progress bar for long-running installation steps
- Cloud support flag
- Parameter value validation for service configuration

### Changed

- `size` no longer required in custom models and services

### Fixed

- Custom endpoint query parameters now forwarded when proxying requests
- Slow `vllm` startup
- SHA parsing error on `@@` characters in output
- Error on insufficient CUDA version for Speeches GPU

## [0.25.0] - 2026-04-10

### Added

- Updated Ollama version
- Scrapling MCP server
- Adapter registry integration

### Changed

- Sorted Ollama model index and entries

## [0.24.2] - 2026-04-02

### Added

- Ollama web search support
- Streaming for `/v1/responses`
- Endpoint listing custom services and custom MCP servers
- GPT-5.3 and GPT-5.4 model support
- Open-source infrastructure release

### Fixed

- Rerank service not starting on GPU

## [0.24.1] - 2026-03-20

### Fixed

- Function tool `parameters` field expected as dict instead of string
- Rerank model configuration

## [0.24.0] - 2026-03-10

### Added

- Claude service (Anthropic API proxy)
- Reranking model support
- Intel GPU support for LlamaCpp
- Ollama custom context length option and context length metadata for models
- LlamaCpp context window option
- Ollama model alias info
- Model properties support
- Required headers definition per service
- MCP web search Docker fixes

### Fixed

- `/v1/messages` endpoint
- Unclosed connection issue
- LlamaCpp error during model installation
- MCP and custom endpoint API bypass checks

## [0.23.1] - 2026-02-19

### Fixed

- `json_schema` handling in `/v1/chat/completions`

## [0.23.0] - 2026-02-18

### Added

- Multiple service instances with independent options, models and custom models, sharing downloads
- `services.json` scheme v2 with automatic conversion from v1; v2 is not backward compatible
- MCP service support: Docker streamable HTTP and SSE servers, web search servers, headers, required envs
- `all-MiniLM-L6-v2` embedding model

### Fixed

- Volumes and environment variables no longer required in custom services

## [0.22.0] - 2026-02-13

### Added

- Text to image support in Ollama

### Changed

- Updated Ollama version

## [0.21.5] - 2026-02-05

### Fixed

- Model framing issues in UI

## [0.21.4] - 2026-02-05

### Added

- `verbose_json` option for `/v1/audio/transcription`

### Fixed

- `/v1/models` compatibility with the OpenAI standard

## [0.21.3] - 2026-02-04

### Fixed

- Test model framing issues in UI

## [0.21.2] - 2026-02-04

### Added

- Timestamps in logs

## [0.21.1] - 2026-02-04

### Added

- Prometheus metrics for infra count in mesh
- Debug logs
- MythoMax L2 in base Ollama models

### Fixed

- UI: service failing to install

## [0.21.0] - 2026-02-02

### Fixed

- Hotfixes

## [0.20.0] - 2026-02-02

### Added

- Nvidia Spark support
- New UI

## [0.19.4] - 2026-01-29

### Fixed

- Hotfixes

## [0.19.3] - 2026-01-29

### Added

- Healthcheck script

## [0.19.2] - 2026-01-28

### Fixed

- Hotfixes

## [0.19.1] - 2026-01-28

### Fixed

- Hotfixes

## [0.19.0] - 2026-01-28

### Added

- Two-sided verification in DeepFellow Mesh
- Purge options for services and models (removes all files and Docker images)
- Indicator for downloaded but not installed services and models
- `/health` endpoint
- Prometheus metrics at `/metrics`
- Bielik 11b 3.0 instruct in basic LlamaCpp models
- Native OpenAI-compatible `/v1/responses` endpoint
- Native Claude-compatible `/v1/messages` endpoint

### Fixed

- "Image is not compatible with your system" error
- GPU detection
- Speaches service not removing its directory on purge

## [0.18.0] - 2026-01-19

### Added

- Choice of a specific GPU or all GPUs per service
- Authentication checks for static and runtime code

### Changed

- Hardware selection reflects available hardware

## [0.17.0] - 2026-01-16

### Added

- OpenAI-compatible API server
- Service management for local and remote AI models
- Multiple AI backends (OpenAI, Ollama, LLaMA.cpp, vLLM, etc.)
- Model registry and load balancing
- WebSocket infrastructure for real-time communication
- Proxy support for external AI services
- Docker integration with model management
- Model testing and validation
- Text-to-speech, speech-to-text and image generation
- Lifecycle management for services and models
