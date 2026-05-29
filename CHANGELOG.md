# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Unreleased]

### Added
- Added opt-in OTLP log export (`otel_logging_enabled = true`): Python application logs and uvicorn access/error logs are forwarded to the configured OTLP endpoint alongside traces.

### Fixed
- Fixed command injection vulnerability in `Utils.run_command` — replaced `create_subprocess_shell` with `create_subprocess_exec` and changed the signature to `list[str]`, eliminating shell interpretation of subprocess arguments.

### Changed
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
- Fixed Stable Diffusion `n_iter` value defaulting to `None`/`0` when `body.n` is falsy — now correctly defaults to `1`.
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
