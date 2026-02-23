# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

# Added

- Option to create many service instances.
  - Service instance have independent options, models and custom models.
  - Service instance share download models.
- New services.json scheme.
  - Old service.json (v1) will be automatically converted to v2.
  - Services.json v2 has not backward compatibility.
- Added Mcp Service.
  - Added option to host docker streamable HTTP and SSE MCP servers.
  - Added websearch mcp servers to install.
  - Added headers support.
  - Added required envs and headers for specified model.

### Fixed
- volumes and environmental variables in custom services are not required.

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
