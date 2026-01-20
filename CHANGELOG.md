# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

- Check child Deepfellow Infra existence by parent DeepFellow Infra in mesh connection.
- Extended GPU choice depending on the available hardware.
- Add showing is service downloaded in WebUI.
- Add purge button to WebUI.

## [0.17.0] - 2026-01-16

- Changelog started.
- Added OpenAI-compatible API server
- Added service management for local and remote AI models
- Added support for multiple AI backends (OpenAI, Ollama, LLaMA.cpp, vLLM, etc.)
- Implemented model registry and load balancing
- Added WebSocket infrastructure for real-time communication
- Added proxy support for external AI services
- Implemented Docker integration with model management
- Added model testing and validation capabilities
- Included support for text-to-speech, speech-to-text, and image generation
- Added lifecycle management for services and models
