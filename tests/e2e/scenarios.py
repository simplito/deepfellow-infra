# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""The scenario matrix: one entry per (service, model, modality).

Add a scenario by appending to `SCENARIOS`; `test_lifecycle.py` picks it up. `UPGRADE_SETS` names the scenarios
`test_upgrade.py` installs together before an upgrade. Models are the smallest stable catalog
entries of their kind - the suite checks the lifecycle and the proxy, not output quality.
"""

from typing import Any, Literal

from pydantic import BaseModel, SecretStr

from tests.e2e.config import E2ESettings

type Modality = Literal["chat", "embeddings", "rerank", "image", "speech", "transcription", "decision", "document"]
type Mark = Literal["gate", "cpu", "gpu", "cloud"]
CPU = {"hardware": "CPU"}
GPU = {"hardware": "GPU"}


class Scenario(BaseModel):
    id: str
    service_id: str
    service_spec: dict[str, Any]
    model_id: str
    model_spec: dict[str, Any] | None = None
    # Name the model is served under on `/v1/*`; defaults to `model_id`.
    api_model: str | None = None
    modality: Modality
    marks: frozenset[Mark]
    # Embeddings only: the vector length the model must return, so a request routed to the wrong model fails.
    dimensions: int | None = None
    # Service spec fields filled from secret settings at install time: spec field -> E2ESettings field (e.g. an API key).
    # The scenario is skipped while any of those settings is empty.
    service_secrets: dict[str, str] = {}

    @property
    def served_model(self) -> str:
        return self.api_model or self.model_id

    def missing_settings(self, settings: E2ESettings) -> list[str]:
        return [f"DF_E2E_{name.upper()}" for name in self.service_secrets.values() if not _secret(settings, name)]

    def resolved_service_spec(self, settings: E2ESettings) -> dict[str, Any]:
        return {**self.service_spec, **{field: _secret(settings, name) for field, name in self.service_secrets.items()}}


def _secret(settings: E2ESettings, name: str) -> str:
    value: SecretStr = getattr(settings, name)
    return value.get_secret_value()


SCENARIOS: list[Scenario] = [
    # --- gate: the quick default of the on-demand CI jobs, so keep it small and CPU-only ---
    Scenario(
        id="ollama-chat-cpu",
        service_id="ollama",
        service_spec=CPU,
        model_id="smollm2:135m",
        modality="chat",
        marks=frozenset({"gate", "cpu"}),
    ),
    Scenario(
        id="ollama-embeddings-cpu",
        service_id="ollama",
        service_spec=CPU,
        model_id="all-minilm",
        modality="embeddings",
        dimensions=384,
        marks=frozenset({"gate", "cpu"}),
    ),
    # --- cpu: nightly ---
    Scenario(
        id="llamacpp-chat-cpu",
        service_id="llamacpp",
        service_spec=CPU,
        model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF-q4-0",
        modality="chat",
        marks=frozenset({"cpu"}),
    ),
    Scenario(
        id="rerank-cpu",
        service_id="rerank",
        service_spec=CPU,
        model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        modality="rerank",
        marks=frozenset({"cpu"}),
    ),
    Scenario(
        id="ollama-decision-cpu",
        service_id="ollama",
        service_spec=CPU,
        model_id="tev1:0.8b",
        modality="decision",
        marks=frozenset({"cpu"}),
    ),
    Scenario(
        id="doc-chunker-cpu",
        service_id="custom",
        service_spec={},
        model_id="doc_chunker",
        # The prefix is the path Server calls (`/custom/doc_chunker/...`); hardware sits on the model for `custom`.
        model_spec={"hardware": "CPU", "prefix": "doc_chunker"},
        modality="document",
        marks=frozenset({"cpu"}),
    ),
    Scenario(
        id="speaches-speech-cpu",
        service_id="speaches-ai",
        service_spec=CPU,
        model_id="speaches-ai/piper-en_US-amy-low",
        modality="speech",
        marks=frozenset({"cpu"}),
    ),
    Scenario(
        id="speaches-transcription-cpu",
        service_id="speaches-ai",
        service_spec=CPU,
        model_id="Systran/faster-whisper-tiny.en",
        modality="transcription",
        marks=frozenset({"cpu"}),
    ),
    # --- gpu: nightly, skipped when the target offers no GPU for the service ---
    Scenario(
        id="ollama-chat-gpu",
        service_id="ollama",
        service_spec=GPU,
        model_id="smollm2:135m",
        modality="chat",
        marks=frozenset({"gpu"}),
    ),
    Scenario(
        id="vllm-chat-gpu",
        service_id="vllm",
        service_spec=GPU,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_spec={"gpu_memory_utilization": 0.25, "max_model_length": 2048},
        modality="chat",
        marks=frozenset({"gpu"}),
    ),
    Scenario(
        id="vllm-embeddings-gpu",
        service_id="vllm",
        service_spec=GPU,
        model_id="BAAI/bge-small-en-v1.5",
        model_spec={"gpu_memory_utilization": 0.1},
        modality="embeddings",
        dimensions=384,
        marks=frozenset({"gpu"}),
    ),
    Scenario(
        id="sglang-chat-gpu",
        service_id="sglang",
        service_spec=GPU,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_spec={"gpu_memory_utilization": 0.25, "max_model_length": 2048},
        modality="chat",
        marks=frozenset({"gpu"}),
    ),
    Scenario(
        id="stable-diffusion-image-gpu",
        service_id="stable-diffusion-next",
        service_spec=GPU,
        model_id="SDXL Lighting 8 step",
        modality="image",
        marks=frozenset({"gpu"}),
    ),
    # --- cloud: OpenAI only, skipped without DF_E2E_OPENAI_API_KEY ---
    Scenario(
        id="openai-chat",
        service_id="openai",
        service_spec={},
        service_secrets={"api_key": "openai_api_key"},
        model_id="gpt-4o-mini",
        modality="chat",
        marks=frozenset({"cloud"}),
    ),
    Scenario(
        id="openai-embeddings",
        service_id="openai",
        service_spec={},
        service_secrets={"api_key": "openai_api_key"},
        model_id="text-embedding-3-small",
        modality="embeddings",
        dimensions=1536,
        marks=frozenset({"cloud"}),
    ),
]

# Scenarios installed together before an upgrade, selected with `-k <set>`. A set installs each service once, so the
# full set runs Ollama on CPU only. It must fit one 16 GB GPU: vLLM and SGLang models get a fixed share of VRAM instead
# of 90% of what is free, and vLLM embeddings stay out. A scenario joins only once the release the upgrade starts
# from supports it.
UPGRADE_SETS: dict[str, list[str]] = {
    "gate": ["ollama-chat-cpu", "ollama-embeddings-cpu"],
    "full": [
        "ollama-chat-cpu",
        "ollama-embeddings-cpu",
        "llamacpp-chat-cpu",
        "rerank-cpu",
        "speaches-speech-cpu",
        "speaches-transcription-cpu",
        "ollama-decision-cpu",
        "doc-chunker-cpu",
        "stable-diffusion-image-gpu",
        "vllm-chat-gpu",
        "sglang-chat-gpu",
    ],
}


# Known Infra bugs the suite has found, each with what is wrong. A broken scenario is expected to fail (strict xfail)
# and a broken chat API to answer with an error; both fail as soon as the bug is fixed, so the entry gets removed.
BROKEN_SCENARIOS: dict[str, str] = {}
BROKEN_CHAT_APIS: dict[tuple[str, str], str] = {}


# Services the target offers that no scenario covers, each with the reason. `test_coverage.py` fails on an offered
# service that is in neither, so a new service needs a scenario or an entry here.
NOT_COVERED: dict[str, str] = {
    "claude": "cloud provider; cloud scenarios cover OpenAI only for now",
    "deepseek": "cloud provider; cloud scenarios cover OpenAI only for now",
    "google": "cloud provider; cloud scenarios cover OpenAI only for now",
    "kimi": "cloud provider; cloud scenarios cover OpenAI only for now",
    "ollama-cloud": "cloud provider; cloud scenarios cover OpenAI only for now",
    "ollama-external": "proxies an external Ollama; not covered yet",
    "docker-model-runner": "needs Docker Model Runner on the host; not covered yet",
    "finetune": "training jobs, not a lifecycle plus one inference",
    "mcp": "MCP tools, no /v1 inference",
}


def upgrade_set(name: str) -> list[Scenario]:
    """The set's scenarios, without known-broken ones: one broken service must not hide whether the rest upgrades."""
    by_id = {s.id: s for s in SCENARIOS}
    return [by_id[scenario_id] for scenario_id in UPGRADE_SETS[name] if scenario_id not in BROKEN_SCENARIOS]
