# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import errno
import json
import os
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.docker import DockerOptions
from server.models.models import InstallModelIn
from server.models.services import InstallServiceIn
from server.services.ollama_service import (
    InstalledInfo,
    LoadedModelInfo,
    ModelCapabilities,
    ModelSize,
    OllamaModel,
    OllamaOptions,
    OllamaService,
)
from server.utils.core import DownloadedPacket, DownloadPacket, FetchResult, PreDownloadPacket
from server.utils.ollama_registry import BlobLocks, OllamaModelRef, OllamaRegistryClient, OllamaRegistryError, parse_model_ref
from server.utils.vram_calculator import ArchParams

_ARCH = ArchParams(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    num_hidden_layers=32,
)


def _make_service() -> OllamaService:
    svc = object.__new__(OllamaService)
    svc._arch_cache = {}  # pyright: ignore[reportPrivateUsage]
    svc._capabilities_cache = {}  # pyright: ignore[reportPrivateUsage]
    svc._show_failure_at = {}  # pyright: ignore[reportPrivateUsage]
    svc._vram_cache = {}  # pyright: ignore[reportPrivateUsage]
    svc._log_cache = {}  # pyright: ignore[reportPrivateUsage]
    svc._ps_query_failing = set()  # pyright: ignore[reportPrivateUsage]
    svc._blob_locks = BlobLocks()  # pyright: ignore[reportPrivateUsage]
    svc._installing = set()  # pyright: ignore[reportPrivateUsage]
    svc.models = {}
    svc.models_download_progress = {}
    svc.config = MagicMock()
    svc.config.ollama_vram_overhead_factor = 1.0
    return svc


def _make_installed_info(container_name: str = "ollama-default") -> InstalledInfo:
    docker = object.__new__(DockerOptions)
    docker.container_name = container_name
    docker.name = "ollama"

    options = object.__new__(InstallServiceIn)

    return InstalledInfo(
        instance_name="default",
        docker=docker,
        models={},
        options=options,
        parsed_options=OllamaOptions(),
        container_host="localhost",
        container_port=11434,
        docker_exposed_port=11434,
        base_url="http://localhost:11434",
    )


_LLAMA3_INFO = {
    "general.architecture": "llama",
    "llama.embedding_length": 4096,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.block_count": 32,
}


def _make_fetch_result(status_code: int, data: object) -> FetchResult:
    return FetchResult(status_code=status_code, data=json.dumps(data))


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_returns_arch(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": _LLAMA3_INFO})
    svc = _make_service()

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.hidden_size == 4096
    assert result.num_attention_heads == 32
    assert result.num_key_value_heads == 8
    assert result.num_hidden_layers == 32


@pytest.mark.asyncio
@pytest.mark.parametrize("kv_value", [None, 0])
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_null_head_count_kv_is_not_masked(mock_fetch: AsyncMock, kv_value: int | None):
    # Masking a missing head_count_kv as 1 would understate the KV cache by the GQA ratio,
    # so the raw value is passed through and the calculator picks the MHA fallback.
    info = {**_LLAMA3_INFO, "llama.attention.head_count_kv": kv_value}
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": info})
    svc = _make_service()

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result.num_key_value_heads == kv_value
    assert result.num_attention_heads == 32


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_cached(mock_fetch: AsyncMock):
    cached = ArchParams(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32)
    svc = _make_service()
    svc._arch_cache[("default", "llama3:latest")] = cached  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is cached
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_non_200_returns_none(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(404, {})
    svc = _make_service()

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_missing_critical_fields_returns_none(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": {"general.architecture": "llama"}})
    svc = _make_service()

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_arch_params_fetch_exception_returns_none(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection error")
    svc = _make_service()

    result = await svc._get_arch_params("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_vision_reported(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": _LLAMA3_INFO, "capabilities": ["completion", "vision"]})
    svc = _make_service()

    result = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result.supports_vision is True
    assert result.supports_audio is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_no_vision_or_capabilities_field(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": _LLAMA3_INFO})
    svc = _make_service()

    result = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result.supports_vision is None
    assert result.supports_audio is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_cached(mock_fetch: AsyncMock):
    cached = ModelCapabilities(supports_vision=True, supports_audio=None)
    svc = _make_service()
    svc._capabilities_cache[("default", "llama3:latest")] = cached  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is cached
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_uses_bounded_timeout(mock_fetch: AsyncMock):
    """A down/hung instance must not stall behind aiohttp's 300s default for `/api/show`."""
    mock_fetch.return_value = _make_fetch_result(200, {"model_info": _LLAMA3_INFO})
    svc = _make_service()

    await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert mock_fetch.call_args.kwargs["timeout"] is not None
    assert mock_fetch.call_args.kwargs["timeout"] <= 10


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_caches_failure_and_skips_refetch(mock_fetch: AsyncMock):
    """A failed `/api/show` must not be retried on every call within the failure-cache window."""
    mock_fetch.side_effect = RuntimeError("connection refused")
    svc = _make_service()

    first = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]
    second = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert first.supports_vision is None
    assert second.supports_vision is None
    mock_fetch.assert_called_once()


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_capabilities_retries_after_failure_cache_expires(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection refused")
    svc = _make_service()

    await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    with patch("server.services.ollama_service.time.monotonic", return_value=time.monotonic() + 3600):
        mock_fetch.side_effect = None
        mock_fetch.return_value = _make_fetch_result(200, {"model_info": _LLAMA3_INFO, "capabilities": ["completion", "vision"]})
        result = await svc._get_capabilities("default", "http://localhost:11434", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result.supports_vision is True
    assert mock_fetch.call_count == 2


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_model_sizes_returns_sizes(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(
        200,
        {
            "models": [
                {
                    "name": "llama3:latest",
                    "size": 4 * 1024**3,
                    "details": {"parameter_size": "8B", "quantization_level": "Q4_K_M"},
                }
            ]
        },
    )
    svc = _make_service()

    result = await svc._get_model_sizes("http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert "llama3:latest" in result
    model = result["llama3:latest"]
    assert model.size_bytes == 4 * 1024**3
    assert model.parameters == 8_000_000_000
    assert model.bytes_weight == pytest.approx(4.85)


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_model_sizes_non_200_returns_empty(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(503, {})
    svc = _make_service()

    result = await svc._get_model_sizes("http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert result == {}


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_model_sizes_exception_returns_empty(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection error")
    svc = _make_service()

    result = await svc._get_model_sizes("http://localhost:11434")  # pyright: ignore[reportPrivateUsage]

    assert result == {}


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_returns_value(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 9.5
    svc = _make_service()

    result = await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    assert result == 9.5
    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 16, 1, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_no_arch_returns_none(mock_arch: AsyncMock):
    mock_arch.return_value = None
    svc = _make_service()

    result = await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_no_num_ctx_returns_none(mock_arch: AsyncMock):
    mock_arch.return_value = _ARCH
    svc = _make_service()

    result = await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=None
    )

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_custom_cache_type(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096, kv_cache_type="q8_0"
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 8, 1, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_defaults_cache_type_and_num_parallel(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 16, 1, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_explicit_num_parallel_overrides_default(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096, num_parallel=2
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 16, 2, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_quant_overhead_applied(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 9.5
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096, quantization_level="Q4_K_M"
    )

    assert mock_estimate.call_args.kwargs["overhead_factor"] == 1.02


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_env_overhead_factor_applied(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 9.5
    svc = _make_service()
    svc.config.ollama_vram_overhead_factor = 1.1

    result = await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    # no quantization_level given, so only the env margin scales the (mocked) estimate
    assert mock_estimate.call_args.kwargs["overhead_factor"] == pytest.approx(1.0)
    assert result == pytest.approx(9.5 * 1.1)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_quant_and_env_overhead_combined(mock_arch: AsyncMock, mock_estimate: MagicMock):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 9.5
    svc = _make_service()
    svc.config.ollama_vram_overhead_factor = 1.1

    result = await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096, quantization_level="Q4_K_M"
    )

    # quant overhead scales the model-size portion inside estimate_vram_gb; env margin scales the result
    assert mock_estimate.call_args.kwargs["overhead_factor"] == pytest.approx(1.02)
    assert result == pytest.approx(9.5 * 1.1)


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_models_returns_models(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(
        200,
        {"models": [{"name": "llama3:latest", "context_length": 4096}, {"name": "mistral:latest"}]},
    )
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result == {
        "llama3": LoadedModelInfo(context_length=4096, vram_gb=None),
        "mistral": LoadedModelInfo(context_length=0, vram_gb=None),
    }


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_models_reports_measured_vram(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(
        200,
        {
            "models": [
                {"name": "llama3:latest", "context_length": 4096, "size_vram": 5046586573},
                {"name": "mistral:latest", "context_length": 2048, "size_vram": 0},
            ]
        },
    )
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert result["llama3"].vram_gb == pytest.approx(4.7, abs=0.01)
    assert result["mistral"].vram_gb is None
    # _get_loaded_models no longer mutates _vram_cache directly — _resolve_vram_info owns that.
    assert svc._vram_cache == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_models_non_200_returns_none(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(503, {})
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_models_exception_returns_none(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection error")
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_models_repeated_failure_logs_debug_not_warning(mock_fetch: AsyncMock, caplog: pytest.LogCaptureFixture):
    """A second consecutive /api/ps failure for the same instance must drop to debug, not warn again."""
    mock_fetch.return_value = _make_fetch_result(503, {})
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        with caplog.at_level("WARNING"):
            await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]
        assert len(caplog.records) == 1
        caplog.clear()

        with caplog.at_level("DEBUG"):
            result = await svc._get_loaded_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "DEBUG"


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_model_info_maps_to_context_length_for_base_contract(mock_fetch: AsyncMock):
    """The public, base-class-compatible wrapper exposes only {model: context_length}, dict[str, int] | None."""
    mock_fetch.return_value = _make_fetch_result(
        200,
        {"models": [{"name": "llama3:latest", "context_length": 4096, "size_vram": 5046586573}]},
    )
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_loaded_model_info("default")

    assert result == {"llama3": 4096}


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_model_info_returns_none_on_failure(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection error")
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_loaded_model_info("default")

    assert result is None


@pytest.mark.asyncio
async def test_resolve_vram_info_not_loaded_returns_false_none():
    svc = _make_service()

    result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
        "default",
        "llama3:latest",
        4096,
        loaded_info={},
        sizes={},
        base_url="http://localhost:11434",
        num_parallel=None,
        kv_cache_type=None,
    )

    assert result == (False, None)


@pytest.mark.asyncio
async def test_resolve_vram_info_prefers_fresh_measured_vram_over_stale_cache():
    """A fresh measurement from this poll must win even over a previously cached (possibly stale) value."""
    svc = _make_service()
    svc._vram_cache[("default", "llama3:latest")] = 9.5  # stale, from an earlier estimate  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=None)) as mock_estimate:
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": LoadedModelInfo(context_length=4096, vram_gb=4.7)},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
            kv_cache_type=None,
        )

    assert result == (True, 4.7)
    assert svc._vram_cache[("default", "llama3:latest")] == 4.7  # pyright: ignore[reportPrivateUsage]
    mock_estimate.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_vram_info_measured_zero_is_used_not_treated_as_missing():
    """size_vram: 0 (vram_gb=0.0) must be reported and cached as 0.0, not fall through to the estimate."""
    svc = _make_service()

    with patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=9.5)) as mock_estimate:
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": LoadedModelInfo(context_length=4096, vram_gb=0.0)},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
            kv_cache_type=None,
        )

    assert result == (True, 0.0)
    mock_estimate.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_vram_info_uses_cached_estimate_when_no_fresh_measurement():
    svc = _make_service()
    svc._vram_cache[("default", "llama3:latest")] = 4.7  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=9.5)) as mock_estimate:
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": LoadedModelInfo(context_length=4096, vram_gb=None)},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
            kv_cache_type=None,
        )

    assert result == (True, 4.7)
    mock_estimate.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_vram_info_falls_back_to_estimate_when_not_cached():
    svc = _make_service()
    sizes = {"llama3:latest": ModelSize(8 * 1024**3, 8_000_000_000, 4.85, "Q4_K_M")}

    with patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=9.5)) as mock_estimate:
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": LoadedModelInfo(context_length=4096, vram_gb=None)},
            sizes=sizes,
            base_url="http://localhost:11434",
            num_parallel=2,
            kv_cache_type="q8_0",
        )

    assert result == (True, 9.5)
    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (
        "default",
        "http://localhost:11434",
        "llama3:latest",
        8 * 1024**3,
        4096,
        8_000_000_000,
        4.85,
        2,
        "Q4_K_M",
    )
    assert mock_estimate.call_args.kwargs == {"kv_cache_type": "q8_0"}


@pytest.mark.asyncio
async def test_resolve_vram_info_uses_model_context_as_fallback():
    svc = _make_service()

    with patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=5.0)) as mock_estimate:
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            8192,
            loaded_info={"llama3:latest": LoadedModelInfo(context_length=0, vram_gb=None)},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
            kv_cache_type=None,
        )

    assert result == (True, 5.0)
    assert mock_estimate.call_args.args[4] == 8192


@pytest.mark.asyncio
async def test_arch_cache_separate_per_instance():
    arch_a = ArchParams(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32)
    arch_b = ArchParams(hidden_size=2048, num_attention_heads=16, num_key_value_heads=4, num_hidden_layers=16)
    svc = _make_service()
    svc._arch_cache[("instance-a", "llama3:latest")] = arch_a  # pyright: ignore[reportPrivateUsage]
    svc._arch_cache[("instance-b", "llama3:latest")] = arch_b  # pyright: ignore[reportPrivateUsage]

    result_a = svc._arch_cache.get(("instance-a", "llama3:latest"))  # pyright: ignore[reportPrivateUsage]
    result_b = svc._arch_cache.get(("instance-b", "llama3:latest"))  # pyright: ignore[reportPrivateUsage]

    assert result_a is arch_a
    assert result_b is arch_b
    assert result_a is not result_b


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_cache_hit_skips_run_command(mock_utils: MagicMock):
    mock_utils.run_command = AsyncMock()
    svc = _make_service()
    svc._log_cache["ollama-default"] = (time.monotonic(), "cached output", 8.0)  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("ollama-default")  # pyright: ignore[reportPrivateUsage]

    assert result == "cached output"
    mock_utils.run_command.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_cache_miss_calls_run_command(mock_utils: MagicMock):
    mock_result = MagicMock()
    mock_result.stdout = "fresh logs"
    mock_result.stderr = ""
    mock_utils.run_command = AsyncMock(return_value=mock_result)
    svc = _make_service()

    result = await svc._get_docker_logs("ollama-default")  # pyright: ignore[reportPrivateUsage]

    assert result == "fresh logs"
    mock_utils.run_command.assert_called_once()


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_expired_cache_refreshes(mock_utils: MagicMock):
    mock_result = MagicMock()
    mock_result.stdout = "new logs"
    mock_result.stderr = ""
    mock_utils.run_command = AsyncMock(return_value=mock_result)
    svc = _make_service()
    svc._log_cache["ollama-default"] = (time.monotonic() - 100, "stale logs", 8.0)  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("ollama-default")  # pyright: ignore[reportPrivateUsage]

    assert result == "new logs"
    mock_utils.run_command.assert_called_once()


@pytest.mark.asyncio
async def test_download_with_ollama_raises_on_missing_hf_prefix() -> None:
    svc = _make_service()
    stream = MagicMock()

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(status_code=200, data=json.dumps({"error": "pull model manifest: file does not exist"}))

    with (
        patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_with_ollama(stream, "http://localhost:11434", "bartowski/Qwen3-0.6B-GGUF", "4GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "hf.co/" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_with_ollama_raises_on_non_gguf_repo() -> None:
    svc = _make_service()
    stream = MagicMock()

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(
            status_code=200,
            data=json.dumps({"error": 'pull model manifest: 400: {"error":"Repository is not GGUF or is not compatible with llama.cpp"}'}),
        )

    with (
        patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_with_ollama(stream, "http://localhost:11434", "hf.co/Qwen/Qwen3-0.6B", "4GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "GGUF" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_with_ollama_raises_on_gated_model() -> None:
    svc = _make_service()
    stream = MagicMock()

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"status": "pulling manifest"}))
        yield FetchResult(
            status_code=200,
            data=json.dumps({"error": 'pull model manifest: realm host "huggingface.co" does not match original host "hf.co"'}),
        )

    with (
        patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_with_ollama(stream, "http://localhost:11434", "hf.co/meta-llama/Llama-3.2-1B", "4GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "HuggingFace" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_with_ollama_raises_with_raw_error_for_unknown_errors() -> None:
    svc = _make_service()
    stream = MagicMock()

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"error": "some unexpected ollama error"}))

    with (
        patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "4GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "some unexpected ollama error" in exc_info.value.detail


@pytest.mark.asyncio
async def test_download_with_ollama_logs_progress_percentage(caplog: pytest.LogCaptureFixture) -> None:
    svc = _make_service()
    stream = MagicMock()

    async def mock_stream(*args: object, **kwargs: object):  # type: ignore[misc]
        yield FetchResult(status_code=200, data=json.dumps({"digest": "sha256:abc", "completed": 2_000_000_000}))
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))
        # Ollama keeps streaming "success" statuses after completion for small models — the
        # 100% log line must not repeat for each of them.
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))
        yield FetchResult(status_code=200, data=json.dumps({"status": "success"}))

    with (
        patch("server.services.ollama_service.stream_fetch_from", side_effect=mock_stream),
        caplog.at_level("INFO", logger="uvicorn"),
    ):
        await svc._download_with_ollama(stream, "http://localhost:11434", "llama3", "4GB")  # pyright: ignore[reportPrivateUsage]

    assert "llama3" in caplog.text
    assert "50%" in caplog.text
    assert caplog.text.count("100%") == 1


def test_ollama_options_mode_defaults_to_online() -> None:
    assert OllamaOptions().mode == "online"
    assert OllamaOptions.model_validate({"mode": ""}).mode == "online"
    assert OllamaOptions.model_validate({"mode": None}).mode == "online"


def test_ollama_options_mode_accepts_offline() -> None:
    assert OllamaOptions.model_validate({"mode": "offline"}).mode == "offline"


def test_ollama_options_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        OllamaOptions.model_validate({"mode": "sideways"})


def _ref(model: str) -> OllamaModelRef:
    ref = parse_model_ref(model)
    assert ref is not None
    return ref


@pytest.fixture
def download_service(tmp_path: Path) -> Iterator[OllamaService]:
    with patch.object(OllamaService, "_get_working_dir", return_value=str(tmp_path)):
        yield _make_service()


def _fake_registry_download(
    packets: list[DownloadPacket] | None = None, error: Exception | None = None
) -> Callable[[object, object], AsyncGenerator[DownloadPacket]]:
    async def download(self: object, ref: object) -> AsyncGenerator[DownloadPacket]:
        for packet in packets or []:
            yield packet
        if error is not None:
            raise error

    return download


@pytest.mark.asyncio
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock)
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
@patch.object(OllamaService, "_download_with_ollama", new_callable=AsyncMock)
async def test_download_with_downloader_uses_api_pull_for_online_registry_model(
    mock_download_with_ollama: AsyncMock,
    mock_download_with_registry_client: AsyncMock,
    mock_is_model_installed: AsyncMock,
    download_service: OllamaService,
) -> None:
    await download_service._download_with_downloader(MagicMock(), "http://localhost:11434", "llama3.2", "llama3.2", "2GB", 0, 1)  # pyright: ignore[reportPrivateUsage]

    assert mock_download_with_ollama.await_count == 1
    assert mock_download_with_registry_client.await_count == 0
    assert mock_is_model_installed.await_count == 0


@pytest.mark.asyncio
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=True)
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
@patch.object(OllamaService, "_download_with_ollama", new_callable=AsyncMock)
async def test_download_with_downloader_uses_registry_client_for_offline_registry_model(
    mock_download_with_ollama: AsyncMock,
    mock_download_with_registry_client: AsyncMock,
    mock_is_model_installed: AsyncMock,
    download_service: OllamaService,
) -> None:
    stream = MagicMock()

    await download_service._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
        stream, "http://localhost:11434", "llama3.2:3b", "llama3.2:3b", "2GB", 0, 1, offline=True
    )

    assert mock_download_with_ollama.await_count == 0
    assert mock_download_with_registry_client.await_count == 1
    assert mock_download_with_registry_client.await_args == call(stream, _ref("llama3.2:3b"), "2GB")
    assert mock_is_model_installed.await_args == call("http://localhost:11434", "llama3.2:3b")


@pytest.mark.asyncio
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=False)
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
async def test_download_with_downloader_fails_when_ollama_does_not_see_the_downloaded_model(
    mock_download_with_registry_client: AsyncMock, mock_is_model_installed: AsyncMock, download_service: OllamaService
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), "http://localhost:11434", "llama3.2:3b", "llama3.2:3b", "2GB", 0, 1, offline=True
        )

    assert exc_info.value.status_code == 500
    assert "does not see it" in exc_info.value.detail
    assert mock_download_with_registry_client.await_count == 1
    assert mock_is_model_installed.await_count == 1


@pytest.mark.asyncio
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
@patch.object(OllamaService, "_download_with_ollama", new_callable=AsyncMock)
async def test_download_with_downloader_rejects_other_registry_host_in_offline_mode(
    mock_download_with_ollama: AsyncMock, mock_download_with_registry_client: AsyncMock, download_service: OllamaService
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), "http://localhost:11434", "hf.co/org/repo", "hf.co/org/repo", "2GB", 0, 1, offline=True
        )

    assert exc_info.value.status_code == 400
    assert "offline mode" in exc_info.value.detail
    assert mock_download_with_ollama.await_count == 0
    assert mock_download_with_registry_client.await_count == 0


@pytest.mark.asyncio
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
@patch.object(OllamaService, "_download_with_ollama", new_callable=AsyncMock)
async def test_download_with_downloader_keeps_http_downloader_for_direct_url_in_offline_mode(
    mock_download_with_ollama: AsyncMock, mock_download_with_registry_client: AsyncMock, download_service: OllamaService
) -> None:
    async def download(*args: object) -> AsyncGenerator[DownloadPacket]:
        yield PreDownloadPacket(10)
        yield DownloadedPacket(10)

    download_service.model_downloader = MagicMock(download=download)

    await download_service._download_with_downloader(  # pyright: ignore[reportPrivateUsage]
        MagicMock(), "http://localhost:11434", "https://example.com/m.gguf", "m", "10B", 0, 1, offline=True
    )

    assert mock_download_with_ollama.await_count == 0
    assert mock_download_with_registry_client.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("offline", [True, False])
@patch.object(OllamaService, "_download_with_downloader", new_callable=AsyncMock)
async def test_download_model_or_set_progress_forwards_offline(
    mock_download_with_downloader: AsyncMock, offline: bool, download_service: OllamaService
) -> None:
    await download_service._download_model_or_set_progress(  # pyright: ignore[reportPrivateUsage]
        MagicMock(), "http://localhost:11434", "llama3.2", "llama3.2", "2GB", offline=offline
    )

    assert mock_download_with_downloader.await_count == 1
    assert mock_download_with_downloader.await_args.args[-1] is offline  # pyright: ignore[reportOptionalMemberAccess]


def _install_service(download_service: OllamaService, model: OllamaModel, mode: str) -> InstalledInfo:
    info = _make_installed_info()
    info.parsed_options = OllamaOptions(mode=mode)  # pyright: ignore[reportArgumentType]
    download_service.models = {"default": {model.id: model}}
    return info


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "offline"), [("offline", True), ("online", False)])
@patch.object(OllamaService, "_download_model_or_set_progress", new_callable=AsyncMock, side_effect=HTTPException(418))
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=False)
@patch.object(OllamaService, "get_instance_installed_info")
async def test_install_model_forwards_offline_mode_of_the_instance(
    mock_get_instance_installed_info: MagicMock,
    mock_is_model_installed: AsyncMock,
    mock_download_model_or_set_progress: AsyncMock,
    mode: str,
    offline: bool,
    download_service: OllamaService,
) -> None:
    model = OllamaModel(id="llama3.2", size="2GB", type="llm", hash="h", context=None)
    mock_get_instance_installed_info.return_value = _install_service(download_service, model, mode)

    promise = await download_service._install_model("default", "llama3.2", InstallModelIn())  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc_info:
        await promise.wait()

    assert exc_info.value.status_code == 418
    assert mock_is_model_installed.await_count == 1
    assert mock_download_model_or_set_progress.await_count == 1
    assert mock_download_model_or_set_progress.await_args.kwargs["offline"] is offline  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "offline"), [("offline", True), ("online", False)])
@patch.object(OllamaService, "_download_model_or_set_progress", new_callable=AsyncMock, side_effect=HTTPException(418))
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=False)
async def test_install_from_modelfile_forwards_offline_mode_of_the_instance(
    mock_is_model_installed: AsyncMock,
    mock_download_model_or_set_progress: AsyncMock,
    mode: str,
    offline: bool,
    download_service: OllamaService,
) -> None:
    model = OllamaModel(id="my-llama", size="2GB", type="llm", hash="h", context=None)
    info = _install_service(download_service, model, mode)

    with pytest.raises(HTTPException) as exc_info:
        await download_service._install_from_modelfile(  # pyright: ignore[reportPrivateUsage]
            MagicMock(), "my-llama", "FROM llama3.2:3b", "my-llama", info, model
        )

    assert exc_info.value.status_code == 418
    assert mock_is_model_installed.await_args == call(info.base_url, "llama3.2:3b")
    assert mock_download_model_or_set_progress.await_count == 1
    assert mock_download_model_or_set_progress.await_args.kwargs["offline"] is offline  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
@patch.object(
    OllamaRegistryClient,
    "download",
    _fake_registry_download(
        [PreDownloadPacket(0), PreDownloadPacket(100), DownloadedPacket(1), DownloadedPacket(39), DownloadedPacket(60)]
    ),
)
async def test_download_with_registry_client_reports_progress_up_to_one(download_service: OllamaService) -> None:
    stream = MagicMock()

    await download_service._download_with_registry_client(stream, _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    values = [emit_call.args[0]["value"] for emit_call in stream.emit.call_args_list]
    assert values == pytest.approx([0, 0.01, 0.4, 1.0, 1])


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download([DownloadedPacket(500_000_000)]))
async def test_download_with_registry_client_reports_the_catalog_size_as_total(download_service: OllamaService) -> None:
    stream = MagicMock()

    await download_service._download_with_registry_client(stream, _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_args_list[1].args[0]["data"] == {"completed": 500_000_000, "total": 1_000_000_000}
    assert stream.emit.call_args_list[-1].args[0]["data"] == {}


@pytest.mark.asyncio
@patch.object(
    OllamaRegistryClient, "download", _fake_registry_download([DownloadedPacket(10), PreDownloadPacket(100), DownloadedPacket(40)])
)
async def test_download_with_registry_client_emits_no_byte_event_until_the_size_is_known(download_service: OllamaService) -> None:
    stream = MagicMock()

    await download_service._download_with_registry_client(stream, _ref("llama3.2"), "unknown")  # pyright: ignore[reportPrivateUsage]

    datas = [emit_call.args[0]["data"] for emit_call in stream.emit.call_args_list]
    assert datas == [{}, {"completed": 40, "total": 100}, {}]


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download(error=OllamaRegistryError("Model nope not found", 404)))
async def test_download_with_registry_client_maps_registry_error_to_http_exception(download_service: OllamaService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_registry_client(MagicMock(), _ref("nope"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download(error=OllamaRegistryError("registry down")))
async def test_download_with_registry_client_keeps_default_502_status(download_service: OllamaService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_registry_client(MagicMock(), _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download(error=PermissionError("denied")))
async def test_download_with_registry_client_maps_os_error_to_http_exception(download_service: OllamaService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_registry_client(MagicMock(), _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500
    assert "chown" not in exc_info.value.detail


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download(error=OSError(errno.EACCES, "Permission denied")))
async def test_download_with_registry_client_tells_how_to_fix_a_permission_error(download_service: OllamaService, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_registry_client(MagicMock(), _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500
    assert "chown" in exc_info.value.detail
    assert f"sudo chown -R {os.getuid()}:{os.getgid()} {tmp_path / 'main' / 'models'}" in exc_info.value.detail


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download(error=OSError(errno.ENOSPC, "No space left on device")))
async def test_download_with_registry_client_maps_full_disk_to_507(download_service: OllamaService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await download_service._download_with_registry_client(MagicMock(), _ref("llama3.2"), "1GB")  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 507


@pytest.mark.asyncio
@patch("server.services.ollama_service.OllamaRegistryClient")
async def test_download_with_registry_client_writes_into_the_mounted_models_dir(
    mock_registry_client: MagicMock, download_service: OllamaService, tmp_path: Path
) -> None:
    fake_download = _fake_registry_download([PreDownloadPacket(10), DownloadedPacket(10)])

    def download(ref: OllamaModelRef) -> AsyncGenerator[DownloadPacket]:
        return fake_download(None, ref)

    mock_registry_client.return_value.download = download

    await download_service._download_with_registry_client(MagicMock(), _ref("llama3.2"), "10B")  # pyright: ignore[reportPrivateUsage]

    assert mock_registry_client.call_count == 1
    assert mock_registry_client.call_args == call(tmp_path / "main" / "models", blob_locks=download_service._blob_locks)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "download", _fake_registry_download([DownloadedPacket(10)]))
async def test_download_with_registry_client_without_known_size_still_ends_at_one(download_service: OllamaService) -> None:
    stream = MagicMock()

    await download_service._download_with_registry_client(stream, _ref("llama3.2"), "")  # pyright: ignore[reportPrivateUsage]

    assert [emit_call.args[0]["value"] for emit_call in stream.emit.call_args_list] == [0, 1]


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_build_env_vars_leaves_ollama_blob_pruning_enabled(mode: str) -> None:
    svc = _make_service()

    envs = svc._build_env_vars(OllamaOptions(mode=mode), [])  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert "OLLAMA_NOPRUNE" not in envs


def test_create_model_store_dirs_creates_the_dirs_ollama_would_create_as_root(download_service: OllamaService, tmp_path: Path) -> None:
    download_service._create_model_store_dirs()  # pyright: ignore[reportPrivateUsage]

    assert (tmp_path / "main" / "models" / "blobs").is_dir()
    assert (tmp_path / "main" / "models" / "manifests" / "registry.ollama.ai" / "library").is_dir()


def test_create_model_store_dirs_keeps_the_manifest_dir_from_being_pruned_as_empty(download_service: OllamaService, tmp_path: Path) -> None:
    download_service._create_model_store_dirs()  # pyright: ignore[reportPrivateUsage]

    assert (tmp_path / "main" / "models" / "manifests" / "registry.ollama.ai" / "library" / ".keep").is_file()


def test_create_model_store_dirs_keeps_existing_content(download_service: OllamaService, tmp_path: Path) -> None:
    blob = tmp_path / "main" / "models" / "blobs" / "sha256-x"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x")

    download_service._create_model_store_dirs()  # pyright: ignore[reportPrivateUsage]

    assert blob.read_bytes() == b"x"


@patch("server.services.ollama_service.Path.touch", side_effect=PermissionError(13, "Permission denied"))
def test_create_model_store_dirs_does_not_fail_on_a_store_owned_by_root(
    mock_touch: MagicMock, download_service: OllamaService, tmp_path: Path
) -> None:
    download_service._create_model_store_dirs()  # pyright: ignore[reportPrivateUsage]

    assert mock_touch.called
    assert (tmp_path / "main" / "models" / "blobs").is_dir()


def _installed_info() -> InstalledInfo:
    info = MagicMock(spec=InstalledInfo)
    info.docker = MagicMock()
    info.docker.name = "ollama"
    info.docker.service_name = "ollama-service"
    return info


@pytest.mark.asyncio
@patch("server.services.ollama_service.os.access", return_value=True)
async def test_ensure_model_store_writable_does_nothing_when_dirs_are_writable(
    mock_access: MagicMock, download_service: OllamaService
) -> None:
    download_service.docker_service = MagicMock(run_command_docker_compose=AsyncMock())

    await download_service._ensure_model_store_writable(_ref("llama3.2"), _installed_info())  # pyright: ignore[reportPrivateUsage]

    assert mock_access.called
    assert download_service.docker_service.run_command_docker_compose.await_count == 0  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.asyncio
@patch("server.services.ollama_service.os.access", side_effect=lambda path, mode: not str(path).endswith("registry.ollama.ai/library"))  # pyright: ignore[reportUnknownLambdaType]
async def test_ensure_model_store_writable_takes_over_the_store_in_the_container(
    mock_access: MagicMock, download_service: OllamaService, tmp_path: Path
) -> None:
    (tmp_path / "main" / "models" / "manifests" / "registry.ollama.ai" / "library").mkdir(parents=True)
    run_command = AsyncMock()
    download_service.docker_service = MagicMock(run_command_docker_compose=run_command)

    await download_service._ensure_model_store_writable(_ref("llama3.2"), _installed_info())  # pyright: ignore[reportPrivateUsage]

    assert run_command.await_count == 1
    assert run_command.await_args.args[1:] == (  # pyright: ignore[reportOptionalMemberAccess]
        "ollama-service",
        ["chown", "-R", f"{os.getuid()}:{os.getgid()}", "/root/.ollama/models"],
    )


@pytest.mark.asyncio
@patch("server.services.ollama_service.os.access", return_value=False)
async def test_ensure_model_store_writable_gives_the_manual_command_when_the_container_chown_fails(
    mock_access: MagicMock, download_service: OllamaService, tmp_path: Path
) -> None:
    download_service.docker_service = MagicMock(run_command_docker_compose=AsyncMock(side_effect=RuntimeError("no such container")))

    with pytest.raises(HTTPException) as exc_info:
        await download_service._ensure_model_store_writable(_ref("llama3.2"), _installed_info())  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500
    assert f"sudo chown -R {os.getuid()}:{os.getgid()} {tmp_path / 'main' / 'models'}" in exc_info.value.detail


@pytest.mark.asyncio
@patch("server.services.ollama_service.os.access", return_value=False)
async def test_ensure_model_store_writable_skips_without_instance_info(mock_access: MagicMock, download_service: OllamaService) -> None:
    await download_service._ensure_model_store_writable(_ref("llama3.2"), None)  # pyright: ignore[reportPrivateUsage]

    assert mock_access.call_count == 0


@pytest.mark.asyncio
@patch.object(OllamaService, "is_model_installed", new_callable=AsyncMock, return_value=True)
@patch.object(OllamaService, "_download_with_registry_client", new_callable=AsyncMock)
@patch.object(OllamaService, "_ensure_model_store_writable", new_callable=AsyncMock)
async def test_download_offline_registry_model_ensures_the_store_is_writable_before_downloading(
    mock_ensure: AsyncMock, mock_download: AsyncMock, mock_installed: AsyncMock, download_service: OllamaService
) -> None:
    info = _installed_info()
    mock_download.side_effect = lambda *_: mock_ensure.assert_awaited_once_with(_ref("llama3.2:3b"), info)  # pyright: ignore[reportUnknownLambdaType]

    await download_service._download_offline_registry_model(MagicMock(), "http://x", "llama3.2:3b", "2GB", info)  # pyright: ignore[reportPrivateUsage]

    assert mock_download.await_count == 1
