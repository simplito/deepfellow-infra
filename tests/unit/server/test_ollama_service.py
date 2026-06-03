# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from server.docker import DockerOptions
from server.models.services import InstallServiceIn, MemoryLoadComponent, MemoryLoadOut, MemoryLoadSession
from server.services.ollama_service import InstalledInfo, ModelSize, OllamaOptions, OllamaService
from server.utils.core import CommandResult, FetchResult
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
    svc._blob_to_model = {}  # pyright: ignore[reportPrivateUsage]
    svc._model_manifest_loaded = False  # pyright: ignore[reportPrivateUsage]
    svc._vram_cache = {}  # pyright: ignore[reportPrivateUsage]
    svc._log_cache = {}  # pyright: ignore[reportPrivateUsage]
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


@pytest.mark.asyncio
async def test_resolve_blob_known_digest():
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]
    svc._blob_to_model = {"sha256:abc123": "library/llama3:latest"}  # pyright: ignore[reportPrivateUsage]

    result = await svc.resolve_blob("/root/.ollama/blobs/sha256-abc123")

    assert result == "llama3:latest"


@pytest.mark.asyncio
async def test_resolve_blob_unknown_digest():
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]
    svc._blob_to_model = {}  # pyright: ignore[reportPrivateUsage]

    result = await svc.resolve_blob("/root/.ollama/blobs/sha256-abc123def456")

    assert result == "unknown (sha256:abc12)"


@pytest.mark.asyncio
async def test_resolve_blob_no_digest_in_path():
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]

    result = await svc.resolve_blob("/some/path/without/hash")

    assert result == "/some/path/without/hash"


_SINGLE_SESSION_LOGS = """\
msg="model weights" device=cuda0 size="4.20 GiB"
msg="kv cache" device=cuda0 size="0.50 GiB"
msg="total memory" size="4.70 GiB"
"""

_TWO_SESSION_LOGS = """\
msg="model weights" device=cuda0 size="4.20 GiB"
msg="kv cache" device=cuda0 size="0.50 GiB"
msg="total memory" size="4.70 GiB"
msg="model weights" device=cuda0 size="8.00 GiB"
msg="kv cache" device=cuda0 size="1.00 GiB"
msg="total memory" size="9.00 GiB"
"""


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock)
async def test_get_memory_load_single_session(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult(exit_code=0, stdout=_SINGLE_SESSION_LOGS, stderr="")
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_memory_load("default")

    assert len(result.sessions) == 1
    session = result.sessions[0]
    assert session.total == "4.70 GiB"
    assert session.components == [
        MemoryLoadComponent(name="model weights", device="cuda0", size="4.20 GiB"),
        MemoryLoadComponent(name="kv cache", device="cuda0", size="0.50 GiB"),
    ]


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock)
async def test_get_memory_load_two_sessions(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult(exit_code=0, stdout=_TWO_SESSION_LOGS, stderr="")
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_memory_load("default")

    assert len(result.sessions) == 2
    assert result.sessions[0].total == "4.70 GiB"
    assert result.sessions[1].total == "9.00 GiB"


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock)
async def test_get_memory_load_empty_logs(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult(exit_code=0, stdout="", stderr="")
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_memory_load("default")

    assert result.sessions == []


@pytest.mark.asyncio
async def test_load_model_manifest_index_already_loaded():
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]
    svc._blob_to_model = {"sha256:abc": "model:latest"}  # pyright: ignore[reportPrivateUsage]

    await svc._load_model_manifest_index()  # pyright: ignore[reportPrivateUsage]

    assert svc._blob_to_model == {"sha256:abc": "model:latest"}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_model_manifest_index_no_manifests_dir(tmp_path: Path):
    svc = _make_service()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._load_model_manifest_index()  # pyright: ignore[reportPrivateUsage]

    assert svc._blob_to_model == {}  # pyright: ignore[reportPrivateUsage]
    assert not svc._model_manifest_loaded  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_model_manifest_index_builds_index(tmp_path: Path):
    manifests_dir = tmp_path / "main" / "models" / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifests_dir.mkdir(parents=True)
    manifest = {
        "config": {"digest": "sha256:config1"},
        "layers": [{"digest": "sha256:layer1"}, {"digest": "sha256:layer2"}],
    }
    (manifests_dir / "latest").write_text(json.dumps(manifest))
    svc = _make_service()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._load_model_manifest_index()  # pyright: ignore[reportPrivateUsage]

    assert svc._blob_to_model == {  # pyright: ignore[reportPrivateUsage]
        "sha256:config1": "registry.ollama.ai/library/llama3:latest",
        "sha256:layer1": "registry.ollama.ai/library/llama3:latest",
        "sha256:layer2": "registry.ollama.ai/library/llama3:latest",
    }
    assert svc._model_manifest_loaded  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_model_manifest_index_custom_registry(tmp_path: Path):
    manifests_dir = tmp_path / "main" / "models" / "manifests" / "my-registry.internal" / "models" / "llama3"
    manifests_dir.mkdir(parents=True)
    manifest = {
        "config": {"digest": "sha256:cfgcustom"},
        "layers": [{"digest": "sha256:layercustom"}],
    }
    (manifests_dir / "latest").write_text(json.dumps(manifest))
    svc = _make_service()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._load_model_manifest_index()  # pyright: ignore[reportPrivateUsage]

    assert "sha256:cfgcustom" in svc._blob_to_model  # pyright: ignore[reportPrivateUsage]
    assert "sha256:layercustom" in svc._blob_to_model  # pyright: ignore[reportPrivateUsage]
    tag = svc._blob_to_model["sha256:cfgcustom"]  # pyright: ignore[reportPrivateUsage]
    assert tag.startswith("my-registry.internal/")
    assert tag.endswith(":latest")


@pytest.mark.asyncio
async def test_load_model_manifest_index_skips_invalid_json(tmp_path: Path):
    manifests_dir = tmp_path / "main" / "models" / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "latest").write_text("not valid json {{{")
    svc = _make_service()

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):
        await svc._load_model_manifest_index()  # pyright: ignore[reportPrivateUsage]

    assert svc._blob_to_model == {}  # pyright: ignore[reportPrivateUsage]
    assert svc._model_manifest_loaded  # pyright: ignore[reportPrivateUsage]


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
async def test_get_vram_estimate_custom_cache_type(mock_arch: AsyncMock, mock_estimate: MagicMock, monkeypatch: pytest.MonkeyPatch):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 8, 1, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_num_parallel_from_env(mock_arch: AsyncMock, mock_estimate: MagicMock, monkeypatch: pytest.MonkeyPatch):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 16, 4, None, None)


@pytest.mark.asyncio
@patch("server.services.ollama_service.estimate_vram_gb")
@patch("server.services.ollama_service.OllamaService._get_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_explicit_num_parallel_overrides_env(
    mock_arch: AsyncMock, mock_estimate: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    mock_arch.return_value = _ARCH
    mock_estimate.return_value = 7.0
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    svc = _make_service()

    await svc._get_vram_estimate(  # pyright: ignore[reportPrivateUsage]
        "default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, num_ctx=4096, num_parallel=2
    )

    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == (_ARCH, 8 * 1024**3, 4096, 16, 2, None, None)


def _make_memory_load(*sessions: tuple[str | None, str]) -> MemoryLoadOut:
    return MemoryLoadOut(sessions=[MemoryLoadSession(model=model, total=total, components=[]) for model, total in sessions])


@pytest.mark.asyncio
@patch("server.services.ollama_service.get_vram_gb")
async def test_get_vram_from_logs_returns_value(mock_vram_gb: MagicMock):
    mock_vram_gb.return_value = 4.7
    svc = _make_service()
    memory_load = _make_memory_load(("llama3:latest", "4.70 GiB"))

    with patch.object(svc, "get_memory_load", new=AsyncMock(return_value=memory_load)):
        result = await svc._get_vram_from_logs("default", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result == 4.7


@pytest.mark.asyncio
@patch("server.services.ollama_service.get_vram_gb")
async def test_get_vram_from_logs_last_matching_session_wins(mock_vram_gb: MagicMock):
    mock_vram_gb.side_effect = [4.7, 9.0]
    svc = _make_service()
    memory_load = _make_memory_load(("llama3:latest", "4.70 GiB"), ("llama3:latest", "9.00 GiB"))

    with patch.object(svc, "get_memory_load", new=AsyncMock(return_value=memory_load)):
        result = await svc._get_vram_from_logs("default", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result == 9.0


@pytest.mark.asyncio
async def test_get_vram_from_logs_exception_returns_none():
    svc = _make_service()

    with patch.object(svc, "get_memory_load", new=AsyncMock(side_effect=RuntimeError("fail"))):
        result = await svc._get_vram_from_logs("default", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.get_vram_gb")
async def test_get_vram_from_logs_no_matching_model_returns_none(mock_vram_gb: MagicMock):
    mock_vram_gb.return_value = 4.7
    svc = _make_service()
    memory_load = _make_memory_load(("other-model:latest", "4.70 GiB"))

    with patch.object(svc, "get_memory_load", new=AsyncMock(return_value=memory_load)):
        result = await svc._get_vram_from_logs("default", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.get_vram_gb")
async def test_get_vram_from_logs_zero_vram_returns_none(mock_vram_gb: MagicMock):
    mock_vram_gb.return_value = 0.0
    svc = _make_service()
    memory_load = _make_memory_load(("llama3:latest", "0.00 GiB"))

    with patch.object(svc, "get_memory_load", new=AsyncMock(return_value=memory_load)):
        result = await svc._get_vram_from_logs("default", "llama3:latest")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_model_info_returns_models(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(
        200,
        {"models": [{"name": "llama3:latest", "context_length": 4096}, {"name": "mistral:latest"}]},
    )
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_loaded_model_info("default")

    assert result == {"llama3:latest": 4096, "mistral:latest": 0}


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_model_info_non_200_returns_empty(mock_fetch: AsyncMock):
    mock_fetch.return_value = _make_fetch_result(503, {})
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_loaded_model_info("default")

    assert result == {}


@pytest.mark.asyncio
@patch("server.services.ollama_service.fetch_from", new_callable=AsyncMock)
async def test_get_loaded_model_info_exception_returns_empty(mock_fetch: AsyncMock):
    mock_fetch.side_effect = RuntimeError("connection error")
    svc = _make_service()

    with patch.object(svc, "get_instance_installed_info", return_value=_make_installed_info()):
        result = await svc.get_loaded_model_info("default")

    assert result == {}


@pytest.mark.asyncio
async def test_resolve_vram_info_not_loaded_returns_false_none():
    svc = _make_service()

    result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
        "default", "llama3:latest", 4096, loaded_info={}, sizes={}, base_url="http://localhost:11434", num_parallel=None
    )

    assert result == (False, None)


@pytest.mark.asyncio
async def test_resolve_vram_info_vram_from_logs():
    svc = _make_service()

    with (
        patch.object(svc, "_get_vram_from_logs", new=AsyncMock(return_value=4.7)),
        patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=9.5)),
    ):
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": 4096},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
        )

    assert result == (True, 4.7)


@pytest.mark.asyncio
async def test_resolve_vram_info_falls_back_to_estimate():
    svc = _make_service()
    sizes = {"llama3:latest": ModelSize(8 * 1024**3, 8_000_000_000, 4.85)}

    with (
        patch.object(svc, "_get_vram_from_logs", new=AsyncMock(return_value=None)),
        patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=9.5)) as mock_estimate,
    ):
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            4096,
            loaded_info={"llama3:latest": 4096},
            sizes=sizes,
            base_url="http://localhost:11434",
            num_parallel=2,
        )

    assert result == (True, 9.5)
    assert mock_estimate.call_count == 1
    assert mock_estimate.call_args.args == ("default", "http://localhost:11434", "llama3:latest", 8 * 1024**3, 4096, 8_000_000_000, 4.85, 2)


@pytest.mark.asyncio
async def test_resolve_vram_info_uses_model_context_as_fallback():
    svc = _make_service()

    with (
        patch.object(svc, "_get_vram_from_logs", new=AsyncMock(return_value=None)),
        patch.object(svc, "_get_vram_estimate", new=AsyncMock(return_value=5.0)) as mock_estimate,
    ):
        result = await svc._resolve_vram_info(  # pyright: ignore[reportPrivateUsage]
            "default",
            "llama3:latest",
            8192,
            loaded_info={"llama3:latest": 0},
            sizes={},
            base_url="http://localhost:11434",
            num_parallel=None,
        )

    assert result == (True, 5.0)
    assert mock_estimate.call_args.args[4] == 8192


@pytest.mark.asyncio
async def test_resolve_blob_non_root_prefix():
    svc = _make_service()
    svc._model_manifest_loaded = True  # pyright: ignore[reportPrivateUsage]
    svc._blob_to_model = {"sha256:abc123": "library/llama3:latest"}  # pyright: ignore[reportPrivateUsage]

    result = await svc.resolve_blob("/home/ollama/.ollama/blobs/sha256-abc123")

    assert result == "llama3:latest"


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
    svc._log_cache["ollama-default"] = (time.monotonic(), "cached output")  # pyright: ignore[reportPrivateUsage]

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
    svc._log_cache["ollama-default"] = (time.monotonic() - 100, "stale logs")  # pyright: ignore[reportPrivateUsage]

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
