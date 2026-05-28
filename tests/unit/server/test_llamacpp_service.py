# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services.llamacpp_service import LLamacppOptions, LLamacppService
from server.utils.vram_calculator import ArchParams


@pytest.mark.parametrize(
    ("mib", "expected"),
    [
        (0.0, None),
        (4096.0, 4.0),
        (8192.0, 8.0),
        (512.5, round(512.5 / 1024, 2)),
    ],
)
def test_mib_to_gib(mib: float, expected: float | None):
    result = LLamacppService.mib_to_gib(mib)

    assert result == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "CUDA0 model buffer size = 4200.50 MiB\nCUDA0 KV buffer size = 512.00 MiB\nCUDA0 compute buffer size = 200.00 MiB\n",
            4912.5,
        ),
        (
            "CUDA0 model buffer size = 4000.00 MiB\nCUDA1 model buffer size = 4000.00 MiB\nCUDA0 KV buffer size = 256.00 MiB\nCUDA1 KV buffer size = 256.00 MiB\n",  # noqa: E501
            8512.0,
        ),
        ("", 0.0),
        ("some unrelated log output\nno matching lines here\n", 0.0),
    ],
)
def test_parse_llamacpp_vram_mib(raw: str, expected: float):
    result = LLamacppService._parse_llamacpp_vram_mib(raw)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


ARCH = ArchParams(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    num_hidden_layers=32,
)


def _make_service_with_model(
    model_path: Path,
    context_window: int | None = 4096,
    options: LLamacppOptions | None = None,
) -> LLamacppService:
    model_info = MagicMock()
    model_info.model_path = model_path
    model_info.context_window = context_window
    installed_info = MagicMock()
    installed_info.models = {"my-model": model_info}
    installed_info.parsed_options = options or LLamacppOptions()
    svc = object.__new__(LLamacppService)
    svc.get_instance_installed_info = MagicMock(return_value=installed_info)
    return svc


@pytest.mark.asyncio
@patch("server.services.llamacpp_service.get_gguf_arch_params", new_callable=AsyncMock)
@patch("server.services.llamacpp_service.estimate_vram_gb")
async def test_get_vram_estimate_returns_value(mock_estimate: MagicMock, mock_arch: AsyncMock):
    model_path = MagicMock(spec=Path)
    model_path.stat.return_value.st_size = 8 * 1024**3
    mock_arch.return_value = ARCH
    mock_estimate.return_value = 9.5
    ctx_size = 4096
    svc = _make_service_with_model(model_path, context_window=ctx_size)

    result = await svc._get_vram_estimate("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result == 9.5
    assert mock_arch.call_count == 1
    assert mock_arch.call_args.args[0] == model_path
    mock_estimate.assert_called_once_with(ARCH, 8 * 1024**3, ctx_size, 16, 1)


@pytest.mark.asyncio
@patch("server.services.llamacpp_service.get_gguf_arch_params", new_callable=AsyncMock)
@patch("server.services.llamacpp_service.estimate_vram_gb")
async def test_get_vram_estimate_uses_cache_type_and_parallel(mock_estimate: MagicMock, mock_arch: AsyncMock):
    model_path = MagicMock(spec=Path)
    model_path.stat.return_value.st_size = 8 * 1024**3
    mock_arch.return_value = ARCH
    mock_estimate.return_value = 7.0
    options = LLamacppOptions(kv_cache_type="q8_0", num_parallel=4)
    svc = _make_service_with_model(model_path, context_window=2048, options=options)

    await svc._get_vram_estimate("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    mock_estimate.assert_called_once_with(ARCH, 8 * 1024**3, 2048, 8, 4)


@pytest.mark.asyncio
async def test_get_vram_estimate_unknown_model_returns_none():
    installed_info = MagicMock()
    installed_info.models = {}
    svc = object.__new__(LLamacppService)
    svc.get_instance_installed_info = MagicMock(return_value=installed_info)

    result = await svc._get_vram_estimate("inst", "missing-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.llamacpp_service.get_gguf_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_no_arch_returns_none(mock_arch: AsyncMock):
    model_path = MagicMock(spec=Path)
    mock_arch.return_value = None
    svc = _make_service_with_model(model_path)

    result = await svc._get_vram_estimate("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.llamacpp_service.get_gguf_arch_params", new_callable=AsyncMock)
async def test_get_vram_estimate_missing_file_returns_none(mock_arch: AsyncMock):
    model_path = MagicMock(spec=Path)
    model_path.stat.side_effect = FileNotFoundError
    mock_arch.return_value = ARCH
    svc = _make_service_with_model(model_path)

    result = await svc._get_vram_estimate("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


def _make_service_with_container(container_name: str) -> LLamacppService:
    model_info = MagicMock()
    model_info.docker.container_name = container_name
    installed_info = MagicMock()
    installed_info.models = {"my-model": model_info}
    svc = object.__new__(LLamacppService)
    svc.get_instance_installed_info = MagicMock(return_value=installed_info)
    svc._log_cache = {}  # pyright: ignore[reportPrivateUsage]
    return svc


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_value():
    svc = _make_service_with_container("my-container")
    raw = "CUDA0 model buffer size = 4096.00 MiB\n"

    with patch.object(svc, "_get_docker_logs", new=AsyncMock(return_value=raw)):
        result = await svc._get_vram_from_logs("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result == 4.0


@pytest.mark.asyncio
async def test_get_vram_from_logs_unknown_model_returns_none():
    installed_info = MagicMock()
    installed_info.models = {}
    svc = object.__new__(LLamacppService)
    svc.get_instance_installed_info = MagicMock(return_value=installed_info)
    svc._log_cache = {}  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_vram_from_logs("inst", "missing-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_vram_from_logs_no_container_returns_none():
    svc = _make_service_with_container("")

    result = await svc._get_vram_from_logs("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_vram_from_logs_command_error_returns_none():
    svc = _make_service_with_container("my-container")

    with patch.object(svc, "_get_docker_logs", new=AsyncMock(side_effect=RuntimeError("docker error"))):
        result = await svc._get_vram_from_logs("inst", "my-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_cache_hit_skips_run_command(mock_utils: MagicMock):
    mock_utils.run_command = AsyncMock(return_value=MagicMock(stdout="log", stderr=""))

    svc = object.__new__(LLamacppService)
    svc._log_cache = {"my-container": (time.monotonic(), "cached log")}  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "cached log"
    mock_utils.run_command.assert_not_called()


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_cache_miss_calls_run_command(mock_utils: MagicMock):
    mock_result = MagicMock()
    mock_result.stdout = "fresh log"
    mock_result.stderr = ""
    mock_utils.run_command = AsyncMock(return_value=mock_result)

    svc = object.__new__(LLamacppService)
    svc._log_cache = {}  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "fresh log"
    mock_utils.run_command.assert_called_once()


@pytest.mark.asyncio
@patch("server.services.base2_service.Utils", new_callable=MagicMock)
async def test_get_docker_logs_cache_expired_calls_run_command(mock_utils: MagicMock):
    mock_result = MagicMock()
    mock_result.stdout = "new log"
    mock_result.stderr = ""
    mock_utils.run_command = AsyncMock(return_value=mock_result)

    svc = object.__new__(LLamacppService)
    svc._log_cache = {"my-container": (time.monotonic() - 100, "stale log")}  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "new log"
    mock_utils.run_command.assert_called_once()
