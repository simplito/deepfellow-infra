# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.models.services import GpuCardStats, GpuStats
from server.utils.core import CommandResult2
from server.utils.hardware import (
    CpuInfo,
    GpuInfo,
    Hardware,
    IntelGpuInfo,
    NvidiaGpuInfo,
    _get_intel_gpu_name,  # pyright: ignore[reportPrivateUsage]
    convert_mib_to_gb,
    create_nvidia_gpu_info_list,
    get_cpu_info,
    get_hardware_info,
    get_intel_gpus_info,
    get_nvidia_gpu_info_raw,
    get_nvidia_gpus_info,
    get_vram_gb,
    is_cpu_has_avx512,
)


@pytest.mark.parametrize(
    ("nvida_smi_output", "expectation"),
    [
        (
            "index, name, memory.total [MiB]\n0, NVIDIA GB10, [N/A]",
            [
                (NvidiaGpuInfo(id=0, name="NVIDIA GB10", vram=None), "NVIDIA GB10 | 0"),
            ],
        ),
        (
            "index, name, memory.total [MiB]\n0, NVIDIA GeForce RTX 5070 Ti, 16303 MiB",
            [
                (NvidiaGpuInfo(id=0, name="NVIDIA GeForce RTX 5070 Ti", vram="16 GB"), "NVIDIA GeForce RTX 5070 Ti | 16 GB | 0"),
            ],
        ),
        (
            "index, name, memory.total [MiB]\n0, NVIDIA GeForce RTX 5070 Ti, 16303 MiB\n1, NVIDIA GeForce RTX 5070 Ti, 16303 MiB",
            [
                (NvidiaGpuInfo(id=0, name="NVIDIA GeForce RTX 5070 Ti", vram="16 GB"), "NVIDIA GeForce RTX 5070 Ti | 16 GB | 0"),
                (NvidiaGpuInfo(id=1, name="NVIDIA GeForce RTX 5070 Ti", vram="16 GB"), "NVIDIA GeForce RTX 5070 Ti | 16 GB | 1"),
            ],
        ),
        (
            "index, name, memory.total [MiB]\n0, NVIDIA GeForce RTX 4090, 24564 MiB\n1, NVIDIA GeForce RTX 4090, 24564 MiB",
            [
                (NvidiaGpuInfo(id=0, name="NVIDIA GeForce RTX 4090", vram="24 GB"), "NVIDIA GeForce RTX 4090 | 24 GB | 0"),
                (NvidiaGpuInfo(id=1, name="NVIDIA GeForce RTX 4090", vram="24 GB"), "NVIDIA GeForce RTX 4090 | 24 GB | 1"),
            ],
        ),
    ],
)
def test_join_url(nvida_smi_output: str, expectation: list[tuple[NvidiaGpuInfo, str]]):
    result = create_nvidia_gpu_info_list(nvida_smi_output)

    assert len(result) == len(expectation), f"Lists size do not match {len(result)} != {len(expectation)}"
    for i, (res, exp) in enumerate(zip(result, expectation, strict=False)):
        assert res.id == exp[0].id, f"Id on index does not match index={i}: '{res.id}' != '{exp[0].id}'"
        assert res.name == exp[0].name, f"Name on index does not match index={i}: '{res.name}' != '{exp[0].name}'"
        assert res.vram == exp[0].vram, f"VRAM on index does not match index={i}: '{res.vram}' != '{exp[0].vram}'"
        assert res.long_name == exp[0].long_name, f"Long name on index does not match index={i}: '{res.long_name}' != '{exp[0].long_name}'"
        assert res.long_name == exp[1], f"Long name (2) on index does not match index={i}: '{res.long_name}' != '{exp[1]}'"


@pytest.mark.parametrize(
    ("device_id_content", "expected"),
    [
        ("0xe20b", "Intel GPU (0xe20b)"),
        ("0x56a1", "Intel GPU (0x56a1)"),
        (None, "Intel GPU"),
    ],
)
def test_get_intel_gpu_name(tmp_path: Path, device_id_content: str | None, expected: str):
    card_path = tmp_path / "card1"
    card_path.mkdir()
    device_dir = card_path / "device"
    device_dir.mkdir()
    if device_id_content is not None:
        (device_dir / "device").write_text(device_id_content)

    result = _get_intel_gpu_name(card_path)

    assert result == expected


@pytest.mark.parametrize(
    ("name", "vram", "id", "expected_long_name"),
    [
        ("Intel GPU (0xe20b)", None, 1, "Intel GPU (0xe20b) | 1"),
        ("Intel GPU (0x56a1)", None, 0, "Intel GPU (0x56a1) | 0"),
        ("Intel GPU", None, 2, "Intel GPU | 2"),
    ],
)
def test_intel_gpu_info_long_name(name: str, vram: str | None, id: int, expected_long_name: str):
    gpu = IntelGpuInfo(name=name, vram=vram, id=id)

    assert gpu.long_name == expected_long_name


def _make_hardware() -> Hardware:
    hw = Hardware()
    hw.is_info_collected = True
    hw._gpus = []  # pyright: ignore[reportPrivateUsage]
    hw._nvidia_gpus = []  # pyright: ignore[reportPrivateUsage]
    hw._amd_gpus = []  # pyright: ignore[reportPrivateUsage]
    hw._intel_gpus = []  # pyright: ignore[reportPrivateUsage]
    hw._total_vram_gb = 0.0  # pyright: ignore[reportPrivateUsage]
    return hw


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_realtime_stats_single_gpu(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult2(stdout="NVIDIA GeForce RTX 4090, 24576, 8192\n", stderr="")

    result = await _make_hardware().get_realtime_stats()

    assert result == GpuStats(total_vram_gb=24.0, used_vram_gb=8.0, gpus=None)


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_realtime_stats_two_gpus_sums_totals(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult2(stdout="NVIDIA RTX 4090, 24576, 4096\nNVIDIA RTX 4090, 24576, 8192\n", stderr="")

    result = await _make_hardware().get_realtime_stats()

    assert result is not None
    assert result.total_vram_gb == 48.0
    assert result.used_vram_gb == 12.0
    assert result.gpus == [
        GpuCardStats(name="NVIDIA RTX 4090", total_vram_gb=24.0, used_vram_gb=4.0),
        GpuCardStats(name="NVIDIA RTX 4090", total_vram_gb=24.0, used_vram_gb=8.0),
    ]


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_realtime_stats_empty_output_returns_none(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult2(stdout="", stderr="")

    result = await _make_hardware().get_realtime_stats()

    assert result is None


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_realtime_stats_command_fails_returns_none(mock_cmd: AsyncMock):
    mock_cmd.side_effect = RuntimeError

    result = await _make_hardware().get_realtime_stats()

    assert result is None


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_realtime_stats_malformed_lines_are_skipped(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult2(stdout="bad line\nNVIDIA RTX 4090, 24576, 8192\nnot,enough\n", stderr="")

    result = await _make_hardware().get_realtime_stats()

    assert result is not None
    assert result.total_vram_gb == 24.0
    assert result.gpus is None


@pytest.mark.asyncio
async def test_get_amd_stats_returns_cards():
    hw = _make_hardware()
    total_bytes = 8 * 1024**3
    used_bytes = 3 * 1024**3
    rocm_output = f"0, {total_bytes}, {used_bytes}\n"

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock) as mock_cmd:
        mock_cmd.return_value = CommandResult2(stdout=rocm_output, stderr="")
        cards = await hw._get_amd_stats()  # pyright: ignore[reportPrivateUsage]

    assert cards is not None
    assert len(cards) == 1
    assert cards[0].name == "AMD GPU 0"
    assert cards[0].total_vram_gb == pytest.approx(8.0, abs=0.1)
    assert cards[0].used_vram_gb == pytest.approx(3.0, abs=0.1)


@pytest.mark.asyncio
async def test_get_amd_stats_tool_unavailable_returns_none():
    hw = _make_hardware()

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock) as mock_cmd:
        mock_cmd.side_effect = RuntimeError("rocm-smi not found")
        cards = await hw._get_amd_stats()  # pyright: ignore[reportPrivateUsage]

    assert cards is None


@pytest.mark.asyncio
async def test_get_realtime_stats_falls_back_to_amd_when_nvidia_unavailable():
    hw = _make_hardware()
    total_bytes = 16 * 1024**3
    used_bytes = 4 * 1024**3
    rocm_output = f"0, {total_bytes}, {used_bytes}\n"

    amd_ok = CommandResult2(stdout=rocm_output, stderr="")

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock) as mock_cmd:
        mock_cmd.side_effect = [RuntimeError("no nvidia"), amd_ok]
        result = await hw.get_realtime_stats()

    assert result is not None
    assert result.total_vram_gb == pytest.approx(16.0, abs=0.1)
    assert result.used_vram_gb == pytest.approx(4.0, abs=0.1)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [({"flags": ["avx512f", "sse4_2"]}, True), ({"flags": ["sse4_2", "avx2"]}, False), ({"flags": []}, False), ({}, False)],
)
def test_is_cpu_has_avx512(flags: dict[str, list[str]], expected: bool):
    with patch("server.utils.hardware.cpuinfo.get_cpu_info", return_value=flags):
        assert is_cpu_has_avx512() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("avx512", [True, False])
async def test_get_cpu_info_avx512(avx512: bool):
    with patch("server.utils.hardware.is_cpu_has_avx512", return_value=avx512):
        result = await get_cpu_info()
    assert result == CpuInfo(avx512=avx512)


@pytest.mark.parametrize(
    ("mib_str", "expected"),
    [
        ("400 MiB", "512 MB"),
        ("300 MiB", "256 MB"),
        ("16303 MiB", "16 GB"),
        ("1024 MiB", "1 GB"),
    ],
)
def test_convert_mib_to_gb(mib_str: str, expected: str):
    assert convert_mib_to_gb(mib_str) == expected


@pytest.mark.parametrize(
    ("vram_str", "expected"),
    [
        ("16 GB", 16.0),
        ("512 MB", 512 / 1024),
        ("1 TB", 1024.0),
        (None, 0.0),
        ("", 0.0),
        ("unknown", 0.0),
    ],
)
def test_get_vram_gb(vram_str: str | None, expected: float):
    assert get_vram_gb(vram_str) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_get_nvidia_gpu_info_raw_success():
    mock_result = MagicMock()
    mock_result.stdout = "index, name, memory.total [MiB]\n0, NVIDIA RTX 4090, 24564 MiB"

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock, return_value=mock_result):
        result = await get_nvidia_gpu_info_raw()

    assert result == mock_result.stdout


@pytest.mark.asyncio
async def test_get_nvidia_gpu_info_raw_failure():
    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock, side_effect=RuntimeError("docker not found")):
        result = await get_nvidia_gpu_info_raw()

    assert result == ""


@pytest.mark.asyncio
async def test_get_nvidia_gpus_info_non_empty():
    raw = "index, name, memory.total [MiB]\n0, NVIDIA RTX 4090, 24564 MiB"

    with patch("server.utils.hardware.get_nvidia_gpu_info_raw", new_callable=AsyncMock, return_value=raw):
        result = await get_nvidia_gpus_info()

    assert len(result) == 1
    assert result[0].name == "NVIDIA RTX 4090"


@pytest.mark.asyncio
async def test_get_nvidia_gpus_info_empty():
    with patch("server.utils.hardware.get_nvidia_gpu_info_raw", new_callable=AsyncMock, return_value=""):
        result = await get_nvidia_gpus_info()

    assert result == []


def _make_intel_sysfs(
    tmp_path: Path, card_name: str = "card0", vendor: str = "0x8086", boot_vga: str = "0", device_id: str = "0xe20b"
) -> Path:
    device_dir = tmp_path / card_name / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "vendor").write_text(vendor)
    (device_dir / "boot_vga").write_text(boot_vga)
    (device_dir / "device").write_text(device_id)
    return tmp_path


@pytest.mark.asyncio
async def test_get_intel_gpus_info_no_drm_path(tmp_path: Path):
    non_existent = tmp_path / "nonexistent"

    with patch("server.utils.hardware.DRM_PATH", non_existent):
        result = await get_intel_gpus_info()

    assert result == []


@pytest.mark.asyncio
async def test_get_intel_gpus_info_skips_non_intel_vendor(tmp_path: Path):
    _make_intel_sysfs(tmp_path, vendor="0x1002")

    with patch("server.utils.hardware.DRM_PATH", tmp_path):
        result = await get_intel_gpus_info()

    assert result == []


@pytest.mark.asyncio
async def test_get_intel_gpus_info_skips_integrated_gpu(tmp_path: Path):
    _make_intel_sysfs(tmp_path, boot_vga="1")

    with patch("server.utils.hardware.DRM_PATH", tmp_path):
        result = await get_intel_gpus_info()

    assert result == []


@pytest.mark.asyncio
async def test_get_intel_gpus_info_finds_discrete_gpu(tmp_path: Path):
    _make_intel_sysfs(tmp_path, card_name="card0", vendor="0x8086", boot_vga="0", device_id="0xe20b")

    with patch("server.utils.hardware.DRM_PATH", tmp_path):
        result = await get_intel_gpus_info()

    assert len(result) == 1
    assert result[0].name == "Intel GPU (0xe20b)"
    assert result[0].id == 0
    assert result[0].vram is None


@pytest.mark.asyncio
async def test_get_intel_gpus_info_skips_non_card_entries(tmp_path: Path):
    (tmp_path / "renderD128").mkdir()

    with patch("server.utils.hardware.DRM_PATH", tmp_path):
        result = await get_intel_gpus_info()

    assert result == []


@pytest.mark.asyncio
async def test_get_hardware_info():
    cpu = CpuInfo(avx512=False)
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=0)

    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[gpu]),
    ):
        result = await get_hardware_info()

    assert result[0] == cpu
    assert result[1] == gpu
    assert len(result) == 2


def test_hardware_raises_before_init():
    hw = Hardware()
    with pytest.raises(RuntimeError):
        _ = hw.cpu


@pytest.mark.asyncio
async def test_hardware_init_async_no_gpus():
    cpu = CpuInfo(avx512=False)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[]),
    ):
        hw = Hardware()

        await hw.init_async()

    assert hw.has_gpu_support is False
    assert hw.total_vram_gb == 0.0
    assert hw.gpus == []


@pytest.mark.asyncio
async def test_hardware_init_async_with_nvidia_gpu():
    cpu = CpuInfo(avx512=True)
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=0)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[gpu]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[]),
    ):
        hw = Hardware()

        await hw.init_async()

    assert hw.has_gpu_support is True
    assert hw.nvidia_gpus == [gpu]
    assert hw.cpu.avx512 is True


@pytest.mark.asyncio
async def test_hardware_init_async_with_intel_gpu():
    cpu = CpuInfo(avx512=False)
    gpu = IntelGpuInfo(name="Intel GPU (0xe20b)", vram=None, id=0)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[gpu]),
    ):
        hw = Hardware()

        await hw.init_async()

    assert hw.intel_gpus == [gpu]
    assert hw.has_gpu_support is True


@pytest.mark.asyncio
async def test_hardware_total_vram_gb():
    cpu = CpuInfo(avx512=False)
    gpu0 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=0)
    gpu1 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=1)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[gpu0, gpu1]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[]),
    ):
        hw = Hardware()

        await hw.init_async()

    assert hw.total_vram_gb == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_hardware_parts():
    cpu = CpuInfo(avx512=False)
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=0)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[gpu]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[]),
    ):
        hw = Hardware()

        await hw.init_async()

    assert hw.parts == [cpu, gpu]


@pytest.mark.asyncio
async def test_hardware_amd_gpus_empty():
    cpu = CpuInfo(avx512=False)
    with (
        patch("server.utils.hardware.get_cpu_info", new_callable=AsyncMock, return_value=cpu),
        patch("server.utils.hardware.get_nvidia_gpus_info", new_callable=AsyncMock, return_value=[]),
        patch("server.utils.hardware.get_intel_gpus_info", new_callable=AsyncMock, return_value=[]),
    ):
        hw = Hardware()

        await hw.init_async()
    assert hw.amd_gpus == []


def test_gpu_info_long_name_with_vram():
    gpu = GpuInfo(name="Test GPU", vram="8 GB")

    assert gpu.long_name == "Test GPU | 8 GB"


def test_gpu_info_long_name_without_vram():
    gpu = GpuInfo(name="Test GPU", vram=None)

    assert gpu.long_name == "Test GPU"


def test_create_nvidia_gpu_info_list_skips_empty_lines():
    # Trailing newline produces an empty string that should be skipped (line 148)
    raw = "index, name, memory.total [MiB]\n0, NVIDIA RTX 4090, 24564 MiB\n"

    result = create_nvidia_gpu_info_list(raw)

    assert len(result) == 1
    assert result[0].name == "NVIDIA RTX 4090"


@pytest.mark.asyncio
@patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock)
async def test_get_nvidia_stats_skips_malformed_values(mock_cmd: AsyncMock):
    mock_cmd.return_value = CommandResult2(stdout="GPU Name, not_a_number, 4096\nNVIDIA RTX 4090, 24576, 8192\n", stderr="")

    result = await _make_hardware()._get_nvidia_stats()  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert len(result) == 1
    assert result[0].name == "NVIDIA RTX 4090"


@pytest.mark.asyncio
async def test_get_amd_stats_skips_lines_with_fewer_than_three_parts():
    hw = _make_hardware()
    total_bytes = 8 * 1024**3
    used_bytes = 2 * 1024**3
    # First line has only 2 parts (no comma-separated used), second is valid
    rocm_output = f"0,{total_bytes}\n1, {total_bytes}, {used_bytes}\n"

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock) as mock_cmd:
        mock_cmd.return_value = CommandResult2(stdout=rocm_output, stderr="")
        cards = await hw._get_amd_stats()  # pyright: ignore[reportPrivateUsage]

    assert cards is not None
    assert len(cards) == 1
    assert cards[0].name == "AMD GPU 1"


@pytest.mark.asyncio
async def test_get_amd_stats_skips_malformed_numeric_values():
    hw = _make_hardware()
    # Valid format but non-integer values trigger ValueError
    rocm_output = "0, not_a_number, 1073741824\n"

    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock) as mock_cmd:
        mock_cmd.return_value = CommandResult2(stdout=rocm_output, stderr="")
        cards = await hw._get_amd_stats()  # pyright: ignore[reportPrivateUsage]

    assert cards is None


@pytest.mark.asyncio
async def test_get_realtime_stats_returns_cached_result():
    hw = _make_hardware()
    cached_stats = GpuStats(total_vram_gb=16.0, used_vram_gb=4.0, gpus=None)
    hw._realtime_stats_cache = cached_stats  # pyright: ignore[reportPrivateUsage]
    hw._realtime_stats_cache_ts = float("inf")  # pyright: ignore[reportPrivateUsage]

    result = await hw.get_realtime_stats()

    assert result is cached_stats
