# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

import server.utils.hardware as hw
from server.utils.hardware import (
    IntelGpuInfo,
    NvidiaGpuInfo,
    _get_intel_gpu_vram_mib,  # pyright: ignore[reportPrivateUsage]
    create_nvidia_gpu_info_list,
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
    ("vram0_mm_content", "expected"),
    [
        ("chunk_size: 4KiB, total: 16304MiB, free: 16282MiB, clear_free: 0MiB", "16304 MiB"),
        ("chunk_size: 4KiB, total: 14336MiB, free: 14000MiB, clear_free: 0MiB", "14336 MiB"),
        ("chunk_size: 4KiB, total:   6144MiB, free: 6000MiB, clear_free: 0MiB", "6144 MiB"),
        ("some unrelated content without total", None),
    ],
)
def test_get_intel_gpu_vram_mib(tmp_path: Path, vram0_mm_content: str, expected: str | None):
    original = hw.DRM_DEBUG_PATH
    hw.DRM_DEBUG_PATH = tmp_path
    try:
        card_dir = tmp_path / "1"
        card_dir.mkdir()
        (card_dir / "vram0_mm").write_text(vram0_mm_content)
        assert _get_intel_gpu_vram_mib("1") == expected
    finally:
        hw.DRM_DEBUG_PATH = original


@pytest.mark.parametrize(
    ("name", "vram", "id", "expected_long_name"),
    [
        ("Intel Corporation Battlemage G21 [Intel Graphics]", "16 GB", 1, "Intel Corporation Battlemage G21 [Intel Graphics] | 16 GB | 1"),
        ("Intel Corporation Arc A770", "16 GB", 0, "Intel Corporation Arc A770 | 16 GB | 0"),
        ("Intel Corporation Device e212", None, 2, "Intel Corporation Device e212 | 2"),
    ],
)
def test_intel_gpu_info_long_name(name: str, vram: str | None, id: int, expected_long_name: str):
    gpu = IntelGpuInfo(name=name, vram=vram, id=id)
    assert gpu.long_name == expected_long_name
