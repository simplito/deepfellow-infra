# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.utils.hardware import NvidiaGpuInfo, create_nvidia_gpu_info_list


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
