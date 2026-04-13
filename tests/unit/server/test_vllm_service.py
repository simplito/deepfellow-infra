# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from server.services.vllm_service import VllmModelOptions, VllmService


def build_cmd(
    docker_model_path: Path = Path("/mnt/hf/hub/my-model"),
    model_id: str = "Qwen/Qwen3-0.6B",
    opts: VllmModelOptions | None = None,
    quantization: str | None = None,
    gpu_memory_utilization: float | None = None,
    user_model_length: int | None = None,
    use_gpu: bool = True,
) -> list[str]:
    if opts is None:
        opts = VllmModelOptions()
    return VllmService._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        MagicMock(),
        docker_model_path=docker_model_path,
        model_id=model_id,
        opts=opts,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        user_model_length=user_model_length,
        use_gpu=use_gpu,
    )


def test_base_flags_always_present() -> None:
    """Core flags are always included regardless of options."""
    cmd = build_cmd(docker_model_path=Path("/mnt/hf/hub/my-model"), model_id="Qwen/Qwen3-0.6B")
    assert str(Path("/mnt/hf/hub/my-model")) in cmd
    assert "--host" in cmd
    assert "0.0.0.0" in cmd
    assert "--port" in cmd
    assert "8000" in cmd
    assert "--served-model-name" in cmd
    assert "Qwen/Qwen3-0.6B" in cmd


@pytest.mark.parametrize(
    ("use_gpu", "user_model_length", "expected_flag", "expected_value"),
    [
        (True, 4096, "--max-model-len", "4096"),
        (True, None, None, None),
        (False, 4096, "--max-model-len", "4096"),
        (False, None, "--disable-sliding-window", None),
    ],
)
def test_model_length_flags(
    use_gpu: bool,
    user_model_length: int | None,
    expected_flag: str | None,
    expected_value: str | None,
) -> None:
    """Correct max-model-len / disable-sliding-window flags based on GPU mode and length."""
    cmd = build_cmd(use_gpu=use_gpu, user_model_length=user_model_length)
    if expected_flag is None:
        assert "--max-model-len" not in cmd
        assert "--disable-sliding-window" not in cmd
    elif expected_value is None:
        assert expected_flag in cmd
    else:
        assert expected_flag in cmd
        idx = cmd.index(expected_flag)
        assert cmd[idx + 1] == expected_value


def test_quantization_added_when_set() -> None:
    cmd = build_cmd(quantization="fp8")
    idx = cmd.index("--quantization")
    assert cmd[idx + 1] == "fp8"


def test_quantization_omitted_when_none() -> None:
    cmd = build_cmd(quantization=None)
    assert "--quantization" not in cmd


def test_gpu_memory_utilization_added_when_set() -> None:
    cmd = build_cmd(gpu_memory_utilization=0.85)
    idx = cmd.index("--gpu-memory-utilization")
    assert cmd[idx + 1] == "0.85"


def test_gpu_memory_utilization_omitted_when_none() -> None:
    cmd = build_cmd(gpu_memory_utilization=None)
    assert "--gpu-memory-utilization" not in cmd


def test_extra_args_kv_flag_included() -> None:
    """extra_args key-value pairs are appended to the command."""
    opts = VllmModelOptions(extra_args={"--dtype": "bfloat16", "--tensor-parallel-size": "2"})
    cmd = build_cmd(opts=opts)
    idx = cmd.index("--dtype")
    assert cmd[idx + 1] == "bfloat16"
    idx = cmd.index("--tensor-parallel-size")
    assert cmd[idx + 1] == "2"


def test_extra_args_boolean_flag_included() -> None:
    """extra_args entry with empty value produces a standalone flag."""
    opts = VllmModelOptions(extra_args={"--enable-prefix-caching": ""})
    cmd = build_cmd(opts=opts)
    assert "--enable-prefix-caching" in cmd
    idx = cmd.index("--enable-prefix-caching")
    # next token must not be another flag value pair for this flag
    assert idx == len(cmd) - 1 or cmd[idx + 1].startswith("--") or cmd[idx + 1] in ("0.0.0.0", "8000")


def test_extra_args_empty_when_not_set() -> None:
    cmd = build_cmd(opts=VllmModelOptions())
    # only built-in flags should be present
    assert "--dtype" not in cmd
    assert "--enable-prefix-caching" not in cmd
