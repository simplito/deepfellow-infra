# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware module."""

from dataclasses import dataclass

import cpuinfo  # type: ignore

from server.utils.core import Utils


@dataclass
class HardwarePartInfo:
    pass


@dataclass
class CpuInfo(HardwarePartInfo):
    avx512: bool


@dataclass
class GpuInfo(HardwarePartInfo):
    name: str
    vram: str | None

    @property
    def long_name(self) -> str:
        """Return long_name."""
        return f"{self.name}{f' | {self.vram}' if self.vram else ''}"


@dataclass
class NvidiaGpuInfo(GpuInfo):
    id: int

    @property
    def long_name(self) -> str:
        """Return long_name."""
        return f"{self.name}{f' | {self.vram}' if self.vram else ''} | {self.id}"


@dataclass
class AmdGpuInfo(GpuInfo):
    pass


@dataclass
class IntelGpuInfo(GpuInfo):
    pass


def is_cpu_has_avx512() -> bool:
    """Chech if CPU supports avx512 instructions."""
    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])

    return any(flag.startswith("avx512") for flag in flags)


async def get_cpu_info() -> CpuInfo:
    """Get CPU info."""
    avx512: bool = is_cpu_has_avx512()
    return CpuInfo(avx512)


def convert_mib_to_gb(mib: str) -> str:
    """Convert MiB to GB, with specialized rounding for values < 1GB.

    Eg: "16303 MiB" -> "16 GB"
    Eg: "400 MiB" -> "512 MB"
    """
    mib_int = int(mib.split()[0])

    # Handle values less than 1 GB (1024 MiB)
    if mib_int < 800:
        # Define the allowed thresholds
        thresholds = [512, 256, 128, 64]

        # Find the threshold with the minimum absolute difference
        # This rounds to the nearest step (e.g., 400 becomes 512)
        best_fit = min(thresholds, key=lambda x: abs(x - mib_int))
        return f"{best_fit} MB"

    # Standard rounding for values >= 1 GB
    gb_int = round(mib_int / 1024)
    return f"{gb_int} GB"


async def get_nvidia_gpu_info_raw() -> str:
    """Get Nvidia GPU info."""
    cmd = "docker run --gpus all --rm gcr.io/distroless/base nvidia-smi --query-gpu=index,name,memory.total --format=csv"
    try:
        result = await Utils.run_command_for_success(cmd)
    except RuntimeError:
        return ""

    return result.stdout


def create_nvidia_gpu_info_list(raw_info: str) -> list[NvidiaGpuInfo]:
    """From raw info create list of Nvidia GPU Info."""
    gpus_info: list[NvidiaGpuInfo] = []
    for i, line in enumerate(raw_info.split("\n")):
        if i == 0:
            # skip headers line
            continue

        if not line:
            continue

        parts = line.split(",")
        id = int(parts[0])
        name = parts[1].strip()
        raw_vram = parts[2].strip()
        vram = None if raw_vram == "[N/A]" else convert_mib_to_gb(parts[2])
        gpu_info = NvidiaGpuInfo(name, vram, id)
        gpus_info.append(gpu_info)

    return gpus_info


async def get_nvidia_gpus_info() -> list[NvidiaGpuInfo]:
    """Get NVIDIA gpu info."""
    raw_info = await get_nvidia_gpu_info_raw()
    return create_nvidia_gpu_info_list(raw_info) if raw_info else []


async def get_hardware_info() -> list[HardwarePartInfo]:
    """Return hardware info."""
    cpu_info = await get_cpu_info()
    nvidia_gpus_info = await get_nvidia_gpus_info()
    return [cpu_info, *nvidia_gpus_info]


class Hardware:
    is_info_collected: bool
    _parts: list[HardwarePartInfo]
    _cpu: CpuInfo
    _gpus: list[GpuInfo]
    _nvidia_gpus: list[NvidiaGpuInfo]
    _amd_gpus: list[AmdGpuInfo]
    _intel_gpus: list[IntelGpuInfo]

    def __init__(self) -> None:
        self.is_info_collected = False
        self._parts = []
        self._gpus = []
        self._nvidia_gpus = []
        self._amd_gpus = []
        self._intel_gpus = []

    async def init_async(self) -> None:
        """Init async."""
        self._cpu = await get_cpu_info()
        self._nvidia_gpus = await get_nvidia_gpus_info()
        # Ensure these are assigned (even if empty) before calling set_gpus
        self._amd_gpus = []
        self._intel_gpus = []

        self._gpus = self.set_gpus()
        self._parts = self.set_info()
        self.is_info_collected = True

    @property
    def parts(self) -> list[HardwarePartInfo]:
        """Returns a combined list of all hardware parts (CPU and all GPUs)."""
        self._ensure_collected()
        return self._parts

    @property
    def cpu(self) -> CpuInfo:
        """Returns the CPU information. Raises RuntimeError if info is not yet collected."""
        self._ensure_collected()
        return self._cpu

    @property
    def gpus(self) -> list[GpuInfo]:
        """Returns a list of all detected GPUs regardless of manufacturer."""
        self._ensure_collected()
        return self._gpus

    @property
    def nvidia_gpus(self) -> list[NvidiaGpuInfo]:
        """Returns a list specifically containing NVIDIA GPU information."""
        self._ensure_collected()
        return self._nvidia_gpus

    @property
    def amd_gpus(self) -> list[AmdGpuInfo]:
        """Returns a list specifically containing AMD GPU information."""
        self._ensure_collected()
        return self._amd_gpus

    @property
    def intel_gpus(self) -> list[IntelGpuInfo]:
        """Returns a list specifically containing Intel GPU information."""
        self._ensure_collected()
        return self._intel_gpus

    @property
    def has_gpu_support(self) -> bool:
        """Return whether current machine supports GPU."""
        return bool(self.gpus)

    # --- Helpers ---

    def _ensure_collected(self) -> None:
        """Ensure data is ready before access."""
        if not self.is_info_collected:
            raise RuntimeError("Hardware info has not been collected. Call await init_async() first.")

    def set_gpus(self) -> list[GpuInfo]:
        """Combine all manufacturer-specific GPU lists into a single list."""
        return [*self._nvidia_gpus, *self._amd_gpus, *self._intel_gpus]

    def set_info(self) -> list[HardwarePartInfo]:
        """Combine CPU and GPU data into the master parts list."""
        return [self._cpu, *self._gpus]
