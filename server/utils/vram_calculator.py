# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""VRAM estimator for LLM models.

Formula ported from https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator
"""

import re
from dataclasses import dataclass

GGUF_QUANTS: dict[str, float] = {
    "IQ1_S": 1.56,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.5,
    "IQ2_M": 2.7,
    "IQ3_XXS": 3.06,
    "IQ3_XS": 3.3,
    "Q2_K": 3.35,
    "Q3_K_S": 3.5,
    "IQ3_S": 3.5,
    "IQ3_M": 3.7,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "Q4_0": 4.55,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.85,
    "Q5_0": 5.54,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.59,
    "Q8_0": 8.5,
}

MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}

GIB = 2**30
MIB = 2**20


@dataclass
class ArchParams:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    sliding_window: int | None = None


def parse_cache_type_bits(cache_type: str, default: int = 16) -> int:
    """Parse bit width from a cache type string (e.g. 'f16' -> 16, 'q8_0' -> 8)."""
    m = re.search(r"[fq](\d+)", cache_type)
    return int(m.group(1)) if m else default


def parse_parameter_count(s: str) -> int | None:
    """Parse Ollama parameter_size string (e.g. '999.89M', '7B') to integer count."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMBT]?)", s.strip(), re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * MULTIPLIERS.get(m.group(2).upper(), 1))


def cal_kv_cache_bytes(arch: ArchParams, num_ctx: int, cache_bit: int = 16, num_parallel: int = 1) -> float:
    """Return KV cache size in bytes.

    Stores key and value tensors for all past tokens so they are not recomputed.

    Args:
        arch: Model architecture parameters.
            num_attention_heads — total number of query attention heads.
            num_key_value_heads — number of K/V heads (may be smaller due to GQA).
            num_hidden_layers   — number of transformer layers.
        num_ctx: Context window length in tokens.
        cache_bit: Precision of KV cache storage, default 16 (fp16). Can be 8 or 4.
        num_parallel: Number of parallel request slots (sequences held in memory at once).

    Formula:
        n_gqa      = num_attention_heads / num_key_value_heads
          GQA (Grouped Query Attention) — multiple query heads share one K/V head to save memory.
          Example: Llama 3 8B has 32 query heads and 8 K/V heads, so n_gqa = 4.
          Standard MHA has n_gqa = 1 (every query head has its own K/V head).

        n_embed_gqa = hidden_size / n_gqa    (effective K/V embedding dimension after GQA reduction)
        n_elements  = n_embed_gqa * num_hidden_layers * num_ctx
        kv_cache   = 2 * n_elements * (cache_bit / 8) * num_parallel
          Factor 2 accounts for both K and V tensors.
    """
    n_gqa = arch.num_attention_heads / arch.num_key_value_heads
    n_embd_gqa = arch.hidden_size / n_gqa
    n_elements = n_embd_gqa * arch.num_hidden_layers * num_ctx
    size = 2 * n_elements
    return size * (cache_bit / 8) * num_parallel


def cal_input_buffer_bytes(arch: ArchParams, num_ctx: int, batch_size: int = 512) -> float:
    """Return input buffer size in bytes.

    Derived from llama.cpp source (ggml tensor allocations at context creation).
    Each tensor element is 4 bytes (i32 for index tensors, f32 for value tensors).
    """
    bytes_per_element = 4  # i32 / f32
    input_tokens = batch_size * bytes_per_element
    input_embed = arch.hidden_size * batch_size * bytes_per_element
    input_pos = batch_size * bytes_per_element
    input_attention_mask = num_ctx * batch_size * bytes_per_element
    input_rope_shift = num_ctx * bytes_per_element
    input_row_sum = batch_size * bytes_per_element
    return input_tokens + input_embed + input_pos + input_attention_mask + input_rope_shift + input_row_sum


def cal_compute_buffer_bytes(arch: ArchParams, num_ctx: int) -> float:
    """Return compute buffer size in bytes.

    Temporary workspace allocated by llama.cpp for attention score computation.
    Formula is empirical and hardcoded for bsz=512 in the original calculator.
    0.75 Value is probably a heuristic.
    """
    return (num_ctx / 1024 * 2 + 0.75) * arch.num_attention_heads * MIB


def cal_context_size_bytes(
    arch: ArchParams,
    num_ctx: int,
    cache_bit: int = 16,
    num_parallel: int = 1,
) -> float:
    """Return total context size in bytes (input buffer + KV cache + compute buffer)."""
    return (
        cal_input_buffer_bytes(arch, num_ctx)
        + cal_kv_cache_bytes(arch, num_ctx, cache_bit, num_parallel)
        + cal_compute_buffer_bytes(arch, num_ctx)
    )


def cal_model_size_bytes(
    weights_bytes: int,
    parameters: int | None,
    bits_per_weight: float | None,
) -> float:
    """Return model size preferring calculated value over reported one. Return value in bytes (not bits)."""
    if parameters and bits_per_weight:
        return parameters * bits_per_weight / 8

    return float(weights_bytes)


def estimate_vram_gb(
    arch: ArchParams,
    weights_bytes: int,
    num_ctx: int | None = None,
    cache_bit: int = 16,
    num_parallel: int = 1,
    parameters: int | None = None,
    bits_per_weight: float | None = None,
) -> float | None:
    """Return estimated VRAM usage per model."""
    if not num_ctx:
        return None

    model_size = cal_model_size_bytes(weights_bytes, parameters, bits_per_weight)
    context_size = cal_context_size_bytes(arch, num_ctx, cache_bit, num_parallel)
    return round((model_size + context_size) / GIB, 2)
