# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common types for Ollama/Ollama External."""

import json

from fastapi import HTTPException

ERROR_MAPPING = {
    "file does not exist": (
        400,
        "Model not found — check if the model ID is correct "
        "(for HuggingFace models use the hf.co/ prefix, "
        "e.g. hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF)",
    ),
    "not GGUF or is not compatible": (
        400,
        "This repository has no GGUF build compatible with llama.cpp — pick a GGUF version of the model",
    ),
    'realm host "huggingface.co" does not match': (
        400,
        "Cannot authenticate with HuggingFace — this may be a gated model. Try using a direct GGUF download URL instead.",
    ),
}


def raise_ollama_pull_error(raw_data: str) -> None:
    """Parse a failed Ollama /api/pull response and raise an appropriate HTTPException."""
    try:
        record = json.loads(raw_data)
    except json.JSONDecodeError:
        raise HTTPException(400, "Model not available") from None
    ollama_error: str = record.get("error", "")
    for error_fragment, (status_code, message) in ERROR_MAPPING.items():
        if error_fragment in ollama_error:
            raise HTTPException(status_code, message)
    raise HTTPException(400, f"Model not available: {ollama_error}" if ollama_error else "Model not available")
