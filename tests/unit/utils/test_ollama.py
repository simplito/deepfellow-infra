# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/utils/ollama.py."""

import json

import pytest
from fastapi import HTTPException

from server.utils.ollama import raise_ollama_pull_error


def test_raise_ollama_pull_error_invalid_json_raises_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error("not valid json {{{")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"


def test_raise_ollama_pull_error_known_error_fragment() -> None:
    raw = json.dumps({"error": "file does not exist at path"})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert "check if the model ID is correct" in exc_info.value.detail


def test_raise_ollama_pull_error_unknown_error_with_message() -> None:
    raw = json.dumps({"error": "something unexpected happened"})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert "something unexpected happened" in exc_info.value.detail


def test_raise_ollama_pull_error_empty_error_field() -> None:
    raw = json.dumps({"error": ""})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"


def test_raise_ollama_pull_error_missing_error_field() -> None:
    raw = json.dumps({})

    with pytest.raises(HTTPException) as exc_info:
        raise_ollama_pull_error(raw)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not available"
