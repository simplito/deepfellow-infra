# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import MagicMock

import pytest

from server.services.claude_service import ClaudeService, ClaudeServiceOptions, _const  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def claude_service_options() -> ClaudeServiceOptions:
    return ClaudeServiceOptions(
        api_url="https://api.anthropic.com",
        api_key="claude-api-key",
        anthropic_version="anthropic-version",
    )


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[], total_vram_gb=8.0),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> ClaudeService:
    return ClaudeService(**deps)


def test_claude_service_options_headers_provides_dict(claude_service_options: ClaudeServiceOptions):
    assert claude_service_options.headers == {"anthropic-version": "anthropic-version", "x-api-key": "claude-api-key"}


def test_claude_service_options_headers_includes_beta_when_set():
    opts = ClaudeServiceOptions(
        api_url="https://api.anthropic.com",
        api_key="key",
        anthropic_version="2023-06-01",
        anthropic_beta="context-1m-2025-08-07",
    )

    assert opts.headers == {
        "x-api-key": "key",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "context-1m-2025-08-07",
    }


def test_get_type(svc: ClaudeService) -> None:
    assert svc.get_type() == "claude"


def test_get_description(svc: ClaudeService) -> None:
    assert svc.get_description() == "Remote access to Claude models."


def test_get_default_url(svc: ClaudeService) -> None:
    assert svc.get_default_url() == "https://api.anthropic.com"


def test_get_models_registry(svc: ClaudeService) -> None:
    assert svc.get_models_registry() is _const


def test_get_spec_returns_service_specification(svc: ClaudeService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "api_url" in field_names
    assert "api_key" in field_names
    assert "anthropic_version" in field_names
    assert "anthropic_beta" in field_names
