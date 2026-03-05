# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.services.claude_service import ClaudeServiceOptions


@pytest.fixture
def claude_service_options() -> ClaudeServiceOptions:
    return ClaudeServiceOptions(
        api_url="https://api.anthropic.com",
        api_key="claude-api-key",
        anthropic_version="anthropic-version",
    )


def test_claude_service_options_headers_provides_dict(claude_service_options: ClaudeServiceOptions):
    assert claude_service_options.headers == {"anthropic-version": "anthropic-version", "x-api-key": "claude-api-key"}
