# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.services.remote_service import DefaultRemoteServiceOptions


@pytest.fixture
def default_remote_service_options():
    return DefaultRemoteServiceOptions(api_url="https://api.openai.com", api_key="test-api-key")


def test_default_remote_service_options_headers_provides_dict(default_remote_service_options: DefaultRemoteServiceOptions):
    assert default_remote_service_options.headers == {"Authorization": "Bearer test-api-key"}
