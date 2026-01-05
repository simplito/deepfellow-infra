# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.utils.core import Utils


@pytest.fixture
def utils():
    return Utils()


@pytest.mark.parametrize(
    ("a", "b", "expectation"),
    [
        ("http://google.com", "article/123", "http://google.com/article/123"),
        ("http://google.com", "/article/123", "http://google.com/article/123"),
        ("http://google.com/", "article/123", "http://google.com/article/123"),
        ("http://google.com/", "/article/123", "http://google.com/article/123"),
    ],
)
def test_join_url(a: str, b: str, expectation: str, utils: Utils):
    result = utils.join_url(a, b)

    assert result == expectation
