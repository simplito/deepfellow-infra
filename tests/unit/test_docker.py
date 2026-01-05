# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.docker import normalize_docker_platform


@pytest.mark.parametrize(
    ("platform_str", "expectation"),
    [
        ("linux/amd64", "linux/amd64"),
        ("linux/x86_64", "linux/amd64"),
        ("linux/arm64/v8", "linux/arm64/v8"),
        ("linux/arm64", "linux/arm64/v8"),
        ("linux/aarch64/v8", "linux/arm64/v8"),
        ("linux/aarch64", "linux/arm64/v8"),
        ("linux/armhf", "linux/arm/v7"),
        ("linux/armhf/v7", "linux/arm/v7"),
        ("linux/armv7l", "linux/arm/v7"),
        ("linux/armv7", "linux/arm/v7"),
        ("linux/arm", "linux/arm/v7"),
        ("linux/arm/v7", "linux/arm/v7"),
        ("linux/arm/v6", "linux/arm/v6"),
        ("linux/arm/v5", "linux/arm/v5"),
        ("linux/i386", "linux/386"),
        ("linux/386", "linux/386"),
        ("linux/arm64/v9", "linux/arm64/v9"),
        ("linux/fake", "linux/fake"),
        ("linux/fake/v1", "linux/fake/v1"),
    ],
)
def test_join_url(platform_str: str, expectation: str):
    result = normalize_docker_platform(platform_str)

    assert result == expectation
