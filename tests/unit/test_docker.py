# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.docker import DockerImageNameInfo, normalize_docker_platform


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
def test_normalize_docker_platform(platform_str: str, expectation: str):
    result = normalize_docker_platform(platform_str)

    assert result == expectation


@pytest.mark.parametrize(
    ("full_image", "expected_registry", "expected_namespace", "expected_image_name"),
    [
        # Official Docker Hub image (one part)
        ("python", "docker.io", "library", "python"),
        # Docker Hub image with namespace (two parts)
        ("bitnami/redis", "docker.io", "bitnami", "redis"),
        # Third-party registry (three parts)
        ("ghcr.io/username/image", "ghcr.io", "username", "image"),
        # Registry with a port
        ("localhost:5000/my-app", "localhost:5000", "library", "my-app"),
        # Deeply nested namespace (e.g., AWS ECR or GitLab)
        ("123456789.dkr.ecr.us-east-1.amazonaws.com/org/team/app", "123456789.dkr.ecr.us-east-1.amazonaws.com", "org/team", "app"),
        # Registry with namespace and image
        ("my-reg.internal/dev-team/api-server", "my-reg.internal", "dev-team", "api-server"),
    ],
)
def test_docker_image_name_info_parse(full_image: str, expected_registry: str, expected_namespace: str, expected_image_name: str):
    """Test that various image strings are correctly parsed into components."""
    # Act
    info = DockerImageNameInfo.parse(full_image)

    # Assert
    assert info.registry == expected_registry
    assert info.namespace == expected_namespace
    assert info.image_name == expected_image_name


def test_docker_image_name_info_is_frozen():
    """Verify that the dataclass is indeed frozen (immutable)."""
    info = DockerImageNameInfo.parse("alpine")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError  # noqa: B017, PT011
        info.image_name = "ubuntu"  # type: ignore
