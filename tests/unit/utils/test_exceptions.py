# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/utils/exceptions.py and server/utils/logger.py."""

import pytest

from server.utils.exceptions import (
    DockerComposeStartError,
    DockerImageAuthorizationError,
    DockerImageDoesNotExistError,
)


def test_docker_compose_start_error_stores_stdout_and_stderr() -> None:
    err = DockerComposeStartError(stdout="out", stderr="err")

    assert err.stdout == "out"
    assert err.stderr == "err"


def test_docker_compose_start_error_message_prefers_stderr() -> None:
    err = DockerComposeStartError(stdout="out", stderr="err")

    assert str(err) == "err"


def test_docker_compose_start_error_falls_back_to_stdout_when_no_stderr() -> None:
    err = DockerComposeStartError(stdout="out", stderr="")

    assert str(err) == "out"


def test_docker_compose_start_error_default_message_when_both_empty() -> None:
    err = DockerComposeStartError(stdout="", stderr="")

    assert str(err) == "docker compose failed"


def test_docker_image_does_not_exist_error_stores_image_name() -> None:
    err = DockerImageDoesNotExistError("my-image:latest")

    assert err.image == "my-image:latest"


def test_docker_image_does_not_exist_error_message_contains_image() -> None:
    err = DockerImageDoesNotExistError("alpine")
    assert "alpine" in str(err)


def test_docker_image_does_not_exist_error_is_exception() -> None:
    with pytest.raises(DockerImageDoesNotExistError):
        raise DockerImageDoesNotExistError("img")


def test_docker_image_authorization_error_stores_image_name() -> None:
    err = DockerImageAuthorizationError("private/repo:tag")

    assert err.image == "private/repo:tag"


def test_docker_image_authorization_error_message_contains_image() -> None:
    err = DockerImageAuthorizationError("secret-image")

    assert "secret-image" in str(err)


def test_docker_image_authorization_error_is_exception() -> None:
    with pytest.raises(DockerImageAuthorizationError):
        raise DockerImageAuthorizationError("img")
