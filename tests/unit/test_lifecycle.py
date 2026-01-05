# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import subprocess
from unittest import mock
from unittest.mock import Mock, patch

import pytest

from server.lifecycle import check_subnet
from server.utils.exceptions import AppStartError


@pytest.fixture(name="subnet")
def subnet_fixture() -> str:
    return "subnet"


def test_check_subnet_invalid_name() -> None:
    subnet = ";@#$%^"

    with pytest.raises(AppStartError, match="Invalid subnet name"):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 0))
def test_check_subnet_subprocess_timeout(mock_subprocess_run: Mock, subnet: str) -> None:
    with pytest.raises(
        subprocess.CalledProcessError,
        match=re.escape("Command '['docker', 'network', 'inspect', 'subnet']' returned non-zero exit status 124."),
    ):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run")
def test_check_subnet_subprocess_other_error(mock_subprocess_run: Mock, subnet: str) -> None:
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(999, ["docker", "network", "inspect", subnet], "output", "error msg")

    with pytest.raises(
        subprocess.CalledProcessError,
        match=re.escape("Command '['docker', 'network', 'inspect', 'subnet']' returned non-zero exit status 999."),
    ):
        check_subnet(subnet)


@patch("server.lifecycle.subprocess.run")
def test_check_subnet_subprocess_success(mock_subprocess_run: Mock, subnet: str) -> None:
    result = check_subnet(subnet)

    assert result is None
    assert mock_subprocess_run.call_count == 1
    assert mock_subprocess_run.call_args == mock.call(["docker", "network", "inspect", subnet], capture_output=True, check=True, timeout=10)
