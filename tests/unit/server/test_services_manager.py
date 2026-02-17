# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from fastapi import HTTPException

from server.services_manager import ServicesManager


@pytest.mark.parametrize(
    ("input_id", "expected_tid", "expected_inst"),
    [
        ("my-service|inst_01", "my-service", "inst_01"),
        ("simple-service", "simple-service", "default"),
        ("123-456", "123-456", "default"),
        ("A-z_0-9", "A-z_0-9", "default"),
        ("a" * 64 + "|" + "b" * 64, "a" * 64, "b" * 64),
    ],
)
def test_split_success(input_id: str, expected_tid: str, expected_inst: str, services_manager: ServicesManager):
    """Test various valid formats and boundary lengths."""
    tid, inst = services_manager.split_service_type_and_instance(input_id)
    assert tid == expected_tid
    assert inst == expected_inst


@pytest.mark.parametrize(
    ("invalid_id", "expected_status", "error_part"),
    [
        # Character violations
        ("service.name", 400, "invalid characters"),
        ("service name|inst1", 400, "invalid characters"),
        ("serviceID|inst!", 400, "invalid characters"),
        ("service@domain", 400, "invalid characters"),
        # Length violations
        ("a" * 65, 400, "exceeds maximum length"),
        ("valid-id|" + "b" * 65, 400, "exceeds maximum length"),
        # Format violations
        ("part1|part2|part3", 404, "Incorrect service_id"),
        ("", 400, "invalid characters"),  # Empty string fails regex
    ],
)
def test_split_failures(invalid_id: str, expected_status: int, error_part: str, services_manager: ServicesManager):
    """Test that invalid inputs raise the correct HTTPException."""
    with pytest.raises(HTTPException) as exc:
        services_manager.split_service_type_and_instance(invalid_id)

    assert exc.value.status_code == expected_status
    assert error_part in exc.value.detail
