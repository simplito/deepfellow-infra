# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from server.utils.loading import Progress


def test_progress_initial_values():
    p = Progress(max_value=100.0)
    assert p.max == 100.0
    assert p.actual == 0.0
    assert p.percentage == 0


@pytest.mark.parametrize(
    ("max_value", "actual_value", "expected_percentage"),
    [
        (200.0, 50.0, 0.25),
        (100.0, 100.0, 1.0),
        (100.0, 0.0, 0.0),
    ],
)
def test_calculate_percentage(max_value: float, actual_value: float, expected_percentage: float) -> None:
    p = Progress(max_value=max_value, actual_value=actual_value)
    result = p.calculate_percentage()
    assert result == expected_percentage
    assert p.percentage == expected_percentage


@pytest.mark.parametrize(
    ("method_name", "max_value", "delta", "exp_actual", "exp_max", "exp_pct"),
    [
        ("add_to_actual_value", 100.0, 30.0, 30.0, 100.0, 0.3),
        ("add_to_actual_value", 50.0, 80.0, 80.0, 80.0, 1.0),
        ("set_actual_value", 100.0, 40.0, 40.0, 100.0, 0.4),
        ("set_actual_value", 50.0, 75.0, 75.0, 75.0, 1.0),
    ],
)
def test_update_actual_value(method_name: str, max_value: float, delta: float, exp_actual: float, exp_max: float, exp_pct: float) -> None:
    p = Progress(max_value=max_value)
    getattr(p, method_name)(delta)
    assert p.actual == exp_actual
    assert p.max == exp_max
    assert p.percentage == exp_pct


def test_set_max_value():
    p = Progress(max_value=100.0, actual_value=50.0)
    p.calculate_percentage()
    p.set_max_value(200.0)
    assert p.max == 200.0
    assert p.percentage == 0.25


def test_get_percentage_returns_stored_percentage():
    p = Progress(max_value=100.0, actual_value=50.0)
    p.calculate_percentage()
    assert p.get_percentage() == 0.5
