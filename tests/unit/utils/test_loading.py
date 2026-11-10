# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

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


def test_to_data_returns_completed_and_total_bytes():
    p = Progress(max_value=2048.0, actual_value=512.0)
    assert p.to_data() == {"completed": 512, "total": 2048}


def test_to_data_omits_total_when_size_unknown():
    p = Progress(max_value=0)
    assert p.to_data() == {"completed": 0}


@pytest.mark.parametrize("method_name", ["add_to_actual_value", "set_actual_value"])
def test_to_data_omits_total_after_bytes_when_size_unknown(method_name: str) -> None:
    p = Progress(0)
    getattr(p, method_name)(1200)
    assert p.to_data() == {"completed": 1200}


def test_to_data_includes_total_once_max_value_is_set() -> None:
    p = Progress(0)
    p.add_to_actual_value(100)
    p.set_max_value(1000)
    assert p.to_data() == {"completed": 100, "total": 1000}


def test_to_data_reports_raised_total_when_known_size_is_exceeded() -> None:
    p = Progress(1000)
    p.set_actual_value(1500)
    assert p.to_data() == {"completed": 1500, "total": 1500}
