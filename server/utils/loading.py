# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Loading utils."""


class Progress:
    """Return progress info from data."""

    percentage: float
    actual: float
    max: float
    max_known: bool

    def __init__(self, max_value: float, actual_value: float = 0.0):
        self.percentage = 0
        self.max = max_value
        self.max_known = max_value > 0
        self.actual = actual_value

    def calculate_percentage(self) -> float:
        """Return percentage."""
        self.percentage = self.actual / self.max
        return self.percentage

    def add_to_actual_value(self, value: float) -> None:
        """Add percentage."""
        actual_value = self.actual + value
        if actual_value > self.max:
            self.max = actual_value

        self.actual = actual_value
        self.calculate_percentage()

    def set_actual_value(self, value: float) -> None:
        """Set percentage."""
        if value > self.max:
            self.max = value

        self.actual = value
        self.calculate_percentage()

    def set_max_value(self, value: float) -> None:
        """Set max value."""
        self.max = value
        self.max_known = value > 0
        self.calculate_percentage()

    def get_percentage(self) -> float:
        """Return percentage."""
        return self.percentage

    def to_data(self) -> dict[str, int]:
        """Return byte counts for a progress chunk's `data`: `completed`, plus `total` once it is known.

        `max` grows with `actual` while the size is unknown, so `total` is sent only if a real size was set.
        """
        data = {"completed": int(self.actual)}
        if self.max_known:
            data["total"] = int(self.max)
        return data
