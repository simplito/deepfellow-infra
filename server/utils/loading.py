"""Loading utils."""


class Progress:
    """Return progress info from data."""

    percentage: float
    actual: float
    max: float

    def __init__(self, max_value: float, actual_value: float = 0.0):
        self.percentage = 0
        self.max = max_value
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
        self.calculate_percentage()

    def get_percentage(self) -> float:
        """Return percentage."""
        return self.percentage
