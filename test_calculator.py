"""Unit tests for calculator module."""

import unittest
import calculator


class TestCalculator(unittest.TestCase):
    """Test suite for calculator functions."""

    def test_add(self):
        """Test addition works correctly."""
        assert calculator.add(2, 3) == 5
        assert calculator.add(0, 0) == 0
        assert calculator.add(-1, 1) == 0

    def test_subtract(self):
        """Test subtraction works correctly."""
        assert calculator.subtract(5, 3) == 2
        assert calculator.subtract(0, 0) == 0
        assert calculator.subtract(-1, -1) == 0

    def test_multiply(self):
        """Test multiplication works correctly."""
        assert calculator.multiply(2, 3) == 6
        assert calculator.multiply(0, 5) == 0
        assert calculator.multiply(-2, 3) == -6

    def test_divide(self):
        """Test division works correctly."""
        assert calculator.divide(6, 3) == 2
        assert calculator.divide(0, 5) == 0
        assert calculator.divide(7, 3) == 2  # Integer division
        assert calculator.divide(-6, 3) == -2
        assert calculator.divide(6, -3) == -2
        assert calculator.divide(-6, -3) == 2

    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculator.divide(5, 0)
        assert str(context.exception) == "Cannot divide by zero"

        with self.assertRaises(ValueError) as context:
            calculator.divide(-5, 0)
        assert str(context.exception) == "Cannot divide by zero"


if __name__ == "__main__":
    unittest.main()
