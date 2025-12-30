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

    def test_sqrt(self):
        """Test square root works correctly."""
        # Perfect squares
        assert abs(calculator.sqrt(4) - 2.0) < 0.0001
        assert abs(calculator.sqrt(9) - 3.0) < 0.0001
        assert abs(calculator.sqrt(16) - 4.0) < 0.0001

        # Non-perfect squares
        assert abs(calculator.sqrt(2) - 1.41421) < 0.001
        assert abs(calculator.sqrt(3) - 1.73205) < 0.001

        # Edge cases
        assert calculator.sqrt(0) == 0.0
        assert calculator.sqrt(1) == 1.0

    def test_sqrt_negative(self):
        """Test square root of negative number raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculator.sqrt(-1)
        assert str(context.exception) == "Cannot calculate square root of negative number"

        with self.assertRaises(ValueError) as context:
            calculator.sqrt(-4)
        assert str(context.exception) == "Cannot calculate square root of negative number"

        with self.assertRaises(ValueError) as context:
            calculator.sqrt(-10)
        assert str(context.exception) == "Cannot calculate square root of negative number"


if __name__ == "__main__":
    unittest.main()
