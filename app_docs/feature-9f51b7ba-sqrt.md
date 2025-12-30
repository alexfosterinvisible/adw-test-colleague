# Square Root Function

**ADW ID:** 9f51b7ba
**Date:** 2025-12-30
**Specification:** specs/issue-13-adw-9f51b7ba-sdlc_planner-add-sqrt-function.md

## Overview

Implemented a square root function using Newton's method (Babylonian method) that computes the square root of non-negative integers. The function provides accurate approximations with proper error handling for negative inputs, extending the calculator's capabilities beyond basic arithmetic operations.

## What Was Built

- `sqrt(n: int) -> float` function in calculator.py using Newton's iterative method
- Input validation that raises ValueError for negative numbers
- Comprehensive test suite covering perfect squares, non-perfect squares, edge cases, and error handling
- Demonstration examples in the calculator's main block

## Technical Implementation

### Files Modified

- `calculator.py`: Added sqrt function with Newton's method implementation (30 lines)
- `test_calculator.py`: Added comprehensive test coverage with test_sqrt and test_sqrt_negative methods (24 lines)
- `.ports.env`: Minor configuration updates (6 lines changed)
- `uv.lock`: Dependency lock file updates (45 lines)

### Key Changes

- **Newton's Method Algorithm**: Implements iterative square root calculation using formula `guess = (guess + n / guess) / 2` with convergence threshold of 0.00001
- **Edge Case Handling**: Special case for zero input returns 0.0 immediately; negative inputs raise descriptive ValueError
- **Type Hints**: Function signature follows existing pattern with `sqrt(n: int) -> float`
- **Test Coverage**: Tests validate perfect squares (0, 1, 4, 16, 100), non-perfect squares (2, 10), large numbers (10000), and negative input errors
- **Documentation**: Comprehensive docstring explains Newton's method, parameters, return value, and exceptions

## How to Use

### Basic Usage

```python
from calculator import sqrt

# Perfect square
result = sqrt(16)  # Returns 4.0

# Non-perfect square
result = sqrt(10)  # Returns ~3.162...

# Zero
result = sqrt(0)   # Returns 0.0

# Negative number (raises error)
try:
    result = sqrt(-5)
except ValueError as e:
    print(e)  # "Cannot calculate square root of negative number"
```

### Running the Demonstration

```bash
python calculator.py
```

This will output sqrt examples along with other calculator operations.

## Configuration

No configuration required. The function uses a hardcoded convergence threshold of 0.00001 which provides good balance between accuracy (within 0.001 tolerance) and performance (typically 5-7 iterations).

## Testing

### Run All Tests

```bash
uv run pytest test_calculator.py -v
```

### Run Specific sqrt Tests

```bash
# Test normal operations
uv run pytest test_calculator.py::TestCalculator::test_sqrt -v

# Test error handling
uv run pytest test_calculator.py::TestCalculator::test_sqrt_negative -v
```

### Direct Function Testing

```bash
# Perfect and non-perfect squares
python -c "from calculator import sqrt; print(f'sqrt(16) = {sqrt(16)}'); print(f'sqrt(10) = {sqrt(10):.3f}')"

# Verify error for negative input (should raise ValueError)
python -c "from calculator import sqrt; sqrt(-5)"
```

## Notes

- Newton's method converges rapidly (typically 5-7 iterations) for most inputs
- Initial guess of n/2 works well across the input range
- Convergence threshold of 0.00001 ensures accuracy within 0.001 tolerance
- Returns float type since square roots are often irrational numbers
- Self-contained implementation without requiring math module imports
- Function maintains consistency with existing calculator functions (type hints, error handling pattern)
- No iteration limit needed as convergence is guaranteed for non-negative inputs
- Future enhancement: Could extend to support float inputs instead of just integers
