# Add Divide Function to Calculator

**ADW ID:** de3d954e
**Date:** 2025-12-30
**Specification:** specs/issue-1-adw-de3d954e-sdlc_planner-add-divide-function.md

## Overview

Added a divide function to the calculator module that performs integer division with proper error handling for division by zero. The implementation maintains consistency with existing calculator functions and includes comprehensive unit tests covering normal operations, edge cases, and error scenarios.

## What Was Built

- `divide(a: int, b: int) -> int` function in calculator.py
- Comprehensive test suite in test_calculator.py covering all calculator functions
- Division by zero error handling with descriptive error messages
- Integration with existing calculator demonstration code
- Development dependency configuration with ruff linter

## Technical Implementation

### Files Modified

- `calculator.py`: Added divide function with type hints, zero-check validation, and integer division logic. Updated main block to demonstrate divide usage with two examples.
- `test_calculator.py`: Created comprehensive test suite with TestCalculator class covering all functions (add, subtract, multiply, divide) including edge cases and division by zero error handling.
- `pyproject.toml`: Initialized project configuration with Python >=3.11 requirement, adw-framework dependency, and ruff linter as dev dependency.

### Key Changes

- Implemented divide function following existing pattern with type hints (int parameters, int return type)
- Added zero-check that raises `ValueError("Cannot divide by zero")` when b=0
- Used integer division (a // b) to maintain consistency with int type hints
- Created 49-line test file with 5 test methods covering normal operations, negative numbers, and error cases
- Added pytest-compatible test structure using unittest framework

## How to Use

### Basic Division

```python
from calculator import divide

# Normal division
result = divide(10, 2)  # Returns 5

# Integer division (truncates decimal)
result = divide(7, 3)   # Returns 2

# Negative numbers
result = divide(-6, 3)  # Returns -2
```

### Error Handling

```python
try:
    result = divide(5, 0)
except ValueError as e:
    print(e)  # "Cannot divide by zero"
```

### Running the Calculator

```bash
# Run main demonstration
python calculator.py

# Output:
# 2 + 3 = 5
# 10 / 2 = 5
# 7 / 3 = 2
```

## Configuration

No configuration required. The function works with the existing calculator module structure.

**Dependencies:**
- Python >=3.11
- Development: ruff linter for code quality

## Testing

### Run Tests

```bash
# Using pytest (verbose)
python -m pytest test_calculator.py -v

# Using unittest
python test_calculator.py

# Direct function testing
python -c "from calculator import divide; print(divide(10, 2))"

# Test error handling (should raise ValueError)
python -c "from calculator import divide; divide(5, 0)"
```

### Test Coverage

- **test_add** - Addition operations with positive, negative, and zero values
- **test_subtract** - Subtraction operations with various integer combinations
- **test_multiply** - Multiplication including zero and negative numbers
- **test_divide** - Normal division, integer truncation, negative number division
- **test_divide_by_zero** - Error handling for both positive and negative numerators

All tests pass with 100% success rate.

## Notes

- Uses integer division (`//`) to maintain consistency with int type hints
- Division by zero raises ValueError rather than ZeroDivisionError for explicit error messaging
- Edge case: 7 / 3 returns 2 (integer division truncates decimal)
- Edge case: 0 / 5 returns 0 (dividing zero by non-zero is valid)
- Future enhancement: Consider supporting float division with a separate function or parameter
- Project uses uv for dependency management
- Ruff linter added to dev dependencies for code quality checks
