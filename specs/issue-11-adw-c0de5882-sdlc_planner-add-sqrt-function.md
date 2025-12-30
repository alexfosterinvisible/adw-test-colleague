# Feature: Add square root function to calculator

## Metadata
issue_number: `11`
adw_id: `c0de5882`
issue_json: `{"number":11,"title":"Add square root function","body":"Add sqrt(n) function using Newton's method. Raise ValueError for negative inputs."}`

## Feature Description
Add a square root function to the calculator module that computes the square root of a number using Newton's method (also known as the Newton-Raphson method). This iterative numerical method provides accurate approximations of square roots. The function should validate inputs and raise ValueError for negative numbers, as real square roots of negative numbers are undefined.

## User Story
As a calculator user
I want to calculate the square root of a number
So that I can perform square root operations efficiently using Newton's method with confidence that invalid inputs (negative numbers) are handled with clear error messages

## Problem Statement
The calculator module currently supports basic arithmetic operations (addition, subtraction, multiplication, and division) but lacks square root functionality. Users need the ability to calculate square roots using an efficient algorithmic approach, with appropriate error handling to prevent runtime errors when attempting to compute the square root of negative numbers.

## Solution Statement
Implement a sqrt function in calculator.py that takes a numeric argument and returns its square root using Newton's method. The function will validate that the input is non-negative, raising a ValueError with a clear error message for negative inputs. Newton's method will iteratively refine an initial guess until the result converges to an accurate approximation. The implementation will follow the existing code style with type hints for consistency.

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Main calculator module where the sqrt function will be added after the divide function. Currently contains add, subtract, multiply, and divide functions following a consistent pattern with type hints. The sqrt function will need to handle float return types since square roots are often non-integer values.

- **test_calculator.py** - Existing test file that contains comprehensive unit tests for all calculator functions. Will be updated to include tests for the sqrt function covering normal operations, edge cases (0, 1, perfect squares, non-perfect squares), and negative input error handling.

- **app_docs/feature-de3d954e-divide-function.md** - Reference documentation showing the established pattern for calculator functions including error handling, type hints, test structure, and documentation format. This feature should follow the same patterns.

## Implementation Plan
### Phase 1: Foundation
Review the existing calculator.py structure and test_calculator.py test patterns to ensure the sqrt function maintains consistency. Understand Newton's method algorithm: starting with an initial guess, iteratively refine using the formula `x_new = (x_old + n/x_old) / 2` until convergence. Plan the function signature to accept int or float input and return float output.

### Phase 2: Core Implementation
Implement the sqrt function in calculator.py using Newton's method. Add input validation to check for negative numbers and raise ValueError("Cannot calculate square root of negative number"). Implement the iterative Newton's method algorithm with a convergence tolerance (e.g., when the difference between iterations is less than 0.000001). Handle the edge case of sqrt(0) which should return 0 immediately. Use type hints: `def sqrt(n: float) -> float` to support both integer and float inputs while returning float results.

### Phase 3: Integration
Update test_calculator.py with comprehensive tests for the sqrt function including normal cases (perfect squares like 4, 9, 16), non-perfect squares (2, 3, 5), edge cases (0, 1), and the negative input error case. Update the calculator.py main block to demonstrate the sqrt function. Run all validation commands to ensure zero regressions and that the implementation works correctly.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Implement sqrt function in calculator.py
- Add sqrt function to calculator.py after the divide function
- Add type hints: `def sqrt(n: float) -> float` to accept numeric input and return float
- Implement input validation: if n < 0, raise ValueError("Cannot calculate square root of negative number")
- Handle edge case: if n == 0, return 0.0 immediately
- Implement Newton's method algorithm:
  - Start with initial guess (e.g., n / 2 or 1.0 for small n)
  - Iteratively apply formula: `guess = (guess + n / guess) / 2`
  - Continue until convergence (when abs(guess * guess - n) < 0.000001)
  - Return the final approximation
- Maintain consistent spacing and style with existing functions
- Add brief inline comment explaining Newton's method approach

### 2. Add sqrt tests to test_calculator.py
- Add test_sqrt method to TestCalculator class
- Test perfect squares: assert abs(calculator.sqrt(4) - 2.0) < 0.0001
- Test perfect squares: assert abs(calculator.sqrt(9) - 3.0) < 0.0001
- Test perfect squares: assert abs(calculator.sqrt(16) - 4.0) < 0.0001
- Test non-perfect squares: assert abs(calculator.sqrt(2) - 1.41421) < 0.001
- Test edge case zero: assert calculator.sqrt(0) == 0.0
- Test edge case one: assert calculator.sqrt(1) == 1.0
- Add test_sqrt_negative method to verify error handling
- Use assertRaises(ValueError) to verify exception for negative input
- Verify error message content: assert str(context.exception) == "Cannot calculate square root of negative number"
- Test with multiple negative values: -1, -4, -10

### 3. Update main block demonstration
- Add sqrt examples to the if __name__ == "__main__" block in calculator.py
- Include examples: sqrt(4), sqrt(2), sqrt(9)
- Format output to show reasonable precision (e.g., f"{result:.4f}" for 4 decimal places)
- Keep format consistent with existing print statements

### 4. Run validation commands
- Execute all validation commands listed below to ensure zero regressions
- Verify all existing tests still pass (add, subtract, multiply, divide)
- Verify new sqrt tests pass with expected accuracy
- Confirm calculator works end-to-end with manual testing

## Testing Strategy
### Unit Tests
- **test_sqrt** - Verify square root works correctly for various inputs:
  - Perfect squares: sqrt(4) ≈ 2.0, sqrt(9) ≈ 3.0, sqrt(16) ≈ 4.0
  - Non-perfect squares: sqrt(2) ≈ 1.41421, sqrt(3) ≈ 1.73205
  - Edge cases: sqrt(0) = 0.0, sqrt(1) = 1.0
- **test_sqrt_negative** - Verify ValueError is raised with correct message when input is negative
- All existing tests (add, subtract, multiply, divide) should continue to pass

### Edge Cases
- Square root of zero (should return 0.0)
- Square root of one (should return 1.0)
- Square roots of perfect squares (4, 9, 16, 25, etc.)
- Square roots of non-perfect squares (2, 3, 5, 7, etc.)
- Negative number inputs (should raise ValueError)
- Large numbers to verify Newton's method converges correctly
- Very small positive numbers (e.g., 0.01, 0.0001)

## Acceptance Criteria
- sqrt function exists in calculator.py with proper type hints (float parameter, float return)
- sqrt function uses Newton's method for calculation
- sqrt function returns accurate results (within 0.0001 for perfect squares, within 0.001 for others)
- sqrt function raises ValueError with message "Cannot calculate square root of negative number" when n < 0
- sqrt function handles edge cases: sqrt(0) = 0.0, sqrt(1) = 1.0
- All unit tests pass with 100% success rate (existing and new tests)
- test_calculator.py has comprehensive coverage of sqrt function
- Code style matches existing functions (spacing, type hints, formatting)
- Main block demonstrates sqrt function usage with clear output
- No regressions in existing add, subtract, multiply, divide functions

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python -m pytest test_calculator.py -v` - Run all calculator tests with verbose output, verify zero failures
- `python test_calculator.py` - Run tests using unittest directly, confirm all tests pass
- `python calculator.py` - Execute main block to verify demonstrations work without errors
- `python -c "from calculator import sqrt; print(f'sqrt(4) = {sqrt(4):.4f}'); print(f'sqrt(2) = {sqrt(2):.4f}'); print(f'sqrt(9) = {sqrt(9):.4f}')"` - Test sqrt function directly with various inputs
- `python -c "from calculator import sqrt; sqrt(-4)"` - Verify negative input raises ValueError (should exit with error)
- `python -c "from calculator import sqrt; import math; n=16; result=sqrt(n); expected=math.sqrt(n); print(f'sqrt({n}) = {result:.6f}, expected = {expected:.6f}, diff = {abs(result-expected):.9f}')"` - Verify accuracy against Python's built-in math.sqrt

## Notes
- Newton's method is an iterative algorithm that converges quadratically, making it very efficient for square root calculations
- The function uses float type hints to support both integer and float inputs while returning precise float results
- Convergence tolerance of 0.000001 provides high accuracy while ensuring reasonable iteration counts
- The implementation avoids using Python's built-in math.sqrt to demonstrate Newton's method as specified
- Edge case handling for sqrt(0) avoids unnecessary iterations by returning 0.0 immediately
- Initial guess for Newton's method can be n/2 for most numbers, or 1.0 for very small numbers to ensure fast convergence
- Type hints use float rather than int|float syntax for compatibility with Python 3.11
- Future enhancement: Could add optional precision parameter to allow users to specify convergence tolerance
- Future enhancement: Could add support for complex numbers to handle negative inputs (return complex results)
