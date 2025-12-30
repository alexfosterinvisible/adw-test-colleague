# Feature: Add Square Root Function

## Metadata
issue_number: `25`
adw_id: `7da97a75`
issue_json: `{"number":25,"title":"Add square root function","body":"Add some new maths function that's not too complicated"}`

## Feature Description
Add a square root function to the calculator module that computes the integer square root of a non-negative integer. The function will follow the existing pattern established by add, subtract, multiply, and divide functions, including proper error handling for invalid inputs (negative numbers) and comprehensive unit testing.

## User Story
As a calculator user
I want to calculate the square root of a number
So that I can perform mathematical operations that require square root calculations

## Problem Statement
The calculator currently supports basic arithmetic operations (addition, subtraction, multiplication, and division) but lacks support for square root calculations. Users need the ability to compute square roots as part of their mathematical operations without having to use external tools or libraries.

## Solution Statement
Implement a `sqrt(n: int) -> int` function that calculates the integer square root of a non-negative integer. The function will:
- Accept a single integer parameter
- Return the integer (floor) square root using Python's built-in `math.isqrt()` function
- Raise a `ValueError` with a descriptive message when given a negative number
- Follow the existing code style and patterns established in calculator.py
- Include comprehensive unit tests covering normal operations, edge cases, and error scenarios

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Main calculator module where the sqrt function will be added. This file contains all existing mathematical operations (add, subtract, multiply, divide) and follows a consistent pattern with type hints and error handling.

- **test_calculator.py** - Test suite where sqrt unit tests will be added. This file uses unittest framework and contains comprehensive tests for all calculator functions including edge cases and error handling.

- **pyproject.toml** - Project configuration file for dependency management. No changes needed, but referenced for context on project structure.

- **app_docs/feature-de3d954e-divide-function.md** - Reference documentation showing the pattern for implementing calculator functions with error handling, type hints, and testing strategy. This provides the template for our implementation approach.

### New Files
- **app_docs/feature-7da97a75-square-root-function.md** - Feature documentation that will be created at the end to document the implementation, usage examples, testing strategy, and technical details following the pattern established in feature-de3d954e-divide-function.md.

## Implementation Plan
### Phase 1: Foundation
Review existing calculator patterns and error handling conventions. The divide function provides an excellent reference for:
- Type hint usage (int parameters, int return type)
- Error handling pattern (raising ValueError with descriptive messages)
- Integer-only return values (using integer division or floor operations)
- Test structure and coverage expectations

### Phase 2: Core Implementation
Implement the sqrt function in calculator.py following the established pattern:
- Add function immediately after the divide function to maintain code organization
- Use type hints: `sqrt(n: int) -> int`
- Validate input: raise ValueError for negative numbers with message "Cannot calculate square root of negative number"
- Use `math.isqrt()` for calculation (returns integer square root, equivalent to floor(sqrt(n)))
- Import math module at the top of the file

Add comprehensive unit tests in test_calculator.py:
- Normal operations: sqrt(4) = 2, sqrt(9) = 3, sqrt(16) = 4
- Edge cases: sqrt(0) = 0, sqrt(1) = 1, sqrt(2) = 1 (floor behavior)
- Perfect squares and non-perfect squares
- Error handling: negative number raises ValueError with correct message

### Phase 3: Integration
Update the main block demonstration code to include sqrt examples, showing both perfect squares and floor behavior for non-perfect squares. Update feature documentation to capture implementation details, usage patterns, and testing strategy.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Task 1: Implement sqrt function in calculator.py
- Import math module at the top of calculator.py
- Add sqrt function after divide function with type hints: `def sqrt(n: int) -> int:`
- Add input validation: if n < 0, raise ValueError("Cannot calculate square root of negative number")
- Implement calculation: return math.isqrt(n)
- Add docstring following the pattern of other functions if needed

### Task 2: Add sqrt demonstration to main block
- Update the `if __name__ == "__main__":` block in calculator.py
- Add two print statements demonstrating sqrt usage:
  - One showing a perfect square: print(f"sqrt(16) = {sqrt(16)}")
  - One showing floor behavior: print(f"sqrt(10) = {sqrt(10)}")

### Task 3: Add unit tests for sqrt function
- Add `test_sqrt` method to TestCalculator class in test_calculator.py
- Test normal operations with perfect squares: sqrt(0), sqrt(1), sqrt(4), sqrt(9), sqrt(16), sqrt(25)
- Test floor behavior with non-perfect squares: sqrt(2), sqrt(10), sqrt(15)
- Add `test_sqrt_negative` method to test error handling
- Verify ValueError is raised with correct message: "Cannot calculate square root of negative number"
- Test with multiple negative values: -1, -5, -10

### Task 4: Run validation commands
- Execute all validation commands listed below to ensure zero regressions
- Verify all tests pass including the new sqrt tests
- Confirm calculator.py runs without errors and displays sqrt examples

### Task 5: Create feature documentation
- Create app_docs/feature-7da97a75-square-root-function.md
- Document the implementation following the pattern in app_docs/feature-de3d954e-divide-function.md
- Include: Overview, What Was Built, Technical Implementation, How to Use, Testing, and Notes sections
- Provide usage examples showing both perfect squares and floor behavior
- Document edge cases and error handling approach

## Testing Strategy
### Unit Tests
All tests will be added to test_calculator.py following the existing unittest pattern:

**test_sqrt**:
- Test perfect squares: 0, 1, 4, 9, 16, 25 (returns 0, 1, 2, 3, 4, 5)
- Test non-perfect squares demonstrating floor behavior: 2, 10, 15 (returns 1, 3, 3)
- Test large perfect square: 100 (returns 10)

**test_sqrt_negative**:
- Test that sqrt(-1) raises ValueError
- Test that sqrt(-5) raises ValueError
- Test that sqrt(-10) raises ValueError
- Verify error message is exactly: "Cannot calculate square root of negative number"

### Edge Cases
- **sqrt(0)** - Should return 0 without error
- **sqrt(1)** - Should return 1 (identity case)
- **sqrt(2)** - Should return 1 (demonstrates floor behavior)
- **sqrt(negative)** - Should raise ValueError with descriptive message
- **Large perfect squares** - Should handle correctly (e.g., sqrt(10000) = 100)
- **Large non-perfect squares** - Should return floor value correctly

## Acceptance Criteria
- sqrt function is implemented in calculator.py with proper type hints
- Function validates input and raises ValueError for negative numbers
- Function returns integer square root (floor) for all non-negative inputs
- All existing tests continue to pass (zero regressions)
- New unit tests provide comprehensive coverage including edge cases
- Main block demonstrates sqrt functionality with clear examples
- Calculator.py runs without errors when executed directly
- Feature documentation is created following established patterns
- Code follows the existing style and conventions in the codebase

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python calculator.py` - Run calculator demonstration, verify sqrt examples display correctly
- `python -m pytest test_calculator.py -v` - Run all unit tests with verbose output, verify 100% pass rate
- `python -c "from calculator import sqrt; print(sqrt(16))"` - Direct function test for perfect square
- `python -c "from calculator import sqrt; print(sqrt(10))"` - Direct function test for floor behavior
- `python -c "from calculator import sqrt; sqrt(-5)"` - Direct test for error handling (should raise ValueError)
- `uv run pytest test_calculator.py -v` - Run backend tests with zero regressions (alternative test runner)
- `uv run ruff check calculator.py test_calculator.py` - Run linter to verify code quality

## Notes
- Uses `math.isqrt()` which is available in Python 3.8+ and returns the integer square root (floor)
- Follows the same error handling pattern as divide() by raising ValueError rather than a generic exception
- The function returns an integer (floor) value, not a float, maintaining consistency with other calculator functions
- Edge case: sqrt(2) returns 1 (integer square root truncates decimal)
- Edge case: sqrt(0) returns 0 (valid operation)
- Perfect squares (4, 9, 16, 25, etc.) return exact integer results
- Future enhancement: Could add a float version if decimal precision is needed
- No new dependencies required - math module is part of Python standard library
- The implementation prioritizes simplicity and consistency with existing code patterns
