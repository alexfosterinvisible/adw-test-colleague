# Feature: Add Square Root Function

## Metadata
issue_number: `13`
adw_id: `9f51b7ba`
issue_json: `{"number":13,"title":"Add square root function","body":"Add sqrt(n) function using Newton's method. Raise ValueError for negative inputs."}`

## Feature Description
Add a square root function to the calculator module that computes the square root of a non-negative number using Newton's method (also known as the Babylonian method). The function will raise a ValueError for negative inputs to maintain mathematical correctness. This enhancement extends the calculator's mathematical capabilities with a commonly used operation while demonstrating iterative algorithm implementation.

## User Story
As a calculator user
I want to compute square roots of numbers
So that I can perform advanced mathematical calculations beyond basic arithmetic

## Problem Statement
The calculator module currently supports only basic arithmetic operations (add, subtract, multiply, divide). Users need access to more advanced mathematical functions like square root to solve real-world problems that require root calculations. The implementation must handle edge cases properly, particularly negative inputs which have no real square root.

## Solution Statement
Implement a `sqrt(n: int) -> float` function using Newton's method, an iterative approximation algorithm that converges rapidly to the square root. The function will validate input to reject negative numbers with a descriptive ValueError, ensuring mathematical correctness. Newton's method provides excellent accuracy with few iterations and is a well-understood algorithm suitable for educational and practical purposes.

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Main calculator module where sqrt function will be added
  - Contains existing arithmetic functions (add, subtract, multiply, divide)
  - Follows pattern of type-hinted functions with error handling
  - Includes demonstration code in `__main__` block

- **test_calculator.py** - Test suite for calculator functions
  - Uses unittest framework with TestCalculator class
  - Follows pattern of testing normal operations, edge cases, and error handling
  - Tests need to be added for sqrt function covering various scenarios

- **pyproject.toml** - Project configuration
  - Contains Python version requirement (>=3.11)
  - May need math library if using math.sqrt for validation in tests

- **.adw.yaml** - ADW configuration
  - Defines test command: `uv run pytest`
  - Backend dir: `app/server` (though this project uses root-level files)
  - Test command will be used for validation

### New Files
No new files required. All implementation will be added to existing files.

## Implementation Plan
### Phase 1: Foundation
Implement the core sqrt function with Newton's method algorithm in calculator.py. Add input validation to handle negative numbers appropriately. Follow the existing code style with type hints and clear function structure.

### Phase 2: Core Implementation
Add comprehensive test coverage in test_calculator.py including normal operations (perfect squares, non-perfect squares), edge cases (zero, very small numbers, very large numbers), and error handling (negative inputs). Ensure tests validate both accuracy and error conditions.

### Phase 3: Integration
Update the demonstration code in calculator.py's `__main__` block to showcase the sqrt function. Validate all existing tests still pass to ensure no regressions. Run the full test suite to confirm the feature integrates seamlessly with existing functionality.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Implement sqrt function in calculator.py
- Add `sqrt(n: int) -> float` function after the divide function
- Implement input validation to raise ValueError for negative inputs with message "Cannot calculate square root of negative number"
- Implement Newton's method algorithm:
  - Handle edge case: return 0.0 for input 0
  - Initialize guess as n / 2
  - Iterate until convergence (difference < 0.00001)
  - Use formula: guess = (guess + n / guess) / 2
  - Return the final approximation
- Add function docstring explaining Newton's method and parameters

### Step 2: Update __main__ demonstration block
- Add 2-3 sqrt examples to the demonstration code
- Include examples showing: perfect square (e.g., sqrt(16)), non-perfect square (e.g., sqrt(10)), and zero
- Ensure output format matches existing print statements

### Step 3: Add unit tests for sqrt function
- Add `test_sqrt` method to TestCalculator class
- Test perfect squares: sqrt(0) == 0, sqrt(1) == 1, sqrt(4) == 2, sqrt(16) == 4
- Test non-perfect squares with tolerance: abs(sqrt(10) - 3.162...) < 0.001
- Test large numbers: sqrt(10000) == 100
- Test small numbers: sqrt(1) == 1

### Step 4: Add error handling test for sqrt
- Add `test_sqrt_negative` method to TestCalculator class
- Test that sqrt(-1) raises ValueError
- Test that sqrt(-100) raises ValueError
- Verify error message matches "Cannot calculate square root of negative number"

### Step 5: Run validation commands
- Execute all validation commands listed below to ensure zero regressions
- Verify all tests pass including new sqrt tests
- Confirm calculator.py runs without errors when executed directly
- Check that demonstration output includes sqrt examples

## Testing Strategy
### Unit Tests
- **test_sqrt**: Validates correct sqrt computation for perfect squares (0, 1, 4, 16, 100), non-perfect squares (2, 10, 50), and edge cases (0, 1, large numbers like 10000)
- **test_sqrt_negative**: Validates ValueError is raised for negative inputs (-1, -100) with correct error message
- **Accuracy validation**: Ensure Newton's method converges to within 0.001 of actual value (can compare with math.sqrt if needed for test validation)

### Edge Cases
- **Zero input**: sqrt(0) should return 0.0 without iteration
- **Perfect squares**: sqrt(1), sqrt(4), sqrt(16), sqrt(100) should return exact integer values as floats
- **Non-perfect squares**: sqrt(2), sqrt(10) should converge to accurate approximations
- **Large numbers**: sqrt(10000), sqrt(1000000) should handle efficiently
- **Negative numbers**: sqrt(-1), sqrt(-5), sqrt(-100) should raise ValueError
- **Convergence**: Algorithm should converge within reasonable iterations (typically < 10 for most inputs)

## Acceptance Criteria
- sqrt function is implemented in calculator.py using Newton's method
- Function has proper type hints: `sqrt(n: int) -> float`
- Function raises ValueError with message "Cannot calculate square root of negative number" for negative inputs
- Function returns accurate results within 0.001 for all non-negative inputs
- Zero input returns 0.0 correctly
- Perfect squares return exact values (as floats)
- All existing calculator tests continue to pass (no regressions)
- New test methods test_sqrt and test_sqrt_negative are added to test_calculator.py
- Tests achieve 100% pass rate
- Demonstration code in __main__ includes sqrt examples
- Code follows existing style and conventions in the codebase

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python calculator.py` - Run calculator demonstration to verify sqrt examples work
- `uv run pytest test_calculator.py -v` - Run test suite with verbose output to see all test results
- `uv run pytest test_calculator.py::TestCalculator::test_sqrt -v` - Run sqrt test specifically
- `uv run pytest test_calculator.py::TestCalculator::test_sqrt_negative -v` - Run negative input test specifically
- `python -c "from calculator import sqrt; print(f'sqrt(16) = {sqrt(16)}'); print(f'sqrt(10) = {sqrt(10):.3f}')"` - Direct function test for perfect and non-perfect squares
- `python -c "from calculator import sqrt; sqrt(-5)"` - Verify ValueError is raised for negative input (should error)
- `uv run pytest test_calculator.py` - Run all calculator tests to ensure zero regressions

## Notes
- Newton's method converges very quickly (typically 5-7 iterations) for most inputs
- The algorithm uses initial guess of n/2 which works well for most values
- Convergence threshold of 0.00001 provides good balance between accuracy and performance
- Function returns float type (not int) because square roots are often irrational numbers
- For production code, consider adding iteration limit to prevent infinite loops, but for this implementation the convergence is guaranteed for non-negative inputs
- The implementation does not require importing the math module, making it self-contained
- This feature demonstrates iterative algorithms and numerical methods in the calculator
- Future enhancement: Could support float inputs instead of just int, but keeping int maintains consistency with existing functions
