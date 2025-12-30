# Feature: Add divide function to calculator

## Metadata
issue_number: `1`
adw_id: `de3d954e`
issue_json: `{"number":1,"title":"Add divide function to calculator","body":"Add a divide(a, b) function to calculator.py that divides a by b. Should handle division by zero gracefully."}`

## Feature Description
Add a divide function to the calculator module that performs division operations with proper error handling for division by zero. The function should follow the existing pattern established by the add, subtract, and multiply functions in calculator.py, maintaining consistency in type hints and implementation style.

## User Story
As a calculator user
I want to divide two numbers
So that I can perform division operations with confidence that division by zero is handled gracefully

## Problem Statement
The calculator module currently supports addition, subtraction, and multiplication operations but lacks division functionality. Users need the ability to divide numbers with appropriate error handling to prevent runtime crashes when attempting to divide by zero.

## Solution Statement
Implement a divide function in calculator.py that takes two numeric arguments and returns their quotient. The function will check for division by zero and raise a descriptive ValueError with a clear error message. The implementation will match the existing code style with type hints for integers and return type.

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Main calculator module where the divide function will be added. Currently contains add, subtract, and multiply functions following a consistent pattern with type hints (int parameters and int return type).

### New Files
- **test_calculator.py** - New test file to be created with comprehensive unit tests for all calculator functions including the new divide function, edge cases, and division by zero handling.

## Implementation Plan
### Phase 1: Foundation
Create a comprehensive test suite for the calculator module. This establishes test coverage for existing functionality (add, subtract, multiply) before adding new features, ensuring no regressions occur. The test file will use Python's unittest framework and include tests for normal operations, edge cases, and the division by zero scenario.

### Phase 2: Core Implementation
Implement the divide function in calculator.py following the established pattern. The function will accept two integer parameters (a and b), check if b equals zero, raise a ValueError with message "Cannot divide by zero" if true, otherwise return a // b for integer division (matching the existing function signatures that work with integers).

### Phase 3: Integration
Update the calculator.py main block to demonstrate the new divide function alongside existing operations. Validate that all tests pass and the function integrates seamlessly with the existing module structure.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Create test infrastructure
- Create test_calculator.py in the project root directory
- Import unittest and calculator module
- Create TestCalculator class inheriting from unittest.TestCase
- Add test methods for existing functions: test_add, test_subtract, test_multiply
- Each test should verify basic functionality with simple assertions

### 2. Add division-specific tests
- Create test_divide method to test normal division operations
- Create test_divide_by_zero method using assertRaises to verify ValueError is raised
- Verify error message content matches expected "Cannot divide by zero"
- Add edge case tests: dividing zero by a number, negative number divisions

### 3. Implement divide function
- Add divide function to calculator.py after the multiply function
- Add type hints: def divide(a: int, b: int) -> int
- Implement zero check: if b == 0, raise ValueError("Cannot divide by zero")
- Return integer division: return a // b
- Maintain consistent spacing and style with existing functions

### 4. Update main block demonstration
- Add divide example to the if __name__ == "__main__" block
- Include both a successful division and demonstrate proper usage pattern
- Keep format consistent with existing print statements

### 5. Run validation commands
- Execute all validation commands listed below to ensure zero regressions
- Verify all tests pass
- Confirm calculator works end-to-end with manual testing

## Testing Strategy
### Unit Tests
- **test_add** - Verify addition works correctly (e.g., 2 + 3 = 5)
- **test_subtract** - Verify subtraction works correctly (e.g., 5 - 3 = 2)
- **test_multiply** - Verify multiplication works correctly (e.g., 2 * 3 = 6)
- **test_divide** - Verify division works correctly (e.g., 6 / 3 = 2)
- **test_divide_by_zero** - Verify ValueError is raised with correct message when dividing by zero

### Edge Cases
- Division of zero by a non-zero number (should return 0)
- Division with negative numbers (e.g., -6 / 3 = -2, 6 / -3 = -2, -6 / -3 = 2)
- Division where result would be fractional using integer division (e.g., 7 / 3 = 2)
- Division by zero from both positive and negative numerators

## Acceptance Criteria
- divide function exists in calculator.py with proper type hints
- divide function returns correct integer division results
- divide function raises ValueError with message "Cannot divide by zero" when b=0
- All unit tests pass with 100% success rate
- test_calculator.py has comprehensive coverage of all calculator functions
- Code style matches existing functions (spacing, type hints, formatting)
- Main block demonstrates divide function usage
- No regressions in existing add, subtract, multiply functions

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python -m pytest test_calculator.py -v` - Run all calculator tests with verbose output, verify zero failures
- `python test_calculator.py` - Run tests using unittest directly, confirm all tests pass
- `python calculator.py` - Execute main block to verify demonstrations work without errors
- `python -c "from calculator import divide; print(divide(10, 2)); print(divide(7, 3))"` - Test divide function directly
- `python -c "from calculator import divide; divide(5, 0)"` - Verify division by zero raises ValueError (should exit with error)

## Notes
- The project uses uv for dependency management with Python >=3.11
- No new dependencies are required for this feature
- Using integer division (a // b) maintains consistency with the int type hints
- Consider adding pytest as a dev dependency for better test output: `uv add --dev pytest`
- The existing calculator.py uses a simple functional style without classes or decorators - maintain this pattern
- Future enhancement: Consider supporting float division with a separate function or parameter
