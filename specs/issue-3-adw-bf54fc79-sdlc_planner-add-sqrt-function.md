# Feature: Add Square Root Function

## Metadata
issue_number: `3`
adw_id: `bf54fc79`
issue_json: `{"number":3,"title":"Add square root function","body":"Add sqrt(n) function using Newton's method. Raise ValueError for negative inputs."}`

## Feature Description
Add a mathematical square root function to the calculator module that computes the square root of a number using Newton's method (also known as the Heron method). The function will follow the existing pattern of type-hinted integer parameters and return values, and will include proper error handling for invalid inputs (negative numbers). This extends the calculator's capabilities beyond basic arithmetic operations to include a fundamental mathematical operation.

## User Story
As a calculator user
I want to compute square roots of numbers
So that I can perform more advanced mathematical calculations beyond basic arithmetic

## Problem Statement
The calculator currently only supports basic arithmetic operations (addition, subtraction, multiplication, division). Users need the ability to calculate square roots, which is a common mathematical operation required in many applications including geometry, physics, statistics, and general computation.

## Solution Statement
Implement a `sqrt(n: int) -> int` function using Newton's method for iterative approximation. The function will validate inputs (rejecting negative numbers with ValueError), use Newton's iterative formula (x_new = (x + n/x) / 2) to converge on the square root, and return the integer floor of the result to maintain consistency with the calculator's integer-focused design. The implementation will follow the existing patterns in calculator.py including type hints, error handling conventions, and test coverage.

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Main calculator module where sqrt function will be added alongside existing add, subtract, multiply, and divide functions. Follows pattern of type-hinted functions with proper error handling.

- **test_calculator.py** - Unit test suite using unittest framework. Contains TestCalculator class with test methods for each calculator function. Will need new test methods for sqrt: test_sqrt (normal cases) and test_sqrt_negative (error handling).

- **pyproject.toml** - Project configuration with Python >=3.11 requirement and dev dependencies (ruff linter). No changes needed unless new dependencies are required.

- **app_docs/feature-de3d954e-divide-function.md** - Reference documentation showing the established pattern for calculator functions: type hints, error handling with ValueError, comprehensive tests including edge cases and error scenarios, and demonstration in main block.

- **.adw.yaml** - ADW configuration specifying test_command: "uv run pytest". Used for validation commands.

### New Files
- **.claude/commands/e2e/test_sqrt_function.md** - E2E test specification following the pattern of test_basic_query.md. Will validate sqrt functionality through demonstration script execution, verifying normal operations (sqrt(16) = 4, sqrt(9) = 3, sqrt(2) = 1), edge cases (sqrt(0) = 0, sqrt(1) = 1), and error handling (sqrt(-4) raises ValueError).

## Implementation Plan
### Phase 1: Foundation
Review existing calculator patterns and Newton's method algorithm. Understand the integer division convention used in divide() function and how to apply similar logic to sqrt(). Confirm error handling pattern (ValueError with descriptive messages). Review test structure in test_calculator.py to prepare for adding sqrt tests.

### Phase 2: Core Implementation
Implement sqrt(n: int) -> int function in calculator.py using Newton's method with proper input validation, convergence logic, and integer result. Add comprehensive unit tests in test_calculator.py covering normal operations, edge cases (0, 1, perfect squares, non-perfect squares), and error handling (negative inputs). Update the main block demonstration code in calculator.py to showcase sqrt usage.

### Phase 3: Integration
Create E2E test specification document to validate sqrt works correctly in demonstration context. Run all existing tests to ensure no regressions. Verify the function integrates seamlessly with existing calculator operations. Document the implementation following the pattern established in app_docs/feature-de3d954e-divide-function.md.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Understand Existing Patterns
- Read calculator.py to understand function signature patterns (type hints, docstrings)
- Read test_calculator.py to understand test structure and assertion patterns
- Review app_docs/feature-de3d954e-divide-function.md to understand error handling conventions and documentation requirements

### Implement sqrt Function
- Add sqrt(n: int) -> int function to calculator.py after the divide function
- Add docstring explaining the function uses Newton's method
- Implement input validation: raise ValueError("Cannot calculate square root of negative number") for n < 0
- Handle edge cases: return 0 for n=0, return 1 for n=1
- Implement Newton's method iteration:
  - Start with initial guess x = n
  - Iterate: x_new = (x + n//x) // 2 (using integer division)
  - Convergence condition: when x_new >= x, return x
  - Use integer arithmetic throughout to maintain type consistency
- Add type hints: def sqrt(n: int) -> int

### Add Unit Tests
- Add test_sqrt method to TestCalculator class in test_calculator.py
- Test perfect squares: sqrt(0)=0, sqrt(1)=1, sqrt(4)=2, sqrt(9)=3, sqrt(16)=4, sqrt(25)=5, sqrt(100)=10
- Test non-perfect squares (integer floor): sqrt(2)=1, sqrt(3)=1, sqrt(5)=2, sqrt(8)=2, sqrt(10)=3
- Test larger numbers: sqrt(144)=12, sqrt(1000)=31
- Add test_sqrt_negative method to TestCalculator class
- Test negative input raises ValueError: sqrt(-1), sqrt(-4), sqrt(-100)
- Verify error message matches: "Cannot calculate square root of negative number"

### Update Main Block Demonstration
- Add sqrt examples to the __main__ block in calculator.py
- Show perfect square: sqrt(16) = 4
- Show non-perfect square: sqrt(10) = 3 (demonstrating integer floor behavior)
- Show edge case: sqrt(1) = 1

### Create E2E Test Specification
- Read .claude/commands/test_e2e.md to understand E2E test runner requirements
- Read .claude/commands/e2e/test_basic_query.md to understand E2E test specification format
- Create .claude/commands/e2e/test_sqrt_function.md following the established template
- Define User Story for sqrt validation
- List test steps:
  1. Run calculator.py directly and capture output
  2. Verify sqrt(16) output shows "= 4"
  3. Verify sqrt(10) output shows "= 3" (integer division)
  4. Verify sqrt(1) output shows "= 1"
  5. Test direct function import and execution
  6. Test error handling with python -c command attempting sqrt(-4)
  7. Verify ValueError is raised with correct message
- Define success criteria: all outputs match expected values, error handling works correctly
- Specify validation commands to execute the demonstration script

### Run Validation Commands
- Execute: uv run pytest test_calculator.py -v (validate all tests pass with zero regressions)
- Execute: python calculator.py (validate main block demonstration runs without errors)
- Execute: python -c "from calculator import sqrt; print(sqrt(16))" (validate direct function usage)
- Execute: python -c "from calculator import sqrt; sqrt(-4)" (validate error handling - should raise ValueError)
- Execute: uv run ruff check calculator.py test_calculator.py (validate code quality)
- Read .claude/commands/test_e2e.md
- Execute the new E2E test: Read and follow instructions in .claude/commands/e2e/test_sqrt_function.md

## Testing Strategy
### Unit Tests
- **test_sqrt**: Tests normal sqrt operations with perfect squares (0, 1, 4, 9, 16, 25, 100), non-perfect squares returning integer floor (2→1, 3→1, 5→2, 8→2, 10→3), and larger numbers (144→12, 1000→31)
- **test_sqrt_negative**: Tests error handling for negative inputs (-1, -4, -100), verifying ValueError is raised with message "Cannot calculate square root of negative number"

### Edge Cases
- Zero input (sqrt(0) = 0)
- One input (sqrt(1) = 1)
- Perfect squares vs non-perfect squares (integer floor behavior)
- Large numbers (sqrt(1000) = 31, demonstrating algorithm stability)
- Negative numbers (proper ValueError with descriptive message)
- Newton's method convergence (implicit - algorithm must converge for all valid inputs)

## Acceptance Criteria
- sqrt(n: int) -> int function exists in calculator.py with proper type hints and docstring
- Function uses Newton's method for square root calculation
- Function raises ValueError("Cannot calculate square root of negative number") for n < 0
- Function returns integer floor of square root for all non-negative inputs
- Function handles edge cases: sqrt(0)=0, sqrt(1)=1
- All unit tests in test_calculator.py pass (both new sqrt tests and all existing tests)
- Main block in calculator.py demonstrates sqrt usage with at least 2-3 examples
- Code passes ruff linter checks with zero warnings
- E2E test specification document exists and validates sqrt functionality
- Implementation follows existing patterns in calculator.py (type hints, error handling, code style)
- No regressions in existing calculator functions (add, subtract, multiply, divide)

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python calculator.py` - Run demonstration script showing sqrt examples work correctly
- `python -c "from calculator import sqrt; print(f'sqrt(16) = {sqrt(16)}')"` - Test sqrt(16) = 4
- `python -c "from calculator import sqrt; print(f'sqrt(10) = {sqrt(10)}')"` - Test sqrt(10) = 3 (integer floor)
- `python -c "from calculator import sqrt; print(f'sqrt(0) = {sqrt(0)}')"` - Test sqrt(0) = 0
- `python -c "from calculator import sqrt; print(f'sqrt(1) = {sqrt(1)}')"` - Test sqrt(1) = 1
- `python -c "from calculator import sqrt; sqrt(-4)"` - Verify ValueError raised (should fail with error message)
- `uv run pytest test_calculator.py -v` - Run all unit tests with zero regressions
- `uv run ruff check calculator.py test_calculator.py` - Lint code with zero warnings
- Read [.claude/commands/test_e2e.md], then read and execute your new E2E [.claude/commands/e2e/test_sqrt_function.md] test file to validate sqrt functionality works as demonstrated

## Notes
- Newton's method (Heron method) formula: x_new = (x + n/x) / 2, converges quickly for square roots
- Using integer division throughout maintains type consistency: x_new = (x + n//x) // 2
- Convergence condition for integers: when x_new >= x, we've found the floor(sqrt(n))
- Initial guess x = n works well and simplifies the algorithm (alternative: x = n // 2 + 1)
- Edge case handling: sqrt(0)=0 and sqrt(1)=1 can be returned immediately without iteration
- Error handling follows established pattern: ValueError with descriptive message (consistent with divide-by-zero)
- Integer floor behavior: sqrt(10) = 3 (not 3.16...), sqrt(2) = 1 (not 1.41...) - consistent with calculator's integer-focused design
- Future consideration: Could add a sqrt_float function if floating-point results are needed
- Algorithm complexity: O(log n) iterations for convergence, very efficient
- The implementation should be self-contained in calculator.py with no external dependencies (no math.sqrt)
- Test coverage should match the comprehensive pattern established for divide function
- E2E test validates the function works in demonstration/usage context, complementing unit tests
