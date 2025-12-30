# Feature: Add Symbolic Math Engine

## Metadata
issue_number: `33`
adw_id: `d3cd90df`
issue_json: `{"number":33,"title":"Add symbolic math engine","body":"Implement a symbolic mathematics engine:\n\n1. **Expression tree**: Parse `'x^2 + 2*x + 1'` into AST, support variables, constants, operators\n2. **Simplification**: `simplify('x + x')` → `'2*x'`, `simplify('x*1')` → `'x'`, collect like terms\n3. **Differentiation**: `differentiate('x^3 + 2*x', 'x')` → `'3*x^2 + 2'` (power rule, chain rule, product rule)\n4. **Integration**: `integrate('2*x', 'x')` → `'x^2 + C'` (basic polynomial integration)\n5. **Substitution**: `substitute('x^2 + y', {'x': 3, 'y': 1})` → `10`\n6. **Equation solving**: `solve('x^2 - 4 = 0', 'x')` → `[2, -2]` (quadratic formula, linear equations)\n7. **Pretty printing**: `latex('x^2/2')` → `'\\\\frac{x^2}{2}'`\n\nRequirements:\n- Tokenizer → Parser → AST → Evaluator pipeline\n- Support: `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `log`, `exp`, `sqrt`\n- Raise `SyntaxError` for malformed expressions\n- Raise `ValueError` for unsolvable equations\n- Handle nested parentheses and operator precedence\n\nExample:\n```python\nexpr = parse('(x + 1)^2')\nexpanded = expand(expr)        # 'x^2 + 2*x + 1'\nderiv = differentiate(expr, 'x')  # '2*(x + 1)' or '2*x + 2'\nresult = substitute(expr, {'x': 2})  # 9\n```"}`

## Feature Description
Implement a comprehensive symbolic mathematics engine that can parse mathematical expressions into an Abstract Syntax Tree (AST), perform symbolic manipulation operations (simplification, differentiation, integration, expansion), evaluate expressions through substitution, solve equations, and generate LaTeX-formatted output. The engine follows a classical compiler architecture: Tokenizer → Parser → AST → Evaluator/Transformer pipeline. It supports variables, constants, arithmetic operators (+, -, *, /, ^), and mathematical functions (sin, cos, log, exp, sqrt). The implementation will be pure Python with proper operator precedence handling, nested parentheses support, and comprehensive error handling for malformed expressions and unsolvable equations.

## User Story
As a developer using this calculator application
I want to perform symbolic mathematical operations on expressions
So that I can parse, simplify, differentiate, integrate, and solve algebraic expressions programmatically without external CAS dependencies

## Problem Statement
The calculator module currently only supports basic arithmetic and statistical operations on numeric values. Users need symbolic mathematics capabilities to manipulate algebraic expressions - parsing expressions containing variables, simplifying expressions, computing symbolic derivatives and integrals, solving equations, and generating formatted output. This requires a complete expression tree architecture that can represent and transform mathematical expressions symbolically.

## Solution Statement
Create a new `symbolic_math.py` module following existing code patterns (type hints, specific error types, comprehensive tests). The module will implement:
1. **Tokenizer**: Convert string expressions into a stream of tokens (numbers, variables, operators, functions, parentheses)
2. **Parser**: Build AST using recursive descent with proper operator precedence (PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)
3. **AST Node Classes**: Number, Variable, BinaryOp, UnaryOp, FunctionCall nodes
4. **Transformers**: Simplify, differentiate, integrate, expand, substitute operations on AST
5. **Evaluator**: Numerical evaluation with variable substitution
6. **Solver**: Linear and quadratic equation solving
7. **Formatter**: String and LaTeX output generation

Key implementation details:
- Recursive descent parser with explicit precedence levels
- Pattern matching for symbolic simplification rules
- Chain rule, product rule, power rule for differentiation
- Power rule for polynomial integration
- Quadratic formula and linear equation solving
- Proper error handling with SyntaxError and ValueError

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Reference for existing code patterns: type hints, docstrings, error handling with ValueError, and main block demonstration. The symbolic math module should follow the same patterns for consistency.

- **statistics.py** - Reference for a larger module structure with multiple function groups, helper functions, comprehensive docstrings, and validation patterns. Shows how to organize a complex module with clear section headers and consistent patterns.

- **test_calculator.py** - Reference for test structure using unittest framework with TestCase class. Symbolic math tests should follow the same pattern with comprehensive test methods covering normal operations, edge cases, and error handling.

- **test_statistics.py** - Reference for comprehensive test coverage patterns including testing edge cases, error conditions, and numerical accuracy assertions.

- **pyproject.toml** - Project configuration. No new dependencies needed since we're using pure Python. The symbolic math engine will be entirely self-contained.

- **.adw.yaml** - ADW configuration specifying test_command: "uv run pytest". Used for validation commands.

### New Files
- **symbolic_math.py** - New module containing the symbolic mathematics engine. Will be placed at project root alongside calculator.py and statistics.py. Contains:
  - Token types and Tokenizer class
  - AST node classes: Expr (base), Num, Var, BinOp, UnaryOp, FuncCall
  - Parser class with recursive descent implementation
  - Public API functions: parse, simplify, differentiate, integrate, expand, substitute, solve, latex, to_string
  - Helper functions for pattern matching and tree transformation

- **test_symbolic_math.py** - Comprehensive test suite for all symbolic math functions. Will follow test_calculator.py pattern with TestSymbolicMath class containing test methods for each function group.

## Implementation Plan
### Phase 1: Foundation
Build the lexical analysis (tokenizer) and parsing infrastructure. Define token types for numbers, variables, operators (+, -, *, /, ^), functions (sin, cos, log, exp, sqrt), and parentheses. Implement the Tokenizer class to convert string input into token streams. Define the AST node class hierarchy. Implement the recursive descent Parser with proper operator precedence handling.

### Phase 2: Core Implementation
Implement the core symbolic operations:
1. **to_string** and **latex** formatters for AST output
2. **substitute** for variable replacement and numerical evaluation
3. **simplify** with algebraic simplification rules (collect like terms, identity rules)
4. **differentiate** with power rule, product rule, chain rule, function derivatives
5. **integrate** with power rule for polynomials
6. **expand** for expanding products and powers
7. **solve** for linear and quadratic equations

### Phase 3: Integration
Create comprehensive test suite covering all functions with normal cases, edge cases, and error handling. Add main block demonstration showing example usage. Validate all examples from the issue specification work correctly. Ensure proper error messages for malformed expressions (SyntaxError) and unsolvable equations (ValueError).

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Create Symbolic Math Module Foundation - Token Types and Tokenizer
- Create symbolic_math.py file at project root
- Add module docstring explaining purpose and listing all public functions
- Define TokenType enum: NUMBER, VARIABLE, PLUS, MINUS, MULTIPLY, DIVIDE, POWER, LPAREN, RPAREN, FUNCTION, COMMA, EQUALS, EOF
- Define Token dataclass with type, value, and position fields
- Implement Tokenizer class:
  - __init__(self, text: str) stores input and position
  - _peek() returns current character without advancing
  - _advance() returns current character and advances position
  - _skip_whitespace() skips whitespace characters
  - _number() reads a number (int or float)
  - _identifier() reads identifier (variable or function name)
  - tokenize() -> list[Token] returns all tokens
- Handle function names: sin, cos, tan, log, ln, exp, sqrt
- Raise SyntaxError for unrecognized characters with position info

### Step 2: Define AST Node Classes
- Define base Expr class (abstract) with __repr__ method
- Define Num(Expr) class with value: float attribute
- Define Var(Expr) class with name: str attribute
- Define BinOp(Expr) class with op: str, left: Expr, right: Expr attributes
- Define UnaryOp(Expr) class with op: str, operand: Expr attributes (for negation)
- Define FuncCall(Expr) class with name: str, arg: Expr attributes
- Add __eq__ method to each class for testing equality
- Add __hash__ method to each class for use in collections

### Step 3: Implement Recursive Descent Parser
- Implement Parser class with tokens list and position tracking
- Define operator precedence levels:
  - Level 1 (lowest): Addition, Subtraction (+, -)
  - Level 2: Multiplication, Division (*, /)
  - Level 3: Power (^) - right associative
  - Level 4 (highest): Unary operators, Functions, Parentheses
- Implement helper methods:
  - _current_token() returns token at current position
  - _eat(token_type) consumes expected token or raises SyntaxError
  - _peek_token() looks ahead without consuming
- Implement parsing methods by precedence:
  - _parse_expression() entry point, handles lowest precedence
  - _parse_term() handles +, -
  - _parse_factor() handles *, /
  - _parse_power() handles ^ (right associative)
  - _parse_unary() handles unary -
  - _parse_primary() handles numbers, variables, functions, parentheses
- Implement parse(text: str) -> Expr public function that tokenizes and parses

### Step 4: Implement String Conversion and LaTeX Formatting
- Implement to_string(expr: Expr) -> str function:
  - Num: return str(value), handle int vs float display
  - Var: return name
  - BinOp: return f"({left} op {right})" with minimal parentheses
  - UnaryOp: return f"(-{operand})"
  - FuncCall: return f"name({arg})"
- Implement latex(expr: str | Expr) -> str function:
  - If string input, parse first
  - Num: return str(value)
  - Var: return name
  - BinOp with /: return f"\\frac{{{left}}}{{{right}}}"
  - BinOp with ^: return f"{{{left}}}^{{{right}}}"
  - BinOp with *: return f"{left} \\cdot {right}"
  - FuncCall: return f"\\{name}({arg})" for sin, cos, etc.
  - Handle sqrt specially: return f"\\sqrt{{{arg}}}"

### Step 5: Implement Substitution and Evaluation
- Implement substitute(expr: str | Expr, values: dict[str, float]) -> float | Expr:
  - If string input, parse first
  - Recursively traverse AST:
    - Num: return value
    - Var: if name in values, return Num(values[name]) else return Var unchanged
    - BinOp: substitute in both operands, if both Num compute result
    - FuncCall: substitute in arg, if Num evaluate function
  - Supported functions: sin, cos, tan, log (natural), ln, exp, sqrt
  - Return float if fully evaluated, Expr if variables remain
- Import math module for function evaluation
- Raise ValueError for undefined functions or invalid operations (log of negative, sqrt of negative)

### Step 6: Implement Simplification
- Implement simplify(expr: str | Expr) -> Expr function:
  - If string input, parse first
  - Apply simplification rules recursively:
  - Identity rules:
    - x + 0 = x, 0 + x = x
    - x - 0 = x
    - x * 1 = x, 1 * x = x
    - x * 0 = 0, 0 * x = 0
    - x / 1 = x
    - x ^ 0 = 1
    - x ^ 1 = x
  - Constant folding: if both operands are Num, compute result
  - Like terms collection: x + x = 2*x, a*x + b*x = (a+b)*x
  - Negation simplification: --x = x, -(- x) = x
  - Apply rules until no more changes (fixed point)
- Return simplified AST

### Step 7: Implement Differentiation
- Implement differentiate(expr: str | Expr, var: str) -> Expr function:
  - If string input, parse first
  - Apply differentiation rules recursively:
  - d/dx(c) = 0 for constants
  - d/dx(x) = 1
  - d/dx(y) = 0 for other variables
  - Sum rule: d/dx(u + v) = du/dx + dv/dx
  - Difference rule: d/dx(u - v) = du/dx - dv/dx
  - Product rule: d/dx(u * v) = u * dv/dx + v * du/dx
  - Quotient rule: d/dx(u / v) = (v * du/dx - u * dv/dx) / v^2
  - Power rule: d/dx(x^n) = n * x^(n-1)
  - Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
  - Function derivatives:
    - d/dx(sin(u)) = cos(u) * du/dx
    - d/dx(cos(u)) = -sin(u) * du/dx
    - d/dx(exp(u)) = exp(u) * du/dx
    - d/dx(log(u)) = (1/u) * du/dx
    - d/dx(sqrt(u)) = (1/(2*sqrt(u))) * du/dx
- Simplify result before returning

### Step 8: Implement Integration
- Implement integrate(expr: str | Expr, var: str) -> Expr function:
  - If string input, parse first
  - Apply integration rules for polynomials:
  - Integral of c = c*x (constant)
  - Integral of x = x^2/2
  - Integral of x^n = x^(n+1)/(n+1) for n != -1
  - Integral of 1/x = log(x)
  - Sum rule: integral(u + v) = integral(u) + integral(v)
  - Constant multiple: integral(c*f) = c * integral(f)
  - Basic function integrals:
    - integral(sin(x)) = -cos(x)
    - integral(cos(x)) = sin(x)
    - integral(exp(x)) = exp(x)
  - Add "+ C" constant of integration representation
  - Raise ValueError for expressions that cannot be integrated symbolically
- Simplify result before returning

### Step 9: Implement Expansion
- Implement expand(expr: str | Expr) -> Expr function:
  - If string input, parse first
  - Expand products: (a + b) * (c + d) = a*c + a*d + b*c + b*d
  - Expand powers: (a + b)^n using binomial expansion
  - Expand nested expressions recursively
  - Combine and simplify result
- Helper function for binomial coefficients (can reuse from statistics.py concept)
- Handle integer powers only (raise ValueError for non-integer exponents in expansion)

### Step 10: Implement Equation Solving
- Implement solve(equation: str, var: str) -> list[float] function:
  - Parse equation containing '='
  - Rearrange to form: expr = 0
  - For linear equations (a*x + b = 0): x = -b/a
  - For quadratic equations (a*x^2 + b*x + c = 0): use quadratic formula
    - discriminant = b^2 - 4*a*c
    - If discriminant < 0: raise ValueError("No real solutions")
    - If discriminant = 0: return [-b/(2*a)]
    - If discriminant > 0: return [(-b + sqrt(d))/(2*a), (-b - sqrt(d))/(2*a)]
  - Extract coefficients by analyzing AST structure
  - Raise ValueError for unsupported equation types (higher degree, transcendental)
- Return list of solutions sorted ascending

### Step 11: Add Main Block Demonstration
- Add `if __name__ == "__main__":` block to symbolic_math.py
- Demonstrate parsing: parse('x^2 + 2*x + 1')
- Demonstrate to_string conversion
- Demonstrate latex output for 'x^2/2' -> '\frac{x^2}{2}'
- Demonstrate substitution: substitute('x^2 + y', {'x': 3, 'y': 1}) -> 10
- Demonstrate simplification: simplify('x + x') -> '2*x', simplify('x*1') -> 'x'
- Demonstrate differentiation: differentiate('x^3 + 2*x', 'x') -> '3*x^2 + 2'
- Demonstrate integration: integrate('2*x', 'x') -> 'x^2 + C'
- Demonstrate expansion: expand('(x + 1)^2') -> 'x^2 + 2*x + 1'
- Demonstrate solving: solve('x^2 - 4 = 0', 'x') -> [2, -2]
- Print all results with clear labels

### Step 12: Create Test Suite
- Create test_symbolic_math.py following test_calculator.py pattern
- Create TestSymbolicMath class inheriting from unittest.TestCase

**Tokenizer tests:**
- test_tokenize_numbers: integers, floats, negative numbers
- test_tokenize_variables: single letter, multi-letter
- test_tokenize_operators: +, -, *, /, ^
- test_tokenize_functions: sin, cos, log, exp, sqrt
- test_tokenize_parentheses: nested parentheses
- test_tokenize_invalid: SyntaxError for unrecognized characters

**Parser tests:**
- test_parse_simple: numbers, variables
- test_parse_binary_ops: arithmetic operations
- test_parse_precedence: correct operator precedence
- test_parse_parentheses: override precedence
- test_parse_functions: function calls
- test_parse_unary_minus: negative numbers and expressions
- test_parse_complex: nested expressions
- test_parse_invalid: SyntaxError for malformed expressions

**Substitute tests:**
- test_substitute_simple: single variable
- test_substitute_multi: multiple variables
- test_substitute_partial: some variables unsubstituted
- test_substitute_functions: evaluate sin, cos, etc.

**Simplify tests:**
- test_simplify_identity: x + 0, x * 1, x ^ 1, etc.
- test_simplify_zero: x * 0, 0 / x
- test_simplify_like_terms: x + x = 2*x
- test_simplify_constants: 2 + 3 = 5
- test_simplify_nested: complex nested simplification

**Differentiate tests:**
- test_differentiate_constant: d/dx(5) = 0
- test_differentiate_variable: d/dx(x) = 1
- test_differentiate_power: d/dx(x^3) = 3*x^2
- test_differentiate_sum: d/dx(x^2 + x) = 2*x + 1
- test_differentiate_product: product rule
- test_differentiate_quotient: quotient rule
- test_differentiate_chain: chain rule with functions
- test_differentiate_functions: sin, cos, exp, log derivatives

**Integrate tests:**
- test_integrate_constant: integral of 5 = 5*x
- test_integrate_variable: integral of x = x^2/2
- test_integrate_power: integral of x^n
- test_integrate_polynomial: integral of polynomial
- test_integrate_functions: integral of sin, cos, exp

**Expand tests:**
- test_expand_product: (a + b) * (c + d)
- test_expand_square: (x + 1)^2 = x^2 + 2*x + 1
- test_expand_cube: (x + 1)^3
- test_expand_nested: nested expansions

**Solve tests:**
- test_solve_linear: x + 2 = 0 -> [-2]
- test_solve_quadratic_two_roots: x^2 - 4 = 0 -> [-2, 2]
- test_solve_quadratic_one_root: x^2 = 0 -> [0]
- test_solve_quadratic_no_real_roots: x^2 + 1 = 0 raises ValueError
- test_solve_complex_quadratic: 2*x^2 - 5*x + 2 = 0

**Latex tests:**
- test_latex_fraction: x/2 -> \frac{x}{2}
- test_latex_power: x^2 -> {x}^{2}
- test_latex_functions: sin(x) -> \sin(x)
- test_latex_sqrt: sqrt(x) -> \sqrt{x}
- test_latex_complex: complex expressions

**Error tests:**
- test_syntax_error_unclosed_paren: missing closing parenthesis
- test_syntax_error_invalid_char: @ # etc
- test_syntax_error_missing_operand: 2 +
- test_value_error_no_solution: unsolvable equations
- test_value_error_undefined_function: unknown function names

### Step 13: Run Validation Commands
- Execute all validation commands to verify implementation
- Fix any failing tests or issues discovered
- Ensure 100% test pass rate with zero regressions

## Testing Strategy
### Unit Tests
- **Tokenizer tests**: Verify tokenization of numbers (int, float), variables, operators, functions, parentheses, and error handling for invalid characters
- **Parser tests**: Verify AST construction with correct precedence, associativity, nested expressions, and error handling for syntax errors
- **Substitute tests**: Verify variable replacement, partial evaluation, function evaluation, and error handling
- **Simplify tests**: Verify identity rules, zero rules, constant folding, like terms collection
- **Differentiate tests**: Verify power rule, product rule, quotient rule, chain rule, function derivatives
- **Integrate tests**: Verify polynomial integration, function integrals, constant of integration
- **Expand tests**: Verify product expansion, binomial expansion, nested expansion
- **Solve tests**: Verify linear equation solving, quadratic formula, discriminant handling
- **Latex tests**: Verify proper LaTeX formatting for fractions, powers, functions

### Edge Cases
- Empty string input: raise SyntaxError
- Single number or variable: parse and return correctly
- Multiple consecutive operators: raise SyntaxError
- Unbalanced parentheses: raise SyntaxError
- Unknown function names: raise SyntaxError
- Division by zero during evaluation: raise ValueError
- Negative sqrt during evaluation: raise ValueError
- Variable not in substitution dict: keep as variable
- Differentiate with respect to non-existent variable: return 0
- Integrate non-polynomial: raise ValueError
- Solve unsupported equation type: raise ValueError
- Solve equation with no real solutions: raise ValueError

## Acceptance Criteria
- symbolic_math.py module exists with all functions implemented
- All functions have proper type hints and docstrings following project conventions
- Tokenizer correctly handles: numbers, variables, operators (+, -, *, /, ^), functions (sin, cos, log, exp, sqrt), parentheses
- Parser implements correct operator precedence (PEMDAS) and right-associativity for ^
- parse() raises SyntaxError for malformed expressions with helpful error messages
- to_string() and latex() produce correct formatted output
- substitute() correctly replaces variables and evaluates to numbers when possible
- simplify() applies algebraic simplification rules correctly
- differentiate() implements power, product, quotient, and chain rules correctly
- integrate() handles polynomial integration correctly
- expand() correctly expands products and powers
- solve() handles linear and quadratic equations, raises ValueError for unsolvable
- All unit tests pass (test_symbolic_math.py)
- All existing tests pass (test_calculator.py, test_statistics.py) - zero regressions
- Code passes ruff linter with zero warnings
- Main block demonstrates all example usage from issue specification
- Examples match: parse('(x + 1)^2'), expand -> 'x^2 + 2*x + 1', differentiate -> '2*x + 2', substitute({'x': 2}) -> 9

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python symbolic_math.py` - Run demonstration script showing all symbolic math examples work correctly
- `python -c "from symbolic_math import parse, to_string; print(to_string(parse('x^2 + 2*x + 1')))"` - Verify parsing works
- `python -c "from symbolic_math import simplify; print(simplify('x + x'))"` - Verify simplify produces '2*x'
- `python -c "from symbolic_math import simplify; print(simplify('x*1'))"` - Verify simplify produces 'x'
- `python -c "from symbolic_math import differentiate; print(differentiate('x^3 + 2*x', 'x'))"` - Verify differentiation produces '3*x^2 + 2'
- `python -c "from symbolic_math import integrate; print(integrate('2*x', 'x'))"` - Verify integration produces 'x^2 + C'
- `python -c "from symbolic_math import substitute; print(substitute('x^2 + y', {'x': 3, 'y': 1}))"` - Verify substitution produces 10
- `python -c "from symbolic_math import solve; print(solve('x^2 - 4 = 0', 'x'))"` - Verify solving produces [-2, 2]
- `python -c "from symbolic_math import latex; print(latex('x^2/2'))"` - Verify LaTeX produces '\frac{x^2}{2}'
- `python -c "from symbolic_math import expand; print(expand('(x + 1)^2'))"` - Verify expansion produces 'x^2 + 2*x + 1'
- `python -c "from symbolic_math import parse; parse('2 +')"` - Verify SyntaxError raised for malformed expression
- `python -c "from symbolic_math import solve; solve('x^2 + 1 = 0', 'x')"` - Verify ValueError raised for no real solutions
- `uv run pytest test_symbolic_math.py -v` - Run all symbolic math tests with zero failures
- `uv run pytest test_calculator.py -v` - Run all calculator tests with zero regressions
- `uv run pytest test_statistics.py -v` - Run all statistics tests with zero regressions
- `uv run ruff check symbolic_math.py test_symbolic_math.py` - Lint new code with zero warnings

## Notes
- Tokenizer should handle both integer and floating-point numbers, including scientific notation if feasible
- Parser uses recursive descent which naturally handles precedence by method call hierarchy
- For simplification, use a fixed-point approach: apply rules until AST stops changing
- Differentiation is straightforward recursive application of calculus rules
- Integration is limited to polynomial forms - complex integrals (e.g., involving products of transcendental functions) should raise ValueError
- For expand(), integer powers only - non-integer powers are not expanded
- Equation solver extracts coefficients by pattern matching on the AST structure:
  - Linear: look for a*x + b form
  - Quadratic: look for a*x^2 + b*x + c form
- LaTeX output should use proper escaping and formatting conventions
- The module name "symbolic_math" avoids conflicts with the statistics module
- Consider adding a repr for AST nodes that shows the tree structure for debugging
- Future enhancements could include: trigonometric identities, more integration techniques, higher-degree polynomial solving, matrix operations
- For chain rule in differentiation, need to recursively differentiate the inner function
- binomial coefficients for expansion can be computed using the formula C(n,k) = n!/(k!(n-k)!)
