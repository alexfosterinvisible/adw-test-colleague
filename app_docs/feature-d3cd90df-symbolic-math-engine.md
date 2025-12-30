# Symbolic Math Engine

**ADW ID:** d3cd90df
**Date:** 2025-12-30
**Specification:** specs/issue-33-adw-d3cd90df-sdlc_planner-symbolic-math-engine.md

## Overview

A comprehensive symbolic mathematics engine implemented in pure Python. The module provides a complete pipeline for parsing mathematical expressions into an Abstract Syntax Tree (AST), performing symbolic operations (simplification, differentiation, integration, expansion), evaluating expressions through substitution, solving equations, and generating LaTeX-formatted output.

## What Was Built

- **Tokenizer**: Converts string expressions into token streams (numbers, variables, operators, functions)
- **Parser**: Recursive descent parser with proper operator precedence (PEMDAS)
- **AST Node Classes**: Num, Var, BinOp, UnaryOp, FuncCall expression nodes
- **Symbolic Operations**: parse, simplify, differentiate, integrate, expand, substitute
- **Equation Solver**: Linear and quadratic equation solving
- **Formatters**: String conversion and LaTeX output generation
- **Comprehensive Test Suite**: 547 lines of tests covering all functionality

## Technical Implementation

### Files Modified

| File                   | Lines   | Description                                               |
|:-----------------------|:--------|:----------------------------------------------------------|
| `symbolic_math.py`     | +1654   | New module containing the complete symbolic math engine   |
| `test_symbolic_math.py`| +547    | Comprehensive test suite with unittest framework          |
| `.ports.env`           | +3      | Minor port configuration changes                          |

### Key Changes

- **Tokenizer**: Handles integers, floats, scientific notation, variables, operators (+, -, *, /, ^), and functions (sin, cos, tan, log, ln, exp, sqrt)
- **Parser**: Implements recursive descent with explicit precedence levels (addition/subtraction < multiplication/division < power < unary/functions/parentheses)
- **Differentiation**: Implements power rule, product rule, quotient rule, chain rule, and derivatives of trigonometric/exponential/logarithmic functions
- **Integration**: Supports polynomial integration using power rule, plus basic function integrals (sin, cos, exp)
- **Equation Solver**: Handles linear (a*x + b = 0) and quadratic (a*x^2 + b*x + c = 0) equations using the quadratic formula

## How to Use

1. **Parse an expression**:
   ```python
   from symbolic_math import parse, to_string
   expr = parse('x^2 + 2*x + 1')
   print(to_string(expr))  # x^2 + 2*x + 1
   ```

2. **Substitute variables**:
   ```python
   from symbolic_math import substitute
   result = substitute('x^2 + y', {'x': 3, 'y': 1})
   print(result)  # 10
   ```

3. **Simplify expressions**:
   ```python
   from symbolic_math import simplify
   print(simplify('x + x'))  # 2*x
   print(simplify('x * 1'))  # x
   ```

4. **Differentiate**:
   ```python
   from symbolic_math import differentiate
   print(differentiate('x^3 + 2*x', 'x'))  # 3*x^2 + 2
   ```

5. **Integrate**:
   ```python
   from symbolic_math import integrate
   print(integrate('2*x', 'x'))  # x^2 + C
   ```

6. **Expand expressions**:
   ```python
   from symbolic_math import expand
   print(expand('(x + 1)^2'))  # x^2 + 2*x + 1
   ```

7. **Solve equations**:
   ```python
   from symbolic_math import solve
   print(solve('x^2 - 4 = 0', 'x'))  # [-2, 2]
   ```

8. **Generate LaTeX**:
   ```python
   from symbolic_math import latex
   print(latex('x^2/2'))  # \frac{{x}^{2}}{2}
   ```

## Configuration

No configuration required. The module is self-contained and uses only the Python standard library `math` module.

## Testing

Run the test suite:
```bash
uv run pytest test_symbolic_math.py -v
```

Run the demonstration script:
```bash
python symbolic_math.py
```

## Notes

- **Supported operators**: +, -, *, /, ^ (power)
- **Supported functions**: sin, cos, tan, log, ln, exp, sqrt
- **Error handling**: Raises `SyntaxError` for malformed expressions, `ValueError` for unsolvable equations or undefined operations
- **Integration limitations**: Only supports polynomial integration; complex integrals raise `ValueError`
- **Equation solving**: Limited to linear and quadratic equations; higher-degree or transcendental equations raise `ValueError`
- **Pure Python**: No external dependencies (e.g., SymPy) required
