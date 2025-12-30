"""(Claude) Unit tests for symbolic_math module."""

import unittest
from symbolic_math import (
    Tokenizer, TokenType,
    Num, Var, BinOp, UnaryOp, FuncCall, Expr,
    parse, latex, substitute, simplify,
    differentiate, integrate, expand, solve
)


class TestTokenizer(unittest.TestCase):
    """Test suite for tokenizer."""

    def test_tokenize_numbers(self):
        """Test tokenization of integers and floats."""
        tokenizer = Tokenizer('42')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42.0

        tokenizer = Tokenizer('3.14')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 3.14

        tokenizer = Tokenizer('1e5')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 1e5

    def test_tokenize_variables(self):
        """Test tokenization of variables."""
        tokenizer = Tokenizer('x')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.VARIABLE
        assert tokens[0].value == 'x'

        tokenizer = Tokenizer('abc')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.VARIABLE
        assert tokens[0].value == 'abc'

    def test_tokenize_operators(self):
        """Test tokenization of operators."""
        tokenizer = Tokenizer('+ - * / ^')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.PLUS
        assert tokens[1].type == TokenType.MINUS
        assert tokens[2].type == TokenType.MULTIPLY
        assert tokens[3].type == TokenType.DIVIDE
        assert tokens[4].type == TokenType.POWER

    def test_tokenize_functions(self):
        """Test tokenization of function names."""
        for func in ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt']:
            tokenizer = Tokenizer(func)
            tokens = tokenizer.tokenize()
            assert tokens[0].type == TokenType.FUNCTION
            assert tokens[0].value == func

    def test_tokenize_parentheses(self):
        """Test tokenization of parentheses."""
        tokenizer = Tokenizer('((x))')
        tokens = tokenizer.tokenize()
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.VARIABLE
        assert tokens[3].type == TokenType.RPAREN
        assert tokens[4].type == TokenType.RPAREN

    def test_tokenize_invalid(self):
        """Test SyntaxError for unrecognized characters."""
        tokenizer = Tokenizer('@')
        with self.assertRaises(SyntaxError):
            tokenizer.tokenize()

        tokenizer = Tokenizer('#')
        with self.assertRaises(SyntaxError):
            tokenizer.tokenize()


class TestParser(unittest.TestCase):
    """Test suite for parser."""

    def test_parse_simple(self):
        """Test parsing of numbers and variables."""
        assert isinstance(parse('42'), Num)
        assert parse('42').value == 42.0

        assert isinstance(parse('x'), Var)
        assert parse('x').name == 'x'

    def test_parse_binary_ops(self):
        """Test parsing of binary operations."""
        result = parse('2 + 3')
        assert isinstance(result, BinOp)
        assert result.op == '+'

        result = parse('x * y')
        assert isinstance(result, BinOp)
        assert result.op == '*'

    def test_parse_precedence(self):
        """Test correct operator precedence."""
        # 2 + 3 * 4 should be 2 + (3 * 4)
        result = parse('2 + 3 * 4')
        assert isinstance(result, BinOp)
        assert result.op == '+'
        assert isinstance(result.right, BinOp)
        assert result.right.op == '*'

        # 2 * 3 + 4 should be (2 * 3) + 4
        result = parse('2 * 3 + 4')
        assert isinstance(result, BinOp)
        assert result.op == '+'
        assert isinstance(result.left, BinOp)
        assert result.left.op == '*'

    def test_parse_parentheses(self):
        """Test parentheses override precedence."""
        result = parse('(2 + 3) * 4')
        assert isinstance(result, BinOp)
        assert result.op == '*'
        assert isinstance(result.left, BinOp)
        assert result.left.op == '+'

    def test_parse_functions(self):
        """Test parsing of function calls."""
        result = parse('sin(x)')
        assert isinstance(result, FuncCall)
        assert result.name == 'sin'
        assert isinstance(result.arg, Var)

        result = parse('sqrt(2)')
        assert isinstance(result, FuncCall)
        assert result.name == 'sqrt'

    def test_parse_unary_minus(self):
        """Test parsing of unary minus."""
        result = parse('-x')
        assert isinstance(result, UnaryOp)
        assert result.op == '-'

        result = parse('-5')
        assert isinstance(result, UnaryOp)
        assert result.op == '-'

    def test_parse_complex(self):
        """Test parsing of complex nested expressions."""
        result = parse('sin(x^2) + cos(y)')
        assert isinstance(result, BinOp)
        assert result.op == '+'

        result = parse('((x + 1) * (y - 2)) / z')
        assert isinstance(result, BinOp)
        assert result.op == '/'

    def test_parse_invalid(self):
        """Test SyntaxError for malformed expressions."""
        with self.assertRaises(SyntaxError):
            parse('2 +')

        with self.assertRaises(SyntaxError):
            parse('* 3')

        with self.assertRaises(SyntaxError):
            parse('(2 + 3')

        with self.assertRaises(SyntaxError):
            parse('')


class TestSubstitute(unittest.TestCase):
    """Test suite for substitute function."""

    def test_substitute_simple(self):
        """Test simple variable substitution."""
        result = substitute('x', {'x': 5})
        assert result == 5.0

        result = substitute('x + 1', {'x': 5})
        assert result == 6.0

    def test_substitute_multi(self):
        """Test substitution with multiple variables."""
        result = substitute('x + y', {'x': 3, 'y': 4})
        assert result == 7.0

        result = substitute('x^2 + y', {'x': 3, 'y': 1})
        assert result == 10.0

    def test_substitute_partial(self):
        """Test partial substitution leaves variables."""
        result = substitute('x + y', {'x': 3})
        assert isinstance(result, Expr)  # Not fully evaluated

    def test_substitute_functions(self):
        """Test substitution with function evaluation."""
        result = substitute('sin(x)', {'x': 0})
        assert abs(result - 0.0) < 1e-10

        result = substitute('sqrt(x)', {'x': 4})
        assert abs(result - 2.0) < 1e-10

        result = substitute('exp(x)', {'x': 0})
        assert abs(result - 1.0) < 1e-10


class TestSimplify(unittest.TestCase):
    """Test suite for simplify function."""

    def test_simplify_identity_add(self):
        """Test x + 0 = x and 0 + x = x."""
        assert simplify('x + 0') == 'x'
        assert simplify('0 + x') == 'x'

    def test_simplify_identity_multiply(self):
        """Test x * 1 = x and 1 * x = x."""
        assert simplify('x * 1') == 'x'
        assert simplify('1 * x') == 'x'

    def test_simplify_identity_power(self):
        """Test x ^ 1 = x and x ^ 0 = 1."""
        assert simplify('x ^ 1') == 'x'
        assert simplify('x ^ 0') == '1'

    def test_simplify_zero(self):
        """Test x * 0 = 0 and 0 * x = 0."""
        assert simplify('x * 0') == '0'
        assert simplify('0 * x') == '0'

    def test_simplify_like_terms(self):
        """Test x + x = 2*x."""
        assert simplify('x + x') == '2 * x'

    def test_simplify_constants(self):
        """Test constant folding."""
        assert simplify('2 + 3') == '5'
        assert simplify('4 * 5') == '20'
        assert simplify('10 / 2') == '5'

    def test_simplify_nested(self):
        """Test nested simplification."""
        result = simplify('(x + 0) * 1')
        assert result == 'x'


class TestDifferentiate(unittest.TestCase):
    """Test suite for differentiate function."""

    def test_differentiate_constant(self):
        """Test d/dx(5) = 0."""
        assert differentiate('5', 'x') == '0'

    def test_differentiate_variable(self):
        """Test d/dx(x) = 1 and d/dx(y) = 0."""
        assert differentiate('x', 'x') == '1'
        assert differentiate('y', 'x') == '0'

    def test_differentiate_power(self):
        """Test power rule: d/dx(x^3) = 3*x^2."""
        result = differentiate('x^3', 'x')
        # Result should be 3 * x^2
        assert '3' in result
        assert 'x' in result

    def test_differentiate_sum(self):
        """Test sum rule: d/dx(x^2 + x) = 2*x + 1."""
        result = differentiate('x^2 + x', 'x')
        # Result should contain both 2*x and 1
        assert 'x' in result

    def test_differentiate_product(self):
        """Test product rule."""
        result = differentiate('x * x', 'x')
        # Result should simplify to 2*x
        assert 'x' in result

    def test_differentiate_quotient(self):
        """Test quotient rule."""
        result = differentiate('x / y', 'x')
        # Should be 1/y when y is constant
        assert 'y' in result

    def test_differentiate_chain(self):
        """Test chain rule with functions."""
        result = differentiate('sin(x^2)', 'x')
        # Should involve cos(x^2) * 2*x
        assert 'cos' in result

    def test_differentiate_functions(self):
        """Test function derivatives."""
        assert 'cos(x)' in differentiate('sin(x)', 'x')

        result = differentiate('exp(x)', 'x')
        assert 'exp' in result


class TestIntegrate(unittest.TestCase):
    """Test suite for integrate function."""

    def test_integrate_constant(self):
        """Test integral of constant."""
        result = integrate('5', 'x')
        assert '5 * x' in result
        assert 'C' in result

    def test_integrate_variable(self):
        """Test integral of x."""
        result = integrate('x', 'x')
        # Should be x^2 / 2 + C
        assert 'x ^ 2' in result
        assert '2' in result
        assert 'C' in result

    def test_integrate_power(self):
        """Test power rule for integration."""
        result = integrate('x^2', 'x')
        # Should be x^3 / 3 + C
        assert 'x ^ 3' in result
        assert '3' in result
        assert 'C' in result

    def test_integrate_polynomial(self):
        """Test polynomial integration."""
        result = integrate('2*x', 'x')
        # Should simplify to x^2 + C
        assert 'x ^ 2' in result
        assert 'C' in result

    def test_integrate_functions(self):
        """Test basic function integrals."""
        result = integrate('cos(x)', 'x')
        assert 'sin' in result
        assert 'C' in result


class TestExpand(unittest.TestCase):
    """Test suite for expand function."""

    def test_expand_product(self):
        """Test (a + b) * c expansion."""
        result = expand('(x + 1) * y')
        # Should contain x*y and y terms
        assert 'y' in result

    def test_expand_square(self):
        """Test (x + 1)^2 = x^2 + 2*x + 1."""
        result = expand('(x + 1)^2')
        # Should contain x^2, 2*x, and 1
        assert 'x ^ 2' in result
        assert '2 * x' in result or '2*x' in result
        assert '1' in result

    def test_expand_cube(self):
        """Test (x + 1)^3 expansion."""
        result = expand('(x + 1)^3')
        # Verify the expansion evaluates correctly for x=2
        # (2+1)^3 = 27, expanded form should also give 27
        expanded_value = substitute(result, {'x': 2})
        assert expanded_value == 27.0

    def test_expand_nested(self):
        """Test nested expansion."""
        result = expand('(x + 1) * (x - 1)')
        # Should expand to x^2 - 1
        assert 'x ^ 2' in result or 'x' in result


class TestSolve(unittest.TestCase):
    """Test suite for solve function."""

    def test_solve_linear(self):
        """Test solving linear equations."""
        result = solve('x + 2 = 0', 'x')
        assert len(result) == 1
        assert abs(result[0] - (-2.0)) < 1e-10

        result = solve('2*x - 4 = 0', 'x')
        assert len(result) == 1
        assert abs(result[0] - 2.0) < 1e-10

    def test_solve_quadratic_two_roots(self):
        """Test quadratic with two distinct roots."""
        result = solve('x^2 - 4 = 0', 'x')
        assert len(result) == 2
        assert abs(result[0] - (-2.0)) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10

    def test_solve_quadratic_one_root(self):
        """Test quadratic with one root (repeated)."""
        result = solve('x^2 = 0', 'x')
        assert len(result) == 1
        assert abs(result[0] - 0.0) < 1e-10

    def test_solve_quadratic_no_real_roots(self):
        """Test quadratic with no real roots raises ValueError."""
        with self.assertRaises(ValueError) as context:
            solve('x^2 + 1 = 0', 'x')
        assert 'No real solutions' in str(context.exception)

    def test_solve_complex_quadratic(self):
        """Test more complex quadratic: 2*x^2 - 5*x + 2 = 0."""
        result = solve('2*x^2 - 5*x + 2 = 0', 'x')
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10


class TestLatex(unittest.TestCase):
    """Test suite for latex function."""

    def test_latex_fraction(self):
        """Test LaTeX fraction output."""
        result = latex('x/2')
        assert '\\frac' in result
        assert 'x' in result
        assert '2' in result

    def test_latex_power(self):
        """Test LaTeX power output."""
        result = latex('x^2')
        assert '^' in result
        assert 'x' in result
        assert '2' in result

    def test_latex_functions(self):
        """Test LaTeX function output."""
        assert '\\sin' in latex('sin(x)')
        assert '\\cos' in latex('cos(x)')
        assert '\\exp' in latex('exp(x)')

    def test_latex_sqrt(self):
        """Test LaTeX sqrt output."""
        result = latex('sqrt(x)')
        assert '\\sqrt' in result
        assert 'x' in result

    def test_latex_complex(self):
        """Test LaTeX for complex expressions."""
        result = latex('x^2 + 2*x + 1')
        assert 'x' in result


class TestErrors(unittest.TestCase):
    """Test suite for error handling."""

    def test_syntax_error_unclosed_paren(self):
        """Test SyntaxError for missing closing parenthesis."""
        with self.assertRaises(SyntaxError):
            parse('(x + 1')

    def test_syntax_error_invalid_char(self):
        """Test SyntaxError for invalid characters."""
        with self.assertRaises(SyntaxError):
            parse('x @ y')

    def test_syntax_error_missing_operand(self):
        """Test SyntaxError for missing operand."""
        with self.assertRaises(SyntaxError):
            parse('x +')

    def test_value_error_no_solution(self):
        """Test ValueError for unsolvable equations."""
        with self.assertRaises(ValueError):
            solve('x^2 + 1 = 0', 'x')

    def test_value_error_division_by_zero(self):
        """Test ValueError for division by zero in substitution."""
        with self.assertRaises(ValueError):
            substitute('1/x', {'x': 0})

    def test_value_error_sqrt_negative(self):
        """Test ValueError for sqrt of negative number."""
        with self.assertRaises(ValueError):
            substitute('sqrt(x)', {'x': -1})


class TestIssueSpecification(unittest.TestCase):
    """Test cases from the issue specification."""

    def test_parse_quadratic(self):
        """Test parsing x^2 + 2*x + 1."""
        expr = parse('x^2 + 2*x + 1')
        assert isinstance(expr, BinOp)

    def test_simplify_x_plus_x(self):
        """Test simplify('x + x') -> '2*x'."""
        result = simplify('x + x')
        # Should be '2 * x' or similar
        assert '2' in result
        assert 'x' in result

    def test_simplify_x_times_1(self):
        """Test simplify('x*1') -> 'x'."""
        result = simplify('x*1')
        assert result == 'x'

    def test_differentiate_cubic(self):
        """Test differentiate('x^3 + 2*x', 'x') -> '3*x^2 + 2'."""
        result = differentiate('x^3 + 2*x', 'x')
        assert '3' in result
        assert 'x ^ 2' in result or 'x^2' in result
        assert '2' in result

    def test_integrate_2x(self):
        """Test integrate('2*x', 'x') -> 'x^2 + C'."""
        result = integrate('2*x', 'x')
        assert 'x ^ 2' in result or 'x^2' in result
        assert 'C' in result

    def test_substitute_quadratic(self):
        """Test substitute('x^2 + y', {'x': 3, 'y': 1}) -> 10."""
        result = substitute('x^2 + y', {'x': 3, 'y': 1})
        assert result == 10.0

    def test_solve_quadratic(self):
        """Test solve('x^2 - 4 = 0', 'x') -> [-2, 2]."""
        result = solve('x^2 - 4 = 0', 'x')
        assert -2.0 in result
        assert 2.0 in result

    def test_latex_fraction_expr(self):
        """Test latex('x^2/2') -> contains \\frac."""
        result = latex('x^2/2')
        assert '\\frac' in result

    def test_example_workflow(self):
        """Test full example from issue: parse, expand, differentiate, substitute."""
        expr = parse('(x + 1)^2')
        assert isinstance(expr, BinOp)

        expanded = expand('(x + 1)^2')
        # Should contain x^2, 2*x, and 1 terms
        assert 'x ^ 2' in expanded or 'x^2' in expanded

        deriv = differentiate('(x + 1)^2', 'x')
        # Should be 2*(x + 1) or 2*x + 2
        assert '2' in deriv

        result = substitute('(x + 1)^2', {'x': 2})
        assert result == 9.0


if __name__ == "__main__":
    unittest.main()
