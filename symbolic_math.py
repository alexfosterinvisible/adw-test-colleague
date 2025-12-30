"""(Claude) Symbolic mathematics engine with pure Python implementation.

Provides comprehensive symbolic math operations including:
- Expression parsing: parse() - Convert string expressions to AST
- String conversion: to_string() - Convert AST to human-readable string
- LaTeX formatting: latex() - Generate LaTeX output
- Substitution: substitute() - Replace variables with values
- Simplification: simplify() - Apply algebraic simplification rules
- Differentiation: differentiate() - Symbolic derivatives
- Integration: integrate() - Symbolic integration (polynomial)
- Expansion: expand() - Expand products and powers
- Equation solving: solve() - Solve linear and quadratic equations

All implementations use pure Python with only the math module.
Supports: +, -, *, /, ^, sin, cos, tan, log, ln, exp, sqrt
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


# =============================================================================
# Token Types and Tokenizer
# =============================================================================

class TokenType(Enum):
    """Types of tokens recognized by the tokenizer."""
    NUMBER = auto()
    VARIABLE = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    LPAREN = auto()
    RPAREN = auto()
    FUNCTION = auto()
    COMMA = auto()
    EQUALS = auto()
    EOF = auto()


@dataclass
class Token:
    """A token produced by the tokenizer.

    Attributes:
        type: The type of token.
        value: The token's value (number, variable name, etc.).
        position: Position in the input string.
    """
    type: TokenType
    value: Union[str, float]
    position: int


class Tokenizer:
    """Converts a string expression into a stream of tokens.

    Attributes:
        text: The input string to tokenize.
        pos: Current position in the input.
    """

    FUNCTIONS = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt'}

    def __init__(self, text: str) -> None:
        """Initialize tokenizer with input text.

        Args:
            text: The mathematical expression to tokenize.
        """
        self.text = text
        self.pos = 0

    def _peek(self) -> str:
        """Return current character without advancing."""
        if self.pos < len(self.text):
            return self.text[self.pos]
        return ''

    def _advance(self) -> str:
        """Return current character and advance position."""
        char = self._peek()
        self.pos += 1
        return char

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self._peek().isspace():
            self._advance()

    def _number(self) -> Token:
        """Read a number (int or float)."""
        start_pos = self.pos
        result = ''

        # Integer part
        while self._peek().isdigit():
            result += self._advance()

        # Decimal part
        if self._peek() == '.':
            result += self._advance()
            while self._peek().isdigit():
                result += self._advance()

        # Scientific notation
        if self._peek() in 'eE':
            result += self._advance()
            if self._peek() in '+-':
                result += self._advance()
            while self._peek().isdigit():
                result += self._advance()

        return Token(TokenType.NUMBER, float(result), start_pos)

    def _identifier(self) -> Token:
        """Read an identifier (variable or function name)."""
        start_pos = self.pos
        result = ''

        while self._peek().isalnum() or self._peek() == '_':
            result += self._advance()

        if result in self.FUNCTIONS:
            return Token(TokenType.FUNCTION, result, start_pos)
        return Token(TokenType.VARIABLE, result, start_pos)

    def tokenize(self) -> list:
        """Convert input string to list of tokens.

        Returns:
            List of Token objects.

        Raises:
            SyntaxError: For unrecognized characters.
        """
        tokens = []

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            char = self._peek()

            if char.isdigit() or (char == '.' and self.pos + 1 < len(self.text)
                                  and self.text[self.pos + 1].isdigit()):
                tokens.append(self._number())
            elif char.isalpha() or char == '_':
                tokens.append(self._identifier())
            elif char == '+':
                tokens.append(Token(TokenType.PLUS, '+', self.pos))
                self._advance()
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, '-', self.pos))
                self._advance()
            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*', self.pos))
                self._advance()
            elif char == '/':
                tokens.append(Token(TokenType.DIVIDE, '/', self.pos))
                self._advance()
            elif char == '^':
                tokens.append(Token(TokenType.POWER, '^', self.pos))
                self._advance()
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', self.pos))
                self._advance()
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', self.pos))
                self._advance()
            elif char == ',':
                tokens.append(Token(TokenType.COMMA, ',', self.pos))
                self._advance()
            elif char == '=':
                tokens.append(Token(TokenType.EQUALS, '=', self.pos))
                self._advance()
            else:
                raise SyntaxError(f"Unrecognized character '{char}' at position {self.pos}")

        tokens.append(Token(TokenType.EOF, '', self.pos))
        return tokens


# =============================================================================
# AST Node Classes
# =============================================================================

class Expr:
    """Base class for all AST expression nodes."""

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class Num(Expr):
    """Numeric constant node.

    Attributes:
        value: The numeric value.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Num):
            return self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(('Num', self.value))

    def __repr__(self) -> str:
        return f"Num({self.value})"


class Var(Expr):
    """Variable node.

    Attributes:
        name: The variable name.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Var):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(('Var', self.name))

    def __repr__(self) -> str:
        return f"Var('{self.name}')"


class BinOp(Expr):
    """Binary operation node.

    Attributes:
        op: The operator (+, -, *, /, ^).
        left: Left operand expression.
        right: Right operand expression.
    """

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BinOp):
            return (self.op == other.op and
                    self.left == other.left and
                    self.right == other.right)
        return False

    def __hash__(self) -> int:
        return hash(('BinOp', self.op, self.left, self.right))

    def __repr__(self) -> str:
        return f"BinOp('{self.op}', {self.left!r}, {self.right!r})"


class UnaryOp(Expr):
    """Unary operation node (for negation).

    Attributes:
        op: The operator (typically '-').
        operand: The operand expression.
    """

    def __init__(self, op: str, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnaryOp):
            return self.op == other.op and self.operand == other.operand
        return False

    def __hash__(self) -> int:
        return hash(('UnaryOp', self.op, self.operand))

    def __repr__(self) -> str:
        return f"UnaryOp('{self.op}', {self.operand!r})"


class FuncCall(Expr):
    """Function call node.

    Attributes:
        name: Function name (sin, cos, etc.).
        arg: Argument expression.
    """

    def __init__(self, name: str, arg: Expr) -> None:
        self.name = name
        self.arg = arg

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FuncCall):
            return self.name == other.name and self.arg == other.arg
        return False

    def __hash__(self) -> int:
        return hash(('FuncCall', self.name, self.arg))

    def __repr__(self) -> str:
        return f"FuncCall('{self.name}', {self.arg!r})"


# =============================================================================
# Parser
# =============================================================================

class Parser:
    """Recursive descent parser for mathematical expressions.

    Implements proper operator precedence:
    - Level 1 (lowest): Addition, Subtraction (+, -)
    - Level 2: Multiplication, Division (*, /)
    - Level 3: Power (^) - right associative
    - Level 4 (highest): Unary operators, Functions, Parentheses
    """

    def __init__(self, tokens: list) -> None:
        """Initialize parser with token list.

        Args:
            tokens: List of tokens from tokenizer.
        """
        self.tokens = tokens
        self.pos = 0

    def _current_token(self) -> Token:
        """Return the current token."""
        return self.tokens[self.pos]

    def _peek_token(self) -> Token:
        """Look ahead to next token without consuming."""
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return self.tokens[-1]

    def _eat(self, token_type: TokenType) -> Token:
        """Consume expected token or raise error.

        Args:
            token_type: Expected token type.

        Returns:
            The consumed token.

        Raises:
            SyntaxError: If current token doesn't match expected type.
        """
        token = self._current_token()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at position {token.position}"
            )
        self.pos += 1
        return token

    def parse(self) -> Expr:
        """Parse tokens into AST.

        Returns:
            Root expression node.

        Raises:
            SyntaxError: For invalid syntax.
        """
        result = self._parse_expression()
        if self._current_token().type != TokenType.EOF:
            token = self._current_token()
            raise SyntaxError(
                f"Unexpected token {token.type.name} at position {token.position}"
            )
        return result

    def _parse_expression(self) -> Expr:
        """Parse addition and subtraction (lowest precedence)."""
        left = self._parse_term()

        while self._current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._current_token().value
            self.pos += 1
            right = self._parse_term()
            left = BinOp(op, left, right)

        return left

    def _parse_term(self) -> Expr:
        """Parse multiplication and division."""
        left = self._parse_power()

        while self._current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self._current_token().value
            self.pos += 1
            right = self._parse_power()
            left = BinOp(op, left, right)

        return left

    def _parse_power(self) -> Expr:
        """Parse exponentiation (right associative)."""
        base = self._parse_unary()

        if self._current_token().type == TokenType.POWER:
            self.pos += 1
            # Right associative: x^y^z = x^(y^z)
            exponent = self._parse_power()
            return BinOp('^', base, exponent)

        return base

    def _parse_unary(self) -> Expr:
        """Parse unary minus."""
        if self._current_token().type == TokenType.MINUS:
            self.pos += 1
            operand = self._parse_unary()
            return UnaryOp('-', operand)
        elif self._current_token().type == TokenType.PLUS:
            self.pos += 1
            return self._parse_unary()

        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        """Parse numbers, variables, functions, and parentheses."""
        token = self._current_token()

        if token.type == TokenType.NUMBER:
            self.pos += 1
            return Num(token.value)

        elif token.type == TokenType.VARIABLE:
            self.pos += 1
            return Var(token.value)

        elif token.type == TokenType.FUNCTION:
            func_name = token.value
            self.pos += 1
            self._eat(TokenType.LPAREN)
            arg = self._parse_expression()
            self._eat(TokenType.RPAREN)
            return FuncCall(func_name, arg)

        elif token.type == TokenType.LPAREN:
            self.pos += 1
            expr = self._parse_expression()
            self._eat(TokenType.RPAREN)
            return expr

        else:
            raise SyntaxError(
                f"Unexpected token {token.type.name} at position {token.position}"
            )


# =============================================================================
# Public API Functions
# =============================================================================

def parse(text: str) -> Expr:
    """Parse a mathematical expression string into an AST.

    Args:
        text: Mathematical expression string.

    Returns:
        Root of the expression tree.

    Raises:
        SyntaxError: For malformed expressions.

    Example:
        >>> parse('x^2 + 2*x + 1')
        BinOp('+', BinOp('+', BinOp('^', Var('x'), Num(2.0)), ...
    """
    if not text or not text.strip():
        raise SyntaxError("Empty expression")

    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


def to_string(expr: Union[str, Expr]) -> str:
    """Convert an expression to a human-readable string.

    Args:
        expr: Expression (string or Expr object).

    Returns:
        String representation.

    Example:
        >>> to_string(parse('x^2 + 1'))
        'x^2 + 1'
    """
    if isinstance(expr, str):
        expr = parse(expr)

    return _to_string_impl(expr)


def _to_string_impl(expr: Expr, parent_op: str = None, is_right: bool = False) -> str:
    """Internal implementation of to_string with precedence handling."""
    if isinstance(expr, Num):
        # Display integers without decimal point
        if expr.value == int(expr.value):
            return str(int(expr.value))
        return str(expr.value)

    elif isinstance(expr, Var):
        return expr.name

    elif isinstance(expr, UnaryOp):
        if expr.op == '-':
            operand_str = _to_string_impl(expr.operand, '-')
            if isinstance(expr.operand, (BinOp, UnaryOp)):
                return f"-({operand_str})"
            return f"-{operand_str}"
        return f"{expr.op}{_to_string_impl(expr.operand)}"

    elif isinstance(expr, BinOp):
        left_str = _to_string_impl(expr.left, expr.op, False)
        right_str = _to_string_impl(expr.right, expr.op, True)

        result = f"{left_str} {expr.op} {right_str}"

        # Add parentheses based on precedence
        if parent_op:
            needs_parens = _needs_parentheses(expr.op, parent_op, is_right)
            if needs_parens:
                result = f"({result})"

        return result

    elif isinstance(expr, FuncCall):
        arg_str = _to_string_impl(expr.arg)
        return f"{expr.name}({arg_str})"

    raise ValueError(f"Unknown expression type: {type(expr)}")


def _needs_parentheses(inner_op: str, outer_op: str, is_right: bool) -> bool:
    """Determine if parentheses are needed based on precedence."""
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    inner_prec = precedence.get(inner_op, 0)
    outer_prec = precedence.get(outer_op, 0)

    if inner_prec < outer_prec:
        return True

    # Handle right associativity and non-associativity
    if inner_prec == outer_prec:
        if outer_op in ('-', '/') and is_right:
            return True
        if outer_op == '^' and not is_right:
            return True

    return False


def latex(expr: Union[str, Expr]) -> str:
    """Generate LaTeX representation of an expression.

    Args:
        expr: Expression (string or Expr object).

    Returns:
        LaTeX formatted string.

    Example:
        >>> latex('x^2/2')
        '\\\\frac{x^{2}}{2}'
    """
    if isinstance(expr, str):
        expr = parse(expr)

    return _latex_impl(expr)


def _latex_impl(expr: Expr) -> str:
    """Internal implementation of latex conversion."""
    if isinstance(expr, Num):
        if expr.value == int(expr.value):
            return str(int(expr.value))
        return str(expr.value)

    elif isinstance(expr, Var):
        return expr.name

    elif isinstance(expr, UnaryOp):
        if expr.op == '-':
            operand_latex = _latex_impl(expr.operand)
            if isinstance(expr.operand, BinOp):
                return f"-\\left({operand_latex}\\right)"
            return f"-{operand_latex}"
        return f"{expr.op}{_latex_impl(expr.operand)}"

    elif isinstance(expr, BinOp):
        left_latex = _latex_impl(expr.left)
        right_latex = _latex_impl(expr.right)

        if expr.op == '/':
            return f"\\frac{{{left_latex}}}{{{right_latex}}}"
        elif expr.op == '^':
            # Wrap base in braces if it's complex
            if isinstance(expr.left, BinOp):
                left_latex = f"\\left({left_latex}\\right)"
            return f"{{{left_latex}}}^{{{right_latex}}}"
        elif expr.op == '*':
            return f"{left_latex} \\cdot {right_latex}"
        elif expr.op == '+':
            return f"{left_latex} + {right_latex}"
        elif expr.op == '-':
            return f"{left_latex} - {right_latex}"

        return f"{left_latex} {expr.op} {right_latex}"

    elif isinstance(expr, FuncCall):
        arg_latex = _latex_impl(expr.arg)

        if expr.name == 'sqrt':
            return f"\\sqrt{{{arg_latex}}}"
        elif expr.name == 'ln':
            return f"\\ln\\left({arg_latex}\\right)"
        elif expr.name in ('sin', 'cos', 'tan', 'log', 'exp'):
            return f"\\{expr.name}\\left({arg_latex}\\right)"

        return f"\\mathrm{{{expr.name}}}\\left({arg_latex}\\right)"

    raise ValueError(f"Unknown expression type: {type(expr)}")


def substitute(expr: Union[str, Expr], values: dict) -> Union[float, Expr]:
    """Substitute variables with values and evaluate if possible.

    Args:
        expr: Expression (string or Expr object).
        values: Dictionary mapping variable names to numeric values.

    Returns:
        Float if fully evaluated, Expr if variables remain.

    Raises:
        ValueError: For undefined functions or invalid operations.

    Example:
        >>> substitute('x^2 + y', {'x': 3, 'y': 1})
        10.0
    """
    if isinstance(expr, str):
        expr = parse(expr)

    result = _substitute_impl(expr, values)

    # If result is a Num, return just the value
    if isinstance(result, Num):
        return result.value

    return result


def _substitute_impl(expr: Expr, values: dict) -> Expr:
    """Internal implementation of substitute."""
    if isinstance(expr, Num):
        return expr

    elif isinstance(expr, Var):
        if expr.name in values:
            return Num(float(values[expr.name]))
        return expr

    elif isinstance(expr, UnaryOp):
        operand = _substitute_impl(expr.operand, values)
        if isinstance(operand, Num):
            if expr.op == '-':
                return Num(-operand.value)
        return UnaryOp(expr.op, operand)

    elif isinstance(expr, BinOp):
        left = _substitute_impl(expr.left, values)
        right = _substitute_impl(expr.right, values)

        if isinstance(left, Num) and isinstance(right, Num):
            return Num(_evaluate_binop(expr.op, left.value, right.value))

        return BinOp(expr.op, left, right)

    elif isinstance(expr, FuncCall):
        arg = _substitute_impl(expr.arg, values)

        if isinstance(arg, Num):
            return Num(_evaluate_function(expr.name, arg.value))

        return FuncCall(expr.name, arg)

    raise ValueError(f"Unknown expression type: {type(expr)}")


def _evaluate_binop(op: str, left: float, right: float) -> float:
    """Evaluate a binary operation."""
    if op == '+':
        return left + right
    elif op == '-':
        return left - right
    elif op == '*':
        return left * right
    elif op == '/':
        if right == 0:
            raise ValueError("Division by zero")
        return left / right
    elif op == '^':
        return left ** right

    raise ValueError(f"Unknown operator: {op}")


def _evaluate_function(name: str, arg: float) -> float:
    """Evaluate a function call."""
    if name == 'sin':
        return math.sin(arg)
    elif name == 'cos':
        return math.cos(arg)
    elif name == 'tan':
        return math.tan(arg)
    elif name in ('log', 'ln'):
        if arg <= 0:
            raise ValueError("Logarithm of non-positive number")
        return math.log(arg)
    elif name == 'exp':
        return math.exp(arg)
    elif name == 'sqrt':
        if arg < 0:
            raise ValueError("Square root of negative number")
        return math.sqrt(arg)

    raise ValueError(f"Unknown function: {name}")


def simplify(expr: Union[str, Expr]) -> Union[str, Expr]:
    """Simplify an expression using algebraic rules.

    Args:
        expr: Expression (string or Expr object).

    Returns:
        Simplified expression (string if input was string).

    Example:
        >>> simplify('x + x')
        '2 * x'
        >>> simplify('x * 1')
        'x'
    """
    return_string = isinstance(expr, str)
    if return_string:
        expr = parse(expr)

    # Apply simplification rules until fixed point
    prev = None
    current = expr
    max_iterations = 100
    i = 0

    while current != prev and i < max_iterations:
        prev = current
        current = _simplify_impl(current)
        i += 1

    if return_string:
        return to_string(current)
    return current


def _simplify_impl(expr: Expr) -> Expr:
    """Internal implementation of simplify."""
    if isinstance(expr, (Num, Var)):
        return expr

    elif isinstance(expr, UnaryOp):
        operand = _simplify_impl(expr.operand)

        # --x = x
        if expr.op == '-' and isinstance(operand, UnaryOp) and operand.op == '-':
            return operand.operand

        # -constant = computed value
        if expr.op == '-' and isinstance(operand, Num):
            return Num(-operand.value)

        return UnaryOp(expr.op, operand)

    elif isinstance(expr, BinOp):
        left = _simplify_impl(expr.left)
        right = _simplify_impl(expr.right)

        # Constant folding
        if isinstance(left, Num) and isinstance(right, Num):
            return Num(_evaluate_binop(expr.op, left.value, right.value))

        # Addition rules
        if expr.op == '+':
            # x + 0 = x
            if isinstance(right, Num) and right.value == 0:
                return left
            # 0 + x = x
            if isinstance(left, Num) and left.value == 0:
                return right
            # x + x = 2*x (like terms)
            if left == right:
                return BinOp('*', Num(2), left)
            # a*x + b*x = (a+b)*x
            if (isinstance(left, BinOp) and left.op == '*' and
                isinstance(right, BinOp) and right.op == '*'):
                if isinstance(left.left, Num) and isinstance(right.left, Num):
                    if left.right == right.right:
                        return BinOp('*', Num(left.left.value + right.left.value), left.right)
                if isinstance(left.right, Num) and isinstance(right.right, Num):
                    if left.left == right.left:
                        return BinOp('*', Num(left.right.value + right.right.value), left.left)
            # n*x + x = (n+1)*x
            if isinstance(left, BinOp) and left.op == '*' and isinstance(left.left, Num):
                if left.right == right:
                    return BinOp('*', Num(left.left.value + 1), right)
            # x + n*x = (n+1)*x
            if isinstance(right, BinOp) and right.op == '*' and isinstance(right.left, Num):
                if right.right == left:
                    return BinOp('*', Num(right.left.value + 1), left)
            # (a + x) + x = a + 2*x
            if isinstance(left, BinOp) and left.op == '+' and left.right == right:
                return BinOp('+', left.left, BinOp('*', Num(2), right))
            # x + (a + x) = a + 2*x
            if isinstance(right, BinOp) and right.op == '+' and right.right == left:
                return BinOp('+', right.left, BinOp('*', Num(2), left))

        # Subtraction rules
        elif expr.op == '-':
            # x - 0 = x
            if isinstance(right, Num) and right.value == 0:
                return left
            # 0 - x = -x
            if isinstance(left, Num) and left.value == 0:
                return UnaryOp('-', right)
            # x - x = 0
            if left == right:
                return Num(0)

        # Multiplication rules
        elif expr.op == '*':
            # x * 1 = x
            if isinstance(right, Num) and right.value == 1:
                return left
            # 1 * x = x
            if isinstance(left, Num) and left.value == 1:
                return right
            # x * 0 = 0
            if isinstance(right, Num) and right.value == 0:
                return Num(0)
            # 0 * x = 0
            if isinstance(left, Num) and left.value == 0:
                return Num(0)
            # x * x = x^2
            if left == right:
                return BinOp('^', left, Num(2))
            # n * (x / n) = x (simplify 2 * x^2 / 2 to x^2)
            if isinstance(left, Num) and isinstance(right, BinOp) and right.op == '/':
                if isinstance(right.right, Num) and left.value == right.right.value:
                    return right.left
            # (x / n) * n = x
            if isinstance(right, Num) and isinstance(left, BinOp) and left.op == '/':
                if isinstance(left.right, Num) and right.value == left.right.value:
                    return left.left

        # Division rules
        elif expr.op == '/':
            # x / 1 = x
            if isinstance(right, Num) and right.value == 1:
                return left
            # 0 / x = 0 (when x != 0)
            if isinstance(left, Num) and left.value == 0:
                return Num(0)
            # x / x = 1
            if left == right:
                return Num(1)
            # (n * x) / n = x
            if isinstance(left, BinOp) and left.op == '*' and isinstance(right, Num):
                if isinstance(left.left, Num) and left.left.value == right.value:
                    return left.right
                if isinstance(left.right, Num) and left.right.value == right.value:
                    return left.left

        # Power rules
        elif expr.op == '^':
            # x ^ 0 = 1
            if isinstance(right, Num) and right.value == 0:
                return Num(1)
            # x ^ 1 = x
            if isinstance(right, Num) and right.value == 1:
                return left
            # 0 ^ x = 0 (for x > 0)
            if isinstance(left, Num) and left.value == 0:
                if isinstance(right, Num) and right.value > 0:
                    return Num(0)
            # 1 ^ x = 1
            if isinstance(left, Num) and left.value == 1:
                return Num(1)

        return BinOp(expr.op, left, right)

    elif isinstance(expr, FuncCall):
        arg = _simplify_impl(expr.arg)
        return FuncCall(expr.name, arg)

    return expr


def differentiate(expr: Union[str, Expr], var: str) -> Union[str, Expr]:
    """Compute the symbolic derivative of an expression.

    Args:
        expr: Expression (string or Expr object).
        var: Variable to differentiate with respect to.

    Returns:
        Derivative expression (string if input was string).

    Example:
        >>> differentiate('x^3 + 2*x', 'x')
        '3 * x^2 + 2'
    """
    return_string = isinstance(expr, str)
    if return_string:
        expr = parse(expr)

    result = _differentiate_impl(expr, var)
    # Apply simplification - need to ensure result is Expr for simplify
    if isinstance(result, Expr):
        result = simplify(result)
        # simplify might return str if input was str, so convert back
        if isinstance(result, str):
            result = parse(result)

    if return_string:
        return to_string(result)
    return result


def _differentiate_impl(expr: Expr, var: str) -> Expr:
    """Internal implementation of differentiate."""
    if isinstance(expr, Num):
        # d/dx(c) = 0
        return Num(0)

    elif isinstance(expr, Var):
        # d/dx(x) = 1, d/dx(y) = 0
        if expr.name == var:
            return Num(1)
        return Num(0)

    elif isinstance(expr, UnaryOp):
        if expr.op == '-':
            # d/dx(-u) = -du/dx
            return UnaryOp('-', _differentiate_impl(expr.operand, var))
        return UnaryOp(expr.op, _differentiate_impl(expr.operand, var))

    elif isinstance(expr, BinOp):
        if expr.op == '+':
            # Sum rule: d/dx(u + v) = du/dx + dv/dx
            return BinOp('+',
                        _differentiate_impl(expr.left, var),
                        _differentiate_impl(expr.right, var))

        elif expr.op == '-':
            # Difference rule: d/dx(u - v) = du/dx - dv/dx
            return BinOp('-',
                        _differentiate_impl(expr.left, var),
                        _differentiate_impl(expr.right, var))

        elif expr.op == '*':
            # Product rule: d/dx(u * v) = u * dv/dx + v * du/dx
            du = _differentiate_impl(expr.left, var)
            dv = _differentiate_impl(expr.right, var)
            return BinOp('+',
                        BinOp('*', expr.left, dv),
                        BinOp('*', expr.right, du))

        elif expr.op == '/':
            # Quotient rule: d/dx(u / v) = (v * du/dx - u * dv/dx) / v^2
            du = _differentiate_impl(expr.left, var)
            dv = _differentiate_impl(expr.right, var)
            numerator = BinOp('-',
                            BinOp('*', expr.right, du),
                            BinOp('*', expr.left, dv))
            denominator = BinOp('^', expr.right, Num(2))
            return BinOp('/', numerator, denominator)

        elif expr.op == '^':
            # Check if exponent is constant
            if _is_constant(expr.right, var):
                # Power rule: d/dx(u^n) = n * u^(n-1) * du/dx
                du = _differentiate_impl(expr.left, var)
                return BinOp('*',
                           BinOp('*', expr.right,
                                 BinOp('^', expr.left,
                                       BinOp('-', expr.right, Num(1)))),
                           du)
            elif _is_constant(expr.left, var):
                # d/dx(a^u) = a^u * ln(a) * du/dx
                du = _differentiate_impl(expr.right, var)
                return BinOp('*',
                           BinOp('*', expr, FuncCall('ln', expr.left)),
                           du)
            else:
                # General case: d/dx(u^v) = u^v * (v'*ln(u) + v*u'/u)
                du = _differentiate_impl(expr.left, var)
                dv = _differentiate_impl(expr.right, var)
                term1 = BinOp('*', dv, FuncCall('ln', expr.left))
                term2 = BinOp('/', BinOp('*', expr.right, du), expr.left)
                return BinOp('*', expr, BinOp('+', term1, term2))

    elif isinstance(expr, FuncCall):
        # Chain rule for functions
        arg = expr.arg
        darg = _differentiate_impl(arg, var)

        if expr.name == 'sin':
            # d/dx(sin(u)) = cos(u) * du/dx
            return BinOp('*', FuncCall('cos', arg), darg)

        elif expr.name == 'cos':
            # d/dx(cos(u)) = -sin(u) * du/dx
            return BinOp('*', UnaryOp('-', FuncCall('sin', arg)), darg)

        elif expr.name == 'tan':
            # d/dx(tan(u)) = (1/cos^2(u)) * du/dx
            return BinOp('*',
                        BinOp('/', Num(1), BinOp('^', FuncCall('cos', arg), Num(2))),
                        darg)

        elif expr.name in ('log', 'ln'):
            # d/dx(ln(u)) = (1/u) * du/dx
            return BinOp('*', BinOp('/', Num(1), arg), darg)

        elif expr.name == 'exp':
            # d/dx(exp(u)) = exp(u) * du/dx
            return BinOp('*', expr, darg)

        elif expr.name == 'sqrt':
            # d/dx(sqrt(u)) = (1/(2*sqrt(u))) * du/dx
            return BinOp('*',
                        BinOp('/', Num(1), BinOp('*', Num(2), FuncCall('sqrt', arg))),
                        darg)

    raise ValueError(f"Cannot differentiate: {type(expr)}")


def _is_constant(expr: Expr, var: str) -> bool:
    """Check if expression is constant with respect to variable."""
    if isinstance(expr, Num):
        return True
    elif isinstance(expr, Var):
        return expr.name != var
    elif isinstance(expr, UnaryOp):
        return _is_constant(expr.operand, var)
    elif isinstance(expr, BinOp):
        return _is_constant(expr.left, var) and _is_constant(expr.right, var)
    elif isinstance(expr, FuncCall):
        return _is_constant(expr.arg, var)
    return False


def integrate(expr: Union[str, Expr], var: str) -> Union[str, Expr]:
    """Compute the symbolic integral of a polynomial expression.

    Args:
        expr: Expression (string or Expr object).
        var: Variable to integrate with respect to.

    Returns:
        Integral expression with '+ C' (string if input was string).

    Raises:
        ValueError: For non-integrable expressions.

    Example:
        >>> integrate('2*x', 'x')
        'x^2 + C'
    """
    return_string = isinstance(expr, str)
    if return_string:
        expr = parse(expr)

    result = _integrate_impl(expr, var)
    result = simplify(result)

    # Add constant of integration
    result = BinOp('+', result, Var('C'))

    if return_string:
        return to_string(result)
    return result


def _integrate_impl(expr: Expr, var: str) -> Expr:
    """Internal implementation of integrate."""
    if isinstance(expr, Num):
        # Integral of c = c*x
        return BinOp('*', expr, Var(var))

    elif isinstance(expr, Var):
        if expr.name == var:
            # Integral of x = x^2/2
            return BinOp('/', BinOp('^', Var(var), Num(2)), Num(2))
        else:
            # Integral of y (with respect to x) = y*x
            return BinOp('*', expr, Var(var))

    elif isinstance(expr, UnaryOp):
        if expr.op == '-':
            return UnaryOp('-', _integrate_impl(expr.operand, var))
        raise ValueError(f"Cannot integrate unary operator: {expr.op}")

    elif isinstance(expr, BinOp):
        if expr.op == '+':
            # Sum rule
            return BinOp('+',
                        _integrate_impl(expr.left, var),
                        _integrate_impl(expr.right, var))

        elif expr.op == '-':
            # Difference rule
            return BinOp('-',
                        _integrate_impl(expr.left, var),
                        _integrate_impl(expr.right, var))

        elif expr.op == '*':
            # Constant multiple rule: integral(c*f) = c * integral(f)
            if _is_constant(expr.left, var):
                return BinOp('*', expr.left, _integrate_impl(expr.right, var))
            elif _is_constant(expr.right, var):
                return BinOp('*', expr.right, _integrate_impl(expr.left, var))
            else:
                raise ValueError("Cannot integrate non-polynomial product")

        elif expr.op == '/':
            # Check for 1/x case
            if isinstance(expr.left, Num) and expr.left.value == 1:
                if isinstance(expr.right, Var) and expr.right.name == var:
                    return FuncCall('ln', Var(var))
            if _is_constant(expr.right, var):
                return BinOp('/', _integrate_impl(expr.left, var), expr.right)
            raise ValueError("Cannot integrate non-polynomial division")

        elif expr.op == '^':
            # Power rule: integral(x^n) = x^(n+1)/(n+1)
            if isinstance(expr.left, Var) and expr.left.name == var:
                if isinstance(expr.right, Num):
                    n = expr.right.value
                    if n == -1:
                        return FuncCall('ln', Var(var))
                    return BinOp('/',
                               BinOp('^', Var(var), Num(n + 1)),
                               Num(n + 1))
            raise ValueError("Cannot integrate non-polynomial power")

    elif isinstance(expr, FuncCall):
        # Only integrate if argument is just the variable
        if isinstance(expr.arg, Var) and expr.arg.name == var:
            if expr.name == 'sin':
                return UnaryOp('-', FuncCall('cos', Var(var)))
            elif expr.name == 'cos':
                return FuncCall('sin', Var(var))
            elif expr.name == 'exp':
                return FuncCall('exp', Var(var))
        raise ValueError(f"Cannot integrate function: {expr.name}")

    raise ValueError(f"Cannot integrate: {type(expr)}")


def expand(expr: Union[str, Expr]) -> Union[str, Expr]:
    """Expand products and powers of sums.

    Args:
        expr: Expression (string or Expr object).

    Returns:
        Expanded expression (string if input was string).

    Raises:
        ValueError: For non-integer exponents in expansion.

    Example:
        >>> expand('(x + 1)^2')
        'x^2 + 2 * x + 1'
    """
    return_string = isinstance(expr, str)
    if return_string:
        expr = parse(expr)

    result = _expand_impl(expr)
    # Collect like terms after expansion
    result = _collect_like_terms(result)
    result = simplify(result)

    if return_string:
        return to_string(result)
    return result


def _collect_like_terms(expr: Expr) -> Expr:
    """Collect like terms in an expression."""
    # Flatten the expression into terms
    terms = _flatten_sum(expr)

    # Group terms by their variable structure (ignoring coefficient)
    term_groups = {}  # Maps term structure to coefficient

    for coeff, term_expr in terms:
        # Create a key for this term structure
        key = _term_key(term_expr)
        if key in term_groups:
            old_coeff, old_expr = term_groups[key]
            term_groups[key] = (old_coeff + coeff, old_expr)
        else:
            term_groups[key] = (coeff, term_expr)

    # Rebuild expression from collected terms
    result = None
    for key in sorted(term_groups.keys(), reverse=True):
        coeff, term_expr = term_groups[key]

        if coeff == 0:
            continue

        if term_expr is None:
            # Constant term
            term = Num(coeff)
        elif coeff == 1:
            term = term_expr
        elif coeff == -1:
            term = UnaryOp('-', term_expr)
        else:
            term = BinOp('*', Num(coeff), term_expr)

        if result is None:
            result = term
        else:
            result = BinOp('+', result, term)

    return result if result else Num(0)


def _flatten_sum(expr: Expr, sign: float = 1.0) -> list:
    """Flatten an expression into a list of (coefficient, term) pairs."""
    if isinstance(expr, Num):
        return [(sign * expr.value, None)]

    elif isinstance(expr, Var):
        return [(sign, expr)]

    elif isinstance(expr, UnaryOp) and expr.op == '-':
        return _flatten_sum(expr.operand, -sign)

    elif isinstance(expr, BinOp):
        if expr.op == '+':
            return _flatten_sum(expr.left, sign) + _flatten_sum(expr.right, sign)
        elif expr.op == '-':
            return _flatten_sum(expr.left, sign) + _flatten_sum(expr.right, -sign)
        elif expr.op == '*':
            # Check for coefficient * term
            if isinstance(expr.left, Num):
                coeff = expr.left.value * sign
                return [(coeff, expr.right)]
            elif isinstance(expr.right, Num):
                coeff = expr.right.value * sign
                return [(coeff, expr.left)]
            else:
                return [(sign, expr)]
        else:
            return [(sign, expr)]

    elif isinstance(expr, FuncCall):
        return [(sign, expr)]

    return [(sign, expr)]


def _term_key(term_expr) -> tuple:
    """Create a hashable key for a term structure."""
    if term_expr is None:
        return (0, '')  # Constant term
    elif isinstance(term_expr, Var):
        return (1, term_expr.name)
    elif isinstance(term_expr, BinOp) and term_expr.op == '^':
        if isinstance(term_expr.left, Var) and isinstance(term_expr.right, Num):
            return (term_expr.right.value, term_expr.left.name)
    return (999, str(term_expr))  # Complex term - high priority to keep separate


def _expand_impl(expr: Expr) -> Expr:
    """Internal implementation of expand."""
    if isinstance(expr, (Num, Var)):
        return expr

    elif isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _expand_impl(expr.operand))

    elif isinstance(expr, BinOp):
        left = _expand_impl(expr.left)
        right = _expand_impl(expr.right)

        if expr.op == '*':
            # Distribute multiplication
            return _distribute(left, right)

        elif expr.op == '^':
            # Expand powers of sums
            if isinstance(right, Num):
                n = int(right.value)
                if n != right.value:
                    raise ValueError("Non-integer exponent in expansion")
                if n < 0:
                    return BinOp('^', left, right)
                if n == 0:
                    return Num(1)
                if n == 1:
                    return left

                # Expand (a + b)^n using repeated multiplication
                result = left
                for _ in range(n - 1):
                    result = _distribute(result, left)
                return result

            return BinOp('^', left, right)

        elif expr.op == '+':
            return BinOp('+', left, right)

        elif expr.op == '-':
            return BinOp('-', left, right)

        elif expr.op == '/':
            return BinOp('/', left, right)

    elif isinstance(expr, FuncCall):
        return FuncCall(expr.name, _expand_impl(expr.arg))

    return expr


def _distribute(left: Expr, right: Expr) -> Expr:
    """Distribute multiplication over addition/subtraction."""
    # (a + b) * c = a*c + b*c
    if isinstance(left, BinOp) and left.op == '+':
        return BinOp('+',
                    _distribute(left.left, right),
                    _distribute(left.right, right))

    if isinstance(left, BinOp) and left.op == '-':
        return BinOp('-',
                    _distribute(left.left, right),
                    _distribute(left.right, right))

    # a * (b + c) = a*b + a*c
    if isinstance(right, BinOp) and right.op == '+':
        return BinOp('+',
                    _distribute(left, right.left),
                    _distribute(left, right.right))

    if isinstance(right, BinOp) and right.op == '-':
        return BinOp('-',
                    _distribute(left, right.left),
                    _distribute(left, right.right))

    return BinOp('*', left, right)


def solve(equation: str, var: str) -> list:
    """Solve an equation for a variable.

    Supports linear and quadratic equations.

    Args:
        equation: Equation string containing '='.
        var: Variable to solve for.

    Returns:
        List of solutions sorted ascending.

    Raises:
        ValueError: For unsolvable or unsupported equations.

    Example:
        >>> solve('x^2 - 4 = 0', 'x')
        [-2.0, 2.0]
    """
    if '=' not in equation:
        raise SyntaxError("Equation must contain '='")

    parts = equation.split('=')
    if len(parts) != 2:
        raise SyntaxError("Equation must contain exactly one '='")

    left_str, right_str = parts[0].strip(), parts[1].strip()

    # Parse both sides
    left = parse(left_str) if left_str else Num(0)
    right = parse(right_str) if right_str else Num(0)

    # Rearrange to: left - right = 0
    expr = simplify(BinOp('-', left, right))

    # Extract polynomial coefficients
    coeffs = _extract_polynomial_coefficients(expr, var)

    if coeffs is None:
        raise ValueError("Cannot solve this type of equation")

    degree = max(coeffs.keys()) if coeffs else 0

    if degree == 0:
        # Constant equation
        c = coeffs.get(0, 0)
        if abs(c) < 1e-10:
            raise ValueError("Infinite solutions (identity)")
        raise ValueError("No solutions (contradiction)")

    elif degree == 1:
        # Linear: ax + b = 0 => x = -b/a
        a = coeffs.get(1, 0)
        b = coeffs.get(0, 0)

        if a == 0:
            raise ValueError("Not a valid linear equation")

        return [-b / a]

    elif degree == 2:
        # Quadratic: ax^2 + bx + c = 0
        a = coeffs.get(2, 0)
        b = coeffs.get(1, 0)
        c = coeffs.get(0, 0)

        if a == 0:
            # Degenerate to linear
            if b == 0:
                raise ValueError("Not a valid equation")
            return [-c / b]

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            raise ValueError("No real solutions")
        elif discriminant == 0:
            return [-b / (2 * a)]
        else:
            sqrt_disc = math.sqrt(discriminant)
            x1 = (-b + sqrt_disc) / (2 * a)
            x2 = (-b - sqrt_disc) / (2 * a)
            return sorted([x1, x2])

    raise ValueError(f"Cannot solve polynomial of degree {degree}")


def _extract_polynomial_coefficients(expr: Expr, var: str) -> dict:
    """Extract polynomial coefficients from expression.

    Returns dict mapping degree to coefficient, or None if not polynomial.
    """
    expr = _expand_impl(expr)
    expr = simplify(expr)

    coeffs = {}

    def add_term(degree: int, coeff: float):
        coeffs[degree] = coeffs.get(degree, 0) + coeff

    def process(node: Expr, sign: float = 1.0) -> bool:
        if isinstance(node, Num):
            add_term(0, sign * node.value)
            return True

        elif isinstance(node, Var):
            if node.name == var:
                add_term(1, sign)
            else:
                # Other variable treated as constant
                return False
            return True

        elif isinstance(node, UnaryOp) and node.op == '-':
            return process(node.operand, -sign)

        elif isinstance(node, BinOp):
            if node.op == '+':
                return process(node.left, sign) and process(node.right, sign)

            elif node.op == '-':
                return process(node.left, sign) and process(node.right, -sign)

            elif node.op == '*':
                # Check for coefficient * variable^n
                coeff = _get_constant_value(node.left, var)
                if coeff is not None:
                    # coeff * something
                    if isinstance(node.right, Var) and node.right.name == var:
                        add_term(1, sign * coeff)
                        return True
                    elif isinstance(node.right, BinOp) and node.right.op == '^':
                        if (isinstance(node.right.left, Var) and
                            node.right.left.name == var and
                            isinstance(node.right.right, Num)):
                            degree = int(node.right.right.value)
                            add_term(degree, sign * coeff)
                            return True

                coeff = _get_constant_value(node.right, var)
                if coeff is not None:
                    # something * coeff
                    if isinstance(node.left, Var) and node.left.name == var:
                        add_term(1, sign * coeff)
                        return True
                    elif isinstance(node.left, BinOp) and node.left.op == '^':
                        if (isinstance(node.left.left, Var) and
                            node.left.left.name == var and
                            isinstance(node.left.right, Num)):
                            degree = int(node.left.right.value)
                            add_term(degree, sign * coeff)
                            return True

                return False

            elif node.op == '^':
                if (isinstance(node.left, Var) and node.left.name == var and
                    isinstance(node.right, Num)):
                    degree = int(node.right.value)
                    add_term(degree, sign)
                    return True
                return False

            else:
                return False

        return False

    if process(expr):
        return coeffs
    return None


def _get_constant_value(expr: Expr, var: str) -> float:
    """Get the constant value of an expression, or None if not constant."""
    if isinstance(expr, Num):
        return expr.value
    elif isinstance(expr, Var):
        if expr.name != var:
            return None  # Other variable - can't evaluate as constant number
        return None
    elif isinstance(expr, UnaryOp) and expr.op == '-':
        val = _get_constant_value(expr.operand, var)
        return -val if val is not None else None
    elif isinstance(expr, BinOp):
        left_val = _get_constant_value(expr.left, var)
        right_val = _get_constant_value(expr.right, var)
        if left_val is not None and right_val is not None:
            return _evaluate_binop(expr.op, left_val, right_val)
    return None


# =============================================================================
# Main Demonstration
# =============================================================================

if __name__ == "__main__":
    print("Symbolic Math Engine Demonstration")
    print("=" * 50)

    # Parsing demonstration
    print("\n1. Parsing:")
    expr = parse('x^2 + 2*x + 1')
    print(f"   parse('x^2 + 2*x + 1') -> {to_string(expr)}")

    # LaTeX output
    print("\n2. LaTeX formatting:")
    print(f"   latex('x^2/2') -> {latex('x^2/2')}")
    print(f"   latex('sqrt(x)') -> {latex('sqrt(x)')}")

    # Substitution
    print("\n3. Substitution:")
    result = substitute('x^2 + y', {'x': 3, 'y': 1})
    print(f"   substitute('x^2 + y', {{'x': 3, 'y': 1}}) -> {result}")

    # Simplification
    print("\n4. Simplification:")
    print(f"   simplify('x + x') -> {simplify('x + x')}")
    print(f"   simplify('x * 1') -> {simplify('x * 1')}")
    print(f"   simplify('x + 0') -> {simplify('x + 0')}")
    print(f"   simplify('x * 0') -> {simplify('x * 0')}")
    print(f"   simplify('x ^ 1') -> {simplify('x ^ 1')}")

    # Differentiation
    print("\n5. Differentiation:")
    print(f"   differentiate('x^3 + 2*x', 'x') -> {differentiate('x^3 + 2*x', 'x')}")
    print(f"   differentiate('sin(x)', 'x') -> {differentiate('sin(x)', 'x')}")

    # Integration
    print("\n6. Integration:")
    print(f"   integrate('2*x', 'x') -> {integrate('2*x', 'x')}")
    print(f"   integrate('x^2', 'x') -> {integrate('x^2', 'x')}")

    # Expansion
    print("\n7. Expansion:")
    print(f"   expand('(x + 1)^2') -> {expand('(x + 1)^2')}")

    # Equation solving
    print("\n8. Equation solving:")
    print(f"   solve('x^2 - 4 = 0', 'x') -> {solve('x^2 - 4 = 0', 'x')}")
    print(f"   solve('x + 2 = 0', 'x') -> {solve('x + 2 = 0', 'x')}")
    print(f"   solve('2*x^2 - 5*x + 2 = 0', 'x') -> {solve('2*x^2 - 5*x + 2 = 0', 'x')}")

    # Example from issue specification
    print("\n9. Issue specification example:")
    expr = parse('(x + 1)^2')
    print(f"   parse('(x + 1)^2') -> {to_string(expr)}")
    expanded = expand(expr)
    print(f"   expand((x + 1)^2) -> {to_string(expanded) if isinstance(expanded, Expr) else expanded}")
    deriv = differentiate(expr, 'x')
    print(f"   differentiate((x + 1)^2, 'x') -> {to_string(deriv) if isinstance(deriv, Expr) else deriv}")
    result = substitute(expr, {'x': 2})
    print(f"   substitute((x + 1)^2, {{'x': 2}}) -> {result}")
