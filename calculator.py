"""Simple calculator module for testing ADW."""

def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b


def sqrt(n: int) -> float:
    """Calculate square root using Newton's method (Babylonian method).

    Args:
        n: Non-negative integer to calculate square root of.

    Returns:
        Float approximation of the square root.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")

    if n == 0:
        return 0.0

    guess = n / 2
    while True:
        new_guess = (guess + n / guess) / 2
        if abs(new_guess - guess) < 0.00001:
            return new_guess
        guess = new_guess


if __name__ == "__main__":
    print(f"2 + 3 = {add(2, 3)}")
    print(f"10 / 2 = {divide(10, 2)}")
    print(f"7 / 3 = {divide(7, 3)}")
    print(f"sqrt(16) = {sqrt(16)}")
    print(f"sqrt(10) = {sqrt(10):.3f}")
    print(f"sqrt(0) = {sqrt(0)}")
