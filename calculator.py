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

def sqrt(n: float) -> float:
    """Calculate square root using Newton's method."""
    # Validate input
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")

    # Handle edge case
    if n == 0:
        return 0.0

    # Newton's method: iteratively refine guess using formula x_new = (x_old + n/x_old) / 2
    guess = n / 2 if n > 1 else 1.0
    tolerance = 0.000001

    while abs(guess * guess - n) >= tolerance:
        guess = (guess + n / guess) / 2

    return guess

if __name__ == "__main__":
    print(f"2 + 3 = {add(2, 3)}")
    print(f"10 / 2 = {divide(10, 2)}")
    print(f"7 / 3 = {divide(7, 3)}")
    print(f"sqrt(4) = {sqrt(4):.4f}")
    print(f"sqrt(2) = {sqrt(2):.4f}")
    print(f"sqrt(9) = {sqrt(9):.4f}")
