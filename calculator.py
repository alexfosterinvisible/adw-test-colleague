"""Simple calculator module for testing ADW."""

def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b

if __name__ == "__main__":
    print(f"2 + 3 = {add(2, 3)}")
