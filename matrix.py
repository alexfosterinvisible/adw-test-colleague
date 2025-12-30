"""(Claude) Pure Python matrix operations module for linear algebra.

Provides a Matrix class and functions for matrix operations including:
- Basic operations: add, subtract, multiply, scalar_multiply
- Advanced operations: transpose, determinant, inverse, trace
- Decompositions: lu_decompose (LU with partial pivoting)
- Solvers: solve (Ax=b using LU decomposition)
- Eigenvalues: eigenvalues using QR algorithm
"""

from typing import Union
import copy
import math

# Tolerance for floating point comparisons
TOLERANCE = 1e-10


class Matrix:
    """Matrix class supporting standard linear algebra operations.

    Attributes:
        _data: Internal storage as list of lists of floats
        rows: Number of rows
        cols: Number of columns
    """

    def __init__(self, data: list[list[Union[int, float]]]) -> None:
        """Initialize matrix from nested list.

        Args:
            data: Nested list of numbers representing matrix rows

        Raises:
            ValueError: If rows have inconsistent lengths or data is empty
        """
        if not data:
            raise ValueError("Matrix cannot be empty")
        if not data[0]:
            raise ValueError("Matrix rows cannot be empty")

        row_len = len(data[0])
        for i, row in enumerate(data):
            if len(row) != row_len:
                raise ValueError(f"Inconsistent row length at row {i}: expected {row_len}, got {len(row)}")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Non-numeric value at position ({i}, {j})")

        # Deep copy and convert to floats
        self._data: list[list[float]] = [[float(val) for val in row] for row in data]
        self._rows = len(data)
        self._cols = row_len

    @property
    def rows(self) -> int:
        """Number of rows in the matrix."""
        return self._rows

    @property
    def cols(self) -> int:
        """Number of columns in the matrix."""
        return self._cols

    @property
    def data(self) -> list[list[float]]:
        """Return a deep copy of the matrix data."""
        return copy.deepcopy(self._data)

    def __getitem__(self, index: int) -> list[float]:
        """Get row by index for m[i][j] access pattern.

        Args:
            index: Row index

        Returns:
            Copy of the row as a list
        """
        return self._data[index][:]

    def __eq__(self, other: object) -> bool:
        """Check matrix equality with float tolerance.

        Args:
            other: Another matrix to compare

        Returns:
            True if matrices are equal within tolerance
        """
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self._data[i][j] - other._data[i][j]) > TOLERANCE:
                    return False
        return True

    def __str__(self) -> str:
        """Pretty-print matrix with aligned columns."""
        if not self._data:
            return "[]"

        # Format each number
        formatted = []
        for row in self._data:
            formatted_row = []
            for val in row:
                if abs(val - round(val)) < TOLERANCE:
                    formatted_row.append(str(int(round(val))))
                else:
                    formatted_row.append(f"{val:.6g}")
            formatted.append(formatted_row)

        # Find max width for each column
        col_widths = []
        for j in range(self.cols):
            max_width = max(len(formatted[i][j]) for i in range(self.rows))
            col_widths.append(max_width)

        # Build output with aligned columns
        lines = []
        for row in formatted:
            aligned = [val.rjust(col_widths[j]) for j, val in enumerate(row)]
            lines.append("[" + ", ".join(aligned) + "]")

        return "[" + ",\n ".join(lines) + "]"

    def __repr__(self) -> str:
        """Debug representation of matrix."""
        return f"Matrix({self._data})"

    @staticmethod
    def identity(n: int) -> 'Matrix':
        """Create an n x n identity matrix.

        Args:
            n: Size of the identity matrix

        Returns:
            Identity matrix of size n x n
        """
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Matrix(data)

    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create a matrix of zeros.

        Args:
            rows: Number of rows
            cols: Number of columns

        Returns:
            Zero matrix of given dimensions
        """
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)


def add(A: Matrix, B: Matrix) -> Matrix:
    """Add two matrices element-wise.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Sum of A and B

    Raises:
        ValueError: If dimensions don't match
    """
    if A.rows != B.rows or A.cols != B.cols:
        raise ValueError(f"Dimension mismatch: ({A.rows}x{A.cols}) + ({B.rows}x{B.cols})")

    result = [[A[i][j] + B[i][j] for j in range(A.cols)] for i in range(A.rows)]
    return Matrix(result)


def subtract(A: Matrix, B: Matrix) -> Matrix:
    """Subtract matrix B from matrix A element-wise.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Difference A - B

    Raises:
        ValueError: If dimensions don't match
    """
    if A.rows != B.rows or A.cols != B.cols:
        raise ValueError(f"Dimension mismatch: ({A.rows}x{A.cols}) - ({B.rows}x{B.cols})")

    result = [[A[i][j] - B[i][j] for j in range(A.cols)] for i in range(A.rows)]
    return Matrix(result)


def scalar_multiply(A: Matrix, k: float) -> Matrix:
    """Multiply matrix by a scalar.

    Args:
        A: Matrix to scale
        k: Scalar multiplier

    Returns:
        Scaled matrix k * A
    """
    result = [[A[i][j] * k for j in range(A.cols)] for i in range(A.rows)]
    return Matrix(result)


def multiply(A: Matrix, B: Matrix) -> Matrix:
    """Multiply two matrices.

    Args:
        A: Left matrix (m x n)
        B: Right matrix (n x p)

    Returns:
        Product matrix A * B (m x p)

    Raises:
        ValueError: If inner dimensions don't match
    """
    if A.cols != B.rows:
        raise ValueError(f"Dimension mismatch: ({A.rows}x{A.cols}) * ({B.rows}x{B.cols})")

    result = []
    for i in range(A.rows):
        row = []
        for j in range(B.cols):
            val = sum(A[i][k] * B[k][j] for k in range(A.cols))
            row.append(val)
        result.append(row)
    return Matrix(result)


def transpose(A: Matrix) -> Matrix:
    """Transpose a matrix.

    Args:
        A: Matrix to transpose

    Returns:
        Transposed matrix A^T
    """
    result = [[A[j][i] for j in range(A.rows)] for i in range(A.cols)]
    return Matrix(result)


def trace(A: Matrix) -> float:
    """Calculate the trace (sum of diagonal elements).

    Args:
        A: Square matrix

    Returns:
        Sum of diagonal elements

    Raises:
        ValueError: If matrix is not square
    """
    if A.rows != A.cols:
        raise ValueError(f"Trace requires square matrix, got {A.rows}x{A.cols}")

    return sum(A[i][i] for i in range(A.rows))


def determinant(A: Matrix) -> float:
    """Calculate the determinant of a square matrix.

    Uses cofactor expansion for small matrices, LU decomposition for larger.

    Args:
        A: Square matrix

    Returns:
        Determinant value

    Raises:
        ValueError: If matrix is not square
    """
    if A.rows != A.cols:
        raise ValueError(f"Determinant requires square matrix, got {A.rows}x{A.cols}")

    n = A.rows

    # Base cases
    if n == 1:
        return A[0][0]
    elif n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    elif n == 3:
        # Direct formula for 3x3
        return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

    # For larger matrices, use LU decomposition
    # det(A) = det(P)^(-1) * det(L) * det(U) = (-1)^swaps * product of U diagonal
    data = copy.deepcopy(A._data)
    swaps = 0

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(data[col][col])
        for row in range(col + 1, n):
            if abs(data[row][col]) > max_val:
                max_val = abs(data[row][col])
                max_row = row

        if max_val < TOLERANCE:
            return 0.0  # Singular matrix

        if max_row != col:
            data[col], data[max_row] = data[max_row], data[col]
            swaps += 1

        # Eliminate below
        for row in range(col + 1, n):
            factor = data[row][col] / data[col][col]
            for k in range(col, n):
                data[row][k] -= factor * data[col][k]

    # Product of diagonal
    det = 1.0
    for i in range(n):
        det *= data[i][i]

    return det * ((-1) ** swaps)


def lu_decompose(A: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """LU decomposition with partial pivoting.

    Decomposes A into PA = LU where:
    - P is a permutation matrix
    - L is lower triangular with 1s on diagonal
    - U is upper triangular

    Args:
        A: Square matrix to decompose

    Returns:
        Tuple (L, U, P)

    Raises:
        ValueError: If matrix is not square or is singular
    """
    if A.rows != A.cols:
        raise ValueError(f"LU decomposition requires square matrix, got {A.rows}x{A.cols}")

    n = A.rows

    # Initialize
    L = [[0.0] * n for _ in range(n)]
    U = copy.deepcopy(A._data)
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Partial pivoting - find max in column
        max_row = col
        max_val = abs(U[col][col])
        for row in range(col + 1, n):
            if abs(U[row][col]) > max_val:
                max_val = abs(U[row][col])
                max_row = row

        if max_val < TOLERANCE:
            raise ValueError("Singular matrix")

        # Swap rows in U, P, and L
        if max_row != col:
            U[col], U[max_row] = U[max_row], U[col]
            P[col], P[max_row] = P[max_row], P[col]
            # Swap L rows up to current column
            for k in range(col):
                L[col][k], L[max_row][k] = L[max_row][k], L[col][k]

        L[col][col] = 1.0

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for k in range(col, n):
                U[row][k] -= factor * U[col][k]

    return Matrix(L), Matrix(U), Matrix(P)


def inverse(A: Matrix) -> Matrix:
    """Calculate the inverse of a matrix using Gauss-Jordan elimination.

    Args:
        A: Square matrix to invert

    Returns:
        Inverse matrix A^(-1)

    Raises:
        ValueError: If matrix is not square or is singular
    """
    if A.rows != A.cols:
        raise ValueError(f"Inverse requires square matrix, got {A.rows}x{A.cols}")

    n = A.rows

    # Create augmented matrix [A | I]
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < TOLERANCE:
            raise ValueError("Singular matrix")

        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract inverse from right half
    result = [row[n:] for row in aug]
    return Matrix(result)


def solve(A: Matrix, b: Matrix) -> Matrix:
    """Solve linear system Ax = b using LU decomposition.

    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side column vector (n x 1)

    Returns:
        Solution vector x (n x 1)

    Raises:
        ValueError: If dimensions are incompatible or system is singular
    """
    if A.rows != A.cols:
        raise ValueError(f"Coefficient matrix must be square, got {A.rows}x{A.cols}")
    if b.rows != A.rows:
        raise ValueError(f"Dimension mismatch: A is {A.rows}x{A.cols}, b has {b.rows} rows")
    if b.cols != 1:
        raise ValueError(f"b must be a column vector, got {b.rows}x{b.cols}")

    n = A.rows

    # LU decomposition: PA = LU
    L, U, P = lu_decompose(A)

    # Solve Pb = Ly (forward substitution)
    Pb = multiply(P, b)
    y = [0.0] * n
    for i in range(n):
        y[i] = Pb[i][0] - sum(L[i][j] * y[j] for j in range(i))

    # Solve Ux = y (backward substitution)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return Matrix([[xi] for xi in x])


def eigenvalues(A: Matrix) -> list[float]:
    """Compute eigenvalues of a square matrix.

    Uses analytical solution for 2x2, QR algorithm for larger matrices.
    Only returns real eigenvalues.

    Args:
        A: Square matrix

    Returns:
        List of real eigenvalues

    Raises:
        ValueError: If matrix is not square
    """
    if A.rows != A.cols:
        raise ValueError(f"Eigenvalues require square matrix, got {A.rows}x{A.cols}")

    n = A.rows

    # 1x1 case
    if n == 1:
        return [A[0][0]]

    # 2x2 case: use quadratic formula on characteristic polynomial
    # det(A - λI) = λ² - (a+d)λ + (ad-bc) = 0
    if n == 2:
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        tr = a + d  # trace
        det = a * d - b * c  # determinant

        discriminant = tr * tr - 4 * det
        if discriminant < -TOLERANCE:
            # Complex eigenvalues - return empty for real-only implementation
            return []
        elif abs(discriminant) < TOLERANCE:
            return [tr / 2]
        else:
            sqrt_disc = math.sqrt(discriminant)
            return [(tr + sqrt_disc) / 2, (tr - sqrt_disc) / 2]

    # QR algorithm for larger matrices
    # Use Hessenberg form and shifted QR for faster convergence
    H = _to_hessenberg(A)
    return _qr_algorithm(H)


def _to_hessenberg(A: Matrix) -> Matrix:
    """Convert matrix to upper Hessenberg form using Householder reflections."""
    n = A.rows
    H = copy.deepcopy(A._data)

    for k in range(n - 2):
        # Extract column below diagonal
        x = [H[i][k] for i in range(k + 1, n)]

        # Compute Householder vector
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < TOLERANCE:
            continue

        sign = 1 if x[0] >= 0 else -1
        x[0] += sign * norm_x
        norm_v = math.sqrt(sum(xi * xi for xi in x))
        if norm_v < TOLERANCE:
            continue
        v = [xi / norm_v for xi in x]

        # Apply H = I - 2vv^T to columns
        for j in range(k, n):
            dot = sum(v[i - k - 1] * H[i][j] for i in range(k + 1, n))
            for i in range(k + 1, n):
                H[i][j] -= 2 * v[i - k - 1] * dot

        # Apply to rows
        for i in range(n):
            dot = sum(v[j - k - 1] * H[i][j] for j in range(k + 1, n))
            for j in range(k + 1, n):
                H[i][j] -= 2 * v[j - k - 1] * dot

    return Matrix(H)


def _qr_algorithm(H: Matrix, max_iter: int = 100) -> list[float]:
    """QR algorithm with Wilkinson shifts for eigenvalue computation."""
    n = H.rows
    data = copy.deepcopy(H._data)
    eigenvals = []

    m = n
    iteration = 0

    while m > 1 and iteration < max_iter * n:
        iteration += 1

        # Check for convergence on subdiagonal
        if abs(data[m - 1][m - 2]) < TOLERANCE * (abs(data[m - 1][m - 1]) + abs(data[m - 2][m - 2]) + 1):
            eigenvals.append(data[m - 1][m - 1])
            m -= 1
            continue

        # Wilkinson shift
        d = (data[m - 2][m - 2] - data[m - 1][m - 1]) / 2
        if abs(d) < TOLERANCE:
            shift = data[m - 1][m - 1] - abs(data[m - 1][m - 2])
        else:
            sign = 1 if d >= 0 else -1
            shift = data[m - 1][m - 1] - data[m - 1][m - 2] ** 2 / (d + sign * math.sqrt(d * d + data[m - 1][m - 2] ** 2))

        # QR step with shift
        for i in range(m):
            data[i][i] -= shift

        # QR factorization using Givens rotations
        cs = []
        sn = []
        for i in range(m - 1):
            a = data[i][i]
            b = data[i + 1][i]
            r = math.sqrt(a * a + b * b)
            if r < TOLERANCE:
                c, s = 1.0, 0.0
            else:
                c, s = a / r, -b / r
            cs.append(c)
            sn.append(s)

            # Apply Givens rotation to rows
            for j in range(i, m):
                temp = c * data[i][j] - s * data[i + 1][j]
                data[i + 1][j] = s * data[i][j] + c * data[i + 1][j]
                data[i][j] = temp

        # Apply Q^T from right (multiply by Q)
        for i in range(m - 1):
            c, s = cs[i], sn[i]
            for j in range(min(i + 2, m)):
                temp = c * data[j][i] - s * data[j][i + 1]
                data[j][i + 1] = s * data[j][i] + c * data[j][i + 1]
                data[j][i] = temp

        # Undo shift
        for i in range(m):
            data[i][i] += shift

    # Add remaining diagonal elements
    for i in range(m):
        eigenvals.append(data[i][i])

    return sorted(eigenvals, reverse=True)


if __name__ == "__main__":
    # Example from issue description
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    print("A * B:")
    print(multiply(A, B))
    print(f"\ndet(A) = {determinant(A)}")
    print("\ninv(A):")
    print(inverse(A))
