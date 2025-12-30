"""(Claude) Unit tests for matrix module."""

import unittest
from matrix import (
    Matrix, add, subtract, scalar_multiply, multiply,
    transpose, trace, determinant, lu_decompose, inverse,
    solve, eigenvalues, TOLERANCE
)


class TestMatrixConstruction(unittest.TestCase):
    """Test Matrix class construction and basic properties."""

    def test_basic_construction(self):
        """if Matrix construction from nested list fails then broken"""
        m = Matrix([[1, 2], [3, 4]])
        assert m.rows == 2
        assert m.cols == 2

    def test_rectangular_matrix(self):
        """if rectangular matrix dimensions wrong then broken"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.rows == 2
        assert m.cols == 3

    def test_single_element(self):
        """if 1x1 matrix construction fails then broken"""
        m = Matrix([[42]])
        assert m.rows == 1
        assert m.cols == 1
        assert m[0][0] == 42

    def test_empty_matrix_error(self):
        """if empty matrix doesn't raise ValueError then broken"""
        with self.assertRaises(ValueError):
            Matrix([])

    def test_empty_row_error(self):
        """if empty row doesn't raise ValueError then broken"""
        with self.assertRaises(ValueError):
            Matrix([[]])

    def test_inconsistent_row_lengths(self):
        """if inconsistent rows don't raise ValueError then broken"""
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_non_numeric_value(self):
        """if non-numeric value doesn't raise ValueError then broken"""
        with self.assertRaises(ValueError):
            Matrix([[1, "a"], [3, 4]])

    def test_integer_conversion_to_float(self):
        """if integers aren't stored as floats then broken"""
        m = Matrix([[1, 2], [3, 4]])
        assert isinstance(m[0][0], float)

    def test_data_immutability(self):
        """if external mutation affects matrix then broken"""
        data = [[1, 2], [3, 4]]
        m = Matrix(data)
        data[0][0] = 999
        assert m[0][0] == 1.0


class TestMatrixIndexing(unittest.TestCase):
    """Test matrix indexing operations."""

    def test_basic_indexing(self):
        """if m[i][j] indexing wrong then broken"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m[0][0] == 1
        assert m[0][2] == 3
        assert m[1][1] == 5

    def test_row_access(self):
        """if row access returns wrong values then broken"""
        m = Matrix([[1, 2], [3, 4]])
        row = m[0]
        assert row == [1.0, 2.0]

    def test_row_immutability(self):
        """if modifying returned row affects matrix then broken"""
        m = Matrix([[1, 2], [3, 4]])
        row = m[0]
        row[0] = 999
        assert m[0][0] == 1.0


class TestMatrixEquality(unittest.TestCase):
    """Test matrix equality comparison."""

    def test_equal_matrices(self):
        """if equal matrices don't compare equal then broken"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        assert m1 == m2

    def test_different_values(self):
        """if different matrices compare equal then broken"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        assert m1 != m2

    def test_different_dimensions(self):
        """if different dimension matrices compare equal then broken"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m1 != m2

    def test_tolerance_comparison(self):
        """if float tolerance not applied then broken"""
        m1 = Matrix([[1.0, 2.0]])
        m2 = Matrix([[1.0 + TOLERANCE / 10, 2.0]])
        assert m1 == m2

    def test_non_matrix_comparison(self):
        """if comparing to non-matrix doesn't return False then broken"""
        m = Matrix([[1, 2]])
        assert m != [[1, 2]]
        assert m != "matrix"
        assert m != 42


class TestMatrixStringRepresentation(unittest.TestCase):
    """Test matrix string representation."""

    def test_str_basic(self):
        """if str representation wrong format then broken"""
        m = Matrix([[1, 2], [3, 4]])
        s = str(m)
        assert "[" in s
        assert "1" in s
        assert "4" in s

    def test_repr(self):
        """if repr doesn't include Matrix then broken"""
        m = Matrix([[1, 2]])
        assert "Matrix" in repr(m)


class TestMatrixAdd(unittest.TestCase):
    """Test matrix addition."""

    def test_basic_add(self):
        """if basic addition wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = add(A, B)
        expected = Matrix([[6, 8], [10, 12]])
        assert C == expected

    def test_add_zeros(self):
        """if adding zero matrix changes result then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Z = Matrix([[0, 0], [0, 0]])
        assert add(A, Z) == A

    def test_add_negative(self):
        """if adding negatives fails then broken"""
        A = Matrix([[1, 2]])
        B = Matrix([[-1, -2]])
        expected = Matrix([[0, 0]])
        assert add(A, B) == expected

    def test_add_dimension_mismatch(self):
        """if dimension mismatch doesn't raise ValueError then broken"""
        A = Matrix([[1, 2]])
        B = Matrix([[1], [2]])
        with self.assertRaises(ValueError):
            add(A, B)


class TestMatrixSubtract(unittest.TestCase):
    """Test matrix subtraction."""

    def test_basic_subtract(self):
        """if basic subtraction wrong then broken"""
        A = Matrix([[5, 6], [7, 8]])
        B = Matrix([[1, 2], [3, 4]])
        C = subtract(A, B)
        expected = Matrix([[4, 4], [4, 4]])
        assert C == expected

    def test_subtract_self(self):
        """if subtracting self doesn't give zero then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Z = subtract(A, A)
        expected = Matrix([[0, 0], [0, 0]])
        assert Z == expected

    def test_subtract_dimension_mismatch(self):
        """if dimension mismatch doesn't raise ValueError then broken"""
        A = Matrix([[1, 2]])
        B = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            subtract(A, B)


class TestScalarMultiply(unittest.TestCase):
    """Test scalar multiplication."""

    def test_basic_scalar_multiply(self):
        """if scalar multiplication wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        B = scalar_multiply(A, 2)
        expected = Matrix([[2, 4], [6, 8]])
        assert B == expected

    def test_multiply_by_zero(self):
        """if multiplying by zero doesn't give zero matrix then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Z = scalar_multiply(A, 0)
        expected = Matrix([[0, 0], [0, 0]])
        assert Z == expected

    def test_multiply_by_negative(self):
        """if multiplying by negative fails then broken"""
        A = Matrix([[1, 2]])
        B = scalar_multiply(A, -1)
        expected = Matrix([[-1, -2]])
        assert B == expected

    def test_multiply_by_fraction(self):
        """if multiplying by fraction fails then broken"""
        A = Matrix([[2, 4]])
        B = scalar_multiply(A, 0.5)
        expected = Matrix([[1, 2]])
        assert B == expected


class TestMatrixMultiply(unittest.TestCase):
    """Test matrix multiplication."""

    def test_basic_multiply(self):
        """if matrix multiplication wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = multiply(A, B)
        expected = Matrix([[19, 22], [43, 50]])
        assert C == expected

    def test_multiply_identity(self):
        """if multiplying by identity changes matrix then broken"""
        A = Matrix([[1, 2], [3, 4]])
        ident = Matrix.identity(2)
        assert multiply(A, ident) == A
        assert multiply(ident, A) == A

    def test_multiply_rectangular(self):
        """if rectangular multiplication wrong then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        B = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2
        C = multiply(A, B)  # 2x2
        expected = Matrix([[58, 64], [139, 154]])
        assert C == expected
        assert C.rows == 2
        assert C.cols == 2

    def test_multiply_dimension_mismatch(self):
        """if inner dimension mismatch doesn't raise ValueError then broken"""
        A = Matrix([[1, 2]])  # 1x2
        B = Matrix([[1, 2]])  # 1x2 - can't multiply
        with self.assertRaises(ValueError):
            multiply(A, B)

    def test_multiply_zeros(self):
        """if multiplying by zero matrix doesn't give zeros then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Z = Matrix.zeros(2, 2)
        assert multiply(A, Z) == Z


class TestTranspose(unittest.TestCase):
    """Test matrix transpose."""

    def test_basic_transpose(self):
        """if transpose wrong then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        At = transpose(A)
        expected = Matrix([[1, 4], [2, 5], [3, 6]])
        assert At == expected

    def test_transpose_square(self):
        """if square transpose wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        At = transpose(A)
        expected = Matrix([[1, 3], [2, 4]])
        assert At == expected

    def test_transpose_twice(self):
        """if double transpose doesn't give original then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        assert transpose(transpose(A)) == A

    def test_transpose_dimensions(self):
        """if transpose dimensions wrong then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        At = transpose(A)
        assert At.rows == 3
        assert At.cols == 2


class TestTrace(unittest.TestCase):
    """Test matrix trace."""

    def test_basic_trace(self):
        """if trace calculation wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        assert trace(A) == 5  # 1 + 4

    def test_trace_identity(self):
        """if identity trace wrong then broken"""
        ident = Matrix.identity(3)
        assert trace(ident) == 3

    def test_trace_single_element(self):
        """if 1x1 trace wrong then broken"""
        A = Matrix([[42]])
        assert trace(A) == 42

    def test_trace_non_square(self):
        """if non-square trace doesn't raise ValueError then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            trace(A)


class TestDeterminant(unittest.TestCase):
    """Test determinant calculation."""

    def test_det_1x1(self):
        """if 1x1 determinant wrong then broken"""
        A = Matrix([[5]])
        assert abs(determinant(A) - 5) < TOLERANCE

    def test_det_2x2(self):
        """if 2x2 determinant wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        assert abs(determinant(A) - (-2)) < TOLERANCE

    def test_det_3x3(self):
        """if 3x3 determinant wrong then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        # det = 1*(50-48) - 2*(40-42) + 3*(32-35) = 2 + 4 - 9 = -3
        assert abs(determinant(A) - (-3)) < TOLERANCE

    def test_det_identity(self):
        """if identity determinant not 1 then broken"""
        ident = Matrix.identity(3)
        assert abs(determinant(ident) - 1) < TOLERANCE

    def test_det_singular(self):
        """if singular matrix determinant not 0 then broken"""
        A = Matrix([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
        assert abs(determinant(A)) < TOLERANCE

    def test_det_4x4(self):
        """if 4x4 determinant wrong then broken"""
        # Block diagonal matrix: det = det(2x2 block1) * det(2x2 block2)
        # Block1: [[1,2],[3,4]] has det = -2
        # Block2: [[1,2],[3,4]] has det = -2
        # Total: (-2) * (-2) = 4
        A = Matrix([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 3, 4]
        ])
        assert abs(determinant(A) - 4) < TOLERANCE

    def test_det_non_square(self):
        """if non-square determinant doesn't raise ValueError then broken"""
        A = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            determinant(A)


class TestLUDecompose(unittest.TestCase):
    """Test LU decomposition."""

    def test_basic_lu(self):
        """if PA != LU then broken"""
        A = Matrix([[4, 3], [6, 3]])
        L, U, P = lu_decompose(A)

        # Verify PA = LU
        PA = multiply(P, A)
        LU = multiply(L, U)
        assert PA == LU

    def test_lu_3x3(self):
        """if 3x3 PA != LU then broken"""
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        L, U, P = lu_decompose(A)

        PA = multiply(P, A)
        LU = multiply(L, U)
        assert PA == LU

    def test_lu_lower_triangular(self):
        """if L not lower triangular with 1s diagonal then broken"""
        A = Matrix([[4, 3], [6, 3]])
        L, U, P = lu_decompose(A)

        # Check L is lower triangular with 1s on diagonal
        for i in range(L.rows):
            assert abs(L[i][i] - 1.0) < TOLERANCE  # Diagonal is 1
            for j in range(i + 1, L.cols):
                assert abs(L[i][j]) < TOLERANCE  # Upper triangle is 0

    def test_lu_upper_triangular(self):
        """if U not upper triangular then broken"""
        A = Matrix([[4, 3], [6, 3]])
        L, U, P = lu_decompose(A)

        # Check U is upper triangular
        for i in range(U.rows):
            for j in range(i):
                assert abs(U[i][j]) < TOLERANCE  # Lower triangle is 0

    def test_lu_singular(self):
        """if singular matrix doesn't raise ValueError then broken"""
        A = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            lu_decompose(A)


class TestInverse(unittest.TestCase):
    """Test matrix inverse."""

    def test_basic_inverse(self):
        """if A * inv(A) != I then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Ainv = inverse(A)
        ident = Matrix.identity(2)
        product = multiply(A, Ainv)
        assert product == ident

    def test_inverse_values(self):
        """if inverse values wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Ainv = inverse(A)
        # inv([[1,2],[3,4]]) = [[-2, 1], [1.5, -0.5]]
        expected = Matrix([[-2, 1], [1.5, -0.5]])
        assert Ainv == expected

    def test_inverse_3x3(self):
        """if 3x3 A * inv(A) != I then broken"""
        A = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        Ainv = inverse(A)
        ident = Matrix.identity(3)
        product = multiply(A, Ainv)
        assert product == ident

    def test_inverse_identity(self):
        """if inverse of identity not identity then broken"""
        ident = Matrix.identity(3)
        assert inverse(ident) == ident

    def test_inverse_singular(self):
        """if singular matrix inverse doesn't raise ValueError('Singular matrix') then broken"""
        A = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError) as ctx:
            inverse(A)
        assert "Singular matrix" in str(ctx.exception)

    def test_inverse_non_square(self):
        """if non-square inverse doesn't raise ValueError then broken"""
        A = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            inverse(A)


class TestSolve(unittest.TestCase):
    """Test linear system solver."""

    def test_basic_solve(self):
        """if Ax != b for computed x then broken"""
        A = Matrix([[2, 1], [1, 3]])
        b = Matrix([[3], [4]])
        x = solve(A, b)

        # Verify Ax = b
        Ax = multiply(A, x)
        assert Ax == b

    def test_solve_3x3(self):
        """if 3x3 Ax != b then broken"""
        A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = Matrix([[8], [-11], [-3]])
        x = solve(A, b)

        Ax = multiply(A, x)
        assert Ax == b

    def test_solve_identity(self):
        """if Ix = b doesn't give x = b then broken"""
        ident = Matrix.identity(2)
        b = Matrix([[3], [4]])
        x = solve(ident, b)
        assert x == b

    def test_solve_singular(self):
        """if singular system doesn't raise ValueError then broken"""
        A = Matrix([[1, 2], [2, 4]])
        b = Matrix([[3], [6]])
        with self.assertRaises(ValueError):
            solve(A, b)

    def test_solve_dimension_mismatch(self):
        """if dimension mismatch doesn't raise ValueError then broken"""
        A = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1], [2], [3]])  # Wrong number of rows
        with self.assertRaises(ValueError):
            solve(A, b)

    def test_solve_non_column_vector(self):
        """if non-column b doesn't raise ValueError then broken"""
        A = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 4]])  # Not a column vector
        with self.assertRaises(ValueError):
            solve(A, b)


class TestEigenvalues(unittest.TestCase):
    """Test eigenvalue computation."""

    def test_eigenvalues_2x2(self):
        """if 2x2 eigenvalues wrong then broken"""
        # [[3, 1], [1, 3]] has eigenvalues 4 and 2
        A = Matrix([[3, 1], [1, 3]])
        eigs = eigenvalues(A)
        assert len(eigs) == 2
        assert abs(max(eigs) - 4) < TOLERANCE
        assert abs(min(eigs) - 2) < TOLERANCE

    def test_eigenvalues_identity(self):
        """if identity eigenvalues not all 1 then broken"""
        ident = Matrix.identity(3)
        eigs = eigenvalues(ident)
        assert len(eigs) == 3
        for e in eigs:
            assert abs(e - 1) < TOLERANCE

    def test_eigenvalues_diagonal(self):
        """if diagonal matrix eigenvalues wrong then broken"""
        A = Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 5]])
        eigs = sorted(eigenvalues(A), reverse=True)
        assert abs(eigs[0] - 5) < TOLERANCE
        assert abs(eigs[1] - 3) < TOLERANCE
        assert abs(eigs[2] - 2) < TOLERANCE

    def test_eigenvalues_1x1(self):
        """if 1x1 eigenvalue wrong then broken"""
        A = Matrix([[7]])
        eigs = eigenvalues(A)
        assert len(eigs) == 1
        assert abs(eigs[0] - 7) < TOLERANCE

    def test_eigenvalues_non_square(self):
        """if non-square eigenvalues doesn't raise ValueError then broken"""
        A = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            eigenvalues(A)

    def test_eigenvalues_symmetric_3x3(self):
        """if symmetric 3x3 eigenvalues wrong then broken"""
        # Symmetric matrices have real eigenvalues
        A = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
        eigs = eigenvalues(A)
        assert len(eigs) == 3
        # Sum of eigenvalues = trace
        assert abs(sum(eigs) - trace(A)) < 0.01
        # Product of eigenvalues = determinant
        prod = 1
        for e in eigs:
            prod *= e
        assert abs(prod - determinant(A)) < 0.01


class TestIdentityAndZeros(unittest.TestCase):
    """Test static matrix constructors."""

    def test_identity_2x2(self):
        """if 2x2 identity wrong then broken"""
        ident = Matrix.identity(2)
        expected = Matrix([[1, 0], [0, 1]])
        assert ident == expected

    def test_identity_3x3(self):
        """if 3x3 identity wrong then broken"""
        ident = Matrix.identity(3)
        assert ident[0][0] == 1 and ident[1][1] == 1 and ident[2][2] == 1
        assert ident[0][1] == 0 and ident[0][2] == 0

    def test_zeros_2x3(self):
        """if zeros matrix wrong then broken"""
        Z = Matrix.zeros(2, 3)
        assert Z.rows == 2 and Z.cols == 3
        for i in range(2):
            for j in range(3):
                assert Z[i][j] == 0


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and numerical stability."""

    def test_large_matrix_multiply(self):
        """if 10x10 multiplication fails then broken"""
        A = Matrix([[float(i + j) for j in range(10)] for i in range(10)])
        B = Matrix.identity(10)
        C = multiply(A, B)
        assert C == A

    def test_large_matrix_determinant(self):
        """if 10x10 determinant fails then broken"""
        # Use identity - determinant is 1
        ident = Matrix.identity(10)
        assert abs(determinant(ident) - 1) < TOLERANCE

    def test_nearly_singular(self):
        """if nearly singular matrix handled incorrectly then broken"""
        # Very small but non-zero determinant
        eps = 1e-8
        A = Matrix([[1, 2], [0.5, 1 + eps]])
        det = determinant(A)
        assert abs(det - eps) < 1e-6

    def test_negative_values(self):
        """if negative values cause issues then broken"""
        A = Matrix([[-1, -2], [-3, -4]])
        B = Matrix([[-5, -6], [-7, -8]])
        C = multiply(A, B)
        expected = Matrix([[19, 22], [43, 50]])
        assert C == expected

    def test_fractional_values(self):
        """if fractional values cause issues then broken"""
        A = Matrix([[0.5, 0.25], [0.125, 0.0625]])
        At = transpose(A)
        expected = Matrix([[0.5, 0.125], [0.25, 0.0625]])
        assert At == expected


class TestIssueExamples(unittest.TestCase):
    """Test examples from the issue description."""

    def test_issue_multiply_example(self):
        """if issue multiply example wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        result = multiply(A, B)
        # [[19, 22], [43, 50]]
        assert result[0][0] == 19
        assert result[0][1] == 22
        assert result[1][0] == 43
        assert result[1][1] == 50

    def test_issue_determinant_example(self):
        """if issue determinant example wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        det = determinant(A)
        assert abs(det - (-2.0)) < TOLERANCE

    def test_issue_inverse_example(self):
        """if issue inverse example wrong then broken"""
        A = Matrix([[1, 2], [3, 4]])
        Ainv = inverse(A)
        # [[-2, 1], [1.5, -0.5]]
        assert abs(Ainv[0][0] - (-2)) < TOLERANCE
        assert abs(Ainv[0][1] - 1) < TOLERANCE
        assert abs(Ainv[1][0] - 1.5) < TOLERANCE
        assert abs(Ainv[1][1] - (-0.5)) < TOLERANCE


if __name__ == "__main__":
    unittest.main()
