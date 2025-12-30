# Feature: Add matrix operations module

## Metadata
issue_number: `28`
adw_id: `4e37141a`
issue_json: `{"number":28,"title":"Add matrix operations module","body":"Implement a matrix operations module with:\n\n1. **Matrix class**: Create `Matrix(rows)` from nested lists, support `m[i][j]` access\n2. **Basic ops**: `add(A, B)`, `subtract(A, B)`, `multiply(A, B)`, `scalar_multiply(A, k)`\n3. **Advanced ops**: `transpose(A)`, `determinant(A)`, `inverse(A)`, `trace(A)`\n4. **Decompositions**: `lu_decompose(A)` returns (L, U, P) matrices\n5. **Solvers**: `solve(A, b)` for Ax=b using Gaussian elimination with partial pivoting\n6. **Eigenvalues**: `eigenvalues(A)` using QR algorithm (power iteration acceptable for 2x2)\n\nRequirements:\n- No numpy/scipy - pure Python implementation\n- Raise `ValueError` for dimension mismatches\n- Raise `ValueError('Singular matrix')` when inverse doesn't exist\n- Support matrices up to 10x10 efficiently\n- Pretty-print matrices with aligned columns\n\nExample:\n```python\nA = Matrix([[1, 2], [3, 4]])\nB = Matrix([[5, 6], [7, 8]])\nprint(multiply(A, B))  # [[19, 22], [43, 50]]\nprint(determinant(A))  # -2.0\nprint(inverse(A))      # [[-2, 1], [1.5, -0.5]]\n```"}`

## Feature Description
Implement a comprehensive pure-Python matrix operations module with a Matrix class and a full suite of linear algebra operations. The module will support matrix construction from nested lists, indexing via `m[i][j]`, basic arithmetic (add, subtract, multiply, scalar multiply), advanced operations (transpose, determinant, inverse, trace), LU decomposition with partial pivoting, linear system solving (Ax=b), and eigenvalue computation. All implementations must be pure Python without numpy or scipy dependencies.

## User Story
As a developer or mathematician
I want to perform matrix operations without external dependencies
So that I can solve linear algebra problems in Python environments where numpy/scipy are unavailable or unwanted

## Problem Statement
The calculator module currently supports only basic arithmetic operations on scalars. There is no support for matrix operations, which are essential for linear algebra, computer graphics, machine learning, and scientific computing applications. Users need a self-contained matrix library that can perform standard linear algebra operations without requiring heavy dependencies like numpy or scipy.

## Solution Statement
Create a new `matrix.py` module containing a `Matrix` class and comprehensive matrix operation functions. The Matrix class will store data as nested lists of floats, support indexing, and provide pretty-printing with aligned columns. The module will implement all standard matrix operations using pure Python, with proper error handling for dimension mismatches and singular matrices. Numerical algorithms like LU decomposition and QR iteration will use partial pivoting for numerical stability.

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Reference for existing code patterns, type hint style, and error handling approach (raises ValueError for invalid operations). The matrix module should follow similar patterns.
- **test_calculator.py** - Reference for test structure using unittest framework. The matrix tests should follow the same pattern with TestMatrix class.
- **pyproject.toml** - Project configuration. No new dependencies needed since implementation is pure Python.
- **app_docs/feature-de3d954e-divide-function.md** - Reference for documentation format after feature completion.

### New Files
- **matrix.py** - New module containing the Matrix class and all matrix operation functions
- **test_matrix.py** - Comprehensive test suite for all matrix operations including edge cases and error conditions

## Implementation Plan
### Phase 1: Foundation
Build the Matrix class with core infrastructure: data storage, construction from nested lists, row/column accessors, dimension validation, indexing support (`__getitem__`), equality checking (`__eq__`), and pretty-print formatting (`__str__` and `__repr__`). This foundation must be solid before implementing operations.

### Phase 2: Core Implementation
Implement matrix operations in order of increasing complexity:
1. **Basic operations**: add, subtract, scalar_multiply - straightforward element-wise operations
2. **Matrix multiplication**: multiply - requires proper dimension checking and row-by-column computation
3. **Simple advanced ops**: transpose, trace - relatively simple structural operations
4. **Determinant**: Use recursive cofactor expansion for small matrices, LU-based for larger
5. **Inverse**: Implement using Gauss-Jordan elimination or adjugate method
6. **LU decomposition**: Implement Doolittle's method with partial pivoting, returning (L, U, P)
7. **Linear solver**: Use LU decomposition for forward/backward substitution
8. **Eigenvalues**: Implement QR algorithm with shifts for general matrices, power iteration for 2x2

### Phase 3: Integration
Validate all operations work together (e.g., solve uses LU decomposition correctly). Ensure comprehensive test coverage. Document the module with docstrings and usage examples. Verify numerical accuracy with known test cases.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Create Matrix class foundation
- Create matrix.py in the project root
- Implement Matrix class with `__init__(self, rows: list[list[float]])` constructor
- Add validation: ensure all rows have same length, data is numeric
- Store data as list of lists (deep copy to prevent external mutation)
- Add `rows` and `cols` properties for dimensions
- Add `__getitem__` for `m[i][j]` access (return row as list, then index into it)
- Add `__eq__` for matrix equality comparison (with float tolerance)
- Add `__str__` for pretty-printing with aligned columns
- Add `__repr__` for debugging representation

### 2. Create initial test infrastructure
- Create test_matrix.py with unittest framework
- Create TestMatrix class with setUp method creating sample matrices
- Add tests for Matrix construction, dimensions, indexing
- Add tests for equality checking
- Add tests for dimension validation errors

### 3. Implement basic matrix operations
- Implement `add(A: Matrix, B: Matrix) -> Matrix` with dimension checking
- Implement `subtract(A: Matrix, B: Matrix) -> Matrix` with dimension checking
- Implement `scalar_multiply(A: Matrix, k: float) -> Matrix`
- Add tests for all basic operations including edge cases
- Verify dimension mismatch raises ValueError

### 4. Implement matrix multiplication
- Implement `multiply(A: Matrix, B: Matrix) -> Matrix`
- Validate dimensions: A.cols must equal B.rows
- Implement row-by-column dot product computation
- Add comprehensive tests including identity matrix multiplication
- Test dimension mismatch error handling

### 5. Implement transpose and trace
- Implement `transpose(A: Matrix) -> Matrix`
- Implement `trace(A: Matrix) -> float` with square matrix validation
- Add tests for transpose correctness
- Add tests for trace on various square matrices
- Test trace error on non-square matrices

### 6. Implement determinant calculation
- Implement `determinant(A: Matrix) -> float` for square matrices
- Use recursive cofactor expansion for matrices up to 3x3
- Use LU decomposition approach for larger matrices (product of diagonal of U)
- Handle singular matrices (det = 0) correctly
- Add comprehensive tests with known determinant values
- Test various matrix sizes (1x1, 2x2, 3x3, 4x4)

### 7. Implement LU decomposition
- Implement `lu_decompose(A: Matrix) -> tuple[Matrix, Matrix, Matrix]`
- Use Doolittle's algorithm with partial pivoting
- Return (L, U, P) where PA = LU
- L is lower triangular with 1s on diagonal
- U is upper triangular
- P is permutation matrix
- Add tests verifying PA = LU for various matrices
- Test singular matrix detection

### 8. Implement matrix inverse
- Implement `inverse(A: Matrix) -> Matrix`
- Use Gauss-Jordan elimination for numerical stability
- Raise ValueError('Singular matrix') when determinant is zero/near-zero
- Add tests for inverse correctness: A * inv(A) = I
- Test singular matrix error handling
- Test various matrix sizes

### 9. Implement linear system solver
- Implement `solve(A: Matrix, b: Matrix) -> Matrix` for Ax = b
- Use LU decomposition with forward/backward substitution
- b should be a column vector (Matrix with 1 column)
- Return x as column vector
- Validate dimensions: A must be square, b must have A.rows rows
- Add tests with known solutions
- Test singular system detection

### 10. Implement eigenvalue computation
- Implement `eigenvalues(A: Matrix) -> list[float]`
- For 2x2 matrices: use quadratic formula on characteristic polynomial
- For larger matrices: implement QR algorithm with shifts
- Handle real eigenvalues (complex eigenvalues can return as NaN or raise error)
- Add tests with matrices of known eigenvalues
- Test symmetric matrices (all real eigenvalues)

### 11. Add numerical stability and edge cases
- Add tolerance parameter for near-zero comparisons (default 1e-10)
- Handle near-singular matrices appropriately
- Add tests for numerical edge cases
- Verify operations work correctly for 10x10 matrices
- Add performance sanity check tests

### 12. Run validation commands
- Execute all validation commands listed below
- Verify all tests pass with zero failures
- Test the example code from the issue description
- Confirm matrix module integrates cleanly with project

## Testing Strategy
### Unit Tests
- **test_matrix_construction** - Verify Matrix created from nested lists with correct dimensions
- **test_matrix_indexing** - Verify m[i][j] returns correct elements
- **test_matrix_equality** - Verify equality with float tolerance
- **test_matrix_pretty_print** - Verify string output is aligned
- **test_add** - Verify element-wise addition
- **test_subtract** - Verify element-wise subtraction
- **test_scalar_multiply** - Verify scalar multiplication
- **test_multiply** - Verify matrix multiplication with various dimensions
- **test_transpose** - Verify transpose correctness
- **test_trace** - Verify trace calculation
- **test_determinant** - Verify determinant for 1x1, 2x2, 3x3, larger matrices
- **test_inverse** - Verify A * inv(A) = I
- **test_lu_decompose** - Verify PA = LU
- **test_solve** - Verify Ax = b solutions
- **test_eigenvalues** - Verify eigenvalues for known matrices

### Edge Cases
- Empty matrix or 1x1 matrix operations
- Identity matrix operations (multiply, inverse, determinant)
- Zero matrix operations
- Singular matrix detection (determinant = 0, inverse fails)
- Dimension mismatch errors for all operations
- Non-square matrix errors for trace, determinant, inverse, eigenvalues
- Nearly singular matrices (numerical stability)
- Negative and fractional values
- Large matrices (10x10) for performance validation

## Acceptance Criteria
- Matrix class exists with construction from nested lists and m[i][j] indexing
- add, subtract, multiply, scalar_multiply functions work correctly
- transpose, determinant, inverse, trace functions work correctly
- lu_decompose returns valid (L, U, P) decomposition
- solve correctly solves Ax=b systems
- eigenvalues returns correct eigenvalues for square matrices
- All operations raise ValueError for dimension mismatches
- inverse raises ValueError('Singular matrix') for non-invertible matrices
- Pretty-print displays matrices with aligned columns
- No numpy or scipy dependencies used
- All unit tests pass with 100% success rate
- Example code from issue description produces expected output
- Operations work efficiently for matrices up to 10x10

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `uv run pytest test_matrix.py -v` - Run all matrix tests with verbose output, verify zero failures
- `python test_matrix.py` - Run tests using unittest directly, confirm all tests pass
- `uv run pytest test_calculator.py -v` - Verify no regressions in existing calculator tests
- `python -c "from matrix import Matrix, multiply, determinant, inverse; A = Matrix([[1, 2], [3, 4]]); B = Matrix([[5, 6], [7, 8]]); print(multiply(A, B)); print(determinant(A)); print(inverse(A))"` - Test example code from issue
- `python -c "from matrix import Matrix, solve; A = Matrix([[2, 1], [1, 3]]); b = Matrix([[3], [4]]); print(solve(A, b))"` - Test linear solver
- `python -c "from matrix import Matrix, lu_decompose; A = Matrix([[4, 3], [6, 3]]); L, U, P = lu_decompose(A); print('L:'); print(L); print('U:'); print(U); print('P:'); print(P)"` - Test LU decomposition
- `uv run ruff check matrix.py test_matrix.py` - Run linter on new files

## Notes
- Using floats internally for all matrix operations to support fractional results (inverse, etc.)
- LU decomposition uses partial pivoting for numerical stability
- Eigenvalue computation for matrices larger than 2x2 uses QR algorithm - may not converge for all matrices
- Complex eigenvalues are not supported in this implementation
- For determinant calculation, using LU decomposition for efficiency on larger matrices
- Matrix equality uses tolerance-based comparison (1e-10) for float precision
- The `solve` function assumes the system has a unique solution; underdetermined/overdetermined systems not supported
- Consider adding identity matrix constructor (Matrix.identity(n)) as future enhancement
- Consider adding matrix slicing support as future enhancement
