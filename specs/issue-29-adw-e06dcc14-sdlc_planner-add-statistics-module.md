# Feature: Add Statistics Module with Distributions

## Metadata
issue_number: `29`
adw_id: `e06dcc14`
issue_json: `{"number":29,"title":"Add statistics module with distributions","body":"Implement a statistics module with:\n\n1. **Descriptive stats**: `mean(data)`, `median(data)`, `mode(data)`, `variance(data)`, `std_dev(data)`\n2. **Percentiles**: `percentile(data, p)`, `quartiles(data)`, `iqr(data)`\n3. **Correlation**: `correlation(x, y)`, `covariance(x, y)`, `linear_regression(x, y)` returns (slope, intercept, r_squared)\n4. **Distributions**:\n   - `normal_pdf(x, mu, sigma)`, `normal_cdf(x, mu, sigma)`\n   - `binomial_pmf(k, n, p)`, `binomial_cdf(k, n, p)`\n   - `poisson_pmf(k, lambda_)`, `poisson_cdf(k, lambda_)`\n5. **Hypothesis testing**: `t_test(sample1, sample2)` returns (t_stat, p_value), `chi_square_test(observed, expected)`\n6. **Sampling**: `bootstrap_sample(data, n)`, `confidence_interval(data, confidence=0.95)`\n\nRequirements:\n- No scipy/statsmodels - pure Python (math module OK)\n- Use Taylor series for erf/normal CDF approximation\n- Raise `ValueError` for empty datasets\n- Handle edge cases: single element, all same values\n- Results accurate to 6 decimal places\n\nExample:\n```python\ndata = [2, 4, 4, 4, 5, 5, 7, 9]\nprint(mean(data))      # 5.0\nprint(std_dev(data))   # 2.0\nprint(percentile(data, 75))  # 6.0\n\nx = [1, 2, 3, 4, 5]\ny = [2, 4, 5, 4, 5]\nslope, intercept, r2 = linear_regression(x, y)\nprint(f'y = {slope:.2f}x + {intercept:.2f}, R²={r2:.3f}')\n```"}`

## Feature Description
Implement a comprehensive statistics module that provides pure Python implementations of common statistical functions. The module will include descriptive statistics (mean, median, mode, variance, standard deviation), percentile calculations, correlation and regression analysis, probability distribution functions (normal, binomial, Poisson), hypothesis testing (t-test, chi-square), and sampling methods (bootstrap, confidence intervals). All implementations must use pure Python with only the math module allowed - no scipy or statsmodels dependencies. Results must be accurate to 6 decimal places, and all functions must properly handle edge cases including empty datasets, single-element datasets, and datasets with all identical values.

## User Story
As a developer using this calculator application
I want to perform statistical analysis on datasets
So that I can analyze data distributions, test hypotheses, and make statistical inferences without requiring heavy external dependencies

## Problem Statement
The calculator module currently only supports basic arithmetic operations. Users need comprehensive statistical analysis capabilities without requiring heavy dependencies like scipy or statsmodels. This includes descriptive statistics for summarizing data, percentile calculations for understanding data distribution, correlation/regression for analyzing relationships between variables, probability distribution functions for statistical modeling, hypothesis testing for statistical inference, and sampling methods for statistical estimation.

## Solution Statement
Create a new `statistics.py` module following the existing calculator.py patterns (type hints, ValueError for invalid inputs, comprehensive tests). The module will implement all requested statistical functions using pure Python with the math module for mathematical constants and functions. Key implementation details:
- Taylor series expansion for the error function (erf) to calculate normal CDF
- Newton's method concepts for iterative calculations where needed
- Proper handling of edge cases (empty data, single element, all same values)
- 6 decimal place accuracy through careful numerical computation
- Consistent error handling with descriptive ValueError messages

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Reference for existing code patterns: type hints, docstrings, error handling with ValueError, and main block demonstration. The statistics module should follow the same patterns for consistency.

- **test_calculator.py** - Reference for test structure using unittest framework with TestCase class. Statistics tests should follow the same pattern with comprehensive test methods covering normal operations, edge cases, and error handling.

- **pyproject.toml** - Project configuration. No new dependencies needed since we're using pure Python with only the math module. May need to add pytest to dependencies if not already available via adw-framework.

- **app_docs/feature-de3d954e-divide-function.md** - Reference documentation showing established documentation pattern for features. Will create similar documentation for the statistics module.

- **.adw.yaml** - ADW configuration specifying test_command: "uv run pytest". Used for validation commands.

### New Files
- **statistics.py** - New module containing all statistics functions. Will be placed at project root alongside calculator.py. Contains:
  - Descriptive stats: mean, median, mode, variance, std_dev
  - Percentiles: percentile, quartiles, iqr
  - Correlation: correlation, covariance, linear_regression
  - Distributions: normal_pdf, normal_cdf, binomial_pmf, binomial_cdf, poisson_pmf, poisson_cdf
  - Hypothesis testing: t_test, chi_square_test
  - Sampling: bootstrap_sample, confidence_interval
  - Helper functions: _erf (error function via Taylor series), _factorial, _combinations

- **test_statistics.py** - Comprehensive test suite for all statistics functions. Will follow test_calculator.py pattern with TestStatistics class containing test methods for each function group.

## Implementation Plan
### Phase 1: Foundation
Implement core helper functions and descriptive statistics. This includes the error function (erf) using Taylor series approximation for later use in normal distribution calculations, factorial and combinations functions for discrete distributions, and the basic descriptive statistics functions (mean, median, mode, variance, std_dev). Establish the module structure and error handling patterns.

### Phase 2: Core Implementation
Build out the remaining statistical functions in logical groups:
1. Percentile functions (percentile, quartiles, iqr) - depend on sorting and interpolation
2. Correlation functions (covariance, correlation, linear_regression) - depend on mean, variance
3. Distribution functions (normal, binomial, Poisson PDFs and CDFs) - depend on helpers
4. Hypothesis testing (t_test, chi_square_test) - depend on distributions
5. Sampling methods (bootstrap_sample, confidence_interval) - depend on descriptive stats

### Phase 3: Integration
Create comprehensive test suite covering all functions with normal cases, edge cases, and error handling. Add main block demonstration showing example usage of key functions. Validate accuracy against known statistical values. Ensure all tests pass and code follows project conventions.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Create Statistics Module Foundation
- Create statistics.py file at project root
- Add module docstring explaining purpose and listing all functions
- Import math module (only external dependency allowed)
- Define List type alias for type hints
- Implement input validation helper: _validate_data(data) raises ValueError for empty lists
- Implement _factorial(n) helper function for discrete distributions
- Implement _combinations(n, k) helper for binomial calculations

### Step 2: Implement Error Function for Normal Distribution
- Implement _erf(x) using Taylor series approximation
- Taylor series: erf(x) = (2/sqrt(pi)) * sum((-1)^n * x^(2n+1) / (n! * (2n+1)))
- Use sufficient terms (at least 50) for 6 decimal place accuracy
- Handle edge cases: erf(0) = 0, erf(large) approaches +/-1

### Step 3: Implement Descriptive Statistics
- Implement mean(data: list[float]) -> float
- Implement median(data: list[float]) -> float (handle odd/even lengths)
- Implement mode(data: list[float]) -> float (return first mode if multiple)
- Implement variance(data: list[float], population: bool = True) -> float
- Implement std_dev(data: list[float], population: bool = True) -> float
- All functions raise ValueError for empty data
- Handle edge cases: single element returns that element for mean/median/mode, 0.0 for variance/std_dev

### Step 4: Implement Percentile Functions
- Implement percentile(data: list[float], p: float) -> float using linear interpolation
- Validate p is between 0 and 100, raise ValueError otherwise
- Implement quartiles(data: list[float]) -> tuple[float, float, float] (Q1, Q2, Q3)
- Implement iqr(data: list[float]) -> float (Q3 - Q1)
- Handle edge cases: single element returns that element for all percentiles

### Step 5: Implement Correlation and Regression
- Implement covariance(x: list[float], y: list[float]) -> float
- Validate x and y have same length, raise ValueError otherwise
- Implement correlation(x: list[float], y: list[float]) -> float (Pearson correlation coefficient)
- Handle edge case: if std_dev is 0, correlation is undefined (raise ValueError or return 0.0)
- Implement linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]
- Returns (slope, intercept, r_squared)
- Handle edge case: vertical line (x all same) - raise ValueError

### Step 6: Implement Distribution Functions
- Implement normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float
- Validate sigma > 0, raise ValueError otherwise
- Implement normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float using _erf
- Implement binomial_pmf(k: int, n: int, p: float) -> float
- Validate 0 <= p <= 1 and 0 <= k <= n, raise ValueError otherwise
- Implement binomial_cdf(k: int, n: int, p: float) -> float (sum of pmf from 0 to k)
- Implement poisson_pmf(k: int, lambda_: float) -> float
- Validate lambda_ > 0 and k >= 0, raise ValueError otherwise
- Implement poisson_cdf(k: int, lambda_: float) -> float (sum of pmf from 0 to k)

### Step 7: Implement Hypothesis Testing
- Implement t_test(sample1: list[float], sample2: list[float]) -> tuple[float, float]
- Calculate two-sample t-statistic: t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
- Approximate p-value using normal distribution (for large samples) or t-distribution approximation
- Returns (t_stat, p_value)
- Implement chi_square_test(observed: list[float], expected: list[float]) -> tuple[float, float]
- Validate lengths match and expected values are positive
- Calculate chi-square statistic: sum((O-E)^2/E)
- Approximate p-value using chi-square distribution approximation
- Returns (chi_stat, p_value)

### Step 8: Implement Sampling Methods
- Implement bootstrap_sample(data: list[float], n: int) -> list[float]
- Validate n > 0, raise ValueError otherwise
- Use random module for sampling with replacement
- Implement confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]
- Validate 0 < confidence < 1
- Calculate interval: mean +/- z * (std_dev / sqrt(n))
- Use normal distribution critical values (z-score from confidence level)
- Returns (lower_bound, upper_bound)

### Step 9: Add Main Block Demonstration
- Add `if __name__ == "__main__":` block to statistics.py
- Demonstrate descriptive stats with example data [2, 4, 4, 4, 5, 5, 7, 9]
- Show mean, median, mode, std_dev outputs
- Demonstrate percentile and quartiles
- Show correlation and linear regression example
- Demonstrate distribution function calculations
- Include examples from issue specification

### Step 10: Create Test Suite
- Create test_statistics.py following test_calculator.py pattern
- Create TestStatistics class inheriting from unittest.TestCase
- Add test_mean, test_median, test_mode, test_variance, test_std_dev methods
- Add test_percentile, test_quartiles, test_iqr methods
- Add test_covariance, test_correlation, test_linear_regression methods
- Add test_normal_pdf, test_normal_cdf methods
- Add test_binomial_pmf, test_binomial_cdf methods
- Add test_poisson_pmf, test_poisson_cdf methods
- Add test_t_test, test_chi_square_test methods
- Add test_bootstrap_sample, test_confidence_interval methods
- Add test_empty_data method verifying ValueError for all functions
- Add test_single_element method verifying edge case handling
- Add test_same_values method verifying edge case handling
- Use assertAlmostEqual with places=6 for float comparisons

### Step 11: Run Validation Commands
- Execute all validation commands to verify implementation
- Fix any failing tests or issues discovered
- Ensure 100% test pass rate with zero regressions

## Testing Strategy
### Unit Tests
- **test_mean**: Test mean calculation with integers, floats, positive/negative values. Verify mean([2,4,4,4,5,5,7,9]) = 5.0
- **test_median**: Test odd and even length lists, sorted and unsorted data
- **test_mode**: Test single mode, multiple modes (return first), all unique (return first)
- **test_variance**: Test population variance, verify variance([2,4,4,4,5,5,7,9]) = 4.0
- **test_std_dev**: Test standard deviation, verify std_dev([2,4,4,4,5,5,7,9]) = 2.0
- **test_percentile**: Test 0th, 25th, 50th, 75th, 100th percentiles, edge cases
- **test_quartiles**: Test Q1, Q2, Q3 calculations
- **test_iqr**: Test interquartile range calculation
- **test_covariance**: Test positive, negative, zero covariance
- **test_correlation**: Test perfect positive (1.0), perfect negative (-1.0), no correlation (0.0)
- **test_linear_regression**: Verify slope, intercept, r_squared for known datasets
- **test_normal_pdf**: Verify standard normal PDF values, custom mu/sigma
- **test_normal_cdf**: Verify CDF(0, 0, 1) = 0.5, CDF(-inf) -> 0, CDF(inf) -> 1
- **test_binomial_pmf**: Test binomial(5, 10, 0.5), edge cases
- **test_binomial_cdf**: Test cumulative binomial probabilities
- **test_poisson_pmf**: Test Poisson(k=2, lambda=3), edge cases
- **test_poisson_cdf**: Test cumulative Poisson probabilities
- **test_t_test**: Test with known samples, verify t-statistic calculation
- **test_chi_square_test**: Test with known observed/expected values
- **test_bootstrap_sample**: Test returns correct length, values from original data
- **test_confidence_interval**: Test returns tuple of two floats, lower < mean < upper

### Edge Cases
- Empty dataset: All functions should raise ValueError("Data cannot be empty")
- Single element: mean/median/mode return that element, variance/std_dev return 0.0
- All same values: variance = 0, std_dev = 0, correlation undefined
- Percentile p = 0 and p = 100: return min and max respectively
- Negative values in data: should work correctly
- Very large/small numbers: test numerical stability
- Correlation with constant x or y: handle gracefully
- Binomial/Poisson with edge k=0, k=n: test boundary conditions

## Acceptance Criteria
- statistics.py module exists with all 20+ functions implemented
- All functions have proper type hints and docstrings
- All functions raise ValueError for empty datasets with message "Data cannot be empty"
- All functions handle edge cases: single element, all same values
- Results accurate to 6 decimal places (verified by tests)
- No scipy/statsmodels imports - pure Python with only math module
- All unit tests pass (test_statistics.py)
- All existing tests pass (test_calculator.py) - zero regressions
- Code passes ruff linter with zero warnings
- Main block demonstrates example usage matching issue specification
- Example outputs match: mean([2,4,4,4,5,5,7,9]) = 5.0, std_dev = 2.0

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python statistics.py` - Run demonstration script showing all statistics examples work correctly
- `python -c "from statistics import mean; print(mean([2,4,4,4,5,5,7,9]))"` - Verify mean = 5.0
- `python -c "from statistics import std_dev; print(std_dev([2,4,4,4,5,5,7,9]))"` - Verify std_dev = 2.0
- `python -c "from statistics import percentile; print(percentile([2,4,4,4,5,5,7,9], 75))"` - Verify percentile(75) = 6.0
- `python -c "from statistics import linear_regression; print(linear_regression([1,2,3,4,5], [2,4,5,4,5]))"` - Verify regression output
- `python -c "from statistics import normal_cdf; print(round(normal_cdf(0, 0, 1), 6))"` - Verify CDF(0,0,1) = 0.5
- `python -c "from statistics import mean; mean([])"` - Verify ValueError raised for empty data
- `uv run pytest test_statistics.py -v` - Run all statistics tests with zero failures
- `uv run pytest test_calculator.py -v` - Run all calculator tests with zero regressions
- `uv run ruff check statistics.py test_statistics.py calculator.py test_calculator.py` - Lint all code with zero warnings

## Notes
- Taylor series for erf converges quickly: 50 terms gives ~15 decimal places accuracy
- erf(x) formula: (2/sqrt(pi)) * sum from n=0 to inf of ((-1)^n * x^(2n+1)) / (n! * (2n+1))
- normal_cdf(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
- For t-test p-value approximation with large samples, use normal distribution; for small samples, could implement t-distribution CDF or use conservative approximation
- Chi-square p-value approximation: use incomplete gamma function or Wilson-Hilferty approximation
- Python's random module is acceptable for bootstrap sampling
- Module name "statistics" may shadow stdlib statistics - consider importing specific functions in tests to avoid conflicts
- Future enhancements could include: weighted statistics, moving averages, additional distributions, ANOVA
- The issue example shows std_dev([2,4,4,4,5,5,7,9]) = 2.0, which implies population std dev; verify this interpretation
- For percentile calculation, using linear interpolation method (same as numpy's default)
- Confidence interval assumes normal distribution of sample mean (Central Limit Theorem)
