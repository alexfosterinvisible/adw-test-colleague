# Feature: Add Statistics Module with Distributions

## Metadata
issue_number: `29`
adw_id: `f9c30308`
issue_json: `{"number":29,"title":"Add statistics module with distributions","body":"Implement a statistics module with:\n\n1. **Descriptive stats**: `mean(data)`, `median(data)`, `mode(data)`, `variance(data)`, `std_dev(data)`\n2. **Percentiles**: `percentile(data, p)`, `quartiles(data)`, `iqr(data)`\n3. **Correlation**: `correlation(x, y)`, `covariance(x, y)`, `linear_regression(x, y)` returns (slope, intercept, r_squared)\n4. **Distributions**:\n   - `normal_pdf(x, mu, sigma)`, `normal_cdf(x, mu, sigma)`\n   - `binomial_pmf(k, n, p)`, `binomial_cdf(k, n, p)`\n   - `poisson_pmf(k, lambda_)`, `poisson_cdf(k, lambda_)`\n5. **Hypothesis testing**: `t_test(sample1, sample2)` returns (t_stat, p_value), `chi_square_test(observed, expected)`\n6. **Sampling**: `bootstrap_sample(data, n)`, `confidence_interval(data, confidence=0.95)`\n\nRequirements:\n- No scipy/statsmodels - pure Python (math module OK)\n- Use Taylor series for erf/normal CDF approximation\n- Raise `ValueError` for empty datasets\n- Handle edge cases: single element, all same values\n- Results accurate to 6 decimal places"}`

## Feature Description
Implement a comprehensive statistics module providing pure Python implementations of statistical functions. The module will include descriptive statistics (mean, median, mode, variance, standard deviation), percentile calculations (percentile, quartiles, interquartile range), correlation analysis (correlation, covariance, linear regression), probability distributions (normal, binomial, Poisson), hypothesis testing (t-test, chi-square test), and sampling utilities (bootstrap sampling, confidence intervals). All implementations must be pure Python using only the math module, with Taylor series approximations for mathematical functions like the error function (erf) for normal CDF calculations.

## User Story
As a calculator/data analysis user
I want to perform statistical computations on datasets
So that I can analyze data, understand distributions, test hypotheses, and make data-driven decisions without external dependencies

## Problem Statement
The calculator currently only supports basic arithmetic operations. Users need the ability to perform statistical analysis including computing descriptive statistics, analyzing correlations, working with probability distributions, and performing hypothesis tests. These are fundamental operations required in data analysis, research, and scientific computing.

## Solution Statement
Create a new `statistics.py` module containing pure Python implementations of all required statistical functions. The implementation will:
- Use only the built-in `math` module (no scipy, statsmodels, numpy)
- Implement Taylor series approximations for the error function (erf) to calculate normal CDF
- Use iterative/recursive implementations for factorial and combinations
- Raise `ValueError` for invalid inputs (empty datasets, negative parameters where inappropriate)
- Handle edge cases: single-element datasets, all-same-values datasets, zero variance
- Achieve 6 decimal place accuracy through careful numerical implementation

## Relevant Files
Use these files to implement the feature:

- **calculator.py** - Reference for existing function patterns, type hints, error handling conventions, and main block demonstration structure. The statistics module will follow the same patterns established here.

- **test_calculator.py** - Reference for test structure and patterns using unittest framework. The statistics tests will follow the same TestClass pattern with comprehensive test methods.

- **pyproject.toml** - Project configuration. No changes needed as we're using pure Python with only the math module.

- **app_docs/feature-de3d954e-divide-function.md** - Reference documentation showing the established pattern for feature documentation including code examples, error handling, and testing sections.

- **.adw.yaml** - ADW configuration specifying test_command: "uv run pytest". Used for validation commands.

- **specs/issue-3-adw-bf54fc79-sdlc_planner-add-sqrt-function.md** - Reference for plan format and detail level expected.

### New Files
- **statistics.py** - New statistics module containing all statistical functions organized into logical sections: descriptive stats, percentiles, correlation, distributions, hypothesis testing, and sampling.

- **test_statistics.py** - Comprehensive test suite for all statistical functions following the unittest pattern established in test_calculator.py. Will test normal operations, edge cases, and error handling.

## Implementation Plan
### Phase 1: Foundation
Set up the statistics.py module structure with proper imports (math module only). Implement helper functions needed across multiple features:
- `_validate_data(data)` - Check for empty datasets and raise ValueError
- `_factorial(n)` - Iterative factorial for combinations/permutations
- `_combinations(n, k)` - Binomial coefficient calculation
- `_erf(x)` - Taylor series approximation of error function for normal CDF

### Phase 2: Core Implementation
Implement statistical functions in logical groups:
1. Descriptive statistics: mean, median, mode, variance, std_dev
2. Percentiles: percentile, quartiles, iqr
3. Correlation: covariance, correlation, linear_regression
4. Distributions: normal_pdf/cdf, binomial_pmf/cdf, poisson_pmf/cdf
5. Hypothesis testing: t_test, chi_square_test
6. Sampling: bootstrap_sample, confidence_interval

### Phase 3: Integration
Add comprehensive unit tests for all functions. Add main block demonstration showing example usage. Verify accuracy to 6 decimal places against known values. Run all tests to ensure no regressions.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Task 1: Create Statistics Module Foundation
- Create statistics.py with module docstring
- Add `from math import sqrt, exp, pi, log, factorial as math_factorial` imports
- Implement `_validate_data(data: list) -> None` helper that raises ValueError("Data cannot be empty") for empty lists
- Implement `_factorial(n: int) -> int` using iterative approach (for use in combinations)
- Implement `_combinations(n: int, k: int) -> int` for binomial coefficient (n choose k)
- Implement `_erf(x: float) -> float` using Taylor series approximation for error function

### Task 2: Implement Descriptive Statistics
- Implement `mean(data: list[float]) -> float` - arithmetic mean with empty data validation
- Implement `median(data: list[float]) -> float` - middle value (average of two middle for even length)
- Implement `mode(data: list[float]) -> float` - most frequent value (first if tie)
- Implement `variance(data: list[float], population: bool = True) -> float` - population variance by default, sample variance when population=False
- Implement `std_dev(data: list[float], population: bool = True) -> float` - standard deviation (sqrt of variance)

### Task 3: Implement Percentile Functions
- Implement `percentile(data: list[float], p: float) -> float` - p-th percentile (0-100 scale), using linear interpolation
- Implement `quartiles(data: list[float]) -> tuple[float, float, float]` - returns (Q1, Q2, Q3)
- Implement `iqr(data: list[float]) -> float` - interquartile range (Q3 - Q1)

### Task 4: Implement Correlation and Regression
- Implement `covariance(x: list[float], y: list[float]) -> float` - population covariance with length validation
- Implement `correlation(x: list[float], y: list[float]) -> float` - Pearson correlation coefficient
- Implement `linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]` - returns (slope, intercept, r_squared)

### Task 5: Implement Normal Distribution
- Implement `normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float` - probability density function
- Implement `normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float` - cumulative distribution function using erf approximation

### Task 6: Implement Binomial Distribution
- Implement `binomial_pmf(k: int, n: int, p: float) -> float` - probability mass function
- Implement `binomial_cdf(k: int, n: int, p: float) -> float` - cumulative distribution function (sum of pmf from 0 to k)

### Task 7: Implement Poisson Distribution
- Implement `poisson_pmf(k: int, lambda_: float) -> float` - probability mass function
- Implement `poisson_cdf(k: int, lambda_: float) -> float` - cumulative distribution function

### Task 8: Implement Hypothesis Testing
- Implement `t_test(sample1: list[float], sample2: list[float]) -> tuple[float, float]` - two-sample t-test returning (t_statistic, p_value)
- Implement `chi_square_test(observed: list[float], expected: list[float]) -> tuple[float, float]` - chi-square test returning (chi2_statistic, p_value)

### Task 9: Implement Sampling Functions
- Implement `bootstrap_sample(data: list[float], n: int) -> list[float]` - random sampling with replacement
- Implement `confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]` - confidence interval for mean

### Task 10: Add Main Block Demonstration
- Add `if __name__ == "__main__":` block with examples:
  - Descriptive stats: mean, std_dev of sample data
  - Percentile calculation
  - Linear regression with R-squared
  - Normal distribution values
  - Bootstrap sample

### Task 11: Create Test File Foundation
- Create test_statistics.py with unittest imports
- Import all functions from statistics module
- Create TestDescriptiveStats class
- Create TestPercentiles class
- Create TestCorrelation class
- Create TestDistributions class
- Create TestHypothesisTesting class
- Create TestSampling class

### Task 12: Implement Descriptive Statistics Tests
- Test mean with positive, negative, mixed values
- Test median with odd and even length lists
- Test mode with single mode, multiple modes (returns first), all unique
- Test variance with known values, single element (should be 0)
- Test std_dev verifying it equals sqrt(variance)
- Test empty data raises ValueError for all functions

### Task 13: Implement Percentile Tests
- Test percentile with p=0, 25, 50, 75, 100
- Test quartiles returns correct Q1, Q2, Q3
- Test iqr equals Q3 - Q1
- Test edge cases: single element, all same values

### Task 14: Implement Correlation Tests
- Test covariance with known values
- Test correlation with perfect positive (1.0), perfect negative (-1.0), no correlation (~0)
- Test linear_regression slope, intercept, r_squared with known data
- Test mismatched length raises ValueError

### Task 15: Implement Distribution Tests
- Test normal_pdf at mean (maximum), symmetric points
- Test normal_cdf at mu (should be 0.5), known z-scores
- Test binomial_pmf for known probabilities
- Test binomial_cdf cumulative property
- Test poisson_pmf against known values
- Test poisson_cdf cumulative property

### Task 16: Implement Hypothesis Testing Tests
- Test t_test with identical samples (t_stat ~0)
- Test t_test with different samples
- Test chi_square_test with uniform observed/expected (low chi2)
- Test chi_square_test with mismatched arrays raises ValueError

### Task 17: Implement Sampling Tests
- Test bootstrap_sample returns correct length
- Test bootstrap_sample values are from original data
- Test confidence_interval returns (lower, upper) tuple
- Test confidence_interval contains mean for large samples

### Task 18: Run Validation Commands
- Execute: `uv run pytest test_statistics.py -v` - validate all statistics tests pass
- Execute: `uv run pytest test_calculator.py -v` - validate existing tests still pass (no regressions)
- Execute: `python statistics.py` - validate main block demonstration runs
- Execute: `python -c "from statistics import mean; print(mean([2,4,4,4,5,5,7,9]))"` - verify outputs 5.0
- Execute: `python -c "from statistics import std_dev; print(std_dev([2,4,4,4,5,5,7,9]))"` - verify outputs 2.0
- Execute: `uv run ruff check statistics.py test_statistics.py` - lint code with zero warnings

## Testing Strategy
### Unit Tests
- **TestDescriptiveStats**: Tests mean, median, mode, variance, std_dev with various inputs, verifying accuracy to 6 decimal places
- **TestPercentiles**: Tests percentile, quartiles, iqr with edge cases including single elements and all-same values
- **TestCorrelation**: Tests covariance, correlation, linear_regression with known mathematical results
- **TestDistributions**: Tests PDF/PMF and CDF functions against known probability values
- **TestHypothesisTesting**: Tests t_test and chi_square_test statistical properties
- **TestSampling**: Tests bootstrap_sample randomness and confidence_interval bounds

### Edge Cases
- Empty datasets (all functions should raise ValueError)
- Single element datasets (variance=0, std_dev=0, etc.)
- All identical values (variance=0, correlation undefined)
- Negative numbers in appropriate contexts
- Large datasets for numerical stability
- Perfect correlation (r=1.0 or r=-1.0)
- Zero variance handling in correlation/regression
- Percentile boundary values (0 and 100)
- Distribution parameter validation (n, p, lambda_ must be valid)

## Acceptance Criteria
- statistics.py module exists with all required functions
- All functions have proper type hints
- All functions use only math module (no scipy, statsmodels, numpy)
- Empty datasets raise ValueError("Data cannot be empty")
- Normal CDF uses Taylor series erf approximation
- Results accurate to 6 decimal places
- mean([2,4,4,4,5,5,7,9]) returns 5.0
- std_dev([2,4,4,4,5,5,7,9]) returns 2.0 (population std dev)
- percentile([2,4,4,4,5,5,7,9], 75) returns approximately 6.0
- linear_regression returns (slope, intercept, r_squared) tuple
- All unit tests pass with zero failures
- Existing calculator tests pass (no regressions)
- Code passes ruff linter checks
- Main block demonstrates key functionality

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python statistics.py` - Run demonstration script showing statistics examples work correctly
- `python -c "from statistics import mean; print(mean([2,4,4,4,5,5,7,9]))"` - Verify mean returns 5.0
- `python -c "from statistics import std_dev; print(std_dev([2,4,4,4,5,5,7,9]))"` - Verify std_dev returns 2.0
- `python -c "from statistics import percentile; print(percentile([2,4,4,4,5,5,7,9], 75))"` - Verify percentile returns ~6.0
- `python -c "from statistics import linear_regression; x=[1,2,3,4,5]; y=[2,4,5,4,5]; s,i,r=linear_regression(x,y); print(f'slope={s:.2f}, intercept={i:.2f}, r2={r:.3f}')"` - Verify regression output
- `python -c "from statistics import normal_cdf; print(round(normal_cdf(0, 0, 1), 6))"` - Verify normal_cdf(0,0,1) returns 0.5
- `python -c "from statistics import mean; mean([])"` - Verify ValueError raised for empty data
- `uv run pytest test_statistics.py -v` - Run all statistics unit tests with zero failures
- `uv run pytest test_calculator.py -v` - Run calculator tests with zero regressions
- `uv run ruff check statistics.py test_statistics.py` - Lint code with zero warnings

## Notes
- The error function (erf) Taylor series approximation: erf(x) = (2/sqrt(pi)) * sum((-1)^n * x^(2n+1) / (n! * (2n+1))) converges well for |x| < 3
- For larger x values, use asymptotic approximation: erf(x) ≈ 1 - exp(-x^2) / (x * sqrt(pi)) for x > 3
- Normal CDF formula: CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
- t-distribution p-value approximation uses normal distribution for large sample sizes (n > 30)
- For chi-square p-value, use incomplete gamma function approximation
- Bootstrap sampling uses random.choices for sampling with replacement
- Confidence interval uses t-distribution critical values approximated for common confidence levels
- The module name "statistics" shadows the built-in module - use explicit imports
- All float comparisons in tests should use assertAlmostEqual with places=6
- Future enhancement: Consider adding skewness, kurtosis, Shapiro-Wilk test for normality
