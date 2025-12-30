# Statistics Module with Distributions

**ADW ID:** f9c30308
**Date:** 2025-12-30
**Specification:** specs/issue-29-adw-f9c30308-sdlc_planner-add-statistics-module.md

## Overview

A comprehensive pure Python statistics module providing descriptive statistics, percentile calculations, correlation analysis, probability distributions (normal, binomial, Poisson), hypothesis testing (t-test, chi-square), and sampling utilities. Implemented using only the built-in `math` module with Taylor series approximations for mathematical functions like the error function.

## What Was Built

- **Descriptive Statistics**: `mean`, `median`, `mode`, `variance`, `std_dev`
- **Percentile Functions**: `percentile`, `quartiles`, `iqr`
- **Correlation Analysis**: `covariance`, `correlation`, `linear_regression`
- **Normal Distribution**: `normal_pdf`, `normal_cdf`
- **Binomial Distribution**: `binomial_pmf`, `binomial_cdf`
- **Poisson Distribution**: `poisson_pmf`, `poisson_cdf`
- **Hypothesis Testing**: `t_test`, `chi_square_test`
- **Sampling Utilities**: `bootstrap_sample`, `confidence_interval`
- **Helper Functions**: `_validate_data`, `_factorial`, `_combinations`, `_erf`, `_gamma`, `_chi2_cdf`, `_t_cdf`, `_betainc`

## Technical Implementation

### Files Modified

- `statistics.py`: New 689-line module containing all statistical functions organized into logical sections
- `test_statistics.py`: New 531-line comprehensive test suite with 6 test classes covering all functionality
- `uv.lock`: Updated with any new dependencies
- `.ports.env`: Minor updates

### Key Changes

- **Pure Python Implementation**: Uses only `math` and `random` modules - no scipy, statsmodels, or numpy dependencies
- **Taylor Series Error Function**: Custom `_erf(x)` implementation using Taylor series for |x| <= 3.5 and asymptotic approximation for larger values
- **Gamma Function**: Lanczos approximation for gamma function, required for chi-square CDF calculations
- **Incomplete Beta Function**: Custom implementation for t-distribution CDF calculations
- **Robust Input Validation**: All functions validate for empty datasets and raise `ValueError` with descriptive messages
- **6 Decimal Place Accuracy**: Careful numerical implementation ensures results accurate to 6 decimal places

## How to Use

### Descriptive Statistics

```python
from statistics import mean, median, mode, variance, std_dev

data = [2, 4, 4, 4, 5, 5, 7, 9]
print(mean(data))      # 5.0
print(median(data))    # 4.5
print(mode(data))      # 4
print(variance(data))  # 4.0 (population variance)
print(std_dev(data))   # 2.0 (population std dev)
```

### Percentiles

```python
from statistics import percentile, quartiles, iqr

data = [2, 4, 4, 4, 5, 5, 7, 9]
print(percentile(data, 75))  # ~6.0
print(quartiles(data))       # (Q1, Q2, Q3)
print(iqr(data))             # Q3 - Q1
```

### Correlation and Regression

```python
from statistics import correlation, covariance, linear_regression

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
print(correlation(x, y))        # Pearson r
print(covariance(x, y))         # Population covariance
slope, intercept, r_squared = linear_regression(x, y)
print(f"y = {slope:.2f}x + {intercept:.2f}, R^2 = {r_squared:.3f}")
```

### Probability Distributions

```python
from statistics import normal_pdf, normal_cdf, binomial_pmf, poisson_pmf

# Normal distribution
print(normal_pdf(0, mu=0, sigma=1))   # PDF at x=0
print(normal_cdf(0, mu=0, sigma=1))   # 0.5 (CDF at mean)

# Binomial distribution
print(binomial_pmf(k=3, n=10, p=0.5)) # P(X=3) for n=10, p=0.5

# Poisson distribution
print(poisson_pmf(k=5, lambda_=3))    # P(X=5) for lambda=3
```

### Hypothesis Testing

```python
from statistics import t_test, chi_square_test

sample1 = [23, 25, 28, 30, 32]
sample2 = [20, 22, 24, 26, 28]
t_stat, p_value = t_test(sample1, sample2)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

observed = [10, 20, 30]
expected = [15, 20, 25]
chi2, p_val = chi_square_test(observed, expected)
```

### Sampling

```python
from statistics import bootstrap_sample, confidence_interval

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sample = bootstrap_sample(data, n=100)  # 100 samples with replacement
lower, upper = confidence_interval(data, confidence=0.95)
print(f"95% CI: ({lower:.2f}, {upper:.2f})")
```

## Configuration

No configuration required. The module uses only standard library dependencies (`math`, `random`).

## Testing

Run the test suite:

```bash
uv run pytest test_statistics.py -v
```

The test suite includes:
- `TestDescriptiveStats`: Tests for mean, median, mode, variance, std_dev
- `TestPercentiles`: Tests for percentile, quartiles, iqr
- `TestCorrelation`: Tests for covariance, correlation, linear_regression
- `TestDistributions`: Tests for normal, binomial, and Poisson distributions
- `TestHypothesisTesting`: Tests for t_test and chi_square_test
- `TestSampling`: Tests for bootstrap_sample and confidence_interval

Verify specific values:
```bash
python -c "from statistics import mean; print(mean([2,4,4,4,5,5,7,9]))"  # 5.0
python -c "from statistics import std_dev; print(std_dev([2,4,4,4,5,5,7,9]))"  # 2.0
python -c "from statistics import normal_cdf; print(round(normal_cdf(0, 0, 1), 6))"  # 0.5
```

## Notes

- **Module Name Shadowing**: The module name `statistics` shadows Python's built-in `statistics` module. Use explicit imports from this module.
- **Error Function Approximation**: Uses Taylor series for |x| <= 3.5 and asymptotic expansion for larger values to ensure accuracy
- **t-Distribution Approximation**: Uses normal approximation for df > 100, incomplete beta function for smaller degrees of freedom
- **Sample vs Population Variance**: Both `variance()` and `std_dev()` accept a `population` parameter (default `True` for population, `False` for sample)
- **Future Enhancements**: Consider adding skewness, kurtosis, and Shapiro-Wilk normality test
