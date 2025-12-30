# Statistics Module with Distributions

**ADW ID:** e06dcc14
**Date:** 2025-12-30
**Specification:** specs/issue-29-adw-e06dcc14-sdlc_planner-add-statistics-module.md

## Overview

A comprehensive pure Python statistics module implementing descriptive statistics, percentile calculations, correlation/regression analysis, probability distributions (normal, binomial, Poisson), hypothesis testing (t-test, chi-square), and sampling methods. All implementations use only the math module with no scipy/statsmodels dependencies, achieving 6 decimal place accuracy.

## What Was Built

- **Descriptive Statistics**: mean, median, mode, variance, std_dev
- **Percentile Functions**: percentile, quartiles, iqr (interquartile range)
- **Correlation/Regression**: covariance, correlation (Pearson), linear_regression (returns slope, intercept, r_squared)
- **Distribution Functions**: normal_pdf, normal_cdf, binomial_pmf, binomial_cdf, poisson_pmf, poisson_cdf
- **Hypothesis Testing**: t_test (two-sample), chi_square_test
- **Sampling Methods**: bootstrap_sample, confidence_interval
- **Helper Functions**: _erf (Taylor series), _factorial, _combinations, _gamma_incomplete_upper, _t_distribution_cdf

## Technical Implementation

### Files Modified

- `statistics.py`: New 1055-line module with all statistical functions
- `test_statistics.py`: New 507-line comprehensive test suite
- `uv.lock`: Updated with project dependencies

### Key Changes

- Implemented error function (_erf) using Taylor series with 100 terms for high precision
- Used incomplete gamma function approximation for chi-square p-values
- Applied Student's t-distribution CDF approximation for t-test p-values
- All functions validate input with `_validate_data()` raising ValueError for empty datasets
- Population vs sample variance supported via `population` parameter (default True)

## How to Use

1. Import functions from the statistics module:
   ```python
   from statistics import mean, std_dev, percentile, linear_regression
   ```

2. Calculate descriptive statistics:
   ```python
   data = [2, 4, 4, 4, 5, 5, 7, 9]
   print(mean(data))      # 5.0
   print(std_dev(data))   # 2.0
   print(median(data))    # 4.5
   ```

3. Calculate percentiles:
   ```python
   print(percentile(data, 75))  # 5.5
   q1, q2, q3 = quartiles(data)
   print(iqr(data))  # Q3 - Q1
   ```

4. Perform regression analysis:
   ```python
   x = [1, 2, 3, 4, 5]
   y = [2, 4, 5, 4, 5]
   slope, intercept, r2 = linear_regression(x, y)
   print(f'y = {slope:.2f}x + {intercept:.2f}, R^2={r2:.3f}')
   ```

5. Use distribution functions:
   ```python
   print(normal_cdf(0, 0, 1))  # 0.5 (standard normal at x=0)
   print(binomial_pmf(5, 10, 0.5))  # P(X=5) for Bin(10, 0.5)
   ```

6. Hypothesis testing:
   ```python
   t_stat, p_value = t_test(sample1, sample2)
   chi_stat, p_value = chi_square_test(observed, expected)
   ```

7. Sampling methods:
   ```python
   bootstrap = bootstrap_sample(data, 1000)
   lower, upper = confidence_interval(data, confidence=0.95)
   ```

## Configuration

No configuration required. All functions use sensible defaults:
- `variance(data, population=True)` - population variance by default
- `confidence_interval(data, confidence=0.95)` - 95% confidence by default
- Distribution functions default to standard parameters (mu=0, sigma=1 for normal)

## Testing

Run the test suite:
```bash
uv run pytest test_statistics.py -v
```

Tests cover:
- Normal operation for all functions
- Edge cases: single element, all same values
- Error handling: empty datasets raise ValueError
- Accuracy verification to 6 decimal places

## Notes

- Module name "statistics" shadows Python's stdlib statistics module - import specific functions or use alias
- For large samples, t-test uses normal approximation; for small samples uses t-distribution CDF
- Taylor series for erf converges within 100 terms for all practical inputs
- Linear interpolation method used for percentiles (numpy-compatible)
- Confidence intervals assume normal distribution via Central Limit Theorem
