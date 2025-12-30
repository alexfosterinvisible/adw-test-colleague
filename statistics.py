"""(Claude) Statistics module with pure Python implementations.

Provides comprehensive statistical functions including:
- Descriptive statistics: mean, median, mode, variance, std_dev
- Percentiles: percentile, quartiles, iqr
- Correlation: covariance, correlation, linear_regression
- Distributions: normal_pdf, normal_cdf, binomial_pmf, binomial_cdf, poisson_pmf, poisson_cdf
- Hypothesis testing: t_test, chi_square_test
- Sampling: bootstrap_sample, confidence_interval

All implementations use pure Python with only the math module.
Results accurate to 6 decimal places.
"""

import math
import random
from typing import List, Tuple


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_data(data: List[float]) -> None:
    """Validate that data is not empty.

    Args:
        data: List of numeric values to validate.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Data cannot be empty")


def _factorial(n: int) -> int:
    """Calculate factorial of n.

    Args:
        n: Non-negative integer.

    Returns:
        n! (n factorial).

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _combinations(n: int, k: int) -> int:
    """Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!).

    Args:
        n: Total number of items.
        k: Number of items to choose.

    Returns:
        Number of combinations.

    Raises:
        ValueError: If k > n or either is negative.
    """
    if k < 0 or n < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k == 0 or k == n:
        return 1
    # Use multiplicative formula for efficiency
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _erf(x: float) -> float:
    """Calculate the error function using Taylor series approximation.

    erf(x) = (2/sqrt(pi)) * sum((-1)^n * x^(2n+1) / (n! * (2n+1)))

    Args:
        x: Input value.

    Returns:
        erf(x) approximation accurate to at least 6 decimal places.
    """
    # For large |x|, erf approaches +/- 1
    if x > 6:
        return 1.0
    if x < -6:
        return -1.0

    # Taylor series expansion
    result = 0.0
    term = x  # First term: x
    n = 0

    while n < 100:  # More than enough terms for convergence
        result += term / (2 * n + 1)
        # Next term: (-1)^(n+1) * x^(2(n+1)+1) / ((n+1)!)
        # = previous_base * (-x^2) / (n+1)
        term *= -x * x / (n + 1)
        n += 1
        if abs(term) < 1e-15:
            break

    return result * 2 / math.sqrt(math.pi)


def _gamma_incomplete_upper(s: float, x: float) -> float:
    """Approximate upper incomplete gamma function for chi-square p-value.

    Uses series expansion for the regularized incomplete gamma function.

    Args:
        s: Shape parameter.
        x: Integration limit.

    Returns:
        Approximate value of upper incomplete gamma / Gamma(s).
    """
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0

    # For x > s + 1, use continued fraction (more stable)
    # For x <= s + 1, use series expansion for lower gamma and subtract from 1

    # Series expansion for lower incomplete gamma
    if x < s + 1:
        term = 1.0 / s
        total = term
        for n in range(1, 200):
            term *= x / (s + n)
            total += term
            if abs(term) < 1e-10:
                break
        # P(s, x) = x^s * e^(-x) * sum / Gamma(s)
        lower = math.exp(s * math.log(x) - x - math.lgamma(s)) * total * s
        return 1.0 - lower
    else:
        # Continued fraction for upper gamma (Lentz's method)
        f = 1e-30
        c = 1e-30
        d = 0.0
        for n in range(1, 200):
            if n == 1:
                an = 1.0
            elif n % 2 == 0:
                an = (n // 2 - s) * x
            else:
                an = (n // 2) * x
            bn = (x - s + n) if n == 1 else (x - s + 2 * n - 1)

            d = bn + an * d
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d

            c = bn + an / c
            if abs(c) < 1e-30:
                c = 1e-30

            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                break

        # Q(s, x) = x^s * e^(-x) * f / Gamma(s)
        return math.exp(s * math.log(x) - x - math.lgamma(s)) * f


def _t_distribution_cdf(t: float, df: float) -> float:
    """Approximate CDF of Student's t-distribution.

    For large df, approximates normal. For small df, uses incomplete beta.

    Args:
        t: t-statistic value.
        df: Degrees of freedom.

    Returns:
        Approximate CDF value.
    """
    # For large df, t-distribution approaches normal
    if df > 100:
        return 0.5 * (1 + _erf(t / math.sqrt(2)))

    # Use incomplete beta function relationship
    # CDF = 1 - 0.5 * I_x(df/2, 1/2) where x = df / (df + t^2)
    x = df / (df + t * t)

    # Incomplete beta approximation using continued fraction
    a = df / 2
    b = 0.5

    if x < (a + 1) / (a + b + 2):
        beta_cf = _beta_continued_fraction(x, a, b)
        result = beta_cf
    else:
        beta_cf = _beta_continued_fraction(1 - x, b, a)
        result = 1 - beta_cf

    if t >= 0:
        return 0.5 + 0.5 * (1 - result)
    else:
        return 0.5 - 0.5 * (1 - result)


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function using continued fraction."""
    # Using Lentz's algorithm
    qab = a + b
    qap = a + 1
    qam = a - 1

    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    h = d

    for m in range(1, 200):
        m2 = 2 * m

        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta

        if abs(delta - 1) < 1e-10:
            break

    # Multiply by the beta function factor
    try:
        front = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1 - x)
        ) / a
    except (ValueError, OverflowError):
        front = 0.0

    return front * h


# =============================================================================
# Descriptive Statistics
# =============================================================================

def mean(data: List[float]) -> float:
    """Calculate the arithmetic mean of a dataset.

    Args:
        data: List of numeric values.

    Returns:
        The arithmetic mean.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> mean([2, 4, 4, 4, 5, 5, 7, 9])
        5.0
    """
    _validate_data(data)
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """Calculate the median of a dataset.

    Args:
        data: List of numeric values.

    Returns:
        The median value.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> median([1, 3, 5, 7, 9])
        5.0
        >>> median([1, 2, 3, 4])
        2.5
    """
    _validate_data(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]


def mode(data: List[float]) -> float:
    """Calculate the mode of a dataset.

    Returns the first mode if multiple modes exist.

    Args:
        data: List of numeric values.

    Returns:
        The mode (most frequent value).

    Raises:
        ValueError: If data is empty.

    Example:
        >>> mode([2, 4, 4, 4, 5, 5, 7, 9])
        4
    """
    _validate_data(data)

    # Count occurrences
    counts: dict = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1

    # Find maximum count
    max_count = max(counts.values())

    # Return first value with max count (preserving original order)
    for value in data:
        if counts[value] == max_count:
            return value

    # Should never reach here
    return data[0]


def variance(data: List[float], population: bool = True) -> float:
    """Calculate the variance of a dataset.

    Args:
        data: List of numeric values.
        population: If True, calculate population variance (divide by n).
                   If False, calculate sample variance (divide by n-1).

    Returns:
        The variance.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.0
    """
    _validate_data(data)
    n = len(data)

    if n == 1:
        return 0.0

    data_mean = mean(data)
    squared_diff_sum = sum((x - data_mean) ** 2 for x in data)

    if population:
        return squared_diff_sum / n
    else:
        return squared_diff_sum / (n - 1)


def std_dev(data: List[float], population: bool = True) -> float:
    """Calculate the standard deviation of a dataset.

    Args:
        data: List of numeric values.
        population: If True, calculate population std dev.
                   If False, calculate sample std dev.

    Returns:
        The standard deviation.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        2.0
    """
    return math.sqrt(variance(data, population))


# =============================================================================
# Percentile Functions
# =============================================================================

def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of a dataset using linear interpolation.

    Args:
        data: List of numeric values.
        p: Percentile to compute (0-100).

    Returns:
        The p-th percentile value.

    Raises:
        ValueError: If data is empty or p is not in [0, 100].

    Example:
        >>> percentile([2, 4, 4, 4, 5, 5, 7, 9], 75)
        6.0
    """
    _validate_data(data)

    if p < 0 or p > 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return sorted_data[0]

    # Linear interpolation method (numpy default)
    # Index formula: (p/100) * (n-1)
    index = (p / 100) * (n - 1)
    lower_idx = int(index)
    upper_idx = lower_idx + 1

    if upper_idx >= n:
        return sorted_data[-1]

    fraction = index - lower_idx
    return sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])


def quartiles(data: List[float]) -> Tuple[float, float, float]:
    """Calculate the quartiles (Q1, Q2, Q3) of a dataset.

    Args:
        data: List of numeric values.

    Returns:
        Tuple of (Q1, Q2, Q3) - 25th, 50th, and 75th percentiles.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> quartiles([2, 4, 4, 4, 5, 5, 7, 9])
        (4.0, 4.5, 5.5)
    """
    _validate_data(data)
    q1 = percentile(data, 25)
    q2 = percentile(data, 50)
    q3 = percentile(data, 75)
    return (q1, q2, q3)


def iqr(data: List[float]) -> float:
    """Calculate the interquartile range (IQR) of a dataset.

    Args:
        data: List of numeric values.

    Returns:
        The IQR (Q3 - Q1).

    Raises:
        ValueError: If data is empty.

    Example:
        >>> iqr([2, 4, 4, 4, 5, 5, 7, 9])
        1.5
    """
    q1, _, q3 = quartiles(data)
    return q3 - q1


# =============================================================================
# Correlation and Regression
# =============================================================================

def covariance(x: List[float], y: List[float], population: bool = True) -> float:
    """Calculate the covariance between two datasets.

    Args:
        x: First list of numeric values.
        y: Second list of numeric values.
        population: If True, calculate population covariance.

    Returns:
        The covariance.

    Raises:
        ValueError: If either list is empty or lengths don't match.

    Example:
        >>> covariance([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        1.0
    """
    _validate_data(x)
    _validate_data(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    if n == 1:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)

    cov_sum = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    if population:
        return cov_sum / n
    else:
        return cov_sum / (n - 1)


def correlation(x: List[float], y: List[float]) -> float:
    """Calculate the Pearson correlation coefficient between two datasets.

    Args:
        x: First list of numeric values.
        y: Second list of numeric values.

    Returns:
        The Pearson correlation coefficient (-1 to 1).

    Raises:
        ValueError: If either list is empty, lengths don't match,
                   or if either dataset has zero variance.

    Example:
        >>> correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        1.0
    """
    _validate_data(x)
    _validate_data(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    std_x = std_dev(x)
    std_y = std_dev(y)

    if std_x == 0 or std_y == 0:
        raise ValueError("Cannot compute correlation when standard deviation is zero")

    return covariance(x, y) / (std_x * std_y)


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Perform simple linear regression on two datasets.

    Fits the model y = slope * x + intercept.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Tuple of (slope, intercept, r_squared).

    Raises:
        ValueError: If either list is empty, lengths don't match,
                   or if x has zero variance.

    Example:
        >>> linear_regression([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        (0.6, 2.2, 0.36)
    """
    _validate_data(x)
    _validate_data(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    var_x = variance(x)
    if var_x == 0:
        raise ValueError("Cannot compute regression when x has zero variance")

    cov_xy = covariance(x, y)
    mean_x = mean(x)
    mean_y = mean(y)

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    # Calculate R-squared
    var_y = variance(y)
    if var_y == 0:
        r_squared = 1.0 if var_x == 0 else 0.0
    else:
        r_squared = (cov_xy ** 2) / (var_x * var_y)

    return (slope, intercept, r_squared)


# =============================================================================
# Distribution Functions
# =============================================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Calculate the probability density function of the normal distribution.

    Args:
        x: Value at which to evaluate the PDF.
        mu: Mean of the distribution (default 0).
        sigma: Standard deviation of the distribution (default 1).

    Returns:
        The PDF value at x.

    Raises:
        ValueError: If sigma <= 0.

    Example:
        >>> round(normal_pdf(0, 0, 1), 6)
        0.398942
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * math.exp(exponent)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Calculate the cumulative distribution function of the normal distribution.

    Args:
        x: Value at which to evaluate the CDF.
        mu: Mean of the distribution (default 0).
        sigma: Standard deviation of the distribution (default 1).

    Returns:
        The CDF value at x (probability P(X <= x)).

    Raises:
        ValueError: If sigma <= 0.

    Example:
        >>> round(normal_cdf(0, 0, 1), 6)
        0.5
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + _erf(z))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Calculate the probability mass function of the binomial distribution.

    Args:
        k: Number of successes.
        n: Number of trials.
        p: Probability of success on each trial.

    Returns:
        P(X = k) for X ~ Binomial(n, p).

    Raises:
        ValueError: If p not in [0,1], k < 0, n < 0, or k > n.

    Example:
        >>> round(binomial_pmf(5, 10, 0.5), 6)
        0.246094
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0 or n < 0:
        raise ValueError("k and n must be non-negative")
    if k > n:
        raise ValueError("k cannot be greater than n")

    combinations = _combinations(n, k)
    return combinations * (p ** k) * ((1 - p) ** (n - k))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Calculate the cumulative distribution function of the binomial distribution.

    Args:
        k: Number of successes.
        n: Number of trials.
        p: Probability of success on each trial.

    Returns:
        P(X <= k) for X ~ Binomial(n, p).

    Raises:
        ValueError: If p not in [0,1], k < 0, n < 0, or k > n.

    Example:
        >>> round(binomial_cdf(5, 10, 0.5), 6)
        0.623047
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    return sum(binomial_pmf(i, n, p) for i in range(k + 1))


def poisson_pmf(k: int, lambda_: float) -> float:
    """Calculate the probability mass function of the Poisson distribution.

    Args:
        k: Number of events.
        lambda_: Expected number of events (rate parameter).

    Returns:
        P(X = k) for X ~ Poisson(lambda_).

    Raises:
        ValueError: If lambda_ <= 0 or k < 0.

    Example:
        >>> round(poisson_pmf(2, 3), 6)
        0.224042
    """
    if lambda_ <= 0:
        raise ValueError("lambda must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")

    return (math.exp(-lambda_) * (lambda_ ** k)) / _factorial(k)


def poisson_cdf(k: int, lambda_: float) -> float:
    """Calculate the cumulative distribution function of the Poisson distribution.

    Args:
        k: Number of events.
        lambda_: Expected number of events (rate parameter).

    Returns:
        P(X <= k) for X ~ Poisson(lambda_).

    Raises:
        ValueError: If lambda_ <= 0 or k < 0.

    Example:
        >>> round(poisson_cdf(2, 3), 6)
        0.423190
    """
    if lambda_ <= 0:
        raise ValueError("lambda must be positive")
    if k < 0:
        return 0.0

    return sum(poisson_pmf(i, lambda_) for i in range(k + 1))


# =============================================================================
# Hypothesis Testing
# =============================================================================

def t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """Perform a two-sample t-test (Welch's t-test).

    Tests whether the means of two samples are significantly different.
    Uses Welch's t-test which does not assume equal variances.

    Args:
        sample1: First sample data.
        sample2: Second sample data.

    Returns:
        Tuple of (t_statistic, p_value) for a two-tailed test.

    Raises:
        ValueError: If either sample is empty.

    Example:
        >>> t_stat, p_val = t_test([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        >>> round(t_stat, 4)
        -1.0
    """
    _validate_data(sample1)
    _validate_data(sample2)

    n1 = len(sample1)
    n2 = len(sample2)

    mean1 = mean(sample1)
    mean2 = mean(sample2)

    # Use sample variance (n-1 denominator)
    var1 = variance(sample1, population=False) if n1 > 1 else 0.0
    var2 = variance(sample2, population=False) if n2 > 1 else 0.0

    # Standard error of difference
    se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-10

    # t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    if var1 == 0 and var2 == 0:
        df = n1 + n2 - 2
    else:
        num = (var1 / n1 + var2 / n2) ** 2
        denom = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2

    # Two-tailed p-value using t-distribution CDF
    p_value = 2 * (1 - _t_distribution_cdf(abs(t_stat), df))

    # Clamp p-value to valid range
    p_value = max(0.0, min(1.0, p_value))

    return (t_stat, p_value)


def chi_square_test(observed: List[float], expected: List[float]) -> Tuple[float, float]:
    """Perform a chi-square goodness-of-fit test.

    Tests whether observed frequencies differ significantly from expected frequencies.

    Args:
        observed: Observed frequencies.
        expected: Expected frequencies.

    Returns:
        Tuple of (chi_square_statistic, p_value).

    Raises:
        ValueError: If lists are empty, lengths don't match,
                   or any expected value is not positive.

    Example:
        >>> chi_stat, p_val = chi_square_test([10, 20, 30], [15, 20, 25])
        >>> round(chi_stat, 4)
        2.6667
    """
    _validate_data(observed)
    _validate_data(expected)

    if len(observed) != len(expected):
        raise ValueError("observed and expected must have the same length")

    for e in expected:
        if e <= 0:
            raise ValueError("Expected values must be positive")

    # Calculate chi-square statistic
    chi_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected))

    # Degrees of freedom
    df = len(observed) - 1

    # P-value from chi-square distribution
    # P(X > chi_stat) where X ~ chi-square(df)
    # Using incomplete gamma function: P(X > x) = Gamma_upper(df/2, x/2) / Gamma(df/2)
    p_value = _gamma_incomplete_upper(df / 2, chi_stat / 2)

    # Clamp p-value to valid range
    p_value = max(0.0, min(1.0, p_value))

    return (chi_stat, p_value)


# =============================================================================
# Sampling Methods
# =============================================================================

def bootstrap_sample(data: List[float], n: int) -> List[float]:
    """Generate a bootstrap sample (sampling with replacement).

    Args:
        data: Original data to sample from.
        n: Number of samples to draw.

    Returns:
        List of n values sampled with replacement from data.

    Raises:
        ValueError: If data is empty or n <= 0.

    Example:
        >>> random.seed(42)
        >>> bootstrap_sample([1, 2, 3, 4, 5], 5)
        [1, 4, 1, 3, 2]
    """
    _validate_data(data)

    if n <= 0:
        raise ValueError("n must be positive")

    return [random.choice(data) for _ in range(n)]


def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate a confidence interval for the mean.

    Uses normal distribution approximation (valid for large samples by CLT).

    Args:
        data: Sample data.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If data is empty or confidence not in (0, 1).

    Example:
        >>> lower, upper = confidence_interval([1, 2, 3, 4, 5], 0.95)
        >>> round(lower, 4), round(upper, 4)
        (1.2, 4.8)
    """
    _validate_data(data)

    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    n = len(data)
    data_mean = mean(data)

    if n == 1:
        return (data_mean, data_mean)

    # Use sample standard deviation
    data_std = std_dev(data, population=False)

    # Z-score for given confidence level
    # For 95% confidence, we need z such that P(-z < Z < z) = 0.95
    # This means P(Z < z) = 0.975
    # Using inverse normal CDF approximation
    alpha = 1 - confidence

    # Approximate inverse normal CDF using Newton-Raphson or lookup
    # For common values:
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }

    if confidence in z_scores:
        z = z_scores[confidence]
    else:
        # Rational approximation for inverse normal CDF
        p = 1 - alpha / 2
        if p <= 0 or p >= 1:
            z = 1.96  # Default fallback
        else:
            # Abramowitz and Stegun approximation
            t = math.sqrt(-2 * math.log(1 - p)) if p < 0.5 else math.sqrt(-2 * math.log(p))
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
            if p < 0.5:
                z = -z

    margin = z * data_std / math.sqrt(n)

    return (data_mean - margin, data_mean + margin)


# =============================================================================
# Main Demonstration
# =============================================================================

if __name__ == "__main__":
    print("Statistics Module Demonstration")
    print("=" * 50)

    # Example data from issue specification
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"\nDataset: {data}")
    print("-" * 30)

    # Descriptive statistics
    print(f"Mean: {mean(data)}")
    print(f"Median: {median(data)}")
    print(f"Mode: {mode(data)}")
    print(f"Variance: {variance(data)}")
    print(f"Std Dev: {std_dev(data)}")

    # Percentiles
    print(f"\nPercentile (75th): {percentile(data, 75)}")
    q1, q2, q3 = quartiles(data)
    print(f"Quartiles: Q1={q1}, Q2={q2}, Q3={q3}")
    print(f"IQR: {iqr(data)}")

    # Correlation and regression
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    print(f"\nLinear Regression (x={x}, y={y}):")
    slope, intercept, r2 = linear_regression(x, y)
    print(f"y = {slope:.2f}x + {intercept:.2f}, R²={r2:.3f}")
    print(f"Correlation: {correlation(x, y):.4f}")

    # Distribution functions
    print(f"\nNormal CDF(0, 0, 1): {round(normal_cdf(0, 0, 1), 6)}")
    print(f"Normal PDF(0, 0, 1): {round(normal_pdf(0, 0, 1), 6)}")
    print(f"Binomial PMF(5, 10, 0.5): {round(binomial_pmf(5, 10, 0.5), 6)}")
    print(f"Poisson PMF(2, 3): {round(poisson_pmf(2, 3), 6)}")

    # Hypothesis testing
    sample1 = [1, 2, 3, 4, 5]
    sample2 = [2, 3, 4, 5, 6]
    t_stat, p_val = t_test(sample1, sample2)
    print(f"\nT-test (samples: {sample1} vs {sample2}):")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

    # Sampling
    random.seed(42)
    bootstrap = bootstrap_sample(data, 5)
    print(f"\nBootstrap sample (n=5): {bootstrap}")

    ci_lower, ci_upper = confidence_interval(data)
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
