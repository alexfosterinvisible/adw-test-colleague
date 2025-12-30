"""(Claude) Statistics module with pure Python implementations.

Provides descriptive statistics, percentiles, correlation analysis,
probability distributions, hypothesis testing, and sampling utilities.
Uses only the math module - no scipy, statsmodels, or numpy.
"""

import random
from math import exp, factorial as math_factorial, log, pi, sqrt


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_data(data: list) -> None:
    """Validate that data is not empty."""
    if not data:
        raise ValueError("Data cannot be empty")


def _factorial(n: int) -> int:
    """Iterative factorial calculation."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _combinations(n: int, k: int) -> int:
    """Binomial coefficient (n choose k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use the smaller k for efficiency
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _erf(x: float) -> float:
    """Error function approximation using Taylor series and asymptotic formula."""
    # For large |x|, use asymptotic approximation
    if abs(x) > 3.5:
        sign = 1 if x >= 0 else -1
        x_abs = abs(x)
        # Asymptotic expansion: erf(x) ≈ 1 - exp(-x^2) / (x * sqrt(pi))
        result = 1.0 - exp(-x_abs * x_abs) / (x_abs * sqrt(pi))
        return sign * result

    # Taylor series: erf(x) = (2/sqrt(pi)) * sum((-1)^n * x^(2n+1) / (n! * (2n+1)))
    result = 0.0
    for n in range(50):  # Enough terms for 6 decimal precision
        term = ((-1) ** n) * (x ** (2 * n + 1)) / (_factorial(n) * (2 * n + 1))
        result += term
        if abs(term) < 1e-15:
            break
    return result * 2 / sqrt(pi)


def _gamma_inc_lower(a: float, x: float) -> float:
    """Lower incomplete gamma function approximation using series expansion."""
    if x <= 0:
        return 0.0
    # Series expansion: gamma_inc(a, x) = x^a * e^(-x) * sum(x^n / gamma(a+n+1))
    result = 0.0
    term = 1.0 / a
    result = term
    for n in range(1, 200):
        term *= x / (a + n)
        result += term
        if abs(term) < 1e-15:
            break
    return result * (x ** a) * exp(-x)


def _gamma(a: float) -> float:
    """Gamma function approximation using Lanczos approximation."""
    if a <= 0 and a == int(a):
        raise ValueError("Gamma function not defined for non-positive integers")
    if a < 0.5:
        # Reflection formula: gamma(1-z) * gamma(z) = pi / sin(pi*z)
        import math

        return pi / (math.sin(pi * a) * _gamma(1 - a))

    a -= 1
    # Lanczos coefficients
    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (a + i)

    t = a + g + 0.5
    return sqrt(2 * pi) * (t ** (a + 0.5)) * exp(-t) * x


def _chi2_cdf(x: float, df: int) -> float:
    """Chi-square CDF using incomplete gamma function."""
    if x <= 0:
        return 0.0
    # CDF = gamma_inc_lower(df/2, x/2) / gamma(df/2)
    return _gamma_inc_lower(df / 2, x / 2) / _gamma(df / 2)


def _t_cdf(t: float, df: int) -> float:
    """Student's t-distribution CDF approximation."""
    # Use normal approximation for large df
    if df > 100:
        return normal_cdf(t, 0, 1)

    # For smaller df, use incomplete beta function approximation
    x = df / (df + t * t)
    # Approximation based on regularized incomplete beta function
    if t < 0:
        return 0.5 * _betainc(df / 2, 0.5, x)
    else:
        return 1 - 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function approximation."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use continued fraction expansion
    bt = exp(
        _lgamma(a + b)
        - _lgamma(a)
        - _lgamma(b)
        + a * log(x)
        + b * log(1 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    else:
        return 1 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function."""
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
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c

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

    return h


def _lgamma(x: float) -> float:
    """Log-gamma function using Lanczos approximation (non-recursive)."""
    if x <= 0:
        raise ValueError("lgamma not defined for non-positive values")

    # Lanczos approximation coefficients
    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if x < 0.5:
        # Reflection formula: gamma(1-z) * gamma(z) = pi / sin(pi*z)
        import math

        return log(pi / math.sin(pi * x)) - _lgamma(1 - x)

    x -= 1
    a = c[0]
    for i in range(1, g + 2):
        a += c[i] / (x + i)

    t = x + g + 0.5
    return 0.5 * log(2 * pi) + (x + 0.5) * log(t) - t + log(a)


# =============================================================================
# Descriptive Statistics
# =============================================================================


def mean(data: list[float]) -> float:
    """Calculate arithmetic mean."""
    _validate_data(data)
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calculate median (middle value)."""
    _validate_data(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]


def mode(data: list[float]) -> float:
    """Find most frequent value (returns first if tie)."""
    _validate_data(data)
    counts: dict[float, int] = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    # Return first value with max count
    max_count = max(counts.values())
    for value in data:
        if counts[value] == max_count:
            return value
    return data[0]  # Fallback


def variance(data: list[float], population: bool = True) -> float:
    """Calculate variance (population by default, sample if population=False)."""
    _validate_data(data)
    n = len(data)
    if n == 1:
        return 0.0
    data_mean = mean(data)
    squared_diff_sum = sum((x - data_mean) ** 2 for x in data)
    if population:
        return squared_diff_sum / n
    return squared_diff_sum / (n - 1)


def std_dev(data: list[float], population: bool = True) -> float:
    """Calculate standard deviation (sqrt of variance)."""
    return sqrt(variance(data, population))


# =============================================================================
# Percentile Functions
# =============================================================================


def percentile(data: list[float], p: float) -> float:
    """Calculate p-th percentile (0-100 scale) using linear interpolation."""
    _validate_data(data)
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return sorted_data[0]

    # Calculate position
    pos = (p / 100) * (n - 1)
    lower_idx = int(pos)
    upper_idx = lower_idx + 1
    fraction = pos - lower_idx

    if upper_idx >= n:
        return sorted_data[-1]

    return sorted_data[lower_idx] + fraction * (
        sorted_data[upper_idx] - sorted_data[lower_idx]
    )


def quartiles(data: list[float]) -> tuple[float, float, float]:
    """Calculate Q1, Q2 (median), Q3."""
    _validate_data(data)
    q1 = percentile(data, 25)
    q2 = percentile(data, 50)
    q3 = percentile(data, 75)
    return (q1, q2, q3)


def iqr(data: list[float]) -> float:
    """Calculate interquartile range (Q3 - Q1)."""
    q1, _, q3 = quartiles(data)
    return q3 - q1


# =============================================================================
# Correlation and Regression
# =============================================================================


def covariance(x: list[float], y: list[float]) -> float:
    """Calculate population covariance."""
    _validate_data(x)
    _validate_data(y)
    if len(x) != len(y):
        raise ValueError("Lists must have the same length")

    n = len(x)
    x_mean = mean(x)
    y_mean = mean(y)
    return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / n


def correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    _validate_data(x)
    _validate_data(y)
    if len(x) != len(y):
        raise ValueError("Lists must have the same length")

    std_x = std_dev(x)
    std_y = std_dev(y)

    if std_x == 0 or std_y == 0:
        raise ValueError("Cannot compute correlation with zero variance")

    return covariance(x, y) / (std_x * std_y)


def linear_regression(
    x: list[float], y: list[float]
) -> tuple[float, float, float]:
    """Perform linear regression, returning (slope, intercept, r_squared)."""
    _validate_data(x)
    _validate_data(y)
    if len(x) != len(y):
        raise ValueError("Lists must have the same length")

    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate slope
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        raise ValueError("Cannot compute regression with zero variance in x")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate R-squared
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    if ss_tot == 0:
        r_squared = 1.0 if ss_res == 0 else 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return (slope, intercept, r_squared)


# =============================================================================
# Normal Distribution
# =============================================================================


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Normal probability density function."""
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    coefficient = 1 / (sigma * sqrt(2 * pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * exp(exponent)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Normal cumulative distribution function using erf approximation."""
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    z = (x - mu) / (sigma * sqrt(2))
    return 0.5 * (1 + _erf(z))


# =============================================================================
# Binomial Distribution
# =============================================================================


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass function."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0 or k > n:
        return 0.0

    return _combinations(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Binomial cumulative distribution function."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    return sum(binomial_pmf(i, n, p) for i in range(k + 1))


# =============================================================================
# Poisson Distribution
# =============================================================================


def poisson_pmf(k: int, lambda_: float) -> float:
    """Poisson probability mass function."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if k < 0:
        return 0.0

    return (lambda_ ** k) * exp(-lambda_) / math_factorial(k)


def poisson_cdf(k: int, lambda_: float) -> float:
    """Poisson cumulative distribution function."""
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if k < 0:
        return 0.0

    return sum(poisson_pmf(i, lambda_) for i in range(k + 1))


# =============================================================================
# Hypothesis Testing
# =============================================================================


def t_test(
    sample1: list[float], sample2: list[float]
) -> tuple[float, float]:
    """Two-sample t-test returning (t_statistic, p_value)."""
    _validate_data(sample1)
    _validate_data(sample2)

    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = mean(sample1)
    mean2 = mean(sample2)
    var1 = variance(sample1, population=False) if n1 > 1 else 0
    var2 = variance(sample2, population=False) if n2 > 1 else 0

    # Pooled standard error
    se = sqrt(var1 / n1 + var2 / n2) if (var1 > 0 or var2 > 0) else 0

    if se == 0:
        # Both samples have zero variance
        if mean1 == mean2:
            return (0.0, 1.0)
        else:
            return (float("inf") if mean1 > mean2 else float("-inf"), 0.0)

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    if var1 == 0 and var2 == 0:
        df = n1 + n2 - 2
    elif var1 == 0:
        df = n2 - 1
    elif var2 == 0:
        df = n1 - 1
    else:
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = int(num / denom) if denom > 0 else n1 + n2 - 2

    # Two-tailed p-value
    p_value = 2 * (1 - _t_cdf(abs(t_stat), max(df, 1)))

    return (t_stat, p_value)


def chi_square_test(
    observed: list[float], expected: list[float]
) -> tuple[float, float]:
    """Chi-square test returning (chi2_statistic, p_value)."""
    _validate_data(observed)
    _validate_data(expected)
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have the same length")

    if any(e <= 0 for e in expected):
        raise ValueError("Expected values must be positive")

    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1

    # p-value using chi-square CDF
    p_value = 1 - _chi2_cdf(chi2, df) if df > 0 else 1.0

    return (chi2, p_value)


# =============================================================================
# Sampling Functions
# =============================================================================


def bootstrap_sample(data: list[float], n: int) -> list[float]:
    """Generate bootstrap sample (random sampling with replacement)."""
    _validate_data(data)
    if n < 0:
        raise ValueError("Sample size must be non-negative")
    return random.choices(data, k=n)


def confidence_interval(
    data: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for the mean."""
    _validate_data(data)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")

    n = len(data)
    data_mean = mean(data)

    if n == 1:
        return (data_mean, data_mean)

    data_std = std_dev(data, population=False)
    se = data_std / sqrt(n)

    # t-critical value approximation for common confidence levels
    # For large samples, use z-scores
    if n > 30:
        if confidence == 0.95:
            t_crit = 1.96
        elif confidence == 0.99:
            t_crit = 2.576
        elif confidence == 0.90:
            t_crit = 1.645
        else:
            # Use normal approximation for other values
            # z = norm.ppf((1 + confidence) / 2)
            alpha = 1 - confidence
            z = _z_score_approx(1 - alpha / 2)
            t_crit = z
    else:
        # Use approximate t-critical values for small samples
        df = n - 1
        alpha = 1 - confidence
        t_crit = _t_critical_approx(df, alpha / 2)

    margin = t_crit * se
    return (data_mean - margin, data_mean + margin)


def _z_score_approx(p: float) -> float:
    """Approximate z-score for probability p using rational approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1")

    # Rational approximation for inverse normal CDF
    # Abramowitz and Stegun approximation 26.2.23
    # This approximation is for p in (0, 0.5] and gives a negative z-score
    if p > 0.5:
        # For p > 0.5, use symmetry: z(p) = -z(1-p)
        # Since _z_score_approx(1-p) for (1-p) < 0.5 gives negative,
        # negating it gives positive, which is correct for p > 0.5
        return -_z_score_approx(1 - p)

    # For p <= 0.5, compute the negative z-score
    t = sqrt(-2 * log(p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    # This gives a positive value, but we need negative for p < 0.5
    approx = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t ** 3)
    return -approx  # Return negative for p < 0.5


def _t_critical_approx(df: int, alpha: float) -> float:
    """Approximate t-critical value."""
    # Use normal approximation and scale for small df
    z = _z_score_approx(1 - alpha)
    if df >= 30:
        return z
    # Adjust for small df
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    return z + g1 / df + g2 / (df ** 2)


# =============================================================================
# Main Demonstration
# =============================================================================

if __name__ == "__main__":
    print("Statistics Module Demonstration")
    print("=" * 40)

    # Descriptive statistics
    sample_data = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"\nSample data: {sample_data}")
    print(f"Mean: {mean(sample_data)}")
    print(f"Median: {median(sample_data)}")
    print(f"Mode: {mode(sample_data)}")
    print(f"Variance: {variance(sample_data)}")
    print(f"Std Dev: {std_dev(sample_data)}")

    # Percentiles
    print(f"\n75th Percentile: {percentile(sample_data, 75)}")
    print(f"Quartiles: {quartiles(sample_data)}")
    print(f"IQR: {iqr(sample_data)}")

    # Linear regression
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    slope, intercept, r_squared = linear_regression(x, y)
    print(f"\nLinear Regression (x={x}, y={y}):")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R-squared: {r_squared:.4f}")

    # Normal distribution
    print("\nNormal Distribution:")
    print(f"  PDF(0, mu=0, sigma=1): {normal_pdf(0, 0, 1):.6f}")
    print(f"  CDF(0, mu=0, sigma=1): {normal_cdf(0, 0, 1):.6f}")

    # Bootstrap sample
    print(f"\nBootstrap sample (n=5): {bootstrap_sample(sample_data, 5)}")

    # Confidence interval
    ci = confidence_interval(sample_data, 0.95)
    print(f"95% CI for mean: ({ci[0]:.4f}, {ci[1]:.4f})")
