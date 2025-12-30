"""(Claude) Unit tests for statistics module."""

import unittest
import random

# Import our statistics module (not stdlib)
import statistics as stats


class TestDescriptiveStatistics(unittest.TestCase):
    """Test suite for descriptive statistics functions."""

    def test_mean(self):
        """Test mean calculation."""
        # Example from spec
        self.assertEqual(stats.mean([2, 4, 4, 4, 5, 5, 7, 9]), 5.0)
        # Additional cases
        self.assertEqual(stats.mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(stats.mean([10]), 10.0)
        self.assertAlmostEqual(stats.mean([1.5, 2.5, 3.5]), 2.5, places=6)
        # Negative values
        self.assertEqual(stats.mean([-1, 0, 1]), 0.0)
        self.assertEqual(stats.mean([-5, -3, -1]), -3.0)

    def test_median(self):
        """Test median calculation."""
        # Odd length
        self.assertEqual(stats.median([1, 3, 5, 7, 9]), 5.0)
        self.assertEqual(stats.median([5]), 5.0)
        # Even length
        self.assertEqual(stats.median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(stats.median([2, 4, 4, 4, 5, 5, 7, 9]), 4.5, places=6)
        # Unsorted input
        self.assertEqual(stats.median([9, 1, 5, 3, 7]), 5.0)

    def test_mode(self):
        """Test mode calculation."""
        # Single mode
        self.assertEqual(stats.mode([2, 4, 4, 4, 5, 5, 7, 9]), 4)
        # Multiple modes (return first)
        self.assertEqual(stats.mode([1, 1, 2, 2, 3]), 1)
        # All unique (return first)
        self.assertEqual(stats.mode([1, 2, 3, 4, 5]), 1)
        # Single element
        self.assertEqual(stats.mode([42]), 42)

    def test_variance(self):
        """Test variance calculation."""
        # Example from spec (population variance)
        self.assertAlmostEqual(stats.variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.0, places=6)
        # Single element
        self.assertEqual(stats.variance([5]), 0.0)
        # All same values
        self.assertEqual(stats.variance([3, 3, 3, 3]), 0.0)
        # Sample variance
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(
            stats.variance(data, population=False),
            4.571428571,
            places=6
        )

    def test_std_dev(self):
        """Test standard deviation calculation."""
        # Example from spec
        self.assertAlmostEqual(stats.std_dev([2, 4, 4, 4, 5, 5, 7, 9]), 2.0, places=6)
        # Single element
        self.assertEqual(stats.std_dev([5]), 0.0)
        # All same values
        self.assertEqual(stats.std_dev([3, 3, 3, 3]), 0.0)


class TestPercentileFunctions(unittest.TestCase):
    """Test suite for percentile functions."""

    def test_percentile(self):
        """Test percentile calculation using linear interpolation."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        # 75th percentile with linear interpolation: index = 0.75 * 7 = 5.25
        # sorted[5] = 5, sorted[6] = 7, result = 5 + 0.25 * 2 = 5.5
        self.assertAlmostEqual(stats.percentile(data, 75), 5.5, places=6)
        # Edge percentiles
        self.assertEqual(stats.percentile(data, 0), 2)
        self.assertEqual(stats.percentile(data, 100), 9)
        # Median
        self.assertAlmostEqual(stats.percentile(data, 50), 4.5, places=6)
        # Single element
        self.assertEqual(stats.percentile([42], 50), 42)
        self.assertEqual(stats.percentile([42], 0), 42)
        self.assertEqual(stats.percentile([42], 100), 42)

    def test_percentile_invalid(self):
        """Test percentile with invalid p values."""
        with self.assertRaises(ValueError):
            stats.percentile([1, 2, 3], -1)
        with self.assertRaises(ValueError):
            stats.percentile([1, 2, 3], 101)

    def test_quartiles(self):
        """Test quartiles calculation using linear interpolation."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        q1, q2, q3 = stats.quartiles(data)
        self.assertAlmostEqual(q1, 4.0, places=6)
        self.assertAlmostEqual(q2, 4.5, places=6)
        self.assertAlmostEqual(q3, 5.5, places=6)

    def test_iqr(self):
        """Test interquartile range calculation."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        # IQR = Q3 - Q1 = 5.5 - 4.0 = 1.5
        self.assertAlmostEqual(stats.iqr(data), 1.5, places=6)


class TestCorrelationRegression(unittest.TestCase):
    """Test suite for correlation and regression functions."""

    def test_covariance(self):
        """Test covariance calculation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        # Population covariance = sum((xi-mx)(yi-my))/n = 6/5 = 1.2
        self.assertAlmostEqual(stats.covariance(x, y), 1.2, places=6)
        # Negative covariance
        y_neg = [5, 4, 3, 2, 1]
        self.assertAlmostEqual(stats.covariance(x, y_neg), -2.0, places=6)
        # Zero covariance
        y_zero = [3, 3, 3, 3, 3]
        self.assertEqual(stats.covariance(x, y_zero), 0.0)

    def test_covariance_length_mismatch(self):
        """Test covariance with mismatched lengths."""
        with self.assertRaises(ValueError):
            stats.covariance([1, 2, 3], [1, 2])

    def test_correlation(self):
        """Test Pearson correlation coefficient."""
        # Perfect positive correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(stats.correlation(x, y), 1.0, places=6)
        # Perfect negative correlation
        y_neg = [10, 8, 6, 4, 2]
        self.assertAlmostEqual(stats.correlation(x, y_neg), -1.0, places=6)
        # Partial correlation
        y_partial = [2, 4, 5, 4, 5]
        r = stats.correlation(x, y_partial)
        self.assertGreater(r, 0)
        self.assertLess(r, 1)

    def test_correlation_zero_variance(self):
        """Test correlation with zero variance raises error."""
        with self.assertRaises(ValueError):
            stats.correlation([1, 2, 3], [5, 5, 5])
        with self.assertRaises(ValueError):
            stats.correlation([5, 5, 5], [1, 2, 3])

    def test_linear_regression(self):
        """Test linear regression."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        slope, intercept, r2 = stats.linear_regression(x, y)
        self.assertAlmostEqual(slope, 0.6, places=6)
        self.assertAlmostEqual(intercept, 2.2, places=6)
        # R² = cov(x,y)² / (var(x) * var(y)) = 1.2² / (2 * 1.2) = 1.44 / 2.4 = 0.6
        self.assertAlmostEqual(r2, 0.6, places=6)

    def test_linear_regression_perfect_fit(self):
        """Test linear regression with perfect fit."""
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]  # y = 2x + 1
        slope, intercept, r2 = stats.linear_regression(x, y)
        self.assertAlmostEqual(slope, 2.0, places=6)
        self.assertAlmostEqual(intercept, 1.0, places=6)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_linear_regression_zero_variance_x(self):
        """Test linear regression with zero variance in x."""
        with self.assertRaises(ValueError):
            stats.linear_regression([5, 5, 5], [1, 2, 3])


class TestDistributionFunctions(unittest.TestCase):
    """Test suite for distribution functions."""

    def test_normal_pdf(self):
        """Test normal PDF."""
        # Standard normal at x=0
        self.assertAlmostEqual(stats.normal_pdf(0, 0, 1), 0.398942, places=5)
        # Standard normal at x=1
        self.assertAlmostEqual(stats.normal_pdf(1, 0, 1), 0.241971, places=5)
        # Custom mu, sigma
        self.assertAlmostEqual(stats.normal_pdf(5, 5, 2), 0.199471, places=5)

    def test_normal_pdf_invalid(self):
        """Test normal PDF with invalid sigma."""
        with self.assertRaises(ValueError):
            stats.normal_pdf(0, 0, 0)
        with self.assertRaises(ValueError):
            stats.normal_pdf(0, 0, -1)

    def test_normal_cdf(self):
        """Test normal CDF."""
        # Standard normal at x=0
        self.assertAlmostEqual(stats.normal_cdf(0, 0, 1), 0.5, places=6)
        # Standard normal tails
        self.assertAlmostEqual(stats.normal_cdf(-3, 0, 1), 0.00135, places=4)
        self.assertAlmostEqual(stats.normal_cdf(3, 0, 1), 0.99865, places=4)
        # At +/- 1 std dev
        self.assertAlmostEqual(stats.normal_cdf(1, 0, 1), 0.841345, places=4)
        self.assertAlmostEqual(stats.normal_cdf(-1, 0, 1), 0.158655, places=4)

    def test_normal_cdf_invalid(self):
        """Test normal CDF with invalid sigma."""
        with self.assertRaises(ValueError):
            stats.normal_cdf(0, 0, 0)

    def test_binomial_pmf(self):
        """Test binomial PMF."""
        # P(X=5) for X ~ Binomial(10, 0.5)
        self.assertAlmostEqual(stats.binomial_pmf(5, 10, 0.5), 0.246094, places=5)
        # Edge cases
        self.assertEqual(stats.binomial_pmf(0, 10, 0.5), 0.5 ** 10)
        self.assertEqual(stats.binomial_pmf(10, 10, 0.5), 0.5 ** 10)
        # p=0 and p=1
        self.assertEqual(stats.binomial_pmf(0, 5, 0), 1.0)
        self.assertEqual(stats.binomial_pmf(5, 5, 1), 1.0)

    def test_binomial_pmf_invalid(self):
        """Test binomial PMF with invalid inputs."""
        with self.assertRaises(ValueError):
            stats.binomial_pmf(5, 10, -0.1)
        with self.assertRaises(ValueError):
            stats.binomial_pmf(5, 10, 1.1)
        with self.assertRaises(ValueError):
            stats.binomial_pmf(-1, 10, 0.5)
        with self.assertRaises(ValueError):
            stats.binomial_pmf(11, 10, 0.5)

    def test_binomial_cdf(self):
        """Test binomial CDF."""
        # P(X<=5) for X ~ Binomial(10, 0.5)
        self.assertAlmostEqual(stats.binomial_cdf(5, 10, 0.5), 0.623047, places=5)
        # Edge cases
        self.assertEqual(stats.binomial_cdf(10, 10, 0.5), 1.0)
        self.assertAlmostEqual(stats.binomial_cdf(0, 10, 0.5), 0.5 ** 10, places=10)

    def test_poisson_pmf(self):
        """Test Poisson PMF."""
        # P(X=2) for X ~ Poisson(3)
        self.assertAlmostEqual(stats.poisson_pmf(2, 3), 0.224042, places=5)
        # k=0
        self.assertAlmostEqual(stats.poisson_pmf(0, 3), 0.049787, places=5)

    def test_poisson_pmf_invalid(self):
        """Test Poisson PMF with invalid inputs."""
        with self.assertRaises(ValueError):
            stats.poisson_pmf(2, 0)
        with self.assertRaises(ValueError):
            stats.poisson_pmf(2, -1)
        with self.assertRaises(ValueError):
            stats.poisson_pmf(-1, 3)

    def test_poisson_cdf(self):
        """Test Poisson CDF."""
        # P(X<=2) for X ~ Poisson(3)
        self.assertAlmostEqual(stats.poisson_cdf(2, 3), 0.423190, places=5)


class TestHypothesisTesting(unittest.TestCase):
    """Test suite for hypothesis testing functions."""

    def test_t_test(self):
        """Test two-sample t-test."""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [2, 3, 4, 5, 6]
        t_stat, p_val = stats.t_test(sample1, sample2)
        # t-statistic should be negative (sample1 mean < sample2 mean)
        self.assertLess(t_stat, 0)
        self.assertAlmostEqual(t_stat, -1.0, places=4)
        # p-value should be > 0.05 (not significant)
        self.assertGreater(p_val, 0.05)

    def test_t_test_identical_samples(self):
        """Test t-test with identical samples."""
        sample = [1, 2, 3, 4, 5]
        t_stat, p_val = stats.t_test(sample, sample.copy())
        self.assertAlmostEqual(t_stat, 0.0, places=6)

    def test_t_test_different_means(self):
        """Test t-test with clearly different means."""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [10, 11, 12, 13, 14]
        t_stat, p_val = stats.t_test(sample1, sample2)
        self.assertLess(t_stat, 0)
        # Should be significant
        self.assertLess(p_val, 0.05)

    def test_chi_square_test(self):
        """Test chi-square goodness-of-fit test."""
        observed = [10, 20, 30]
        expected = [15, 20, 25]
        chi_stat, p_val = stats.chi_square_test(observed, expected)
        self.assertAlmostEqual(chi_stat, 2.6667, places=3)
        # p-value should be > 0.05 (not significant)
        self.assertGreater(p_val, 0.05)

    def test_chi_square_test_perfect_fit(self):
        """Test chi-square with perfect fit."""
        observed = [10, 20, 30]
        chi_stat, p_val = stats.chi_square_test(observed, observed)
        self.assertEqual(chi_stat, 0.0)

    def test_chi_square_test_invalid(self):
        """Test chi-square with invalid inputs."""
        with self.assertRaises(ValueError):
            stats.chi_square_test([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            stats.chi_square_test([1, 2, 3], [1, 0, 3])
        with self.assertRaises(ValueError):
            stats.chi_square_test([1, 2, 3], [1, -1, 3])


class TestSamplingMethods(unittest.TestCase):
    """Test suite for sampling methods."""

    def test_bootstrap_sample(self):
        """Test bootstrap sampling."""
        data = [1, 2, 3, 4, 5]
        random.seed(42)
        sample = stats.bootstrap_sample(data, 10)
        self.assertEqual(len(sample), 10)
        # All values should be from original data
        for val in sample:
            self.assertIn(val, data)

    def test_bootstrap_sample_invalid(self):
        """Test bootstrap with invalid inputs."""
        with self.assertRaises(ValueError):
            stats.bootstrap_sample([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            stats.bootstrap_sample([1, 2, 3], -1)

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        data = [1, 2, 3, 4, 5]
        lower, upper = stats.confidence_interval(data, 0.95)
        data_mean = stats.mean(data)
        # Interval should contain the mean
        self.assertLess(lower, data_mean)
        self.assertGreater(upper, data_mean)
        # Lower should be less than upper
        self.assertLess(lower, upper)

    def test_confidence_interval_narrowing(self):
        """Test that higher confidence = wider interval."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        l90, u90 = stats.confidence_interval(data, 0.90)
        l95, u95 = stats.confidence_interval(data, 0.95)
        l99, u99 = stats.confidence_interval(data, 0.99)
        # Width should increase with confidence
        width90 = u90 - l90
        width95 = u95 - l95
        width99 = u99 - l99
        self.assertLess(width90, width95)
        self.assertLess(width95, width99)

    def test_confidence_interval_invalid(self):
        """Test confidence interval with invalid confidence."""
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 1)
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], -0.5)
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 1.5)

    def test_confidence_interval_single_element(self):
        """Test confidence interval with single element."""
        lower, upper = stats.confidence_interval([5], 0.95)
        self.assertEqual(lower, 5)
        self.assertEqual(upper, 5)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases across all functions."""

    def test_empty_data(self):
        """Test that all functions raise ValueError for empty data."""
        with self.assertRaises(ValueError) as ctx:
            stats.mean([])
        self.assertEqual(str(ctx.exception), "Data cannot be empty")

        with self.assertRaises(ValueError):
            stats.median([])
        with self.assertRaises(ValueError):
            stats.mode([])
        with self.assertRaises(ValueError):
            stats.variance([])
        with self.assertRaises(ValueError):
            stats.std_dev([])
        with self.assertRaises(ValueError):
            stats.percentile([], 50)
        with self.assertRaises(ValueError):
            stats.quartiles([])
        with self.assertRaises(ValueError):
            stats.iqr([])
        with self.assertRaises(ValueError):
            stats.covariance([], [])
        with self.assertRaises(ValueError):
            stats.correlation([], [])
        with self.assertRaises(ValueError):
            stats.linear_regression([], [])
        with self.assertRaises(ValueError):
            stats.t_test([], [1, 2])
        with self.assertRaises(ValueError):
            stats.chi_square_test([], [])
        with self.assertRaises(ValueError):
            stats.bootstrap_sample([], 5)
        with self.assertRaises(ValueError):
            stats.confidence_interval([])

    def test_single_element(self):
        """Test single element datasets."""
        data = [42]
        self.assertEqual(stats.mean(data), 42)
        self.assertEqual(stats.median(data), 42)
        self.assertEqual(stats.mode(data), 42)
        self.assertEqual(stats.variance(data), 0.0)
        self.assertEqual(stats.std_dev(data), 0.0)
        self.assertEqual(stats.percentile(data, 50), 42)
        self.assertEqual(stats.quartiles(data), (42, 42, 42))
        self.assertEqual(stats.iqr(data), 0.0)

    def test_same_values(self):
        """Test datasets with all identical values."""
        data = [5, 5, 5, 5, 5]
        self.assertEqual(stats.mean(data), 5.0)
        self.assertEqual(stats.median(data), 5.0)
        self.assertEqual(stats.mode(data), 5)
        self.assertEqual(stats.variance(data), 0.0)
        self.assertEqual(stats.std_dev(data), 0.0)
        self.assertEqual(stats.percentile(data, 25), 5.0)
        self.assertEqual(stats.quartiles(data), (5.0, 5.0, 5.0))
        self.assertEqual(stats.iqr(data), 0.0)
        # Covariance with constant y
        self.assertEqual(stats.covariance([1, 2, 3, 4, 5], data), 0.0)

    def test_negative_values(self):
        """Test datasets with negative values."""
        data = [-5, -3, -1, 0, 1, 3, 5]
        self.assertEqual(stats.mean(data), 0.0)
        self.assertEqual(stats.median(data), 0.0)

    def test_large_numbers(self):
        """Test numerical stability with large numbers."""
        data = [1e10, 1e10 + 1, 1e10 + 2]
        m = stats.mean(data)
        self.assertAlmostEqual(m, 1e10 + 1, places=0)


class TestHelperFunctions(unittest.TestCase):
    """Test suite for helper functions."""

    def test_factorial(self):
        """Test factorial calculation."""
        self.assertEqual(stats._factorial(0), 1)
        self.assertEqual(stats._factorial(1), 1)
        self.assertEqual(stats._factorial(5), 120)
        self.assertEqual(stats._factorial(10), 3628800)

    def test_factorial_negative(self):
        """Test factorial with negative input."""
        with self.assertRaises(ValueError):
            stats._factorial(-1)

    def test_combinations(self):
        """Test combinations calculation."""
        self.assertEqual(stats._combinations(5, 0), 1)
        self.assertEqual(stats._combinations(5, 5), 1)
        self.assertEqual(stats._combinations(5, 2), 10)
        self.assertEqual(stats._combinations(10, 3), 120)

    def test_combinations_invalid(self):
        """Test combinations with invalid inputs."""
        with self.assertRaises(ValueError):
            stats._combinations(-1, 2)
        with self.assertRaises(ValueError):
            stats._combinations(5, -1)
        with self.assertRaises(ValueError):
            stats._combinations(5, 6)

    def test_erf(self):
        """Test error function approximation."""
        # erf(0) = 0
        self.assertAlmostEqual(stats._erf(0), 0.0, places=6)
        # erf(1) ~ 0.8427
        self.assertAlmostEqual(stats._erf(1), 0.842701, places=5)
        # erf(-1) = -erf(1)
        self.assertAlmostEqual(stats._erf(-1), -0.842701, places=5)
        # Large values approach +/- 1
        self.assertAlmostEqual(stats._erf(10), 1.0, places=6)
        self.assertAlmostEqual(stats._erf(-10), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
