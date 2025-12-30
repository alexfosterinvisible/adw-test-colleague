"""(Claude) Unit tests for statistics module."""

import unittest

import statistics as stats


class TestDescriptiveStats(unittest.TestCase):
    """Tests for mean, median, mode, variance, std_dev."""

    def test_mean_positive(self):
        """Test mean with positive numbers."""
        self.assertAlmostEqual(stats.mean([1, 2, 3, 4, 5]), 3.0, places=6)

    def test_mean_mixed(self):
        """Test mean with mixed positive/negative."""
        self.assertAlmostEqual(stats.mean([-2, -1, 0, 1, 2]), 0.0, places=6)

    def test_mean_known_value(self):
        """Test mean returns 5.0 for [2,4,4,4,5,5,7,9]."""
        self.assertAlmostEqual(
            stats.mean([2, 4, 4, 4, 5, 5, 7, 9]), 5.0, places=6
        )

    def test_mean_single_element(self):
        """Test mean with single element."""
        self.assertAlmostEqual(stats.mean([42]), 42.0, places=6)

    def test_mean_empty_raises(self):
        """Test mean raises ValueError for empty data."""
        with self.assertRaises(ValueError) as ctx:
            stats.mean([])
        self.assertEqual(str(ctx.exception), "Data cannot be empty")

    def test_median_odd_length(self):
        """Test median with odd-length list."""
        self.assertAlmostEqual(stats.median([1, 2, 3, 4, 5]), 3.0, places=6)

    def test_median_even_length(self):
        """Test median with even-length list."""
        self.assertAlmostEqual(stats.median([1, 2, 3, 4]), 2.5, places=6)

    def test_median_single_element(self):
        """Test median with single element."""
        self.assertAlmostEqual(stats.median([7]), 7.0, places=6)

    def test_median_unsorted(self):
        """Test median sorts data correctly."""
        self.assertAlmostEqual(stats.median([5, 1, 3, 2, 4]), 3.0, places=6)

    def test_median_empty_raises(self):
        """Test median raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.median([])

    def test_mode_single_mode(self):
        """Test mode with clear single mode."""
        self.assertEqual(stats.mode([1, 2, 2, 3]), 2)

    def test_mode_multiple_modes_returns_first(self):
        """Test mode returns first value when tie."""
        self.assertEqual(stats.mode([1, 1, 2, 2, 3]), 1)

    def test_mode_all_unique(self):
        """Test mode with all unique values."""
        self.assertEqual(stats.mode([1, 2, 3, 4]), 1)

    def test_mode_empty_raises(self):
        """Test mode raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.mode([])

    def test_variance_population(self):
        """Test population variance."""
        self.assertAlmostEqual(
            stats.variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.0, places=6
        )

    def test_variance_sample(self):
        """Test sample variance."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        expected_sample_var = 4.0 * 8 / 7  # n/(n-1) * population_var
        self.assertAlmostEqual(
            stats.variance(data, population=False), expected_sample_var, places=6
        )

    def test_variance_single_element(self):
        """Test variance with single element is 0."""
        self.assertAlmostEqual(stats.variance([5]), 0.0, places=6)

    def test_variance_all_same(self):
        """Test variance with all same values is 0."""
        self.assertAlmostEqual(stats.variance([3, 3, 3, 3]), 0.0, places=6)

    def test_variance_empty_raises(self):
        """Test variance raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.variance([])

    def test_std_dev_known_value(self):
        """Test std_dev returns 2.0 for [2,4,4,4,5,5,7,9]."""
        self.assertAlmostEqual(
            stats.std_dev([2, 4, 4, 4, 5, 5, 7, 9]), 2.0, places=6
        )

    def test_std_dev_is_sqrt_variance(self):
        """Test std_dev equals sqrt(variance)."""
        data = [1, 2, 3, 4, 5]
        import math

        self.assertAlmostEqual(
            stats.std_dev(data), math.sqrt(stats.variance(data)), places=6
        )

    def test_std_dev_empty_raises(self):
        """Test std_dev raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.std_dev([])


class TestPercentiles(unittest.TestCase):
    """Tests for percentile, quartiles, iqr."""

    def test_percentile_0(self):
        """Test 0th percentile is minimum."""
        self.assertAlmostEqual(stats.percentile([1, 2, 3, 4, 5], 0), 1.0, places=6)

    def test_percentile_100(self):
        """Test 100th percentile is maximum."""
        self.assertAlmostEqual(stats.percentile([1, 2, 3, 4, 5], 100), 5.0, places=6)

    def test_percentile_50(self):
        """Test 50th percentile is median."""
        data = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(
            stats.percentile(data, 50), stats.median(data), places=6
        )

    def test_percentile_75_known(self):
        """Test 75th percentile on known data."""
        # For [2,4,4,4,5,5,7,9], 75th percentile should be around 6
        result = stats.percentile([2, 4, 4, 4, 5, 5, 7, 9], 75)
        self.assertTrue(5.5 <= result <= 6.5)

    def test_percentile_single_element(self):
        """Test percentile with single element."""
        self.assertAlmostEqual(stats.percentile([42], 50), 42.0, places=6)

    def test_percentile_invalid_raises(self):
        """Test invalid percentile raises ValueError."""
        with self.assertRaises(ValueError):
            stats.percentile([1, 2, 3], 101)
        with self.assertRaises(ValueError):
            stats.percentile([1, 2, 3], -1)

    def test_percentile_empty_raises(self):
        """Test percentile raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.percentile([], 50)

    def test_quartiles_returns_tuple(self):
        """Test quartiles returns (Q1, Q2, Q3) tuple."""
        q1, q2, q3 = stats.quartiles([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertIsInstance(q1, float)
        self.assertIsInstance(q2, float)
        self.assertIsInstance(q3, float)
        self.assertLess(q1, q2)
        self.assertLess(q2, q3)

    def test_quartiles_q2_is_median(self):
        """Test Q2 equals median."""
        data = [1, 2, 3, 4, 5, 6, 7]
        _, q2, _ = stats.quartiles(data)
        self.assertAlmostEqual(q2, stats.median(data), places=6)

    def test_iqr_equals_q3_minus_q1(self):
        """Test IQR equals Q3 - Q1."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        q1, _, q3 = stats.quartiles(data)
        self.assertAlmostEqual(stats.iqr(data), q3 - q1, places=6)

    def test_iqr_all_same_is_zero(self):
        """Test IQR with all same values is 0."""
        self.assertAlmostEqual(stats.iqr([5, 5, 5, 5]), 0.0, places=6)


class TestCorrelation(unittest.TestCase):
    """Tests for covariance, correlation, linear_regression."""

    def test_covariance_positive(self):
        """Test covariance with positively related data."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        self.assertGreater(stats.covariance(x, y), 0)

    def test_covariance_negative(self):
        """Test covariance with negatively related data."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        self.assertLess(stats.covariance(x, y), 0)

    def test_covariance_mismatched_length_raises(self):
        """Test covariance raises ValueError for mismatched lengths."""
        with self.assertRaises(ValueError):
            stats.covariance([1, 2, 3], [1, 2])

    def test_covariance_empty_raises(self):
        """Test covariance raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.covariance([], [])

    def test_correlation_perfect_positive(self):
        """Test correlation of 1.0 for perfectly positive relationship."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(stats.correlation(x, y), 1.0, places=6)

    def test_correlation_perfect_negative(self):
        """Test correlation of -1.0 for perfectly negative relationship."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        self.assertAlmostEqual(stats.correlation(x, y), -1.0, places=6)

    def test_correlation_no_relationship(self):
        """Test correlation near 0 for uncorrelated data."""
        x = [1, 2, 3, 4, 5]
        y = [1, -1, 1, -1, 1]
        result = stats.correlation(x, y)
        self.assertTrue(-0.3 < result < 0.3)

    def test_correlation_zero_variance_raises(self):
        """Test correlation raises ValueError for zero variance."""
        with self.assertRaises(ValueError):
            stats.correlation([1, 1, 1], [1, 2, 3])

    def test_correlation_mismatched_length_raises(self):
        """Test correlation raises ValueError for mismatched lengths."""
        with self.assertRaises(ValueError):
            stats.correlation([1, 2, 3], [1, 2])

    def test_linear_regression_perfect_line(self):
        """Test regression on perfect line y = 2x + 1."""
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]
        slope, intercept, r_squared = stats.linear_regression(x, y)
        self.assertAlmostEqual(slope, 2.0, places=6)
        self.assertAlmostEqual(intercept, 1.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_linear_regression_returns_tuple(self):
        """Test regression returns (slope, intercept, r_squared)."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result = stats.linear_regression(x, y)
        self.assertEqual(len(result), 3)

    def test_linear_regression_r_squared_bounds(self):
        """Test R-squared is between 0 and 1."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        _, _, r_squared = stats.linear_regression(x, y)
        self.assertTrue(0 <= r_squared <= 1)

    def test_linear_regression_mismatched_length_raises(self):
        """Test regression raises ValueError for mismatched lengths."""
        with self.assertRaises(ValueError):
            stats.linear_regression([1, 2, 3], [1, 2])

    def test_linear_regression_zero_variance_raises(self):
        """Test regression raises ValueError for zero variance in x."""
        with self.assertRaises(ValueError):
            stats.linear_regression([1, 1, 1], [1, 2, 3])


class TestDistributions(unittest.TestCase):
    """Tests for normal, binomial, and Poisson distributions."""

    def test_normal_pdf_at_mean(self):
        """Test normal PDF is maximum at mean."""
        pdf_at_mean = stats.normal_pdf(0, 0, 1)
        pdf_away = stats.normal_pdf(1, 0, 1)
        self.assertGreater(pdf_at_mean, pdf_away)

    def test_normal_pdf_symmetric(self):
        """Test normal PDF is symmetric around mean."""
        self.assertAlmostEqual(
            stats.normal_pdf(-1, 0, 1), stats.normal_pdf(1, 0, 1), places=6
        )

    def test_normal_pdf_known_value(self):
        """Test normal PDF at mu=0, sigma=1, x=0."""
        import math

        expected = 1 / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(stats.normal_pdf(0, 0, 1), expected, places=6)

    def test_normal_pdf_negative_sigma_raises(self):
        """Test normal PDF raises for negative sigma."""
        with self.assertRaises(ValueError):
            stats.normal_pdf(0, 0, -1)

    def test_normal_cdf_at_mean(self):
        """Test normal CDF at mean is 0.5."""
        self.assertAlmostEqual(stats.normal_cdf(0, 0, 1), 0.5, places=6)

    def test_normal_cdf_z_scores(self):
        """Test normal CDF at known z-scores."""
        # z = 1.96 should give ~0.975
        self.assertAlmostEqual(stats.normal_cdf(1.96, 0, 1), 0.975, places=2)
        # z = -1.96 should give ~0.025
        self.assertAlmostEqual(stats.normal_cdf(-1.96, 0, 1), 0.025, places=2)

    def test_normal_cdf_increasing(self):
        """Test normal CDF is monotonically increasing."""
        cdf_neg = stats.normal_cdf(-1, 0, 1)
        cdf_zero = stats.normal_cdf(0, 0, 1)
        cdf_pos = stats.normal_cdf(1, 0, 1)
        self.assertLess(cdf_neg, cdf_zero)
        self.assertLess(cdf_zero, cdf_pos)

    def test_normal_cdf_negative_sigma_raises(self):
        """Test normal CDF raises for negative sigma."""
        with self.assertRaises(ValueError):
            stats.normal_cdf(0, 0, -1)

    def test_binomial_pmf_known(self):
        """Test binomial PMF with known values."""
        # P(X=2) for n=5, p=0.5 should be C(5,2) * 0.5^5 = 10 * 0.03125 = 0.3125
        self.assertAlmostEqual(stats.binomial_pmf(2, 5, 0.5), 0.3125, places=6)

    def test_binomial_pmf_boundary(self):
        """Test binomial PMF at boundaries."""
        # P(X=0) for n=5, p=0.5 = 0.5^5 = 0.03125
        self.assertAlmostEqual(stats.binomial_pmf(0, 5, 0.5), 0.03125, places=6)
        # P(X=5) for n=5, p=0.5 = 0.5^5 = 0.03125
        self.assertAlmostEqual(stats.binomial_pmf(5, 5, 0.5), 0.03125, places=6)

    def test_binomial_pmf_out_of_range(self):
        """Test binomial PMF returns 0 outside valid range."""
        self.assertEqual(stats.binomial_pmf(-1, 5, 0.5), 0.0)
        self.assertEqual(stats.binomial_pmf(6, 5, 0.5), 0.0)

    def test_binomial_pmf_invalid_p_raises(self):
        """Test binomial PMF raises for invalid p."""
        with self.assertRaises(ValueError):
            stats.binomial_pmf(2, 5, 1.5)
        with self.assertRaises(ValueError):
            stats.binomial_pmf(2, 5, -0.1)

    def test_binomial_cdf_cumulative(self):
        """Test binomial CDF is cumulative sum."""
        n, p = 5, 0.5
        expected = sum(stats.binomial_pmf(i, n, p) for i in range(4))
        self.assertAlmostEqual(stats.binomial_cdf(3, n, p), expected, places=6)

    def test_binomial_cdf_full_range(self):
        """Test binomial CDF at n equals 1."""
        self.assertAlmostEqual(stats.binomial_cdf(5, 5, 0.5), 1.0, places=6)

    def test_poisson_pmf_known(self):
        """Test Poisson PMF with known values."""
        import math

        # P(X=3) for lambda=2: e^-2 * 2^3 / 3! = e^-2 * 8 / 6
        expected = math.exp(-2) * 8 / 6
        self.assertAlmostEqual(stats.poisson_pmf(3, 2), expected, places=6)

    def test_poisson_pmf_zero(self):
        """Test Poisson PMF at k=0."""
        import math

        # P(X=0) for lambda=3: e^-3
        self.assertAlmostEqual(stats.poisson_pmf(0, 3), math.exp(-3), places=6)

    def test_poisson_pmf_negative_k(self):
        """Test Poisson PMF returns 0 for negative k."""
        self.assertEqual(stats.poisson_pmf(-1, 2), 0.0)

    def test_poisson_pmf_negative_lambda_raises(self):
        """Test Poisson PMF raises for negative lambda."""
        with self.assertRaises(ValueError):
            stats.poisson_pmf(2, -1)

    def test_poisson_cdf_cumulative(self):
        """Test Poisson CDF is cumulative sum."""
        lambda_ = 3
        expected = sum(stats.poisson_pmf(i, lambda_) for i in range(4))
        self.assertAlmostEqual(stats.poisson_cdf(3, lambda_), expected, places=6)


class TestHypothesisTesting(unittest.TestCase):
    """Tests for t_test and chi_square_test."""

    def test_t_test_identical_samples(self):
        """Test t-test with identical samples gives t_stat near 0."""
        sample = [1, 2, 3, 4, 5]
        t_stat, _ = stats.t_test(sample, sample)
        self.assertAlmostEqual(t_stat, 0.0, places=6)

    def test_t_test_different_samples(self):
        """Test t-test with different samples."""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [6, 7, 8, 9, 10]
        t_stat, p_value = stats.t_test(sample1, sample2)
        self.assertLess(t_stat, 0)  # sample1 mean < sample2 mean
        self.assertLess(p_value, 0.05)  # Significantly different

    def test_t_test_returns_tuple(self):
        """Test t-test returns (t_stat, p_value)."""
        result = stats.t_test([1, 2, 3], [4, 5, 6])
        self.assertEqual(len(result), 2)

    def test_t_test_p_value_bounds(self):
        """Test p-value is between 0 and 1."""
        _, p_value = stats.t_test([1, 2, 3], [4, 5, 6])
        self.assertTrue(0 <= p_value <= 1)

    def test_t_test_empty_raises(self):
        """Test t-test raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.t_test([], [1, 2, 3])

    def test_chi_square_uniform(self):
        """Test chi-square with uniform observed/expected gives low chi2."""
        observed = [10, 10, 10, 10]
        expected = [10, 10, 10, 10]
        chi2, p_value = stats.chi_square_test(observed, expected)
        self.assertAlmostEqual(chi2, 0.0, places=6)

    def test_chi_square_different(self):
        """Test chi-square with different distributions."""
        observed = [20, 15, 10, 5]
        expected = [12.5, 12.5, 12.5, 12.5]
        chi2, p_value = stats.chi_square_test(observed, expected)
        self.assertGreater(chi2, 0)

    def test_chi_square_returns_tuple(self):
        """Test chi-square returns (chi2, p_value)."""
        result = stats.chi_square_test([10, 20], [15, 15])
        self.assertEqual(len(result), 2)

    def test_chi_square_mismatched_raises(self):
        """Test chi-square raises ValueError for mismatched arrays."""
        with self.assertRaises(ValueError):
            stats.chi_square_test([1, 2, 3], [1, 2])

    def test_chi_square_zero_expected_raises(self):
        """Test chi-square raises ValueError for zero expected."""
        with self.assertRaises(ValueError):
            stats.chi_square_test([1, 2, 3], [1, 0, 2])


class TestSampling(unittest.TestCase):
    """Tests for bootstrap_sample and confidence_interval."""

    def test_bootstrap_correct_length(self):
        """Test bootstrap sample returns correct length."""
        data = [1, 2, 3, 4, 5]
        sample = stats.bootstrap_sample(data, 10)
        self.assertEqual(len(sample), 10)

    def test_bootstrap_values_from_data(self):
        """Test bootstrap sample values are from original data."""
        data = [1, 2, 3, 4, 5]
        sample = stats.bootstrap_sample(data, 100)
        for value in sample:
            self.assertIn(value, data)

    def test_bootstrap_empty_data_raises(self):
        """Test bootstrap raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.bootstrap_sample([], 10)

    def test_bootstrap_negative_n_raises(self):
        """Test bootstrap raises ValueError for negative n."""
        with self.assertRaises(ValueError):
            stats.bootstrap_sample([1, 2, 3], -1)

    def test_confidence_interval_returns_tuple(self):
        """Test confidence interval returns (lower, upper) tuple."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = stats.confidence_interval(data, 0.95)
        self.assertEqual(len(result), 2)
        lower, upper = result
        self.assertLess(lower, upper)

    def test_confidence_interval_contains_mean(self):
        """Test confidence interval contains the mean."""
        data = list(range(1, 101))
        lower, upper = stats.confidence_interval(data, 0.95)
        data_mean = stats.mean(data)
        self.assertLessEqual(lower, data_mean)
        self.assertGreaterEqual(upper, data_mean)

    def test_confidence_interval_wider_at_higher_confidence(self):
        """Test higher confidence gives wider interval."""
        data = list(range(1, 51))
        ci_90 = stats.confidence_interval(data, 0.90)
        ci_95 = stats.confidence_interval(data, 0.95)
        ci_99 = stats.confidence_interval(data, 0.99)

        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)

    def test_confidence_interval_single_element(self):
        """Test confidence interval with single element."""
        lower, upper = stats.confidence_interval([5], 0.95)
        self.assertEqual(lower, 5.0)
        self.assertEqual(upper, 5.0)

    def test_confidence_interval_invalid_confidence_raises(self):
        """Test confidence interval raises for invalid confidence."""
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 1)
        with self.assertRaises(ValueError):
            stats.confidence_interval([1, 2, 3], 1.5)

    def test_confidence_interval_empty_raises(self):
        """Test confidence interval raises ValueError for empty data."""
        with self.assertRaises(ValueError):
            stats.confidence_interval([], 0.95)


if __name__ == "__main__":
    unittest.main()
