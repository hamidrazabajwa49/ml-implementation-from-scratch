"""
test_descriptive_stats.py

Run with:  pytest test_descriptive_stats.py -v
Requires: pytest, numpy, scipy (regression oracle only).
"""

import sys
import os
import math
import numpy as np
import pytest
from scipy import stats as spstats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from descriptive_stats import DescriptiveStats


class TestConstruction:
    def test_basic(self):
        s = DescriptiveStats([3, 1, 2])
        assert s.data == [1, 2, 3]
        assert s.n == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats([])

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            DescriptiveStats([1, "a", 3])

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            DescriptiveStats([1, True, 3])

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats([1, float("nan"), 3])

    def test_repr(self):
        s = DescriptiveStats([1, 2, 3])
        assert "n=3" in repr(s)

    def test_iteration_is_sorted(self):
        s = DescriptiveStats([3, 1, 2])
        assert list(s) == [1, 2, 3]

    def test_len(self):
        assert len(DescriptiveStats([1, 2, 3])) == 3


class TestCentralTendency:
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_mean_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        assert s.mean() == pytest.approx(np.mean(self.DATA))

    def test_median_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        assert s.median() == pytest.approx(np.median(self.DATA))

    def test_median_odd_length(self):
        s = DescriptiveStats([1, 2, 3, 4, 5])
        assert s.median() == pytest.approx(np.median([1, 2, 3, 4, 5]))

    def test_mode(self):
        s = DescriptiveStats(self.DATA)
        assert s.mode() == [4]

    def test_mode_multimodal(self):
        s = DescriptiveStats([1, 1, 2, 2, 3])
        assert s.mode() == [1, 2]

    def test_percentile_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        for p in [0, 10, 25, 50, 75, 90, 100]:
            assert s.percentile(p) == pytest.approx(np.percentile(self.DATA, p))

    def test_percentile_out_of_range_raises(self):
        s = DescriptiveStats(self.DATA)
        with pytest.raises(ValueError):
            s.percentile(-1)
        with pytest.raises(ValueError):
            s.percentile(101)


class TestSpread:
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_variance_population_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        assert s.variance(ddof=0) == pytest.approx(np.var(self.DATA, ddof=0))

    def test_variance_sample_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        assert s.variance(ddof=1) == pytest.approx(np.var(self.DATA, ddof=1))

    def test_variance_invalid_ddof_raises(self):
        s = DescriptiveStats(self.DATA)
        with pytest.raises(ValueError):
            s.variance(ddof=2)

    def test_sample_variance_requires_2_points(self):
        with pytest.raises(ValueError):
            DescriptiveStats([5]).variance(ddof=1)

    def test_std_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        assert s.std(ddof=1) == pytest.approx(np.std(self.DATA, ddof=1))

    def test_data_range(self):
        s = DescriptiveStats(self.DATA)
        assert s.data_range() == pytest.approx(9 - 2)

    def test_iqr_matches_numpy(self):
        s = DescriptiveStats(self.DATA)
        expected = np.percentile(self.DATA, 75) - np.percentile(self.DATA, 25)
        assert s.iqr() == pytest.approx(expected)


class TestShape:
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_skewness_matches_scipy(self):
        s = DescriptiveStats(self.DATA)
        assert s.skewness() == pytest.approx(spstats.skew(self.DATA))

    def test_kurtosis_matches_scipy(self):
        s = DescriptiveStats(self.DATA)
        assert s.kurtosis() == pytest.approx(spstats.kurtosis(self.DATA))

    def test_skewness_constant_data_is_zero(self):
        assert DescriptiveStats([3, 3, 3]).skewness() == 0.0

    def test_kurtosis_constant_data_is_zero(self):
        assert DescriptiveStats([3, 3, 3]).kurtosis() == 0.0


class TestZScores:
    def test_z_scores_mean_zero(self):
        s = DescriptiveStats([1, 2, 3, 4, 5])
        z = s.z_scores()
        assert sum(z) / len(z) == pytest.approx(0.0, abs=1e-9)

    def test_z_scores_preserve_input_order(self):
        s = DescriptiveStats([5, 1, 3])
        z = s.z_scores()
        assert len(z) == 3
        # original order is [5, 1, 3]; largest value should have largest z
        assert z[0] > z[2] > z[1]

    def test_z_scores_constant_data_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats([3, 3, 3]).z_scores()


class TestCovarianceCorrelation:
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 4, 5]

    def test_covariance_matches_numpy(self):
        expected = np.cov(self.X, self.Y, ddof=1)[0, 1]
        assert DescriptiveStats.covariance(self.X, self.Y, ddof=1) == pytest.approx(expected)

    def test_correlation_matches_numpy(self):
        expected = np.corrcoef(self.X, self.Y)[0, 1]
        assert DescriptiveStats.correlation(self.X, self.Y) == pytest.approx(expected)

    def test_correlation_constant_series_is_zero(self):
        assert DescriptiveStats.correlation([1, 1, 1], [1, 2, 3]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats.covariance([1, 2], [1, 2, 3])
        with pytest.raises(ValueError):
            DescriptiveStats.correlation([1, 2], [1, 2, 3])

    def test_correlation_needs_2_points(self):
        with pytest.raises(ValueError):
            DescriptiveStats.correlation([1], [1])

    def test_covariance_matrix_matches_numpy(self):
        dataset = [self.X, self.Y, [5, 3, 2, 1, 1]]
        cm = DescriptiveStats.covariance_matrix(dataset)
        expected = np.cov(dataset, ddof=1)
        for i in range(3):
            for j in range(3):
                assert cm.rows[i].components[j] == pytest.approx(expected[i, j])

    def test_covariance_matrix_is_symmetric(self):
        dataset = [self.X, self.Y, [5, 3, 2, 1, 1]]
        cm = DescriptiveStats.covariance_matrix(dataset)
        for i in range(3):
            for j in range(3):
                assert cm.rows[i].components[j] == pytest.approx(cm.rows[j].components[i])

    def test_correlation_matrix_matches_numpy(self):
        dataset = [self.X, self.Y, [5, 3, 2, 1, 1]]
        corr = DescriptiveStats.correlation_matrix(dataset)
        expected = np.corrcoef(dataset)
        for i in range(3):
            for j in range(3):
                assert corr.rows[i].components[j] == pytest.approx(expected[i, j])

    def test_correlation_matrix_diagonal_is_one(self):
        dataset = [self.X, self.Y]
        corr = DescriptiveStats.correlation_matrix(dataset)
        assert corr.rows[0].components[0] == 1.0
        assert corr.rows[1].components[1] == 1.0

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats.covariance_matrix([])
        with pytest.raises(ValueError):
            DescriptiveStats.correlation_matrix([])


class TestSummary:
    def test_summary_contains_key_stats(self):
        s = DescriptiveStats([1, 2, 3, 4, 5])
        text = s.summary()
        assert "Mean:" in text
        assert "Median:" in text
        assert "Std Dev:" in text
