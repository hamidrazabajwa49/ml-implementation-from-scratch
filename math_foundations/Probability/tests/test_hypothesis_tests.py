"""
test_hypothesis_tests.py

Run with:  pytest test_hypothesis_tests.py -v
Requires: pytest, numpy, scipy (regression oracle only).
"""

import sys
import os
import math
import warnings
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hypothesis_tests import (
    anova_oneway,
    chisquare_gof,
    chisquare_independence,
    confidence_interval,
    ttest_1samp,
    ttest_ind,
    ttest_paired,
    ztest_1samp,
)


class TestTtest1Samp:
    DATA = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2, 5.4, 4.7]

    def test_matches_scipy(self):
        r = ttest_1samp(self.DATA, 5.0)
        sp = stats.ttest_1samp(self.DATA, 5.0)
        assert r["t_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue, abs=1e-6)

    def test_alternative_greater(self):
        r = ttest_1samp(self.DATA, 5.0, alternative="greater")
        sp = stats.ttest_1samp(self.DATA, 5.0, alternative="greater")
        assert r["p_value"] == pytest.approx(sp.pvalue, abs=1e-6)

    def test_alternative_less(self):
        r = ttest_1samp(self.DATA, 5.0, alternative="less")
        sp = stats.ttest_1samp(self.DATA, 5.0, alternative="less")
        assert r["p_value"] == pytest.approx(sp.pvalue, abs=1e-6)

    def test_cohens_d_present(self):
        r = ttest_1samp(self.DATA, 5.0)
        expected_d = (np.mean(self.DATA) - 5.0) / np.std(self.DATA, ddof=1)
        assert r["cohens_d"] == pytest.approx(expected_d)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            ttest_1samp([1], 5.0)

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            ttest_1samp([3, 3, 3], 5.0)

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError):
            ttest_1samp([1, 2], 5.0, alternative="bogus")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ttest_1samp([1, 2], 5.0, alpha=1.5)

    def test_conclusion_field(self):
        r = ttest_1samp(self.DATA, 5.0)
        assert r["conclusion"] in ("Reject H0", "Fail to reject H0")


class TestTtestInd:
    D1 = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2]
    D2 = [4.5, 4.6, 4.8, 4.4, 4.7, 4.9]

    def test_matches_scipy_welch(self):
        r = ttest_ind(self.D1, self.D2)
        sp = stats.ttest_ind(self.D1, self.D2, equal_var=False)
        assert r["t_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue, abs=1e-6)
        assert r["df"] == pytest.approx(sp.df, abs=1e-6)

    def test_cohens_d_present(self):
        r = ttest_ind(self.D1, self.D2)
        assert isinstance(r["cohens_d"], float)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            ttest_ind([1], [1, 2, 3])
        with pytest.raises(ValueError):
            ttest_ind([1, 2, 3], [1])

    def test_both_zero_variance_same_mean_raises(self):
        with pytest.raises(ValueError):
            ttest_ind([3, 3, 3], [3, 3, 3])


class TestTtestPaired:
    def test_matches_scipy(self):
        before = [10, 12, 9, 11, 13]
        after = [12, 13, 10, 12, 15]
        r = ttest_paired(before, after)
        sp = stats.ttest_rel(before, after)
        assert r["t_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue, abs=1e-6)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ttest_paired([1, 2], [1, 2, 3])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ttest_paired([], [])


class TestZTest:
    def test_matches_manual_calculation(self):
        data = [102, 98, 101, 99, 103, 97, 100]
        pop_mean, pop_std = 100, 5
        r = ztest_1samp(data, pop_mean, pop_std)
        n = len(data)
        z_expected = (np.mean(data) - pop_mean) / (pop_std / math.sqrt(n))
        p_expected = 2 * stats.norm.sf(abs(z_expected))
        assert r["z_statistic"] == pytest.approx(z_expected)
        assert r["p_value"] == pytest.approx(p_expected)

    def test_nonpositive_pop_std_raises(self):
        with pytest.raises(ValueError):
            ztest_1samp([1, 2, 3], 0, pop_std=0)
        with pytest.raises(ValueError):
            ztest_1samp([1, 2, 3], 0, pop_std=-1)

    def test_alternative_variants(self):
        data = [102, 98, 101, 99, 103]
        r_two = ztest_1samp(data, 100, 5, alternative="two-sided")
        r_greater = ztest_1samp(data, 100, 5, alternative="greater")
        r_less = ztest_1samp(data, 100, 5, alternative="less")
        assert r_greater["p_value"] + r_less["p_value"] == pytest.approx(1.0, abs=1e-9)


class TestConfidenceInterval:
    DATA = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2, 5.4, 4.7]

    def test_matches_scipy(self):
        ci = confidence_interval(self.DATA, 0.95)
        mean, se = np.mean(self.DATA), stats.sem(self.DATA)
        tcrit = stats.t.ppf(0.975, len(self.DATA) - 1)
        assert ci["lower"] == pytest.approx(mean - tcrit * se, abs=1e-6)
        assert ci["upper"] == pytest.approx(mean + tcrit * se, abs=1e-6)

    def test_mean_is_inside_interval(self):
        ci = confidence_interval(self.DATA, 0.95)
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            confidence_interval(self.DATA, 1.5)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            confidence_interval([1], 0.95)


class TestChiSquareGof:
    def test_matches_scipy_uniform_expected(self):
        obs = [18, 22, 20, 15, 25]
        r = chisquare_gof(obs)
        sp = stats.chisquare(obs)
        assert r["chi2_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue)

    def test_matches_scipy_custom_expected(self):
        obs = [18, 22, 20, 15, 25]
        exp = [20, 20, 20, 20, 20]
        r = chisquare_gof(obs, expected=exp)
        sp = stats.chisquare(obs, exp)
        assert r["chi2_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue)

    def test_empty_observed_raises(self):
        with pytest.raises(ValueError):
            chisquare_gof([])

    def test_negative_observed_raises(self):
        with pytest.raises(ValueError):
            chisquare_gof([-1, 2, 3])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chisquare_gof([1, 2, 3], expected=[1, 2])

    def test_nonpositive_expected_raises(self):
        with pytest.raises(ValueError):
            chisquare_gof([1, 2, 3], expected=[1, 0, 2])

    def test_single_category_warns_and_returns_p1(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = chisquare_gof([10])
            assert r["p_value"] == 1.0
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            chisquare_gof([0, 0, 0])


class TestChiSquareIndependence:
    def test_matches_scipy(self):
        table = [[10, 20, 30], [6, 9, 17]]
        r = chisquare_independence(table)
        sp = stats.chi2_contingency(table, correction=False)
        assert r["chi2_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue)
        assert r["df"] == sp.dof

    def test_empty_table_raises(self):
        with pytest.raises(ValueError):
            chisquare_independence([])

    def test_ragged_table_raises(self):
        with pytest.raises(ValueError):
            chisquare_independence([[1, 2], [3]])

    def test_non_sequence_rows_raises(self):
        with pytest.raises(TypeError):
            chisquare_independence([5, 6])

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            chisquare_independence([[1, -2], [3, 4]])

    def test_zero_row_raises(self):
        with pytest.raises(ValueError):
            chisquare_independence([[0, 0], [3, 4]])


class TestAnovaOneway:
    def test_matches_scipy(self):
        g1, g2, g3 = [1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]
        r = anova_oneway(g1, g2, g3)
        sp = stats.f_oneway(g1, g2, g3)
        assert r["F_statistic"] == pytest.approx(sp.statistic)
        assert r["p_value"] == pytest.approx(sp.pvalue)

    def test_group_means_correct(self):
        g1, g2 = [1, 2, 3], [4, 5, 6]
        r = anova_oneway(g1, g2)
        assert r["group_means"] == pytest.approx([2.0, 5.0])

    def test_single_group_raises(self):
        with pytest.raises(ValueError):
            anova_oneway([1, 2, 3])

    def test_empty_group_raises(self):
        with pytest.raises(ValueError):
            anova_oneway([], [1, 2, 3])

    def test_single_observation_group_raises(self):
        with pytest.raises(ValueError):
            anova_oneway([1], [1, 2, 3])

    def test_all_identical_groups_gives_nan_f(self):
        r = anova_oneway([1, 1], [1, 1])
        assert math.isnan(r["F_statistic"])
        assert r["reject_H0"] is False

    def test_zero_within_variance_nonzero_between(self):
        r = anova_oneway([1, 1], [5, 5])
        assert r["F_statistic"] == math.inf
        assert r["p_value"] == 0.0
        assert r["reject_H0"] is True
