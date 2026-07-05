"""
test_continuous.py

Run with:  pytest test_continuous.py -v
Requires: pytest, scipy (regression oracle only).
"""

import os
import sys
import math
import pytest
from scipy import stats


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from continuous import (
    BetaDistribution,
    Chi2Distribution,
    FDistribution,
    GammaDistribution,
    NormalDistribution,
    TDistribution,
)


class TestNormal:
    def test_pdf_matches_scipy(self):
        n = NormalDistribution(2, 3)
        sp = stats.norm(2, 3)
        for x in [-5, -1, 0, 1, 2, 5, 10]:
            assert n.pdf(x) == pytest.approx(sp.pdf(x), abs=1e-9)

    def test_cdf_matches_scipy(self):
        n = NormalDistribution(2, 3)
        sp = stats.norm(2, 3)
        for x in [-5, -1, 0, 1, 2, 5, 10]:
            assert n.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-9)

    def test_sf_matches_scipy(self):
        n = NormalDistribution(0, 1)
        sp = stats.norm(0, 1)
        assert n.sf(3.0) == pytest.approx(sp.sf(3.0), rel=1e-6)

    @pytest.mark.parametrize("p", [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999])
    def test_ppf_matches_scipy_high_precision(self, p):
        n = NormalDistribution(2, 3)
        sp = stats.norm(2, 3)
        assert n.ppf(p) == pytest.approx(sp.ppf(p), abs=1e-8)

    def test_ppf_out_of_range_raises(self):
        n = NormalDistribution()
        with pytest.raises(ValueError):
            n.ppf(0.0)
        with pytest.raises(ValueError):
            n.ppf(1.0)

    def test_nonpositive_sigma_raises(self):
        with pytest.raises(ValueError):
            NormalDistribution(0, 0)
        with pytest.raises(ValueError):
            NormalDistribution(0, -1)

    def test_nan_mu_raises(self):
        with pytest.raises(ValueError):
            NormalDistribution(float("nan"), 1)

    def test_sample_reproducible_with_seed(self):
        n = NormalDistribution(0, 1)
        assert n.sample(5, seed=1) == n.sample(5, seed=1)

    def test_sample_mean_close_to_true_mean(self):
        n = NormalDistribution(5, 2)
        s = n.sample(5000, seed=1)
        assert sum(s) / len(s) == pytest.approx(5, abs=0.3)


class TestT:
    @pytest.mark.parametrize("df", [1.5, 3, 10, 50])
    def test_pdf_cdf_match_scipy(self, df):
        t = TDistribution(df)
        sp = stats.t(df)
        for x in [-3, -1, 0, 1, 3]:
            assert t.pdf(x) == pytest.approx(sp.pdf(x), abs=1e-8)
            assert t.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-6)

    @pytest.mark.parametrize("df", [3, 10, 50])
    def test_ppf_matches_scipy(self, df):
        t = TDistribution(df)
        sp = stats.t(df)
        for p in [0.05, 0.5, 0.95]:
            assert t.ppf(p) == pytest.approx(sp.ppf(p), abs=1e-4)

    def test_variance_matches_scipy_when_defined(self):
        t = TDistribution(10)
        assert t.variance() == pytest.approx(stats.t(10).var())

    def test_mean_undefined_for_df_leq_1_raises(self):
        with pytest.raises(ValueError):
            TDistribution(1).mean()

    def test_variance_infinite_for_1_lt_df_leq_2(self):
        assert TDistribution(1.5).variance() == math.inf

    def test_variance_undefined_for_df_leq_1_raises(self):
        with pytest.raises(ValueError):
            TDistribution(1).variance()

    def test_nonpositive_df_raises(self):
        with pytest.raises(ValueError):
            TDistribution(0)
        with pytest.raises(ValueError):
            TDistribution(-1)

    def test_bool_df_rejected(self):
        with pytest.raises(TypeError):
            TDistribution(True)

    def test_p_value_two_sided(self):
        t = TDistribution(10)
        pv = t.p_value(2.0, "two-sided")
        assert pv == pytest.approx(2 * t.sf(2.0))

    def test_p_value_invalid_alternative_raises(self):
        t = TDistribution(10)
        with pytest.raises(ValueError):
            t.p_value(1.0, "bogus")

    def test_sample_reproducible_with_seed(self):
        t = TDistribution(10)
        assert t.sample(5, seed=42) == t.sample(5, seed=42)

    def test_sample_mean_close_to_zero(self):
        t = TDistribution(10)
        s = t.sample(5000, seed=1)
        assert sum(s) / len(s) == pytest.approx(0.0, abs=0.3)


class TestChi2:
    @pytest.mark.parametrize("df", [1, 2, 3, 10, 50])
    def test_mean_variance_match_scipy(self, df):
        c = Chi2Distribution(df)
        sp = stats.chi2(df)
        assert c.mean() == pytest.approx(sp.mean())
        assert c.variance() == pytest.approx(sp.var())

    @pytest.mark.parametrize("df", [1, 2, 3, 10, 50])
    def test_cdf_matches_scipy(self, df):
        c = Chi2Distribution(df)
        sp = stats.chi2(df)
        for x in [0.1, 1, 5, 20]:
            assert c.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-6)

    def test_ppf_matches_scipy(self):
        c = Chi2Distribution(5)
        sp = stats.chi2(5)
        for p in [0.05, 0.5, 0.95]:
            assert c.ppf(p) == pytest.approx(sp.ppf(p), abs=1e-4)

    def test_pdf_zero_boundary_df_greater_than_2(self):
        assert Chi2Distribution(4).pdf(0) == 0.0

    def test_pdf_zero_boundary_df_equals_2(self):
        assert Chi2Distribution(2).pdf(0) == pytest.approx(0.5)

    def test_pdf_zero_boundary_df_less_than_2_is_infinite(self):
        assert Chi2Distribution(1).pdf(0) == math.inf

    def test_pdf_negative_x_is_zero(self):
        assert Chi2Distribution(3).pdf(-1) == 0.0

    def test_nonpositive_df_raises(self):
        with pytest.raises(ValueError):
            Chi2Distribution(0)

    def test_sample_reproducible_with_seed(self):
        c = Chi2Distribution(5)
        assert c.sample(5, seed=1) == c.sample(5, seed=1)

    def test_sample_mean_close_to_df(self):
        c = Chi2Distribution(5)
        s = c.sample(5000, seed=1)
        assert sum(s) / len(s) == pytest.approx(5, abs=0.5)


class TestF:
    @pytest.mark.parametrize("df1,df2", [(3, 10), (5, 20), (1, 5)])
    def test_cdf_matches_scipy(self, df1, df2):
        f = FDistribution(df1, df2)
        sp = stats.f(df1, df2)
        for x in [0.5, 1, 2, 5]:
            assert f.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-6)

    def test_mean_matches_scipy_when_defined(self):
        f = FDistribution(5, 10)
        assert f.mean() == pytest.approx(stats.f(5, 10).mean())

    def test_mean_undefined_for_df2_leq_2_raises(self):
        with pytest.raises(ValueError):
            FDistribution(5, 2).mean()

    def test_variance_matches_scipy_when_defined(self):
        f = FDistribution(5, 10)
        assert f.variance() == pytest.approx(stats.f(5, 10).var(), rel=1e-6)

    def test_variance_undefined_for_df2_leq_4_raises(self):
        with pytest.raises(ValueError):
            FDistribution(5, 4).variance()

    def test_pdf_negative_is_zero(self):
        assert FDistribution(3, 5).pdf(-1) == 0.0

    def test_nonpositive_df_raises(self):
        with pytest.raises(ValueError):
            FDistribution(0, 5)
        with pytest.raises(ValueError):
            FDistribution(5, 0)

    def test_sample_reproducible_with_seed(self):
        f = FDistribution(5, 10)
        assert f.sample(5, seed=1) == f.sample(5, seed=1)


class TestBeta:
    def test_cdf_exact_matches_scipy(self):
        """Regression test: cdf must use the exact incomplete beta function,
        not the old trapezoidal-integration approximation."""
        b = BetaDistribution(2, 5)
        sp = stats.beta(2, 5)
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert b.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-9)

    def test_ppf_matches_scipy(self):
        b = BetaDistribution(2, 5)
        sp = stats.beta(2, 5)
        for p in [0.1, 0.5, 0.9]:
            assert b.ppf(p) == pytest.approx(sp.ppf(p), abs=1e-6)

    def test_mean_variance_match_scipy(self):
        b = BetaDistribution(2, 5)
        sp = stats.beta(2, 5)
        assert b.mean() == pytest.approx(sp.mean())
        assert b.variance() == pytest.approx(sp.var())

    def test_pdf_boundary_alpha_less_than_1(self):
        b = BetaDistribution(0.5, 2)
        assert b.pdf(0.0) == math.inf

    def test_pdf_boundary_alpha_geq_1(self):
        b = BetaDistribution(2, 2)
        assert b.pdf(0.0) == 0.0

    def test_pdf_outside_support_is_zero(self):
        b = BetaDistribution(2, 2)
        assert b.pdf(-0.1) == 0.0
        assert b.pdf(1.1) == 0.0

    def test_cdf_boundary(self):
        b = BetaDistribution(2, 3)
        assert b.cdf(0.0) == 0.0
        assert b.cdf(1.0) == 1.0

    def test_nonpositive_shape_raises(self):
        with pytest.raises(ValueError):
            BetaDistribution(0, 2)
        with pytest.raises(ValueError):
            BetaDistribution(2, -1)

    def test_sample_reproducible_with_seed(self):
        b = BetaDistribution(2, 5)
        assert b.sample(5, seed=1) == b.sample(5, seed=1)

    def test_sample_mean_close_to_true_mean(self):
        b = BetaDistribution(2, 5)
        s = b.sample(3000, seed=1)
        assert sum(s) / len(s) == pytest.approx(b.mean(), abs=0.02)

    def test_sample_values_in_unit_interval(self):
        b = BetaDistribution(2, 5)
        s = b.sample(200, seed=1)
        assert all(0.0 <= v <= 1.0 for v in s)


class TestGamma:
    def test_cdf_exact_matches_scipy(self):
        """Regression test: cdf must use the exact incomplete gamma function,
        not the old trapezoidal-integration approximation."""
        g = GammaDistribution(3, 2)
        sp = stats.gamma(3, scale=1 / 2)
        for x in [0.5, 1, 2, 5]:
            assert g.cdf(x) == pytest.approx(sp.cdf(x), abs=1e-9)

    def test_ppf_matches_scipy(self):
        g = GammaDistribution(3, 2)
        sp = stats.gamma(3, scale=1 / 2)
        for p in [0.1, 0.5, 0.9]:
            assert g.ppf(p) == pytest.approx(sp.ppf(p), abs=1e-6)

    def test_mean_variance_match_scipy(self):
        g = GammaDistribution(3, 2)
        sp = stats.gamma(3, scale=1 / 2)
        assert g.mean() == pytest.approx(sp.mean())
        assert g.variance() == pytest.approx(sp.var())

    def test_pdf_boundary_alpha_less_than_1(self):
        g = GammaDistribution(0.5, 2)
        assert g.pdf(0.0) == math.inf

    def test_pdf_negative_is_zero(self):
        assert GammaDistribution(2, 2).pdf(-1) == 0.0

    def test_nonpositive_shape_raises(self):
        with pytest.raises(ValueError):
            GammaDistribution(0, 2)
        with pytest.raises(ValueError):
            GammaDistribution(2, -1)

    def test_sample_reproducible_with_seed(self):
        g = GammaDistribution(3, 2)
        assert g.sample(5, seed=1) == g.sample(5, seed=1)

    def test_sample_mean_close_to_true_mean(self):
        g = GammaDistribution(3, 2)
        s = g.sample(5000, seed=1)
        assert sum(s) / len(s) == pytest.approx(g.mean(), abs=0.1)

    def test_cdf_speed_is_fast(self):
        """Regression test: cdf should be O(iterations of continued fraction),
        not O(n_steps=2000) trapezoidal quadrature. 2000 calls should take
        well under a second."""
        import time

        g = GammaDistribution(3, 2)
        start = time.time()
        for _ in range(2000):
            g.cdf(0.5)
        assert time.time() - start < 2.0
