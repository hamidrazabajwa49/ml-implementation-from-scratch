"""
test_discrete.py

Run with:  pytest test_discrete.py -v
Requires: pytest, scipy (regression oracle only).
"""

import os
import sys
import math
import pytest
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from discrete import BernoulliDistribution, BinomialDistribution, PoissonDistribution


class TestBernoulli:
    def test_mean_variance(self):
        b = BernoulliDistribution(0.3)
        assert b.mean() == pytest.approx(0.3)
        assert b.variance() == pytest.approx(0.21)

    def test_pmf(self):
        b = BernoulliDistribution(0.3)
        assert b.pmf(0) == pytest.approx(0.7)
        assert b.pmf(1) == pytest.approx(0.3)
        assert b.pmf(2) == 0.0

    def test_cdf(self):
        b = BernoulliDistribution(0.3)
        assert b.cdf(-1) == 0.0
        assert b.cdf(0) == pytest.approx(0.7)
        assert b.cdf(1) == 1.0

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.69, 0.71, 0.99, 1.0])
    def test_ppf_matches_scipy(self, q):
        b = BernoulliDistribution(0.3)
        assert b.ppf(q) == stats.bernoulli(0.3).ppf(q)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            BernoulliDistribution(1.5)
        with pytest.raises(ValueError):
            BernoulliDistribution(-0.1)

    def test_bool_p_rejected(self):
        with pytest.raises(TypeError):
            BernoulliDistribution(True)

    def test_nan_p_rejected(self):
        with pytest.raises(ValueError):
            BernoulliDistribution(float("nan"))

    def test_sample_reproducible_with_seed(self):
        b = BernoulliDistribution(0.5)
        assert b.sample(10, seed=1) == b.sample(10, seed=1)

    def test_sample_values_in_range(self):
        b = BernoulliDistribution(0.5)
        assert all(v in (0, 1) for v in b.sample(50, seed=1))


class TestBinomial:
    @pytest.mark.parametrize("n,p", [(10, 0.3), (1, 0.5), (50, 0.9), (0, 0.5)])
    def test_mean_variance_matches_scipy(self, n, p):
        b = BinomialDistribution(n, p)
        sp = stats.binom(n, p)
        assert b.mean() == pytest.approx(sp.mean())
        assert b.variance() == pytest.approx(sp.var())

    def test_pmf_matches_scipy(self):
        b = BinomialDistribution(10, 0.3)
        sp = stats.binom(10, 0.3)
        for k in range(11):
            assert b.pmf(k) == pytest.approx(sp.pmf(k), abs=1e-9)

    def test_pmf_out_of_range_is_zero(self):
        b = BinomialDistribution(10, 0.3)
        assert b.pmf(-1) == 0.0
        assert b.pmf(11) == 0.0

    def test_cdf_matches_scipy(self):
        b = BinomialDistribution(10, 0.3)
        sp = stats.binom(10, 0.3)
        for k in range(10):
            assert b.cdf(k) == pytest.approx(sp.cdf(k), abs=1e-9)

    def test_cdf_boundary(self):
        b = BinomialDistribution(10, 0.3)
        assert b.cdf(-1) == 0.0
        assert b.cdf(10) == 1.0

    @pytest.mark.parametrize("q", [0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_ppf_matches_scipy(self, q):
        b = BinomialDistribution(10, 0.3)
        assert b.ppf(q) == stats.binom(10, 0.3).ppf(q)

    def test_n_zero_edge_case(self):
        b = BinomialDistribution(0, 0.5)
        assert b.pmf(0) == 1.0
        assert b.cdf(0) == 1.0

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            BinomialDistribution(-1, 0.5)

    def test_non_int_n_raises(self):
        with pytest.raises(TypeError):
            BinomialDistribution(5.5, 0.5)

    def test_bool_n_rejected(self):
        with pytest.raises(TypeError):
            BinomialDistribution(True, 0.5)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            BinomialDistribution(10, 1.5)

    def test_non_int_k_in_pmf_raises(self):
        b = BinomialDistribution(10, 0.3)
        with pytest.raises(TypeError):
            b.pmf(2.5)

    def test_sample_reproducible_with_seed(self):
        b = BinomialDistribution(20, 0.4)
        assert b.sample(5, seed=123) == b.sample(5, seed=123)

    def test_sample_values_within_bounds(self):
        b = BinomialDistribution(20, 0.4)
        samples = b.sample(50, seed=1)
        assert all(0 <= s <= 20 for s in samples)

    def test_sample_num_samples_validation(self):
        with pytest.raises(ValueError):
            BinomialDistribution(10, 0.5).sample(0)


class TestPoisson:
    @pytest.mark.parametrize("lam", [0.0, 1.0, 5.0, 20.0, 50.0, 200.0, 1000.0])
    def test_pmf_matches_scipy(self, lam):
        p = PoissonDistribution(lam)
        sp = stats.poisson(lam)
        for k in [0, 1, int(lam), int(lam) + 5]:
            if k >= 0:
                assert p.pmf(k) == pytest.approx(sp.pmf(k), abs=1e-6, rel=1e-6)

    def test_pmf_lambda_zero(self):
        p = PoissonDistribution(0.0)
        assert p.pmf(0) == 1.0
        assert p.pmf(1) == 0.0

    def test_cdf_matches_scipy(self):
        p = PoissonDistribution(5.0)
        sp = stats.poisson(5.0)
        for k in range(15):
            assert p.cdf(k) == pytest.approx(sp.cdf(k), abs=1e-8)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            PoissonDistribution(-1.0)

    def test_bool_lambda_rejected(self):
        with pytest.raises(TypeError):
            PoissonDistribution(True)

    def test_infinite_lambda_raises(self):
        with pytest.raises(ValueError):
            PoissonDistribution(float("inf"))

    def test_nan_lambda_raises(self):
        with pytest.raises(ValueError):
            PoissonDistribution(float("nan"))

    def test_non_int_k_raises(self):
        p = PoissonDistribution(5.0)
        with pytest.raises(TypeError):
            p.pmf(2.5)

    def test_large_lambda_sampling_does_not_hang(self):
        """Regression test: Knuth's algorithm underflows (exp(-lam) == 0.0)
        for lam above ~700, which without the chunk-decomposition fix
        would loop for an extremely long time instead of the few
        milliseconds this should take."""
        import time

        p = PoissonDistribution(1000.0)
        start = time.time()
        samples = p.sample(100, seed=42)
        elapsed = time.time() - start
        assert elapsed < 5.0
        sample_mean = sum(samples) / len(samples)
        # Loose sanity check: sample mean should be in the right ballpark
        # (Poisson(1000) has std ~31.6, so 10 std devs is generous).
        assert abs(sample_mean - 1000.0) < 316

    def test_large_lambda_pmf_still_accurate(self):
        p = PoissonDistribution(500.0)
        sp = stats.poisson(500.0)
        assert p.pmf(500) == pytest.approx(sp.pmf(500), rel=1e-6)

    def test_sample_zero_lambda(self):
        p = PoissonDistribution(0.0)
        assert p.sample(5, seed=1) == [0, 0, 0, 0, 0]

    def test_sample_reproducible_with_seed(self):
        p = PoissonDistribution(10.0)
        assert p.sample(5, seed=7) == p.sample(5, seed=7)

    def test_sample_num_samples_validation(self):
        with pytest.raises(ValueError):
            PoissonDistribution(5.0).sample(0)

    def test_ppf_matches_scipy(self):
        p = PoissonDistribution(5.0)
        sp = stats.poisson(5.0)
        for q in [0.1, 0.5, 0.9, 0.99]:
            assert p.ppf(q) == sp.ppf(q)
