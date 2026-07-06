"""
test_inference.py

Run with:  pytest test_inference.py -v
Requires: pytest, numpy (regression oracle / RNG only).
"""

import os
import sys
import math
import time
import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conjugate import BetaBinomial
from inference import (
    bayes_factor,
    credible_interval,
    gelman_rubin,
    hdi,
    log_marginal_likelihood,
    map_estimate,
    posterior_summary,
    sequential_update,
)


class TestMapEstimate:
    def test_basic(self):
        assert map_estimate([1, 2, 3], [0.1, 0.5, 0.2]) == 2

    def test_tie_returns_first(self):
        assert map_estimate([1, 2, 3], [0.5, 0.5, 0.1]) == 1

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError):
            map_estimate([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            map_estimate([1, 2], [0.1])

    def test_nan_posterior_value_raises(self):
        with pytest.raises(ValueError):
            map_estimate([1, 2], [0.1, float("nan")])

    def test_negative_posterior_value_raises(self):
        with pytest.raises(ValueError):
            map_estimate([1, 2], [0.1, -0.2])


class TestCredibleInterval:
    def test_contains_reasonable_range(self):
        samples = list(np.random.RandomState(1).normal(0, 1, 1000))
        ci = credible_interval(samples, 0.9)
        assert ci["lower"] < 0 < ci["upper"]

    def test_wider_prob_gives_wider_interval(self):
        samples = list(np.random.RandomState(1).normal(0, 1, 1000))
        ci90 = credible_interval(samples, 0.90)
        ci99 = credible_interval(samples, 0.99)
        assert (ci99["upper"] - ci99["lower"]) > (ci90["upper"] - ci90["lower"])

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            credible_interval([], 0.95)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            credible_interval([1, 2, 3], 1.5)

    def test_nan_in_samples_raises(self):
        with pytest.raises(ValueError):
            credible_interval([1, 2, float("nan")], 0.95)


class TestHdi:
    def test_narrower_than_or_equal_to_equal_tailed_ci_for_symmetric_data(self):
        samples = list(np.random.RandomState(1).normal(0, 1, 2000))
        ci = credible_interval(samples, 0.9)
        hdi_res = hdi(samples, 0.9)
        # For unimodal symmetric data, HDI width should be close to (or
        # narrower than) the equal-tailed CI width.
        assert hdi_res["width"] <= (ci["upper"] - ci["lower"]) + 0.2

    def test_full_coverage_when_interval_size_geq_n(self):
        samples = [1, 2, 3]
        result = hdi(samples, 0.99)
        assert result["lower"] == 1
        assert result["upper"] == 3

    def test_narrower_for_concentrated_cluster(self):
        # A tight cluster plus a couple of outliers: HDI should find the
        # tight cluster, not be dragged wide by the outliers.
        samples = [10.0, 10.1, 9.9, 10.05, 9.95, -50.0, 80.0]
        result = hdi(samples, prob=5 / 7)
        assert result["lower"] >= 9.0
        assert result["upper"] <= 11.0

    def test_insufficient_samples_raises(self):
        with pytest.raises(ValueError):
            hdi([1], 0.95)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hdi([], 0.95)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            hdi([1, 2, float("nan")], 0.95)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            hdi([1, 2, 3], 1.5)


class TestPosteriorSummary:
    def test_basic_fields_present(self):
        samples = list(np.random.RandomState(1).normal(5, 2, 500))
        summ = posterior_summary(samples, 0.95)
        for key in ["mean", "std", "median", "mode_approx", "ci_95", "hdi_95", "n_samples"]:
            assert key in summ

    def test_mean_close_to_true_mean(self):
        samples = list(np.random.RandomState(1).normal(5, 2, 5000))
        summ = posterior_summary(samples, 0.95)
        assert summ["mean"] == pytest.approx(5.0, abs=0.2)

    def test_mode_approx_close_to_true_mode(self):
        samples = list(np.random.RandomState(1).normal(5, 2, 5000))
        summ = posterior_summary(samples, 0.95)
        assert summ["mode_approx"] == pytest.approx(5.0, abs=0.5)

    def test_large_sample_performance(self):
        """Regression test: mode estimation must use a fixed-size KDE grid
        (O(n * grid_size)), not O(n^2) evaluation at every sample point,
        which would take far too long for large MCMC chains."""
        samples = list(np.random.RandomState(2).normal(0, 1, 20000))
        start = time.time()
        posterior_summary(samples, 0.95)
        elapsed = time.time() - start
        assert elapsed < 10.0

    def test_insufficient_samples_raises(self):
        with pytest.raises(ValueError):
            posterior_summary([1], 0.95)

    def test_invalid_ci_prob_raises(self):
        with pytest.raises(ValueError):
            posterior_summary([1, 2, 3], 1.5)

    def test_constant_samples_does_not_crash(self):
        summ = posterior_summary([3.0, 3.0, 3.0, 3.0], 0.95)
        assert summ["mode_approx"] == 3.0


class TestLogMarginalLikelihood:
    def test_uniform_grid_matches_known_integral(self):
        """A standard normal 'prior' with a flat (zero) log-likelihood should
        integrate to ~1 (its own normalization)."""
        grid = list(np.linspace(-6, 6, 1000))
        log_prior = [-0.5 * x ** 2 - 0.5 * math.log(2 * math.pi) for x in grid]
        log_lik = [0.0] * len(grid)
        lml = log_marginal_likelihood(grid, log_prior, log_lik)
        assert math.exp(lml) == pytest.approx(1.0, abs=0.01)

    def test_nonuniform_grid_matches_known_integral(self):
        """Regression test: the old implementation assumed a uniform grid
        (using a single fixed delta) without validating it, silently giving
        wrong answers on non-uniform grids. Trapezoidal quadrature must
        handle non-uniform spacing correctly."""
        rng = np.random.RandomState(3)
        grid = sorted(set(list(rng.uniform(-6, 6, 400)) + [-6.0, 6.0]))
        log_prior = [-0.5 * x ** 2 - 0.5 * math.log(2 * math.pi) for x in grid]
        log_lik = [0.0] * len(grid)
        lml = log_marginal_likelihood(grid, log_prior, log_lik)
        assert math.exp(lml) == pytest.approx(1.0, abs=0.05)

    def test_grid_not_strictly_increasing_raises(self):
        with pytest.raises(ValueError):
            log_marginal_likelihood([1, 1, 2], [0, 0, 0], [0, 0, 0])

    def test_grid_descending_raises(self):
        with pytest.raises(ValueError):
            log_marginal_likelihood([2, 1, 3], [0, 0, 0], [0, 0, 0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            log_marginal_likelihood([1], [0], [0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            log_marginal_likelihood([1, 2], [0], [0, 0])

    def test_nan_input_raises(self):
        with pytest.raises(ValueError):
            log_marginal_likelihood([1, 2], [0, float("nan")], [0, 0])


class TestBayesFactor:
    def test_decisive_for_m1(self):
        result = bayes_factor(math.log(200), math.log(1))
        assert result["evidence"] == "Decisive for M1"

    def test_strong_for_m1(self):
        result = bayes_factor(math.log(50), math.log(1))
        assert result["evidence"] == "Strong for M1"

    def test_no_preference(self):
        result = bayes_factor(math.log(5), math.log(5))
        assert result["evidence"] == "No preference"
        assert result["K"] == pytest.approx(1.0)

    def test_decisive_for_m2(self):
        result = bayes_factor(math.log(1), math.log(200))
        assert result["evidence"] == "Decisive for M2"

    def test_overflow_handled_as_infinity(self):
        result = bayes_factor(10000.0, 0.0)
        assert result["K"] == float("inf")
        assert result["evidence"] == "Decisive for M1"

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            bayes_factor(float("nan"), 0.0)
        with pytest.raises(ValueError):
            bayes_factor(0.0, float("nan"))


class TestSequentialUpdate:
    def test_snapshots_have_expected_length_and_keys(self):
        model = BetaBinomial(1.0, 1.0)
        snaps = sequential_update(model, [(10, 7), (5, 2)])
        assert len(snaps) == 2
        for s in snaps:
            assert set(s.keys()) == {"step", "posterior_mean", "posterior_std"}

    def test_final_state_matches_direct_update(self):
        model1 = BetaBinomial(1.0, 1.0)
        sequential_update(model1, [(10, 7), (5, 2)])

        model2 = BetaBinomial(1.0, 1.0)
        model2.update(15, 9)

        assert model1.alpha == pytest.approx(model2.alpha)
        assert model1.beta == pytest.approx(model2.beta)

    def test_invalid_model_raises(self):
        with pytest.raises(TypeError):
            sequential_update("not a model", [(1, 1)])

    def test_invalid_batch_type_raises(self):
        model = BetaBinomial(1.0, 1.0)
        with pytest.raises(TypeError):
            sequential_update(model, [42])

    def test_wrong_tuple_length_raises(self):
        model = BetaBinomial(1.0, 1.0)
        with pytest.raises(ValueError):
            sequential_update(model, [(10, 7, 3)])


class TestGelmanRubin:
    def test_converged_chains_near_one(self):
        rng = np.random.RandomState(4)
        chains = [list(rng.normal(0, 1, 500)) for _ in range(4)]
        rhat = gelman_rubin(chains)
        assert 0.95 < rhat < 1.05

    def test_diverged_chains_much_greater_than_one(self):
        rng = np.random.RandomState(4)
        chains = [list(rng.normal(-5, 0.1, 500)), list(rng.normal(5, 0.1, 500))]
        rhat = gelman_rubin(chains)
        assert rhat > 1.5

    def test_all_identical_constant_chains_is_exactly_one(self):
        """Regression test: previously, W == 0 always returned infinity, even
        when chains perfectly agree (B == 0 too). That's a trivial converged
        case and should report R-hat = 1.0, not infinity."""
        assert gelman_rubin([[3, 3, 3], [3, 3, 3]]) == 1.0

    def test_zero_within_variance_but_disagreeing_chains_is_infinity(self):
        assert gelman_rubin([[1, 1], [5, 5]]) == float("inf")

    def test_single_chain_raises(self):
        with pytest.raises(ValueError):
            gelman_rubin([[1, 2, 3]])

    def test_short_chain_raises(self):
        with pytest.raises(ValueError):
            gelman_rubin([[1], [2]])

    def test_unequal_length_chains_truncated(self):
        # Should not raise; truncates to shortest chain's length.
        rhat = gelman_rubin([[1, 2, 3, 4, 5], [1, 2, 3]])
        assert isinstance(rhat, float)

    def test_nan_in_chain_raises(self):
        with pytest.raises(ValueError):
            gelman_rubin([[1, 2, float("nan")], [1, 2, 3]])
