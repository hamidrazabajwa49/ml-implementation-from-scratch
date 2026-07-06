"""
test_samplers.py

Run with:  pytest test_samplers.py -v
Requires: pytest, numpy (regression oracle / RNG only).
"""

import os
import sys
import logging
import math
import random
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from samplers import (
    effective_sample_size,
    grid_approximation,
    metropolis_hastings,
    rejection_sampling,
)


class TestGridApproximation:
    def setup_method(self):
        random.seed(0)

    def test_posterior_normalizes_to_one(self):
        grid = list(np.linspace(0, 1, 1000))
        result = grid_approximation(lambda t: 1.0, lambda t: t ** 7 * (1 - t) ** 3, grid)
        integral = sum(result["posterior"]) * (grid[1] - grid[0])
        assert integral == pytest.approx(1.0, abs=1e-6)

    def test_posterior_mean_matches_beta_binomial_conjugacy(self):
        """Uniform prior + Binomial(10, k=7) likelihood should give a
        posterior with mean matching Beta(8, 4) = 8/12."""
        grid = list(np.linspace(0, 1, 1000))
        result = grid_approximation(lambda t: 1.0, lambda t: t ** 7 * (1 - t) ** 3, grid)
        delta = grid[1] - grid[0]
        post_mean = sum(g * p for g, p in zip(grid, result["posterior"])) * delta
        assert post_mean == pytest.approx(8 / 12, abs=0.01)

    def test_descending_grid_raises(self):
        """Regression test: a descending grid used to silently flip the sign
        of the normalization constant instead of raising."""
        grid = list(reversed(np.linspace(0, 1, 100)))
        with pytest.raises(ValueError, match="strictly increasing"):
            grid_approximation(lambda t: 1.0, lambda t: 1.0, grid)

    def test_non_monotonic_grid_raises(self):
        with pytest.raises(ValueError):
            grid_approximation(lambda t: 1.0, lambda t: 1.0, [0.1, 0.1, 0.2, 0.3])

    def test_non_uniform_grid_raises(self):
        with pytest.raises(ValueError, match="uniformly spaced"):
            grid_approximation(lambda t: 1.0, lambda t: 1.0, [0.0, 0.1, 0.5, 1.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            grid_approximation(lambda t: 1.0, lambda t: 1.0, [0.5])

    def test_zero_everywhere_raises(self):
        grid = list(np.linspace(0, 1, 10))
        with pytest.raises(ValueError):
            grid_approximation(lambda t: 0.0, lambda t: 1.0, grid)


class TestRejectionSampling:
    def setup_method(self):
        random.seed(0)

    def test_samples_match_target_mean(self):
        """Target: triangular density f(x) = 2x on [0,1], mean = 2/3."""
        target = lambda x: 2 * x if 0 <= x <= 1 else 0.0
        proposal_sampler = lambda: random.random()
        proposal_fn = lambda x: 1.0 if 0 <= x <= 1 else 0.0
        result = rejection_sampling(target, proposal_sampler, proposal_fn, M=2.0, n_samples=5000)
        mean_est = sum(result["samples"]) / len(result["samples"])
        assert mean_est == pytest.approx(2 / 3, abs=0.03)

    def test_acceptance_rate_matches_theory(self):
        """Acceptance rate should equal integral(target)/M = 1/M for a properly
        normalized target and M=2."""
        target = lambda x: 2 * x if 0 <= x <= 1 else 0.0
        proposal_sampler = lambda: random.random()
        proposal_fn = lambda x: 1.0 if 0 <= x <= 1 else 0.0
        result = rejection_sampling(target, proposal_sampler, proposal_fn, M=2.0, n_samples=3000)
        assert result["acceptance_rate"] == pytest.approx(0.5, abs=0.05)

    def test_bound_violation_logs_warning(self, caplog):
        """Regression test: an M too small to bound the target must emit a
        diagnostic warning (accepted samples would otherwise silently not
        follow the target distribution)."""
        target = lambda x: 2 * x if 0 <= x <= 1 else 0.0
        proposal_sampler = lambda: random.random()
        proposal_fn = lambda x: 1.0 if 0 <= x <= 1 else 0.0
        with caplog.at_level(logging.WARNING):
            rejection_sampling(target, proposal_sampler, proposal_fn, M=1.0, n_samples=50, max_iter=100000)
        assert any("exceeds M*proposal_fn" in r.message for r in caplog.records)

    def test_nonpositive_M_raises(self):
        with pytest.raises(ValueError):
            rejection_sampling(lambda x: 1.0, lambda: 0.5, lambda x: 1.0, M=0, n_samples=10)

    def test_nonpositive_n_samples_raises(self):
        with pytest.raises(ValueError):
            rejection_sampling(lambda x: 1.0, lambda: 0.5, lambda x: 1.0, M=1.0, n_samples=0)

    def test_max_iter_exceeded_raises_runtime_error(self):
        # target always 0 in support region -> proposal always p_x<=0 rejected forever... use impossible target
        target = lambda x: 0.0
        proposal_sampler = lambda: random.random()
        proposal_fn = lambda x: 1.0
        with pytest.raises(RuntimeError):
            rejection_sampling(target, proposal_sampler, proposal_fn, M=1.0, n_samples=5, max_iter=50)

    def test_target_fn_exception_wrapped(self):
        def bad_target(x):
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="target_fn raised"):
            rejection_sampling(bad_target, lambda: 0.5, lambda x: 1.0, M=1.0, n_samples=5)


class TestMetropolisHastings:
    def setup_method(self):
        random.seed(0)

    def test_samples_recover_target_normal(self):
        log_post = lambda theta: -0.5 * (theta - 2.0) ** 2
        result = metropolis_hastings(log_post, init=0.0, n_samples=5000, step_size=1.0, burn_in=500)
        samples = result["samples"]
        mean_est = sum(samples) / len(samples)
        var_est = sum((x - mean_est) ** 2 for x in samples) / len(samples)
        assert mean_est == pytest.approx(2.0, abs=0.15)
        assert var_est == pytest.approx(1.0, abs=0.2)

    def test_output_lengths(self):
        log_post = lambda theta: -0.5 * theta ** 2
        result = metropolis_hastings(log_post, init=0.0, n_samples=1000, burn_in=200)
        assert len(result["samples"]) == 1000
        assert result["n_samples"] == 1000
        assert result["burn_in"] == 200

    def test_acceptance_rate_in_valid_range(self):
        log_post = lambda theta: -0.5 * theta ** 2
        result = metropolis_hastings(log_post, init=0.0, n_samples=1000, burn_in=100)
        assert 0.0 <= result["acceptance_rate"] <= 1.0

    def test_invalid_start_point_neg_inf_raises(self):
        """Regression test: starting at a zero-density point should raise a
        clear error rather than silently running (even though the chain
        might eventually self-correct)."""
        def bad_log_post(theta):
            return -0.5 * (theta - 2.0) ** 2 if theta > 100 else float("-inf")

        with pytest.raises(ValueError, match="-inf"):
            metropolis_hastings(bad_log_post, init=0.0, n_samples=100)

    def test_invalid_start_point_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            metropolis_hastings(lambda theta: float("nan"), init=0.0, n_samples=100)

    def test_nonpositive_n_samples_raises(self):
        with pytest.raises(ValueError):
            metropolis_hastings(lambda t: -0.5 * t ** 2, init=0.0, n_samples=0)

    def test_nonpositive_step_size_raises(self):
        with pytest.raises(ValueError):
            metropolis_hastings(lambda t: -0.5 * t ** 2, init=0.0, n_samples=10, step_size=0.0)

    def test_negative_burn_in_raises(self):
        with pytest.raises(ValueError):
            metropolis_hastings(lambda t: -0.5 * t ** 2, init=0.0, n_samples=10, burn_in=-1)

    def test_proposal_evaluation_errors_treated_as_rejection(self):
        """log_posterior_fn raising on a *proposed* (not initial) point should
        be treated as -inf (reject), not crash the sampler."""
        def flaky_log_post(theta):
            if theta > 1e6:
                raise ValueError("out of domain")
            return -0.5 * theta ** 2

        result = metropolis_hastings(flaky_log_post, init=0.0, n_samples=200, step_size=0.5, burn_in=50)
        assert len(result["samples"]) == 200


class TestEffectiveSampleSize:
    def test_iid_samples_have_high_ess(self):
        samples = list(np.random.RandomState(1).normal(0, 1, 2000))
        ess = effective_sample_size(samples)
        assert ess > 1500

    def test_random_walk_has_low_ess(self):
        rng = np.random.RandomState(2)
        rw = [0.0]
        for _ in range(1999):
            rw.append(rw[-1] + rng.normal(0, 0.1))
        ess = effective_sample_size(rw)
        assert ess < 500

    def test_constant_sequence_ess_is_one(self):
        assert effective_sample_size([5.0] * 100) == 1.0

    def test_short_sequence_returns_n(self):
        assert effective_sample_size([1.0, 2.0, 3.0]) == 3.0

    def test_alternating_sequence_does_not_crash(self):
        """Regression test: strong anti-correlation could previously push the
        ESS denominator to zero or negative, risking a ZeroDivisionError or
        a nonsensical negative ESS."""
        alt = [1.0, -1.0] * 1000
        ess = effective_sample_size(alt)
        assert 1.0 <= ess <= 2000.0

    def test_ess_never_exceeds_n(self):
        samples = list(np.random.RandomState(3).normal(0, 1, 500))
        assert effective_sample_size(samples) <= 500.0

    def test_ess_at_least_one(self):
        rng = np.random.RandomState(4)
        rw = [0.0]
        for _ in range(999):
            rw.append(rw[-1] + rng.normal(0, 1))
        assert effective_sample_size(rw) >= 1.0

    def test_large_chain_respects_max_lag_performance(self):
        """Regression test: without a lag cap, a slowly-decaying
        autocorrelation on a very long chain could force examination of
        O(n) lags each costing O(n) work -> O(n^2) total."""
        samples = list(np.random.RandomState(5).normal(0, 1, 50000))
        start = time.time()
        effective_sample_size(samples, max_lag=1000)
        elapsed = time.time() - start
        assert elapsed < 5.0
