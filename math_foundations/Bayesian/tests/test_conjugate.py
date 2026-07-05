"""
test_conjugate.py

Run with:  pytest test_conjugate.py -v
Requires: pytest, scipy (regression oracle only).
"""

import os
import sys
import math
import pytest
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conjugate import BetaBinomial, GammaPoisson, GaussianGaussian


class TestBetaBinomial:
    def test_update_matches_conjugate_formula(self):
        bb = BetaBinomial(1.0, 1.0)
        bb.update(10, 7)
        assert bb.alpha == pytest.approx(8.0)
        assert bb.beta == pytest.approx(4.0)

    def test_posterior_mean_matches_scipy(self):
        bb = BetaBinomial(1.0, 1.0)
        bb.update(10, 7)
        assert bb.posterior_mean() == pytest.approx(stats.beta(8, 4).mean())

    def test_posterior_std_matches_scipy(self):
        bb = BetaBinomial(2.0, 3.0)
        bb.update(20, 12)
        assert bb.posterior_std() == pytest.approx(stats.beta(14, 11).std())

    def test_map_estimate_unimodal(self):
        bb = BetaBinomial(1.0, 1.0)
        bb.update(10, 7)
        a, b = bb.alpha, bb.beta
        assert bb.map_estimate() == pytest.approx((a - 1) / (a + b - 2))

    def test_map_estimate_boundary_a_gt_1_b_leq_1(self):
        bb = BetaBinomial(2.0, 0.5)
        assert bb.map_estimate() == 1.0

    def test_map_estimate_boundary_b_gt_1_a_leq_1(self):
        bb = BetaBinomial(0.5, 2.0)
        assert bb.map_estimate() == 0.0

    def test_map_estimate_bimodal_raises(self):
        """Regression test: alpha<=1, beta<=1 has a bimodal/uniform density with
        no single finite mode; must raise instead of silently returning 0.5
        (which is the *antimode*, not a mode, for this parameter regime)."""
        with pytest.raises(ValueError, match="No unique mode"):
            BetaBinomial(0.5, 0.5).map_estimate()

    def test_map_estimate_uniform_case_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(1.0, 1.0).map_estimate()

    def test_credible_interval_matches_scipy(self):
        bb = BetaBinomial(1.0, 1.0)
        bb.update(10, 7)
        ci = bb.credible_interval(0.95)
        sp = stats.beta(8, 4)
        assert ci["lower"] == pytest.approx(sp.ppf(0.025), abs=1e-6)
        assert ci["upper"] == pytest.approx(sp.ppf(0.975), abs=1e-6)

    def test_nonpositive_prior_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(0, 1)
        with pytest.raises(ValueError):
            BetaBinomial(1, -1)

    def test_bool_prior_rejected(self):
        with pytest.raises(TypeError):
            BetaBinomial(True, 1)

    def test_negative_n_trials_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(1, 1).update(-1, 0)

    def test_k_exceeds_n_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(1, 1).update(5, 6)

    def test_non_int_trials_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(1, 1).update(5.5, 2)

    def test_credible_interval_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            BetaBinomial(1, 1).credible_interval(1.5)

    def test_sequential_updates_are_associative(self):
        """Updating in two batches should equal updating with the combined total."""
        bb1 = BetaBinomial(1.0, 1.0)
        bb1.update(10, 7)
        bb1.update(5, 2)

        bb2 = BetaBinomial(1.0, 1.0)
        bb2.update(15, 9)

        assert bb1.alpha == pytest.approx(bb2.alpha)
        assert bb1.beta == pytest.approx(bb2.beta)


class TestGammaPoisson:
    def test_update_matches_conjugate_formula(self):
        gp = GammaPoisson(2.0, 1.0)
        gp.update(5, 3.0)
        assert gp.alpha == pytest.approx(7.0)
        assert gp.beta == pytest.approx(4.0)

    def test_posterior_mean_matches_scipy(self):
        gp = GammaPoisson(2.0, 1.0)
        gp.update(5, 3.0)
        sp = stats.gamma(7.0, scale=1 / 4.0)
        assert gp.posterior_mean() == pytest.approx(sp.mean())

    def test_posterior_std_matches_scipy(self):
        gp = GammaPoisson(2.0, 1.0)
        gp.update(5, 3.0)
        sp = stats.gamma(7.0, scale=1 / 4.0)
        assert gp.posterior_std() == pytest.approx(sp.std())

    def test_map_estimate_alpha_geq_1(self):
        gp = GammaPoisson(2.0, 1.0)
        gp.update(5, 3.0)
        assert gp.map_estimate() == pytest.approx((7.0 - 1.0) / 4.0)

    def test_map_estimate_alpha_lt_1_is_zero(self):
        gp = GammaPoisson(0.5, 1.0)
        assert gp.map_estimate() == 0.0

    def test_credible_interval_matches_scipy(self):
        gp = GammaPoisson(2.0, 1.0)
        gp.update(5, 3.0)
        ci = gp.credible_interval(0.9)
        sp = stats.gamma(7.0, scale=1 / 4.0)
        assert ci["lower"] == pytest.approx(sp.ppf(0.05), abs=1e-6)
        assert ci["upper"] == pytest.approx(sp.ppf(0.95), abs=1e-6)

    def test_nonpositive_prior_raises(self):
        with pytest.raises(ValueError):
            GammaPoisson(0, 1)

    def test_negative_events_raises(self):
        with pytest.raises(ValueError):
            GammaPoisson(1, 1).update(-1)

    def test_nonpositive_exposure_raises(self):
        with pytest.raises(ValueError):
            GammaPoisson(1, 1).update(1, 0)

    def test_non_numeric_events_raises(self):
        with pytest.raises(TypeError):
            GammaPoisson(1, 1).update("x")

    def test_default_exposure_is_one(self):
        gp = GammaPoisson(1.0, 1.0)
        gp.update(3)
        assert gp.beta == pytest.approx(2.0)


class TestGaussianGaussian:
    def test_update_matches_conjugate_formula(self):
        gg = GaussianGaussian(prior_mu=0.0, prior_sigma=10.0, obs_sigma=2.0)
        data = [5.0, 5.5, 4.5, 5.2]
        gg.update(data)
        n = len(data)
        x_bar = sum(data) / n
        prior_prec = 1 / 100.0
        obs_prec = n / 4.0
        post_prec = prior_prec + obs_prec
        expected_mu = (prior_prec * 0.0 + obs_prec * x_bar) / post_prec
        expected_var = 1 / post_prec
        assert gg.posterior_mean() == pytest.approx(expected_mu)
        assert gg.posterior_std() == pytest.approx(math.sqrt(expected_var))

    def test_credible_interval_matches_manual_zscore(self):
        """Regression test: after consolidating into the ConjugateModel base
        class, credible_interval now goes through NormalDistribution.ppf
        instead of a custom z-score calculation -- verify they agree."""
        gg = GaussianGaussian(prior_mu=0.0, prior_sigma=10.0, obs_sigma=2.0)
        gg.update([5.0, 5.5, 4.5, 5.2])
        ci = gg.credible_interval(0.95)
        mu, std = gg.posterior_mean(), gg.posterior_std()
        z = stats.norm.ppf(0.975)
        assert ci["lower"] == pytest.approx(mu - z * std, abs=1e-6)
        assert ci["upper"] == pytest.approx(mu + z * std, abs=1e-6)

    def test_map_estimate_equals_mean_for_normal(self):
        gg = GaussianGaussian(0.0, 10.0, 2.0)
        gg.update([1.0, 2.0, 3.0])
        assert gg.map_estimate() == pytest.approx(gg.posterior_mean())

    def test_more_data_narrows_posterior(self):
        gg = GaussianGaussian(0.0, 10.0, 2.0)
        gg.update([5.0])
        std_after_1 = gg.posterior_std()
        gg.update([5.0] * 100)
        std_after_101 = gg.posterior_std()
        assert std_after_101 < std_after_1

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            GaussianGaussian(0, 1, 1).update([])

    def test_non_list_tuple_data_raises(self):
        with pytest.raises(TypeError):
            GaussianGaussian(0, 1, 1).update("not a list")

    def test_tuple_data_accepted(self):
        gg = GaussianGaussian(0, 1, 1)
        gg.update((1, 2, 3))  # should not raise

    def test_nan_in_data_raises(self):
        with pytest.raises(ValueError):
            GaussianGaussian(0, 1, 1).update([1.0, float("nan")])

    def test_inf_in_data_raises(self):
        with pytest.raises(ValueError):
            GaussianGaussian(0, 1, 1).update([1.0, float("inf")])

    def test_nan_prior_mu_raises(self):
        with pytest.raises(ValueError):
            GaussianGaussian(float("nan"), 1, 1)

    def test_nonpositive_sigma_raises(self):
        with pytest.raises(ValueError):
            GaussianGaussian(0, -1, 1)
        with pytest.raises(ValueError):
            GaussianGaussian(0, 1, 0)


class TestConjugateModelSharedBehavior:
    """Tests targeting the shared ConjugateModel base-class logic itself."""

    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: BetaBinomial(2.0, 2.0),
            lambda: GammaPoisson(2.0, 2.0),
            lambda: GaussianGaussian(0.0, 1.0, 1.0),
        ],
    )
    def test_credible_interval_invalid_prob_raises(self, model_factory):
        model = model_factory()
        with pytest.raises(ValueError):
            model.credible_interval(0.0)
        with pytest.raises(ValueError):
            model.credible_interval(1.0)

    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: BetaBinomial(2.0, 2.0),
            lambda: GammaPoisson(2.0, 2.0),
            lambda: GaussianGaussian(0.0, 1.0, 1.0),
        ],
    )
    def test_credible_interval_contains_mean(self, model_factory):
        model = model_factory()
        ci = model.credible_interval(0.95)
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda: BetaBinomial(2.0, 2.0),
            lambda: GammaPoisson(2.0, 2.0),
            lambda: GaussianGaussian(0.0, 1.0, 1.0),
        ],
    )
    def test_posterior_std_matches_sqrt_variance(self, model_factory):
        model = model_factory()
        assert model.posterior_std() == pytest.approx(math.sqrt(model.posterior.variance()))
