"""
test_information_theory.py

Run with:  pytest test_information_theory.py -v
Requires: pytest, numpy, scipy (regression oracle only).
"""

import os
import sys
import math
import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as sp_entropy


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from information_theory import (
    binary_cross_entropy,
    binary_entropy,
    conditional_entropy,
    cross_entropy,
    entropy,
    gini_gain,
    gini_impurity,
    information_gain,
    joint_entropy,
    js_divergence,
    kl_divergence,
    marginal_from_joint,
    mutual_information,
    normalized_mutual_information,
    perplexity,
    renyi_entropy,
)


class TestEntropy:
    @pytest.mark.parametrize("base", [2.0, math.e, 10.0])
    def test_matches_scipy(self, base):
        p = [0.5, 0.25, 0.25]
        assert entropy(p, base=base) == pytest.approx(sp_entropy(p, base=base), abs=1e-9)

    def test_uniform_distribution_max_entropy(self):
        n = 8
        p = [1.0 / n] * n
        assert entropy(p, base=2) == pytest.approx(math.log2(n))

    def test_deterministic_distribution_zero_entropy(self):
        assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            entropy([])

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError):
            entropy([-0.5, 1.5])

    def test_does_not_sum_to_one_raises(self):
        with pytest.raises(ValueError):
            entropy([0.3, 0.3])

    def test_nan_probability_raises(self):
        """Regression test: NaN comparisons are always False, so a naive
        `x < 0.0` non-negativity check silently lets NaN through. Must be
        explicitly rejected."""
        with pytest.raises(ValueError, match="NaN"):
            entropy([0.5, float("nan")])

    def test_inf_probability_raises(self):
        with pytest.raises(ValueError):
            entropy([float("inf"), 0.0])

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            entropy([True, False])

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            entropy([0.5, "a"])

    def test_custom_tolerance_allows_small_error(self):
        entropy([0.5, 0.5000001])  # should not raise with default tol=1e-6

    def test_custom_tolerance_rejects_larger_error(self):
        with pytest.raises(ValueError):
            entropy([0.5, 0.51], tol=1e-6)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            entropy([0.5, 0.5], base=1.0)
        with pytest.raises(ValueError):
            entropy([0.5, 0.5], base=0.0)
        with pytest.raises(ValueError):
            entropy([0.5, 0.5], base=-2.0)

    def test_bool_base_rejected(self):
        with pytest.raises(TypeError):
            entropy([0.5, 0.5], base=True)


class TestBinaryEntropy:
    def test_max_at_half(self):
        assert binary_entropy(0.5) == pytest.approx(1.0)

    def test_zero_at_boundaries(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_matches_general_entropy(self):
        p = 0.3
        assert binary_entropy(p) == pytest.approx(entropy([p, 1 - p]))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            binary_entropy(1.5)
        with pytest.raises(ValueError):
            binary_entropy(-0.1)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            binary_entropy(float("nan"))


class TestJointDistributionFunctions:
    JOINT = [[0.1, 0.2], [0.3, 0.4]]

    def test_joint_entropy_matches_manual(self):
        expected = -sum(v * math.log2(v) for row in self.JOINT for v in row if v > 0)
        assert joint_entropy(self.JOINT) == pytest.approx(expected)

    def test_marginal_axis1_is_row_sums(self):
        mx = marginal_from_joint(self.JOINT, axis=1)
        np.testing.assert_allclose(mx, np.array(self.JOINT).sum(axis=1))

    def test_marginal_axis0_is_col_sums(self):
        my = marginal_from_joint(self.JOINT, axis=0)
        np.testing.assert_allclose(my, np.array(self.JOINT).sum(axis=0))

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            marginal_from_joint(self.JOINT, axis=2)

    def test_mutual_information_matches_manual(self):
        mx = marginal_from_joint(self.JOINT, axis=1)
        my = marginal_from_joint(self.JOINT, axis=0)
        H_x = entropy(mx)
        H_y = entropy(my)
        H_xy = joint_entropy(self.JOINT)
        assert mutual_information(self.JOINT) == pytest.approx(H_x + H_y - H_xy)

    def test_mutual_information_independent_vars_is_zero(self):
        # p(x,y) = p(x)*p(y): independence -> MI = 0
        independent = [[0.25, 0.25], [0.25, 0.25]]
        assert mutual_information(independent) == pytest.approx(0.0, abs=1e-9)

    def test_conditional_entropy_given_x(self):
        H_x = entropy(marginal_from_joint(self.JOINT, axis=1))
        H_xy = joint_entropy(self.JOINT)
        assert conditional_entropy(self.JOINT, given="X") == pytest.approx(H_xy - H_x)

    def test_conditional_entropy_given_y(self):
        H_y = entropy(marginal_from_joint(self.JOINT, axis=0))
        H_xy = joint_entropy(self.JOINT)
        assert conditional_entropy(self.JOINT, given="Y") == pytest.approx(H_xy - H_y)

    def test_conditional_entropy_invalid_given_raises(self):
        with pytest.raises(ValueError):
            conditional_entropy(self.JOINT, given="Z")

    def test_normalized_mutual_information_range(self):
        nmi = normalized_mutual_information(self.JOINT)
        assert 0.0 <= nmi <= 1.0

    def test_normalized_mutual_information_perfect_dependence(self):
        # Perfectly dependent: NMI should be 1
        perfect = [[0.5, 0.0], [0.0, 0.5]]
        assert normalized_mutual_information(perfect) == pytest.approx(1.0, abs=1e-6)

    def test_ragged_joint_raises(self):
        with pytest.raises(ValueError):
            joint_entropy([[0.5, 0.5], [0.5]])

    def test_negative_joint_value_raises(self):
        with pytest.raises(ValueError):
            joint_entropy([[-0.1, 0.6], [0.3, 0.2]])

    def test_nan_joint_value_raises(self):
        with pytest.raises(ValueError):
            marginal_from_joint([[0.5, float("nan")], [0.3, 0.2]])

    def test_empty_joint_raises(self):
        with pytest.raises(ValueError):
            joint_entropy([])


class TestCrossEntropyAndDivergences:
    P = [0.1, 0.4, 0.5]
    Q = [0.2, 0.3, 0.5]

    def test_cross_entropy_matches_manual(self):
        expected = -sum(pi * math.log2(qi) for pi, qi in zip(self.P, self.Q))
        assert cross_entropy(self.P, self.Q) == pytest.approx(expected)

    def test_cross_entropy_infinite_when_q_zero_p_positive(self):
        assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_cross_entropy_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cross_entropy([0.5, 0.5], [1.0, 0.0, 0.0])

    @pytest.mark.parametrize("base", [2.0, math.e])
    def test_kl_divergence_matches_scipy(self, base):
        assert kl_divergence(self.P, self.Q, base=base) == pytest.approx(
            sp_entropy(self.P, self.Q, base=base), abs=1e-9
        )

    def test_kl_divergence_self_is_zero(self):
        assert kl_divergence(self.P, self.P) == pytest.approx(0.0, abs=1e-9)

    def test_kl_divergence_infinite_when_q_zero_p_positive(self):
        assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_kl_divergence_nonnegative(self):
        assert kl_divergence(self.P, self.Q) >= 0.0

    def test_js_divergence_matches_scipy(self):
        jsd = js_divergence(self.P, self.Q, base=2)
        jsd_scipy = jensenshannon(self.P, self.Q, base=2) ** 2  # scipy returns sqrt(JSD)
        assert jsd == pytest.approx(jsd_scipy, abs=1e-9)

    def test_js_divergence_symmetric(self):
        assert js_divergence(self.P, self.Q) == pytest.approx(js_divergence(self.Q, self.P))

    def test_js_divergence_self_is_zero(self):
        assert js_divergence(self.P, self.P) == pytest.approx(0.0, abs=1e-9)

    def test_js_divergence_finite_even_when_disjoint_support(self):
        # Unlike KL, JS divergence stays finite for disjoint support.
        result = js_divergence([1.0, 0.0], [0.0, 1.0])
        assert math.isfinite(result)


class TestBinaryCrossEntropy:
    def test_matches_manual(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [0.9, 0.1, 0.8, 0.6, 0.3]
        expected = -sum(
            yt * math.log(yp) + (1 - yt) * math.log(1 - yp) for yt, yp in zip(y_true, y_pred)
        ) / len(y_true)
        assert binary_cross_entropy(y_true, y_pred) == pytest.approx(expected)

    def test_perfect_predictions_near_zero_loss(self):
        loss = binary_cross_entropy([1, 0, 1], [1.0, 0.0, 1.0])
        assert loss < 1e-10

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            binary_cross_entropy([1, 0], [0.5, 0.5, 0.5])

    def test_empty_returns_zero(self):
        assert binary_cross_entropy([], []) == 0.0

    def test_non_binary_y_true_raises(self):
        with pytest.raises(ValueError):
            binary_cross_entropy([0, 2], [0.5, 0.5])

    def test_out_of_range_proba_raises(self):
        """Regression test: probabilities outside [0,1] used to be silently
        clipped (masking real bugs in the caller's model) instead of raising."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            binary_cross_entropy([1, 0], [1.5, 0.2])
        with pytest.raises(ValueError):
            binary_cross_entropy([1, 0], [-0.1, 0.2])

    def test_nan_proba_raises(self):
        with pytest.raises(ValueError):
            binary_cross_entropy([1, 0], [float("nan"), 0.2])


class TestInformationGainAndGiniGain:
    PARENT = [10, 10]
    SUBSETS = [[8, 2], [2, 8]]

    def test_information_gain_matches_manual(self):
        h_parent = entropy([0.5, 0.5])
        h_c1 = entropy([0.8, 0.2])
        h_c2 = entropy([0.2, 0.8])
        expected = h_parent - 0.5 * h_c1 - 0.5 * h_c2
        assert information_gain(self.PARENT, self.SUBSETS) == pytest.approx(expected)

    def test_gini_gain_matches_manual(self):
        g_parent = gini_impurity([0.5, 0.5])
        g_c1 = gini_impurity([0.8, 0.2])
        g_c2 = gini_impurity([0.2, 0.8])
        expected = g_parent - 0.5 * g_c1 - 0.5 * g_c2
        assert gini_gain(self.PARENT, self.SUBSETS) == pytest.approx(expected)

    def test_perfect_split_gives_max_information_gain(self):
        parent = [10, 10]
        subsets = [[10, 0], [0, 10]]
        ig = information_gain(parent, subsets)
        assert ig == pytest.approx(entropy([0.5, 0.5]))

    def test_useless_split_gives_zero_gain(self):
        parent = [10, 10]
        subsets = [[5, 5], [5, 5]]
        assert information_gain(parent, subsets) == pytest.approx(0.0, abs=1e-9)

    def test_category_count_mismatch_raises(self):
        """Regression test: a subset with a different number of categories
        than the parent is a silent alignment bug; must raise clearly."""
        with pytest.raises(ValueError, match="categories"):
            information_gain([10, 10], [[5, 5, 0], [5, 5]])

    def test_negative_subset_count_raises(self):
        with pytest.raises(ValueError):
            information_gain([10, 10], [[-5, 5], [15, 5]])

    def test_nan_parent_count_raises(self):
        """Regression test: NaN counts used to silently bypass the `c < 0`
        non-negativity check and propagate into entropy calculations."""
        with pytest.raises(ValueError, match="NaN"):
            information_gain([10, float("nan")], [[5, 0], [5, 10]])

    def test_empty_parent_raises(self):
        with pytest.raises(ValueError):
            information_gain([], [[1, 2]])

    def test_zero_parent_total_raises(self):
        with pytest.raises(ValueError):
            information_gain([0, 0], [[1, 2]])

    def test_empty_subsets_raises(self):
        with pytest.raises(ValueError):
            information_gain([10, 10], [])

    def test_zero_count_subset_skipped_not_erroring(self):
        # A subset that received zero samples should just be skipped, not error.
        result = information_gain([10, 10], [[10, 10], [0, 0]])
        assert result == pytest.approx(0.0, abs=1e-9)


class TestGiniImpurity:
    def test_pure_node_is_zero(self):
        assert gini_impurity([1.0, 0.0]) == pytest.approx(0.0)

    def test_maximally_impure_binary(self):
        assert gini_impurity([0.5, 0.5]) == pytest.approx(0.5)

    def test_matches_formula(self):
        p = [0.2, 0.3, 0.5]
        assert gini_impurity(p) == pytest.approx(1 - sum(pi ** 2 for pi in p))


class TestPerplexity:
    def test_deterministic_is_one(self):
        assert perplexity([1.0]) == pytest.approx(1.0)

    def test_uniform_binary_is_two(self):
        assert perplexity([0.5, 0.5]) == pytest.approx(2.0)

    def test_matches_base_power_entropy(self):
        p = [0.2, 0.3, 0.5]
        assert perplexity(p, base=2) == pytest.approx(2 ** entropy(p, base=2))


class TestRenyiEntropy:
    P = [0.9, 0.05, 0.05]

    def test_alpha_one_matches_shannon(self):
        assert renyi_entropy(self.P, alpha=1.0) == pytest.approx(entropy(self.P), abs=1e-9)

    def test_alpha_two_matches_manual_collision_entropy(self):
        expected = math.log2(sum(pi ** 2 for pi in self.P)) / (1 - 2)
        assert renyi_entropy(self.P, alpha=2.0) == pytest.approx(expected)

    def test_min_entropy_limit_at_infinity(self):
        expected = -math.log2(max(self.P))
        assert renyi_entropy(self.P, alpha=float("inf")) == pytest.approx(expected)

    def test_large_alpha_approaches_min_entropy_without_underflow(self):
        """Regression test: naive p_i**alpha for large alpha underflows to
        exactly 0.0 for all i, which used to make the function incorrectly
        return 0.0 instead of approaching -log(max(p))."""
        expected = -math.log2(max(self.P))
        result = renyi_entropy(self.P, alpha=1000.0)
        assert result == pytest.approx(expected, abs=0.01)
        assert result != 0.0

    def test_uniform_distribution_alpha_independent(self):
        # For a uniform distribution, Renyi entropy equals Shannon entropy
        # for every alpha (all orders agree when there's no "peakiness").
        p = [0.25, 0.25, 0.25, 0.25]
        h = entropy(p)
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            assert renyi_entropy(p, alpha=alpha) == pytest.approx(h, abs=1e-6)

    def test_nonpositive_alpha_raises(self):
        with pytest.raises(ValueError):
            renyi_entropy(self.P, alpha=0.0)
        with pytest.raises(ValueError):
            renyi_entropy(self.P, alpha=-1.0)

    def test_nan_alpha_raises(self):
        with pytest.raises(ValueError):
            renyi_entropy(self.P, alpha=float("nan"))
