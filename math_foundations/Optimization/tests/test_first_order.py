"""
test_first_order.py

Run with:  pytest test_first_order.py -v
Requires: pytest.
"""

import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from first_order import GradientDescent, SGD, Momentum, RMSProp, Adam, AdaGrad
from Vectors.vector import Vector
from Matrix.matrix import Matrix


def _converge_on_square(optimizer, start=10.0, n_iter=500):
    """Run `optimizer` on f(x) = x^2 (grad = 2x), minimum at x=0."""
    params = [start]
    for _ in range(n_iter):
        grad = [2 * params[0]]
        optimizer.step(params, grad)
    return params[0]


class TestGradientDescent:
    def test_converges_on_quadratic(self):
        assert abs(_converge_on_square(GradientDescent(lr=0.1))) < 0.01

    def test_mismatched_lengths_raises(self):
        opt = GradientDescent(lr=0.1)
        with pytest.raises(ValueError):
            opt.step([1.0, 2.0], [1.0])

    def test_works_with_vector_params(self):
        opt = GradientDescent(lr=0.1)
        params = [Vector([4.0, 4.0])]
        for _ in range(200):
            grad = [Vector([2 * c for c in params[0].components])]
            opt.step(params, grad)
        assert all(abs(c) < 0.01 for c in params[0].components)

    def test_iterations_increment(self):
        opt = GradientDescent(lr=0.1)
        params = [1.0]
        opt.step(params, [1.0])
        opt.step(params, [1.0])
        assert opt.iterations == 2


class TestSGD:
    def test_converges_plain(self):
        assert abs(_converge_on_square(SGD(lr=0.1))) < 0.01

    def test_converges_with_momentum(self):
        assert abs(_converge_on_square(SGD(lr=0.05, momentum=0.9))) < 0.01

    def test_converges_with_nesterov(self):
        assert abs(_converge_on_square(SGD(lr=0.05, momentum=0.9, nesterov=True))) < 0.01

    def test_nesterov_requires_momentum(self):
        with pytest.raises(ValueError):
            SGD(momentum=0.0, nesterov=True)

    def test_momentum_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SGD(momentum=1.0)
        with pytest.raises(ValueError):
            SGD(momentum=-0.1)

    def test_negative_decay_raises(self):
        with pytest.raises(ValueError):
            SGD(decay=-0.1)

    def test_decay_reduces_effective_lr_over_time(self):
        opt = SGD(lr=1.0, decay=1.0)
        params1 = [10.0]
        opt.step(params1, [1.0])  # iteration 0: lr_eff = 1.0
        step1_size = 10.0 - params1[0]

        opt2 = SGD(lr=1.0, decay=1.0)
        params2 = [10.0]
        for _ in range(5):
            opt2.step(params2, [1.0])
        params2_before = params2[0]
        opt2.step(params2, [1.0])
        step_later_size = params2_before - params2[0]

        assert step_later_size < step1_size

    def test_reset_clears_velocity(self):
        opt = SGD(lr=0.1, momentum=0.9)
        opt.step([1.0], [1.0])
        opt.reset()
        assert opt._velocity is None

    def test_get_config(self):
        opt = SGD(lr=0.1, momentum=0.5, decay=0.01, nesterov=True)
        config = opt.get_config()
        assert config["momentum"] == 0.5
        assert config["nesterov"] is True


class TestMomentum:
    def test_converges(self):
        assert abs(_converge_on_square(Momentum(lr=0.1, beta=0.9))) < 0.01

    def test_beta_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Momentum(beta=1.0)

    def test_differs_from_sgd_momentum_for_same_beta(self):
        """Regression/clarity test: SGD(momentum=b) and Momentum(beta=b) are
        genuinely different update rules (heavy-ball velocity of raw lr*g
        vs. an EMA of g scaled by lr), so they should NOT produce identical
        trajectories despite sharing a hyperparameter name/value."""
        sgd = SGD(lr=0.1, momentum=0.9)
        mom = Momentum(lr=0.1, beta=0.9)
        p_sgd, p_mom = [10.0], [10.0]
        for _ in range(5):
            sgd.step(p_sgd, [2 * p_sgd[0]])
            mom.step(p_mom, [2 * p_mom[0]])
        assert p_sgd[0] != pytest.approx(p_mom[0])


class TestAdaGrad:
    def test_converges(self):
        assert abs(_converge_on_square(AdaGrad(lr=0.5))) < 0.2

    def test_nonpositive_epsilon_raises(self):
        with pytest.raises(ValueError):
            AdaGrad(epsilon=0.0)

    def test_learning_rate_effectively_shrinks_over_time(self):
        opt = AdaGrad(lr=1.0)
        params = [10.0]
        opt.step(params, [5.0])
        step1 = 10.0 - params[0]

        for _ in range(19):
            opt.step(params, [5.0])
        before_last = params[0]
        opt.step(params, [5.0])
        step_later = before_last - params[0]

        # AdaGrad's cache only accumulates, so a later single-step delta
        # (even with identical gradient magnitude) should be smaller than
        # the first single-step delta.
        assert abs(step_later) < abs(step1)


class TestRMSProp:
    def test_converges(self):
        assert abs(_converge_on_square(RMSProp(lr=0.1))) < 0.1

    def test_beta_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RMSProp(beta=1.0)

    def test_nonpositive_epsilon_raises(self):
        with pytest.raises(ValueError):
            RMSProp(epsilon=-1e-8)

    def test_matrix_gradient_elementwise_square_not_matmul(self):
        """Regression test: RMSProp/Adam used to apply their elementwise
        lambdas directly to whole Matrix objects for non-Vector params,
        so `x * x` invoked Matrix.__mul__ (matrix multiplication) instead
        of squaring each entry, and would crash entirely on `_safe_sqrt`.
        A uniform gradient should produce a uniform (all-entries-equal)
        update."""
        opt = RMSProp(lr=0.1)
        params = [Matrix([[4.0, 4.0], [4.0, 4.0]])]
        grad = [Matrix([[2.0, 2.0], [2.0, 2.0]])]
        opt.step(params, grad)  # must not raise
        vals = [params[0].rows[i].components[j] for i in range(2) for j in range(2)]
        assert all(v == pytest.approx(vals[0]) for v in vals)


class TestAdam:
    def test_converges(self):
        assert abs(_converge_on_square(Adam(lr=0.1))) < 0.01

    def test_beta_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Adam(beta1=1.0)
        with pytest.raises(ValueError):
            Adam(beta2=1.0)

    def test_matrix_gradient_works(self):
        """Same Matrix-elementwise regression as RMSProp, for Adam."""
        opt = Adam(lr=0.1)
        params = [Matrix([[4.0, 4.0], [4.0, 4.0]])]
        grad = [Matrix([[2.0, 2.0], [2.0, 2.0]])]
        opt.step(params, grad)  # must not raise
        vals = [params[0].rows[i].components[j] for i in range(2) for j in range(2)]
        assert all(v == pytest.approx(vals[0]) for v in vals)

    def test_first_step_magnitude_approx_lr(self):
        """Well-known Adam property (and a check that bias correction is
        implemented correctly): the very first step's size should be
        approximately `lr`, regardless of the gradient's magnitude."""
        opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
        params = [100.0]
        opt.step(params, [50.0])
        step_size = 100.0 - params[0]
        assert step_size == pytest.approx(0.1, abs=0.02)

    def test_matches_textbook_formula_manually(self):
        """Regression test: verifies the exact textbook/PyTorch-compatible
        bias-correction formula (m_hat, v_hat computed explicitly, epsilon
        applied to sqrt(v_hat)) rather than the paper's alternate
        combined-alpha_t reformulation, which uses epsilon at a different
        effective scale and would diverge from this at t=1 when epsilon
        is non-negligible relative to the gradient."""
        lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-2  # large eps to make the two formulas diverge visibly
        g = 0.5

        m = b1 * 0.0 + (1 - b1) * g
        v = b2 * 0.0 + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** 1)
        v_hat = v / (1 - b2 ** 1)
        expected_update = lr * m_hat / (math.sqrt(v_hat) + eps)

        opt = Adam(lr=lr, beta1=b1, beta2=b2, epsilon=eps)
        params = [10.0]
        opt.step(params, [g])
        actual_update = 10.0 - params[0]

        assert actual_update == pytest.approx(expected_update, abs=1e-9)

    def test_get_config(self):
        opt = Adam(lr=0.01, beta1=0.8, beta2=0.99, epsilon=1e-7)
        config = opt.get_config()
        assert config["beta1"] == 0.8
        assert config["beta2"] == 0.99

    def test_reset_clears_moments(self):
        opt = Adam()
        opt.step([1.0], [1.0])
        opt.reset()
        assert opt._m is None
        assert opt._v is None
