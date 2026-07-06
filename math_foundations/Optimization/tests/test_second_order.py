"""
test_second_order.py

Run with:  pytest test_second_order.py -v
Requires: pytest, scipy (regression oracle only).
"""

import os
import sys
import math

import pytest
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from second_order import (
    BFGS,
    NewtonMethod,
    _identity,
    _mat_add,
    _mat_sub,
    _mat_vec,
    _outer,
    _solve_linear,
    _vec_dot,
)
from utils import numerical_gradient


class TestSolveLinear:
    def test_matches_known_solution(self):
        A = [[2, 1], [1, 3]]
        b = [3, 5]
        x = _solve_linear(A, b)
        resid = [sum(A[i][j] * x[j] for j in range(2)) - b[i] for i in range(2)]
        assert all(abs(r) < 1e-9 for r in resid)

    def test_identity_system(self):
        A = [[1, 0], [0, 1]]
        b = [7, -3]
        assert _solve_linear(A, b) == pytest.approx([7, -3])

    def test_singular_matrix_raises(self):
        with pytest.raises(ValueError, match="Singular"):
            _solve_linear([[1, 2], [2, 4]], [1, 2])

    def test_empty_system_returns_empty(self):
        assert _solve_linear([], []) == []

    def test_nan_in_matrix_raises(self):
        with pytest.raises(ValueError):
            _solve_linear([[1, 2], [float("nan"), 4]], [1, 2])

    def test_inf_in_rhs_raises(self):
        with pytest.raises(ValueError):
            _solve_linear([[1, 0], [0, 1]], [1, float("inf")])

    def test_ragged_matrix_raises(self):
        with pytest.raises(ValueError):
            _solve_linear([[1, 2], [3]], [1, 2])

    def test_scale_aware_tolerance_large_magnitude(self):
        """A well-conditioned but large-magnitude system should still solve
        correctly (tests the scale-aware pivot tolerance)."""
        A = [[1e8, 0], [0, 1e8]]
        b = [2e8, 3e8]
        x = _solve_linear(A, b)
        assert x == pytest.approx([2.0, 3.0])


class TestMatrixHelpers:
    def test_identity(self):
        assert _identity(2) == [[1.0, 0.0], [0.0, 1.0]]

    def test_outer(self):
        assert _outer([1, 2], [3, 4]) == [[3, 4], [6, 8]]

    def test_mat_add_sub_roundtrip(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        assert _mat_sub(_mat_add(A, B), B) == pytest.approx(A)


class TestNewtonMethod:
    def test_converges_on_quadratic_bowl(self):
        f = lambda x: (x[0] - 3) ** 2 + 2 * (x[1] + 1) ** 2
        nm = NewtonMethod(f, lr=1.0)
        x = [0.0, 0.0]
        for _ in range(5):
            g = numerical_gradient(f, x)
            x = nm.step(x, g)
        assert x[0] == pytest.approx(3.0, abs=1e-2)
        assert x[1] == pytest.approx(-1.0, abs=1e-2)

    def test_matches_scipy_newton_cg_result(self):
        """Also exercises the numerical_gradient/numerical_hessian NumPy-array
        safety fix: scipy.optimize passes NumPy arrays (not plain lists) to
        jac/hess, and x[:] on a NumPy array is a view, not a copy -- without
        defensively copying to a list first, in-place perturbation would
        corrupt scipy's array in place and produce wrong results."""
        from utils import numerical_hessian

        f = lambda x: (x[0] - 3) ** 2 + 2 * (x[1] + 1) ** 2
        nm = NewtonMethod(f, lr=1.0)
        x = [0.0, 0.0]
        for _ in range(5):
            x = nm.step(x, numerical_gradient(f, x))

        sp = minimize(
            f, [0.0, 0.0], method="Newton-CG",
            jac=lambda x: numerical_gradient(f, x),
            hess=lambda x: numerical_hessian(f, x),
        )
        assert x[0] == pytest.approx(sp.x[0], abs=1e-2)
        assert x[1] == pytest.approx(sp.x[1], abs=1e-2)
        assert x[0] == pytest.approx(3.0, abs=1e-2)
        assert x[1] == pytest.approx(-1.0, abs=1e-2)

    def test_step_with_hessian(self):
        f = lambda x: x[0] ** 2
        nm = NewtonMethod(f)
        # analytic Hessian of x^2 is [[2]]
        new_params = nm.step_with_hessian([5.0], [10.0], [[2.0]])
        assert new_params[0] == pytest.approx(0.0)

    def test_falls_back_to_gradient_descent_on_singular_hessian(self, caplog):
        import logging

        f = lambda x: x[0]  # linear -> Hessian is exactly 0 (singular)
        nm = NewtonMethod(f, lr=0.1)
        with caplog.at_level(logging.WARNING):
            new_params = nm.step_with_hessian([1.0], [1.0], [[0.0]])
        assert new_params[0] == pytest.approx(1.0 - 0.1 * 1.0)
        assert any("singular" in r.message.lower() for r in caplog.records)

    def test_falls_back_on_non_descent_direction(self, caplog):
        """Regression-adjacent test: negative curvature can make the raw
        Newton direction an ascent direction; must fall back to gradient
        descent with a warning rather than silently taking a bad step."""
        import logging

        f = lambda x: x[0] ** 2  # irrelevant; using step_with_hessian directly
        nm = NewtonMethod(f, lr=0.1)
        # Negative Hessian: H^-1 @ grad would point the "wrong way".
        with caplog.at_level(logging.WARNING):
            new_params = nm.step_with_hessian([1.0], [1.0], [[-1.0]])
        # Should have fallen back to direction=grads=[1.0]: new = 1.0 - 0.1*1.0
        assert new_params[0] == pytest.approx(1.0 - 0.1 * 1.0)
        assert any("descent" in r.message.lower() for r in caplog.records)

    def test_mismatched_lengths_raises(self):
        nm = NewtonMethod(lambda x: x[0])
        with pytest.raises(ValueError):
            nm.step([1.0, 2.0], [1.0])

    def test_hessian_dimension_mismatch_raises(self):
        nm = NewtonMethod(lambda x: x[0])
        with pytest.raises(ValueError):
            nm.step_with_hessian([1.0], [1.0], [[1, 2], [3, 4]])

    def test_nonpositive_hessian_h_raises(self):
        with pytest.raises(ValueError):
            NewtonMethod(lambda x: x[0], hessian_h=0.0)


class TestBFGS:
    def test_converges_on_quadratic(self):
        f = lambda x: sum(xi ** 2 for xi in x)
        bfgs = BFGS(lr=1.0)
        x = [5.0, -3.0]
        for _ in range(50):
            x = bfgs.step(x, numerical_gradient(f, x))
        assert x[0] == pytest.approx(0.0, abs=1e-3)
        assert x[1] == pytest.approx(0.0, abs=1e-3)

    def test_matches_scipy_on_rosenbrock(self):
        rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        rosen_grad = lambda x: numerical_gradient(rosen, x, h=1e-6)

        bfgs = BFGS(lr=1.0)
        x0 = [-1.2, 1.0]
        x = x0[:]
        for _ in range(2000):
            g = rosen_grad(x)
            if sum(gi ** 2 for gi in g) ** 0.5 < 1e-6:
                break
            x = bfgs.step(x, g)

        sp = minimize(rosen, x0, method="BFGS")
        assert x[0] == pytest.approx(sp.x[0], abs=0.05)
        assert x[1] == pytest.approx(sp.x[1], abs=0.05)

    def test_curvature_condition_preserves_positive_definiteness(self):
        """Regression test for the core bug: the curvature check used to be
        `abs(s.y) > eps`, which allows updates even when s.y is strongly
        NEGATIVE. That corrupts positive-definiteness of H_inv (which can
        turn the BFGS search direction into an ascent direction). The
        fixed condition `s.y > eps` must skip the update in that case,
        leaving H_inv untouched (still the identity, still positive
        definite)."""
        bfgs = BFGS(lr=1.0, eps=1e-10)
        n = 2
        bfgs._H_inv = _identity(n)
        bfgs._x_prev = [0.0, 0.0]
        bfgs._g_prev = [1.0, 0.0]

        # Craft params/grads so that s=[1,0], y=[-1,0] -> s.y = -1 (negative curvature).
        params = [1.0, 0.0]
        grads = [0.0, 0.0]  # g_prev + y = [1,0] + [-1,0] = [0,0]
        bfgs.step(params, grads)

        H = bfgs._H_inv
        # Positive-definite 2x2 check via leading principal minors.
        assert H[0][0] > 0
        assert (H[0][0] * H[1][1] - H[0][1] * H[1][0]) > 0

    def test_step_size_clamped_to_max_step(self):
        bfgs = BFGS(lr=1000.0, max_step=1.0)
        x = [0.0, 0.0]
        x_new = bfgs.step(x, [10.0, 10.0])
        step_norm = sum((a - b) ** 2 for a, b in zip(x_new, x)) ** 0.5
        assert step_norm == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_lengths_raises(self):
        bfgs = BFGS()
        with pytest.raises(ValueError):
            bfgs.step([1.0, 2.0], [1.0])

    def test_nonpositive_eps_raises(self):
        with pytest.raises(ValueError):
            BFGS(eps=0.0)

    def test_nonpositive_max_step_raises(self):
        with pytest.raises(ValueError):
            BFGS(max_step=-1.0)

    def test_reset_clears_state(self):
        bfgs = BFGS()
        bfgs.step([1.0], [1.0])
        bfgs.reset()
        assert bfgs._H_inv is None
        assert bfgs._x_prev is None
        assert bfgs._g_prev is None

    def test_get_config(self):
        bfgs = BFGS(lr=0.5, eps=1e-8, max_step=5.0)
        config = bfgs.get_config()
        assert config["eps"] == 1e-8
        assert config["max_step"] == 5.0
