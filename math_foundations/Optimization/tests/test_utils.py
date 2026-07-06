"""
test_utils.py

Run with:  pytest test_utils.py -v
Requires: pytest.
"""

import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    ConvergenceTracker,
    _mat_vec,
    _vec_add,
    _vec_dot,
    _vec_norm,
    _vec_scale,
    _vec_sub,
    numerical_gradient,
    numerical_hessian,
    optimize,
)
from first_order import GradientDescent, Adam
from second_order import BFGS


class TestNumericalGradient:
    def test_matches_analytic_gradient(self):
        # f(x,y) = x^2 + 3xy + y^3, grad = [2x+3y, 3x+3y^2]
        f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 3
        x = [2.0, 1.0]
        g = numerical_gradient(f, x)
        assert g[0] == pytest.approx(2 * 2 + 3 * 1, abs=1e-4)
        assert g[1] == pytest.approx(3 * 2 + 3 * 1 ** 2, abs=1e-4)

    def test_empty_x_raises(self):
        with pytest.raises(ValueError):
            numerical_gradient(lambda x: 0.0, [])

    def test_nonpositive_h_raises(self):
        with pytest.raises(ValueError):
            numerical_gradient(lambda x: x[0], [1.0], h=0.0)

    def test_safe_with_numpy_array_input(self):
        """Regression test: x[:] is a VIEW (not a copy) for numpy arrays,
        so without defensively converting to a list first, perturbing one
        coordinate for the finite-difference step would corrupt the
        caller's original array in place and silently produce wrong
        gradients (or corrupt data the caller still holds a reference to)."""
        np = pytest.importorskip("numpy")
        f = lambda x: x[0] ** 2 + x[1] ** 2
        x = np.array([3.0, 4.0])
        x_before = x.copy()
        g = numerical_gradient(f, x)
        assert g[0] == pytest.approx(6.0, abs=1e-4)
        assert g[1] == pytest.approx(8.0, abs=1e-4)
        # The caller's original array must be untouched.
        assert np.array_equal(x, x_before)


class TestNumericalHessian:
    def test_matches_analytic_hessian(self):
        # f(x,y) = x^2 + 3xy + y^3, Hessian = [[2, 3], [3, 6y]]
        f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 3
        x = [2.0, 1.0]
        H = numerical_hessian(f, x)
        expected = [[2.0, 3.0], [3.0, 6.0]]
        for i in range(2):
            for j in range(2):
                assert H[i][j] == pytest.approx(expected[i][j], abs=1e-2)

    def test_is_symmetric(self):
        f = lambda x: x[0] ** 3 * x[1] ** 2 + x[2]
        x = [1.0, 2.0, 3.0]
        H = numerical_hessian(f, x)
        n = len(x)
        for i in range(n):
            for j in range(n):
                assert H[i][j] == pytest.approx(H[j][i])

    def test_diagonal_uses_correct_step_size(self):
        """Regression test: the diagonal used to be computed via the 4-point
        mixed-partial formula applied to i==j, which (due to how += / -=
        compose on the same index) silently used an effective step of 2h
        instead of h, and needlessly re-evaluated f(x) twice per diagonal
        entry. A pure quadratic's Hessian should be recovered essentially
        exactly regardless."""
        f = lambda x: 3.0 * x[0] ** 2
        H = numerical_hessian(f, [5.0], h=1e-4)
        assert H[0][0] == pytest.approx(6.0, abs=1e-4)

    def test_function_call_count_roughly_halved(self):
        """Regression test: exploiting symmetry should evaluate roughly
        2n^2 + 1 times rather than a naive 4n^2 (which also wastes calls
        re-evaluating f(x) redundantly on the diagonal)."""
        calls = {"n": 0}

        def counted_f(x):
            calls["n"] += 1
            return sum(xi ** 2 for xi in x)

        n = 4
        numerical_hessian(counted_f, [1.0] * n)
        naive_upper_bound = 4 * n * n
        expected = 2 * n * n + 1
        assert calls["n"] == expected
        assert calls["n"] < naive_upper_bound

    def test_empty_x_raises(self):
        with pytest.raises(ValueError):
            numerical_hessian(lambda x: 0.0, [])

    def test_nonpositive_h_raises(self):
        with pytest.raises(ValueError):
            numerical_hessian(lambda x: x[0], [1.0], h=-1.0)

    def test_safe_with_numpy_array_input(self):
        """Regression test: same numpy-array view-vs-copy hazard as
        numerical_gradient, but for the Hessian's 4-point stencil."""
        np = pytest.importorskip("numpy")
        f = lambda x: x[0] ** 2 + x[1] ** 2
        x = np.array([3.0, 4.0])
        x_before = x.copy()
        H = numerical_hessian(f, x)
        assert H[0][0] == pytest.approx(2.0, abs=1e-2)
        assert H[1][1] == pytest.approx(2.0, abs=1e-2)
        assert np.array_equal(x, x_before)


class TestVectorHelpers:
    def test_vec_add_sub(self):
        assert _vec_add([1, 2], [3, 4]) == [4, 6]
        assert _vec_sub([5, 5], [1, 2]) == [4, 3]

    def test_vec_scale(self):
        assert _vec_scale([1, 2, 3], 2) == [2, 4, 6]

    def test_vec_dot(self):
        assert _vec_dot([1, 2], [3, 4]) == 11

    def test_vec_norm(self):
        assert _vec_norm([3, 4]) == pytest.approx(5.0)

    def test_mat_vec(self):
        assert _mat_vec([[1, 2], [3, 4]], [1, 1]) == [3, 7]

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            _vec_add([1, 2], [1, 2, 3])
        with pytest.raises(ValueError):
            _vec_dot([1, 2], [1])


class TestConvergenceTracker:
    def test_update_detects_improvement(self):
        t = ConvergenceTracker(tol=1e-3, patience=3)
        assert t.update(10.0) is False
        assert t.update(5.0) is False  # improved

    def test_patience_triggers_after_n_stalls(self):
        t = ConvergenceTracker(tol=1e-3, patience=3)
        t.update(10.0)
        assert t.update(10.0) is False  # stall 1
        assert t.update(10.0) is False  # stall 2
        assert t.update(10.0) is True  # stall 3 -> patience exceeded

    def test_converged_needs_at_least_two_points(self):
        t = ConvergenceTracker()
        assert t.converged() is False
        t.update(1.0)
        assert t.converged() is False

    def test_converged_true_when_close(self):
        t = ConvergenceTracker(tol=1e-3)
        t.update(1.0)
        t.update(1.0000001)
        assert t.converged() is True

    def test_reset(self):
        t = ConvergenceTracker()
        t.update(1.0)
        t.reset()
        assert t.history == []

    def test_nonpositive_tol_raises(self):
        with pytest.raises(ValueError):
            ConvergenceTracker(tol=0.0)

    def test_nonpositive_patience_raises(self):
        with pytest.raises(ValueError):
            ConvergenceTracker(patience=0)

    def test_plot_ascii_height_one_raises(self):
        """Regression test: height=1 used to crash with ZeroDivisionError
        (dividing by height-1) instead of raising a clear error."""
        t = ConvergenceTracker()
        t.update(1.0)
        with pytest.raises(ValueError):
            t.plot_ascii(height=1)

    def test_plot_ascii_empty_history_does_not_crash(self, capsys):
        t = ConvergenceTracker()
        t.plot_ascii()
        captured = capsys.readouterr()
        assert "No history" in captured.out

    def test_plot_ascii_runs_without_error(self):
        t = ConvergenceTracker()
        for v in [5.0, 4.0, 3.0, 2.0, 1.0]:
            t.update(v)
        t.plot_ascii(height=5)  # should not raise


class TestOptimizeDriver:
    def test_gradient_descent_converges(self):
        f = lambda x: sum(xi ** 2 for xi in x)
        grad_f = lambda x: numerical_gradient(f, x)
        result = optimize(f, grad_f, [5.0, -3.0], GradientDescent(lr=0.1), max_iter=500, patience=200)
        assert result["converged"]
        assert result["loss"] < 1e-8

    def test_adam_converges(self):
        f = lambda x: sum(xi ** 2 for xi in x)
        grad_f = lambda x: numerical_gradient(f, x)
        result = optimize(f, grad_f, [5.0, -3.0], Adam(lr=0.1), max_iter=1000, patience=200)
        assert result["loss"] < 1e-4

    def test_second_order_optimizer_dispatch(self):
        """BFGS.step() returns a new list (rather than mutating in place);
        optimize() must correctly detect and use that return value."""
        f = lambda x: sum(xi ** 2 for xi in x)
        grad_f = lambda x: numerical_gradient(f, x)
        result = optimize(f, grad_f, [5.0, -3.0], BFGS(lr=1.0), max_iter=100, patience=100)
        assert result["loss"] < 1e-6

    def test_plateau_early_stopping_is_wired_up(self):
        """Regression test: ConvergenceTracker.update()'s return value used
        to be computed but never checked by optimize(), so patience-based
        early stopping never actually triggered. Isolate with a constant
        loss (never improves after the first call) and a large constant
        gradient (so the gradient-norm criterion can't be what stops it)."""
        f = lambda x: 5.0
        grad_f = lambda x: [10.0]
        result = optimize(f, grad_f, [1.0], GradientDescent(lr=0.001), max_iter=1000, patience=5, tol=1e-6)
        assert result["iterations"] < 1000
        assert "plateau" in result["stop_reason"]

    def test_gradient_norm_stopping_reason(self):
        f = lambda x: sum(xi ** 2 for xi in x)
        grad_f = lambda x: numerical_gradient(f, x)
        result = optimize(f, grad_f, [0.0000001], GradientDescent(lr=0.1), max_iter=500)
        assert "gradient norm" in result["stop_reason"]

    def test_empty_x0_raises(self):
        with pytest.raises(ValueError):
            optimize(lambda x: 0.0, lambda x: [], [], GradientDescent())

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError):
            optimize(lambda x: 0.0, lambda x: [0.0], [1.0], GradientDescent(), max_iter=0)

    def test_invalid_optimizer_raises(self):
        with pytest.raises(TypeError):
            optimize(lambda x: 0.0, lambda x: [0.0], [1.0], "not an optimizer")

    def test_history_is_recorded(self):
        f = lambda x: sum(xi ** 2 for xi in x)
        grad_f = lambda x: numerical_gradient(f, x)
        result = optimize(f, grad_f, [5.0], GradientDescent(lr=0.1), max_iter=50, patience=200)
        assert len(result["history"]) == result["iterations"]
