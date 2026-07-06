"""
test_line_search.py

Run with:  pytest test_line_search.py -v
Requires: pytest, scipy (regression oracle only).
"""

import os
import sys
import math
import pytest
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from line_search import backtracking, golden_section


class TestBacktracking:
    def test_returns_valid_alpha(self):
        f = lambda x: x[0] ** 2
        alpha = backtracking(f, [3.0], grad=[6.0], direction=[-1.0])
        assert 0.0 < alpha <= 1.0

    def test_satisfies_armijo_condition(self):
        f = lambda x: x[0] ** 2
        x, grad, direction = [3.0], [6.0], [-1.0]
        c = 1e-4
        alpha = backtracking(f, x, grad, direction, c=c)
        f0 = f(x)
        slope = sum(g * d for g, d in zip(grad, direction))
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        assert f(x_new) <= f0 + c * alpha * slope

    def test_ascent_direction_raises(self):
        f = lambda x: x[0] ** 2
        with pytest.raises(ValueError, match="descent"):
            backtracking(f, [3.0], grad=[6.0], direction=[1.0])

    def test_nonpositive_alpha_raises(self):
        with pytest.raises(ValueError):
            backtracking(lambda x: x[0] ** 2, [1.0], [2.0], [-1.0], alpha=0.0)

    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError):
            backtracking(lambda x: x[0] ** 2, [1.0], [2.0], [-1.0], rho=1.5)

    def test_c_out_of_range_raises(self):
        with pytest.raises(ValueError):
            backtracking(lambda x: x[0] ** 2, [1.0], [2.0], [-1.0], c=1.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            backtracking(lambda x: x[0] ** 2, [1.0, 2.0], [2.0], [-1.0])

    def test_objective_exception_wrapped(self):
        def bad_f(x):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="Objective function raised"):
            backtracking(bad_f, [1.0], [2.0], [-1.0])

    def test_max_iter_exhausted_logs_warning(self, caplog):
        import logging

        # f is a flat, zero-everywhere function: f(x_new)=0 never satisfies
        # f(x_new) <= f0 + c*alpha*slope (which is slightly negative here),
        # so every trial fails and max_iter is genuinely exhausted.
        with caplog.at_level(logging.WARNING):
            alpha = backtracking(lambda x: 0.0, [1.0], [1.0], [-1.0], max_iter=2)
        assert alpha > 0
        assert any("did not satisfy" in r.message for r in caplog.records)


class TestGoldenSection:
    def test_matches_scipy(self):
        f = lambda x: (x - 2.0) ** 2 + 1.0
        result = golden_section(f, 0.0, 5.0)
        sp = minimize_scalar(f, bounds=(0, 5), method="bounded")
        assert result["x_min"] == pytest.approx(sp.x, abs=1e-4)
        assert result["f_min"] == pytest.approx(sp.fun, abs=1e-6)

    def test_converged_flag_true_on_success(self):
        f = lambda x: (x - 1.0) ** 2
        result = golden_section(f, -5.0, 5.0)
        assert result["converged"] is True

    def test_a_geq_b_raises(self):
        with pytest.raises(ValueError):
            golden_section(lambda x: x ** 2, 5.0, 0.0)

    def test_nan_bounds_raise(self):
        with pytest.raises(ValueError):
            golden_section(lambda x: x ** 2, float("nan"), 5.0)

    def test_inf_bounds_raise(self):
        with pytest.raises(ValueError):
            golden_section(lambda x: x ** 2, 0.0, float("inf"))

    def test_nonpositive_tol_raises(self):
        with pytest.raises(ValueError):
            golden_section(lambda x: x ** 2, 0.0, 1.0, tol=0.0)

    def test_objective_exception_wrapped(self):
        def bad_f(x):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="Objective function raised"):
            golden_section(bad_f, 0.0, 1.0)

    def test_bracket_shrinks(self):
        f = lambda x: (x - 1.0) ** 2
        result = golden_section(f, -5.0, 5.0)
        a, b = result["bracket"]
        assert (b - a) < 10.0
