"""
test_special_functions.py

Run with:  pytest test_special_functions.py -v
Requires: pytest, scipy (regression oracle only).
"""


import sys
import os
import math
import pytest
from scipy import special

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from special_functions import betainc, betaincc, gammainc, gammaincc


class TestBetaInc:
    @pytest.mark.parametrize(
        "a,b,x",
        [
            (2, 3, 0.5), (0.5, 0.5, 0.3), (10, 10, 0.5), (100, 200, 0.4),
            (1, 1, 0.7), (0.1, 5, 0.01), (5, 0.1, 0.99), (1000, 1000, 0.5),
            (2.5, 7.5, 0.9999), (2.5, 7.5, 0.0001),
        ],
    )
    def test_matches_scipy(self, a, b, x):
        assert betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), abs=1e-8)

    def test_boundary_zero(self):
        assert betainc(2, 3, 0.0) == 0.0

    def test_boundary_one(self):
        assert betainc(2, 3, 1.0) == 1.0

    def test_complement_relation(self):
        assert betaincc(3, 4, 0.6) == pytest.approx(1 - betainc(3, 4, 0.6), abs=1e-9)

    def test_x_out_of_range_raises(self):
        with pytest.raises(ValueError):
            betainc(2, 3, 1.5)
        with pytest.raises(ValueError):
            betainc(2, 3, -0.1)

    def test_nonpositive_shape_raises(self):
        with pytest.raises(ValueError):
            betainc(-1, 2, 0.5)
        with pytest.raises(ValueError):
            betainc(2, 0, 0.5)

    def test_nan_x_raises(self):
        with pytest.raises(ValueError):
            betainc(2, 3, float("nan"))

    def test_nan_shape_raises(self):
        with pytest.raises(ValueError):
            betainc(float("nan"), 2, 0.5)

    def test_inf_shape_raises(self):
        with pytest.raises(ValueError):
            betainc(float("inf"), 2, 0.5)

    def test_bool_shape_rejected(self):
        with pytest.raises(TypeError):
            betainc(True, 2, 0.5)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            betainc("a", 2, 0.5)

    def test_result_within_unit_interval(self):
        for x in [0.01, 0.25, 0.5, 0.75, 0.99]:
            r = betainc(3, 5, x)
            assert 0.0 <= r <= 1.0


class TestGammaInc:
    @pytest.mark.parametrize(
        "a,x",
        [
            (2, 3), (0.5, 0.3), (10, 10), (100, 50), (1, 0.001),
            (0.1, 50), (50, 0.1), (1000, 1000), (5, 1e-8), (0.01, 0.01),
        ],
    )
    def test_matches_scipy(self, a, x):
        assert gammainc(a, x) == pytest.approx(special.gammainc(a, x), abs=1e-8)

    def test_boundary_zero(self):
        assert gammainc(2, 0.0) == 0.0

    def test_infinite_x(self):
        assert gammainc(2, float("inf")) == 1.0

    def test_complement_relation(self):
        assert gammaincc(4, 2.5) == pytest.approx(1 - gammainc(4, 2.5), abs=1e-9)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            gammainc(2, -1)

    def test_nonpositive_a_raises(self):
        with pytest.raises(ValueError):
            gammainc(0, 1)
        with pytest.raises(ValueError):
            gammainc(-1, 1)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            gammainc(2, float("nan"))
        with pytest.raises(ValueError):
            gammainc(float("nan"), 2)

    def test_bool_shape_rejected(self):
        with pytest.raises(TypeError):
            gammainc(True, 2)

    def test_result_within_unit_interval(self):
        for x in [0.01, 1, 5, 50]:
            r = gammainc(3, x)
            assert 0.0 <= r <= 1.0

    def test_series_and_cf_branches_agree_near_crossover(self):
        # x < a+1 uses series; x >= a+1 uses continued fraction. Check
        # continuity right around the crossover point.
        a = 5.0
        x_lo = a + 1.0 - 1e-6
        x_hi = a + 1.0 + 1e-6
        assert gammainc(a, x_lo) == pytest.approx(gammainc(a, x_hi), abs=1e-5)
