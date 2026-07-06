"""
test_base.py

Run with:  pytest test_base.py -v
Requires: pytest.
"""

import os
import sys
import math
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import Optimizer
from Vectors.vector import Vector
from Matrix.matrix import Matrix


class DummyOptimizer(Optimizer):
    """Minimal concrete subclass for exercising the base class's shared behavior."""

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] = params[i] - self.lr * grads[i]


class TestOptimizerConstruction:
    def test_default_lr(self):
        opt = DummyOptimizer()
        assert opt.lr == 0.01

    def test_custom_lr(self):
        opt = DummyOptimizer(lr=0.5)
        assert opt.lr == 0.5

    def test_zero_lr_raises(self):
        with pytest.raises(ValueError):
            DummyOptimizer(lr=0.0)

    def test_negative_lr_raises(self):
        with pytest.raises(ValueError):
            DummyOptimizer(lr=-0.1)

    def test_bool_lr_rejected(self):
        with pytest.raises(TypeError):
            DummyOptimizer(lr=True)

    def test_nan_lr_raises(self):
        with pytest.raises(ValueError):
            DummyOptimizer(lr=float("nan"))

    def test_inf_lr_raises(self):
        with pytest.raises(ValueError):
            DummyOptimizer(lr=float("inf"))

    def test_non_numeric_lr_raises(self):
        with pytest.raises(TypeError):
            DummyOptimizer(lr="0.1")


class TestOptimizerSharedBehavior:
    def test_initial_state(self):
        opt = DummyOptimizer()
        assert opt.iterations == 0
        assert opt.history == []

    def test_record_appends_to_history(self):
        opt = DummyOptimizer()
        opt.record(1.0)
        opt.record(0.5)
        assert opt.history == [1.0, 0.5]

    def test_reset_clears_state(self):
        opt = DummyOptimizer()
        opt.record(1.0)
        params = [1.0]
        opt.step(params, [1.0])
        opt.reset()
        assert opt.iterations == 0
        assert opt.history == []

    def test_get_config(self):
        opt = DummyOptimizer(lr=0.05)
        assert opt.get_config() == {"lr": 0.05}

    def test_repr_contains_class_name_and_config(self):
        opt = DummyOptimizer(lr=0.05)
        text = repr(opt)
        assert "DummyOptimizer" in text
        assert "0.05" in text

    def test_step_not_implemented_on_base(self):
        opt = Optimizer(lr=0.1)
        with pytest.raises(NotImplementedError):
            opt.step([1.0], [1.0])


class TestZerosLike:
    def test_float(self):
        opt = DummyOptimizer()
        assert opt._zeros_like(5.0) == 0.0

    def test_int(self):
        opt = DummyOptimizer()
        assert opt._zeros_like(5) == 0.0

    def test_vector(self):
        opt = DummyOptimizer()
        v = Vector([1.0, 2.0, 3.0])
        z = opt._zeros_like(v)
        assert isinstance(z, Vector)
        assert z.components == [0.0, 0.0, 0.0]

    def test_matrix(self):
        opt = DummyOptimizer()
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        z = opt._zeros_like(m)
        assert isinstance(z, Matrix)
        assert z.shape == (2, 2)
        assert all(v == 0.0 for row in z.rows for v in row.components)

    def test_bool_rejected(self):
        opt = DummyOptimizer()
        with pytest.raises(TypeError):
            opt._zeros_like(True)

    def test_unsupported_type_raises(self):
        opt = DummyOptimizer()
        with pytest.raises(TypeError):
            opt._zeros_like("not a number")
