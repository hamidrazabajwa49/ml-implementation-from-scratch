"""
Correctness tests for `ml_models.linear_models.linear_regression`.

Each model is trained on an identical synthetic regression dataset
(generated once via scikit-learn's `make_regression` for reproducible,
well-conditioned features) and compared against the equivalent
scikit-learn estimator on held-out test data. NumPy and scikit-learn
are test-only dependencies, used exclusively as a correctness oracle
and dataset generator; they are never imported by the library code
under test.

Run with:
    pytest ml_models/linear_models/tests/test_linear_regression.py -v
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.metrics import r2_score as sk_r2_score

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Now that _project_root is in sys.path, use full paths
from math_foundations.Vectors.vector import Vector
from math_foundations.Matrix.matrix import Matrix

# Similarly for ml_models
from ml_models.base_class import NotFittedError
from ml_models.linear_models.linear_regression import (
    GDLinearRegression,
    LassoRegression,
    LinearRegression,
    RidgeRegression,
)

N_SAMPLES = 200
N_FEATURES = 5
TRAIN_FRACTION = 0.75
RANDOM_STATE = 42


def _to_matrix(array: np.ndarray) -> Matrix:
    """Convert a 2D NumPy array into a `Matrix`."""
    return Matrix(array.tolist())


def _to_vector(array: np.ndarray) -> Vector:
    """Convert a 1D NumPy array into a `Vector`."""
    return Vector(array.tolist())
