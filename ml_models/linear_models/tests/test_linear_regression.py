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

@pytest.fixture(scope="module")
def regression_dataset() -> Dict[str, object]:
    """Generate a fixed synthetic regression dataset in both NumPy and Vector/Matrix form.

    Returns
    -------
    dict
        Keys ``X_train_np``, ``X_test_np``, ``y_train_np``, ``y_test_np``
        (NumPy arrays, for scikit-learn) and ``X_train``, ``X_test``,
        ``y_train``, ``y_test`` (Vector/Matrix, for the library under
        test), all derived from the same underlying split.
    """
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES - 1,
        noise=8.0,
        random_state=RANDOM_STATE,
    )
    n_train = int(N_SAMPLES * TRAIN_FRACTION)

    X_train_np, X_test_np = X[:n_train], X[n_train:]
    y_train_np, y_test_np = y[:n_train], y[n_train:]

    return {
        "X_train_np": X_train_np,
        "X_test_np": X_test_np,
        "y_train_np": y_train_np,
        "y_test_np": y_test_np,
        "X_train": _to_matrix(X_train_np),
        "X_test": _to_matrix(X_test_np),
        "y_train": _to_vector(y_train_np),
        "y_test": _to_vector(y_test_np),
    }


class TestLinearRegressionAgainstSklearn:
    """Ordinary least squares must match scikit-learn's closed-form solver."""

    def test_r2_score_matches_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = LinearRegression().fit(regression_dataset["X_train"], regression_dataset["y_train"])
        sk_model = SkLinearRegression().fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_r2 = ours.score(regression_dataset["X_test"], regression_dataset["y_test"])
        sk_r2 = sk_r2_score(
            regression_dataset["y_test_np"], sk_model.predict(regression_dataset["X_test_np"])
        )

        assert our_r2 == pytest.approx(sk_r2, abs=1e-6)

    def test_coefficients_match_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = LinearRegression().fit(regression_dataset["X_train"], regression_dataset["y_train"])
        sk_model = SkLinearRegression().fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_, abs=1e-6)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_):
            assert our_coef == pytest.approx(sk_coef, abs=1e-6)

    def test_fit_intercept_false_matches_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = LinearRegression(fit_intercept=False).fit(
            regression_dataset["X_train"], regression_dataset["y_train"]
        )
        sk_model = SkLinearRegression(fit_intercept=False).fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert sk_model.intercept_ == 0.0
        for our_coef, sk_coef in zip(our_w, sk_model.coef_):
            assert our_coef == pytest.approx(sk_coef, abs=1e-6)
