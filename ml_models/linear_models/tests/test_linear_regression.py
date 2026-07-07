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


class TestRidgeRegressionAgainstSklearn:
    """Closed-form ridge regression must match scikit-learn's `Ridge` (identical convention)."""

    LAM = 2.0

    def test_r2_score_matches_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = RidgeRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], lam=self.LAM
        )
        sk_model = SkRidge(alpha=self.LAM).fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_r2 = ours.score(regression_dataset["X_test"], regression_dataset["y_test"])
        sk_r2 = sk_r2_score(
            regression_dataset["y_test_np"], sk_model.predict(regression_dataset["X_test_np"])
        )

        assert our_r2 == pytest.approx(sk_r2, abs=1e-6)

    def test_coefficients_match_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = RidgeRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], lam=self.LAM
        )
        sk_model = SkRidge(alpha=self.LAM).fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_, abs=1e-6)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_):
            assert our_coef == pytest.approx(sk_coef, abs=1e-6)

    def test_zero_lambda_matches_ordinary_least_squares(
        self, regression_dataset: Dict[str, object]
    ) -> None:
        ridge = RidgeRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], lam=0.0
        )
        ols = LinearRegression().fit(regression_dataset["X_train"], regression_dataset["y_train"])

        ridge_w = ridge.parameters()[0].components
        ols_w = ols.parameters()[0].components
        for ridge_coef, ols_coef in zip(ridge_w, ols_w):
            assert ridge_coef == pytest.approx(ols_coef, abs=1e-6)


class TestLassoRegressionAgainstSklearn:
    """Coordinate-descent lasso must closely track scikit-learn's `Lasso` (same objective)."""

    LAM = 0.5

    def test_r2_score_close_to_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = LassoRegression().fit(
            regression_dataset["X_train"],
            regression_dataset["y_train"],
            lam=self.LAM,
            n_iters=2000,
            tol=1e-6,
        )
        sk_model = SkLasso(alpha=self.LAM, max_iter=10_000).fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_r2 = ours.score(regression_dataset["X_test"], regression_dataset["y_test"])
        sk_r2 = sk_r2_score(
            regression_dataset["y_test_np"], sk_model.predict(regression_dataset["X_test_np"])
        )

        assert our_r2 == pytest.approx(sk_r2, abs=1e-3)

    def test_coefficients_close_to_sklearn(self, regression_dataset: Dict[str, object]) -> None:
        ours = LassoRegression().fit(
            regression_dataset["X_train"],
            regression_dataset["y_train"],
            lam=self.LAM,
            n_iters=2000,
            tol=1e-6,
        )
        sk_model = SkLasso(alpha=self.LAM, max_iter=10_000).fit(
            regression_dataset["X_train_np"], regression_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_, abs=1e-2)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_):
            assert our_coef == pytest.approx(sk_coef, abs=1e-2)

    def test_large_lambda_produces_sparse_coefficients(
        self, regression_dataset: Dict[str, object]
    ) -> None:
        ours = LassoRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], lam=50.0, n_iters=2000
        )
        feature_coefficients = ours.parameters()[0].components[1:]
        assert any(coef == 0.0 for coef in feature_coefficients)

class TestGDLinearRegressionAgainstOLS:
    """Gradient-descent linear regression should converge close to the closed-form solution."""

    def test_r2_score_close_to_closed_form_ols(self, regression_dataset: Dict[str, object]) -> None:
        gd_model = GDLinearRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], n_iter=3000, lr=0.1
        )
        ols_model = LinearRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"]
        )

        gd_r2 = gd_model.score(regression_dataset["X_test"], regression_dataset["y_test"])
        ols_r2 = ols_model.score(regression_dataset["X_test"], regression_dataset["y_test"])

        assert gd_r2 == pytest.approx(ols_r2, abs=1e-3)

    def test_coefficients_close_to_closed_form_ols(
        self, regression_dataset: Dict[str, object]
    ) -> None:
        gd_model = GDLinearRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"], n_iter=3000, lr=0.1
        )
        ols_model = LinearRegression().fit(
            regression_dataset["X_train"], regression_dataset["y_train"]
        )

        gd_w = gd_model.parameters()[0].components
        ols_w = ols_model.parameters()[0].components
        for gd_coef, ols_coef in zip(gd_w, ols_w):
            assert gd_coef == pytest.approx(ols_coef, abs=1e-2)


class TestInputValidationAndEdgeCases:
    """Every model must fail fast and specifically on malformed input or misuse."""

    @pytest.mark.parametrize(
        "model_cls", [LinearRegression, RidgeRegression, LassoRegression, GDLinearRegression]
    )
    def test_predict_before_fit_raises_not_fitted_error(self, model_cls, regression_dataset) -> None:
        model = model_cls()
        with pytest.raises(NotFittedError):
            model.predict(regression_dataset["X_test"])

    @pytest.mark.parametrize(
        "model_cls", [LinearRegression, RidgeRegression, LassoRegression, GDLinearRegression]
    )
    def test_fit_rejects_mismatched_sample_counts(self, model_cls, regression_dataset) -> None:
        y_wrong_length = Vector(list(regression_dataset["y_train"])[:-1])
        with pytest.raises(ValueError):
            model_cls().fit(regression_dataset["X_train"], y_wrong_length)

    @pytest.mark.parametrize(
        "model_cls", [LinearRegression, RidgeRegression, LassoRegression, GDLinearRegression]
    )
    def test_fit_rejects_non_matrix_X(self, model_cls, regression_dataset) -> None:
        with pytest.raises(TypeError):
            model_cls().fit([[1.0, 2.0], [3.0, 4.0]], regression_dataset["y_train"][:2])

    def test_predict_rejects_feature_count_mismatch(self, regression_dataset) -> None:
        model = LinearRegression().fit(regression_dataset["X_train"], regression_dataset["y_train"])
        truncated = _to_matrix(regression_dataset["X_test_np"][:, :N_FEATURES - 2])
        with pytest.raises(ValueError):
            model.predict(truncated)

    def test_ridge_rejects_negative_lambda(self, regression_dataset) -> None:
        with pytest.raises(ValueError):
            RidgeRegression().fit(
                regression_dataset["X_train"], regression_dataset["y_train"], lam=-1.0
            )

    def test_lasso_rejects_non_positive_tolerance(self, regression_dataset) -> None:
        with pytest.raises(ValueError):
            LassoRegression().fit(
                regression_dataset["X_train"], regression_dataset["y_train"], tol=0.0
            )

    def test_gd_rejects_non_positive_learning_rate(self, regression_dataset) -> None:
        with pytest.raises(ValueError):
            GDLinearRegression().fit(
                regression_dataset["X_train"], regression_dataset["y_train"], lr=0.0
            )

    def test_gd_rejects_invalid_optimizer_type(self, regression_dataset) -> None:
        with pytest.raises(TypeError):
            GDLinearRegression().fit(
                regression_dataset["X_train"], regression_dataset["y_train"], optimizer="not_an_optimizer"
            )

    def test_parameters_length_matches_bias_convention(self, regression_dataset) -> None:
        with_intercept = LinearRegression(fit_intercept=True).fit(
            regression_dataset["X_train"], regression_dataset["y_train"]
        )
        without_intercept = LinearRegression(fit_intercept=False).fit(
            regression_dataset["X_train"], regression_dataset["y_train"]
        )
        assert len(with_intercept.parameters()[0]) == N_FEATURES + 1
        assert len(without_intercept.parameters()[0]) == N_FEATURES

