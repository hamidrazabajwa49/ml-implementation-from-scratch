# ml_models/linear_models/tests/test_logistic_regression.py
"""
Correctness tests for `ml_models.linear_models.logistic_regression`.

Each model is trained on an identical synthetic binary classification
dataset (generated once via scikit-learn's `make_classification` for
reproducible, linearly-separable-ish features) and compared against
the equivalent scikit-learn `LogisticRegression` configuration on
held-out test data. NumPy and scikit-learn are test-only dependencies,
used exclusively as a correctness oracle and dataset generator; they
are never imported by the library code under test.

Regularization strengths are related by ``C = 1 / lam``: scikit-learn
minimizes ``penalty(w) + C * sum_i(log_loss_i)``, while this library
minimizes ``mean_i(log_loss_i) + (lam / n_samples) * penalty(w)``
(L2) or ``mean_i(log_loss_i) + (lam / n_samples) * penalty(w)`` (L1);
dividing scikit-learn's objective by ``C * n_samples`` shows the two
conventions coincide exactly when ``lam = 1 / C``.

Run with:
    pytest ml_models/linear_models/tests/test_logistic_regression.py -v
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score as sk_accuracy_score

_current_dir = os.path.dirname(os.path.abspath(__file__))
_linear_models_dir = os.path.abspath(os.path.join(_current_dir, os.pardir))
_ml_models_dir = os.path.abspath(os.path.join(_linear_models_dir, os.pardir))
_main_folder = os.path.abspath(os.path.join(_ml_models_dir, os.pardir))
if _main_folder not in sys.path:
    sys.path.insert(0, _main_folder)

from Vectors.vector import Vector  # type: ignore
from Matrix.matrix import Matrix  # type: ignore
from ml_models.base_class import NotFittedError  # type: ignore
from ml_models.linear_models.logistic_regression import (  # type: ignore
    LogisticRegression,
    LogisticRegressionL1,
    LogisticRegressionL2,
)

N_SAMPLES = 400
N_FEATURES = 6
N_TRAIN = 300
RANDOM_STATE = 42


def _to_matrix(array: np.ndarray) -> Matrix:
    """Convert a 2D NumPy array into a `Matrix`."""
    return Matrix(array.tolist())


def _to_vector(array: np.ndarray) -> Vector:
    """Convert a 1D NumPy array into a `Vector` of floats."""
    return Vector(array.astype(float).tolist())


@pytest.fixture(scope="module")
def classification_dataset() -> Dict[str, object]:
    """Generate a fixed synthetic binary classification dataset.

    Returns
    -------
    dict
        Keys ``X_train_np``, ``X_test_np``, ``y_train_np``, ``y_test_np``
        (NumPy arrays, for scikit-learn) and ``X_train``, ``X_test``,
        ``y_train``, ``y_test`` (Vector/Matrix, for the library under
        test), all derived from the same underlying split.
    """
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_FEATURES - 2,
        n_redundant=0,
        class_sep=1.5,
        random_state=RANDOM_STATE,
    )
    X_train_np, X_test_np = X[:N_TRAIN], X[N_TRAIN:]
    y_train_np, y_test_np = y[:N_TRAIN], y[N_TRAIN:]

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


class TestLogisticRegressionAgainstSklearn:
    """Unregularized logistic regression should reach comparable accuracy and coefficients."""

    def test_accuracy_matches_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegression(lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(penalty=None, max_iter=5000).fit(
            classification_dataset["X_train_np"], classification_dataset["y_train_np"]
        )

        our_acc = ours.score(classification_dataset["X_test"], classification_dataset["y_test"])
        sk_acc = sk_accuracy_score(
            classification_dataset["y_test_np"], sk_model.predict(classification_dataset["X_test_np"])
        )

        assert our_acc == pytest.approx(sk_acc, abs=0.05)

    def test_coefficients_close_to_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegression(lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(penalty=None, max_iter=5000).fit(
            classification_dataset["X_train_np"], classification_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_[0], abs=0.1)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_[0]):
            assert our_coef == pytest.approx(sk_coef, abs=0.1)

    def test_predict_proba_is_bounded(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegression(lr=0.1, n_iter=1000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        probabilities = ours.predict_proba(classification_dataset["X_test"]).components
        assert all(0.0 < p < 1.0 for p in probabilities)


class TestLogisticRegressionL2AgainstSklearn:
    """L2-regularized logistic regression should match scikit-learn under the equivalent C."""

    LAM = 1.0

    def test_accuracy_matches_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegressionL2(lam=self.LAM, lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(penalty="l2", C=1.0 / self.LAM, max_iter=5000).fit(
            classification_dataset["X_train_np"], classification_dataset["y_train_np"]
        )

        our_acc = ours.score(classification_dataset["X_test"], classification_dataset["y_test"])
        sk_acc = sk_accuracy_score(
            classification_dataset["y_test_np"], sk_model.predict(classification_dataset["X_test_np"])
        )

        assert our_acc == pytest.approx(sk_acc, abs=0.05)

    def test_coefficients_close_to_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegressionL2(lam=self.LAM, lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(penalty="l2", C=1.0 / self.LAM, max_iter=5000).fit(
            classification_dataset["X_train_np"], classification_dataset["y_train_np"]
        )

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_[0], abs=0.05)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_[0]):
            assert our_coef == pytest.approx(sk_coef, abs=0.05)

    def test_penalty_shrinks_coefficients_relative_to_unregularized(
        self, classification_dataset: Dict[str, object]
    ) -> None:
        unregularized = LogisticRegression(lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        regularized = LogisticRegressionL2(lam=10.0, lr=0.1, n_iter=3000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )

        unregularized_norm = sum(c ** 2 for c in unregularized.parameters()[0].components[1:])
        regularized_norm = sum(c ** 2 for c in regularized.parameters()[0].components[1:])
        assert regularized_norm < unregularized_norm


class TestLogisticRegressionL1AgainstSklearn:
    """L1-regularized logistic regression should closely track scikit-learn's `liblinear` solver."""

    LAM = 0.1

    def test_accuracy_matches_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegressionL1(lam=self.LAM, lr=0.1, n_iter=5000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(
            penalty="l1", C=1.0 / self.LAM, solver="liblinear", max_iter=5000
        ).fit(classification_dataset["X_train_np"], classification_dataset["y_train_np"])

        our_acc = ours.score(classification_dataset["X_test"], classification_dataset["y_test"])
        sk_acc = sk_accuracy_score(
            classification_dataset["y_test_np"], sk_model.predict(classification_dataset["X_test_np"])
        )

        assert our_acc == pytest.approx(sk_acc, abs=0.05)

    def test_coefficients_close_to_sklearn(self, classification_dataset: Dict[str, object]) -> None:
        ours = LogisticRegressionL1(lam=self.LAM, lr=0.1, n_iter=5000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        sk_model = SkLogisticRegression(
            penalty="l1", C=1.0 / self.LAM, solver="liblinear", max_iter=5000
        ).fit(classification_dataset["X_train_np"], classification_dataset["y_train_np"])

        our_w = ours.parameters()[0].components
        assert our_w[0] == pytest.approx(sk_model.intercept_[0], abs=0.1)
        for our_coef, sk_coef in zip(our_w[1:], sk_model.coef_[0]):
            assert our_coef == pytest.approx(sk_coef, abs=0.1)

    def test_loss_history_is_recorded_and_non_increasing_on_average(
        self, classification_dataset: Dict[str, object]
    ) -> None:
        model = LogisticRegressionL1(lam=self.LAM, lr=0.1, n_iter=1000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        assert len(model.loss_history) > 1
        assert model.loss_history[-1] < model.loss_history[0]

    def test_large_lambda_produces_sparse_coefficients(
        self, classification_dataset: Dict[str, object]
    ) -> None:
        model = LogisticRegressionL1(lam=50.0, lr=0.1, n_iter=2000).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        feature_coefficients = model.parameters()[0].components[1:]
        assert any(coef == 0.0 for coef in feature_coefficients)


class TestInputValidationAndEdgeCases:
    """Every model must fail fast and specifically on malformed input or misuse."""

    @pytest.mark.parametrize(
        "model_cls", [LogisticRegression, LogisticRegressionL2, LogisticRegressionL1]
    )
    def test_predict_before_fit_raises_not_fitted_error(self, model_cls, classification_dataset) -> None:
        model = model_cls()
        with pytest.raises(NotFittedError):
            model.predict(classification_dataset["X_test"])

    @pytest.mark.parametrize(
        "model_cls", [LogisticRegression, LogisticRegressionL2, LogisticRegressionL1]
    )
    def test_fit_rejects_non_binary_targets(self, model_cls, classification_dataset) -> None:
        n = len(classification_dataset["y_train"])
        multiclass_targets = Vector([float(i % 3) for i in range(n)])
        with pytest.raises(ValueError):
            model_cls().fit(classification_dataset["X_train"], multiclass_targets)

    @pytest.mark.parametrize(
        "model_cls", [LogisticRegression, LogisticRegressionL2, LogisticRegressionL1]
    )
    def test_fit_rejects_mismatched_sample_counts(self, model_cls, classification_dataset) -> None:
        y_wrong_length = Vector(list(classification_dataset["y_train"])[:-1])
        with pytest.raises(ValueError):
            model_cls().fit(classification_dataset["X_train"], y_wrong_length)

    def test_predict_rejects_feature_count_mismatch(self, classification_dataset) -> None:
        model = LogisticRegression().fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        truncated = _to_matrix(classification_dataset["X_test_np"][:, : N_FEATURES - 2])
        with pytest.raises(ValueError):
            model.predict(truncated)

    def test_predict_rejects_invalid_threshold(self, classification_dataset) -> None:
        model = LogisticRegression().fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        with pytest.raises(ValueError):
            model.predict(classification_dataset["X_test"], threshold=1.5)

    def test_constructor_rejects_non_positive_learning_rate(self) -> None:
        with pytest.raises(ValueError):
            LogisticRegression(lr=0.0)

    def test_constructor_rejects_non_positive_n_iter(self) -> None:
        with pytest.raises(ValueError):
            LogisticRegression(n_iter=0)

    def test_l2_rejects_negative_lambda(self) -> None:
        with pytest.raises(ValueError):
            LogisticRegressionL2(lam=-1.0)

    def test_l1_rejects_negative_lambda(self) -> None:
        with pytest.raises(ValueError):
            LogisticRegressionL1(lam=-1.0)

    def test_fit_rejects_invalid_optimizer_type(self, classification_dataset) -> None:
        with pytest.raises(TypeError):
            LogisticRegression().fit(
                classification_dataset["X_train"],
                classification_dataset["y_train"],
                optimizer="not_an_optimizer",
            )

    def test_parameters_length_matches_bias_convention(self, classification_dataset) -> None:
        with_intercept = LogisticRegression(fit_intercept=True, n_iter=100).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        without_intercept = LogisticRegression(fit_intercept=False, n_iter=100).fit(
            classification_dataset["X_train"], classification_dataset["y_train"]
        )
        assert len(with_intercept.parameters()[0]) == N_FEATURES + 1
        assert len(without_intercept.parameters()[0]) == N_FEATURES
