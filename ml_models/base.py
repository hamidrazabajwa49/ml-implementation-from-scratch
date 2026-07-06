"""
Abstract base class shared by every supervised learning model in the
ml_models package.

This module centralizes the common fit/predict/score/parameters
interface, input validation, and fitted-state tracking so that concrete
model implementations (linear models, SVMs, tree-based models, neural
networks, and so on) only need to override what is genuinely specific
to that algorithm.

Dependencies are limited to the math_foundations library (`Vector`,
`Matrix`) and the Python standard library. No third-party numerical
package is imported.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Union

_script_dir = os.path.dirname(os.path.abspath(__file__))
_shared_parent = os.path.abspath(os.path.join(_script_dir, os.pardir))
_target_root = os.path.join(_shared_parent, "math_foundations")

if _target_root not in sys.path:
    sys.path.insert(0, _target_root)

from Vectors.vector import Vector  
from Matrix.matrix import Matrix  


Parameter = Union[float, Vector, Matrix]


class NotFittedError(RuntimeError):
    """Raised when an estimator is used before `fit()` has completed.

    This mirrors the intent of scikit-learn's exception of the same
    name: any call to `predict()`, `score()`, or `parameters()` on an
    unfitted model is a programmer error and should fail loudly and
    specifically, rather than raising an unrelated `AttributeError`
    deep inside model-specific code.
    """


class MLModel(ABC):
    """Abstract base class for supervised learning models.

    Concrete subclasses must implement `fit`, `predict`, `score`, and
    `parameters`. Every subclass operates on `Vector`/`Matrix` objects
    from the math_foundations library rather than raw Python lists,
    which keeps the numerical core of every algorithm consistent.

    Attributes
    ----------
    _is_fitted : bool
        True once `fit()` has completed successfully. Subclasses are
        responsible for setting this to True at the end of a
        successful `fit()` call (typically via `self._is_fitted =
        True`), since the specific parameters learned (a weight
        vector, a set of support vectors, a tree structure, and so on)
        differ by algorithm and cannot be generalized here.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: Matrix, y: Vector) -> "MLModel":
        """Fit the model to training data.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector of length n_samples.

        Returns
        -------
        MLModel
            The fitted instance, returned to support method chaining
            (e.g. `model.fit(X, y).predict(X_test)`).
        """
        raise NotImplementedError("Subclasses must implement fit().")

    @abstractmethod
    def predict(self, X: Matrix) -> Vector:
        """Generate predictions for new data.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Vector
            Predicted values, one per row of X.
        """
        raise NotImplementedError("Subclasses must implement predict().")

    @abstractmethod
    def score(self, X: Matrix, y: Vector) -> float:
        """Evaluate the model against held-out data.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            True target values corresponding to the rows of X.

        Returns
        -------
        float
            A model-appropriate performance score (e.g. R-squared for
            regressors, accuracy for classifiers).
        """
        raise NotImplementedError("Subclasses must implement score().")

    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """Return the model's learned parameters.

        Returns
        -------
        list of (float or Vector or Matrix)
            The fitted parameters, in a model-specific, documented
            order (e.g. `[weights, bias]`).
        """
        raise NotImplementedError("Subclasses must implement parameters().")

    def _validate_Xy(self, X: Matrix, y: Vector) -> None:
        """Validate a training feature matrix and target vector.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector; must have one entry per row of X.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix` or `y` is not a `Vector`.
        ValueError
            If `X` has zero rows or zero columns, or if the number of
            rows in `X` does not match the length of `y`.
        """
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}.")
        if not isinstance(y, Vector):
            raise TypeError(f"y must be a Vector, got {type(y).__name__}.")
        if X.n_rows == 0 or X.n_cols == 0:
            raise ValueError("X must not be empty (zero rows or zero columns).")
        if X.n_rows != len(y):
            raise ValueError(
                "X and y must have the same number of samples: "
                f"X has {X.n_rows} rows but y has {len(y)} elements."
            )

    def _validate_X(self, X: Matrix, expected_n_features: Optional[int] = None) -> None:
        """Validate a feature matrix supplied at inference time.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        expected_n_features : int, optional
            If given, `X` must have exactly this many columns. This is
            typically the number of features observed during `fit()`,
            used to reject inference-time data with a mismatched
            schema before it silently produces garbage predictions.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix`.
        ValueError
            If `X` is empty, or its column count does not match
            `expected_n_features`.
        """
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}.")
        if X.n_rows == 0 or X.n_cols == 0:
            raise ValueError("X must not be empty (zero rows or zero columns).")
        if expected_n_features is not None and X.n_cols != expected_n_features:
            raise ValueError(
                f"X has {X.n_cols} feature column(s), but this model was "
                f"fitted on {expected_n_features} feature column(s)."
            )

    def _check_is_fitted(self) -> None:
        """Ensure the model has been fitted before it is used.

        Raises
        ------
        NotFittedError
            If `fit()` has not yet completed successfully on this
            instance.
        """
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted. Call fit() before "
                "predict(), score(), or parameters()."
            )

    @staticmethod
    def _add_bias_column(X: Matrix) -> Matrix:
        """Prepend a constant column of 1.0 to a feature matrix.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Matrix
            A new matrix of shape (n_samples, n_features + 1), where
            column 0 is the intercept (bias) term and the remaining
            columns are the original features, in order.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix`.
        ValueError
            If `X` has zero rows.
        """
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}.")
        if X.n_rows == 0:
            raise ValueError("Cannot add a bias column to an empty matrix.")
        augmented_rows = [[1.0] + list(row.components) for row in X.rows]
        return Matrix(augmented_rows)

    def __repr__(self) -> str:
        status = "fitted" if getattr(self, "_is_fitted", False) else "unfitted"
        return f"{type(self).__name__}({status})"


# `MLModels`. New algorithm modules should import `MLModel`.
MLModels = MLModel
