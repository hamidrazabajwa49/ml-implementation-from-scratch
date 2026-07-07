"""
Binary logistic regression family: unregularized, L2-regularized
(ridge-style), and L1-regularized (lasso-style) variants.

Every model in this module predicts ``P(y = 1 | x) = sigmoid(X @ w)``
(with an optional intercept term folded into `w` via a prepended bias
column) and is fit by minimizing the binary cross-entropy loss. The
implementation is built exclusively on the math_foundations library
(`Vector`, `Matrix`, `Optimization`, `InformationTheory`) and the
`ml_models` core utilities (`MLModel`, `accuracy`, `sigmoid`). No
third-party numerical package is imported.

`LogisticRegressionL1`'s proximal (soft-thresholding) update is reused
from `linear_models.linear_regression`, since the operator is
mathematically identical regardless of the loss function it is applied
to (only the gradient term differs between lasso regression and
L1-regularized logistic regression).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Union

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from math_foundations.Vectors.vector import Vector 
from math_foundations.Matrix.matrix import Matrix  
from math_foundations.Optimization.base import Optimizer 
from math_foundations.Optimization.first_order import Adam  
from math_foundations.InformationTheory.information_theory import binary_cross_entropy  
from ml_models.base_class import MLModel  
from ml_models.metrics import accuracy  
from ml_models.activations import sigmoid  
from ml_models.linear_models.linear_regression import _soft_threshold  # type: ignore

Parameter = Union[float, Vector, Matrix]



class LogisticRegression(MLModel):
    """Binary logistic regression fit via gradient-based optimization.

    Models the log-odds of the positive class as a linear function of
    the input features and minimizes the binary cross-entropy loss:

    ``L(w) = -(1 / n_samples) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))``

    where ``p_i = sigmoid(x_i . w)``. The gradient of this loss with
    respect to `w` has the compact closed form:

    ``grad(L) = (1 / n_samples) * X^T (sigmoid(Xw) - y)``

    which is identical in structure to ordinary least squares
    regression's gradient, differing only through the sigmoid
    nonlinearity applied to the linear predictor.

    Parameters
    ----------
    lr : float, optional
        Learning rate used only when `fit()` is called without an
        explicit `optimizer`; must be positive. Defaults to 0.01.
    n_iter : int, optional
        Number of gradient-descent iterations; must be at least 1.
        Defaults to 1000.
    fit_intercept : bool, optional
        Whether to prepend a bias (intercept) column of ones to the
        design matrix. Defaults to True.

    Raises
    ------
    ValueError
        If `lr` is not positive or `n_iter` is less than 1.
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000, fit_intercept: bool = True) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Matrix, y: Vector, optimizer: Optional[Optimizer] = None) -> "LogisticRegression":
        """Fit the model via iterative gradient-based optimization.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Binary target vector (0 or 1) of length n_samples.
        optimizer : Optimizer, optional
            An instance conforming to `Optimization.base.Optimizer`
            (e.g. `Adam`, `SGD`). If omitted, a fresh `Adam` optimizer
            with learning rate `self.lr` is constructed.

        Returns
        -------
        LogisticRegression
            The fitted instance, for method chaining.

        Raises
        ------
        TypeError
            If `X`/`y` have the wrong type, or `optimizer` is provided
            but is not an `Optimizer` instance.
        ValueError
            If `X`/`y` are empty or mismatched, or `y` contains a
            non-binary label.
        """
        self._validate_Xy(X, y)
        _validate_binary_targets(y)
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}"
            )

        self.n_features_in_ = X.n_cols
        design = self._add_bias_column(X) if self.fit_intercept else X
        self.w: Vector = Vector([0.0] * design.n_cols)

        if optimizer is None:
            optimizer = Adam(lr=self.lr)

        inv_n = 1.0 / design.n_rows
        for iteration in range(self.n_iter):
            y_pred = sigmoid(design * self.w)
            error = y_pred - y
            gradient = inv_n * (design.transpose() * error)

            params = [self.w]
            optimizer.step(params, [gradient])
            self.w = params[0]

            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                optimizer.record(binary_cross_entropy(list(y), list(y_pred)))

        self._is_fitted = True
        return self

    def predict_proba(self, X: Matrix) -> Vector:
        """Predict the probability of the positive class for new samples.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features_in_).

        Returns
        -------
        Vector
            Predicted probabilities in (0, 1), one per row of `X`.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        TypeError
            If `X` is not a `Matrix`.
        ValueError
            If `X` is empty or its column count does not match the
            number of features observed during `fit()`.
        """
        self._check_is_fitted()
        self._validate_X(X, expected_n_features=self.n_features_in_)
        design = self._add_bias_column(X) if self.fit_intercept else X
        return sigmoid(design * self.w)

    def predict(self, X: Matrix, threshold: float = 0.5) -> Vector:
        """Predict binary class labels for new samples.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features_in_).
        threshold : float, optional
            Decision threshold applied to the predicted probability;
            must be strictly between 0 and 1. Defaults to 0.5.

        Returns
        -------
        Vector
            Predicted labels (0.0 or 1.0), one per row of `X`.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        TypeError
            If `X` is not a `Matrix`.
        ValueError
            If `X` is malformed (see `predict_proba`), or `threshold`
            is not strictly between 0 and 1.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        probabilities = self.predict_proba(X)
        return Vector([1.0 if p >= threshold else 0.0 for p in probabilities.components])

    def score(self, X: Matrix, y: Vector) -> float:
        """Compute classification accuracy on (X, y).

        Parameters
        ----------
        X : Matrix
            Feature matrix.
        y : Vector
            True binary labels corresponding to the rows of `X`.

        Returns
        -------
        float
            Accuracy in [0, 1]; see `ml_models.metrics.accuracy`.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        TypeError, ValueError
            See `predict()` and `ml_models.metrics.accuracy`.
        """
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def parameters(self) -> List[Parameter]:
        """Return the fitted parameter vector.

        Returns
        -------
        list of Vector
            A single-element list ``[w]``. If `fit_intercept` is True,
            ``w[0]`` is the intercept and the remaining entries are the
            per-feature coefficients, in the original column order.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        """
        self._check_is_fitted()
        return [self.w]


class LogisticRegressionL2(LogisticRegression):
    """Logistic regression with an L2 (ridge-style) penalty on the feature weights.

    Minimizes the L2-penalized binary cross-entropy loss:

    ``L(w) = BCE(w) + (lam / (2 * n_samples)) * ||w_features||^2``

    where the intercept term (if `fit_intercept` is True) is excluded
    from the penalty, following the standard regularized-GLM
    convention. The corresponding gradient adds ``(lam / n_samples) *
    w_j`` to each penalized coefficient's gradient.

    Parameters
    ----------
    lam : float, optional
        L2 regularization strength; must be non-negative. Defaults to
        1.0.
    lr : float, optional
        Learning rate used only when `fit()` is called without an
        explicit `optimizer`; must be positive. Defaults to 0.01.
    n_iter : int, optional
        Number of gradient-descent iterations; must be at least 1.
        Defaults to 1000.
    fit_intercept : bool, optional
        Whether to prepend a bias (intercept) column of ones to the
        design matrix. Defaults to True.

    Raises
    ------
    ValueError
        If `lam` is negative, `lr` is not positive, or `n_iter` is
        less than 1.
    """
