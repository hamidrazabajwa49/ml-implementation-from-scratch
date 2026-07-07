"""
Linear regression family of models: ordinary least squares (solved both
in closed form and via gradient-based optimization), ridge (L2) regression,
and lasso (L1) regression.

Every model in this module predicts ``y_hat = X @ w`` (with an optional
intercept term folded into `w` via a prepended bias column) and is built
exclusively on the math_foundations library (`Vector`, `Matrix`,
`Optimization`) and the `ml_models` core utilities (`MLModel`, `r2_score`).
No third-party numerical package is imported.
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
from math_foundations.Matrix.matrix import Matrix, SingularMatrixError 
from math_foundations.Optimization.base import Optimizer 
from math_foundations.Optimization.first_order import Adam 

from ml_models.base_class import MLModel 
from ml_models.metrics import r2_score

Parameter = Union[float, Vector, Matrix]


def _safe_inverse(matrix: Matrix, ridge: float = 1e-8) -> Matrix:
    """Invert a square matrix, applying a small ridge correction if singular.

    Attempts a direct inverse first. If the matrix is exactly singular
    (e.g. `X^T X` when features are collinear or `n_samples <
    n_features`), a small multiple of the identity matrix is added
    before retrying. This is a standard numerical safeguard for
    normal-equation solves and does not materially change the solution
    for well-conditioned inputs, since it is only triggered when the
    direct inverse fails.

    Parameters
    ----------
    matrix : Matrix
        A square matrix to invert.
    ridge : float, optional
        Magnitude of the identity-matrix correction applied on retry;
        must be positive.

    Returns
    -------
    Matrix
        The inverse of `matrix`, or of the ridge-corrected matrix if
        `matrix` itself was singular.

    Raises
    ------
    ValueError
        If `ridge` is not positive.
    SingularMatrixError
        If the matrix remains singular even after the ridge correction.
    """
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive, got {ridge}")
    try:
        return matrix.inverse()
    except SingularMatrixError:
        corrected = matrix + Matrix.identity(matrix.n_rows) * ridge
        return corrected.inverse()


def _soft_threshold(z: float, alpha: float) -> float:
    """Apply the soft-thresholding operator used by L1-regularized coordinate descent.

    ``S(z, alpha) = sign(z) * max(|z| - alpha, 0)``

    Parameters
    ----------
    z : float
        The value to threshold.
    alpha : float
        The non-negative threshold magnitude.

    Returns
    -------
    float
        The thresholded value.

    Raises
    ------
    ValueError
        If `alpha` is negative.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if z > alpha:
        return z - alpha
    if z < -alpha:
        return z + alpha
    return 0.0


class LinearRegression(MLModel):
    """Ordinary least squares linear regression, solved via the normal equations.

    Fits ``y = X @ w`` by minimizing the residual sum of squares
    ``||y - Xw||^2`` in closed form:

    ``w = (X^T X)^-1 X^T y``

    where `X` denotes the design matrix, augmented with a leading
    column of ones when `fit_intercept` is True.

    Parameters
    ----------
    fit_intercept : bool, optional
        Whether to prepend a bias (intercept) column of ones to the
        design matrix. Defaults to True.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        super().__init__()
        self.fit_intercept = fit_intercept
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Matrix, y: Vector) -> "LinearRegression":
        """Fit the model using the ordinary least squares normal equations.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector of length n_samples.

        Returns
        -------
        LinearRegression
            The fitted instance, for method chaining.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix` or `y` is not a `Vector`.
        ValueError
            If `X`/`y` are empty or their sample counts do not match.
        SingularMatrixError
            If `X^T X` remains singular even after the ridge-corrected
            fallback in `_safe_inverse` (this indicates a degenerate
            design matrix, e.g. perfectly collinear features).
        """
        self._validate_Xy(X, y)
        self.n_features_in_ = X.n_cols

        design = self._add_bias_column(X) if self.fit_intercept else X
        design_t = design.transpose()
        gram_inv = _safe_inverse(design_t * design)
        self.w: Vector = gram_inv * (design_t * y)

        self._is_fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        """Predict target values for new samples.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features_in_).

        Returns
        -------
        Vector
            Predicted values, one per row of `X`.

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
        return design * self.w

    def score(self, X: Matrix, y: Vector) -> float:
        """Compute the coefficient of determination (R-squared) on (X, y).

        Parameters
        ----------
        X : Matrix
            Feature matrix.
        y : Vector
            True target values corresponding to the rows of `X`.

        Returns
        -------
        float
            The R-squared score; see `ml_models.metrics.r2_score`.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        TypeError, ValueError
            See `predict()` and `ml_models.metrics.r2_score`.
        """
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

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
