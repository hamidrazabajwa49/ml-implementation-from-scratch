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


class GDLinearRegression(MLModel):
    """Linear regression fit via iterative gradient-based optimization.

    Minimizes the mean squared error

    ``L(w) = (1 / n_samples) * ||y - Xw||^2``

    using any optimizer conforming to the `Optimization.base.Optimizer`
    interface, applied to the gradient

    ``grad(L) = (2 / n_samples) * X^T (Xw - y)``

    Defaults to `Adam`, which converges reliably across a wide range of
    learning rates without manual tuning.

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

    def fit(
        self,
        X: Matrix,
        y: Vector,
        optimizer: Optional[Optimizer] = None,
        n_iter: int = 1000,
        lr: float = 0.01,
    ) -> "GDLinearRegression":
        """Fit the model via iterative gradient-based optimization.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector of length n_samples.
        optimizer : Optimizer, optional
            An instance conforming to `Optimization.base.Optimizer`
            (e.g. `Adam`, `SGD`, `RMSProp`). If omitted, a fresh `Adam`
            optimizer with learning rate `lr` is constructed. `lr` is
            ignored if `optimizer` is provided.
        n_iter : int, optional
            Number of gradient-descent iterations; must be at least 1.
        lr : float, optional
            Learning rate used only when `optimizer` is None; must be
            positive.

        Returns
        -------
        GDLinearRegression
            The fitted instance, for method chaining.

        Raises
        ------
        TypeError
            If `X`/`y` have the wrong type, or `optimizer` is provided
            but is not an `Optimizer` instance.
        ValueError
            If `X`/`y` are empty or mismatched, `n_iter` is less than
            1, or `lr` is not positive.
        """
        self._validate_Xy(X, y)
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}"
            )

        self.n_features_in_ = X.n_cols
        design = self._add_bias_column(X) if self.fit_intercept else X
        n_samples = design.n_rows
        self.w: Vector = Vector([0.0] * design.n_cols)

        if optimizer is None:
            optimizer = Adam(lr=lr)

        inv_n = 1.0 / n_samples
        for iteration in range(n_iter):
            residual = (design * self.w) - y
            gradient = (2.0 * inv_n) * (design.transpose() * residual)
            params = [self.w]
            optimizer.step(params, [gradient])
            self.w = params[0]
            if iteration % 100 == 0 or iteration == n_iter - 1:
                optimizer.record(inv_n * residual.dot(residual))

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


class RidgeRegression(MLModel):
    """L2-regularized (ridge) linear regression, solved in closed form.

    Fits ``y = X @ w`` by minimizing the L2-penalized residual sum of
    squares:

    ``||y - Xw||^2 + lam * ||w_features||^2``

    where the intercept term (if `fit_intercept` is True) is excluded
    from the penalty, following the standard ridge regression
    convention. The closed-form solution is:

    ``w = (X^T X + lam * L)^-1 X^T y``

    where `L` is a diagonal matrix with `lam` on every feature
    coefficient's diagonal entry and 0 at the intercept's position.

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

    def fit(self, X: Matrix, y: Vector, lam: float = 1.0) -> "RidgeRegression":
        """Fit the ridge regression model.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector of length n_samples.
        lam : float, optional
            L2 regularization strength; must be non-negative. A value
            of 0 reduces to ordinary least squares.

        Returns
        -------
        RidgeRegression
            The fitted instance, for method chaining.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix` or `y` is not a `Vector`.
        ValueError
            If `X`/`y` are empty or mismatched, or `lam` is negative.
        SingularMatrixError
            If the regularized Gram matrix remains singular even after
            the ridge-corrected fallback in `_safe_inverse`.
        """
        self._validate_Xy(X, y)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")

        self.n_features_in_ = X.n_cols
        design = self._add_bias_column(X) if self.fit_intercept else X
        n_features = design.n_cols
        design_t = design.transpose()
        gram = design_t * design

        penalty_start = 1 if self.fit_intercept else 0
        penalty_data = [
            [lam if (i == j and i >= penalty_start) else 0.0 for j in range(n_features)]
            for i in range(n_features)
        ]
        penalty = Matrix(penalty_data)

        gram_inv = _safe_inverse(gram + penalty)
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
            ``w[0]`` is the (unpenalized) intercept and the remaining
            entries are the L2-penalized feature coefficients.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        """
        self._check_is_fitted()
        return [self.w]


class LassoRegression(MLModel):
    """L1-regularized (lasso) linear regression, solved via coordinate descent.

    Fits ``y = X @ w`` by minimizing the L1-penalized mean squared
    error:

    ``(1 / (2 * n_samples)) * ||y - Xw||^2 + lam * ||w_features||_1``

    using cyclic coordinate descent with soft-thresholding (Friedman,
    Hastie & Tibshirani, 2010). At each coordinate, the partial
    residual is formed, the coordinate-wise least-squares solution is
    shrunk toward zero by `lam * n_samples`, and the residual is
    updated incrementally in O(n_samples) rather than by recomputing
    the full prediction, keeping each full pass O(n_samples *
    n_features). The intercept term (if `fit_intercept` is True) is
    excluded from the penalty and updated with an ordinary
    (unregularized) least-squares coordinate step, following the
    standard lasso convention.

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

    def fit(
        self,
        X: Matrix,
        y: Vector,
        lam: float = 0.1,
        n_iters: int = 1000,
        tol: float = 1e-4,
    ) -> "LassoRegression":
        """Fit the lasso regression model via coordinate descent.

        Parameters
        ----------
        X : Matrix
            Feature matrix of shape (n_samples, n_features).
        y : Vector
            Target vector of length n_samples.
        lam : float, optional
            L1 regularization strength; must be non-negative. A value
            of 0 reduces to (coordinate-descent) ordinary least
            squares.
        n_iters : int, optional
            Maximum number of full coordinate-descent passes; must be
            at least 1.
        tol : float, optional
            Convergence threshold on the largest per-pass coefficient
            change; must be positive.

        Returns
        -------
        LassoRegression
            The fitted instance, for method chaining.

        Raises
        ------
        TypeError
            If `X` is not a `Matrix` or `y` is not a `Vector`.
        ValueError
            If `X`/`y` are empty or mismatched, or `lam`/`n_iters`/`tol`
            are out of their valid ranges.
        """
        self._validate_Xy(X, y)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")

        self.n_features_in_ = X.n_cols
        design = self._add_bias_column(X) if self.fit_intercept else X
        n_samples, n_features = design.n_rows, design.n_cols

        columns = design.columns()
        squared_norms = [column.dot(column) for column in columns]
        weights = [0.0] * n_features
        residual = list(y)
        bias_index = 0 if self.fit_intercept else None
        threshold = lam * n_samples

        for _ in range(n_iters):
            max_change = 0.0
            for j in range(n_features):
                if squared_norms[j] == 0.0:
                    continue
                column = columns[j].components
                rho = (
                    sum(column[i] * residual[i] for i in range(n_samples))
                    + squared_norms[j] * weights[j]
                )
                if j == bias_index:
                    new_weight = rho / squared_norms[j]
                else:
                    new_weight = _soft_threshold(rho, threshold) / squared_norms[j]

                delta = new_weight - weights[j]
                if delta != 0.0:
                    residual = [residual[i] - delta * column[i] for i in range(n_samples)]
                max_change = max(max_change, abs(delta))
                weights[j] = new_weight

            if max_change < tol:
                break

        self.w: Vector = Vector(weights)
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
            ``w[0]`` is the (unpenalized) intercept and the remaining
            entries are the L1-penalized feature coefficients, many of
            which may be exactly 0.0.

        Raises
        ------
        NotFittedError
            If called before `fit()`.
        """
        self._check_is_fitted()
        return [self.w]
