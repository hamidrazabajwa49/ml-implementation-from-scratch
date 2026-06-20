import os
import sys
import random
import warnings
from typing import List, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
main_folder = os.path.abspath(os.path.join(parent_dir, '..'))
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.base_class import MLModels
from ML_Models.metrics import accuracy
from Optimization.first_order import Adam
from ML_Models.svm.kernels import _resolve_kernel


def _check_pm1(y: Vector, name: str = "y") -> None:
    for i, v in enumerate(y):
        if v not in (1, -1, 1.0, -1.0):
            raise ValueError(
                f"{name} must contain only +1/-1 labels for SVM, got {v} at index {i}. "
                f"Convert 0/1 labels via: y_pm1 = [1 if v == 1 else -1 for v in y]"
            )


def _check_positive_number(value, name: str) -> None:
    """Reject bools (e.g. C=True) and non-positive numbers for a numeric hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__} ({value!r})")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_positive_int(value, name: str) -> None:
    """Reject bools and require a positive int for count-like hyperparameters."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__} ({value!r})")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")



# LinearSVM — primal, subgradient descent, explicit hinge loss

class LinearSVM(MLModels):
    """
    Soft-margin linear SVM trained by minimizing:
        (1/2)||w||^2 + C * (1/m) * sum_i max(0, 1 - y_i * (w.x_i + b))
    via subgradient descent. Large C -> hard margin behavior.
    """

    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iter: int = 1000):
        _check_positive_number(C, "C")
        _check_positive_number(lr, "lr")
        _check_positive_int(n_iter, "n_iter")
        self.C = C
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X: Matrix, y: Vector, optimizer=None) -> None:
        self._validate_Xy(X, y)
        _check_pm1(y)

        A = self._add_bias_column(X)        # bias prepended at index 0
        m, d = A.n_rows, A.n_cols
        self.w = Vector([0.0] * d)
        self.loss_history: list = []

        if optimizer is None:
            optimizer = Adam(lr=self.lr)
        elif getattr(optimizer, "t", 0):
            warnings.warn(
                "Optimizer instance passed to fit() already has accumulated "
                "state from a previous fit() call; resetting it so this fit() "
                "starts from a clean state. Pass a fresh optimizer instance "
                "per fit() call to avoid this warning.",
                stacklevel=2,
            )
            optimizer.t = 0
            optimizer.m = None
            optimizer.v = None

        for t in range(self.n_iter):
            margins = [y[i] * A.rows[i].dot(self.w) for i in range(m)]
            violators = [i for i in range(m) if margins[i] < 1.0]

            # Hinge gradient
            grad_components = [0.0] * d
            for i in violators:
                row = A.rows[i].components
                yi = y[i]
                for j in range(d):
                    grad_components[j] -= (self.C / m) * yi * row[j]

            # L2 regularization gradient, bias (index 0) excluded
            for j in range(1, d):
                grad_components[j] += self.w[j]

            grad = Vector(grad_components)
            params = [self.w]
            optimizer.step(params, [grad])
            self.w = params[0]

            if t % 100 == 0 or t == self.n_iter - 1:
                reg_term = 0.5 * sum(self.w[j] ** 2 for j in range(1, d))
                hinge_term = (self.C / m) * sum(max(0.0, 1.0 - margins[i]) for i in range(m))
                optimizer.record(reg_term + hinge_term)
                self.loss_history.append(reg_term + hinge_term)

    def decision_function(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        A = self._add_bias_column(X)
        return A * self.w

    def predict(self, X: Matrix) -> Vector:
        scores = self.decision_function(X)
        return Vector([1.0 if s >= 0.0 else -1.0 for s in scores])

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


# 2. KernelSVM — dual form, simplified SMO, supports RBF / poly / linear

class KernelSVM(MLModels):
    """
    Soft-margin kernel SVM solved via simplified SMO (Sequential Minimal
    Optimization). No QP library needed: optimize one pair (alpha_i, alpha_j)
    at a time, each pair has a closed-form solution.
    """

    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: float = None,
                degree: int = 3, coef0: float = 1.0,
                tol: float = 1e-3, max_passes: int = 10,
                random_state: int = None):
        _check_positive_number(C, "C")
        _check_positive_number(tol, "tol")
        _check_positive_int(max_passes, "max_passes")
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, int)):
            raise TypeError(f"random_state must be an int or None, got {type(random_state).__name__}")
        if isinstance(kernel, str) and kernel == 'rbf' and gamma is not None and gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if isinstance(kernel, str) and kernel == 'poly':
            if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
                raise ValueError(f"degree must be an int >= 1, got {degree}")
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state
        self._kernel_fn = _resolve_kernel(kernel, gamma, degree, coef0)
        self._fitted = False

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() before predict() or score()."
            )
