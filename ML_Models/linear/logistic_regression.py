import os
import sys
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
from ML_Models.activations import sigmoid
from Optimization.first_order import Adam
from InformationTheory.information_theory import binary_cross_entropy


class LogisticRegression(MLModels):
    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X: Matrix, y: Vector, optimizer=None) -> None:
        self._validate_Xy(X, y)
        A = self._add_bias_column(X)
        d = A.n_cols
        self.w = Vector([0.0] * d)

        if optimizer is None:
            optimizer = Adam(lr=self.lr)

        for t in range(self.n_iter):
            y_pred = sigmoid(A * self.w)
            error = y_pred - y
            grad = (1.0 / A.n_rows) * (A.transpose() * error)

            params = [self.w]
            optimizer.step(params, [grad])
            self.w = params[0]

            if t % 100 == 0 or t == self.n_iter - 1:
                optimizer.record(binary_cross_entropy(list(y), list(y_pred)))

    def predict_proba(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        return sigmoid(self._add_bias_column(X) * self.w)

    def predict(self, X: Matrix, threshold: float = 0.5) -> Vector:
        self._check_is_fitted()
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        probs = self.predict_proba(X)
        return Vector([1.0 if p >= threshold else 0.0 for p in probs.components])

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


class LogisticRegressionL2(LogisticRegression):
    def __init__(self, lam: float = 1.0, lr: float = 0.01, n_iter: int = 1000):
        super().__init__(lr=lr, n_iter=n_iter)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.lam = lam

    def fit(self, X: Matrix, y: Vector, optimizer=None) -> None:
        self._validate_Xy(X, y)
        A = self._add_bias_column(X)
        d = A.n_cols
        self.w = Vector([0.0] * d)

        if optimizer is None:
            optimizer = Adam(lr=self.lr)

        inv_m = 1.0 / A.n_rows
        for t in range(self.n_iter):
            y_pred = sigmoid(A * self.w)
            error = y_pred - y
            grad_components = list((A.transpose() * error).components)
            for j in range(d):
                grad_components[j] *= inv_m
            # L2 penalty on non-bias weights
            for j in range(1, d):
                grad_components[j] += (self.lam * inv_m) * self.w[j]
            grad = Vector(grad_components)

            params = [self.w]
            optimizer.step(params, [grad])
            self.w = params[0]

            if t % 100 == 0 or t == self.n_iter - 1:
                l2_penalty = (self.lam * inv_m / 2.0) * sum(
                    self.w[j] ** 2 for j in range(1, d)
                )
                optimizer.record(
                    binary_cross_entropy(list(y), list(y_pred)) + l2_penalty
                )


class LogisticRegressionL1(LogisticRegression):
    def __init__(self, lam: float = 0.1, lr: float = 0.01, n_iter: int = 1000):
        super().__init__(lr=lr, n_iter=n_iter)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.lam = lam
        self.loss_history: list = []

    def fit(self, X: Matrix, y: Vector, optimizer=None) -> None:
        self._validate_Xy(X, y)
        A = self._add_bias_column(X)
        d = A.n_cols
        self.w = Vector([0.0] * d)
        self.loss_history = []

        inv_m = 1.0 / A.n_rows
        threshold = self.lr * self.lam * inv_m

        for t in range(self.n_iter):
            y_pred = sigmoid(A * self.w)
            error = y_pred - y
            grad = (A.transpose() * error)
            w_list = list(self.w.components)

            # Update bias (no regularisation)
            w_list[0] -= self.lr * grad[0] * inv_m

            # Proximal gradient step for regularised weights
            for j in range(1, d):
                step = w_list[j] - self.lr * grad[j] * inv_m
                w_list[j] = _soft_threshold(step, threshold)

            self.w = Vector(w_list)

            if t % 100 == 0 or t == self.n_iter - 1:
                l1_penalty = (self.lam * inv_m) * sum(
                    abs(self.w[j]) for j in range(1, d)
                )
                self.loss_history.append(
                    binary_cross_entropy(list(y), list(y_pred)) + l1_penalty
                )
