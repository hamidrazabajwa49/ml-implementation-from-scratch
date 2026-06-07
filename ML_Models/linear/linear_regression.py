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
from ML_Models.metrics import r2_score
from Optimization.first_order import Adam


def _regularized_inverse(M: Matrix, tol: float = 1e-10) -> Matrix:
    if abs(M.determinant()) < tol:
        I = Matrix.identity(M.n_rows)
        M = M + I * tol
    return M.inverse()


def _soft_threshold(z: float, alpha: float) -> float:
    if z > alpha:
        return z - alpha
    elif z < -alpha:
        return z + alpha
    return 0.0


class LinearRegression(MLModels):

    def fit(self, X: Matrix, y: Vector) -> None:
        self._validate_Xy(X, y)
        A = self._add_bias_column(X)
        At = A.transpose()
        AtA = At * A
        AtA_inv = _regularized_inverse(AtA)
        Aty = (At * y)
        self.w =(AtA_inv * Aty)

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        A = self._add_bias_column(X)
        return (A * self.w)

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


class GDLinearRegression(MLModels):

    def fit(self,X: Matrix,y: Vector,optimizer=None,n_iter: int = 1000,lr: float = 0.01,) -> None:
        self._validate_Xy(X, y)
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")

        A = self._add_bias_column(X)
        d = A.n_cols
        self.w = Vector([0.0] * d)

        if optimizer is None:
            optimizer = Adam(lr=lr)

        inv_m = 1.0 / A.n_rows
        for t in range(n_iter):
            y_pred = (A * self.w)
            error = y_pred - y
            grad = (2.0 * inv_m) * (A.transpose() * error)
            params = [self.w]
            optimizer.step(params, [grad])
            self.w = params[0]
            if t % 100 == 0 or t == n_iter - 1:
                loss = inv_m * error.dot(error)
                optimizer.record(loss)

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        A = self._add_bias_column(X)
        return (A * self.w)

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


class RidgeRegression(MLModels):

    def fit(self, X: Matrix, y: Vector, lam: float = 1.0) -> None:
        self._validate_Xy(X, y)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")

        A = self._add_bias_column(X)
        d = A.n_cols
        At = A.transpose()
        AtA = At * A

        L_data = [[lam if i == j and i > 0 else 0.0 for j in range(d)] for i in range(d)]
        L = Matrix(L_data)

        AtA_reg = AtA + L
        AtA_reg_inv = AtA_reg.inverse()
        Aty = (At * y)
        self.w = (AtA_reg_inv * Aty)

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        A = self._add_bias_column(X)
        return (A * self.w)

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


class LassoRegression(MLModels):

    def fit(self,X: Matrix,y: Vector,lam: float = 0.1,n_iters: int = 1000,tol: float = 1e-4,) -> None:
        self._validate_Xy(X, y)
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")

        A = self._add_bias_column(X)
        m, d = A.n_rows, A.n_cols
        w = [0.0] * d

        cols = A.columns()
        squared_norms = [cols[j].dot(cols[j]) for j in range(d)]

        for _ in range(n_iters):
            max_change = 0.0

            for j in range(1, d):
                if squared_norms[j] == 0.0:
                    continue
                y_pred = (A * Vector(w))
                r = Vector([y[i] - y_pred[i] for i in range(m)])
                rho = cols[j].dot(r) + squared_norms[j] * w[j]
                old_val = w[j]
                w[j] = _soft_threshold(rho, lam * m) / squared_norms[j]
                max_change = max(max_change, abs(w[j] - old_val))

            w[0] = sum(
                y[i] - sum(cols[j][i] * w[j] for j in range(1, d))
                for i in range(m)
            ) / m

            if max_change < tol:
                break

        self.w = Vector(w)

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        A = self._add_bias_column(X)
        return (A * self.w)

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        self._check_is_fitted()
        return [self.w]


