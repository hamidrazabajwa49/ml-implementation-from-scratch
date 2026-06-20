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


# KernelSVM — dual form, simplified SMO, supports RBF / poly / linear

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

    def fit(self, X: Matrix, y: Vector) -> None:
        self._validate_Xy(X, y)
        _check_pm1(y)

        m = X.n_rows
        if m < 2:
            raise ValueError(
                f"KernelSVM requires at least 2 samples to fit (SMO optimizes pairs "
                f"of alphas), got {m}."
            )
        Xs = [X.rows[i] for i in range(m)]
        ys = list(y)

        # _check_pm1 already restricted every label to +1/-1 (int or float,
        # which compare/hash equal in Python), so a plain set() over ys is
        # sufficient here -- no need to re-normalize the values.
        if len(set(ys)) < 2:
            raise ValueError(
                "y must contain both +1 and -1 labels; got samples from only one class."
            )

        # Precompute the full kernel (Gram) matrix once -- O(m^2) kernel calls
        K = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                val = self._kernel_fn(Xs[i], Xs[j])
                K[i][j] = val
                K[j][i] = val

        alpha = [0.0] * m
        b = 0.0
        rng = random.Random(self.random_state)

        def f(i: int) -> float:
            return sum(alpha[k] * ys[k] * K[k][i] for k in range(m)) + b

        passes = 0
        total_sweeps = 0
        while passes < self.max_passes:
            num_changed = 0
            total_sweeps += 1
            for i in range(m):
                Ei = f(i) - ys[i]
                if (ys[i] * Ei < -self.tol and alpha[i] < self.C) or \
                (ys[i] * Ei > self.tol and alpha[i] > 0.0):

                    j = i
                    while j == i:
                        j = rng.randint(0, m - 1)

                    Ej = f(j) - ys[j]
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if ys[i] != ys[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = 2.0 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0.0:
                        continue

                    alpha_j_new = alpha[j] - ys[j] * (Ei - Ej) / eta
                    alpha_j_new = max(L, min(H, alpha_j_new))

                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    alpha[j] = alpha_j_new
                    alpha[i] = alpha_i_old + ys[i] * ys[j] * (alpha_j_old - alpha[j])

                    b1 = (b - Ei
                        - ys[i] * (alpha[i] - alpha_i_old) * K[i][i]
                        - ys[j] * (alpha[j] - alpha_j_old) * K[i][j])
                    b2 = (b - Ej
                        - ys[i] * (alpha[i] - alpha_i_old) * K[i][j]
                        - ys[j] * (alpha[j] - alpha_j_old) * K[j][j])

                    if 0.0 < alpha[i] < self.C:
                        b = b1
                    elif 0.0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        # Keep only support vectors (alpha > tiny threshold) for fast inference
        sv_idx = [i for i in range(m) if alpha[i] > 1e-7]
        self.alphas = [alpha[i] for i in sv_idx]
        self.support_X = [Xs[i] for i in sv_idx]
        self.support_y = [ys[i] for i in sv_idx]
        self.b = b
        self.n_support = len(sv_idx)
        self.n_passes_run = total_sweeps

        # Kept for dual_objective(); not used by predict()/decision_function(),
        # which only need the (typically much smaller) support-vector subset.
        self._all_alpha = alpha
        self._all_ys = ys
        self._K = K

        self._fitted = True

        if self.n_support == 0:
            warnings.warn(
                "KernelSVM converged with 0 support vectors (every alpha stayed "
                "near 0). The model will predict a single constant class based "
                "only on the bias term. This can legitimately happen with a "
                "very small C or a kernel/gamma mismatched to the data scale -- "
                "consider increasing C or checking your kernel parameters.",
                stacklevel=2,
            )

    def dual_objective(self) -> float:
        """
        SMO's dual objective at the current solution:
            W(alpha) = sum(alpha_i) - 0.5 * sum_i sum_j alpha_i*alpha_j*y_i*y_j*K(x_i,x_j)
        Higher is better; useful for confirming the optimizer made progress
        (e.g. plotting W(alpha) across re-fits with different max_passes).
        """
        self._check_is_fitted()
        alpha = self._all_alpha
        ys = self._all_ys
        K = self._K
        m = len(alpha)
        linear_term = sum(alpha)
        quad_term = 0.0
        for i in range(m):
            if alpha[i] == 0.0:
                continue
            ai_yi = alpha[i] * ys[i]
            for j in range(m):
                if alpha[j] == 0.0:
                    continue
                quad_term += ai_yi * alpha[j] * ys[j] * K[i][j]
        return linear_term - 0.5 * quad_term

    def decision_function(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        scores = []
        for i in range(X.n_rows):
            x = X.rows[i]
            s = self.b
            for a, sv_x, sv_y in zip(self.alphas, self.support_X, self.support_y):
                s += a * sv_y * self._kernel_fn(sv_x, x)
            scores.append(s)
        return Vector(scores)

    def predict(self, X: Matrix) -> Vector:
        scores = self.decision_function(X)
        return Vector([1.0 if s >= 0.0 else -1.0 for s in scores])

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def parameters(self) -> dict:
        self._check_is_fitted()
        return {"alphas": self.alphas, "b": self.b, "n_support": self.n_support}

