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
