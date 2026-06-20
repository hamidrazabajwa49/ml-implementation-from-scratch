import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Vectors.vector import Vector


def linear_kernel(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    return a.dot(b)

def rbf_kernel(a: Vector, b: Vector, gamma: float = None) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must be non-empty")
    if gamma is None:
        gamma = 1.0 / len(a)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    diff = a - b
    sq_dist = diff.dot(diff)
    return math.exp(-gamma * sq_dist)

def polynomial_kernel(a: Vector, b: Vector, degree: int = 3, coef0: float = 1.0) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    if not isinstance(degree, int) or isinstance(degree, bool):
        raise TypeError(f"degree must be an int, got {type(degree).__name__} ({degree})")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    return (a.dot(b) + coef0) ** degree


def _resolve_kernel(kernel, gamma: float = None, degree: int = 3, coef0: float = 1.0):
    """
    Resolve a kernel spec into a callable (a: Vector, b: Vector) -> float.

    `kernel` may be the string 'linear' / 'rbf' / 'poly', or any callable
    with that same (Vector, Vector) -> float signature.
    """
    if callable(kernel):
        return kernel
    if not isinstance(kernel, str):
        raise TypeError(
            f"kernel must be a string ('linear', 'rbf', 'poly') or a callable, got {type(kernel).__name__}"
        )
    if kernel == 'linear':
        return lambda a, b: linear_kernel(a, b)
    if kernel == 'rbf':
        return lambda a, b: rbf_kernel(a, b, gamma)
    if kernel == 'poly':
        return lambda a, b: polynomial_kernel(a, b, degree, coef0)
    raise ValueError(f"Unknown kernel '{kernel}'. Choose from 'linear', 'rbf', 'poly', or pass a callable.")

