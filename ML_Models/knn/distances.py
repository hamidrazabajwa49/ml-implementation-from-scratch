import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Vectors.vector import Vector


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    return (a - b).norm(2)


def manhattan(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    return (a - b).norm(1)


def minkowski(a: Vector, b: Vector, p: float) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have equal length, got {len(a)} and {len(b)}")
    if p <= 0:
        raise ValueError(f"Minkowski order p must be > 0, got {p}")
    return sum(abs(a[i] - b[i]) ** p for i in range(len(a))) ** (1.0 / p)


_METRIC_MAP = {
    'euclidean': euclidean,
    'manhattan': manhattan,
}


def _resolve_metric(metric):
    if callable(metric):
        return metric
    if isinstance(metric, str):
        if metric not in _METRIC_MAP:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from {list(_METRIC_MAP)} "
                f"or pass a callable."
            )
        return _METRIC_MAP[metric]
    raise TypeError(f"metric must be a str or callable, got {type(metric)}")
