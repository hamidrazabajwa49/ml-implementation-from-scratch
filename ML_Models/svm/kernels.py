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
