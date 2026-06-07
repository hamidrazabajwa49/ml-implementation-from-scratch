import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
main_folder = os.path.abspath(os.path.join(current_dir, '..'))
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from Vectors.vector import Vector
from Matrix.matrix import Matrix


def sigmoid(z):
    # Works on float, Vector, or Matrix (element‑wise).

    if isinstance(z, (int, float)):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1.0 + exp_z)

    if isinstance(z, Vector):
        return Vector([sigmoid(x) for x in z.components])

    if isinstance(z, Matrix):
        new_rows = []
        for row in z.rows:
            new_rows.append(Vector([sigmoid(x) for x in row.components]))
        return Matrix([v.components for v in new_rows])

    raise TypeError(f"Unsupported type: {type(z)}")
