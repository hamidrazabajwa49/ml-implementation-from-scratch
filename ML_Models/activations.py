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


def relu(z):
    if isinstance(z, (int, float)):
        return max(0.0, float(z))
    if isinstance(z, Vector):
        return Vector([max(0.0, float(x)) for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[max(0.0, float(x)) for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def leaky_relu(z, alpha: float = 0.01):
    if isinstance(z, (int, float)):
        return float(z) if z > 0 else alpha * float(z)
    if isinstance(z, Vector):
        return Vector([float(x) if x > 0 else alpha * float(x) for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[float(x) if x > 0 else alpha * float(x) for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def tanh_act(z):
    import math
    def _tanh(x):
        return math.tanh(x)
    if isinstance(z, (int, float)):
        return _tanh(z)
    if isinstance(z, Vector):
        return Vector([_tanh(x) for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[_tanh(x) for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def linear_act(z):
    if isinstance(z, (int, float)):
        return float(z)
    if isinstance(z, Vector):
        return Vector([float(x) for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[float(x) for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def softmax(z):
    import math
    def _softmax_vec(v):
        mx = max(v)
        exps = [math.exp(x - mx) for x in v]
        s = sum(exps)
        return [e / s for e in exps]
    if isinstance(z, Vector):
        return Vector(_softmax_vec(z.components))
    if isinstance(z, Matrix):
        return Matrix([_softmax_vec(row.components) for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def sigmoid_derivative(a):
    """Derivative of sigmoid given its output a = sigmoid(z)."""
    if isinstance(a, (int, float)):
        return a * (1.0 - a)
    if isinstance(a, Vector):
        return Vector([x * (1.0 - x) for x in a.components])
    if isinstance(a, Matrix):
        return Matrix([[x * (1.0 - x) for x in row.components] for row in a.rows])
    raise TypeError(f"Unsupported type: {type(a)}")


def relu_derivative(z):
    """Derivative of ReLU given pre-activation z."""
    if isinstance(z, (int, float)):
        return 1.0 if z > 0 else 0.0
    if isinstance(z, Vector):
        return Vector([1.0 if x > 0 else 0.0 for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[1.0 if x > 0 else 0.0 for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")


def leaky_relu_derivative(z, alpha: float = 0.01):
    if isinstance(z, (int, float)):
        return 1.0 if z > 0 else alpha
    if isinstance(z, Vector):
        return Vector([1.0 if x > 0 else alpha for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[1.0 if x > 0 else alpha for x in row.components] for row in z.rows])
    raise TypeError(f"Unsupported type: {type(z)}")





ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative, "output"),
    "relu": (relu, relu_derivative, "preactivation"),
    "leaky_relu": (leaky_relu, leaky_relu_derivative, "preactivation"),
    "tanh": (tanh_act, tanh_derivative, "output"),
    "linear": (linear_act, linear_derivative, "preactivation"),
    "softmax": (softmax, None, "output"),
}
