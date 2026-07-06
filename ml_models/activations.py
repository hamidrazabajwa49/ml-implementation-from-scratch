"""
Activation functions and their derivatives, shared across the
ml_models package (primarily neural network layers, but also usable
directly by any model that needs a squashing function, e.g. logistic
regression's sigmoid).

Every activation and derivative operates element-wise and accepts a
`float`/`int`, a `Vector`, or a `Matrix` from the math_foundations
library, dispatching recursively over each container's elements.

The `ACTIVATIONS` registry maps a string name to a
``(function, derivative, derivative_input)`` tuple, where
`derivative_input` is either ``"output"`` (the derivative is expressed
in terms of the activation's own output, e.g. sigmoid and tanh) or
``"preactivation"`` (the derivative is expressed in terms of the raw
input `z`, e.g. ReLU and leaky ReLU). This lets calling code (e.g. a
backpropagation loop) know which cached value to pass in without
inspecting each function individually.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, Optional, Tuple, Union

_script_dir = os.path.dirname(os.path.abspath(__file__))
_shared_parent = os.path.abspath(os.path.join(_script_dir, os.pardir))
_target_root = os.path.join(_shared_parent, "math_foundations")

if _target_root not in sys.path:
    sys.path.insert(0, _target_root)

from Vectors.vector import Vector  
from Matrix.matrix import Matrix  

Number = Union[int, float]
Tensor = Union[Number, Vector, Matrix]


def _is_number(x: object) -> bool:
    """Return True if `x` is an `int` or `float`, excluding `bool`."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_positive(value: Number, name: str) -> None:
    """Validate that `value` is a positive, finite, non-NaN real number.

    Raises
    ------
    TypeError
        If `value` is not a real number.
    ValueError
        If `value` is NaN, infinite, or non-positive.
    """
    if not _is_number(value):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    if math.isnan(value):
        raise ValueError(f"{name} must not be NaN.")
    if math.isinf(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _dispatch(z: Tensor, scalar_fn: Callable[[Number], Number], fn_name: str) -> Tensor:
    """Apply a scalar function element-wise to a float, Vector, or Matrix.

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The input to transform.
    scalar_fn : Callable[[Number], Number]
        A function operating on a single numeric value.
    fn_name : str
        Name of the calling activation function, used in error messages.

    Returns
    -------
    float, Vector, or Matrix
        The element-wise result, matching the type of `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    if _is_number(z):
        return scalar_fn(z)
    if isinstance(z, Vector):
        return Vector([scalar_fn(x) for x in z.components])
    if isinstance(z, Matrix):
        return Matrix([[scalar_fn(x) for x in row.components] for row in z.rows])
    raise TypeError(f"{fn_name} expects a float, int, Vector, or Matrix; got {type(z).__name__}.")


def _scalar_sigmoid(z: Number) -> float:
    """Numerically stable logistic sigmoid for a single scalar."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def sigmoid(z: Tensor) -> Tensor:
    """Compute the logistic sigmoid, element-wise.

    ``sigmoid(z) = 1 / (1 + exp(-z))``

    Implemented with the standard branch-on-sign trick to avoid
    overflow in `math.exp` for large-magnitude negative inputs.

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.

    Returns
    -------
    float, Vector, or Matrix
        Values in the open interval (0, 1), same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, _scalar_sigmoid, "sigmoid")


def sigmoid_derivative(a: Tensor) -> Tensor:
    """Compute the derivative of sigmoid given its own output.

    ``sigmoid'(z) = a * (1 - a)``, where ``a = sigmoid(z)``.

    Parameters
    ----------
    a : float, int, Vector, or Matrix
        The output of `sigmoid`, evaluated at the point of interest.

    Returns
    -------
    float, Vector, or Matrix
        The derivative values, same shape as `a`.

    Raises
    ------
    TypeError
        If `a` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(a, lambda x: x * (1.0 - x), "sigmoid_derivative")


def relu(z: Tensor) -> Tensor:
    """Compute the rectified linear unit, element-wise.

    ``relu(z) = max(0, z)``

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.

    Returns
    -------
    float, Vector, or Matrix
        Non-negative values, same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, lambda x: max(0.0, float(x)), "relu")


def relu_derivative(z: Tensor) -> Tensor:
    """Compute the derivative of ReLU given its pre-activation input.

    ``relu'(z) = 1 if z > 0 else 0``

    The (mathematically undefined) subgradient at exactly ``z == 0``
    is conventionally taken to be 0, matching common deep learning
    framework behavior.

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input (not the activation's output).

    Returns
    -------
    float, Vector, or Matrix
        Values in {0.0, 1.0}, same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, lambda x: 1.0 if x > 0.0 else 0.0, "relu_derivative")


def leaky_relu(z: Tensor, alpha: float = 0.01) -> Tensor:
    """Compute the leaky rectified linear unit, element-wise.

    ``leaky_relu(z) = z if z > 0 else alpha * z``

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.
    alpha : float, optional
        Slope applied to negative inputs; must be positive. Defaults
        to 0.01.

    Returns
    -------
    float, Vector, or Matrix
        Same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix, or `alpha` is
        not a real number.
    ValueError
        If `alpha` is not positive.
    """
    _check_positive(alpha, "alpha")
    return _dispatch(z, lambda x: float(x) if x > 0.0 else alpha * float(x), "leaky_relu")


def leaky_relu_derivative(z: Tensor, alpha: float = 0.01) -> Tensor:
    """Compute the derivative of leaky ReLU given its pre-activation input.

    ``leaky_relu'(z) = 1 if z > 0 else alpha``

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.
    alpha : float, optional
        Slope used for negative inputs during the forward pass; must
        match the value used in `leaky_relu`. Defaults to 0.01.

    Returns
    -------
    float, Vector, or Matrix
        Values in {alpha, 1.0}, same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix, or `alpha` is
        not a real number.
    ValueError
        If `alpha` is not positive.
    """
    _check_positive(alpha, "alpha")
    return _dispatch(z, lambda x: 1.0 if x > 0.0 else alpha, "leaky_relu_derivative")


def tanh_act(z: Tensor) -> Tensor:
    """Compute the hyperbolic tangent activation, element-wise.

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.

    Returns
    -------
    float, Vector, or Matrix
        Values in the open interval (-1, 1), same shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, math.tanh, "tanh_act")


def tanh_derivative(a: Tensor) -> Tensor:
    """Compute the derivative of tanh given its own output.

    ``tanh'(z) = 1 - a ** 2``, where ``a = tanh(z)``.

    Parameters
    ----------
    a : float, int, Vector, or Matrix
        The output of `tanh_act`, evaluated at the point of interest.

    Returns
    -------
    float, Vector, or Matrix
        The derivative values, same shape as `a`.

    Raises
    ------
    TypeError
        If `a` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(a, lambda x: 1.0 - x * x, "tanh_derivative")


def linear_act(z: Tensor) -> Tensor:
    """Compute the identity activation, element-wise.

    Provided for architectural uniformity (e.g. the output layer of a
    regression network), so every layer can be described by an entry
    in `ACTIVATIONS` rather than special-cased as "no activation".

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input.

    Returns
    -------
    float, Vector, or Matrix
        Numerically identical to `z` (cast to float), same shape.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, float, "linear_act")


def linear_derivative(z: Tensor) -> Tensor:
    """Compute the derivative of the identity activation.

    ``linear'(z) = 1`` everywhere.

    Parameters
    ----------
    z : float, int, Vector, or Matrix
        The pre-activation input (used only to determine the output
        shape).

    Returns
    -------
    float, Vector, or Matrix
        A value (or Vector/Matrix of values) of exactly 1.0, same
        shape as `z`.

    Raises
    ------
    TypeError
        If `z` is not a float, int, Vector, or Matrix.
    """
    return _dispatch(z, lambda _: 1.0, "linear_derivative")


def _softmax_row(values: list) -> list:
    """Compute a numerically stable softmax over a single list of scores.

    Subtracts the row maximum before exponentiating, which prevents
    `math.exp` overflow for large positive inputs without changing the
    result (softmax is shift-invariant).

    Raises
    ------
    ValueError
        If `values` is empty.
    """
    if not values:
        raise ValueError("softmax requires at least one value.")
    max_value = max(values)
    exponentials = [math.exp(x - max_value) for x in values]
    total = sum(exponentials)
    return [e / total for e in exponentials]


def softmax(z: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
    """Compute the softmax activation.

    ``softmax(z)_i = exp(z_i) / sum_j(exp(z_j))``

    Unlike the other activations, softmax is not element-wise: each
    output component depends on every component of the same row, so
    scalar (`float`/`int`) input is not supported.

    Parameters
    ----------
    z : Vector or Matrix
        For a `Vector`, softmax is applied across its components. For
        a `Matrix`, softmax is applied independently to each row
        (the typical convention for a batch of logit vectors).

    Returns
    -------
    Vector or Matrix
        Row-wise probability distributions summing to 1.0, same shape
        as `z`.

    Raises
    ------
    TypeError
        If `z` is not a `Vector` or `Matrix`.
    ValueError
        If `z` (or any row of `z`) is empty.
    """
    if isinstance(z, Vector):
        return Vector(_softmax_row(list(z.components)))
    if isinstance(z, Matrix):
        return Matrix([_softmax_row(list(row.components)) for row in z.rows])
    raise TypeError(f"softmax expects a Vector or Matrix; got {type(z).__name__}.")


''' 
Registry mapping an activation's name to its (function, derivative,derivative_input) tuple.
`derivative_input` indicates whether the derivative expects the activation's OUTPUT ("output") or its raw PRE-ACTIVATION input ("preactivation"). 
Softmax's derivative is a full Jacobian rather than an element-wise function, and in practice is fused directly into the cross-entropy loss gradient
(dL/dz = y_pred - y_true); it is therefore left as None here rather than implemented as a standalone element-wise derivative.
'''
ACTIVATIONS: Dict[str, Tuple[Callable[..., Tensor], Optional[Callable[..., Tensor]], str]] = {
    "sigmoid": (sigmoid, sigmoid_derivative, "output"),
    "relu": (relu, relu_derivative, "preactivation"),
    "leaky_relu": (leaky_relu, leaky_relu_derivative, "preactivation"),
    "tanh": (tanh_act, tanh_derivative, "output"),
    "linear": (linear_act, linear_derivative, "preactivation"),
    "softmax": (softmax, None, "output"),
}
