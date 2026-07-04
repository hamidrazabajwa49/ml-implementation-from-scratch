"""
vector.py
=========

A lightweight, dependency-free N-dimensional vector library implementing
standard linear-algebra vector operations (arithmetic, dot/cross product,
norms, normalization, angle, projection).

This module intentionally avoids a hard NumPy dependency so it can serve as
a "from scratch" foundation piece. An optional ``to_numpy()`` bridge is
provided for interoperability and for regression-testing against NumPy.

Design notes
------------
- ``Vector`` is a thin, immutable-by-convention wrapper around a Python
  ``list`` of ``int``/``float`` components. In-place mutation via
  ``__setitem__`` is still supported for performance-sensitive callers.
- ``bool`` is explicitly rejected as a numeric component even though
  ``bool`` is a subclass of ``int`` in Python -- ``True``/``False`` are not
  meaningful vector components and silently accepting them is a common
  source of bugs.
- All zero-division-adjacent operations (``normalize``, ``angle``,
  ``projection_onto``, ``__truediv__``) accept a configurable ``tol``
  parameter so callers can decide what "effectively zero" means for their
  application (defaults to an exact ``0.0`` comparison for
  backward-compatible, unsurprising behavior).

Example
-------
>>> v = Vector([3, 4])
>>> v.norm()
5.0
>>> (v + Vector([1, 1])).components
[4, 5]
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterable, List, Union

Number = Union[int, float]

logger = logging.getLogger(__name__)


def _is_number(x: object) -> bool:
    """Return True if ``x`` is an ``int`` or ``float`` but not a ``bool``.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is ``True``. Vector components should never silently accept booleans,
    since that almost always indicates a bug upstream (e.g. a comparison
    result leaking into numeric data).

    Parameters
    ----------
    x : object
        The value to check.

    Returns
    -------
    bool
        True if ``x`` is a genuine numeric scalar.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class VectorError(Exception):
    """Base class for all Vector-related errors raised by this module."""


class DimensionMismatchError(VectorError, ValueError):
    """Raised when two vectors have incompatible lengths for an operation."""


class ZeroVectorError(VectorError, ValueError):
    """Raised when an operation (e.g. normalize) is undefined for a zero vector."""


class Vector:
    """An N-dimensional mathematical vector backed by a Python list.

    Parameters
    ----------
    components : Iterable[Number]
        An iterable of ``int``/``float`` values. Booleans are rejected.

    Raises
    ------
    TypeError
        If ``components`` is not iterable, or contains a non-numeric
        (including boolean) element.

    Examples
    --------
    >>> Vector([1, 2, 3])
    Vector([1, 2, 3])
    >>> Vector([])  # the zero-dimensional / empty vector is legal
    Vector([])
    """

    __slots__ = ("components",)

    def __init__(self, components: Iterable[Number]) -> None:
        if isinstance(components, str) or not hasattr(components, "__iter__"):
            raise TypeError(
                f"components must be an iterable of numbers, got {type(components).__name__}"
            )
        components = list(components)
        for i, x in enumerate(components):
            if not _is_number(x):
                raise TypeError(
                    f"component at index {i} must be int or float, got {type(x).__name__}"
                )
        self.components: List[Number] = components
        logger.debug("Created Vector with %d components", len(components))


    # Dunder / container protocol
    def __repr__(self) -> str:
        return f"Vector({self.components})"

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, index: Union[int, slice]) -> Union[Number, "Vector"]:
        if isinstance(index, slice):
            return Vector(self.components[index])
        return self.components[index]

    def __setitem__(self, index: int, value: Number) -> None:
        if not _is_number(value):
            raise TypeError(f"component must be int or float, got {type(value).__name__}")
        self.components[index] = value

    def __iter__(self):
        return iter(self.components)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.components == other.components

    def __hash__(self):
        # Vector is mutable (supports __setitem__), so it is intentionally
        # unhashable - matching Python's own list semantics.
        return None  

    def __neg__(self) -> "Vector":
        return Vector([-x for x in self.components])

    def __abs__(self) -> float:
        return self.norm()

    def copy(self) -> "Vector":
        """Return a shallow copy of this vector."""
        return Vector(list(self.components))


    # Functional helpers
    def element_wise(self, func: Callable[[Number], Number]) -> "Vector":
        """Apply ``func`` to every component and return a new Vector.

        Parameters
        ----------
        func : Callable[[Number], Number]
            A unary function applied element-wise.

        Returns
        -------
        Vector
        """
        return Vector([func(x) for x in self.components])

    def element_wise_with(self, other: "Vector", func: Callable[[Number, Number], Number]) -> "Vector":
        """Combine two same-length vectors element-wise with ``func``.

        Parameters
        ----------
        other : Vector
        func : Callable[[Number, Number], Number]

        Returns
        -------
        Vector

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        DimensionMismatchError
            If the two vectors have different lengths.
        """
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        if len(self) != len(other):
            raise DimensionMismatchError(
                f"dimension mismatch: {len(self)} vs {len(other)}"
            )
        return Vector([func(a, b) for a, b in zip(self.components, other.components)])


    # Arithmetic operators

    def __add__(self, other: Union[Number, "Vector"]) -> "Vector":
        if _is_number(other):
            return Vector([x + other for x in self.components])
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise DimensionMismatchError(
                    f"dimension mismatch: {len(self)} vs {len(other)}"
                )
            return Vector([x + y for x, y in zip(self.components, other.components)])
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Union[Number, "Vector"]) -> "Vector":
        if _is_number(other):
            return Vector([x - other for x in self.components])
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise DimensionMismatchError(
                    f"dimension mismatch: {len(self)} vs {len(other)}"
                )
            return Vector([x - y for x, y in zip(self.components, other.components)])
        return NotImplemented

    def __rsub__(self, other: Union[Number, "Vector"]) -> "Vector":
        if _is_number(other):
            return Vector([other - x for x in self.components])
        if isinstance(other, Vector):
            return other.__sub__(self)
        return NotImplemented

    def __mul__(self, num: Number) -> "Vector":
        if not _is_number(num):
            raise TypeError(
                f"unsupported operand type for *: 'Vector' and '{type(num).__name__}'. "
                "Use .dot() for dot product or .cross() for cross product."
            )
        return Vector([num * x for x in self.components])

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number, tol: float = 0.0) -> "Vector":
        if not _is_number(scalar):
            raise TypeError(
                f"unsupported operand type for /: 'Vector' and '{type(scalar).__name__}'"
            )
        if math.isnan(scalar):
            raise ValueError("cannot divide a vector by NaN")
        if abs(scalar) <= tol:
            raise ZeroDivisionError("division by zero (or value within tolerance of zero)")
        return Vector([x / scalar for x in self.components])

    # Core linear algebra

    def dot(self, other: "Vector") -> Number:
        """Compute the dot (inner) product with another vector.

        Parameters
        ----------
        other : Vector

        Returns
        -------
        Number
            0 for two empty vectors, by convention (the identity of sum).

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        DimensionMismatchError
            If lengths differ.
        """
        if not isinstance(other, Vector):
            raise TypeError(f"dot product requires a Vector, got {type(other).__name__}")
        if len(self) != len(other):
            raise DimensionMismatchError(f"dimension mismatch: {len(self)} vs {len(other)}")
        return sum(x * y for x, y in zip(self.components, other.components))

    def cross(self, other: "Vector") -> "Vector":
        """Compute the 3D cross product with another 3-component vector.

        Parameters
        ----------
        other : Vector
            Must have exactly 3 components, as must ``self``.

        Returns
        -------
        Vector
            A new 3-component vector orthogonal to both inputs.

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        ValueError
            If either vector does not have exactly 3 components (the cross
            product is only defined in 3D, and the less common 7D case is
            deliberately out of scope for this "from scratch" library).
        """
        if not isinstance(other, Vector):
            raise TypeError(f"cross product requires a Vector, got {type(other).__name__}")
        if len(self) != 3 or len(other) != 3:
            raise ValueError(
                f"cross product is only defined for 3D vectors, got lengths "
                f"{len(self)} and {len(other)}"
            )
        a1, a2, a3 = self.components
        b1, b2, b3 = other.components
        return Vector(
            [
                a2 * b3 - a3 * b2,
                a3 * b1 - a1 * b3,
                a1 * b2 - a2 * b1,
            ]
        )

    def norm(self, order: Union[int, float] = 2) -> float:
        """Compute the p-norm of the vector.

        Parameters
        ----------
        order : int or float, optional
            The norm order ``p``. Supports ``1`` (Manhattan), ``2``
            (Euclidean, default), ``math.inf`` (Chebyshev / max-abs), and
            any other positive real number (general p-norm).

        Returns
        -------
        float
            ``0.0`` for an empty vector, by convention.

        Raises
        ------
        ValueError
            If ``order`` is not a positive number or ``math.inf``.
        """
        if len(self.components) == 0:
            return 0.0
        if order == 1:
            return float(sum(abs(x) for x in self.components))
        if order == 2:
            return math.sqrt(float(self.dot(self)))
        if order == math.inf:
            return float(max(abs(x) for x in self.components))
        if _is_number(order) and order > 0:
            return float(sum(abs(x) ** order for x in self.components) ** (1.0 / order))
        raise ValueError(f"norm order must be a positive number or math.inf, got {order!r}")

    def normalize(self, tol: float = 0.0) -> "Vector":
        """Return a unit-length copy of this vector.

        Parameters
        ----------
        tol : float, optional
            Norms with absolute value ``<= tol`` are treated as zero.
            Defaults to an exact comparison for backward compatibility.

        Raises
        ------
        ZeroVectorError
            If the vector's norm is (within ``tol`` of) zero.
        """
        n = self.norm()
        if math.isnan(n) or abs(n) <= tol:
            raise ZeroVectorError("cannot normalize a zero (or near-zero) vector")
        return Vector([x / n for x in self.components])

    def is_zero(self, tol: float = 0.0) -> bool:
        """Return True if every component is within ``tol`` of zero."""
        return all(abs(x) <= tol for x in self.components)

    def angle(self, other: "Vector", tol: float = 0.0, degrees: bool = True) -> float:
        """Compute the angle between this vector and ``other``.

        Parameters
        ----------
        other : Vector
        tol : float, optional
            Threshold below which either vector's norm is treated as zero.
        degrees : bool, optional
            Return degrees (default) or radians.

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        ZeroVectorError
            If either vector has (near-)zero norm.
        """
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        denom = self.norm() * other.norm()
        if abs(denom) <= tol:
            raise ZeroVectorError("cannot compute angle with a zero (or near-zero) vector")
        cos = self.dot(other) / denom
        cos = max(-1.0, min(1.0, cos))  # guard against float drift outside [-1, 1]
        radians = math.acos(cos)
        return math.degrees(radians) if degrees else radians

    def projection_onto(self, other: "Vector", tol: float = 0.0) -> "Vector":
        """Compute the vector projection of ``self`` onto ``other``.

        Parameters
        ----------
        other : Vector
        tol : float, optional
            Threshold below which ``other``'s squared norm is treated as zero.

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        ZeroVectorError
            If ``other`` is a (near-)zero vector.
        """
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        n_sq = other.dot(other)
        if abs(n_sq) <= tol:
            raise ZeroVectorError("cannot project onto a zero (or near-zero) vector")
        return (self.dot(other) / n_sq) * other

    def distance_to(self, other: "Vector", order: Union[int, float] = 2) -> float:
        """Compute the p-distance ``||self - other||_order``.

        Parameters
        ----------
        other : Vector
        order : int or float, optional
            Norm order, see :meth:`norm`.

        Raises
        ------
        TypeError
            If ``other`` is not a Vector.
        DimensionMismatchError
            If lengths differ.
        """
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        if len(self) != len(other):
            raise DimensionMismatchError(f"dimension mismatch: {len(self)} vs {len(other)}")
        return (self - other).norm(order)


    # Interop / convenience constructors

    def to_list(self) -> List[Number]:
        """Return a plain Python list copy of the components."""
        return list(self.components)

    def to_numpy(self):
        """Return a NumPy array view of this vector.

        Raises
        ------
        ImportError
            If NumPy is not installed in the current environment.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "NumPy is required for to_numpy() but is not installed"
            ) from exc
        return np.array(self.components, dtype=float)

    @classmethod
    def zeros(cls, n: int) -> "Vector":
        """Create an n-dimensional zero vector.

        Raises
        ------
        ValueError
            If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return cls([0.0] * n)

    @classmethod
    def from_list(cls, values: List[Number]) -> "Vector":
        """Create a Vector from a plain Python list (alias for the constructor)."""
        return cls(values)
