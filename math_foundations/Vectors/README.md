# Vector — Pure Python Vector Library

## Overview

`vector.py` is a pure Python implementation of an n-dimensional mathematical vector, built from scratch as **Project 1** of the *Engineering Redemption Arc* — a structured 60-project ML engineering curriculum covering mathematical foundations through production machine learning systems.

The module provides a `Vector` class with full arithmetic operator support, linear algebra operations, and robust input validation. It serves as a reusable primitive for downstream projects in the curriculum (matrix operations, eigenvalue solvers, SVD, optimization, and ML model implementations).

No third-party libraries are used. The only dependency is Python's standard `math` module.

---

## Project Structure

```
Vector/
└── vector.py       # Vector class — all logic lives here
```

The class is self-contained. There is no separate test file, config, or build step required.

---

## Dependencies

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| `math` (stdlib) | any |

No `pip install` required.

---

## Installation

Clone or copy `vector.py` into your project directory:

```bash
cp vector.py /your/project/
```

Then import it:

```python
from Vector.vector import Vector
```

---

## Usage

### Construction

```python
v = Vector([1, 2, 3])
u = Vector([4.0, 5.0, 6.0])
```

Components must be a finite iterable of `int` or `float` values. Any other type raises `TypeError`.

---

### Arithmetic

All standard arithmetic operators are supported between two `Vector` objects of equal dimension, or between a `Vector` and a scalar.

```python
v + u          # element-wise addition
v - u          # element-wise subtraction
v * 3          # scalar multiplication
3 * v          # scalar multiplication (reversed)
v / 2.0        # scalar division
```

Dimension mismatches raise `ValueError`. Type mismatches raise `TypeError`. Division by zero raises `ZeroDivisionError`.

> **Note:** `*` between two Vectors is intentionally unsupported. Use `.dot()` instead.

---

### Dot Product

```python
result = v.dot(u)    # returns a float
```

---

### Norms

```python
v.norm()          # Euclidean (L2) norm — default
v.norm(order=1)   # Manhattan (L1) norm
v.norm(order=math.inf)   # Chebyshev (max) norm
v.norm(order=3)   # arbitrary Lp norm
```

---

### Normalization

```python
unit = v.normalize()   # returns a new unit Vector
```

Raises `ValueError` on a zero vector.

---

### Angle Between Vectors

```python
degrees = v.angle(u)   # returns angle in degrees
```

Raises `ValueError` if either vector is zero.

---

### Projection

```python
proj = v.projection_onto(u)   # projection of v onto u, returns a Vector
```

Raises `ValueError` if `u` is the zero vector.

---

### Utility Methods

```python
len(v)            # number of components
v[0]              # index access
v[0] = 9.0        # index assignment (validates type)
list(v)           # iterate over components
v == u            # equality check (component-wise)
repr(v)           # Vector([1, 2, 3])
```

#### Element-wise transformations

```python
v.element_wise(lambda x: x ** 2)          # apply a unary function to each component
v.element_wise_with(u, lambda a, b: a*b)  # apply a binary function component-wise
```

---

## Example Session

```python
import math
from vector import Vector

a = Vector([3, 4])
b = Vector([1, 0])

print(a.norm())            # 5.0
print(a.normalize())       # Vector([0.6, 0.8])
print(a.dot(b))            # 3
print(a.angle(b))          # 53.13010235415598
print(a.projection_onto(b))  # Vector([3.0, 0.0])
print(a + b)               # Vector([4, 4])
print(2 * a)               # Vector([6, 8])
```

---

## Error Reference

| Situation | Exception |
|---|---|
| Non-numeric component in constructor | `TypeError` |
| Non-iterable passed to constructor | `TypeError` |
| Dimension mismatch in binary ops | `ValueError` |
| `*` used between two Vectors | `TypeError` (with hint to use `.dot()`) |
| Scalar division by zero | `ZeroDivisionError` |
| Normalizing a zero vector | `ValueError` |
| Angle with a zero vector | `ValueError` |
| Projection onto zero vector | `ValueError` |
| Invalid norm order | `ValueError` |

---

## Design Notes

- **Immutability of results:** All operations return new `Vector` instances; the original is never modified (except `__setitem__`).
- **No NumPy:** Implementation is intentionally from scratch to build foundational understanding before transitioning to library-based workflows in later curriculum phases.
- **Operator clarity:** Multiplication between two `Vector` objects is blocked with a descriptive error to prevent silent dot-product misuse — a common source of bugs in naive implementations.
- **Norm generality:** The `norm()` method handles L1, L2, L∞, and arbitrary Lp norms in a single unified interface.

---

## Roadmap Context

This module is **Project 1 of 60** in the Engineering Redemption Arc curriculum. It underpins:

- **Project 2** — Matrix operations (dot products, projections reused directly)
- **Projects 3–4** — Eigenvalues, SVD (Vector as the base data structure)
- **Projects 11+** — ML model implementations (gradient vectors, weight updates)

The implementation intentionally avoids NumPy to enforce understanding of the underlying mechanics before those abstractions are introduced in later phases.
