# Vector — N-Dimensional Vector Library (From Scratch)

Part of the [`ml-implementation-from-scratch`](../../) curriculum —
**Phase 1: Math Foundations**.

A dependency-free, pure-Python implementation of an N-dimensional
mathematical vector, covering the arithmetic, linear-algebra, and geometric
operations that underpin most ML algorithms (gradient descent, similarity
metrics, PCA, embeddings, etc.). NumPy is used only in the test suite, as a
correctness oracle — never by the library itself.

---

## Overview

`Vector` wraps a plain Python `list` of `int`/`float` components and
implements:

- Full container protocol (`len`, indexing, slicing, iteration, equality)
- Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`) with scalar and
  vector operands
- Core linear algebra: dot product, 3D cross product, p-norms
  (L1, L2, L∞, general p)
- Geometric operations: normalization, angle between vectors, vector
  projection, Euclidean/p-distance
- Functional helpers (`element_wise`, `element_wise_with`) for composing
  custom transformations
- Interop helpers (`to_list`, `to_numpy`) and convenience constructors
  (`zeros`, `from_list`)

Design choices are documented inline in the module docstring and enforced
by the test suite — see [Design Notes](#design-notes) below.

---

## Project Structure

```
math_foundations/
└── Vectors/
    ├── vector.py              # Vector implementation (this module)
    ├── tests/
    │   └── test_vector.py     # Pytest suite (100+ cases, NumPy cross-checked)
    └── README.md
```

---

## Dependencies

| Component        | Requires                          |
|-------------------|-----------------------------------|
| `vector.py`        | Python 3.8+ standard library only (`math`, `logging`, `typing`) |
| `tests/test_vector.py` | `pytest`, `numpy` (test-only, for regression checks) |

Install test dependencies:

```bash
pip install pytest numpy
```

Run the suite from `math_foundations/Vectors/`:

```bash
pytest tests/test_vector.py -v
```

---

## Module Reference

### Exceptions

| Exception | Base | Raised when |
|---|---|---|
| `VectorError` | `Exception` | Base class for all vector errors |
| `DimensionMismatchError` | `VectorError`, `ValueError` | Two vectors have incompatible lengths for an operation |
| `ZeroVectorError` | `VectorError`, `ValueError` | An operation (normalize, angle, projection) is undefined for a (near-)zero vector |

### Constructor

```python
Vector(components: Iterable[int | float])
```

- Accepts any iterable (list, tuple, generator) of numeric values.
- Rejects strings, non-iterables, non-numeric elements, and **booleans**
  (see [Design Notes](#design-notes)).
- The empty vector `Vector([])` is legal and treated as the identity case
  throughout (`dot` → `0`, `norm` → `0.0`).

### Container protocol

| Method | Behavior |
|---|---|
| `len(v)` | Number of components |
| `v[i]` | Get component at index `i` |
| `v[i:j]` | Slice → returns a new `Vector` |
| `v[i] = x` | In-place mutation (type-checked, rejects bool) |
| `iter(v)` | Iterates over components |
| `v1 == v2` | Component-wise equality; `NotImplemented`/`False` against non-`Vector` types |
| `repr(v)` | `Vector([...])` |
| `v.copy()` | Independent shallow copy |
| `hash(v)` | Explicitly unhashable (`Vector` is mutable) |

### Arithmetic

| Operation | Supports | Notes |
|---|---|---|
| `v1 + v2`, `v + scalar`, `scalar + v` | Vector, scalar | `DimensionMismatchError` on length mismatch |
| `v1 - v2`, `v - scalar`, `scalar - v` | Vector, scalar | Same mismatch behavior |
| `v * scalar`, `scalar * v` | Scalar only | Raises `TypeError` for `Vector * Vector` (use `.dot()`/`.cross()` instead) |
| `-v` | — | Element-wise negation |
| `v / scalar` | Scalar only | Raises `ZeroDivisionError` at/below `tol` (default `0.0`), `ValueError` for `NaN`, `TypeError` for non-numeric |
| `abs(v)` | — | Alias for `v.norm()` (L2) |

### Linear algebra

```python
v.dot(other: Vector) -> Number
```
Dot product. Returns `0` for two empty vectors. Raises `DimensionMismatchError` on length mismatch, `TypeError` if `other` is not a `Vector`.

```python
v.cross(other: Vector) -> Vector
```
3D cross product only. Raises `ValueError` if either vector isn't exactly length 3.

```python
v.norm(order: int | float = 2) -> float
```
p-norm. Supports `order=1` (Manhattan), `order=2` (Euclidean, default), `order=math.inf` (Chebyshev), and any positive real `order` (general p-norm). Returns `0.0` for an empty vector. Raises `ValueError` for non-positive or non-numeric `order`.

```python
v.normalize(tol: float = 0.0) -> Vector
```
Unit-length copy. Raises `ZeroVectorError` if the norm is `NaN` or within `tol` of zero.

```python
v.is_zero(tol: float = 0.0) -> bool
```
True if every component's absolute value is `<= tol`.

```python
v.angle(other: Vector, tol: float = 0.0, degrees: bool = True) -> float
```
Angle between two vectors (degrees by default). Cosine is clamped to `[-1, 1]` to absorb floating-point drift before `acos`. Raises `ZeroVectorError` if either vector's norm is within `tol` of zero.

```python
v.projection_onto(other: Vector, tol: float = 0.0) -> Vector
```
Vector projection of `self` onto `other`. Raises `ZeroVectorError` if `other`'s squared norm is within `tol` of zero.

```python
v.distance_to(other: Vector, order: int | float = 2) -> float
```
p-distance, i.e. `(self - other).norm(order)`. Raises `DimensionMismatchError` on length mismatch.

### Functional helpers

```python
v.element_wise(func: Callable[[Number], Number]) -> Vector
v.element_wise_with(other: Vector, func: Callable[[Number, Number], Number]) -> Vector
```
Apply a unary/binary function component-wise. `element_wise_with` raises `TypeError` if `other` isn't a `Vector` and `DimensionMismatchError` on length mismatch.

### Interop & constructors

| Method | Behavior |
|---|---|
| `v.to_list()` | Plain Python `list` copy |
| `v.to_numpy()` | NumPy array (`dtype=float`); raises `ImportError` if NumPy isn't installed |
| `Vector.zeros(n)` | n-dimensional zero vector; raises `ValueError` if `n < 0` |
| `Vector.from_list(values)` | Alias for the constructor |

---

## Example Session

```python
from vector import Vector, DimensionMismatchError, ZeroVectorError

v = Vector([3, 4])
v.norm()                          # 5.0
v.normalize().components          # [0.6, 0.8]

a, b = Vector([1, 2, 3]), Vector([4, 5, 6])
a.dot(b)                           # 32
a.cross(b).components              # [-3, 6, -3]

Vector([1, 0]).angle(Vector([0, 1]))   # 90.0 (degrees)

p = Vector([3, 4]).projection_onto(Vector([1, 0]))
p.components                       # [3.0, 0.0]

a.distance_to(b)                   # 5.196152422706632

try:
    Vector([1, 2]) + Vector([1, 2, 3])
except DimensionMismatchError as e:
    print(e)                       # dimension mismatch: 2 vs 3

try:
    Vector([0, 0]).normalize()
except ZeroVectorError as e:
    print(e)                       # cannot normalize a zero (or near-zero) vector
```

---

## Design Notes

- **Booleans are rejected as numeric components.** `bool` is a subclass of
  `int` in Python, so `isinstance(True, int)` is `True`. Silently accepting
  `True`/`False` as vector components is a common source of upstream bugs
  (e.g. a leaked comparison result), so `_is_number()` explicitly excludes
  `bool` everywhere a component is validated — construction, `__setitem__`,
  scalar operands, and `norm`'s `order` parameter.
- **The empty vector is a legal, well-defined object**, not an edge case to
  special-case away. `dot` on two empty vectors returns `0` (identity of
  sum) and `norm` returns `0.0`, consistent with standard mathematical
  convention.
- **Configurable zero-tolerance.** Every operation that divides by a
  magnitude (`normalize`, `angle`, `projection_onto`, `__truediv__`) accepts
  a `tol` parameter, defaulting to an exact `0.0` comparison. This keeps
  default behavior unsurprising while letting callers opt into
  near-zero-safe numerics where floating-point noise is expected.
  `Vector.__truediv__` is used internally by `/` and does not currently
  expose `tol` through the operator itself — pass it via a direct method
  call if needed.
- **Cross product is intentionally 3D-only.** The lesser-used 7D cross
  product is out of scope for this "from scratch" library; any other
  dimensionality raises `ValueError`.
- **Cosine clamping in `angle`.** Floating-point error can push
  `dot(a, b) / (|a| * |b|)` marginally outside `[-1, 1]` for near-parallel
  vectors, which would make `math.acos` raise a domain error. The result is
  clamped to `[-1.0, 1.0]` before the call.
- **`Vector` is explicitly unhashable.** Because `__setitem__` allows
  in-place mutation, `Vector` follows the same hashability contract as
  Python's built-in `list`.
- **No NumPy dependency in the library itself.** `to_numpy()` performs a
  lazy, optional import and raises `ImportError` with a clear message if
  NumPy isn't installed — the module remains usable in minimal
  environments.
- **NaN/Inf propagate rather than raise**, matching NumPy semantics (e.g.
  `dot` with a `NaN` component returns `NaN`; `norm` with an `Inf`
  component returns `Inf`). This is deliberate and covered by regression
  tests rather than being treated as an error condition.

---

## Test Coverage

`tests/test_vector.py` cross-checks numeric correctness against NumPy
(`np.dot`, `np.cross`, `np.linalg.norm`) and organizes cases into:

- Construction & type validation
- Container protocol
- Arithmetic operators
- Dot / cross product
- Norms (L1, L2, L∞, general p)
- Normalize / `is_zero`
- Angle / projection
- Distance
- Element-wise helpers
- Interop constructors (`to_list`, `to_numpy`, `zeros`, `from_list`)
- NaN / Inf propagation behavior

Run with verbose output:

```bash
pytest tests/test_vector.py -v
```

---

## Roadmap Context

`vector.py` is the first module in **Phase 1: Math Foundations** of the
`ml-implementation-from-scratch` curriculum. It establishes the
conventions carried forward into later math-foundations modules
(explicit exception hierarchy, `tol`-parameterized numerical safety,
NumPy-only-in-tests policy, full docstring + type-hint coverage) before
progressing to matrices, calculus primitives, and the from-scratch
algorithm implementations (KNN, SVM, Decision Trees, Naive Bayes, etc.)
that depend on them.
