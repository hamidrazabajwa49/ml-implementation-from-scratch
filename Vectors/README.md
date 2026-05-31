# 01 · Vector Operations Library

> **Part of ml-implementation-from-scratch · Phase 1 — Math Foundations · Project 1/10**

A pure-Python, dependency-free `Vector` class implementing all the linear algebra primitives needed for ML from scratch — dot products, norms (L1 / L2 / L∞), projections, angle computation, and more.

---

## Why this exists

Every ML algorithm from linear regression to transformers ultimately reduces to vector math. Before reaching for NumPy, this module builds those operations by hand so the intuition is solid before the abstraction is added.

---

## What's implemented

| Method | Description |
|---|---|
| `__add__` / `__radd__` | Element-wise addition; also supports scalar offset |
| `__sub__` / `__rsub__` | Element-wise subtraction; also supports scalar |
| `__mul__` / `__rmul__` | Scalar multiplication |
| `dot(other)` | Dot product — foundation of projections, cosine similarity, and matrix ops |
| `norm(order)` | L1 (Manhattan), L2 (Euclidean), L∞ (Chebyshev) norms |
| `angle(other)` | Angle between two vectors in degrees via the dot-product formula |
| `normalize()` | Unit vector — divides by L2 norm |
| `projection_onto(other)` | Orthogonal projection of `self` onto `other` |
| `element_wise(func)` | Apply any single-argument function element-wise |
| `element_wise_with(other, func)` | Apply any two-argument function element-wise across two vectors |

Python dunder support: `__repr__`, `__getitem__`, `__len__`, `__iter__` — so `Vector` behaves like a native sequence.

---

## Quick start

```python
import math
from vector import Vector

a = Vector([3, 4, 0])
b = Vector([1, 0, 0])

print(a.norm())               # 5.0  (L2)
print(a.norm(order=1))        # 7    (L1)
print(a.norm(order=math.inf)) # 4    (L∞)

print(a.dot(b))               # 3
print(a.angle(b))             # 36.87°
print(a.normalize())          # Vector([0.6, 0.8, 0.0])
print(a.projection_onto(b))   # Vector([3, 0, 0])

# Arithmetic
print(a + b)                  # Vector([4, 4, 0])
print(a * 2)                  # Vector([6, 8, 0])
print(3 * a)                  # Vector([9, 12, 0])
print(a - 1)                  # Vector([2, 3, -1])
```

---

## Math behind the operations

**Dot product**

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i$$

**Norms**

$$\|\mathbf{v}\|_1 = \sum|v_i|, \quad \|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}, \quad \|\mathbf{v}\|_\infty = \max|v_i|$$

**Angle**

$$\theta = \arccos\!\left(\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}\right)$$

**Projection of a onto b**

$$\text{proj}_{\mathbf{b}}\,\mathbf{a} = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\,\mathbf{b}$$

---

## Requirements

- Python 3.8+
- Standard library only (`math`) — no NumPy, no external packages

---

## What comes next

This library feeds directly into the next projects in the series:

- **[02 · Matrix Operations](../02-matrix/)** — builds on `dot` to implement matrix multiply, transpose, determinant, and Gaussian elimination
- **[03 · Eigenvalues & Eigenvectors](../03-eigenvalues/)** — power iteration uses `normalize()` from this module
- **[04 · SVD](../04-svd/)** — requires all of the above

---

## Series context

| Phase | Projects | Status |
|---|---|---|
| **P1 Math Foundations** | 01–10 | 🟢 In progress |
| P2 ML Algorithms | 11–25 | ⬜ Upcoming |
| P3 Real-World Projects | 26–40 | ⬜ Upcoming |
| P4 Advanced + Deployed | 41–52 | ⬜ Upcoming |
| P5 Paper Implementations | 53–60 | ⬜ Upcoming |

---

*Built as part of a 60-day public challenge — follow along at 
-GitHub: [github.com/hamidrazabajwa49](https://github.com/hamidrazabajwa49)
- LinkedIn: [linkedin.com/in/hamid-raza-bajwa-564a91377](https://linkedin.com/in/hamid-raza-bajwa-564a91377)
- Email: hamidrazabajwa49@gmail.com
*
