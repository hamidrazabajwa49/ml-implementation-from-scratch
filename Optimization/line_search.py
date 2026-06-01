import math


def backtracking(f,x: list,grad: list,direction: list,alpha: float = 1.0,rho: float   = 0.5,c: float     = 1e-4,max_iter: int = 100,) -> float:
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0, 1)")

    f0 = f(x)
    slope = sum(g * d for g, d in zip(grad, direction))

    if slope >= 0:
        raise ValueError("direction is not a descent direction")

    for _ in range(max_iter):
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        if f(x_new) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


def golden_section(f,a: float,b: float,tol: float    = 1e-8,max_iter: int = 200,) -> dict:
    if a >= b:
        raise ValueError("a must be less than b")

    phi = (math.sqrt(5.0) - 1.0) / 2.0

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)

    x_min = (a + b) / 2.0

    return {
        "x_min":    x_min,
        "f_min":    f(x_min),
        "bracket":  (a, b),
        "converged": abs(b - a) < tol,
    }
