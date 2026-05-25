import math


def betainc(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).

    Returns the probability P(X <= x) where X ~ Beta(a, b).
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - betainc(b, a, 1.0 - x)

    EPSILON = 1e-15
    MAX_ITER = 200

    C = 1.0
    D = 1.0 - (a + b) * x / (a + 1.0)
    if abs(D) < 1e-300:
        D = 1e-300
    D = 1.0 / D
    f = D

    for i in range(1, MAX_ITER + 1):
        m = i
        delta = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        D = 1.0 + delta * D
        if abs(D) < 1e-300:
            D = 1e-300
        C = 1.0 + delta / C
        if abs(C) < 1e-300:
            C = 1e-300
        D = 1.0 / D
        f *= C * D

        delta = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        D = 1.0 + delta * D
        if abs(D) < 1e-300:
            D = 1e-300
        C = 1.0 + delta / C
        if abs(C) < 1e-300:
            C = 1e-300
        D = 1.0 / D
        delta = C * D
        f *= delta

        if abs(delta - 1.0) < EPSILON:
            break

    return front * f


def gammainc(a: float, x: float) -> float:
    """
    Regularized lower incomplete gamma function P(a, x).

    Returns the probability P(X <= x) where X ~ Gamma(a, 1).
    """
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gammainc_series(a, x)
    return 1.0 - _gammainc_cf(a, x)


def _gammainc_series(a: float, x: float) -> float:
    EPSILON = 1e-12
    MAX_ITER = 300
    term = 1.0 / a
    total = term
    for n in range(1, MAX_ITER + 1):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * EPSILON:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_cf(a: float, x: float) -> float:
    EPSILON = 1e-12
    MAX_ITER = 300
    b = x + 1.0 - a
    C = 1.0 / 1e-300
    D = 1.0 / b if abs(b) > 1e-300 else 1e300
    f = D
    for i in range(1, MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        D = an * D + b
        if abs(D) < 1e-300:
            D = 1e-300
        C = b + an / C
        if abs(C) < 1e-300:
            C = 1e-300
        D = 1.0 / D
        delta = D * C
        f *= delta
        if abs(delta - 1.0) < EPSILON:
            break
    return f * math.exp(-x + a * math.log(x) - math.lgamma(a))
