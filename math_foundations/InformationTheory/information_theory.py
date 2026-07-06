"""
information_theory.py
======================

Information-theoretic quantities used throughout ML: Shannon entropy and
its variants (binary, joint, conditional), mutual information, cross
entropy and KL/Jensen-Shannon divergence, decision-tree splitting
criteria (information gain, Gini gain), perplexity, and Renyi entropy.

All functions operate on plain Python lists/sequences of probabilities
(or, for the tree-splitting criteria, raw non-negative counts) so this
module has no external dependencies. Every "log" is base-configurable
(default base 2, i.e. bits) via a ``base`` parameter, dispatched to the
most numerically precise stdlib routine available for common bases
(2, e, 10) rather than the generic two-argument ``math.log(x, base)``.

Example
-------
>>> round(entropy([0.5, 0.5]), 6)
1.0
>>> round(kl_divergence([0.5, 0.5], [0.9, 0.1]), 4)
0.7369
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Sequence, Union

Number = Union[int, float]
logger = logging.getLogger(__name__)

_DEFAULT_TOL = 1e-6



# Logging helper: dispatch to the most precise stdlib log for common bases

def _log(x: float, base: float) -> float:
    """Logarithm of ``x`` in the given ``base``, using the most precise stdlib routine available.

    ``math.log2``/``math.log10``/natural ``math.log`` are each slightly
    more accurate than the generic two-argument ``math.log(x, base)``
    for their respective bases (fewer floating-point rounding steps),
    which matters when chaining many log evaluations (as entropy sums do).
    """
    if base == 2.0:
        return math.log2(x)
    if base == 10.0:
        return math.log10(x)
    if base == math.e:
        return math.log(x)
    return math.log(x) / math.log(base)


def _validate_base(base: float) -> None:
    """Validate that ``base`` is a positive, finite, non-NaN real number != 1.

    Raises
    ------
    TypeError
        If ``base`` is not a real number (or is a ``bool``).
    ValueError
        If ``base`` is non-positive, equal to 1, NaN, or infinite.
    """
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise TypeError(f"base must be a real number, got {type(base).__name__}")
    if math.isnan(base):
        raise ValueError("base must not be NaN")
    if math.isinf(base):
        raise ValueError("base must be finite")
    if base <= 0.0 or base == 1.0:
        raise ValueError(f"base must be a positive number != 1, got {base}")


def _validate_distribution(p: Sequence[Number], name: str = "p", tol: float = _DEFAULT_TOL) -> None:
    """Validate that ``p`` is a non-empty probability distribution summing to 1.

    Raises
    ------
    TypeError
        If ``p`` is empty of validatable content or contains non-numeric
        (including ``bool``) elements.
    ValueError
        If any probability is negative, NaN, infinite, or the total
        doesn't sum to 1 within ``tol``.
    """
    if len(p) == 0:
        raise ValueError(f"{name} cannot be empty")
    for i, x in enumerate(p):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric, got {type(x).__name__}")
        if math.isnan(x):
            raise ValueError(f"{name}[{i}] is NaN")
        if math.isinf(x):
            raise ValueError(f"{name}[{i}] is infinite")
        if x < 0.0:
            raise ValueError(f"{name} probabilities must be non-negative (got {x} at index {i})")
    total = sum(p)
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1.0 (within tol={tol}), got {total}")


def _validate_joint(joint: Sequence[Sequence[Number]], tol: float = _DEFAULT_TOL) -> None:
    """Validate a rectangular, non-negative, NaN-free joint distribution table.

    Note this does *not* itself check the joint sums to 1 -- callers that
    treat ``joint`` as a probability table (e.g. :func:`joint_entropy`)
    get that check for free via their downstream :func:`entropy` call;
    callers that treat it as raw counts don't want that restriction.

    Raises
    ------
    ValueError
        If empty, ragged, contains negative/NaN/infinite values.
    TypeError
        If any element is non-numeric.
    """
    if len(joint) == 0:
        raise ValueError("joint distribution cannot be empty")
    n_cols = len(joint[0])
    if n_cols == 0:
        raise ValueError("joint distribution rows cannot be empty")
    for i, row in enumerate(joint):
        if len(row) != n_cols:
            raise ValueError(
                f"all rows must have the same length; "
                f"row 0 has {n_cols} cols but row {i} has {len(row)}"
            )
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"joint[{i}][{j}] must be numeric, got {type(v).__name__}")
            if math.isnan(v):
                raise ValueError(f"joint[{i}][{j}] is NaN")
            if math.isinf(v):
                raise ValueError(f"joint[{i}][{j}] is infinite")
            if v < 0.0:
                raise ValueError(f"joint distribution values must be non-negative (row {i}, col {j})")


def _validate_counts(counts: Sequence[Number], name: str) -> None:
    """Validate a non-empty sequence of non-negative, finite, non-NaN counts.

    Raises
    ------
    ValueError
        If empty, negative, NaN, or infinite.
    TypeError
        If any element is non-numeric.
    """
    if len(counts) == 0:
        raise ValueError(f"{name} must be non-empty")
    for i, c in enumerate(counts):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric, got {type(c).__name__}")
        if math.isnan(c):
            raise ValueError(f"{name}[{i}] is NaN")
        if math.isinf(c):
            raise ValueError(f"{name}[{i}] is infinite")
        if c < 0:
            raise ValueError(f"{name}[{i}] must be non-negative, got {c}")



# Shannon entropy family

def entropy(p: Sequence[Number], base: float = 2.0, tol: float = _DEFAULT_TOL) -> float:
    """Shannon entropy :math:`H(p) = -\\sum_i p_i \\log_{base} p_i`.

    Parameters
    ----------
    p : Sequence[Number]
        A probability distribution (non-negative, sums to 1).
    base : float, optional
        Logarithm base (default 2, giving entropy in bits).
    tol : float, optional
        Tolerance for the sum-to-1 check.

    Returns
    -------
    float
        Non-negative entropy value.

    Raises
    ------
    TypeError, ValueError
        See :func:`_validate_distribution` / :func:`_validate_base`.
    """
    _validate_base(base)
    _validate_distribution(p, tol=tol)
    h = -sum(prob * _log(prob, base) for prob in p if prob > 0.0)
    logger.debug("entropy(base=%s) = %s", base, h)
    return h


def binary_entropy(p: float, base: float = 2.0) -> float:
    """Shannon entropy of a Bernoulli(p) distribution.

    Parameters
    ----------
    p : float
        Probability of the "1" outcome; must be in ``[0, 1]``.
    base : float, optional

    Raises
    ------
    ValueError
        If ``p`` is not in ``[0, 1]`` (including NaN, which fails this
        range check naturally).
    """
    _validate_base(base)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * _log(p, base) + (1.0 - p) * _log(1.0 - p, base))


def joint_entropy(joint: Sequence[Sequence[Number]], base: float = 2.0) -> float:
    """Joint entropy :math:`H(X, Y)` of a 2D joint probability table.

    Raises
    ------
    TypeError, ValueError
        See :func:`_validate_joint`; the flattened table must also sum to 1.
    """
    _validate_base(base)
    _validate_joint(joint)
    flat = [v for row in joint for v in row]
    return entropy(flat, base)


def marginal_from_joint(joint: Sequence[Sequence[Number]], axis: int = 0) -> List[float]:
    """Marginal distribution derived from a joint table by summing out one axis.

    Parameters
    ----------
    joint : Sequence[Sequence[Number]]
    axis : {0, 1}, optional
        ``axis=0`` sums over rows, yielding the per-column (Y) marginal.
        ``axis=1`` sums over columns, yielding the per-row (X) marginal.

    Raises
    ------
    ValueError
        If ``axis`` is not 0 or 1, or the joint table is malformed (see
        :func:`_validate_joint`).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    _validate_joint(joint)
    if axis == 0:
        n_cols = len(joint[0])
        return [sum(joint[r][c] for r in range(len(joint))) for c in range(n_cols)]
    return [sum(row) for row in joint]


def conditional_entropy(joint: Sequence[Sequence[Number]], given: str = "X", base: float = 2.0) -> float:
    """Conditional entropy: ``H(Y|X)`` if ``given='X'``, or ``H(X|Y)`` if ``given='Y'``.

    Computed as ``H(X,Y) - H(given)``.

    Raises
    ------
    ValueError
        If ``given`` is not ``'X'`` or ``'Y'``, or the joint table is invalid.
    """
    _validate_base(base)
    if given == "X":
        marginal = marginal_from_joint(joint, axis=1)
    elif given == "Y":
        marginal = marginal_from_joint(joint, axis=0)
    else:
        raise ValueError("given must be 'X' or 'Y'")
    H_xy = joint_entropy(joint, base)
    H_given = entropy(marginal, base)
    return max(0.0, H_xy - H_given)


def mutual_information(joint: Sequence[Sequence[Number]], base: float = 2.0) -> float:
    """Mutual information :math:`I(X;Y) = H(X) + H(Y) - H(X,Y)`.

    Raises
    ------
    TypeError, ValueError
        See :func:`_validate_joint`.
    """
    _validate_base(base)
    H_xy = joint_entropy(joint, base)
    H_x = entropy(marginal_from_joint(joint, axis=1), base)
    H_y = entropy(marginal_from_joint(joint, axis=0), base)
    mi = max(0.0, H_x + H_y - H_xy)
    logger.debug("mutual_information(base=%s) = %s", base, mi)
    return mi


def normalized_mutual_information(joint: Sequence[Sequence[Number]], base: float = 2.0) -> float:
    """NMI via the arithmetic-mean normalization: ``I(X;Y) / (0.5*(H(X)+H(Y)))``.

    Returns 0.0 in the degenerate case where both marginals have zero
    entropy (both variables are deterministic/constant).

    Note
    ----
    This is one of several common NMI normalization conventions
    (arithmetic mean of H(X), H(Y); other conventions use their geometric
    mean, min, or max instead) -- matches scikit-learn's
    ``average_method='arithmetic'``.
    """
    _validate_base(base)
    H_x = entropy(marginal_from_joint(joint, axis=1), base)
    H_y = entropy(marginal_from_joint(joint, axis=0), base)
    denom = 0.5 * (H_x + H_y)
    if denom == 0.0:
        return 0.0
    return mutual_information(joint, base) / denom



# Cross entropy / divergences

def cross_entropy(p: Sequence[Number], q: Sequence[Number], base: float = 2.0) -> float:
    """Cross entropy :math:`H(p, q) = -\\sum_i p_i \\log_{base} q_i`.

    Returns ``float('inf')`` if ``p_i > 0`` where ``q_i == 0`` (q assigns
    zero probability to an event p considers possible).

    Raises
    ------
    ValueError
        If lengths mismatch, or either isn't a valid distribution (see
        :func:`_validate_distribution`).
    """
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    H = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            H -= pi * _log(qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return H


def binary_cross_entropy(
    y_true: Sequence[Number],
    y_pred_proba: Sequence[Number],
    eps: float = 1e-15,
) -> float:
    """Binary (log) cross-entropy loss, averaged over samples.

    Parameters
    ----------
    y_true : Sequence[Number]
        Ground-truth binary labels (each 0 or 1).
    y_pred_proba : Sequence[Number]
        Predicted probabilities of the "1" class; each must be in
        ``[0, 1]``. Values are clipped to ``[eps, 1-eps]`` only to avoid
        ``log(0)`` at the exact boundary -- values genuinely outside
        ``[0, 1]`` are rejected rather than silently clipped, since that
        usually indicates a bug in the caller's model.
    eps : float, optional
        Boundary-clipping epsilon.

    Raises
    ------
    ValueError
        If lengths mismatch, ``y_true`` contains non-binary values, or
        ``y_pred_proba`` contains a value outside ``[0, 1]`` or NaN.
    """
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"y_true and y_pred_proba must have the same length, "
            f"got {len(y_true)} and {len(y_pred_proba)}"
        )
    n = len(y_true)
    if n == 0:
        return 0.0
    for i, yt in enumerate(y_true):
        if yt not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"y_true must contain binary values (0 or 1), got {yt} at index {i}"
            )
    for i, yp in enumerate(y_pred_proba):
        if isinstance(yp, bool) or not isinstance(yp, (int, float)):
            raise TypeError(f"y_pred_proba[{i}] must be numeric, got {type(yp).__name__}")
        if math.isnan(yp):
            raise ValueError(f"y_pred_proba[{i}] is NaN")
        if not (0.0 <= yp <= 1.0):
            raise ValueError(f"y_pred_proba[{i}] must be in [0, 1], got {yp}")

    total = 0.0
    for i in range(n):
        p = max(eps, min(1.0 - eps, y_pred_proba[i]))
        total += y_true[i] * math.log(p) + (1.0 - y_true[i]) * math.log(1.0 - p)
    return -total / n


def kl_divergence(p: Sequence[Number], q: Sequence[Number], base: float = 2.0) -> float:
    """Kullback-Leibler divergence :math:`D_{KL}(p \\| q) = \\sum_i p_i \\log_{base}(p_i/q_i)`.

    Returns ``float('inf')`` if ``p_i > 0`` where ``q_i == 0``.

    Raises
    ------
    ValueError
        If lengths mismatch, or either isn't a valid distribution.
    """
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    D = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            D += pi * _log(pi / qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return max(0.0, D)  # clamp floating-point noise; D_KL >= 0 by Gibbs' inequality


def js_divergence(p: Sequence[Number], q: Sequence[Number], base: float = 2.0) -> float:
    """Jensen-Shannon divergence: the symmetrized, smoothed KL divergence.

    ``JSD(p, q) = 0.5*D_KL(p||m) + 0.5*D_KL(q||m)`` where ``m = (p+q)/2``.
    Always finite (unlike KL divergence) and symmetric in ``p``, ``q``.

    Raises
    ------
    ValueError
        If lengths mismatch, or either isn't a valid distribution.
    """
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)



# Decision-tree splitting criteria

def _weighted_impurity_gain(
    parent: Sequence[Number],
    subsets: Sequence[Sequence[Number]],
    impurity_fn: Callable[[List[float]], float],
) -> float:
    """Shared logic for information gain and Gini gain: parent impurity minus weighted child impurity.

    Parameters
    ----------
    parent : Sequence[Number]
        Non-negative class counts before the split.
    subsets : Sequence[Sequence[Number]]
        One sequence of non-negative class counts per child node after
        the split; each must have the same number of categories as
        ``parent``.
    impurity_fn : Callable[[List[float]], float]
        The impurity measure to apply to each normalized distribution
        (e.g. :func:`entropy` or :func:`gini_impurity`).

    Raises
    ------
    ValueError
        If ``parent``/``subsets`` are empty, contain negative/NaN counts,
        a subset's category count doesn't match ``parent``'s, or both
        parent and subset totals are zero.
    """
    _validate_counts(parent, "parent")
    n_parent = sum(parent)
    if n_parent == 0:
        raise ValueError("parent counts must not all be zero")
    if not subsets:
        raise ValueError("subsets must be non-empty")
    for i, subset in enumerate(subsets):
        _validate_counts(subset, f"subsets[{i}]")
        if len(subset) != len(parent):
            raise ValueError(
                f"subsets[{i}] has {len(subset)} categories but parent has "
                f"{len(parent)}; category counts must align positionally"
            )
    n_total = sum(sum(s) for s in subsets)
    if n_total == 0:
        raise ValueError("subsets must not all be zero")

    p_parent = [c / n_parent for c in parent]
    impurity_parent = impurity_fn(p_parent)

    weighted_child = 0.0
    for subset in subsets:
        n_subset = sum(subset)
        if n_subset == 0:
            continue
        p_norm = [c / n_subset for c in subset]
        weighted_child += (n_subset / n_total) * impurity_fn(p_norm)

    return max(0.0, impurity_parent - weighted_child)


def information_gain(
    parent: Sequence[Number],
    subsets: Sequence[Sequence[Number]],
    base: float = 2.0,
) -> float:
    """Information gain (entropy reduction) from splitting ``parent`` counts into ``subsets``.

    Standard ID3/C4.5 decision-tree splitting criterion.

    Raises
    ------
    See :func:`_weighted_impurity_gain`.
    """
    _validate_base(base)
    return _weighted_impurity_gain(parent, subsets, lambda p: entropy(p, base))


def gini_impurity(p: Sequence[Number]) -> float:
    """Gini impurity :math:`1 - \\sum_i p_i^2` of a class distribution.

    Raises
    ------
    See :func:`_validate_distribution`.
    """
    _validate_distribution(p)
    return 1.0 - sum(pi ** 2 for pi in p)


def gini_gain(parent: Sequence[Number], subsets: Sequence[Sequence[Number]]) -> float:
    """Gini impurity reduction from splitting ``parent`` counts into ``subsets`` (CART criterion).

    Raises
    ------
    See :func:`_weighted_impurity_gain`.
    """
    return _weighted_impurity_gain(parent, subsets, gini_impurity)



# Perplexity / Renyi entropy

def perplexity(p: Sequence[Number], base: float = 2.0) -> float:
    """Perplexity :math:`base^{H(p)}`: the effective number of equally-likely outcomes.

    Raises
    ------
    See :func:`entropy`.
    """
    _validate_base(base)
    return base ** entropy(p, base)


def renyi_entropy(p: Sequence[Number], alpha: float, base: float = 2.0) -> float:
    """Renyi entropy of order ``alpha``, generalizing Shannon entropy (``alpha -> 1``).

    Computed in log-space (log-sum-exp) rather than via direct
    exponentiation of each ``p_i**alpha``, which avoids silent underflow
    to exactly 0.0 for large ``alpha`` (min-entropy limit) that would
    otherwise produce a wrong result instead of the correct
    ``-log(max(p), base)`` limiting value.

    Parameters
    ----------
    p : Sequence[Number]
    alpha : float
        Order parameter; must be positive. ``alpha=1`` (within
        floating-point tolerance) reduces to Shannon entropy;
        ``alpha->0`` approaches Hartley entropy (log of support size);
        ``alpha->inf`` approaches min-entropy (``-log(max(p))``).
    base : float, optional

    Raises
    ------
    ValueError
        If ``alpha`` is non-positive, NaN, or infinite, or ``p`` is invalid.
    """
    _validate_base(base)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}")
    if math.isnan(alpha):
        raise ValueError("alpha must not be NaN")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    _validate_distribution(p)
    if abs(alpha - 1.0) < 1e-10:
        return entropy(p, base)
    if math.isinf(alpha):
        # Min-entropy limit: H_inf(p) = -log(max(p)). Handled as a special
        # case rather than falling through to the log-space formula below,
        # since alpha * log(p_i) is `inf * 0 = nan` whenever any p_i == 1.0
        # exactly.
        return -_log(max(p), base)

    # log-space: log(sum(pi**alpha)) = logsumexp(alpha * log(pi))
    log_terms = [alpha * math.log(pi) for pi in p if pi > 0.0]
    if not log_terms:
        return 0.0
    max_log = max(log_terms)
    log_sum_natural = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))

    result_natural = log_sum_natural / (1.0 - alpha)
    return result_natural / math.log(base)
