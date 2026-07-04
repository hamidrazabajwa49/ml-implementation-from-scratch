import math


def _validate_base(base: float) -> None:
    if not isinstance(base, (int, float)) or base <= 0.0 or base == 1.0:
        raise ValueError(f"base must be a positive number != 1, got {base}")


def _validate_distribution(p: list, name: str = "p") -> None:
    if len(p) == 0:
        raise ValueError(f"{name} cannot be empty")
    if any(x < 0.0 for x in p):
        raise ValueError(f"{name} probabilities must be non-negative")
    total = sum(p)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1.0, got {total}")


def _validate_joint(joint: list) -> None:
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
        if any(v < 0.0 for v in row):
            raise ValueError(f"joint distribution values must be non-negative (row {i})")


def entropy(p: list, base: float = 2.0) -> float:
    _validate_base(base)
    _validate_distribution(p)
    return -sum(prob * math.log(prob, base) for prob in p if prob > 0.0)


def binary_entropy(p: float, base: float = 2.0) -> float:
    _validate_base(base)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log(p, base) + (1.0 - p) * math.log(1.0 - p, base))


def joint_entropy(joint: list, base: float = 2.0) -> float:
    _validate_base(base)
    _validate_joint(joint)
    flat = [v for row in joint for v in row]
    return entropy(flat, base)


def marginal_from_joint(joint: list, axis: int = 0) -> list:
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    _validate_joint(joint)
    if axis == 0:
        n_cols = len(joint[0])
        return [sum(joint[r][c] for r in range(len(joint))) for c in range(n_cols)]
    return [sum(row) for row in joint]


def conditional_entropy(joint: list, given: str = "X", base: float = 2.0) -> float:
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


def mutual_information(joint: list, base: float = 2.0) -> float:
    _validate_base(base)
    H_xy = joint_entropy(joint, base)
    H_x = entropy(marginal_from_joint(joint, axis=1), base)
    H_y = entropy(marginal_from_joint(joint, axis=0), base)
    return max(0.0, H_x + H_y - H_xy)


def normalized_mutual_information(joint: list, base: float = 2.0) -> float:
    _validate_base(base)
    H_x = entropy(marginal_from_joint(joint, axis=1), base)
    H_y = entropy(marginal_from_joint(joint, axis=0), base)
    denom = 0.5 * (H_x + H_y)
    if denom == 0.0:
        return 0.0
    return mutual_information(joint, base) / denom


def cross_entropy(p: list, q: list, base: float = 2.0) -> float:
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    H = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            H -= pi * math.log(qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return H


def binary_cross_entropy(y_true: list, y_pred_proba: list, eps: float = 1e-15) -> float:
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
    total = 0.0
    for i in range(n):
        p = max(eps, min(1.0 - eps, y_pred_proba[i]))
        total += y_true[i] * math.log(p) + (1.0 - y_true[i]) * math.log(1.0 - p)
    return -total / n


def kl_divergence(p: list, q: list, base: float = 2.0) -> float:
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    D = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            D += pi * math.log(pi / qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return max(0.0, D)


def js_divergence(p: list, q: list, base: float = 2.0) -> float:
    _validate_base(base)
    if len(p) != len(q):
        raise ValueError(f"p and q must have the same length, got {len(p)} and {len(q)}")
    _validate_distribution(p, "p")
    _validate_distribution(q, "q")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def information_gain(parent: list, subsets: list, base: float = 2.0) -> float:
    _validate_base(base)
    if len(parent) == 0:
        raise ValueError("parent must be non-empty")
    if any(c < 0 for c in parent):
        raise ValueError("parent counts must be non-negative")
    n_parent = sum(parent)
    if n_parent == 0:
        raise ValueError("parent counts must not all be zero")
    if not subsets:
        raise ValueError("subsets must be non-empty")
    n_total = sum(sum(s) for s in subsets)
    if n_total == 0:
        raise ValueError("subsets must not all be zero")
    p_parent = [c / n_parent for c in parent]
    H_parent = entropy(p_parent, base)
    weighted_child = 0.0
    for subset in subsets:
        n_subset = sum(subset)
        if n_subset == 0:
            continue
        p_norm = [c / n_subset for c in subset]
        weighted_child += (n_subset / n_total) * entropy(p_norm, base)
    return max(0.0, H_parent - weighted_child)


def gini_impurity(p: list) -> float:
    _validate_distribution(p)
    return 1.0 - sum(pi ** 2 for pi in p)


def gini_gain(parent: list, subsets: list) -> float:
    if len(parent) == 0:
        raise ValueError("parent must be non-empty")
    if any(c < 0 for c in parent):
        raise ValueError("parent counts must be non-negative")
    n_parent = sum(parent)
    if n_parent == 0:
        raise ValueError("parent counts must not all be zero")
    if not subsets:
        raise ValueError("subsets must be non-empty")
    n_total = sum(sum(s) for s in subsets)
    if n_total == 0:
        raise ValueError("subsets must not all be zero")
    p_parent = [c / n_parent for c in parent]
    g_parent = gini_impurity(p_parent)
    weighted_child = 0.0
    for subset in subsets:
        n_subset = sum(subset)
        if n_subset == 0:
            continue
        p_norm = [c / n_subset for c in subset]
        weighted_child += (n_subset / n_total) * gini_impurity(p_norm)
    return max(0.0, g_parent - weighted_child)


def perplexity(p: list, base: float = 2.0) -> float:
    _validate_base(base)
    return base ** entropy(p, base)


def renyi_entropy(p: list, alpha: float, base: float = 2.0) -> float:
    _validate_base(base)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    _validate_distribution(p)
    if abs(alpha - 1.0) < 1e-10:
        return entropy(p, base)
    total = sum(pi ** alpha for pi in p if pi > 0.0)
    if total <= 0.0:
        return 0.0
    return math.log(total, base) / (1.0 - alpha)
