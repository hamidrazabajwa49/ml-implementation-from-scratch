import math


def entropy(p: list, base: float = 2.0) -> float:
    if len(p) == 0:
        raise ValueError("distribution cannot be empty")
    if any(x < 0 for x in p):
        raise ValueError("probabilities must be non-negative")
    total = sum(p)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {total}")
    H = 0.0
    for prob in p:
        if prob > 0.0:
            H -= prob * math.log(prob, base)
    return H


def binary_entropy(p: float, base: float = 2.0) -> float:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log(p, base) + (1.0 - p) * math.log(1.0 - p, base))


def joint_entropy(joint: list, base: float = 2.0) -> float:
    if len(joint) == 0:
        raise ValueError("joint distribution cannot be empty")
    flat = [v for row in joint for v in row]
    return entropy(flat, base)


def marginal_from_joint(joint: list, axis: int = 0) -> list:
    if axis == 0:
        n_cols = len(joint[0])
        return [sum(joint[r][c] for r in range(len(joint))) for c in range(n_cols)]
    return [sum(row) for row in joint]


def conditional_entropy(joint: list, given: str = "X", base: float = 2.0) -> float:
    H_xy = joint_entropy(joint, base)
    if given == "X":
        marginal = marginal_from_joint(joint, axis=1)
    elif given == "Y":
        marginal = marginal_from_joint(joint, axis=0)
    else:
        raise ValueError("given must be 'X' or 'Y'")
    H_given = entropy(marginal, base)
    return max(0.0, H_xy - H_given)


def mutual_information(joint: list, base: float = 2.0) -> float:
    H_xy = joint_entropy(joint, base)
    H_x  = entropy(marginal_from_joint(joint, axis=1), base)
    H_y  = entropy(marginal_from_joint(joint, axis=0), base)
    return max(0.0, H_x + H_y - H_xy)


def normalized_mutual_information(joint: list, base: float = 2.0) -> float:
    H_x = entropy(marginal_from_joint(joint, axis=1), base)
    H_y = entropy(marginal_from_joint(joint, axis=0), base)
    denom = 0.5 * (H_x + H_y)
    if denom == 0.0:
        return 0.0
    return mutual_information(joint, base) / denom


def cross_entropy(p: list, q: list, base: float = 2.0) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"p must sum to 1.0, got {sum(p)}")
    if abs(sum(q) - 1.0) > 1e-6:
        raise ValueError(f"q must sum to 1.0, got {sum(q)}")
    H = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            H -= pi * math.log(qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return H


def kl_divergence(p: list, q: list, base: float = 2.0) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"p must sum to 1.0, got {sum(p)}")
    if abs(sum(q) - 1.0) > 1e-6:
        raise ValueError(f"q must sum to 1.0, got {sum(q)}")
    D = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            D += pi * math.log(pi / qi, base)
        elif pi > 0.0 and qi == 0.0:
            return float("inf")
    return max(0.0, D)


def js_divergence(p: list, q: list, base: float = 2.0) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"p must sum to 1.0, got {sum(p)}")
    if abs(sum(q) - 1.0) > 1e-6:
        raise ValueError(f"q must sum to 1.0, got {sum(q)}")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def information_gain(parent: list, subsets: list, base: float = 2.0) -> float:
    if sum(parent) == 0:
        raise ValueError("parent counts must not all be zero")
    n_parent = sum(parent)
    p_parent = [c / n_parent for c in parent]
    H_parent = entropy(p_parent, base)
    n_total = sum(sum(s) for s in subsets)
    if n_total == 0:
        raise ValueError("subsets must not all be zero")
    weighted_child = 0.0
    for subset in subsets:
        n_subset = sum(subset)
        if n_subset == 0:
            continue
        p_norm = [c / n_subset for c in subset]
        weighted_child += (n_subset / n_total) * entropy(p_norm, base)
    return max(0.0, H_parent - weighted_child)


def gini_impurity(p: list) -> float:
    if len(p) == 0:
        raise ValueError("distribution cannot be empty")
    if any(x < 0 for x in p):
        raise ValueError("probabilities must be non-negative")
    total = sum(p)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {total}")
    return 1.0 - sum(pi ** 2 for pi in p)


def gini_gain(parent: list, subsets: list) -> float:
    if sum(parent) == 0:
        raise ValueError("parent counts must not all be zero")
    n_parent = sum(parent)
    p_parent = [c / n_parent for c in parent]
    g_parent = gini_impurity(p_parent)
    n_total = sum(sum(s) for s in subsets)
    if n_total == 0:
        raise ValueError("subsets must not all be zero")
    weighted_child = 0.0
    for subset in subsets:
        n_subset = sum(subset)
        if n_subset == 0:
            continue
        p_norm = [c / n_subset for c in subset]
        weighted_child += (n_subset / n_total) * gini_impurity(p_norm)
    return max(0.0, g_parent - weighted_child)


def perplexity(p: list, base: float = 2.0) -> float:
    H = entropy(p, base)
    return base ** H


def renyi_entropy(p: list, alpha: float, base: float = 2.0) -> float:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if len(p) == 0:
        raise ValueError("distribution cannot be empty")
    if any(x < 0 for x in p):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {sum(p)}")
    if abs(alpha - 1.0) < 1e-10:
        return entropy(p, base)
    total = sum(pi ** alpha for pi in p if pi > 0.0)
    if total <= 0.0:
        return 0.0
    return math.log(total, base) / (1.0 - alpha)

