import os
import sys
import math
from typing import Dict, List, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_models_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(ml_models_dir, '..'))
for p in (root_dir, ml_models_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.base_class import MLModels
from ML_Models.metrics import accuracy_score
from Probability.continuous import NormalDistribution
from Probability.descriptive_stats import DescriptiveStats


# Floor used whenever we are about to take math.log() of something that
_LOG_FLOOR = 1e-300


def _safe_log(x: float) -> float:
    """log(x) that never raises -- floors non-positive/zero input."""
    if x <= 0.0:
        return math.log(_LOG_FLOOR)
    return math.log(x)


def _log_sum_exp_proba(log_scores: Dict[float, float]) -> Dict[float, float]:
    """Convert raw log-posteriors to normalised probabilities via log-sum-exp."""
    if not log_scores:
        raise ValueError("log_scores is empty; cannot compute probabilities")
    vals = list(log_scores.values())
    max_v = max(vals)
    exp_vals = {c: math.exp(v - max_v) for c, v in log_scores.items()}
    denom = sum(exp_vals.values())
    if denom == 0.0:
        # Every class scored at -inf relative to itself. Fall back to a uniform distribution rather than dividing by zero.
        n = len(log_scores)
        return {c: 1.0 / n for c in log_scores}
    return {c: e / denom for c, e in exp_vals.items()}


def _validate_priors(priors: Dict, classes: List) -> None:
    if not isinstance(priors, dict):
        raise TypeError(f"priors must be a dict, got {type(priors).__name__}")
    if set(priors.keys()) != set(classes):
        raise ValueError(
            f"priors keys must match class labels exactly. "
            f"Expected {sorted(classes)}, got {sorted(priors.keys())}"
        )
    for c, p in priors.items():
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError(f"prior for class {c} must be numeric, got {type(p).__name__}")
        if not math.isfinite(p):
            raise ValueError(f"prior for class {c} must be finite, got {p}")
    total = sum(priors.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"priors must sum to 1.0, got {total}")
    if any(p < 0 for p in priors.values()):
        raise ValueError("all priors must be non-negative")
