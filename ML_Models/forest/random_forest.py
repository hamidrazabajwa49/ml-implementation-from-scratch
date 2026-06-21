import os
import sys
import math
import random
import warnings
from typing import List, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_models_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(ml_models_dir, '..'))
for p in (root_dir, ml_models_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.base_class import MLModels
from ML_Models.metrics import accuracy, r2_score
from ML_Models.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _matrix_to_list,
    _vector_to_list,
    _build,
    _predict_one,
    _leaf_value_classifier,
    _leaf_value_regressor,
    Node,
)

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]


# Bootstrap sampling
def _bootstrap_sample(X_data: list, y_data: list, seed: int):
    rng = random.Random(seed)
    n = len(X_data)
    indices = [rng.randint(0, n - 1) for _ in range(n)]
    in_bag = set(indices)
    X_boot = [X_data[i] for i in indices]
    y_boot = [y_data[i] for i in indices]
    oob_indices = [i for i in range(n) if i not in in_bag]
    return X_boot, y_boot, oob_indices


def _sample_features(n_features: int, max_features: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    if max_features >= n_features:
        return list(range(n_features))
    return sorted(rng.sample(range(n_features), max_features))

def _project_X(X_data: list, feature_indices: List[int]) -> list:
    return [[row[j] for j in feature_indices] for row in X_data]


def _resolve_max_features(max_features, n_features: int, task: str) -> int:
    if isinstance(max_features, bool):
        raise ValueError(
            f"max_features must not be a bool, got {max_features!r}. "
            "Use an int, float, 'sqrt', 'log2', or None instead."
        )
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    if max_features is None:
        return max(1, int(round(n_features ** 0.5))) if task == 'classification' else max(1, n_features)
    if max_features == 'sqrt':
        return max(1, int(round(n_features ** 0.5)))
    if max_features == 'log2':
        # log2(1) == 0 → clamped to 1; guarded against n_features < 1 above
        return max(1, int(math.log2(n_features)))
    if isinstance(max_features, int):
        if max_features < 1 or max_features > n_features:
            raise ValueError(
                f"max_features={max_features} out of range [1, {n_features}]"
            )
        return max_features
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(
                f"max_features float must be in (0, 1], got {max_features}"
            )
        return max(1, int(max_features * n_features))
    raise ValueError(f"Unsupported max_features value: {max_features!r}")


# Single tree record — fitted root + feature subset used at build time
class _TreeRecord:
    __slots__ = ("root", "feature_indices")

    def __init__(self, root: Node, feature_indices: List[int]):
        self.root = root
        self.feature_indices = feature_indices

    def predict_one(self, row: list) -> float:
        projected = [row[j] for j in self.feature_indices]
        return _predict_one(self.root, projected)


def _majority_vote(votes: dict):
    """Pick the class with the most votes. Ties broken by lowest label value."""
    best_count = max(votes.values())
    winners = [k for k, v in votes.items() if v == best_count]
    try:
        return min(winners)
    except TypeError:
        return winners[0]


def _check_n_features(X_data: list, expected: int, model_name: str) -> None:
    if not X_data:
        raise ValueError(f"{model_name}.predict received an empty input (0 rows).")
    actual = len(X_data[0])
    if actual != expected:
        raise ValueError(
            f"X has {actual} features, but {model_name} expects "
            f"{expected} features as input."
        )


def _validate_random_state(random_state) -> None:
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError(
            f"random_state must be an int or None, got {type(random_state).__name__!r}"
        )
    if isinstance(random_state, bool):
        raise TypeError(
            f"random_state must not be a bool, got {random_state!r}"
        )


def _validate_common_params(n_estimators, max_depth, min_samples_split, min_impurity_decrease, random_state) -> None:
    if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) or n_estimators < 1:
        raise ValueError(f"n_estimators must be an int >= 1, got {n_estimators!r}")
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or max_depth < 1
    ):
        raise ValueError(f"max_depth must be an int >= 1 or None, got {max_depth!r}")
    if (
        not isinstance(min_samples_split, int)
        or isinstance(min_samples_split, bool)
        or min_samples_split < 2
    ):
        raise ValueError(f"min_samples_split must be an int >= 2, got {min_samples_split!r}")
    if isinstance(min_impurity_decrease, bool) or not isinstance(min_impurity_decrease, (int, float)):
        raise TypeError(
            f"min_impurity_decrease must be a float >= 0, got {min_impurity_decrease!r}"
        )
    if min_impurity_decrease < 0.0:
        raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")
    _validate_random_state(random_state)
