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

