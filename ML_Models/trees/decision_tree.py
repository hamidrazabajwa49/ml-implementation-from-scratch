import os
import sys
from typing import List, Optional, Union
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
main_folder = os.path.abspath(os.path.join(parent_dir, '..'))
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.base_class import MLModels
from ML_Models.metrics import accuracy, r2_score
from InformationTheory.information_theory import information_gain, gini_gain



class Node:
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature_idx}, threshold={self.threshold:.4f})"




def _col(X_data: list, j: int) -> list:
    return [X_data[i][j] for i in range(len(X_data))]


def _split_indices(X_data: list, feature_idx: int, threshold: float):
    left_idx = [i for i, row in enumerate(X_data) if row[feature_idx] <= threshold]
    right_idx = [i for i, row in enumerate(X_data) if row[feature_idx] > threshold]
    return left_idx, right_idx


def _unique_thresholds(values: list) -> list:
    sorted_vals = sorted(set(values))
    if len(sorted_vals) == 1:
        return sorted_vals
    return [(sorted_vals[i] + sorted_vals[i + 1]) / 2.0
            for i in range(len(sorted_vals) - 1)]


def _variance(values: list) -> float:
    if len(values) == 0:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _variance_reduction(y: list, left: list, right: list) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    n_l, n_r = len(left), len(right)
    return _variance(y) - (n_l / n) * _variance(left) - (n_r / n) * _variance(right)


def _information_gain(y: list, left: list, right: list) -> float:
    classes = sorted(set(y))
    if len(classes) == 0:
        return 0.0

    def counts(subset):
        c = Counter(subset)
        return [c.get(cls, 0) for cls in classes]

    return information_gain(counts(y), [counts(left), counts(right)])


def _gini_gain(y: list, left: list, right: list) -> float:
    classes = sorted(set(y))
    if len(classes) == 0:
        return 0.0

    def counts(subset):
        c = Counter(subset)
        return [c.get(cls, 0) for cls in classes]

    return gini_gain(counts(y), [counts(left), counts(right)])


def _best_split(X_data: list, y: list, criterion: str) -> dict:
    n_features = len(X_data[0]) if X_data else 0
    best = {"gain": -1.0, "feature_idx": None, "threshold": None}

    for j in range(n_features):
        col = _col(X_data, j)
        for thresh in _unique_thresholds(col):
            left_idx, right_idx = _split_indices(X_data, j, thresh)
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            y_l = [y[i] for i in left_idx]
            y_r = [y[i] for i in right_idx]

            if criterion == "entropy":
                gain = _information_gain(y, y_l, y_r)
            elif criterion == "gini":
                gain = _gini_gain(y, y_l, y_r)
            elif criterion == "variance":
                gain = _variance_reduction(y, y_l, y_r)
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")

            if gain > best["gain"]:
                best = {"gain": gain, "feature_idx": j, "threshold": thresh}

    return best


def _leaf_value_classifier(y: list) -> float:
    if not y:
        raise ValueError("Cannot compute leaf value for empty node")
    return float(Counter(y).most_common(1)[0][0])


def _leaf_value_regressor(y: list) -> float:
    if not y:
        raise ValueError("Cannot compute leaf value for empty node")
    return sum(y) / len(y)


def _build(
    X_data: list,
    y: list,
    criterion: str,
    leaf_fn,
    depth: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_impurity_decrease: float,
) -> Node:
    n = len(y)

    # Stopping: pure node, too small, or at max depth
    if (
        len(set(y)) == 1
        or n < min_samples_split
        or (max_depth is not None and depth >= max_depth)
    ):
        return Node(value=leaf_fn(y))

    split = _best_split(X_data, y, criterion)

    if split["feature_idx"] is None or split["gain"] < min_impurity_decrease:
        return Node(value=leaf_fn(y))

    left_idx, right_idx = _split_indices(X_data, split["feature_idx"], split["threshold"])

    left = _build(
        [X_data[i] for i in left_idx], [y[i] for i in left_idx],
        criterion, leaf_fn, depth + 1, max_depth, min_samples_split, min_impurity_decrease,
    )
    right = _build(
        [X_data[i] for i in right_idx], [y[i] for i in right_idx],
        criterion, leaf_fn, depth + 1, max_depth, min_samples_split, min_impurity_decrease,
    )

    return Node(
        feature_idx=split["feature_idx"],
        threshold=split["threshold"],
        left=left,
        right=right,
    )


def _predict_one(node: Node, row: list) -> float:
    if node.is_leaf():
        return node.value
    if row[node.feature_idx] <= node.threshold:
        return _predict_one(node.left, row)
    return _predict_one(node.right, row)


def _matrix_to_list(X: Matrix) -> list:
    return [list(X.rows[i].components) for i in range(X.n_rows)]


def _vector_to_list(y: Vector) -> list:
    return list(y.components)





class DecisionTreeClassifier(MLModels):

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_impurity_decrease < 0.0:
            raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self._root: Optional[Node] = None

    def fit(self, X: Matrix, y: Vector) -> None:
        self._validate_Xy(X, y)
        X_data = _matrix_to_list(X)
        y_data = _vector_to_list(y)
        self._root = _build(
            X_data, y_data,
            self.criterion, _leaf_value_classifier,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        # store w as sentinel so _check_is_fitted works
        self.w = True

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        X_data = _matrix_to_list(X)
        return Vector([_predict_one(self._root, row) for row in X_data])

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def parameters(self):
        self._check_is_fitted()
        return [self._root]

    def _check_is_fitted(self) -> None:
        if self._root is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() before predict() or score()."
            )

    def get_depth(self) -> int:
        self._check_is_fitted()
        return _tree_depth(self._root)

    def get_n_leaves(self) -> int:
        self._check_is_fitted()
        return _count_leaves(self._root)




class DecisionTreeRegressor(MLModels):

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
    ):
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_impurity_decrease < 0.0:
            raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self._root: Optional[Node] = None

    def fit(self, X: Matrix, y: Vector) -> None:
        self._validate_Xy(X, y)
        X_data = _matrix_to_list(X)
        y_data = _vector_to_list(y)
        self._root = _build(
            X_data, y_data,
            "variance", _leaf_value_regressor,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        self.w = True

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        X_data = _matrix_to_list(X)
        return Vector([_predict_one(self._root, row) for row in X_data])

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def parameters(self):
        self._check_is_fitted()
        return [self._root]

    def _check_is_fitted(self) -> None:
        if self._root is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() before predict() or score()."
            )

    def get_depth(self) -> int:
        self._check_is_fitted()
        return _tree_depth(self._root)

    def get_n_leaves(self) -> int:
        self._check_is_fitted()
        return _count_leaves(self._root)



def _tree_depth(node: Optional[Node]) -> int:
    if node is None or node.is_leaf():
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _count_leaves(node: Optional[Node]) -> int:
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)
