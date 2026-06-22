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


# RandomForestClassifier
class RandomForestClassifier(MLModels):
    """Random forest classifier: ensemble of decision trees, each trained
    on a bootstrap sample with a random feature subset at every split.

    Parameters
    n_estimators : int, default=100
    criterion : {'gini', 'entropy'}, default='gini'
    max_depth : int or None, default=None
    min_samples_split : int, default=2
    min_impurity_decrease : float, default=0.0
    max_features : {'sqrt', 'log2'}, int, float in (0,1], or None
        None → sqrt(n_features). Default='sqrt'.
    random_state : int or None, default=None
    oob_score : bool, default=False

    Attributes
    oob_score_ : float or None
    n_features_in_ : int
    classes_ : list
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Union[str, int, float, None] = 'sqrt',
        random_state: Optional[int] = None,
        oob_score: bool = False,
    ):
        _validate_common_params(n_estimators, max_depth, min_samples_split, min_impurity_decrease, random_state)
        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.oob_score = oob_score

        self._trees: List[_TreeRecord] = []
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[list] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: Matrix, y: Vector) -> "RandomForestClassifier":
        self._validate_Xy(X, y)
        X_data = _matrix_to_list(X)
        y_data = _vector_to_list(y)
        n_samples = len(X_data)
        n_features = len(X_data[0])
        if n_features == 0:
            raise ValueError("Cannot fit on data with 0 features.")

        self.n_features_in_ = n_features
        self.classes_ = sorted(set(y_data)) if self._labels_orderable(y_data) \
            else sorted(set(y_data), key=str)

        max_feat = _resolve_max_features(self.max_features, n_features, 'classification')
        base_seed = self.random_state if self.random_state is not None \
            else random.randint(0, 2 ** 31 - 1)

        self._trees = []
        oob_votes: List[dict] = [{} for _ in range(n_samples)]
        any_oob = False

        for t in range(self.n_estimators):
            tree_seed = base_seed + 2 * t
            X_boot, y_boot, oob_idx = _bootstrap_sample(X_data, y_data, seed=tree_seed)
            feat_idx = _sample_features(n_features, max_feat, seed=tree_seed + 1)
            X_proj = _project_X(X_boot, feat_idx)

            root = _build(
                X_proj, y_boot,
                self.criterion, _leaf_value_classifier,
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            record = _TreeRecord(root, feat_idx)
            self._trees.append(record)

            if self.oob_score and oob_idx:
                any_oob = True
                for i in oob_idx:
                    pred = record.predict_one(X_data[i])
                    oob_votes[i][pred] = oob_votes[i].get(pred, 0) + 1

        if self.oob_score:
            self.oob_score_ = self._compute_oob_clf(oob_votes, y_data, n_samples, any_oob)

        return self

    def _compute_oob_clf(self, oob_votes, y_data, n_samples, any_oob) -> Optional[float]:
        if not any_oob:
            warnings.warn(
                "No OOB samples found. Increase n_estimators or dataset size.",
                UserWarning,
            )
            return None

        oob_preds, oob_true = [], []
        n_missing = 0
        for i in range(n_samples):
            if oob_votes[i]:
                oob_preds.append(_majority_vote(oob_votes[i]))
                oob_true.append(y_data[i])
            else:
                n_missing += 1

        if n_missing:
            warnings.warn(
                f"{n_missing} sample(s) were never out-of-bag and were "
                "excluded from the OOB score.",
                UserWarning,
            )
        if not oob_true:
            return None

        correct = sum(1 for a, b in zip(oob_true, oob_preds) if a == b)
        return correct / len(oob_true)

    @staticmethod
    def _labels_orderable(values) -> bool:
        try:
            sorted(set(values))
            return True
        except TypeError:
            return False

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        X_data = _matrix_to_list(X)
        _check_n_features(X_data, self.n_features_in_, type(self).__name__)
        preds = []
        for row in X_data:
            votes: dict = {}
            for tree in self._trees:
                p = tree.predict_one(row)
                votes[p] = votes.get(p, 0) + 1
            preds.append(_majority_vote(votes))
        return Vector(preds)

    def predict_proba(self, X: Matrix) -> List[dict]:
        """Vote share for every class seen during fit. Unknown labels
        predicted by any tree are clamped to the closest known class
        rather than silently leaking new keys into the output dict."""
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        X_data = _matrix_to_list(X)
        _check_n_features(X_data, self.n_features_in_, type(self).__name__)
        classes_set = set(self.classes_)
        result = []
        for row in X_data:
            # FIX: pre-seed with all known classes and only increment known keys.
            # A tree predicting an unseen label (impossible in normal flow but
            # possible if _trees is modified externally) is silently dropped.
            # Normalize by actual vote count so probabilities sum to 1.0
            # even if some trees are discarded.
            votes: dict = {c: 0 for c in self.classes_}
            n_valid = 0
            for tree in self._trees:
                p = tree.predict_one(row)
                if p in classes_set:
                    votes[p] += 1
                    n_valid += 1
            denom = n_valid if n_valid > 0 else 1
            result.append({k: v / denom for k, v in votes.items()})
        return result

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy(y, self.predict(X))

    def get_params(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'oob_score': self.oob_score,
        }

    def set_params(self, **params) -> "RandomForestClassifier":
        """Update hyperparameters with validation. Resets fitted state if
        any parameter that affects the trained trees is changed."""
        valid = set(self.get_params().keys())
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")

        # Validate before mutating anything
        merged = {**self.get_params(), **params}
        _validate_common_params(
            merged['n_estimators'], merged['max_depth'],
            merged['min_samples_split'], merged['min_impurity_decrease'],
            merged['random_state'],
        )
        if merged['criterion'] not in ('gini', 'entropy'):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {merged['criterion']!r}")

        # Any change to a training-affecting param invalidates fitted state
        training_params = {
            'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
            'min_impurity_decrease', 'max_features', 'random_state',
        }
        if params.keys() & training_params:
            self._trees = []
            self.n_features_in_ = None
            self.classes_ = None
            self.oob_score_ = None

        for k, v in params.items():
            setattr(self, k, v)
        return self

    def parameters(self) -> dict:
        self._check_is_fitted()
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'n_features_in': self.n_features_in_,
            'classes': self.classes_,
            'oob_score': self.oob_score_,
        }

    def _check_is_fitted(self) -> None:
        if not self._trees:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() before predict()."
            )

    def __repr__(self) -> str:
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"criterion={self.criterion!r}, max_depth={self.max_depth}, "
            f"max_features={self.max_features!r}, random_state={self.random_state})"
        )


# RandomForestRegressor
class RandomForestRegressor(MLModels):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Union[str, int, float, None] = None,
        random_state: Optional[int] = None,
        oob_score: bool = False,
    ):
        _validate_common_params(n_estimators, max_depth, min_samples_split, min_impurity_decrease, random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.oob_score = oob_score

        self._trees: List[_TreeRecord] = []
        self.n_features_in_: Optional[int] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: Matrix, y: Vector) -> "RandomForestRegressor":
        self._validate_Xy(X, y)
        X_data = _matrix_to_list(X)
        y_data = _vector_to_list(y)
        n_samples = len(X_data)
        n_features = len(X_data[0])
        if n_features == 0:
            raise ValueError("Cannot fit on data with 0 features.")

        self.n_features_in_ = n_features

        max_feat = _resolve_max_features(self.max_features, n_features, 'regression')
        base_seed = self.random_state if self.random_state is not None \
            else random.randint(0, 2 ** 31 - 1)

        self._trees = []
        oob_sums = [0.0] * n_samples
        oob_counts = [0] * n_samples
        any_oob = False

        for t in range(self.n_estimators):
            tree_seed = base_seed + 2 * t
            X_boot, y_boot, oob_idx = _bootstrap_sample(X_data, y_data, seed=tree_seed)
            feat_idx = _sample_features(n_features, max_feat, seed=tree_seed + 1)
            X_proj = _project_X(X_boot, feat_idx)

            root = _build(
                X_proj, y_boot,
                'variance', _leaf_value_regressor,
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            record = _TreeRecord(root, feat_idx)
            self._trees.append(record)

            if self.oob_score and oob_idx:
                any_oob = True
                for i in oob_idx:
                    oob_sums[i] += record.predict_one(X_data[i])
                    oob_counts[i] += 1

        if self.oob_score:
            self.oob_score_ = self._compute_oob_reg(oob_sums, oob_counts, y_data, n_samples, any_oob)

        return self

    def _compute_oob_reg(self, oob_sums, oob_counts, y_data, n_samples, any_oob) -> Optional[float]:
        if not any_oob:
            warnings.warn(
                "No OOB samples found. Increase n_estimators or dataset size.",
                UserWarning,
            )
            return None

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        X_data = _matrix_to_list(X)
        _check_n_features(X_data, self.n_features_in_, type(self).__name__)
        preds = []
        for row in X_data:
            tree_preds = [tree.predict_one(row) for tree in self._trees]
            preds.append(sum(tree_preds) / len(tree_preds))
        return Vector(preds)

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return r2_score(y, self.predict(X))

    def get_params(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'oob_score': self.oob_score,
        }

    def set_params(self, **params) -> "RandomForestRegressor":
        # Update hyperparameters with validation. 
        valid = set(self.get_params().keys())
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")

        merged = {**self.get_params(), **params}
        _validate_common_params(
            merged['n_estimators'], merged['max_depth'],
            merged['min_samples_split'], merged['min_impurity_decrease'],
            merged['random_state'],
        )

        training_params = {
            'n_estimators', 'max_depth', 'min_samples_split',
            'min_impurity_decrease', 'max_features', 'random_state',
        }
        if params.keys() & training_params:
            self._trees = []
            self.n_features_in_ = None
            self.oob_score_ = None

        for k, v in params.items():
            setattr(self, k, v)
        return self

    def parameters(self) -> dict:
        self._check_is_fitted()
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'n_features_in': self.n_features_in_,
            'oob_score': self.oob_score_,
        }
