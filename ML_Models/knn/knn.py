import os
import sys
from typing import List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_models_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(ml_models_dir, '..'))
for p in (root_dir, ml_models_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from Vectors.vector import Vector
from Matrix.matrix import Matrix
from ML_Models.base_class import MLModels
from ML_Models.metrics import accuracy_score, r2_score
from ML_Models.knn.distances import _resolve_metric


class _KNNBase(MLModels):

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        self.k = k
        self._metric_fn = _resolve_metric(metric)
        self._metric_name = metric if isinstance(metric, str) else 'custom'
        self._X_train: Optional[List[Vector]] = None
        self._y_train: Optional[Vector] = None

    def fit(self, X: Matrix, y: Vector) -> None:
        if X.n_rows != len(y):
            raise ValueError(f"X has {X.n_rows} rows but y has {len(y)} elements")
        if X.n_rows == 0:
            raise ValueError("Training set is empty")
        self._X_train = [X.rows[i] for i in range(X.n_rows)]
        self._y_train = y

    def _check_fitted(self) -> None:
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict()")

    def _get_neighbors(self, x: Vector) -> List[tuple]:
        distances = []
        for i, train_x in enumerate(self._X_train):
            d = self._metric_fn(x, train_x)
            distances.append((d, self._y_train[i]))
        distances.sort(key=lambda t: t[0])
        return distances[: self.k]

    def parameters(self):
        return {
            'k': self.k,
            'metric': self._metric_name,
            'n_train': len(self._X_train) if self._X_train else 0,
        }


class KNNClassifier(_KNNBase):

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        super().__init__(k=k, metric=metric)

    def _majority_vote(self, labels: List[float]) -> float:
        counts: dict = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=lambda lbl: counts[lbl])

    def _weighted_vote(self, neighbors: List[tuple]) -> float:
        weights: dict = {}
        for dist, label in neighbors:
            w = 1.0 / (dist ** 2) if dist != 0.0 else float('inf')
            weights[label] = weights.get(label, 0.0) + w
        if any(v == float('inf') for v in weights.values()):
            exact = [lbl for d, lbl in neighbors if d == 0.0]
            return self._majority_vote(exact)
        return max(weights, key=lambda lbl: weights[lbl])

    def predict(self, X: Matrix) -> Vector:
        self._check_fitted()
        preds = []
        for i in range(X.n_rows):
            neighbors = self._get_neighbors(X.rows[i])
            preds.append(self._majority_vote([lbl for _, lbl in neighbors]))
        return Vector(preds)

    def predict_weighted(self, X: Matrix) -> Vector:
        self._check_fitted()
        preds = []
        for i in range(X.n_rows):
            neighbors = self._get_neighbors(X.rows[i])
            preds.append(self._weighted_vote(neighbors))
        return Vector(preds)

    def predict_proba(self, X: Matrix) -> List[dict]:
        self._check_fitted()
        result = []
        for i in range(X.n_rows):
            neighbors = self._get_neighbors(X.rows[i])
            counts: dict = {}
            for _, lbl in neighbors:
                counts[lbl] = counts.get(lbl, 0) + 1
            result.append({lbl: cnt / self.k for lbl, cnt in counts.items()})
        return result

    def score(self, X: Matrix, y: Vector) -> float:
        return accuracy_score(y, self.predict(X))


class KNNRegressor(_KNNBase):

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        super().__init__(k=k, metric=metric)

    def _mean_prediction(self, neighbors: List[tuple]) -> float:
        return sum(lbl for _, lbl in neighbors) / len(neighbors)

    def _weighted_prediction(self, neighbors: List[tuple]) -> float:
        exact = [lbl for d, lbl in neighbors if d == 0.0]
        if exact:
            return sum(exact) / len(exact)
        total_w = 0.0
        weighted_sum = 0.0
        for dist, lbl in neighbors:
            w = 1.0 / (dist ** 2)
            weighted_sum += w * lbl
            total_w += w
        return weighted_sum / total_w

    def predict(self, X: Matrix) -> Vector:
        self._check_fitted()
        preds = []
        for i in range(X.n_rows):
            neighbors = self._get_neighbors(X.rows[i])
            preds.append(self._mean_prediction(neighbors))
        return Vector(preds)

    def predict_weighted(self, X: Matrix) -> Vector:
        self._check_fitted()
        preds = []
        for i in range(X.n_rows):
            neighbors = self._get_neighbors(X.rows[i])
            preds.append(self._weighted_prediction(neighbors))
        return Vector(preds)

    def score(self, X: Matrix, y: Vector) -> float:
        return r2_score(y, self.predict(X))
