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

def _validate_X_matrix(X, n_features: Optional[int] = None) -> None:
    if not isinstance(X, Matrix):
        raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
    if X.n_rows == 0:
        raise ValueError("X must not be empty (0 rows)")
    if X.n_cols == 0:
        raise ValueError("X must not be empty (0 feature columns)")
    if n_features is not None and X.n_cols != n_features:
        raise ValueError(
            f"Expected {n_features} feature column(s), got {X.n_cols}. "
            f"X passed to predict()/predict_proba() must have the same "
            f"number of columns as the X used in fit()."
        )
    _validate_finite_matrix(X)


def _validate_finite_matrix(X: "Matrix") -> None:
    for i in range(X.n_rows):
        row = X.rows[i].components
        if len(row) != X.n_cols:
            raise ValueError(
                f"Row {i} has {len(row)} columns, expected {X.n_cols} "
                f"(jagged/ragged matrix)."
            )
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TypeError(
                    f"X[{i}][{j}] must be numeric, got {type(v).__name__}"
                )
            if not math.isfinite(v):
                raise ValueError(
                    f"X[{i}][{j}] is {v!r}; NaN/Inf feature values are not "
                    f"supported -- clean or impute your data first."
                )
                

def _validate_y_vector(y: "Vector") -> None:
    for i in range(len(y)):
        lbl = y[i]
        if isinstance(lbl, float) and not math.isfinite(lbl):
            raise ValueError(f"y[{i}] is {lbl!r}; class labels must be finite.")



# GaussianNB
class GaussianNB(MLModels):

    # Hard floor on variance regardless of var_smoothing
    _MIN_VARIANCE = 1e-12

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        priors: Optional[Dict[float, float]] = None,
    ):
        if not isinstance(var_smoothing, (int, float)) or isinstance(var_smoothing, bool):
            raise TypeError(f"var_smoothing must be numeric, got {type(var_smoothing).__name__}")
        if not math.isfinite(var_smoothing):
            raise ValueError(f"var_smoothing must be finite, got {var_smoothing}")
        if var_smoothing < 0:
            raise ValueError(f"var_smoothing must be non-negative, got {var_smoothing}")
        self.var_smoothing = var_smoothing
        self.priors = priors
        self._classes: Optional[List[float]] = None
        self._log_priors: Optional[Dict[float, float]] = None
        self._distributions: Optional[Dict[float, List[NormalDistribution]]] = None
        self._class_counts: Optional[Dict[float, int]] = None
        self._n_features: Optional[int] = None
        # Running (Welford) sufficient statistics, used by partial_fit().
        self._means: Optional[Dict[float, List[float]]] = None
        self._M2: Optional[Dict[float, List[float]]] = None

    def fit(self, X: Matrix, y: Vector) -> "GaussianNB":
        self._validate_Xy(X, y)
        _validate_finite_matrix(X)
        _validate_y_vector(y)

        n_samples = X.n_rows
        n_features = X.n_cols
        labels = [y[i] for i in range(len(y))]

        self._classes = sorted(set(labels))
        self._n_features = n_features
        self._log_priors = {}
        self._distributions = {}
        self._class_counts = {}
        # Reset incremental state
        self._means = {}
        self._M2 = {}

        if self.priors is not None:
            _validate_priors(self.priors, self._classes)

        for c in self._classes:
            indices = [i for i, lbl in enumerate(labels) if lbl == c]
            n_c = len(indices)
            self._class_counts[c] = n_c

            if self.priors is not None:
                self._log_priors[c] = _safe_log(self.priors[c])
            else:
                self._log_priors[c] = _safe_log(n_c / n_samples)

            dists: List[NormalDistribution] = []
            means_c: List[float] = []
            m2_c: List[float] = []
            for j in range(n_features):
                col_values = [X.rows[i].components[j] for i in indices]
                stats = DescriptiveStats(col_values)
                mu = stats.mean()
                raw_var = stats.variance(ddof=0)
                var = max(raw_var + self.var_smoothing, self._MIN_VARIANCE)
                dists.append(NormalDistribution(mu=mu, sigma=math.sqrt(var)))
                means_c.append(mu)
                m2_c.append(raw_var * n_c)  # store as M2 = n * population_variance

            self._distributions[c] = dists
            self._means[c] = means_c
            self._M2[c] = m2_c

        return self

    def partial_fit(self, X: Matrix, y: Vector) -> "GaussianNB":
        """
        Incremental fit on a new batch. 
        Updates sufficient statistics (mean, variance, count) using Welford's online algorithm so the full dataset never needs to be stored.
        """
        self._validate_Xy(X, y)
        _validate_finite_matrix(X)
        _validate_y_vector(y)

        n_features = X.n_cols
        labels = [y[i] for i in range(len(y))]
        new_classes = sorted(set(labels))

        if self._classes is None:
            # First call ever.
            self._classes = []
            self._log_priors = {}
            self._distributions = {}
            self._class_counts = {}
            self._means = {}
            self._M2 = {}
            self._n_features = n_features
        elif self._n_features is None:
            self._n_features = n_features

        if n_features != self._n_features:
            raise ValueError(
                f"partial_fit expects {self._n_features} feature column(s), got {n_features}"
            )

        # If fit() (not partial_fit) populated the classes/distributions but never initialised the running Welford accumulators
        if self._means is None or self._M2 is None:
            self._means = {}
            self._M2 = {}
        for c in self._classes:
            if c not in self._means:
                if self._distributions is not None and c in self._distributions:
                    self._means[c] = [d.mu for d in self._distributions[c]]
                    n_c = self._class_counts.get(c, 0)
                    self._M2[c] = [
                        max(d.sigma ** 2 - self.var_smoothing, 0.0) * n_c
                        for d in self._distributions[c]
                    ]
                else:
                    self._means[c] = [0.0] * n_features
                    self._M2[c] = [0.0] * n_features

        # merge any new classes seen in this batch
        for c in new_classes:
            if c not in self._classes:
                self._classes = sorted(set(self._classes) | {c})
                self._class_counts[c] = 0
                self._means[c] = [0.0] * n_features
                self._M2[c] = [0.0] * n_features

        # Welford update
        for i, lbl in enumerate(labels):
            self._class_counts[lbl] = self._class_counts.get(lbl, 0) + 1
            n = self._class_counts[lbl]
            for j in range(n_features):
                x_val = X.rows[i].components[j]
                delta = x_val - self._means[lbl][j]
                self._means[lbl][j] += delta / n
                delta2 = x_val - self._means[lbl][j]
                self._M2[lbl][j] += delta * delta2

        # rebuild NormalDistribution objects and log_priors
        total = sum(self._class_counts.values())
        if total == 0:
            raise RuntimeError("partial_fit received no samples to accumulate.")

        if self.priors is not None:
            _validate_priors(self.priors, self._classes)

        for c in self._classes:
            n_c = self._class_counts[c]
            if self.priors is not None:
                self._log_priors[c] = _safe_log(self.priors[c])
            else:
                self._log_priors[c] = _safe_log(n_c / total) if n_c > 0 else _safe_log(0.0)
            dists = []
            for j in range(n_features):
                mu = self._means[c][j]
                raw_var = (self._M2[c][j] / n_c) if n_c > 1 else 0.0
                var = max(raw_var + self.var_smoothing, self._MIN_VARIANCE)
                dists.append(NormalDistribution(mu=mu, sigma=math.sqrt(var)))
            self._distributions[c] = dists

        return self

    def _log_posterior(self, x: Vector) -> Dict[float, float]:
        scores: Dict[float, float] = {}
        for c in self._classes:
            log_p = self._log_priors[c]
            for j, dist in enumerate(self._distributions[c]):
                pdf_val = dist.pdf(x.components[j])
                log_p += _safe_log(pdf_val)
            scores[c] = log_p
        return scores

    def predict(self, X: Matrix) -> Vector:
        self._check_is_fitted()
        _validate_X_matrix(X, n_features=self._n_features)
        preds = []
        for i in range(X.n_rows):
            scores = self._log_posterior(X.rows[i])
            preds.append(max(scores, key=lambda c: scores[c]))
        return Vector(preds)

    def predict_proba(self, X: Matrix) -> List[Dict[float, float]]:
        # Returns normalised class probabilities for each sample via log-sum-exp.Each dict sums to 1.0.
        self._check_is_fitted()
        _validate_X_matrix(X, n_features=self._n_features)
        return [
            _log_sum_exp_proba(self._log_posterior(X.rows[i]))
            for i in range(X.n_rows)
        ]

    def predict_log_proba(self, X: Matrix) -> List[Dict[float, float]]:
        """Returns raw unnormalised log-posteriors (faster, no exp needed)."""
        self._check_is_fitted()
        _validate_X_matrix(X, n_features=self._n_features)
        return [self._log_posterior(X.rows[i]) for i in range(X.n_rows)]

    def score(self, X: Matrix, y: Vector) -> float:
        self._validate_Xy(X, y)
        return accuracy_score(y, self.predict(X))

    def parameters(self) -> dict:
        self._check_is_fitted()
        return {
            "classes": self._classes,
            "class_counts": self._class_counts,
            "log_priors": self._log_priors,
            "theta": {
                c: [(d.mu, d.sigma) for d in dists]
                for c, dists in self._distributions.items()
            },
            "var_smoothing": self.var_smoothing,
            "n_features": self._n_features,
        }

    def _check_is_fitted(self) -> None:
        if self._classes is None or not self._classes:
            raise RuntimeError(
                "GaussianNB is not fitted. Call fit() before predict() or score()."
            )



# BernoulliNB
class BernoulliNB(MLModels):

    def __init__(
        self,
        alpha: float = 1.0,
        binarise_threshold: Optional[float] = 0.0,
        priors: Optional[Dict[float, float]] = None,
    ):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")
        if not math.isfinite(alpha):
            raise ValueError(f"alpha must be finite, got {alpha}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if binarise_threshold is not None and not isinstance(binarise_threshold, (int, float)):
            raise TypeError(
                f"binarise_threshold must be numeric or None, got {type(binarise_threshold).__name__}"
            )
        self.alpha = alpha
        self.binarise_threshold = binarise_threshold
        self.priors = priors
        self._classes: Optional[List[float]] = None
        self._log_priors: Optional[Dict[float, float]] = None
        self._log_p: Optional[Dict[float, List[float]]] = None       # log P(x_j=1|c)
        self._log_1mp: Optional[Dict[float, List[float]]] = None     # log P(x_j=0|c)
        self._class_counts: Optional[Dict[float, int]] = None
        self._feature_counts: Optional[Dict[float, List[float]]] = None
        self._n_features: Optional[int] = None

    def _binarise(self, X: Matrix) -> Matrix:
        if self.binarise_threshold is None:
            return X
        t = self.binarise_threshold
        new_rows = []
        for i in range(X.n_rows):
            new_rows.append(
                [1.0 if v > t else 0.0 for v in X.rows[i].components]
            )
        return Matrix(new_rows)


    def _compute_log_likelihoods(self) -> None:
        self._log_p = {}
        self._log_1mp = {}
        for c in self._classes:
            n_c = self._class_counts[c]
            log_p_c = []
            log_1mp_c = []
            for j in range(self._n_features):
                denom = n_c + 2 * self.alpha
                if denom <= 0:
                    raise ValueError(
                        f"Cannot fit BernoulliNB: class {c!r} has 0 samples "
                        f"and alpha={self.alpha}, making p_cj = 0/0 for "
                        f"feature {j}. Increase alpha or remove the empty class."
                    )
                p_cj = (self._feature_counts[c][j] + self.alpha) / denom
                if p_cj <= 0.0 or p_cj >= 1.0:
                    raise ValueError(
                        f"Cannot fit BernoulliNB: feature {j} is constant "
                        f"(always {'present' if p_cj >= 1.0 else 'absent'}) "
                        f"within class {c!r} and alpha=0.0, making "
                        f"{'log(1-p)' if p_cj >= 1.0 else 'log(p)'} undefined. "
                        f"Increase alpha above 0 to apply Laplace smoothing."
                    )
                log_p_c.append(_safe_log(p_cj))
                log_1mp_c.append(_safe_log(1.0 - p_cj))
            self._log_p[c] = log_p_c
            self._log_1mp[c] = log_1mp_c

    def fit(self, X: Matrix, y: Vector) -> "BernoulliNB":
        self._validate_Xy(X, y)
        _validate_finite_matrix(X)
        _validate_y_vector(y)
        X = self._binarise(X)

        n_samples = X.n_rows
        self._n_features = X.n_cols
        labels = [y[i] for i in range(len(y))]

        self._classes = sorted(set(labels))
        self._log_priors = {}
        self._class_counts = {}
        self._feature_counts = {}

        if self.priors is not None:
            _validate_priors(self.priors, self._classes)

        for c in self._classes:
            indices = [i for i, lbl in enumerate(labels) if lbl == c]
            n_c = len(indices)
            self._class_counts[c] = n_c

            if self.priors is not None:
                self._log_priors[c] = _safe_log(self.priors[c])
            else:
                self._log_priors[c] = _safe_log(n_c / n_samples)

            counts = [0.0] * self._n_features
            for i in indices:
                for j in range(self._n_features):
                    counts[j] += X.rows[i].components[j]
            self._feature_counts[c] = counts

        self._compute_log_likelihoods()
        return self

    def partial_fit(self, X: Matrix, y: Vector) -> "BernoulliNB":
        """Incremental update. Accumulates binary feature counts."""
        self._validate_Xy(X, y)
        _validate_finite_matrix(X)
        _validate_y_vector(y)
        X = self._binarise(X)

        n_features = X.n_cols
        labels = [y[i] for i in range(len(y))]

        if self._classes is None:
            self._classes = []
            self._log_priors = {}
            self._class_counts = {}
            self._feature_counts = {}
            self._n_features = n_features

        if n_features != self._n_features:
            raise ValueError(
                f"partial_fit expects {self._n_features} features, got {n_features}"
            )
