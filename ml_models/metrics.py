# ml_models/metrics.py
"""
Evaluation metrics shared across the ml_models package.

Provides regression metrics (MSE, MAE, RMSE, R-squared) and binary
classification metrics (accuracy, precision, recall, F1, confusion
matrix, ROC-AUC), all operating on `Vector` instances from the
math_foundations library.

The classification metrics in this module (`precision`, `recall`,
`f1_score`, `confusion_matrix`, `roc_auc`) assume binary labels encoded
as 0/1. Multi-class variants are out of scope for this module and are
expected to be provided per-algorithm where needed (e.g. a
one-vs-rest wrapper in a multi-class classifier's own module).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
_shared_parent = os.path.abspath(os.path.join(_script_dir, os.pardir))
_target_root = os.path.join(_shared_parent, "math_foundations")

if _target_root not in sys.path:
    sys.path.insert(0, _target_root)

from Vectors.vector import Vector  

_BINARY_VALUES = (0, 1, 0.0, 1.0)


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the given inputs."""


def _check_lengths(y_true: Vector, y_pred: Vector) -> int:
    """Validate that `y_true` and `y_pred` are equal-length, non-empty Vectors.

    Parameters
    ----------
    y_true : Vector
        Ground-truth values.
    y_pred : Vector
        Predicted values.

    Returns
    -------
    int
        The shared length of both vectors.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the lengths differ or both are empty.
    """
    if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
        raise TypeError("y_true and y_pred must both be Vector instances.")
    if len(y_true) != len(y_pred):
        raise MetricError(
            "y_true and y_pred must have the same length: "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise MetricError("y_true and y_pred must not be empty.")
    return len(y_true)


def _check_binary(y: Vector, name: str) -> None:
    """Validate that every element of `y` is a binary label (0 or 1).

    Parameters
    ----------
    y : Vector
        Vector of labels to validate.
    name : str
        Human-readable name of the argument, used in the error message.

    Raises
    ------
    MetricError
        If any element is not in {0, 1, 0.0, 1.0}.
    """
    for i, value in enumerate(y):
        if value not in _BINARY_VALUES:
            raise MetricError(
                f"{name} must contain only binary values (0 or 1); "
                f"found {value!r} at index {i}."
            )


def mse(y_true: Vector, y_pred: Vector) -> float:
    """Compute the mean squared error.

    ``MSE = (1/n) * sum((y_true_i - y_pred_i) ** 2)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth target values.
    y_pred : Vector
        Predicted target values.

    Returns
    -------
    float
        The mean squared error; always non-negative.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched or zero length.
    """
    n = _check_lengths(y_true, y_pred)
    total = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    return total / n


def mae(y_true: Vector, y_pred: Vector) -> float:
    """Compute the mean absolute error.

    ``MAE = (1/n) * sum(|y_true_i - y_pred_i|)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth target values.
    y_pred : Vector
        Predicted target values.

    Returns
    -------
    float
        The mean absolute error; always non-negative.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched or zero length.
    """
    n = _check_lengths(y_true, y_pred)
    total = sum(abs(y_true[i] - y_pred[i]) for i in range(n))
    return total / n


def rmse(y_true: Vector, y_pred: Vector) -> float:
    """Compute the root mean squared error.

    ``RMSE = sqrt(MSE)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth target values.
    y_pred : Vector
        Predicted target values.

    Returns
    -------
    float
        The root mean squared error; always non-negative.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched or zero length.
    """
    return math.sqrt(mse(y_true, y_pred))


def r2_score(y_true: Vector, y_pred: Vector) -> float:
    """Compute the coefficient of determination (R-squared).

    ``R^2 = 1 - SS_res / SS_tot``

    Parameters
    ----------
    y_true : Vector
        Ground-truth target values.
    y_pred : Vector
        Predicted target values.

    Returns
    -------
    float
        The R-squared score. A value of 1.0 indicates a perfect fit;
        0.0 indicates performance equal to always predicting the mean
        of `y_true`. If `y_true` is constant (``SS_tot == 0``), returns
        1.0 for a perfect prediction and negative infinity otherwise,
        since R-squared is mathematically undefined (division by zero)
        in that degenerate case.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched or zero length.
    """
    n = _check_lengths(y_true, y_pred)
    mean_y = sum(y_true[i] for i in range(n)) / n

    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        residual = y_true[i] - y_pred[i]
        ss_res += residual ** 2
        deviation = y_true[i] - mean_y
        ss_tot += deviation ** 2

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - (ss_res / ss_tot)


def accuracy(y_true: Vector, y_pred: Vector) -> float:
    """Compute classification accuracy.

    ``accuracy = (# correct predictions) / n``

    Works for both binary and multi-class labels, since it relies only
    on element-wise equality rather than a binary-specific formula.

    Parameters
    ----------
    y_true : Vector
        Ground-truth labels.
    y_pred : Vector
        Predicted labels.

    Returns
    -------
    float
        A value in [0, 1].

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched or zero length.
    """
    n = _check_lengths(y_true, y_pred)
    correct = sum(1 for i in range(n) if y_true[i] == y_pred[i])
    return correct / n


def precision(y_true: Vector, y_pred: Vector) -> float:
    """Compute binary classification precision.

    ``precision = TP / (TP + FP)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth binary labels (0 or 1).
    y_pred : Vector
        Predicted binary labels (0 or 1).

    Returns
    -------
    float
        A value in [0, 1]. Returns 0.0 if no positive predictions were
        made (``TP + FP == 0``), since precision is conventionally
        undefined but reported as 0 in that case rather than raising.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched/zero length, or `y_pred`
        contains non-binary values.
    """
    n = _check_lengths(y_true, y_pred)
    _check_binary(y_pred, "y_pred")
    true_positive = false_positive = 0
    for i in range(n):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator > 0 else 0.0


def recall(y_true: Vector, y_pred: Vector) -> float:
    """Compute binary classification recall (sensitivity, true positive rate).

    ``recall = TP / (TP + FN)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth binary labels (0 or 1).
    y_pred : Vector
        Predicted binary labels (0 or 1).

    Returns
    -------
    float
        A value in [0, 1]. Returns 0.0 if there are no actual
        positives (``TP + FN == 0``), since recall is conventionally
        undefined but reported as 0 in that case rather than raising.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched/zero length, or `y_pred`
        contains non-binary values.
    """
    n = _check_lengths(y_true, y_pred)
    _check_binary(y_pred, "y_pred")
    true_positive = false_negative = 0
    for i in range(n):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator > 0 else 0.0


def f1_score(y_true: Vector, y_pred: Vector) -> float:
    """Compute the binary classification F1 score.

    ``F1 = 2 * precision * recall / (precision + recall)``

    Parameters
    ----------
    y_true : Vector
        Ground-truth binary labels (0 or 1).
    y_pred : Vector
        Predicted binary labels (0 or 1).

    Returns
    -------
    float
        The harmonic mean of precision and recall, in [0, 1]. Returns
        0.0 if precision and recall are both 0.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched/zero length, or `y_pred`
        contains non-binary values.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def confusion_matrix(y_true: Vector, y_pred: Vector) -> Dict[str, int]:
    """Compute the binary confusion matrix counts.

    Parameters
    ----------
    y_true : Vector
        Ground-truth binary labels (0 or 1).
    y_pred : Vector
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict
        A dictionary with keys ``"TP"``, ``"TN"``, ``"FP"``, ``"FN"``.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched/zero length, or either contains
        non-binary values.
    """
    n = _check_lengths(y_true, y_pred)
    _check_binary(y_true, "y_true")
    _check_binary(y_pred, "y_pred")

    true_positive = true_negative = false_positive = false_negative = 0
    for i in range(n):
        actual, predicted = y_true[i], y_pred[i]
        if actual == 1 and predicted == 1:
            true_positive += 1
        elif actual == 0 and predicted == 0:
            true_negative += 1
        elif actual == 0 and predicted == 1:
            false_positive += 1
        else:
            false_negative += 1

    return {"TP": true_positive, "TN": true_negative, "FP": false_positive, "FN": false_negative}


def roc_auc(y_true: Vector, y_scores: Vector) -> float:
    """Compute the area under the ROC curve for binary labels.

    Uses a rank-based (Mann-Whitney) formulation, batching tied scores
    together so ties do not artificially inflate the computed area.

    Parameters
    ----------
    y_true : Vector
        Ground-truth binary labels (0 or 1).
    y_scores : Vector
        Real-valued scores or probabilities; higher values are
        interpreted as more likely to belong to the positive class.
        Values need not be restricted to [0, 1].

    Returns
    -------
    float
        The AUC, in [0, 1]. Returns 0.5 (equivalent to random
        guessing) if `y_true` contains only one class, since AUC is
        mathematically undefined without both classes present.

    Raises
    ------
    TypeError
        If either argument is not a `Vector`.
    MetricError
        If the vectors have mismatched/zero length, or `y_true`
        contains non-binary values.
    """
    n = _check_lengths(y_true, y_scores)
    _check_binary(y_true, "y_true")

    n_positive = sum(1 for i in range(n) if y_true[i] == 1)
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5

    paired: List[Tuple[float, float]] = sorted(
        ((y_scores[i], y_true[i]) for i in range(n)),
        key=lambda pair: pair[0],
        reverse=True,
    )

    area = 0.0
    true_positive_rate = 0.0
    false_positive_rate = 0.0
    previous_score = None
    batch_true_positive = 0
    batch_false_positive = 0

    for score, label in paired:
        if previous_score is not None and score != previous_score:
            new_fpr = false_positive_rate + batch_false_positive / n_negative
            new_tpr = true_positive_rate + batch_true_positive / n_positive
            area += (new_fpr - false_positive_rate) * (true_positive_rate + new_tpr) / 2.0
            false_positive_rate, true_positive_rate = new_fpr, new_tpr
            batch_true_positive = batch_false_positive = 0
        if label == 1:
            batch_true_positive += 1
        else:
            batch_false_positive += 1
        previous_score = score

    new_fpr = false_positive_rate + batch_false_positive / n_negative
    new_tpr = true_positive_rate + batch_true_positive / n_positive
    area += (new_fpr - false_positive_rate) * (true_positive_rate + new_tpr) / 2.0

    return area


def accuracy_score(y_true: Vector, y_pred: Vector) -> float:
    """Alias for `accuracy`, provided for naming-convention compatibility.

    See Also
    --------
    accuracy : The underlying implementation.
    """
    return accuracy(y_true, y_pred)
