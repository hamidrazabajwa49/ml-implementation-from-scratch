import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Vectors.vector import Vector


def _check_lengths(y_true: Vector, y_pred: Vector) -> int:
    if not isinstance(y_true, Vector) or not isinstance(y_pred, Vector):
        raise TypeError("y_true and y_pred must both be Vector instances")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"got {len(y_true)} and {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return len(y_true)


def _check_binary(y: Vector, name: str) -> None:
    for i, v in enumerate(y):
        if v not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"{name} must contain binary values (0 or 1), got {v} at index {i}"
            )


def mse(y_true: Vector, y_pred: Vector) -> float:
    n = _check_lengths(y_true, y_pred)
    total = 0.0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        total += diff * diff
    return total / n


def mae(y_true: Vector, y_pred: Vector) -> float:
    n = _check_lengths(y_true, y_pred)
    total = 0.0
    for i in range(n):
        total += abs(y_true[i] - y_pred[i])
    return total / n


def rmse(y_true: Vector, y_pred: Vector) -> float:
    return math.sqrt(mse(y_true, y_pred))


def r2_score(y_true: Vector, y_pred: Vector) -> float:
    n = _check_lengths(y_true, y_pred)
    mean_y = sum(y_true[i] for i in range(n)) / n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        res = y_true[i] - y_pred[i]
        ss_res += res * res
        dev = y_true[i] - mean_y
        ss_tot += dev * dev
    if ss_tot == 0.0:
        return 0.0 if ss_res == 0.0 else float('-inf')
    return 1.0 - (ss_res / ss_tot)


def accuracy(y_true: Vector, y_pred: Vector) -> float:
    n = _check_lengths(y_true, y_pred)
    correct = sum(1 for i in range(n) if y_true[i] == y_pred[i])
    return correct / n


def precision(y_true: Vector, y_pred: Vector) -> float:
    _check_lengths(y_true, y_pred)
    _check_binary(y_pred, "y_pred")
    tp = fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: Vector, y_pred: Vector) -> float:
    _check_lengths(y_true, y_pred)
    _check_binary(y_pred, "y_pred")
    tp = fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: Vector, y_pred: Vector) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def confusion_matrix(y_true: Vector, y_pred: Vector) -> dict:
    n = _check_lengths(y_true, y_pred)
    _check_binary(y_pred, "y_pred")
    tp = tn = fp = fn = 0
    for i in range(n):
        yt, yp = y_true[i], y_pred[i]
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def roc_auc(y_true: Vector, y_proba: Vector) -> float:
    n = _check_lengths(y_true, y_proba)
    pos = sum(1 for i in range(n) if y_true[i] == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5

    paired = sorted(
        zip([y_proba[i] for i in range(n)], [y_true[i] for i in range(n)]),
        key=lambda x: x[0],
        reverse=True,
    )

    # Batch tied scores together so ties don't inflate AUC
    area = 0.0
    tpr = 0.0
    fpr = 0.0
    prev_score = None
    batch_tp = 0
    batch_fp = 0

    for score, label in paired:
        if prev_score is not None and score != prev_score:
            new_fpr = fpr + batch_fp / neg
            new_tpr = tpr + batch_tp / pos
            area += (new_fpr - fpr) * (tpr + new_tpr) / 2.0
            fpr, tpr = new_fpr, new_tpr
            batch_tp = batch_fp = 0
        if label == 1:
            batch_tp += 1
        else:
            batch_fp += 1
        prev_score = score

    new_fpr = fpr + batch_fp / neg
    new_tpr = tpr + batch_tp / pos
    area += (new_fpr - fpr) * (tpr + new_tpr) / 2.0

    return area
