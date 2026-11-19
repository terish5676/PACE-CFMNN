"""Classification metrics and timing (blueprint section 10).

Later phases add XAI, error-detection and refinement metrics here.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, balanced accuracy and per-class recall (in %)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(y_true)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "macro_f1": 100 * f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0),
        "balanced_accuracy": 100 * balanced_accuracy_score(y_true, y_pred),
        "min_class_recall": 100 * float(recalls.min()),
    }
    out.update({f"recall_{lab}": 100 * float(r) for lab, r in zip(labels, recalls)})
    return out


def majority_baseline(y_train, y_test) -> np.ndarray:
    """Predict the most common training class for every test point."""
    vals, counts = np.unique(y_train, return_counts=True)
    return np.full(len(y_test), vals[np.argmax(counts)])


def error_detection_metrics(is_error, score, budget: float = 0.1) -> dict:
    """How well a score ranks the model's errors first (blueprint 10, RQ2).

    AUROC and AUPRC (average precision; read against the base error rate), and
    precision and recall when the top ``budget`` share of points is flagged.
    Ties at the cut-off are shared out evenly (the expected value under random
    tie-breaking), so the result does not depend on row order.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    err = np.asarray(is_error, dtype=bool)
    s = np.asarray(score, dtype=float)
    if len(err) != len(s) or not np.isfinite(s).all():
        raise ValueError("need one finite score per point")
    n, n_err = len(err), int(err.sum())
    out = {"n": n, "n_errors": n_err, "base_rate": n_err / n if n else np.nan}
    if n_err == 0 or n_err == n:
        return {**out, "auroc": np.nan, "auprc": np.nan, "precision_at": np.nan,
                "recall_at": np.nan}
    k = max(1, int(np.ceil(budget * n)))
    t = np.sort(s)[::-1][k - 1]
    gt, eq = s > t, s == t
    caught = err[gt].sum() + (k - gt.sum()) * err[eq].sum() / eq.sum()
    return {**out, "auroc": roc_auc_score(err, s), "auprc": average_precision_score(err, s),
            "precision_at": caught / k, "recall_at": caught / n_err}


@contextmanager
def timer(store: dict, key: str):
    """Record the wall-clock seconds of a block into ``store[key]``."""
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0
