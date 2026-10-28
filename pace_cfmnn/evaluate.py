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


@contextmanager
def timer(store: dict, key: str):
    """Record the wall-clock seconds of a block into ``store[key]``."""
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0
