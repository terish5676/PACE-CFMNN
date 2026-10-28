"""Box-directed counterfactuals, verified on the model (blueprint 5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN


@dataclass
class Counterfactual:
    found: bool
    x_cf: np.ndarray | None
    target: object               # target label
    box: int                     # target box used (-1 if none)
    l1: float = np.nan           # L1 distance from x
    n_changed: int = 0           # features changed
    plausibility: float = np.nan  # L1 distance to nearest training point of the target class
    model_calls: int = 0


def _sparsify(model: CFMNN, x, c, t):
    """Greedily reset changed features to x (smallest change first) while the
    prediction stays t. Every kept step is verified on the model."""
    c = c.copy()
    calls = 0
    changed = np.flatnonzero(~np.isclose(c, x))
    for i in changed[np.argsort(np.abs(c - x)[changed])]:
        trial = c.copy()
        trial[i] = x[i]
        calls += 1
        if model.predict_details(trial[None]).cls[0] == t:
            c = trial
    return c, calls


def counterfactual(model: CFMNN, x, target, locked=None, X_train=None, y_train=None,
                   max_boxes: int = 10, steps=(0.25, 0.5, 0.75, 1.0),
                   sparse: bool = True) -> Counterfactual:
    """Smallest move of x into a box of the target class that the model confirms.

    1. Rank target-class boxes by L1 distance on mutable features, skipping boxes
       whose range excludes a locked feature's current value.
    2. Clip x into the box on mutable features and re-run the model.
    3. If it loses a tie, move towards that box's centroid (staying in the box).
    """
    x = np.asarray(x, float).ravel()
    d = len(x)
    locked = np.zeros(d, bool) if locked is None else np.asarray(locked, bool)
    mut = ~locked
    t = int(np.flatnonzero(model.classes_ == target)[0])
    boxes = model.class_boxes_[t]
    V, W = model.V_[boxes], model.W_[boxes]
    ok = np.all((V[:, locked] <= x[locked]) & (x[locked] <= W[:, locked]), axis=1)
    clipped = np.clip(x, V, W)
    clipped[:, locked] = x[locked]
    dist = np.abs(clipped - x)[:, mut].sum(axis=1)
    calls = 0
    for k in np.argsort(np.where(ok, dist, np.inf), kind="stable")[:max_boxes]:
        if not ok[k]:
            break
        j = boxes[k]
        cand = clipped[k].copy()
        candidates = [cand] + [np.where(mut, np.clip(cand + s * (model.centroid_[j] - cand),
                                                     model.V_[j], model.W_[j]), x)
                               for s in steps]
        for c in candidates:
            calls += 1
            if model.predict_details(c[None]).cls[0] == t:
                if sparse:
                    c, extra = _sparsify(model, x, c, t)
                    calls += extra
                cf = Counterfactual(True, c, target, int(j), float(np.abs(c - x).sum()),
                                    int(np.sum(~np.isclose(c, x))), model_calls=calls)
                if X_train is not None and y_train is not None:
                    pts = np.asarray(X_train)[np.asarray(y_train) == target]
                    if len(pts):
                        cf.plausibility = float(np.abs(pts - c).sum(axis=1).min())
                return cf
    return Counterfactual(False, None, target, -1, model_calls=calls)
