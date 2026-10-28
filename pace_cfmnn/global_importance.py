"""Global feature importance (blueprint 5.4)."""

from __future__ import annotations

import numpy as np

from .cfmnn_core import CFMNN
from .explain import contrastive


def data_based(model: CFMNN, X) -> np.ndarray:
    """G_i = mean |e_i(x)| of the contrastive terms over X, normalised to sum 1.

    Membership-mode and tie-mode rows sit on different scales, so each row is
    normalised to unit L1 norm before averaging.
    """
    e = np.abs(contrastive(model, X).e)
    s = e.sum(axis=1, keepdims=True)
    e = np.divide(e, s, out=np.zeros_like(e), where=s > 0)
    g = e.mean(axis=0)
    return g / g.sum() if g.sum() > 0 else g


def model_only(model: CFMNN) -> np.ndarray:
    """How well the classes' boxes separate on each feature, from the boxes alone.

    For every pair of boxes of different classes, the overlap of their intervals
    on feature i (intersection / union; equal point intervals count as full
    overlap) is weighted by Count_j * Count_k. Importance = 1 - weighted overlap.
    """
    V, W, c, n = model.V_, model.W_, model.box_class_, model.count_.astype(float)
    d = V.shape[1]
    num, den = np.zeros(d), 0.0
    for j in range(len(V)):
        other = np.flatnonzero(c > c[j])  # each cross-class pair once
        if len(other) == 0:
            continue
        inter = np.clip(np.minimum(W[j], W[other]) - np.maximum(V[j], V[other]), 0, None)
        union = np.maximum(W[j], W[other]) - np.minimum(V[j], V[other])
        same_point = (union == 0)
        ov = np.where(same_point, 1.0, np.divide(inter, union, out=np.zeros_like(inter),
                                                 where=union > 0))
        w = n[j] * n[other]
        num += (w[:, None] * ov).sum(axis=0)
        den += w.sum()
    return 1.0 - num / den if den > 0 else np.zeros(d)
