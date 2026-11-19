"""The original per-point, per-feature PACE-flip loop, kept as a reference.

This is the implementation the published faithfulness results were produced
with (``pace_cfmnn.explain.pace_flip`` before it was vectorised). The fast
``PaceFlip`` explainer must return exactly the same arrays, bit for bit.
"""

from __future__ import annotations

import numpy as np

from pace_cfmnn.cfmnn_core import CFMNN, feature_terms


def pace_flip_loop(model: CFMNN, X, background, n_quantiles: int | None = 32,
                   return_parts: bool = False):
    X = model._check_X(X)
    background = model._check_X(background)
    if n_quantiles is not None:
        background = np.quantile(background, (np.arange(n_quantiles) + 0.5) / n_quantiles,
                                 axis=0)
    n, d = X.shape
    V, W, mu, g = model.V_, model.W_, model.centroid_, model.gamma
    tol = float(model.tie_tol)
    r = len(background)
    det = model.predict_details(X)
    groups = [idx for idx in model.class_boxes_ if len(idx)]
    gcls = np.array([c for c, idx in enumerate(model.class_boxes_) if len(idx)])
    flip = np.zeros((n, d))
    drop = np.zeros((n, d))

    def margin(Mz, a):
        s = np.stack([Mz[..., idx].max(axis=-1) for idx in groups], axis=-1)
        s = 2.0 * s - 1.0
        ka = int(np.flatnonzero(gcls == a)[0])
        other = np.delete(s, ka, axis=-1)
        best_other = other.max(axis=-1) if other.shape[-1] else np.zeros(s.shape[:-1])
        return s[..., ka] - best_other

    for p in range(n):
        x = X[p]
        a = det.cls[p]
        T = feature_terms(x, V, W, g)                            # (H, d)
        M = T.mean(axis=1)
        Df = (x - mu) ** 2
        D = Df.sum(axis=1)
        m0 = margin(M, a)
        for i in range(d):
            z = background[:, i][:, None]                        # (r, 1)
            Mz = M + (feature_terms(z, V[:, i], W[:, i], g) - T[:, i]) / d   # (r, H)
            Dz = D + (z - mu[:, i]) ** 2 - Df[:, i]
            top = Mz.max(axis=1, keepdims=True)
            j = np.argmin(np.where(Mz >= top - tol, Dz, np.inf), axis=1)
            flip[p, i] = np.mean(model.box_class_[j] != a)
            drop[p, i] = m0 - margin(Mz, a).mean()
    lo = drop.min(axis=1, keepdims=True)
    hi = drop.max(axis=1, keepdims=True)
    norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
    score = flip + norm * (0.5 / r)
    return (score, flip, drop) if return_parts else score
