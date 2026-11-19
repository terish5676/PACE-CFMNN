"""Simpson's fuzzy min-max classifier trained from scratch, on CFMNN's footing (Run M).

Simpson (IEEE Trans. Neural Networks 3(5), 1992) trains in three steps per training row:
expansion, overlap test and contraction. This model does exactly that, one row at a time in
the given order, with CFMNN's membership, size limit and expansion rule, so that the only
difference from ``CFMNN`` is the contraction inside training:

1. **Expansion** (CFMNN's ``expansion="search"``, which is Simpson's rule): if the row already
   has membership 1 in a box of its class (the first such box), nothing changes. Otherwise the
   box of its class with the highest membership that can take the row within the size limit
   ``sum_i (w_i - v_i) <= d * theta`` (CFMNN's sum-of-sides form, not Simpson's per-feature
   bound) is expanded to hold it; with none, a new point box is made.
2. **Overlap test and contraction,** only after an expansion: the expanded box is tested against
   every box of another class, in box order, with Simpson's published test, and each pair that
   overlaps is contracted on the feature of smallest overlap (``contraction.contract_pair``,
   Run F's code, unchanged). Each box is tested on the corners as they stand at its turn.

Membership is CFMNN's (``cfmnn_core.feature_terms``, averaged over the features), with the same
gamma. Simpson's classifier has no centroid: the class is that of the box with the highest
membership, and an exact tie (within ``tie_tol``) goes to the box made first (the lowest box
index). To keep ``CFMNN.predict_details`` and every tool built on it (``PaceShap``,
``xai_baselines.exact_shap``, ``ties``) unchanged, every centroid is set to the same point (the
origin), so the nearest-centroid step of a tie is always a tie and the lowest box index wins.
``ties.centroid_decided`` then counts the predictions this tie rule decides.

``members_`` keeps the rows each box took when they joined; a later contraction may leave a
member outside its box, and nothing is recomputed from members.
"""

from __future__ import annotations

import numpy as np

from .cfmnn_core import CFMNN, feature_terms
from .contraction import _candidates, contract_pair


class SimpsonFMM(CFMNN):
    """Simpson's FMM trained from scratch: CFMNN's expansion, then Simpson's overlap test and
    contraction after every expansion. Ties go to the lowest box index (no centroid)."""

    def __init__(self, theta=0.3, gamma=4.0, tie_tol=1e-9, chunk_elems=65_536):
        super().__init__(theta=theta, gamma=gamma, centroid_rule="fixed", expansion="search",
                         tie_tol=tie_tol, chunk_elems=chunk_elems)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X must be 2-D with one label per row")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if X.min() < 0.0 or X.max() > 1.0:
            raise ValueError("X must be min-max scaled to [0, 1] before fit")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        gamma, limit = float(self.gamma), d * float(self.theta)
        cap = 64
        V, W = np.empty((cap, d)), np.empty((cap, d))
        bc = np.empty(cap, dtype=np.int64)
        S, N = np.zeros((cap, d)), np.zeros(cap, dtype=np.int64)
        members: list[list[int]] = []
        nb = 0
        stats = {"n_created": 0, "n_expanded": 0, "n_inside": 0, "n_contractions": 0,
                 "case_1": 0, "case_2": 0, "case_3": 0, "case_4": 0}

        for r in range(n):
            a, c = X[r], y_idx[r]
            same = np.flatnonzero(bc[:nb] == c)
            if len(same):
                b = feature_terms(a, V[same], W[same], gamma).mean(axis=1)
                k = int(np.argmax(b))
                if b[k] >= 1.0:
                    j = same[k]
                    members[j].append(r)
                    S[j] += a
                    N[j] += 1
                    stats["n_inside"] += 1
                    continue
                ok = (np.maximum(W[same], a) - np.minimum(V[same], a)).sum(axis=1) <= limit
                if ok.any():
                    j = int(same[int(np.argmax(np.where(ok, b, -np.inf)))])
                    np.minimum(V[j], a, out=V[j])
                    np.maximum(W[j], a, out=W[j])
                    members[j].append(r)
                    S[j] += a
                    N[j] += 1
                    stats["n_expanded"] += 1
                    self._contract_against_others(V[:nb], W[:nb], bc[:nb], j, stats)
                    continue
            if nb == cap:                                    # grow the arrays
                cap *= 2
                V, W, S = (np.vstack([M, np.empty((cap - nb, d))]) for M in (V, W, S))
                bc = np.concatenate([bc, np.empty(cap - nb, dtype=np.int64)])
                N = np.concatenate([N, np.zeros(cap - nb, dtype=np.int64)])
            V[nb], W[nb], bc[nb], S[nb], N[nb] = a, a, c, a, 1
            members.append([r])
            nb += 1
            stats["n_created"] += 1

        self.V_, self.W_ = V[:nb].copy(), W[:nb].copy()
        self.box_class_ = bc[:nb].copy()
        self.sum_, self.count_ = S[:nb].copy(), N[:nb].copy()
        self.centroid_ = np.zeros((nb, d))                   # no centroid: ties -> lowest box index
        self.members_ = [np.asarray(m, dtype=np.int64) for m in members]
        self.n_features_in_ = d
        self.fit_stats_ = {"n_train": n, "n_boxes": nb, **stats}
        self._index_classes()
        return self

    @staticmethod
    def _contract_against_others(V, W, bc, j: int, stats: dict) -> None:
        """Simpson's overlap test of box j against every box of another class, in box order,
        each on the corners as they stand at its turn; contract each pair that overlaps."""
        ks = np.flatnonzero(bc != bc[j])
        pos = 0
        while pos < len(ks):
            hit = _candidates(V, W, j, ks[pos:])            # those that overlap j right now
            if len(hit) == 0:
                return
            k = int(hit[0])
            done = contract_pair(V, W, j, k)
            if done is not None:
                stats["n_contractions"] += 1
                stats[f"case_{done['case']}"] += 1
            pos = int(np.searchsorted(ks, k)) + 1

    @staticmethod
    def overlapping_pairs(model) -> int:
        """Pairs of boxes of different classes that Simpson's test says overlap (any model).

        After this model's fit the count can be above 0: a box made as a new point is not
        tested (Simpson 1992; hyperbox-brain's ``fmnn.py`` does the same), and a later
        contraction can change which features a pair shares."""
        V, W, bc = model.V_, model.W_, model.box_class_
        total = 0
        for j in range(len(V) - 1):
            ks = np.arange(j + 1, len(V))
            ks = ks[bc[ks] != bc[j]]
            if len(ks):
                total += len(_candidates(V, W, j, ks))
        return int(total)

    # Refinement edits assume centroids and members; they are not defined for this model.
    def add_points(self, X, y, rows=None):
        raise NotImplementedError("SimpsonFMM is trained by fit only")
