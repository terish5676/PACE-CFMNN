"""Reference CFMNN: centroid-based fuzzy min-max neural network (single model).

Follows Aadarsh & Prasad (PReMI 2023): Algorithm 2 for training and
Algorithm 4 for testing, without MapReduce or the ensemble step.

Two centroid rules are available:

* ``"original"``: the paper as written. A point that is already inside its
  best same-class box (membership 1) is skipped, so Sum and Count are not
  updated (Algorithm 2, lines 9-10).
* ``"fixed"``: every point a box accepts updates Sum and Count, including
  points already inside it. This is the blueprint's default (section 4.6).

The two variants differ only in this rule, so any accuracy change between them
comes from the centroid fix alone.

A second, separate option is the expansion rule. The paper tests expansion on
the best same-class box only (``expansion="best"``). In many dimensions the
best box is usually a large box already at its size limit, so most new points
become point boxes that never grow. ``expansion="search"`` instead expands the
best box that can still expand (Simpson's FMNN). It is not the paper's rule
and is reported as its own variant.

Inputs must be min-max scaled to [0, 1] before ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

CENTROID_RULES = ("fixed", "original")
EXPANSION_RULES = ("best", "search")


def ramp(t, gamma):
    """Simpson's ramp f(t) = max(0, 1 - max(0, gamma * min(1, t)))."""
    return np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, t)))


def feature_terms(x, v, w, gamma):
    """Per-feature membership terms m_i = [f(x_i - w_i) + f(v_i - x_i)] / 2.

    Broadcasts over any leading axes. Each term lies in [0.5, 1]: because
    v <= w, a point can be outside a box on only one side of each feature.
    """
    return 0.5 * (ramp(x - w, gamma) + ramp(v - x, gamma))


@dataclass
class Prediction:
    """Everything CFMNN computes for a batch of points (one row per point).

    Class indices refer to ``model.classes_``. Box indices refer to the rows
    of ``model.V_``.
    """

    y_pred: np.ndarray            # predicted labels, in the original label values
    cls: np.ndarray               # predicted class index A, shape (n,)
    winner: np.ndarray            # winning box j*, shape (n,)
    n_tied: np.ndarray            # boxes tied at the top membership (|WinList|)
    support: np.ndarray           # raw class support s_c, shape (n, C); nan if class has no boxes
    support_rescaled: np.ndarray  # s'_c = 2 s_c - 1, shape (n, C); 0 if class has no boxes
    best_box: np.ndarray          # best box of each class, shape (n, C); -1 if none
    dist2_best: np.ndarray        # squared distance to that box's centroid, (n, C); inf if none
    runner_up: np.ndarray         # runner-up class B, shape (n,); -1 if no other class has boxes

    @property
    def membership(self) -> np.ndarray:
        """Membership of each point in its winning box."""
        return self.support[np.arange(len(self.cls)), self.cls]


class _ClassBuffer:
    """Growable storage for one class's boxes during training."""

    def __init__(self, d: int, capacity: int = 16):
        self.n = 0
        self.V = np.empty((capacity, d))
        self.W = np.empty((capacity, d))
        self.S = np.empty((capacity, d))
        self.N = np.empty(capacity, dtype=np.int64)
        self.order = np.empty(capacity, dtype=np.int64)  # global creation order
        self.members: list[list[int]] = []

    def add(self, a: np.ndarray, row: int, order: int) -> None:
        if self.n == len(self.N):
            cap = 2 * len(self.N)
            for name in ("V", "W", "S"):
                old = getattr(self, name)
                new = np.empty((cap, old.shape[1]))
                new[: self.n] = old[: self.n]
                setattr(self, name, new)
            for name in ("N", "order"):
                old = getattr(self, name)
                new = np.empty(cap, dtype=np.int64)
                new[: self.n] = old[: self.n]
                setattr(self, name, new)
        k = self.n
        self.V[k] = a
        self.W[k] = a
        self.S[k] = a
        self.N[k] = 1
        self.order[k] = order
        self.members.append([row])
        self.n += 1


class CFMNN(ClassifierMixin, BaseEstimator):
    """Centroid-based fuzzy min-max classifier (expansion only, no contraction).

    Parameters
    ----------
    theta : float
        Box size limit: sum over features of (w - v) must stay <= d * theta.
    gamma : float
        Sensitivity of the membership ramp (the paper uses 4).
    centroid_rule : {"fixed", "original"}
        See the module docstring.
    expansion : {"best", "search"}
        "best" (the paper): test expansion on the highest-membership same-class
        box only; if it is too big, create a point box. "search" (Simpson 1992
        style, not in the paper): expand the highest-membership box that can
        still expand within the size limit.
    tie_tol : float
        Memberships within this tolerance of the maximum count as tied; ties
        go to the nearest centroid, then to the lowest box index.
    chunk_elems : int
        Points are processed in chunks of about this many floats per working
        array. The default (about 0.5 MB) keeps the arrays in the CPU cache,
        which is several times faster with many boxes. Results do not depend
        on it.
    """

    def __init__(self, theta=0.3, gamma=4.0, centroid_rule="fixed", expansion="search",
                 tie_tol=1e-9, chunk_elems=65_536):
        self.theta = theta
        self.gamma = gamma
        self.centroid_rule = centroid_rule
        self.expansion = expansion
        self.tie_tol = tie_tol
        self.chunk_elems = chunk_elems

    # ------------------------------------------------------------------ training
    def fit(self, X, y):
        if self.centroid_rule not in CENTROID_RULES:
            raise ValueError(f"centroid_rule must be one of {CENTROID_RULES}")
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
        if self.expansion not in EXPANSION_RULES:
            raise ValueError(f"expansion must be one of {EXPANSION_RULES}")
        fixed = self.centroid_rule == "fixed"
        search = self.expansion == "search"
        bufs = [_ClassBuffer(d) for _ in self.classes_]
        n_created = n_expanded = n_inside = 0
        n_best_tied = n_floor_tied = 0  # ties for the best same-class box (paper is silent)

        for r in range(n):
            a = X[r]
            buf = bufs[y_idx[r]]
            if buf.n == 0:
                buf.add(a, r, n_created)
                n_created += 1
                continue
            V, W = buf.V[: buf.n], buf.W[: buf.n]
            b = feature_terms(a, V, W, gamma).mean(axis=1)
            k = int(np.argmax(b))  # first box on exact ties
            if buf.n > 1 and np.count_nonzero(b == b[k]) > 1:
                n_best_tied += 1
                n_floor_tied += b[k] == 0.5
            if b[k] >= 1.0:
                # Already inside its best box (Algorithm 2, lines 9-10).
                n_inside += 1
                buf.members[k].append(r)
                if fixed:
                    buf.S[k] += a
                    buf.N[k] += 1
                continue
            if search:
                # Simpson-style: the highest-membership box that can still expand.
                ok = (np.maximum(W, a) - np.minimum(V, a)).sum(axis=1) <= limit
                k = int(np.argmax(np.where(ok, b, -np.inf))) if ok.any() else -1
                can_expand = k >= 0
            else:
                # The paper: expansion is tested on the best box only.
                can_expand = np.sum(np.maximum(W[k], a) - np.minimum(V[k], a)) <= limit
            if can_expand:
                np.minimum(V[k], a, out=V[k])
                np.maximum(W[k], a, out=W[k])
                buf.S[k] += a
                buf.N[k] += 1
                buf.members[k].append(r)
                n_expanded += 1
            else:
                buf.add(a, r, n_created)
                n_created += 1

        # Assemble all boxes in global creation order.
        parts = [(c, buf) for c, buf in enumerate(bufs)]
        order = np.concatenate([buf.order[: buf.n] for _, buf in parts])
        perm = np.argsort(order, kind="stable")
        cat = lambda name: np.concatenate([getattr(buf, name)[: buf.n] for _, buf in parts])[perm]
        self.V_ = cat("V")
        self.W_ = cat("W")
        self.sum_ = cat("S")
        self.count_ = cat("N")
        self.box_class_ = np.concatenate(
            [np.full(buf.n, c, dtype=np.int64) for c, buf in parts])[perm]
        members = [np.asarray(m, dtype=np.int64) for _, buf in parts for m in buf.members]
        self.members_ = [members[i] for i in perm]
        self.centroid_ = self.sum_ / self.count_[:, None]
        self.n_features_in_ = d
        self.fit_stats_ = {"n_train": n, "n_boxes": len(self.V_), "n_created": n_created,
                           "n_expanded": n_expanded, "n_inside": n_inside,
                           "n_best_tied": n_best_tied, "n_floor_tied": int(n_floor_tied)}
        self._index_classes()
        return self

    def _index_classes(self) -> None:
        self.class_boxes_ = [np.flatnonzero(self.box_class_ == c)
                             for c in range(len(self.classes_))]

    @property
    def n_boxes_(self) -> int:
        return len(self.V_)

    # ---------------------------------------------------------------- prediction
    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X

    def _chunk_size(self) -> int:
        return max(1, int(self.chunk_elems // max(1, self.n_boxes_ * self.n_features_in_)))

    def memberships(self, X) -> np.ndarray:
        """Membership of every point in every box, shape (n, H)."""
        X = self._check_X(X)
        out = np.empty((len(X), self.n_boxes_))
        step = self._chunk_size()
        for s in range(0, len(X), step):
            Xc = X[s:s + step, None, :]
            out[s:s + step] = feature_terms(Xc, self.V_, self.W_, self.gamma).mean(axis=2)
        return out

    def box_terms(self, X, boxes) -> np.ndarray:
        """Per-feature terms m_{j,i}(x) of box ``boxes[r]`` at point ``X[r]``, shape (n, d)."""
        X = self._check_X(X)
        boxes = np.asarray(boxes)
        return feature_terms(X, self.V_[boxes], self.W_[boxes], self.gamma)

    def predict_details(self, X) -> Prediction:
        """Prediction plus winning box, class supports and runner-up class."""
        X = self._check_X(X)
        n, C = len(X), len(self.classes_)
        tol = float(self.tie_tol)
        cls = np.empty(n, dtype=np.int64)
        winner = np.empty(n, dtype=np.int64)
        n_tied = np.empty(n, dtype=np.int64)
        support = np.full((n, C), np.nan)
        best_box = np.full((n, C), -1, dtype=np.int64)
        dist2 = np.full((n, C), np.inf)

        step = self._chunk_size()
        for s in range(0, n, step):
            Xc = X[s:s + step]
            rows = np.arange(len(Xc))
            M = feature_terms(Xc[:, None, :], self.V_, self.W_, self.gamma).mean(axis=2)
            D = ((Xc[:, None, :] - self.centroid_[None, :, :]) ** 2).sum(axis=2)

            # Algorithm 4: highest membership, ties to the nearest centroid.
            top = M.max(axis=1)
            tied = M >= top[:, None] - tol
            j = np.argmin(np.where(tied, D, np.inf), axis=1)
            winner[s:s + step] = j
            n_tied[s:s + step] = tied.sum(axis=1)
            cls[s:s + step] = self.box_class_[j]

            # Best box of each class, with the same tie rule.
            for c, idx in enumerate(self.class_boxes_):
                if len(idx) == 0:
                    continue
                Mc = M[:, idx]
                top_c = Mc.max(axis=1)
                Dc = np.where(Mc >= top_c[:, None] - tol, D[:, idx], np.inf)
                k = np.argmin(Dc, axis=1)
                support[s:s + step, c] = top_c
                best_box[s:s + step, c] = idx[k]
                dist2[s:s + step, c] = Dc[rows, k]

            # The predicted class's best box is the winning box itself.
            cw = cls[s:s + step]
            best_box[s + rows, cw] = j
            dist2[s + rows, cw] = D[rows, j]

        rescaled = np.where(np.isnan(support), 0.0, 2.0 * support - 1.0)
        runner_up = self._runner_up(cls, rescaled, best_box, dist2, tol)
        return Prediction(self.classes_[cls], cls, winner, n_tied, support, rescaled,
                          best_box, dist2, runner_up)

    @staticmethod
    def _runner_up(cls, rescaled, best_box, dist2, tol) -> np.ndarray:
        """Class B: highest rescaled support other than A; ties to the nearest prototype."""
        n = len(cls)
        rows = np.arange(n)
        s = np.where(best_box >= 0, rescaled, -np.inf)
        s[rows, cls] = -np.inf
        top = s.max(axis=1)
        tied = np.isfinite(s) & (s >= top[:, None] - 2 * tol)
        b = np.argmin(np.where(tied, dist2, np.inf), axis=1)
        return np.where(np.isfinite(top), b, -1)

    def predict(self, X) -> np.ndarray:
        return self.predict_details(X).y_pred
