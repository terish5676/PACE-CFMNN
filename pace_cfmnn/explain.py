"""PACE local explanations (blueprint 5.1 and 5.2).

Both reuse CFMNN's own per-feature membership terms, so they are exact for the
model's membership score and never change a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN, feature_terms


@dataclass
class Decomposition:
    phi: np.ndarray          # (n, d) centred contributions; sum_i phi_i = membership - base
    base: np.ndarray         # (n,) E_background[b_{j*}(X)]
    membership: np.ndarray   # (n,) b_{j*}(x)
    box: np.ndarray          # (n,) winning box j*


@dataclass
class Contrastive:
    e: np.ndarray            # (n, d) per-feature "why A, not B"; positive favours A
    mode: np.ndarray         # (n,) "membership", "tie" or "none"
    cls_a: np.ndarray        # (n,) predicted class index A
    cls_b: np.ndarray        # (n,) runner-up class index B (-1 if none)
    box_a: np.ndarray        # (n,) winning box of A
    box_b: np.ndarray        # (n,) best box of B (-1 if none)
    gap: np.ndarray          # (n,) sum of e: b_A - b_B, or ||x-mu_B||^2 - ||x-mu_A||^2


def decomposition(model: CFMNN, X, background) -> Decomposition:
    """Centred decomposition: phi_i = (m_{j*,i}(x_i) - E_B[m_{j*,i}(X_i)]) / d.

    Because b_{j*} is a sum of one-feature terms, phi equals the exact
    interventional SHAP value of each feature for that box's membership
    function (with the background as reference). It is not a SHAP value of the
    final prediction, since the winning box can change when features change.
    """
    X = model._check_X(X)
    background = model._check_X(background)
    d = X.shape[1]
    det = model.predict_details(X)
    terms = model.box_terms(X, det.winner)
    expected = np.empty_like(terms)
    for j in np.unique(det.winner):
        mean_j = feature_terms(background, model.V_[j], model.W_[j], model.gamma).mean(axis=0)
        expected[det.winner == j] = mean_j
    phi = (terms - expected) / d
    return Decomposition(phi, expected.sum(axis=1) / d, terms.mean(axis=1), det.winner)


def contrastive(model: CFMNN, X, det=None) -> Contrastive:
    """Why A, not B (blueprint 5.2).

    Membership mode: e_i = (m_{A,i} - m_{B,i}) / d, sum = b_A - b_B.
    Tie mode (memberships within tie_tol, prototypes decide):
    e_i = (x_i - mu_{B,i})^2 - (x_i - mu_{A,i})^2, sum = ||x-mu_B||^2 - ||x-mu_A||^2.
    """
    X = model._check_X(X)
    n, d = X.shape
    det = det if det is not None else model.predict_details(X)
    rows = np.arange(n)
    a, b = det.cls, det.runner_up
    box_a = det.winner
    box_b = np.where(b >= 0, det.best_box[rows, np.maximum(b, 0)], -1)
    has_b = box_b >= 0
    safe_b = np.where(has_b, box_b, box_a)

    m_a = model.box_terms(X, box_a)
    m_b = model.box_terms(X, safe_b)
    diff_b = m_a.mean(axis=1) - m_b.mean(axis=1)
    tie = has_b & (np.abs(diff_b) <= model.tie_tol)

    e = (m_a - m_b) / d
    mu_a, mu_b = model.centroid_[box_a], model.centroid_[safe_b]
    e_tie = (X - mu_b) ** 2 - (X - mu_a) ** 2
    e = np.where(tie[:, None], e_tie, e)
    e[~has_b] = 0.0
    mode = np.where(~has_b, "none", np.where(tie, "tie", "membership"))
    return Contrastive(e, mode, a, b, box_a, box_b, e.sum(axis=1))


def contrastive_multi(model: CFMNN, X, k: int = 5, tau: float = 0.02) -> np.ndarray:
    """Contrastive explanation against the top-k competing boxes of other classes.

    For each competitor box c, the per-feature vector is the membership-mode
    (m_A - m_c)/d, or the tie-mode centroid term when b_A and b_c tie; each is
    scaled to unit L1 norm (the two modes sit on different scales) and weighted
    by w_c ∝ exp(-(b_A - b_c) / tau), so boxes about to win count most. With
    k = 1 this ranks features like ``contrastive``. Returns (n, d).
    """
    X = model._check_X(X)
    n, d = X.shape
    det = model.predict_details(X)
    M = model.memberships(X)
    rows = np.arange(n)
    other = model.box_class_[None, :] != det.cls[:, None]
    Mo = np.where(other, M, -np.inf)
    kk = min(k, max(1, other.sum(axis=1).min()))
    top = np.argsort(-Mo, axis=1, kind="stable")[:, :kk]
    m_a = model.box_terms(X, det.winner)
    b_a = m_a.mean(axis=1)
    mu_a = model.centroid_[det.winner]
    out = np.zeros((n, d))
    wsum = np.zeros(n)
    for c in range(kk):
        j = top[:, c]
        valid = np.isfinite(Mo[rows, j])
        m_c = model.box_terms(X, j)
        gap = b_a - m_c.mean(axis=1)
        tie = np.abs(gap) <= model.tie_tol
        v = np.where(tie[:, None], (X - model.centroid_[j]) ** 2 - (X - mu_a) ** 2,
                     (m_a - m_c) / d)
        norm = np.abs(v).sum(axis=1, keepdims=True)
        v = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
        w = np.where(valid, np.exp(-np.maximum(gap, 0) / tau), 0.0)
        out += w[:, None] * v
        wsum += w
    return np.divide(out, wsum[:, None], out=np.zeros_like(out), where=wsum[:, None] > 0)


def pace_flip(model: CFMNN, X, background, n_quantiles: int | None = 32,
              return_parts: bool = False):
    """PACE-flip: which single-feature changes would change the decision.

    For each point and feature i, x_i is replaced in turn by each of a fixed set
    of values: by default the 32 mid-quantiles of feature i in the background
    (deterministic; same faithfulness as using every background row, at up to
    ~20x lower cost). ``n_quantiles=None`` uses every background row. Because
    every box's membership is a sum of one-feature terms (and squared centroid
    distance is too), the new memberships and distances are updated exactly:
        b_j' = b_j + (m_{j,i}(z) - m_{j,i}(x_i)) / d
        D_j' = D_j + (z - mu_{j,i})^2 - (x_i - mu_{j,i})^2
    and CFMNN's own decision rule is applied, so the result equals re-running
    the model on every modified input.

    Two exact quantities per feature:
      flip  = share of replacements that change the predicted class;
      drop  = mean drop of the margin s'_A - max_{c != A} s'_c.
    Features are ranked by flip, and ties (common: most features never flip the
    class alone) are broken by drop. The returned score encodes that order:
    flip + (0.5 / r) * normalised drop, which never reorders different flips.

    This builds a ``PaceFlip`` explainer and runs it once. To explain many
    batches with the same model and background, build ``PaceFlip`` once.
    """
    return PaceFlip(model, background, n_quantiles).explain(X, return_parts=return_parts)


class PaceFlip:
    """Fast, exact PACE-flip (see ``pace_flip`` for what it computes).

    The output equals the original per-point loop bit for bit
    (``tests/reference_pace_flip.py``); only the work is organised differently:

    1. Tables built once: every box's term m_{j,i}(z) and squared centroid
       distance (z - mu_{j,i})^2 at the fixed replacement values z. They do
       not depend on the point being explained.
    2. Boxes that cannot matter are skipped. Replacing feature i moves box j's
       membership only within [c_j + min_z m_{j,i}(z)/d, c_j + max_z m_{j,i}(z)/d],
       with c_j = b_j - m_{j,i}(x_i)/d. Boxes are split into two groups: the
       predicted class A and all other classes. A box whose ceiling is below
       the floor of its group's strongest box (minus tie_tol and a 1e-12
       rounding slack) can never be its group's maximum, the winner, or tied
       with the winner, so leaving it out changes nothing.
    3. All points and features are processed together in cache-sized chunks
       (the model's ``chunk_elems``). The nearest-centroid tie-break is only
       evaluated where a box of class A and a box of another class tie at the
       top; everywhere else the winner's class follows from the two group
       maxima.

    The tables are rebuilt automatically if the model's boxes change (as in
    refinement). After ``explain``, ``stats_`` reports how much work was skipped.
    """

    def __init__(self, model: CFMNN, background, n_quantiles: int | None = 32,
                 table_elems: int = 16_000_000):
        self.model = model
        self.n_quantiles = n_quantiles
        self.table_elems = table_elems
        B = model._check_X(background)
        if n_quantiles is not None:
            B = np.quantile(B, (np.arange(n_quantiles) + 0.5) / n_quantiles, axis=0)
        self.values_ = B                                   # (r, d) replacement values
        self.stats_: dict = {}
        self._build()

    # ------------------------------------------------------------------ tables
    def _build(self) -> None:
        m = self.model
        H, d = m.V_.shape
        r = len(self.values_)
        # Values are split into blocks only when the tables would be very large
        # (e.g. n_quantiles=None with a big background); then they are rebuilt
        # per block on each call instead of being kept in memory.
        step = max(1, self.table_elems // max(1, d * H))
        self._blocks = [(s, min(r, s + step)) for s in range(0, r, step)]
        amax = np.full((d, H), -np.inf)
        amin = np.full((d, H), np.inf)
        stored = []
        for s, e in self._blocks:
            A, Q = self._tables(s, e)
            np.maximum(amax, A.max(axis=2), out=amax)
            np.minimum(amin, A.min(axis=2), out=amin)
            if len(self._blocks) == 1:
                stored.append((A, Q))
        self._stored = stored or None
        self._ceil = np.ascontiguousarray((amax / d).T)    # (H, d)
        self._floor = np.ascontiguousarray((amin / d).T)
        self._snapshot = (m.V_.copy(), m.W_.copy(), m.centroid_.copy(), float(m.gamma))

    def _tables(self, s: int, e: int):
        """Terms and squared centroid distances of every box at values s:e, (d, H, e - s)."""
        m = self.model
        V, W, mu, g = m.V_, m.W_, m.centroid_, m.gamma
        H, d = V.shape
        Z = self.values_[s:e]
        A = np.empty((d, H, e - s))
        Q = np.empty((d, H, e - s))
        for i in range(d):                                 # one feature at a time stays in cache
            z = Z[:, i]
            A[i] = feature_terms(z, V[:, i, None], W[:, i, None], g)
            Q[i] = (z - mu[:, i, None]) ** 2
        return A, Q

    def _block_tables(self):
        if self._stored is not None:
            return zip(self._blocks, self._stored)
        return ((blk, self._tables(*blk)) for blk in self._blocks)

    def _is_current(self) -> bool:
        m = self.model
        V, W, mu, g = self._snapshot
        return (m.V_.shape == V.shape and m.centroid_.shape == mu.shape
                and float(m.gamma) == g and np.array_equal(m.V_, V)
                and np.array_equal(m.W_, W) and np.array_equal(m.centroid_, mu))

    # ----------------------------------------------------------------- explain
    def explain(self, X, return_parts: bool = False):
        m = self.model
        if not self._is_current():
            self._build()
        X = m._check_X(X)
        n, d = X.shape
        H = m.n_boxes_
        r = len(self.values_)
        flip = np.empty((n, d))
        drop = np.empty((n, d))
        kept = tie_cells = 0
        step = max(1, int(m.chunk_elems) // max(1, H * d))
        for s in range(0, n, step):
            f, dr, k, t = self._chunk(X[s:s + step])
            flip[s:s + step] = f
            drop[s:s + step] = dr
            kept += k
            tie_cells += t
        self.stats_ = {
            "n_points": n, "n_boxes": H, "n_values": r,
            "boxes_kept_per_feature": kept / max(1, n * d),
            "share_of_boxes_skipped": 1.0 - kept / max(1, n * d * H),
            "tie_break_cells": tie_cells, "cells": n * d * r,
        }
        lo = drop.min(axis=1, keepdims=True)
        hi = drop.max(axis=1, keepdims=True)
        norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
        score = flip + norm * (0.5 / r)
        return (score, flip, drop) if return_parts else score

    __call__ = explain

    def _chunk(self, X):
        m = self.model
        n, d = X.shape
        V, W, mu, g = m.V_, m.W_, m.centroid_, m.gamma
        tol = float(m.tie_tol)
        bc = m.box_class_
        H = len(V)
        r = len(self.values_)

        # Memberships, centroid distances and the prediction at x itself,
        # computed exactly as predict_details does.
        T = feature_terms(X[:, None, :], V, W, g)                  # (n, H, d)
        M = T.mean(axis=2)                                          # (n, H)
        D = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)     # (n, H)
        top = M.max(axis=1)
        a = bc[np.argmin(np.where(M >= top[:, None] - tol, D, np.inf), axis=1)]
        isA = bc[None, :] == a[:, None]                             # boxes of the predicted class
        sA0 = np.where(isA, M, -np.inf).max(axis=1)
        sO0 = np.where(isA, -np.inf, M).max(axis=1)
        has_other = np.isfinite(sO0)
        m0 = (2.0 * sA0 - 1.0) - np.where(has_other, 2.0 * sO0 - 1.0, 0.0)

        # Exact bounds on each box's membership when feature i is replaced.
        c = M[:, :, None] - T / d                                   # (n, H, d)
        floor = c + self._floor
        floorA = np.where(isA[:, :, None], floor, -np.inf).max(axis=1)   # (n, d)
        floorO = np.where(isA[:, :, None], -np.inf, floor).max(axis=1)
        c += self._ceil                                             # now each box's ceiling
        slack = tol + 1e-12
        keep = np.empty((n, d, 2, H), dtype=bool)
        keep[:, :, 0, :] = (isA[:, :, None] & (c >= (floorA - slack)[:, None, :])).transpose(0, 2, 1)
        keep[:, :, 1, :] = (~isA[:, :, None] & (c >= (floorO - slack)[:, None, :])).transpose(0, 2, 1)
        p_i, i_i, g_i, j_i = np.nonzero(keep)       # sorted by point, feature, group, box

        seg = (p_i * d + i_i) * 2 + g_i            # one segment per (point, feature, group)
        nseg = 2 * n * d
        cnt = np.bincount(seg, minlength=nseg)
        start = np.zeros(nseg, dtype=np.intp)
        start[1:] = np.cumsum(cnt)[:-1]
        has = cnt > 0
        # Each group's strongest box always survives the bound.
        if not (has[0::2].all() and np.array_equal(has[1::2], np.repeat(has_other, d))):
            raise RuntimeError("PACE-flip pruning lost a group's strongest box")

        T_kept = T[p_i, j_i, i_i][:, None]
        M_kept = M[p_i, j_i][:, None]
        flip_c = np.empty((n * d, r), dtype=bool)
        marg = np.empty((n * d, r))
        tie_cells = 0
        for (s, e), (A, Q) in self._block_tables():
            Mt = A[i_i, j_i]                      # same arithmetic as M + (term - T_i) / d
            Mt -= T_kept
            Mt /= d
            Mt += M_kept
            G = np.full((nseg, e - s), -np.inf)
            G[has] = np.maximum.reduceat(Mt, start[has], axis=0)
            sA, sO = G[0::2], G[1::2]              # class A's and the other classes' maxima
            thr = np.maximum(sA, sO) - tol
            tiedA, tiedO = sA >= thr, sO >= thr
            fc = ~tiedA                            # no class-A box tied at the top: class changes
            both = tiedA & tiedO
            if both.any():
                # Boxes of A and of another class tie: nearest centroid, then lowest box index.
                need = both.any(axis=1)
                rows = np.flatnonzero(need[seg // 2])
                pr, ir, jr, cr = p_i[rows], i_i[rows], j_i[rows], seg[rows] // 2
                Dt = D[pr, jr][:, None] + Q[ir, jr] - ((X[pr, ir] - mu[jr, ir]) ** 2)[:, None]
                Dm = np.where(Mt[rows] >= thr[cr], Dt, np.inf)
                cells, first = np.unique(cr, return_index=True)
                size = np.diff(np.append(first, len(rows)))
                dmin = np.minimum.reduceat(Dm, first, axis=0)
                jwin = np.minimum.reduceat(
                    np.where(Dm == np.repeat(dmin, size, axis=0), jr[:, None], H), first, axis=0)
                fc[cells] = np.where(both[cells], bc[jwin] != a[cells // d][:, None], fc[cells])
                tie_cells += int(both.sum())
            flip_c[:, s:e] = fc
            marg[:, s:e] = (2.0 * sA - 1.0) - np.where(np.isfinite(sO), 2.0 * sO - 1.0, 0.0)

        flip = flip_c.mean(axis=1).reshape(n, d)
        drop = (np.repeat(m0, d) - marg.mean(axis=1)).reshape(n, d)
        return flip, drop, len(j_i), tie_cells
