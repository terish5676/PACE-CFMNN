"""PACE local explanations (blueprint 5.1 and 5.2).

Both reuse CFMNN's own per-feature membership terms, so they are exact for the
model's membership score and never change a prediction.
"""

from __future__ import annotations

import math
import time
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
    """Why A, not B (blueprint 5.2), with A the predicted and B the runner-up class.

    Membership mode: e_i = (m_{A,i} - m_{B,i}) / d, sum = b_A - b_B.
    Tie mode (memberships within tie_tol, prototypes decide):
    e_i = (x_i - mu_{B,i})^2 - (x_i - mu_{A,i})^2, sum = ||x-mu_B||^2 - ||x-mu_A||^2.
    """
    X = model._check_X(X)
    det = det if det is not None else model.predict_details(X)
    return contrastive_pair(model, X, det.cls, det.runner_up, det=det)


def contrastive_pair(model: CFMNN, X, cls_a, cls_b, det=None) -> Contrastive:
    """Why class ``cls_a`` and not ``cls_b`` at each point, for a given class pair.

    Uses the best box of each given class at each point (CFMNN's tie rule), so
    the pair stays fixed even where the prediction changes. This is what
    explanation instability (EI) needs. ``cls_b = -1`` gives mode "none".
    With A the predicted class and B the runner-up it equals ``contrastive``.
    """
    X = model._check_X(X)
    n, d = X.shape
    det = det if det is not None else model.predict_details(X)
    rows = np.arange(n)
    a = np.array(np.broadcast_to(np.asarray(cls_a, dtype=np.int64), (n,)))
    b = np.array(np.broadcast_to(np.asarray(cls_b, dtype=np.int64), (n,)))
    box_a = det.best_box[rows, a]
    box_b = np.where(b >= 0, det.best_box[rows, np.maximum(b, 0)], -1)
    has_b = (box_b >= 0) & (box_a >= 0)
    box_a = np.where(box_a >= 0, box_a, 0)
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


def _row_chunks(cost, budget):
    """Split background rows 0..R-1 into consecutive chunks of total cost <= budget
    (a single row may exceed it)."""
    cum = np.cumsum(cost)
    out, r0, R = [], 0, len(cost)
    while r0 < R:
        base = cum[r0 - 1] if r0 else 0
        r1 = min(R, max(r0 + 1, int(np.searchsorted(cum, base + budget, side="right"))))
        out.append((r0, r1))
        r0 = r1
    return out


class PaceShap:
    """PACE-SHAP: Shapley values of the predicted class's margin, from CFMNN's own terms.

    It explains the quantity the faithfulness tests measure (and that
    KernelSHAP estimates with ``target="margin"``): the margin of the predicted
    class A,
        f(x) = s'_A(x) - max_{c != A} s'_c(x),   s'_c = 2 max_{j in c} b_j - 1,
    with A fixed per point. phi_i is the interventional Shapley value of f with
    the background rows as reference: features outside a coalition S take the
    values of a background row z, and phi is averaged over the rows.

    Every box's membership is a mean of one-feature terms, so for one row z
        b_j(S) = b_j(z) + sum_{i in S} (m_{j,i}(x_i) - m_{j,i}(z_i)) / d,
    and f(S) can be evaluated exactly for any coalition without re-running
    the model. Two modes:

    * d <= ``exact_max_d``: all 2^d coalitions are evaluated and the Shapley
      formula is applied exactly (no sampling).
    * larger d: random feature orders, each paired with its reverse
      (antithetic), ``n_permutations`` per background row, from a fixed seed.
      Each step adds one feature, an O(boxes) update. The estimate is unbiased
      and still adds up exactly: sum_i phi_i = f(x) - mean_z f(z).

    Boxes that can never be their group's maximum (group = class A, or all
    other classes) under any coalition are skipped: box j's membership over
    all S lies in [b_j(z) + sum of its negative changes, b_j(z) + sum of its
    positive changes], and a box whose ceiling is below the best floor in its
    group cannot change f. This is exact, not an approximation.

    The margin needs only the group maxima, not the nearest-centroid
    tie-break, so ties need no special handling. The model is never changed.

    **Conditional game (``sigma`` given; "PACE-SHAP-cond").** Features outside
    S still come from background rows, but each row is weighted by how close
    it is to x on the features in S (the empirical conditional method of Aas,
    Jullum and Løland 2021, with a diagonal covariance):
        v(S) = sum_z w_z(S) f(x_S, z_rest) / sum_z w_z(S),
        w_z(S) = exp(-D_S(x, z)^2 / (2 sigma^2)),
        D_S^2 = (1/|S|) sum_{i in S} ((x_i - z_i) / scale_i)^2,
    with ``scale`` the background's per-feature standard deviation. The
    squared distance is a sum over features, so the weights of every
    coalition come from the same subset sums: exact mode stays exact. Sampled
    mode shares its feature orders across background rows, because each
    v(S) needs every row. With sigma -> infinity all weights are equal and
    this is plain PACE-SHAP.
    """

    def __init__(self, model: CFMNN, background, exact_max_d: int = 12,
                 n_permutations: int = 32, seed: int = 0, chunk_elems: int = 4_000_000,
                 prune: bool = True, sigma: float | None = None, scale=None,
                 exact_algo: str = "matmul"):
        if n_permutations < 2 or n_permutations % 2:
            raise ValueError("n_permutations must be even and >= 2 (antithetic pairs)")
        if sigma is not None and not sigma > 0:
            raise ValueError("sigma must be positive (or None for the interventional game)")
        if exact_algo not in ("matmul", "recurrence"):
            raise ValueError("exact_algo must be 'matmul' or 'recurrence'")
        # "matmul": every coalition's membership as MB + Delta @ ind, K d 2^d multiply-adds.
        # "recurrence" (tough review B9): built one feature at a time, each subset's sum is a
        # smaller subset's sum plus one Delta, K 2^d additions. Same values to rounding.
        self.exact_algo = exact_algo
        self.model = model
        self.background_ = model._check_X(background)
        self.exact_max_d = exact_max_d
        self.n_permutations = n_permutations
        self.seed = seed
        self.chunk_elems = chunk_elems
        self.prune = prune
        self.sigma = sigma
        self.scale_ = cond_scale(self.background_) if scale is None else np.asarray(scale, float)
        self.stats_: dict = {}
        self._pairs = None
        self._build()

    def _build(self) -> None:
        m = self.model
        TB = feature_terms(self.background_[:, None, :], m.V_, m.W_, m.gamma)   # (R, H, d)
        self._TB = TB
        self._MB = TB.mean(axis=2)                                              # (R, H)
        self._snapshot = (m.V_.copy(), m.W_.copy(), float(m.gamma))
        d = TB.shape[2]
        self._exact_tables = None
        if d <= self.exact_max_d:
            masks = np.arange(1 << d)
            ind = ((masks[None, :] >> np.arange(d)[:, None]) & 1).astype(float)  # (d, 2^d)
            size = ind.sum(axis=0).astype(int)
            fact = np.array([float(np.prod(np.arange(1, k + 1))) for k in range(d + 1)])
            w = fact[:d] * fact[d - 1::-1] / fact[d]          # w[s] = s! (d-s-1)! / d!
            lo = [masks[(masks >> i) & 1 == 0] for i in range(d)]
            self._exact_tables = (ind, lo, [w[size[l]] for l in lo])
            self._size = size

    def _is_current(self) -> bool:
        m = self.model
        V, W, g = self._snapshot
        return (m.V_.shape == V.shape and float(m.gamma) == g
                and np.array_equal(m.V_, V) and np.array_equal(m.W_, W))

    @property
    def exact(self) -> bool:
        return self._TB.shape[2] <= self.exact_max_d

    # ----------------------------------------------------------------- explain
    def explain(self, X, seed: int | None = None, return_parts: bool = False):
        """Shapley values (n, d). With ``return_parts``: (phi, f(x), mean_z f(z)).

        ``seed`` overrides the constructor's seed (sampled mode only)."""
        m = self.model
        if not self._is_current():
            self._build()
        X = m._check_X(X)
        n, d = X.shape
        cls = m.predict_details(X).cls
        rng = np.random.default_rng(self.seed if seed is None else seed)
        phi = np.zeros((n, d))
        fx = np.zeros(n)
        f0 = np.zeros(n)
        kept = 0
        for p in range(n):
            phi[p], fx[p], f0[p], k = self._one(X[p], int(cls[p]), rng)
            kept += k
        R, H = self._MB.shape
        self.stats_ = {"n_points": n, "n_boxes": H, "n_background": R,
                       "mode": "exact" if self.exact else "sampled",
                       "game": "interventional" if self.sigma is None else "conditional",
                       "share_of_boxes_skipped": 1.0 - kept / max(1, n * R * H)}
        return (phi, fx, f0) if return_parts else phi

    __call__ = explain

    def explain_games(self, X, sigmas, times: dict | None = None) -> list[np.ndarray]:
        """Exact mode: Shapley values (n, d) of several games at once, one per sigma.

        Each entry of ``sigmas`` is None (the interventional game) or a number
        (the conditional game with that sigma); the instance's own ``sigma`` is
        not used. The games differ only in how background rows are weighted, so
        every row's margin under every coalition is computed once for all of
        them. Each result is identical to ``PaceShap(model, background,
        sigma=s).explain(X)``. ``times``, if a dict, gets ``"own"``: each game's
        own seconds (its weights, weighted sums and Shapley sums); the rest of
        the time is shared.
        """
        if not self.exact:
            raise ValueError("explain_games needs exact mode (d <= exact_max_d)")
        if any(s is not None and not s > 0 for s in sigmas):
            raise ValueError("sigma must be positive (or None for the interventional game)")
        m = self.model
        if not self._is_current():
            self._build()
        X = m._check_X(X)
        n, d = X.shape
        cls = m.predict_details(X).cls
        _, lo, shapley_w = self._exact_tables
        out = [np.zeros((n, d)) for _ in sigmas]
        spent = np.zeros(len(sigmas))
        for p in range(n):
            vs, _ = self._v_exact_games(X[p], int(cls[p]), sigmas, spent)
            for g, v in enumerate(vs):
                t0 = time.perf_counter()
                out[g][p] = [np.dot(shapley_w[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)]
                spent[g] += time.perf_counter() - t0
        if times is not None:
            times["own"] = spent.tolist()
        return out

    def interactions(self, X):
        """SHAP interaction values, shape (n, d, d); exact mode only.

        Off-diagonal entries are half the Shapley interaction index
        (Grabisch and Roubens 1999), as in Lundberg et al. (2020):
            I_ij = sum_{S in N minus {i, j}} |S|! (d - |S| - 2)! / (d - 1)!
                   * [v(S + i + j) - v(S + i) - v(S + j) + v(S)],
        i.e. how much i and j add together beyond what each adds alone. The
        diagonal holds the main effect phi_i - sum_{j != i} I_ij / 2, so each
        row sums to the Shapley value phi_i. Same game as ``explain``
        (interventional, or conditional with ``sigma``), from the same v(S)
        table, so it costs little more than ``explain``.
        """
        if not self.exact:
            raise ValueError("interactions need exact mode (d <= exact_max_d)")
        m = self.model
        if not self._is_current():
            self._build()
        X = m._check_X(X)
        n, d = X.shape
        cls = m.predict_details(X).cls
        _, lo, wts = self._exact_tables
        pairs, plo, pw = self._pair_tables(d)
        out = np.zeros((n, d, d))
        diag = np.diag_indices(d)
        for p in range(n):
            v, _ = self._v_exact(X[p], int(cls[p]))
            phi = np.array([np.dot(wts[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)])
            for k, (i, j) in enumerate(pairs):
                l, bi, bj = plo[k], 1 << i, 1 << j
                half = 0.5 * np.dot(pw[k], v[l | bi | bj] - v[l | bi] - v[l | bj] + v[l])
                out[p, i, j] = out[p, j, i] = half
            out[p][diag] = phi - out[p].sum(axis=1)
        return out

    def _pair_tables(self, d):
        if self._pairs is None or self._pairs[0] != d:
            masks = np.arange(1 << d)
            size = np.array([bin(k).count("1") for k in masks])
            fact = [math.factorial(k) for k in range(d + 1)]
            w2 = np.array([fact[s] * fact[d - s - 2] / fact[d - 1] for s in range(max(d - 1, 1))])
            pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
            plo = [masks[((masks >> i) & 1 == 0) & ((masks >> j) & 1 == 0)] for i, j in pairs]
            self._pairs = (d, pairs, plo, [w2[size[l]] for l in plo])
        return self._pairs[1:]

    def _prepare(self, x, a):
        """Kept (row, box) pairs and each box's per-feature change for point x of class A."""
        m = self.model
        TB, MB = self._TB, self._MB
        R, H, d = TB.shape
        Tx = feature_terms(x, m.V_, m.W_, m.gamma)                 # (H, d)
        Delta = (Tx[None, :, :] - TB) / d                           # (R, H, d)
        isA = m.box_class_ == a
        has_other = bool((~isA).any())
        if self.prune:
            keep = self._prune(MB, Delta, isA)
        else:
            keep = np.ones((R, H), dtype=bool)
        # Kept (row, box) pairs sorted by background row, then group (A first), then box.
        r_i, j_i = np.nonzero(keep)
        g_i = (~isA[j_i]).astype(np.intp)
        order = np.lexsort((j_i, g_i, r_i))
        r_i, j_i, g_i = r_i[order], j_i[order], g_i[order]
        counts = np.bincount(r_i, minlength=R)
        return Delta, r_i, j_i, g_i, counts, has_other

    def _q(self, x):
        """Squared standardised differences to every background row, (R, d)."""
        return ((x - self.background_) / self.scale_) ** 2

    def _v_exact(self, x, a):
        """v(S) for every coalition (bitmask), and the number of kept (row, box) pairs."""
        vs, kept = self._v_exact_games(x, a, (self.sigma,))
        return vs[0], kept

    def _v_exact_games(self, x, a, sigmas, spent=None):
        """``_v_exact`` for several games at once, one per sigma (None = interventional).

        The games share every row's margins; only the row weights differ.
        ``spent`` (one entry per game) collects each game's own seconds."""
        Delta, r_i, j_i, g_i, counts, has_other = self._prepare(x, a)
        wts = []
        for g, sigma in enumerate(sigmas):
            t0 = time.perf_counter()
            wt = None
            if sigma is not None:
                ind = self._exact_tables[0]
                logw = -(self._q(x) @ ind) / (2.0 * sigma ** 2 * np.maximum(self._size, 1))
                wt = _row_weights(logw)                              # (R, 2^d)
            wts.append(wt)
            if spent is not None:
                spent[g] += time.perf_counter() - t0
        vs = self._exact_values(self._MB, Delta, r_i, j_i, g_i, counts, has_other, wts, spent)
        return vs, len(r_i)

    def _one(self, x, a, rng):
        if self.exact:
            v, kept = self._v_exact(x, a)
            _, lo, wts = self._exact_tables
            d = self._TB.shape[2]
            phi = np.array([np.dot(wts[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)])
            return phi, v[-1], v[0], kept
        Delta, r_i, j_i, g_i, counts, has_other = self._prepare(x, a)
        if self.sigma is None:
            phi, fx, f0 = self._sampled(self._MB, Delta, r_i, j_i, g_i, counts, has_other, rng)
        else:
            phi, fx, f0 = self._sampled_cond(self._MB, Delta, r_i, j_i, g_i, counts, has_other,
                                             rng, self._q(x))
        return phi, fx, f0, len(r_i)

    @staticmethod
    def _prune(MB, Delta, isA):
        """Boxes (per background row) that can be a group maximum for some coalition.

        Two exact rules, applied per group (class A / other classes):
        1. bounds: a box whose ceiling (all its positive changes) is below the
           group's best floor (all its negative changes) is never the maximum;
        2. dominance: box k dominates box j if b_k(S) >= b_j(S) for every S,
           i.e. (b_k(z) - b_j(z)) + sum_i min(0, Delta_ki - Delta_ji) >= 0.
           Candidates k are the group's best box at a few probe coalitions
           (none, all, each single feature, each feature left out) plus the
           best floor. A box is dropped only if a candidate beats it by more
           than 1e-12, so rounding can never remove a maximum. Dropping is
           safe even when the candidate itself is dropped by rule 1, because
           then a kept box is above the candidate everywhere.
        """
        R, H, d = Delta.shape
        rows = np.arange(R)
        ceil = MB + np.maximum(Delta, 0).sum(axis=2)
        floor = MB + np.minimum(Delta, 0).sum(axis=2)
        eye = np.eye(d)
        probes = np.concatenate([np.zeros((d, 1)), np.ones((d, 1)), eye, 1.0 - eye], axis=1)
        pv = MB[:, :, None] + Delta @ probes                     # (R, H, 2 + 2d)
        keep = np.zeros((R, H), dtype=bool)
        for grp in (isA, ~isA):
            idx = np.flatnonzero(grp)
            if len(idx) == 0:
                continue
            fl, ce = floor[:, idx], ceil[:, idx]
            kg = ce >= fl.max(axis=1, keepdims=True) - 1e-12      # rule 1
            cand = np.concatenate([pv[:, idx].argmax(axis=1), fl.argmax(axis=1)[:, None]], axis=1)
            cand = np.sort(cand, axis=1)
            Dg = Delta[:, idx]                                    # (R, Hg, d)
            MBg = MB[:, idx]
            dominated = np.zeros_like(kg)
            for c in range(cand.shape[1]):
                kc = cand[:, c]
                if c and np.array_equal(kc, cand[:, c - 1]):
                    continue                                      # same candidates as before
                dom = (MBg[rows, kc][:, None] - MBg
                       + np.minimum(0.0, Dg[rows, kc][:, None, :] - Dg).sum(axis=2))
                dominated |= dom > 1e-12
            keep[:, idx] = kg & ~dominated
        return keep

    @staticmethod
    def _margin(GA, GO, has_other):
        # s'_A - s'_O with s' = 2 s - 1; with no other class, margin_score uses 0.
        return 2.0 * GA - 2.0 * GO if has_other else 2.0 * GA - 1.0

    def _exact_values(self, MB, Delta, r_i, j_i, g_i, counts, has_other, wts, spent=None):
        """v(S) for every coalition S (bitmask), one array per entry of ``wts``.

        An entry None gives the mean over background rows of f(S); a matrix
        (R, 2^d), each column summing to 1, gives a weighted mean instead (the
        conditional game). f is computed once for all entries. ``spent`` (one
        entry per game) collects the seconds of each game's own sums."""
        ind = self._exact_tables[0]
        nm = ind.shape[1]
        R = len(counts)
        bounds = np.r_[0, np.cumsum(counts)]
        vs = [np.zeros(nm) for _ in wts]
        for r0, r1 in _row_chunks(counts * nm, self.chunk_elems):
            rr = r_i[bounds[r0]:bounds[r1]]
            jj = j_i[bounds[r0]:bounds[r1]]
            seg = 2 * (rr - r0) + g_i[bounds[r0]:bounds[r1]]
            if self.exact_algo == "recurrence":
                vals = subset_sums(MB[rr, jj], Delta[rr, jj])      # (rows, 2^d)
            else:
                vals = MB[rr, jj][:, None] + Delta[rr, jj] @ ind  # (rows, 2^d)
            cuts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
            gmax = np.maximum.reduceat(vals, cuts, axis=0)        # (segments, 2^d)
            sid = seg[cuts]
            isa = sid % 2 == 0
            GA = np.full((r1 - r0, nm), -np.inf)
            GO = np.full((r1 - r0, nm), -np.inf)
            GA[sid[isa] // 2] = gmax[isa]
            GO[sid[~isa] // 2] = gmax[~isa]
            f = self._margin(GA, GO, has_other)                   # (rows, 2^d)
            for g, wt in enumerate(wts):
                t0 = time.perf_counter()
                vs[g] += f.sum(axis=0) if wt is None else (wt[r0:r1] * f).sum(axis=0)
                if spent is not None:
                    spent[g] += time.perf_counter() - t0
        return [v / R if wt is None else v for v, wt in zip(vs, wts)]

    def _sampled(self, MB, Delta, r_i, j_i, g_i, counts, has_other, rng):
        """Antithetic permutation sampling: n_permutations orders per background row."""
        P = self.n_permutations
        R, _, d = Delta.shape
        half = rng.permuted(np.tile(np.arange(d), (R * P // 2, 1)), axis=1)
        perms = np.empty((R * P, d), dtype=np.intp)
        perms[0::2] = half
        perms[1::2] = half[:, ::-1]                                # each order and its reverse
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        phi = np.zeros(d)
        fx_sum = f0_sum = 0.0
        for r0, r1 in _row_chunks(counts * P, max(1, self.chunk_elems // 4)):
            ns = (r1 - r0) * P
            rs = np.repeat(np.arange(r0, r1), P)                   # background row of each sample
            L = counts[rs]
            off = np.r_[0, np.cumsum(L)[:-1]]
            samp = np.repeat(np.arange(ns), L)                     # sample of each state row
            src = starts[rs[samp]] + (np.arange(L.sum()) - off[samp])
            rrow, jrow = rs[samp], j_i[src]
            seg = 2 * samp + g_i[src]                              # sorted: sample, then group
            cuts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
            sid = seg[cuts]
            isa = sid % 2 == 0
            val = MB[rrow, jrow].copy()

            def f_now():
                g = np.maximum.reduceat(val, cuts)
                GA = np.full(ns, -np.inf)
                GO = np.full(ns, -np.inf)
                GA[sid[isa] // 2] = g[isa]
                GO[sid[~isa] // 2] = g[~isa]
                return self._margin(GA, GO, has_other)

            prev = f_now()
            f0_sum += prev.sum()
            pr = perms[r0 * P:r1 * P]                             # (ns, d)
            for t in range(d):
                feat = pr[:, t]
                val += Delta[rrow, jrow, feat[samp]]
                cur = f_now()
                phi += np.bincount(feat, weights=cur - prev, minlength=d)
                prev = cur
            fx_sum += prev.sum()
        total = R * P
        return phi / total, fx_sum / total, f0_sum / total

    def _sampled_cond(self, MB, Delta, r_i, j_i, g_i, counts, has_other, rng, Q):
        """Conditional game, sampled: n_permutations antithetic orders shared by all rows.

        For order p and step t (coalition S = the first t features), v_p(t) is
        the weighted mean over rows of f(S); phi gets v_p(t + 1) - v_p(t).
        The weights of every (row, order, step) are fixed in advance, so rows
        can still be processed in chunks.
        """
        P = self.n_permutations
        R, _, d = Delta.shape
        half = rng.permuted(np.tile(np.arange(d), (P // 2, 1)), axis=1)
        perms = np.empty((P, d), dtype=np.intp)
        perms[0::2] = half
        perms[1::2] = half[:, ::-1]
        cq = np.zeros((R, P, d + 1))
        cq[:, :, 1:] = np.cumsum(Q[:, perms], axis=2)               # (R, P, d + 1)
        size = np.maximum(np.arange(d + 1), 1)
        wt = _row_weights(-cq / (2.0 * self.sigma ** 2 * size))     # sums to 1 over rows
        v = np.zeros((P, d + 1))
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        for r0, r1 in _row_chunks(counts * P, max(1, self.chunk_elems // 4)):
            ns = (r1 - r0) * P
            rs = np.repeat(np.arange(r0, r1), P)                   # background row of each sample
            ps = np.tile(np.arange(P), r1 - r0)                    # order of each sample
            L = counts[rs]
            off = np.r_[0, np.cumsum(L)[:-1]]
            samp = np.repeat(np.arange(ns), L)
            src = starts[rs[samp]] + (np.arange(L.sum()) - off[samp])
            rrow, jrow = rs[samp], j_i[src]
            seg = 2 * samp + g_i[src]
            cuts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
            sid = seg[cuts]
            isa = sid % 2 == 0
            val = MB[rrow, jrow].copy()

            def f_now():
                g = np.maximum.reduceat(val, cuts)
                GA = np.full(ns, -np.inf)
                GO = np.full(ns, -np.inf)
                GA[sid[isa] // 2] = g[isa]
                GO[sid[~isa] // 2] = g[~isa]
                return self._margin(GA, GO, has_other)

            v[:, 0] += np.bincount(ps, weights=wt[rs, ps, 0] * f_now(), minlength=P)
            pr = perms[ps]                                         # (ns, d)
            for t in range(d):
                val += Delta[rrow, jrow, pr[samp, t]]
                v[:, t + 1] += np.bincount(ps, weights=wt[rs, ps, t + 1] * f_now(), minlength=P)
        phi = np.zeros(d)
        np.add.at(phi, perms, v[:, 1:] - v[:, :-1])
        return phi / P, v[:, d].mean(), v[:, 0].mean()


def cond_scale(background) -> np.ndarray:
    """Per-feature standard deviation of the rows (1 where a feature is constant)."""
    s = np.asarray(background, float).std(axis=0)
    return np.where(s > 0, s, 1.0)


def _row_weights(logw):
    """exp(logw) normalised to sum 1 over axis 0, computed without underflow."""
    w = np.exp(logw - logw.max(axis=0, keepdims=True))
    return w / w.sum(axis=0, keepdims=True)


def pace_shap(model: CFMNN, X, background, seed: int = 0, **kw) -> np.ndarray:
    """PACE-SHAP attributions (n, d); see ``PaceShap``."""
    return PaceShap(model, background, seed=seed, **kw).explain(X)


def cond_neighbour_values(x, background, k: int = 32, scale=None) -> np.ndarray:
    """Replacement values for conditional PACE-flip, shape (k, d).

    Column i holds feature i of the k background rows nearest to x on all the
    other features (Euclidean on features divided by ``scale``, by default
    the rows' standard deviation). Ties go to the earlier row.
    """
    B = np.asarray(background, float)
    x = np.asarray(x, float).ravel()
    scale = cond_scale(B) if scale is None else np.asarray(scale, float)
    q = ((x - B) / scale) ** 2                                      # (R, d)
    dist = q.sum(axis=1, keepdims=True) - q                         # without feature i
    idx = np.argsort(dist, axis=0, kind="stable")[:min(k, len(B))]  # (k, d)
    return B[idx, np.arange(B.shape[1])]


def pace_flip_cond(model: CFMNN, X, background, k: int = 32, scale=None,
                   return_parts: bool = False):
    """Conditional PACE-flip: PACE-flip with values drawn near the point.

    Same flip and margin-drop scores as ``pace_flip``, but feature i of point
    x is replaced by the values of the k background rows nearest to x on its
    other features (``cond_neighbour_values``), not by the background's
    quantiles. Replaced inputs therefore stay close to the data. Exact: each
    point gets its own ``PaceFlip`` tables, so it equals brute-force
    occlusion with the same values (``xai_baselines.occlusion(values=...)``).
    """
    X = model._check_X(X)
    B = model._check_X(background)
    scale = cond_scale(B) if scale is None else np.asarray(scale, float)
    n, d = X.shape
    score, flip, drop = np.empty((n, d)), np.empty((n, d)), np.empty((n, d))
    for p in range(n):
        vals = cond_neighbour_values(X[p], B, k, scale)
        s, f, dr = PaceFlip(model, vals, n_quantiles=None).explain(X[p:p + 1], return_parts=True)
        score[p], flip[p], drop[p] = s[0], f[0], dr[0]
    return (score, flip, drop) if return_parts else score


# ------------------------------------------------------ tough review: B9 helpers
def subset_sums(base, D) -> np.ndarray:
    """base[r] + sum_{i in S} D[r, i] for every subset S (bit i = feature i), shape (rows, 2^d).

    Built one feature at a time: the subsets containing feature i are those without it plus
    D[:, i], so the whole table costs rows * 2^d additions (no d factor).
    """
    base = np.asarray(base, dtype=float)
    D = np.asarray(D, dtype=float)
    rows, d = D.shape
    out = np.empty((rows, 1 << d))
    out[:, 0] = base
    for i in range(d):
        h = 1 << i
        np.add(out[:, :h], D[:, i:i + 1], out=out[:, h:2 * h])
    return out


def brute_force_shap(model: CFMNN, X, background, chunk_elems: int = 4_000_000) -> np.ndarray:
    """Exact interventional Shapley values of the margin by running the model on every mixed input.

    A fair brute-force baseline for timing (tough review B9): no Proposition 2, no pruning, but
    vectorised numpy instead of shap's general explainer. Same game as ``PaceShap`` (class A fixed
    per point, margin s'_A - max_{c != A} s'_c, background rows as the reference).
    """
    X = model._check_X(X)
    Z = model._check_X(background)
    n, d = X.shape
    R, H = len(Z), model.n_boxes_
    nm = 1 << d
    masks = np.arange(nm)
    bits = ((masks[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)     # (2^d, d)
    size = bits.sum(axis=1)
    fact = [math.factorial(k) for k in range(d + 1)]
    w = np.array([fact[s] * fact[d - s - 1] / fact[d] for s in range(d)])
    cls = model.predict_details(X).cls
    has_other = bool(len(model.classes_) > 1)
    step = max(1, chunk_elems // max(1, H * d * R))
    phi = np.zeros((n, d))
    for p in range(n):
        isA = model.box_class_ == cls[p]
        v = np.zeros(nm)
        for s0 in range(0, nm, step):
            b = bits[s0:s0 + step]
            Hy = np.where(b[:, None, :], X[p][None, None, :], Z[None, :, :]).reshape(-1, d)
            M = feature_terms(Hy[:, None, :], model.V_, model.W_, model.gamma).mean(axis=2)
            GA = np.where(isA, M, -np.inf).max(axis=1)
            if has_other and (~isA).any():
                f = 2.0 * GA - 2.0 * np.where(~isA, M, -np.inf).max(axis=1)
            else:
                f = 2.0 * GA - 1.0
            v[s0:s0 + step] = f.reshape(len(b), R).mean(axis=1)
        for i in range(d):
            lo = masks[~bits[:, i]]
            phi[p, i] = np.dot(w[size[lo]], v[lo | (1 << i)] - v[lo])
    return phi
