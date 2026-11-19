"""PACE-SHAP and PACE-flip for min-membership fuzzy min-max networks (tough review B2).

GFMM (Gabrys and Bargiela 2000) and its descendants in hyperbox-brain score a
point by the *minimum* of one-feature terms:

    b_j(x) = min_i t_{j,i}(x_i),
    t_{j,i}(u) = min(1 - f(u - w_{j,i}), 1 - f(v_{j,i} - u)),
    f(z) = min(1, max(0, gamma_i z)).

They predict the class of the box with the highest membership. Exact ties
between classes follow the model's own rule in hyperbox-brain:
* Online GFMM (and the default): the box whose centre is nearest in Manhattan
  distance, then the first such box (``predict_with_manhattan``);
* IOL-GFMM: the class with the largest share of membership times box sample
  count among the tied boxes, ties in that going to the class whose tied boxes
  hold more samples; at membership 1 a tied box holding one sample is chosen as
  hyperbox-brain does (``predict_with_probability``; P4 Amendment 1).

For a coalition S, with the other features taken from a background row z,

    b_j(S) = min( min_{i in S} t_{j,i}(x_i), min_{i not in S} t_{j,i}(z_i) ).

Both inner minima are minima over a subset of features, and the table of all
2^d subset minima is built one feature at a time: a subset's minimum is the
smaller of the minimum without its last feature and that feature's term. The
minimum over the features outside S is the same table read at the complement
of S. So every coalition's membership is exact, at 2^d work per (row, box),
as for the mean-membership models. The game is the margin
s_A - max_{c != A} s_c, with A the predicted class, as for ``PaceShap``.

``PaceFlipMin`` is PACE-flip for these models. Replacing feature i by u gives
b_j = min(t_{j,i}(u), min_{k != i} t_{j,k}(x_k)), and the minima over the other
features come from prefix and suffix minima. It then applies the model's own
decision rule.
"""

from __future__ import annotations

import inspect
import math
import random

import numpy as np


def _ramp(z, g):
    return np.clip(z * g, 0.0, 1.0)


class GFMMBoxes:
    """The boxes and decision rule of a min-membership model (numeric features in [0, 1])."""

    def __init__(self, V, W, C, gamma=1.0, n_samples=None, tie_rule="manhattan"):
        if tie_rule not in ("manhattan", "probability"):
            raise ValueError(f"unknown tie rule {tie_rule!r}")
        if tie_rule == "probability" and n_samples is None:
            raise ValueError("the probability tie rule needs each box's sample count")
        self.tie_rule = tie_rule
        self.n_samples = None if n_samples is None else np.asarray(n_samples)
        self.V = np.asarray(V, dtype=float)
        self.W = np.asarray(W, dtype=float)
        self.C = np.asarray(C)
        g = np.asarray(gamma, dtype=float)
        self.gamma = np.full(self.V.shape[1], float(g)) if g.ndim == 0 else g
        self.classes_ = np.unique(self.C)
        self.box_cls = np.searchsorted(self.classes_, self.C)
        self.centre = 0.5 * (self.V + self.W)

    @classmethod
    def from_hbbrain(cls, est) -> "GFMMBoxes":
        """The boxes of a fitted hyperbox-brain model, with the tie rule its ``predict`` uses."""
        p = inspect.signature(est.predict).parameters.get("type_boundary_handling")
        from hbbrain.constants import PROBABILITY_MEASURE
        if p is not None and p.default == PROBABILITY_MEASURE:
            return cls(est.V, est.W, est.C, getattr(est, "gamma", 1.0),
                       n_samples=est.N_samples, tie_rule="probability")
        return cls(est.V, est.W, est.C, getattr(est, "gamma", 1.0))

    @property
    def n_features(self) -> int:
        return self.V.shape[1]

    def terms(self, X) -> np.ndarray:
        """t_{j,i}(x_i), shape (n, H, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xb = X[:, None, :]
        return np.minimum(1.0 - _ramp(Xb - self.W[None], self.gamma),
                          1.0 - _ramp(self.V[None] - Xb, self.gamma))

    def memberships(self, X) -> np.ndarray:
        return self.terms(X).min(axis=2)

    def _decide(self, M, X):
        """Class index per row from memberships M (n, H), hyperbox-brain's rule."""
        out = np.empty(len(M), dtype=np.int64)
        top = M.max(axis=1)
        for r in range(len(M)):
            ids = np.flatnonzero(M[r] == top[r])
            if len(np.unique(self.box_cls[ids])) > 1:
                if self.tie_rule == "probability":
                    out[r] = self._probability_winner(M[r], ids, top[r])
                else:
                    dist = np.abs(self.centre[ids] - X[r]).sum(axis=1)
                    out[r] = self.box_cls[ids[int(np.argmin(dist))]]
            else:
                out[r] = self.box_cls[ids[0]]
        return out

    def _probability_winner(self, m, ids, top):
        """hyperbox-brain's ``predict_with_probability`` on tied boxes ``ids`` (memberships m)."""
        N = self.n_samples
        if top == 1:
            one = ids[N[ids] == 1]
            if len(one):
                return self.box_cls[random.Random(0).choice(one)]
        den = (m[ids] * N[ids]).sum()
        best, best_p, best_n = None, -1, None
        for c in np.unique(self.box_cls[ids]):
            sel = ids[self.box_cls[ids] == c]
            p = (m[sel] * N[sel]).sum() / den if den != 0 else 0
            n = N[sel].sum()
            if p > best_p or (p == best_p and best is not None and n > best_n):
                best, best_p, best_n = c, p, n
        return best

    def predict_index(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._decide(self.memberships(X), X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_index(X)]

    def margin(self, X, cls) -> np.ndarray:
        """s_A - max_{c != A} s_c for class index A = cls[r] (s_A alone with one class)."""
        M = self.memberships(X)
        cls = np.asarray(cls)
        isA = self.box_cls[None, :] == cls[:, None]
        GA = np.where(isA, M, -np.inf).max(axis=1)
        if len(self.classes_) < 2:
            return GA
        return GA - np.where(~isA, M, -np.inf).max(axis=1)


def subset_mins(T) -> np.ndarray:
    """min_{i in S} T[r, i] for every subset S (bit i = feature i), +inf for S empty; (rows, 2^d)."""
    T = np.asarray(T, dtype=float)
    rows, d = T.shape
    out = np.empty((rows, 1 << d))
    out[:, 0] = np.inf
    for i in range(d):
        h = 1 << i
        np.minimum(out[:, :h], T[:, i:i + 1], out=out[:, h:2 * h])
    return out


def _shapley_tables(d):
    masks = np.arange(1 << d)
    size = np.array([bin(k).count("1") for k in masks])
    fact = [math.factorial(k) for k in range(d + 1)]
    w = np.array([fact[s] * fact[d - s - 1] / fact[d] for s in range(d)])
    lo = [masks[(masks >> i) & 1 == 0] for i in range(d)]
    return lo, [w[size[l]] for l in lo]


class PaceShapMin:
    """Exact (d <= exact_max_d) or antithetic-sampled Shapley values of the margin for GFMM boxes."""

    def __init__(self, boxes: GFMMBoxes, background, exact_max_d: int = 12,
                 n_permutations: int = 32, seed: int = 0, prune: bool = True):
        if n_permutations < 2 or n_permutations % 2:
            raise ValueError("n_permutations must be even and >= 2")
        self.boxes = boxes
        self.background_ = np.atleast_2d(np.asarray(background, dtype=float))
        self.exact_max_d = exact_max_d
        self.n_permutations = n_permutations
        self.seed = seed
        self.prune = prune
        self._TB = boxes.terms(self.background_)                  # (R, H, d)
        d = boxes.n_features
        self._tables = _shapley_tables(d) if d <= exact_max_d else None
        self.stats_: dict = {}

    @property
    def exact(self) -> bool:
        return self.boxes.n_features <= self.exact_max_d

    def _keep(self, Tx, isA):
        """(R, H): boxes that can be their group's maximum for some coalition."""
        lo = np.minimum(Tx[None], self._TB)
        hi = np.maximum(Tx[None], self._TB)
        floor, ceil = lo.min(axis=2), hi.min(axis=2)
        if not self.prune:
            return np.ones_like(floor, dtype=bool)
        keep = np.zeros_like(floor, dtype=bool)
        for grp in (isA, ~isA):
            if grp.any():
                best = floor[:, grp].max(axis=1, keepdims=True)
                keep[:, grp] = ceil[:, grp] >= best - 1e-12
        return keep

    def explain(self, X, seed: int | None = None, return_parts: bool = False):
        bx = self.boxes
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        cls = bx.predict_index(X)
        has_other = len(bx.classes_) > 1
        rng = np.random.default_rng(self.seed if seed is None else seed)
        phi = np.zeros((n, d))
        fx = np.zeros(n)
        f0 = np.zeros(n)
        R = len(self.background_)
        full = (1 << d) - 1
        kept = 0
        for p in range(n):
            Tx = bx.terms(X[p])[0]                                     # (H, d)
            isA = bx.box_cls == cls[p]
            keep = self._keep(Tx, isA)
            kept += int(keep.sum())
            if self.exact:
                v = np.zeros(1 << d)
                for r in range(R):
                    K = np.flatnonzero(keep[r])
                    b = np.minimum(subset_mins(Tx[K]), subset_mins(self._TB[r, K])[:, full - np.arange(full + 1)])
                    aK = isA[K]
                    GA = b[aK].max(axis=0)
                    v += GA - b[~aK].max(axis=0) if (has_other and (~aK).any()) else GA
                v /= R
                lo, wts = self._tables
                phi[p] = [np.dot(wts[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)]
                fx[p], f0[p] = v[-1], v[0]
            else:
                phi[p], fx[p], f0[p] = self._sampled(Tx, isA, keep, has_other, rng)
        H = len(bx.V)
        self.stats_ = {"n_points": n, "mode": "exact" if self.exact else "sampled",
                       "share_of_boxes_skipped": 1.0 - kept / max(1, n * R * H)}
        return (phi, fx, f0) if return_parts else phi

    __call__ = explain

    def _sampled(self, Tx, isA, keep, has_other, rng):
        P = self.n_permutations
        R, _, d = self._TB.shape
        phi = np.zeros(d)
        s0 = sN = 0.0
        for r in range(R):
            K = np.flatnonzero(keep[r])
            aK = isA[K]
            half = rng.permuted(np.tile(np.arange(d), (P // 2, 1)), axis=1)
            perms = np.empty((P, d), dtype=np.intp)
            perms[0::2] = half
            perms[1::2] = half[:, ::-1]
            TxK, TzK = Tx[K][:, perms], self._TB[r, K][:, perms]        # (K, P, d)
            # after t steps: min over the first t features of x, min over the rest of z
            pre = np.concatenate([np.full(TxK.shape[:2] + (1,), np.inf),
                                  np.minimum.accumulate(TxK, axis=2)], axis=2)       # (K, P, d+1)
            suf = np.concatenate([np.minimum.accumulate(TzK[:, :, ::-1], axis=2)[:, :, ::-1],
                                  np.full(TzK.shape[:2] + (1,), np.inf)], axis=2)    # (K, P, d+1)
            b = np.minimum(pre, suf)
            GA = b[aK].max(axis=0)                                       # (P, d+1)
            f = GA - b[~aK].max(axis=0) if (has_other and (~aK).any()) else GA
            inc = f[:, 1:] - f[:, :-1]                                    # (P, d)
            np.add.at(phi, perms.ravel(), inc.ravel())
            s0 += f[:, 0].sum()
            sN += f[:, -1].sum()
        total = R * P
        return phi / total, sN / total, s0 / total


def brute_force_min(boxes: GFMMBoxes, x, background, chunk_elems: int = 4_000_000) -> np.ndarray:
    """The same values by evaluating the margin on every mixed input (in blocks of coalitions)."""
    x = np.asarray(x, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(background, dtype=float))
    d = len(x)
    nm = 1 << d
    masks = np.arange(nm)
    bits = ((masks[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
    a = int(boxes.predict_index(x[None])[0])
    step = max(1, chunk_elems // max(1, len(Z) * len(boxes.V) * d))
    v = np.empty(nm)
    for s in range(0, nm, step):
        b = bits[s:s + step]
        Hy = np.where(b[:, None, :], x[None, None, :], Z[None, :, :]).reshape(-1, d)
        v[s:s + step] = boxes.margin(Hy, np.full(len(Hy), a)).reshape(len(b), len(Z)).mean(axis=1)
    lo, wts = _shapley_tables(d)
    return np.array([np.dot(wts[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)])


class PaceFlipMin:
    """PACE-flip for GFMM boxes: share of replacement values per feature that change the class."""

    def __init__(self, boxes: GFMMBoxes, values):
        self.boxes = boxes
        self.values_ = np.atleast_2d(np.asarray(values, dtype=float))      # (r, d)
        self._TV = boxes.terms(self.values_)                               # (r, H, d)

    def explain(self, X, return_parts: bool = False):
        bx = self.boxes
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = X.shape
        r = len(self.values_)
        flip = np.zeros((n, d))
        drop = np.zeros((n, d))
        cls = bx.predict_index(X)
        for p in range(n):
            Tx = bx.terms(X[p])[0]                                         # (H, d)
            pre = np.concatenate([np.full((len(Tx), 1), np.inf),
                                  np.minimum.accumulate(Tx, axis=1)[:, :-1]], axis=1)
            suf = np.concatenate([np.minimum.accumulate(Tx[:, ::-1], axis=1)[:, ::-1][:, 1:],
                                  np.full((len(Tx), 1), np.inf)], axis=1)
            other = np.minimum(pre, suf)                                   # (H, d): min over k != i
            isA = bx.box_cls == cls[p]
            m0 = bx.margin(X[p][None], [cls[p]])[0]
            for i in range(d):
                M = np.minimum(other[None, :, i], self._TV[:, :, i])        # (r, H)
                P = np.repeat(X[p][None], r, axis=0)
                P[:, i] = self.values_[:, i]
                c = bx._decide(M, P)
                flip[p, i] = np.mean(c != cls[p])
                GA = np.where(isA, M, -np.inf).max(axis=1)
                mg = GA - np.where(~isA, M, -np.inf).max(axis=1) if (~isA).any() else GA
                drop[p, i] = np.mean(m0 - mg)
        lo = drop.min(axis=1, keepdims=True)
        hi = drop.max(axis=1, keepdims=True)
        norm = np.divide(drop - lo, hi - lo, out=np.zeros_like(drop), where=(hi - lo) > 0)
        score = flip + norm * (0.5 / r)
        return (score, flip, drop) if return_parts else score
