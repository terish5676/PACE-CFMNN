"""PACE-decision: exact Shapley values of CFMNN's decision, ties included (tough review B3).

PACE-SHAP explains the margin s'_A - max_{c != A} s'_c, which is 0 at a tie and
leaves out the nearest-centroid rule that then picks the class. PACE-decision
explains the decision itself, with the interventional game

    v(S) = mean over background rows z of 1[ class(x_S, z_rest) = A ],

where class() is CFMNN's full prediction rule: the highest membership, boxes
within ``tie_tol`` of it tied, the nearest centroid among them, then the
lowest box index. Both quantities the rule reads are additive over features:

    b_j(S)  = b_j(z) + sum_{i in S} Delta_{j,i}                    (Proposition 2)
    D_j(S)  = D_j(z) + sum_{i in S} [(x_i - mu_{j,i})^2 - (z_i - mu_{j,i})^2]

so the class of every coalition comes from two tables. Up to ``exact_max_d``
features every coalition is evaluated (exact); above that, antithetic feature
orders are sampled, as in PACE-SHAP. Boxes that can never reach the tied top
group for a background row (ceiling below the best floor minus tie_tol and
1e-12) are skipped. The values add up to 1 - v(empty set).

Rounding: the tables add the same terms in another order than ``predict``.
Memberships tie within tie_tol, so this does not matter for them. Squared
distances are compared exactly, so two boxes at exactly the same centroid
distance could be ordered differently. This does not happen with continuous
data, and the tests check against the model's own prediction.
"""

from __future__ import annotations

import math

import numpy as np

from .cfmnn_core import CFMNN, feature_terms
from .explain import subset_sums


class PaceDecision:
    def __init__(self, model: CFMNN, background, exact_max_d: int = 12,
                 n_permutations: int = 32, seed: int = 0):
        if n_permutations < 2 or n_permutations % 2:
            raise ValueError("n_permutations must be even and >= 2 (antithetic pairs)")
        self.model = model
        self.background_ = model._check_X(background)
        self.exact_max_d = exact_max_d
        self.n_permutations = n_permutations
        self.seed = seed
        m = model
        self._TB = feature_terms(self.background_[:, None, :], m.V_, m.W_, m.gamma)   # (R, H, d)
        self._MB = self._TB.mean(axis=2)
        self._DB = ((self.background_[:, None, :] - m.centroid_[None]) ** 2)          # (R, H, d)
        self._DZ = self._DB.sum(axis=2)
        d = self._TB.shape[2]
        self.stats_: dict = {}
        if d <= exact_max_d:
            masks = np.arange(1 << d)
            size = np.array([bin(k).count("1") for k in masks])
            fact = [math.factorial(k) for k in range(d + 1)]
            w = np.array([fact[s] * fact[d - s - 1] / fact[d] for s in range(d)])
            lo = [masks[(masks >> i) & 1 == 0] for i in range(d)]
            self._tables = (lo, [w[size[l]] for l in lo])
        else:
            self._tables = None

    @property
    def exact(self) -> bool:
        return self._TB.shape[2] <= self.exact_max_d

    def _row_parts(self, x):
        m = self.model
        R, H, d = self._TB.shape
        Tx = feature_terms(x, m.V_, m.W_, m.gamma)                     # (H, d)
        Delta = (Tx[None] - self._TB) / d                              # (R, H, d)
        Dx = (x[None, :] - m.centroid_) ** 2                           # (H, d)
        dD = Dx[None] - self._DB                                       # (R, H, d)
        ceil = self._MB + np.maximum(Delta, 0).sum(axis=2)
        floor = self._MB + np.minimum(Delta, 0).sum(axis=2)
        keep = ceil >= floor.max(axis=1, keepdims=True) - float(m.tie_tol) - 1e-12
        return Delta, dD, keep

    def _classes(self, vals, dists, boxes):
        """Class index of each column: vals, dists are (K, cols) for boxes ``boxes`` (ascending)."""
        tol = float(self.model.tie_tol)
        top = vals.max(axis=0)
        tied = vals >= top[None, :] - tol
        win = np.argmin(np.where(tied, dists, np.inf), axis=0)
        return self.model.box_class_[boxes[win]]

    def explain(self, X, seed: int | None = None, return_parts: bool = False):
        """Shapley values (n, d) of the decision game. With ``return_parts``: (phi, v(N), v(empty))."""
        m = self.model
        X = m._check_X(X)
        n, d = X.shape
        cls = m.predict_details(X).cls
        rng = np.random.default_rng(self.seed if seed is None else seed)
        phi = np.zeros((n, d))
        vN = np.zeros(n)
        v0 = np.zeros(n)
        kept = 0
        R = len(self.background_)
        for p in range(n):
            Delta, dD, keep = self._row_parts(X[p])
            kept += int(keep.sum())
            a = int(cls[p])
            if self.exact:
                v = np.zeros(1 << d)
                for r in range(R):
                    K = np.flatnonzero(keep[r])
                    vals = subset_sums(self._MB[r, K], Delta[r, K])
                    dists = subset_sums(self._DZ[r, K], dD[r, K])
                    v += self._classes(vals, dists, K) == a
                v /= R
                lo, wts = self._tables
                phi[p] = [np.dot(wts[i], v[lo[i] | (1 << i)] - v[lo[i]]) for i in range(d)]
                vN[p], v0[p] = v[-1], v[0]
            else:
                phi[p], vN[p], v0[p] = self._sampled(Delta, dD, keep, a, rng)
        H = m.n_boxes_
        self.stats_ = {"n_points": n, "mode": "exact" if self.exact else "sampled",
                       "share_of_boxes_skipped": 1.0 - kept / max(1, n * R * H)}
        return (phi, vN, v0) if return_parts else phi

    __call__ = explain

    def _sampled(self, Delta, dD, keep, a, rng):
        P = self.n_permutations
        R, _, d = Delta.shape
        phi = np.zeros(d)
        f0 = fN = 0.0
        for r in range(R):
            K = np.flatnonzero(keep[r])
            half = rng.permuted(np.tile(np.arange(d), (P // 2, 1)), axis=1)
            perms = np.empty((P, d), dtype=np.intp)
            perms[0::2] = half
            perms[1::2] = half[:, ::-1]
            vals = np.repeat(self._MB[r, K][:, None], P, axis=1)        # (K, P)
            dists = np.repeat(self._DZ[r, K][:, None], P, axis=1)
            dK, sK = Delta[r, K], dD[r, K]                              # (K, d)
            prev = (self._classes(vals, dists, K) == a).astype(float)
            f0 += prev.sum()
            for t in range(d):
                feat = perms[:, t]
                vals += dK[:, feat]
                dists += sK[:, feat]
                cur = (self._classes(vals, dists, K) == a).astype(float)
                phi += np.bincount(feat, weights=cur - prev, minlength=d)
                prev = cur
            fN += prev.sum()
        total = R * P
        return phi / total, fN / total, f0 / total


def brute_force_decision(model: CFMNN, x, background) -> np.ndarray:
    """The decision game's exact Shapley values by running ``model.predict`` on every mixed input."""
    x = model._check_X(x)[0]
    Z = model._check_X(background)
    d = len(x)
    nm = 1 << d
    masks = np.arange(nm)
    bits = ((masks[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
    a = int(model.predict_details(x[None]).cls[0])
    Hy = np.where(bits[:, None, :], x[None, None, :], Z[None, :, :]).reshape(-1, d)
    v = (model.predict_details(Hy).cls == a).reshape(nm, len(Z)).mean(axis=1)
    size = bits.sum(axis=1)
    fact = [math.factorial(k) for k in range(d + 1)]
    w = np.array([fact[s] * fact[d - s - 1] / fact[d] for s in range(d)])
    phi = np.zeros(d)
    for i in range(d):
        lo = masks[~bits[:, i]]
        phi[i] = np.dot(w[size[lo]], v[lo | (1 << i)] - v[lo])
    return phi
