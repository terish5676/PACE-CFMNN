"""Exact single-feature flip sets for CFMNN (tough review B8).

PACE-flip tries 32 replacement values per feature. The decision as one feature
varies is simple enough to compute exactly. Fix a point x and a feature i, and
let u be the new value of feature i. Then:

* every box's membership is c_j + t_j(u) / d, with c_j fixed and t_j piecewise
  linear in u, with breakpoints at v - 1/gamma, v, w and w + 1/gamma;
* every squared centroid distance is a fixed part plus (u - mu_{j,i})^2, so the
  difference between two boxes' distances is linear in u.

The prediction (the highest membership, ties within ``tie_tol`` to the nearest
centroid, then to the lowest index) can therefore change only at:

* a breakpoint;
* a point where two memberships differ by 0 or by +-tie_tol;
* a point where two distances are equal.

``flip_sets`` collects these candidate points for the boxes that can reach the
tied top group. It evaluates the model at each candidate and at each midpoint
between candidates, and returns the class on every piece of the domain.

Boxes whose ceiling over u lies below the best floor minus tie_tol (and
1e-12) are never tied at the top, so they can be left out (as in
``PaceFlip``). The model itself, with every box, gives the class at each
evaluation point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN, feature_terms


@dataclass
class FlipSet:
    feature: int
    breaks: np.ndarray      # sorted candidate points, including both domain ends
    point_cls: np.ndarray   # class index at each candidate point
    open_cls: np.ndarray    # class index on each open interval (breaks[k], breaks[k + 1])
    cls_x: int              # the predicted class at x

    def flip_intervals(self) -> list[tuple[float, float]]:
        """Maximal pieces of the domain where the class differs from cls_x.

        Pieces are given as closed [a, b]; whether an end point itself flips is
        in ``point_cls``.
        """
        out: list[list[float]] = []
        for k, c in enumerate(self.open_cls):
            if c == self.cls_x:
                continue
            a, b = float(self.breaks[k]), float(self.breaks[k + 1])
            if out and abs(out[-1][1] - a) == 0.0 and self.point_cls[k] != self.cls_x:
                out[-1][1] = b
            else:
                out.append([a, b])
        # isolated flipping points between two non-flipping pieces
        for k, c in enumerate(self.point_cls):
            if c != self.cls_x:
                p = float(self.breaks[k])
                if not any(lo <= p <= hi for lo, hi in out):
                    out.append([p, p])
        return sorted((lo, hi) for lo, hi in out)

    def smallest_change(self, x_i: float) -> tuple[float, bool]:
        """Distance from x_i to the flip set, and whether a flipping value attains it.

        Returns (inf, False) if no value of this feature flips the decision.
        """
        best, attained = np.inf, False
        for k, c in enumerate(self.point_cls):
            if c != self.cls_x:
                dist = abs(float(self.breaks[k]) - x_i)
                if dist < best:
                    best, attained = dist, True
        for k, c in enumerate(self.open_cls):
            if c == self.cls_x:
                continue
            a, b = float(self.breaks[k]), float(self.breaks[k + 1])
            if a < x_i < b:
                return 0.0, True
            dist = min(abs(a - x_i), abs(b - x_i))
            if dist < best:
                best, attained = dist, False
        return best, attained


def _term_breaks(v, w, gamma):
    r = 1.0 / gamma
    return np.stack([v - r, v, w, w + r], axis=-1)


def flip_sets(model: CFMNN, x, domain=None, features=None) -> list[FlipSet]:
    """The exact class of the model as each feature of x varies over ``domain`` (default [0, 1]).

    ``domain`` is widened to contain x_i. Returns one ``FlipSet`` per feature.
    """
    x = model._check_X(x)[0]
    d = model.n_features_in_
    feats = range(d) if features is None else features
    g, tol = float(model.gamma), float(model.tie_tol)
    T = feature_terms(x[None, :], model.V_, model.W_, g)           # (H, d)
    M = T.mean(axis=1)
    mu = model.centroid_
    Dx = ((x[None, :] - mu) ** 2).sum(axis=1)
    cls_x = int(model.predict_details(x[None]).cls[0])
    out = []
    for i in feats:
        lo, hi = (0.0, 1.0) if domain is None else map(float, domain)
        lo, hi = min(lo, float(x[i])), max(hi, float(x[i]))
        c = M - T[:, i] / d
        v, w = model.V_[:, i], model.W_[:, i]
        t_lo = feature_terms(np.array([[lo]]), v[:, None], w[:, None], g)[:, 0]
        t_hi = feature_terms(np.array([[hi]]), v[:, None], w[:, None], g)[:, 0]
        overlaps = (w >= lo) & (v <= hi)
        t_max = np.where(overlaps, 1.0, np.maximum(t_lo, t_hi))
        ceil = c + t_max / d
        floor = c + np.minimum(t_lo, t_hi) / d
        keep = np.flatnonzero(ceil >= floor.max() - tol - 1e-12)
        cand = [np.array([lo, hi])]
        bk = _term_breaks(v[keep], w[keep], g).ravel()
        cand.append(bk[(bk > lo) & (bk < hi)])     # breakpoints outside the domain are not candidates
        pts = np.unique(np.clip(np.r_[lo, hi, bk], lo, hi))
        # Piece by piece (memberships are linear between consecutive breakpoints): only boxes
        # whose value on the piece reaches the best floor minus tie_tol can be in the tied top
        # group there, so only their pairs can change the prediction. For those pairs: membership
        # crossings at 0 and +-tol, and centroid-distance crossings (D_j - D_k is linear in u).
        cK, vK, wK = c[keep], v[keep], w[keep]
        muK = mu[keep, i]
        baseK = Dx[keep] - (x[i] - muK) ** 2 + muK ** 2          # D_j(u) = baseK_j - 2 u mu_j + u^2
        F = cK[None, :] + feature_terms(pts[:, None, None], vK[:, None], wK[:, None], g)[:, :, 0] / d
        Fa, Fb = F[:-1], F[1:]                                     # (pieces, K) at the two ends
        top = np.maximum(Fa, Fb) >= np.minimum(Fa, Fb).max(axis=1, keepdims=True) - tol - 1e-12
        for k in np.flatnonzero(top.sum(axis=1) >= 2):
            a, b = pts[k], pts[k + 1]
            if b <= a:
                continue
            C = np.flatnonzero(top[k])
            slope = (Fb[k, C] - Fa[k, C]) / (b - a)
            icpt = Fa[k, C] - slope * a
            ds = slope[:, None] - slope[None, :]
            di = icpt[:, None] - icpt[None, :]
            db = baseK[C][:, None] - baseK[C][None, :]
            dm = -2.0 * (muK[C][:, None] - muK[C][None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                roots = [(off - di) / ds for off in (0.0, tol, -tol)] + [-db / dm]
            for root in roots:
                r = root[np.isfinite(root) & (root > a) & (root < b)]
                if len(r):
                    cand.append(r)
        breaks = np.unique(np.concatenate(cand))
        mids = 0.5 * (breaks[:-1] + breaks[1:])
        U = np.r_[breaks, mids]
        P = np.repeat(x[None, :], len(U), axis=0)
        P[:, i] = U
        cl = model.predict_details(P).cls
        out.append(FlipSet(int(i), breaks, cl[:len(breaks)], cl[len(breaks):], cls_x))
    return out
