"""How realistic are the inputs an explainer feeds the model? (RQ1 extension B3.)

Most explainers judge a feature by replacing it and asking the model again.
If the replaced inputs are unlike any real data ("off the data manifold"),
the explanation describes the model where it was never trained. This module
measures that.

* ``Manifold``: an input is **off-manifold** if its nearest training row is
  farther than the 95th percentile of the training rows' own
  nearest-neighbour distances (Euclidean, features divided by the training
  standard deviation). About 5% of real points are off by this rule.
* Samplers that draw the inputs each replacement scheme evaluates, ``k`` per
  explained point: a Shapley game (interventional or conditional),
  single-feature occlusion (fixed or per-point values), LIME's own samples,
  and the deletion and insertion tests (interventional or conditional).

The off-manifold share of a scheme is the mean over points of the
(weighted) share of its inputs that are off-manifold.
"""

from __future__ import annotations

import numpy as np

from .explain import cond_scale


class Manifold:
    """Off-manifold rule fitted on training rows (see the module docstring)."""

    def __init__(self, X_train, quantile: float = 0.95, max_ref: int = 2000, seed: int = 0):
        from sklearn.neighbors import NearestNeighbors
        X = np.asarray(X_train, float)
        self.scale_ = cond_scale(X)
        Xs = X / self.scale_
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(Xs)
        rng = np.random.default_rng(seed)
        ref = Xs if len(Xs) <= max_ref else Xs[rng.choice(len(Xs), max_ref, replace=False)]
        dist = self.nn_.kneighbors(ref, n_neighbors=2)[0][:, 1]   # nearest other row
        self.threshold_ = float(np.quantile(dist, quantile))

    def distance(self, Z) -> np.ndarray:
        """Distance from each input to its nearest training row."""
        Z = np.asarray(Z, float)
        return self.nn_.kneighbors(Z.reshape(-1, Z.shape[-1]) / self.scale_)[0][:, 0].reshape(Z.shape[:-1])

    def off(self, Z) -> np.ndarray:
        return self.distance(Z) > self.threshold_

    def share(self, Z, weights=None) -> float:
        """Mean over points of the weighted share of off-manifold inputs.

        ``Z`` is (n, k, d) (k inputs per point) or (k, d); ``weights`` matches
        Z's leading axes (equal weights if None)."""
        Z = np.asarray(Z, float)
        if Z.ndim == 2:
            Z = Z[None]
        off = self.off(Z).astype(float)                            # (n, k)
        w = np.ones_like(off) if weights is None else np.asarray(weights, float).reshape(off.shape)
        return float(((off * w).sum(axis=1) / w.sum(axis=1)).mean())


def _coalitions(n, d, k, rng):
    """(n, k, d) bool: the first t features of a random order, t uniform on 0..d.

    These are the coalitions a Shapley value visits along a feature order,
    each order and position equally often."""
    perms = rng.permuted(np.tile(np.arange(d), (n * k, 1)), axis=1)
    t = rng.integers(0, d + 1, size=n * k)
    S = np.arange(d)[None, :] < t[:, None]                          # first t positions
    out = np.zeros((n * k, d), dtype=bool)
    np.put_along_axis(out, perms, S, axis=1)
    return out.reshape(n, k, d)


def shapley_inputs(X, background, k: int, rng, sigma: float | None = None, scale=None):
    """Inputs a Shapley game evaluates: (n, k, d).

    Coalition S as in ``_coalitions``; features outside S from one background
    row, drawn uniformly (interventional game) or with probability
    proportional to the conditional weight exp(-D_S^2 / (2 sigma^2)) of
    ``PaceShap(sigma=...)``."""
    X = np.asarray(X, float)
    B = np.asarray(background, float)
    n, d = X.shape
    S = _coalitions(n, d, k, rng)
    if sigma is None:
        rows = rng.integers(len(B), size=(n, k))
    else:
        scale = cond_scale(B) if scale is None else np.asarray(scale, float)
        rows = np.empty((n, k), dtype=np.intp)
        for p in range(n):
            q = ((X[p] - B) / scale) ** 2                            # (R, d)
            size = np.maximum(S[p].sum(axis=1), 1)
            logw = -(S[p].astype(float) @ q.T) / (2.0 * sigma ** 2 * size[:, None])   # (k, R)
            w = np.exp(logw - logw.max(axis=1, keepdims=True))
            cdf = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
            u = rng.uniform(size=(k, 1))
            rows[p] = np.minimum((cdf < u).sum(axis=1), len(B) - 1)
    return np.where(S, X[:, None, :], B[rows])


def occlusion_inputs(X, values, k: int, rng):
    """Single-feature replacements: (n, k, d). Feature and value index uniform;
    ``values`` is (r, d) for all points or (n, r, d) per point."""
    X = np.asarray(X, float)
    n, d = X.shape
    vals = np.asarray(values, float)
    if vals.ndim == 2:
        vals = np.broadcast_to(vals, (n,) + vals.shape)
    feat = rng.integers(d, size=(n, k))
    pick = rng.integers(vals.shape[1], size=(n, k))
    Z = np.repeat(X[:, None, :], k, axis=1)
    pts = np.arange(n)[:, None]
    Z[pts, np.arange(k)[None, :], feat] = vals[pts, pick, feat]
    return Z


def lime_inputs(X, X_ref, k: int, rng):
    """LIME tabular's own samples and kernel weights: (n, k, d), (n, k).

    As ``LimeTabularExplainer(discretize_continuous=False)`` samples them:
    every feature independently from N(mean, std^2) of ``X_ref`` (not around
    the point), weighted by sqrt(exp(-dist^2 / width^2)), width = 0.75 sqrt(d),
    dist = Euclidean distance to the point on standardised features."""
    X = np.asarray(X, float)
    ref = np.asarray(X_ref, float)
    n, d = X.shape
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = rng.normal(size=(n, k, d)) * sd + mu
    dist = np.sqrt((((Z - X[:, None, :]) / sd) ** 2).sum(axis=2))
    width = 0.75 * np.sqrt(d)
    return Z, np.sqrt(np.exp(-dist ** 2 / width ** 2))


def curve_inputs(X, eval_bg, k: int, rng, conditional: bool, reps: int = 10, scale=None):
    """Inputs of the deletion and insertion tests: (n, k, d).

    The features kept from x form a coalition as in ``_coalitions``; the rest
    come from one evaluation row: a random one (interventional test), or the
    q-th nearest to x on the kept features, q uniform on 0..reps-1
    (conditional test; a random row when nothing is kept)."""
    X = np.asarray(X, float)
    B = np.asarray(eval_bg, float)
    n, d = X.shape
    S = _coalitions(n, d, k, rng)
    rows = rng.integers(len(B), size=(n, k))
    if conditional:
        scale = cond_scale(B) if scale is None else np.asarray(scale, float)
        q = rng.integers(0, min(reps, len(B)), size=(n, k))
        for p in range(n):
            dist = S[p].astype(float) @ (((X[p] - B) / scale) ** 2).T      # (k, R)
            near = np.argsort(dist, axis=1, kind="stable")
            some = S[p].any(axis=1)
            rows[p, some] = near[some, q[p, some]]
    return np.where(S, X[:, None, :], B[rows])
