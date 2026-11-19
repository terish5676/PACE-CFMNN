"""Bounded, feature-aware perturbations for the trust signals (blueprint 6.1).

Each point gets K nearby inputs:

* continuous feature i: x_i + N(0, (kappa * std_i)^2), std_i from training rows,
  clipped to [0, 1];
* discrete feature (at most ``max_discrete`` distinct training values): with
  probability ``p_discrete`` the value is redrawn from the training values,
  in proportion to how often each occurs.

The policy is fitted on training rows only and is fixed before experiments.
Perturbations are reproducible for a given seed.
"""

from __future__ import annotations

import numpy as np


class Perturber:
    """Generate K perturbations per point.

    Parameters
    ----------
    kappa : float
        Noise scale for continuous features, as a fraction of each feature's
        training standard deviation (primary 0.05; sensitivity 0.02 and 0.1).
    k : int
        Perturbations per point (primary 50; sensitivity 20 and 100).
    p_discrete : float
        Probability of redrawing a discrete feature.
    max_discrete : int
        A feature with at most this many distinct training values is discrete.
    """

    def __init__(self, kappa: float = 0.05, k: int = 50, p_discrete: float = 0.1,
                 max_discrete: int = 10):
        if kappa < 0 or k < 1 or not 0 <= p_discrete <= 1:
            raise ValueError("need kappa >= 0, k >= 1 and 0 <= p_discrete <= 1")
        self.kappa = kappa
        self.k = k
        self.p_discrete = p_discrete
        self.max_discrete = max_discrete

    def fit(self, X_train) -> "Perturber":
        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X_train must be a non-empty 2-D array")
        d = X.shape[1]
        self.std_ = X.std(axis=0)
        self.discrete_ = np.zeros(d, dtype=bool)
        self.values_: list[np.ndarray | None] = [None] * d
        self.probs_: list[np.ndarray | None] = [None] * d
        for i in range(d):
            vals, counts = np.unique(X[:, i], return_counts=True)
            if len(vals) <= self.max_discrete:
                self.discrete_[i] = True
                self.values_[i] = vals
                self.probs_[i] = counts / counts.sum()
        self.n_features_ = d
        return self

    def sample(self, X, seed=0) -> np.ndarray:
        """Perturbations of shape (n, k, d). ``seed`` may be an int or a Generator."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n, d = X.shape
        Z = np.repeat(X[:, None, :], self.k, axis=1)
        cont = ~self.discrete_
        if cont.any():
            noise = rng.normal(size=(n, self.k, int(cont.sum()))) * (self.kappa * self.std_[cont])
            Z[:, :, cont] = np.clip(Z[:, :, cont] + noise, 0.0, 1.0)
        for i in np.flatnonzero(self.discrete_):
            redraw = rng.random((n, self.k)) < self.p_discrete
            new = rng.choice(self.values_[i], size=(n, self.k), p=self.probs_[i])
            Z[:, :, i] = np.where(redraw, new, Z[:, :, i])
        return Z

    def clip_share(self, X, Z) -> float:
        """Share of continuous coordinates that landed on 0 or 1 although x did not."""
        X = np.asarray(X, dtype=float)
        cont = ~self.discrete_
        if not cont.any():
            return 0.0
        z = Z[:, :, cont]
        x = np.repeat(X[:, None, cont], Z.shape[1], axis=1)
        edge = ((z == 0.0) | (z == 1.0)) & (x != z)
        return float(edge.mean())
