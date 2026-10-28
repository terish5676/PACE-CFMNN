"""Perturbation and neighborhood generation for trust signals (Phase 3 / blueprint 6.1).

Generates reproducible local perturbations respecting feature types and the [0, 1] unit hypercube.
"""

from __future__ import annotations

import numpy as np


def detect_binary_columns(X: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Identify columns that take only values {0, 1} (binary features)."""
    X = np.asarray(X, dtype=float)
    is_binary = np.zeros(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        col = X[:, i]
        is_0 = np.isclose(col, 0.0, atol=tol)
        is_1 = np.isclose(col, 1.0, atol=tol)
        if np.all(is_0 | is_1) and (is_0.any() or is_1.any()):
            is_binary[i] = True
    return is_binary


def generate_perturbations(X: np.ndarray, n_perturb: int = 20, sigma: float = 0.05,
                          binary_cols: np.ndarray | None = None,
                          seed: int | None = 0) -> np.ndarray:
    """Generate K reproducible perturbations for each sample in X.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Input samples, scaled to [0, 1].
    n_perturb : int
        Number of perturbed copies per sample (K).
    sigma : float
        Perturbation scale (standard deviation for continuous features,
        flip probability for binary features).
    binary_cols : boolean array of shape (d,), optional
        Indicator of binary features. If None, detected from X.
    seed : int, optional
        Random seed for reproducible perturbations.

    Returns
    -------
    perturbations : np.ndarray of shape (n, n_perturb, d)
        Perturbed points, all clipped to [0, 1].
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n, d = X.shape
    if n_perturb <= 0:
        raise ValueError("n_perturb must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    if binary_cols is None:
        binary_cols = detect_binary_columns(X)
    else:
        binary_cols = np.asarray(binary_cols, dtype=bool)
        if len(binary_cols) != d:
            raise ValueError(f"binary_cols length {len(binary_cols)} does not match d={d}")

    rng = np.random.default_rng(seed)

    # Base shape: (n, n_perturb, d)
    # Broadcast X across perturbation axis
    pert = np.repeat(X[:, None, :], n_perturb, axis=1)

    if sigma == 0.0:
        return pert

    continuous_cols = ~binary_cols

    # Continuous features: additive Gaussian noise
    if continuous_cols.any():
        noise = rng.normal(0.0, sigma, size=(n, n_perturb, int(continuous_cols.sum())))
        pert[:, :, continuous_cols] += noise
        pert[:, :, continuous_cols] = np.clip(pert[:, :, continuous_cols], 0.0, 1.0)

    # Binary features: Bernoulli bit-flips with probability p = min(0.5, sigma)
    if binary_cols.any():
        p_flip = min(0.5, sigma)
        flips = rng.uniform(size=(n, n_perturb, int(binary_cols.sum()))) < p_flip
        bin_vals = pert[:, :, binary_cols]
        # Flip 0 -> 1 and 1 -> 0
        pert[:, :, binary_cols] = np.where(flips, 1.0 - bin_vals, bin_vals)

    return pert
