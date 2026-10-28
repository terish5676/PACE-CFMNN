"""Trust and uncertainty signal computation for active learning and diagnostics.

Implements:
- compute_uncertainty: continuous membership ambiguity U(x)
- compute_prediction_disagreement: local perturbation disagreement PD(x)
- compute_explanation_instability: within-class attribution instability EI(x)
"""

from __future__ import annotations

import numpy as np
from .cfmnn_core import CFMNN
from .explain import decomposition
from .trust_diagnostics import compute_u


def compute_uncertainty(model: CFMNN, X: np.ndarray, eta: float = 0.5) -> tuple[np.ndarray, dict]:
    """Compute continuous membership ambiguity U(x) in [0, 1]."""
    X = model._check_X(X)
    u = compute_u(model, X, eta=eta)
    det = model.predict_details(X)
    return u, {"support": det.support, "margin": 1.0 - u}


def compute_prediction_disagreement(model: CFMNN, X: np.ndarray, pert: np.ndarray) -> np.ndarray:
    """Compute Prediction Disagreement PD(x) given perturbation array.

    Parameters
    ----------
    model : CFMNN
    X : np.ndarray, shape (n, d)
    pert : np.ndarray, shape (n, n_perturb, d)

    Returns
    -------
    pd_vals : np.ndarray, shape (n,)
    """
    X = model._check_X(X)
    n = len(X)
    n_perturb = pert.shape[1]

    y_orig = model.predict(X)
    pd_vals = np.zeros(n, dtype=float)

    for i in range(n):
        y_pert = model.predict(pert[i])
        pd_vals[i] = np.mean(y_pert != y_orig[i])

    return pd_vals


def compute_explanation_instability(
    model: CFMNN,
    X: np.ndarray,
    pert: np.ndarray,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute within-class explanation instability EI(x) under perturbations.

    Parameters
    ----------
    model : CFMNN
    X : np.ndarray, shape (n, d)
    pert : np.ndarray, shape (n, n_perturb, d)
    background : np.ndarray, optional

    Returns
    -------
    ei_vals : np.ndarray, shape (n,)
    info : dict
    """
    X = model._check_X(X)
    n, d = X.shape
    n_perturb = pert.shape[1]
    if background is None:
        background = model.centroid_

    y_orig = model.predict(X)
    det_orig = model.predict_details(X)
    orig_winner = det_orig.winner

    # Compute base attributions
    dec_orig = decomposition(model, X, background)
    a_orig = dec_orig.phi
    norm_orig = np.abs(a_orig).sum(axis=1, keepdims=True)
    a_orig_norm = np.divide(a_orig, norm_orig, out=np.zeros_like(a_orig), where=norm_orig > 0)

    ei_vals = np.zeros(n, dtype=float)
    eligible_counts = np.zeros(n, dtype=int)

    for i in range(n):
        det_pert = model.predict_details(pert[i])
        y_pert = det_pert.cls
        winner_pert = det_pert.winner

        # Eligible perturbations: same class and same winning box
        eligible = (y_pert == y_orig[i]) & (winner_pert == orig_winner[i])
        eligible_counts[i] = int(np.sum(eligible))

        if eligible_counts[i] > 0:
            dec_p = decomposition(model, pert[i][eligible], background)
            a_p = dec_p.phi
            norm_p = np.abs(a_p).sum(axis=1, keepdims=True)
            a_p_norm = np.divide(a_p, norm_p, out=np.zeros_like(a_p), where=norm_p > 0)

            # L1 distance
            dist = 0.5 * np.abs(a_orig_norm[i] - a_p_norm).sum(axis=1)
            ei_vals[i] = float(np.mean(dist))
        else:
            ei_vals[i] = 1.0  # Maximum instability if no perturbation stayed in same box

    return ei_vals, {"eligible_counts": eligible_counts}
