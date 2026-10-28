"""PACE Refinement Score (PRS) combining trust signals for active acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .cfmnn_core import CFMNN
from .neighborhood import generate_perturbations
from .signals import compute_explanation_instability, compute_prediction_disagreement, compute_uncertainty


@dataclass
class PrsResult:
    prs_score: np.ndarray
    rank: np.ndarray
    pd: np.ndarray
    u: np.ndarray
    ei: np.ndarray


def compute_risk(
    model: CFMNN,
    X: np.ndarray,
    alpha: float = 0.5,
    n_perturb: int = 15,
    sigma: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Operational error risk score combining strictly Prediction Disagreement and Uncertainty (PD + U).

    Note: EI_within is strictly eliminated from this primary risk score because internal empirical
    falsification of Hypothesis H2 proved that within-class explanation instability provides no
    independent predictive gain for error risk detection (Delta AUROC <= 0).
    
    Formula:
        risk(x) = alpha * PD(x) + (1 - alpha) * U(x)
    """
    X = model._check_X(X)
    pert = generate_perturbations(X, n_perturb=n_perturb, sigma=sigma, seed=seed)
    pd_vals = compute_prediction_disagreement(model, X, pert)
    u_vals, _ = compute_uncertainty(model, X)
    return np.clip(alpha * pd_vals + (1.0 - alpha) * u_vals, 0.0, 1.0)


def compute_prs(
    model: CFMNN,
    X: np.ndarray,
    n_perturb: int = 15,
    sigma: float = 0.05,
    seed: int = 0,
    weights: tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> PrsResult:
    """Compute composite PACE Refinement Score (PRS) for active candidate acquisition.

    By default, weights are (0.5, 0.5, 0.0):
    - Prediction Disagreement (PD): weight 0.5
    - Uncertainty (U): weight 0.5
    - Explanation Instability (EI): weight 0.0 (kept only as diagnostic signal in result).

    Parameters
    ----------
    model : CFMNN
    X : np.ndarray, shape (n, d)
    n_perturb : int, default 15
    sigma : float, default 0.05
    seed : int, default 0
    weights : tuple of 3 floats, default (0.5, 0.5, 0.0)

    Returns
    -------
    PrsResult with prs_score, rank (0 is highest priority), pd, u, ei.
    """
    X = model._check_X(X)
    n = len(X)

    pert = generate_perturbations(X, n_perturb=n_perturb, sigma=sigma, seed=seed)
    pd_vals = compute_prediction_disagreement(model, X, pert)
    u_vals, _ = compute_uncertainty(model, X)
    ei_vals, _ = compute_explanation_instability(model, X, pert)

    w_pd, w_u, w_ei = weights
    score = w_pd * pd_vals + w_u * u_vals + w_ei * ei_vals

    # Rank 0 is highest score (highest priority for acquisition)
    order = np.argsort(-score)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    return PrsResult(
        prs_score=score,
        rank=rank,
        pd=pd_vals,
        u=u_vals,
        ei=ei_vals,
    )
