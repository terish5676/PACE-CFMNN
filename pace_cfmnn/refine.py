"""Selective sample refinement engine with Acceptance Gate protection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np

from .cfmnn_core import CFMNN
from .refinement import check_acceptance_gate


@dataclass
class RefinementConfig:
    margin_macro_f1: float = -0.5
    max_per_class_harm: float = 3.0
    max_collateral_pct: float = 4.0
    shrink_rate: float = 0.1


@dataclass
class RefinementResult:
    accepted: bool
    attempted: bool
    rolled_back: bool
    operation: str
    box_modified: int
    feature_modified: int
    reason: str


def refine_sample(
    model: CFMNN,
    x: np.ndarray,
    y_true: object,
    X_accept: np.ndarray,
    y_accept: np.ndarray,
    config: RefinementConfig | None = None,
) -> RefinementResult:
    """Selectively refine CFMNN on a single misclassified sample, protected by the Acceptance Gate.

    Parameters
    ----------
    model : CFMNN
        The model to potentially update in-place.
    x : np.ndarray, shape (d,)
        The acquired sample feature vector.
    y_true : label
        The ground truth label revealed by the oracle.
    X_accept, y_accept :
        Acceptance gate validation data.
    config : RefinementConfig, optional

    Returns
    -------
    RefinementResult
    """
    if config is None:
        config = RefinementConfig()

    x = np.asarray(x, dtype=float).ravel()
    det = model.predict_details(x[None, :])
    y_pred = det.y_pred[0]

    # 1. If already correct, no refinement needed
    if y_pred == y_true:
        return RefinementResult(
            accepted=True,
            attempted=False,
            rolled_back=False,
            operation="none",
            box_modified=-1,
            feature_modified=-1,
            reason="Prediction already matches true label",
        )

    # 2. Misclassified: identify false positive box j*
    j_star = det.winner[0]
    V_j = model.V_[j_star]
    W_j = model.W_[j_star]

    # Find the feature dimension i* with highest attribution or closest boundary
    dist_V = np.abs(x - V_j)
    dist_W = np.abs(x - W_j)
    min_dist = np.minimum(dist_V, dist_W)
    i_star = int(np.argmin(min_dist))

    # 3. Create candidate model with explanation-guided shrink
    candidate = copy.deepcopy(model)
    rate = config.shrink_rate
    delta = (candidate.W_[j_star, i_star] - candidate.V_[j_star, i_star]) * rate
    delta = max(delta, 1e-4)

    # Shrink the offending face away from x
    if dist_V[i_star] < dist_W[i_star]:
        candidate.V_[j_star, i_star] = min(candidate.V_[j_star, i_star] + delta, candidate.W_[j_star, i_star])
    else:
        candidate.W_[j_star, i_star] = max(candidate.W_[j_star, i_star] - delta, candidate.V_[j_star, i_star])

    candidate.centroid_[j_star] = np.clip(
        candidate.centroid_[j_star], candidate.V_[j_star], candidate.W_[j_star]
    )

    # 4. Check candidate against Acceptance Gate
    gate = check_acceptance_gate(
        model_orig=model,
        model_candidate=candidate,
        X_val=X_accept,
        y_val=y_accept,
        margin_macro_f1=config.margin_macro_f1,
        max_per_class_harm=config.max_per_class_harm,
        max_collateral_pct=config.max_collateral_pct,
    )

    if gate.passed:
        # Gate accepted candidate
        model.V_[j_star] = candidate.V_[j_star]
        model.W_[j_star] = candidate.W_[j_star]
        model.centroid_[j_star] = candidate.centroid_[j_star]
        return RefinementResult(
            accepted=True,
            attempted=True,
            rolled_back=False,
            operation="shrink",
            box_modified=j_star,
            feature_modified=i_star,
            reason="Passed acceptance gate",
        )
    else:
        # Gate rejected candidate -> roll back
        return RefinementResult(
            accepted=False,
            attempted=True,
            rolled_back=True,
            operation="shrink",
            box_modified=j_star,
            feature_modified=i_star,
            reason=f"Gate rejected: {'; '.join(gate.reasons)}",
        )
