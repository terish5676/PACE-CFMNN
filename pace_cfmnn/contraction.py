"""Traditional Fuzzy Min-Max (FMM) Contraction baseline.

Implements Simpson's (1992) standard 4-case hyperbox overlap test and contraction
algorithm (also detailed in Gabrys & Bargiela 2000, and Liu et al. 2011).
Used as a standard baseline to compare with explanation-guided refinement.
"""

from __future__ import annotations

import numpy as np
from .cfmnn_core import CFMNN


def test_box_overlap(v_j: np.ndarray, w_j: np.ndarray,
                     v_k: np.ndarray, w_k: np.ndarray) -> tuple[bool, int, int]:
    """Test overlap between box j and box k across all d dimensions.

    Returns
    -------
    overlaps : bool
        True if the two hyperboxes overlap in all dimensions.
    min_dim : int
        The dimension index with the minimal overlap (delta_min).
    case_type : int
        The contraction case (1, 2, 3, or 4) on dimension min_dim.
    """
    d = len(v_j)
    delta_min = np.inf
    dim_min = -1
    case_min = -1

    for i in range(d):
        vji, wji = v_j[i], w_j[i]
        vki, wki = v_k[i], w_k[i]

        # Case 1: vji < vki < wji < wki
        if vji < vki and vki < wji and wji < wki:
            delta = wji - vki
            case = 1
        # Case 2: vki < vji < wki < wji
        elif vki < vji and vji < wki and wki < wji:
            delta = wki - vji
            case = 2
        # Case 3: vji <= vki and wki <= wji
        elif vji <= vki and wki <= wji:
            delta = min(wji - vki, wki - vji)
            case = 3
        # Case 4: vki <= vji and wji <= wki
        elif vki <= vji and wji <= wki:
            delta = min(wki - vji, wji - vki)
            case = 4
        else:
            # No overlap in this dimension
            return False, -1, -1

        if delta < delta_min:
            delta_min = delta
            dim_min = i
            case_min = case

    return True, dim_min, case_min


def contract_boxes(v_j: np.ndarray, w_j: np.ndarray,
                   v_k: np.ndarray, w_k: np.ndarray,
                   dim: int, case: int) -> None:
    """Apply Simpson's 4-case contraction in-place on dimension `dim`."""
    if case == 1:
        # vji < vki < wji < wki
        mid = (w_j[dim] + v_k[dim]) / 2.0
        w_j[dim] = mid
        v_k[dim] = mid
    elif case == 2:
        # vki < vji < wki < wji
        mid = (w_k[dim] + v_j[dim]) / 2.0
        w_k[dim] = mid
        v_j[dim] = mid
    elif case == 3:
        # vji <= vki <= wki <= wji
        if (w_k[dim] - v_j[dim]) < (w_j[dim] - v_k[dim]):
            v_j[dim] = w_k[dim]
        else:
            w_j[dim] = v_k[dim]
    elif case == 4:
        # vki <= vji <= wji <= wki
        if (w_k[dim] - v_j[dim]) < (w_j[dim] - v_k[dim]):
            w_k[dim] = v_j[dim]
        else:
            v_k[dim] = w_j[dim]


def apply_fmm_contraction(model: CFMNN, max_passes: int = 5) -> CFMNN:
    """Apply standard FMM contraction to resolve overlaps between hyperboxes of different classes.

    Parameters
    ----------
    model : CFMNN
        Fitted CFMNN model.
    max_passes : int
        Maximum number of contraction sweep passes.

    Returns
    -------
    model : CFMNN
        The modified model with contracted hyperbox boundaries.
    """
    n_boxes = model.n_boxes_
    if n_boxes < 2:
        return model

    # Copy bounds so we do not mutate if user wants clean copy
    V = model.V_.copy()
    W = model.W_.copy()
    box_class = model.box_class_

    n_contractions = 0
    for _ in range(max_passes):
        changed = False
        # Fast 1D projection dimension for sweep-line candidate filtering
        spreads = np.ptp(W, axis=0)
        proj_dim = int(np.argmax(spreads)) if len(spreads) > 0 else 0

        for j in range(n_boxes):
            diff_mask = np.zeros(n_boxes, dtype=bool)
            diff_mask[j + 1:] = (box_class[j + 1:] != box_class[j])
            # 1D Sweep-line projection filter along max-spread axis
            proj_overlap = (V[j, proj_dim] < W[:, proj_dim]) & (W[j, proj_dim] > V[:, proj_dim])
            valid_cand = diff_mask & proj_overlap
            if not np.any(valid_cand):
                continue
            cand_k = np.flatnonzero(valid_cand)
            # Vectorized overlap check: max(V_j, V_k) < min(W_j, W_k) across all dimensions
            overlap_mask = (np.maximum(V[j], V[cand_k]) < np.minimum(W[j], W[cand_k])).all(axis=1)
            overlapping_k = cand_k[overlap_mask]

            for k in overlapping_k:
                overlaps, dim, case = test_box_overlap(V[j], W[j], V[k], W[k])
                if overlaps and dim >= 0:
                    contract_boxes(V[j], W[j], V[k], W[k], dim, case)
                    n_contractions += 1
                    changed = True
        if not changed:
            break


    # Re-verify V <= W
    np.minimum(V, W, out=V)
    np.maximum(V, W, out=W)

    model.V_ = V
    model.W_ = W
    # Update centroids if needed
    model.centroid_ = np.clip(model.centroid_, V, W)
    model.fit_stats_["n_contractions"] = n_contractions
    return model


def safe_apply_fmm_contraction(
    model: CFMNN,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_passes: int = 5,
    margin_macro_f1: float = -0.5,
    max_per_class_harm: float = 3.0,
    max_collateral_pct: float = 4.0,
) -> tuple[CFMNN, bool, list[str]]:
    """Apply FMM contraction safely protected by the Six-Condition Acceptance Gate.

    Parameters
    ----------
    model : CFMNN
        Fitted model.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation labels.
    max_passes : int
        Maximum number of contraction passes.
    margin_macro_f1 : float
        Allowed drop in macro-F1 before rejection (default -0.5%).
    max_per_class_harm : float
        Allowed drop in per-class recall (default 3.0%).
    max_collateral_pct : float
        Maximum newly misclassified instances allowed (default 4.0%).

    Returns
    -------
    model : CFMNN
        The updated model (or unchanged model if gate rejected).
    passed : bool
        True if contraction passed all 6 acceptance gate conditions.
    reasons : list[str]
        Detailed gate diagnostic messages.
    """
    import copy
    from .refinement import check_acceptance_gate

    candidate = copy.deepcopy(model)
    candidate = apply_fmm_contraction(candidate, max_passes=max_passes)

    gate = check_acceptance_gate(
        model_orig=model,
        model_candidate=candidate,
        X_val=X_val,
        y_val=y_val,
        margin_macro_f1=margin_macro_f1,
        max_per_class_harm=max_per_class_harm,
        max_collateral_pct=max_collateral_pct,
    )

    if gate.passed:
        model.V_ = candidate.V_
        model.W_ = candidate.W_
        model.centroid_ = candidate.centroid_
        model.fit_stats_["n_contractions"] = candidate.fit_stats_.get("n_contractions", 0)
        model.fit_stats_["gate_passed"] = True
        return model, True, gate.reasons
    else:
        model.fit_stats_["gate_passed"] = False
        model.fit_stats_["gate_reasons"] = gate.reasons
        return model, False, gate.reasons
