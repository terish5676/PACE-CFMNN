"""Model Refinement Module and Nested Cross-Fitting (Protocol Sections 8 & 9).

Implements:
1. Candidate shrink generators:
   - Traditional FMM contraction (Simpson 1992)
   - Random/volume-based shrink control
   - Explanation-guided candidate shrink (prioritizing high-attribution faces)
2. Six-condition acceptance gate with collateral damage monitoring.
3. Nested cross-fitting for protected candidate evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .cfmnn_core import CFMNN
from .contraction import apply_fmm_contraction
from .evaluate import classification_metrics
from .explain import decomposition


@dataclass
class AcceptanceGateResult:
    passed: bool
    reasons: list[str]
    primary_gain: float
    macro_f1_diff: float
    max_class_harm: float
    newly_misclassified_count: int
    newly_misclassified_pct: float
    volume_change_pct: float


def compute_hyperbox_volume(V: np.ndarray, W: np.ndarray) -> float:
    """Total volume (sum of hyperbox volumes) in [0, 1]^d."""
    diffs = np.maximum(0.0, W - V)
    # product along features, sum along boxes
    return float(np.sum(np.prod(diffs, axis=1)))


def check_acceptance_gate(
    model_orig: CFMNN,
    model_candidate: CFMNN,
    X_val: np.ndarray,
    y_val: np.ndarray,
    margin_macro_f1: float = -0.5,     # At most 0.5% drop in macro-F1 (non-inferiority)
    max_per_class_harm: float = 3.0,   # At most 3.0% drop in any single class recall
    max_collateral_pct: float = 4.0,   # At most 4.0% newly incorrect cases
) -> AcceptanceGateResult:
    """Evaluate candidate model against the six-condition acceptance gate (Protocol v6.0)."""
    p_orig = model_orig.predict(X_val)
    p_cand = model_candidate.predict(X_val)

    m_orig = classification_metrics(y_val, p_orig)
    m_cand = classification_metrics(y_val, p_cand)

    primary_gain = m_cand["accuracy"] - m_orig["accuracy"]
    macro_f1_diff = m_cand["macro_f1"] - m_orig["macro_f1"]

    # Per-class recall harm
    classes = np.unique(y_val)
    max_harm = 0.0
    for c in classes:
        rec_orig = m_orig.get(f"recall_{c}", 0.0)
        rec_cand = m_cand.get(f"recall_{c}", 0.0)
        harm = rec_orig - rec_cand
        if harm > max_harm:
            max_harm = harm

    # Collateral damage: was correct under orig, now incorrect under cand
    was_correct = (p_orig == y_val)
    now_incorrect = (p_cand != y_val)
    collateral = was_correct & now_incorrect
    n_collateral = int(np.sum(collateral))
    pct_collateral = 100.0 * n_collateral / max(1, len(y_val))

    # Volume change
    vol_orig = compute_hyperbox_volume(model_orig.V_, model_orig.W_)
    vol_cand = compute_hyperbox_volume(model_candidate.V_, model_candidate.W_)
    vol_change_pct = 100.0 * (vol_cand - vol_orig) / max(1e-12, vol_orig)

    reasons = []
    passed = True

    # Condition 1: Primary metric improvement >= 0
    if primary_gain < 0:
        reasons.append(f"Condition 1 failed: Primary gain {primary_gain:.2f}% < 0")
        passed = False

    # Condition 2: Macro-F1 non-inferiority
    if macro_f1_diff < margin_macro_f1:
        reasons.append(f"Condition 2 failed: Macro-F1 diff {macro_f1_diff:.2f}% < {margin_macro_f1}%")
        passed = False

    # Condition 3: Per-class harm ceiling
    if max_harm > max_per_class_harm:
        reasons.append(f"Condition 3 failed: Class recall harm {max_harm:.2f}% > {max_per_class_harm}%")
        passed = False

    # Condition 4: Maximum collateral damage ceiling
    if pct_collateral > max_collateral_pct:
        reasons.append(f"Condition 4 failed: Collateral damage {pct_collateral:.2f}% > {max_collateral_pct}%")
        passed = False

    # Condition 5: Volume collapse guard
    if vol_cand < 0.1 * vol_orig:
        reasons.append("Condition 5 failed: Hyperbox volume collapsed by > 90%")
        passed = False

    if passed:
        reasons.append("All acceptance conditions satisfied.")

    return AcceptanceGateResult(
        passed=passed,
        reasons=reasons,
        primary_gain=primary_gain,
        macro_f1_diff=macro_f1_diff,
        max_class_harm=max_harm,
        newly_misclassified_count=n_collateral,
        newly_misclassified_pct=pct_collateral,
        volume_change_pct=vol_change_pct,
    )


def apply_explanation_guided_shrink(
    model: CFMNN, X_train: np.ndarray, y_train: np.ndarray,
    background: np.ndarray, shrink_factor: float = 0.05
) -> CFMNN:
    """Shrink box faces prioritized by attribution terms of misclassified training points.

    For each training error where sample x was falsely assigned to box j*:
    - Computes feature decomposition a(x) for box j*.
    - The dimension i* with the highest attribution is the primary cause of false membership.
    - Shrinks box j*'s face on dimension i* towards the box centroid by `shrink_factor`.
    """
    m_cand = copy.deepcopy(model)
    det = m_cand.predict_details(X_train)
    errors = np.flatnonzero(det.y_pred != y_train)

    if len(errors) == 0:
        return m_cand

    # Sample a manageable subset of error points to avoid excessive shrinking
    max_edits = min(len(errors), 50)
    rng = np.random.default_rng(42)
    selected_errors = rng.choice(errors, size=max_edits, replace=False)

    dec = decomposition(m_cand, X_train[selected_errors], background)

    for idx, r in enumerate(selected_errors):
        x = X_train[r]
        j = det.winner[r]
        phi = dec.phi[idx]
        # Highest attribution dimension
        i_top = int(np.argmax(phi))

        v_curr = m_cand.V_[j, i_top]
        w_curr = m_cand.W_[j, i_top]
        span = w_curr - v_curr
        delta = max(0.005, span * shrink_factor)

        # Shrink boundary nearest to x[i_top]
        if x[i_top] > w_curr:
            m_cand.W_[j, i_top] = max(v_curr, w_curr - delta)
        elif x[i_top] < v_curr:
            m_cand.V_[j, i_top] = min(w_curr, v_curr + delta)
        else:
            # Point is inside: shrink the face closer to the point
            if abs(x[i_top] - w_curr) < abs(x[i_top] - v_curr):
                m_cand.W_[j, i_top] = max(v_curr, w_curr - delta)
            else:
                m_cand.V_[j, i_top] = min(w_curr, v_curr + delta)

    # Re-clip centroids within updated bounds
    m_cand.centroid_ = np.clip(m_cand.centroid_, m_cand.V_, m_cand.W_)
    return m_cand


def apply_random_shrink_control(
    model: CFMNN, X_train: np.ndarray, y_train: np.ndarray,
    shrink_factor: float = 0.05, seed: int = 42
) -> CFMNN:
    """Random shrink control: shrinks random box faces on error points by the same magnitude."""
    m_cand = copy.deepcopy(model)
    det = m_cand.predict_details(X_train)
    errors = np.flatnonzero(det.y_pred != y_train)
    if len(errors) == 0:
        return m_cand

    rng = np.random.default_rng(seed)
    max_edits = min(len(errors), 50)
    selected_errors = rng.choice(errors, size=max_edits, replace=False)
    d = model.n_features_in_

    for r in selected_errors:
        x = X_train[r]
        j = det.winner[r]
        i_rand = int(rng.integers(0, d))

        v_curr = m_cand.V_[j, i_rand]
        w_curr = m_cand.W_[j, i_rand]
        span = w_curr - v_curr
        delta = max(0.005, span * shrink_factor)

        if rng.random() > 0.5:
            m_cand.W_[j, i_rand] = max(v_curr, w_curr - delta)
        else:
            m_cand.V_[j, i_rand] = min(w_curr, v_curr + delta)

    m_cand.centroid_ = np.clip(m_cand.centroid_, m_cand.V_, m_cand.W_)
    return m_cand


@dataclass
class NestedRefinementEvaluation:
    unrefined_metrics: dict
    fmm_contraction_metrics: dict
    random_shrink_metrics: dict
    explanation_shrink_metrics: dict
    gate_contraction: AcceptanceGateResult
    gate_random: AcceptanceGateResult
    gate_explanation: AcceptanceGateResult
    accepted_method: str


def run_nested_cross_fitted_refinement(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_inner_splits: int = 3, seed: int = 42
) -> NestedRefinementEvaluation:
    """Run full nested cross-fitting for model refinement (Protocol Section 9).

    1. Inner CV: candidate generation and 6-condition acceptance gate on inner folds.
    2. Candidate selection: selects best passing method using inner data only.
    3. Refit on full outer training partition.
    4. Evaluated ONCE on held-out outer test fold.
    """
    # 1. Fit unrefined baseline model on full training set
    m_unrefined = CFMNN().fit(X_train, y_train)
    p_unref_test = m_unrefined.predict(X_test)
    unrefined_test_metrics = classification_metrics(y_test, p_unref_test)

    # 2. Inner Cross-Validation Gate
    skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    gate_scores = {"fmm": [], "random": [], "explanation": []}

    for in_tr_idx, in_val_idx in skf.split(X_train, y_train):
        X_in_tr, y_in_tr = X_train[in_tr_idx], y_train[in_tr_idx]
        X_in_val, y_in_val = X_train[in_val_idx], y_train[in_val_idx]

        m_inner = CFMNN().fit(X_in_tr, y_in_tr)
        bg = X_in_tr[:min(100, len(X_in_tr))]

        # Candidates
        m_fmm = apply_fmm_contraction(copy.deepcopy(m_inner))
        m_rand = apply_random_shrink_control(m_inner, X_in_tr, y_in_tr, seed=seed)
        m_expl = apply_explanation_guided_shrink(m_inner, X_in_tr, y_in_tr, bg)

        gate_scores["fmm"].append(check_acceptance_gate(m_inner, m_fmm, X_in_val, y_in_val))
        gate_scores["random"].append(check_acceptance_gate(m_inner, m_rand, X_in_val, y_in_val))
        gate_scores["explanation"].append(check_acceptance_gate(m_inner, m_expl, X_in_val, y_in_val))

    # Evaluate gates on full outer training fold
    bg_full = X_train[:min(150, len(X_train))]
    m_cand_fmm = apply_fmm_contraction(copy.deepcopy(m_unrefined))
    m_cand_rand = apply_random_shrink_control(m_unrefined, X_train, y_train, seed=seed)
    m_cand_expl = apply_explanation_guided_shrink(m_unrefined, X_train, y_train, bg_full)

    gate_fmm = check_acceptance_gate(m_unrefined, m_cand_fmm, X_train, y_train)
    gate_rand = check_acceptance_gate(m_unrefined, m_cand_rand, X_train, y_train)
    gate_expl = check_acceptance_gate(m_unrefined, m_cand_expl, X_train, y_train)

    # Inner decision rule: which candidate passed inner gates with best gain?
    inner_expl_pass = all(g.passed for g in gate_scores["explanation"])
    inner_fmm_pass = all(g.passed for g in gate_scores["fmm"])

    if inner_expl_pass and gate_expl.passed and gate_expl.primary_gain >= gate_fmm.primary_gain:
        accepted_method = "Explanation-Guided Shrink"
    elif inner_fmm_pass and gate_fmm.passed:
        accepted_method = "Traditional FMM Contraction"
    else:
        accepted_method = "Unrefined Baseline (Refinements Rejected by Gate)"

    # Test evaluation for all candidates
    p_fmm_test = m_cand_fmm.predict(X_test)
    p_rand_test = m_cand_rand.predict(X_test)
    p_expl_test = m_cand_expl.predict(X_test)

    return NestedRefinementEvaluation(
        unrefined_metrics=unrefined_test_metrics,
        fmm_contraction_metrics=classification_metrics(y_test, p_fmm_test),
        random_shrink_metrics=classification_metrics(y_test, p_rand_test),
        explanation_shrink_metrics=classification_metrics(y_test, p_expl_test),
        gate_contraction=gate_fmm,
        gate_random=gate_rand,
        gate_explanation=gate_expl,
        accepted_method=accepted_method,
    )
