"""Trust & Uncertainty Diagnostics and Central Hypothesis Testing.

Implements:
1. Continuous membership ambiguity U(x) (Section 5, Eq. 7).
2. Prediction disagreement PD(x) (Section 5, Eq. 5).
3. Within-class explanation instability EI_within(x) and transition instability EI_transition(x) (Section 5, Eq. 6).
4. Central Hypothesis H2 evaluation: nested cross-validated comparison of
   PD + U vs PD + U + EI_within with AUROC, AUPRC, bootstrap confidence intervals,
   and paired statistical tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .cfmnn_core import CFMNN, feature_terms
from .explain import decomposition


def compute_u(model: CFMNN, X: np.ndarray, eta: float = 0.5) -> np.ndarray:
    """Continuous membership ambiguity score U(x) (Protocol v6.0 Eq. 7).

    U(x) = eta * [1 - (r_{(1)}(x) - r_{(2)}(x))] + (1 - eta) * [1 - r_{(1)}(x)]

    where r_{(1)}(x) and r_{(2)}(x) are the largest and second-largest
    class membership scores in [0, 1].

    Parameters
    ----------
    model : CFMNN
        Fitted classifier.
    X : np.ndarray, shape (n, d)
        Input features in [0, 1].
    eta : float
        Ambiguity balance parameter in [0, 1] (default 0.5).

    Returns
    -------
    U : np.ndarray, shape (n,)
        Continuous ambiguity score in [0, 1].
    """
    X = model._check_X(X)
    n = len(X)
    det = model.predict_details(X)
    # Support values s_c(x) lie in [0.5, 1.0] for non-empty classes
    support = det.support  # shape (n, C)
    C = support.shape[1]

    if C == 1:
        # Trivial single-class case
        r1 = support[:, 0]
        return (1.0 - eta) * (1.0 - r1)

    # Sort class scores along axis 1 descending
    # Replace nan with -inf if any class has no boxes
    clean_sup = np.nan_to_num(support, nan=-np.inf)
    sorted_scores = np.sort(clean_sup, axis=1)[:, ::-1]
    r1 = sorted_scores[:, 0]
    r2 = sorted_scores[:, 1]
    r2 = np.where(np.isfinite(r2), r2, 0.0)

    # Margin g(x) = r_{(1)}(x) - r_{(2)}(x)
    margin = r1 - r2
    u_score = eta * (1.0 - margin) + (1.0 - eta) * (1.0 - r1)
    return np.clip(u_score, 0.0, 1.0)


def compute_pd(model: CFMNN, X: np.ndarray, K: int = 50, radius: float = 0.05,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Prediction Disagreement PD(x) under local bounded perturbations (Eq. 5).

    PD(x) = (1 / K) * sum_{k=1}^K I[f(x^(k)) != f(x)]

    Perturbations are uniformly sampled in Delta = {delta: ||delta||_inf <= radius}
    and clipped to [0, 1].
    """
    X = model._check_X(X)
    n, d = X.shape
    if rng is None:
        rng = np.random.default_rng(42)

    pred_orig = model.predict(X)
    disagreements = np.zeros(n, dtype=float)

    for _ in range(K):
        # Sample delta in [-radius, radius]^d
        delta = rng.uniform(-radius, radius, size=(n, d))
        X_pert = np.clip(X + delta, 0.0, 1.0)
        pred_pert = model.predict(X_pert)
        disagreements += (pred_pert != pred_orig)

    return disagreements / float(K)


@dataclass
class DiagnosticSignals:
    pd: np.ndarray             # Prediction disagreement (n,)
    u: np.ndarray              # Membership ambiguity (n,)
    ei_within: np.ndarray      # Within-class explanation instability (n,), NaN if undefined
    ei_transition: np.ndarray  # Transition instability (n,)
    eligible_count: np.ndarray # Count of eligible perturbations (n,)
    is_defined: np.ndarray     # Boolean mask where ei_within is valid (n,)

    @property
    def operational_risk(self) -> np.ndarray:
        """The validated operational error risk score: 0.5*PD + 0.5*U."""
        return np.clip(0.5 * self.pd + 0.5 * self.u, 0.0, 1.0)


def compute_error_risk_score(
    model: CFMNN,
    X: np.ndarray,
    alpha: float = 0.5,
    eta: float = 0.5,
    K: int = 50,
    radius: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Operational error risk score combining Prediction Disagreement and Ambiguity (PD + U).

    Falsification of Hypothesis H2 proved that EI_within does not improve error detection.
    Therefore, the primary operational error risk score R(x) is:
        R(x) = alpha * PD(x) + (1 - alpha) * U(x)

    Parameters
    ----------
    model : CFMNN
        Fitted model.
    X : np.ndarray
        Input features in [0, 1]^d.
    alpha : float, default 0.5
        Weight between Prediction Disagreement and Ambiguity.
    eta : float, default 0.5
        Ambiguity balance parameter.
    K : int, default 50
        Number of local perturbations.
    radius : float, default 0.05
        L_inf perturbation radius.
    rng : np.random.Generator, optional

    Returns
    -------
    risk_score : np.ndarray
        Continuous error risk score in [0, 1].
    """
    pd = compute_pd(model, X, K=K, radius=radius, rng=rng)
    u = compute_u(model, X, eta=eta)
    return np.clip(alpha * pd + (1.0 - alpha) * u, 0.0, 1.0)


def compute_diagnostics(model: CFMNN, X: np.ndarray, background: np.ndarray,
                        K: int = 50, radius: float = 0.05,
                        eta: float = 0.5, min_eligible: int = 3,
                        rng: np.random.Generator | None = None) -> DiagnosticSignals:
    """Compute PD, U, EI_within, and EI_transition simultaneously and efficiently.

    Parameters
    ----------
    model : CFMNN
        Fitted model.
    X : np.ndarray, shape (n, d)
        Input samples.
    background : np.ndarray
        Background dataset for explanation baseline.
    K : int
        Number of perturbations per sample.
    radius : float
        Perturbation radius epsilon for L_inf bound.
    eta : float
        Ambiguity balance parameter.
    min_eligible : int
        Minimum within-class perturbations required for EI_within.
    rng : np.random.Generator
        RNG instance.
    """
    X = model._check_X(X)
    n, d = X.shape
    if rng is None:
        rng = np.random.default_rng(42)

    # 1. Base prediction details & ambiguity
    det_orig = model.predict_details(X)
    orig_class = det_orig.cls
    orig_winner = det_orig.winner
    u = compute_u(model, X, eta=eta)

    # 2. Base attributions (Target A additive decomposition)
    dec_orig = decomposition(model, X, background)
    a_orig = dec_orig.phi  # shape (n, d)
    # Normalize attribution to unit L1 norm for scale invariance
    norm_orig = np.abs(a_orig).sum(axis=1, keepdims=True)
    a_orig_norm = np.divide(a_orig, norm_orig, out=np.zeros_like(a_orig), where=norm_orig > 0)

    # 3. Perturbation loop
    disagree_count = np.zeros(n, dtype=float)
    eligible_count = np.zeros(n, dtype=int)
    ei_dist_sum = np.zeros(n, dtype=float)

    for _ in range(K):
        delta = rng.uniform(-radius, radius, size=(n, d))
        X_pert = np.clip(X + delta, 0.0, 1.0)
        det_pert = model.predict_details(X_pert)
        pert_class = det_pert.cls
        pert_winner = det_pert.winner

        # Disagreement
        disagree = (pert_class != orig_class)
        disagree_count += disagree

        # Within-class and within-box condition: same class and same winning box
        eligible = (~disagree) & (pert_winner == orig_winner)
        eligible_count += eligible

        if np.any(eligible):
            idx_el = np.flatnonzero(eligible)
            dec_pert = decomposition(model, X_pert[idx_el], background)
            a_pert = dec_pert.phi
            norm_pert = np.abs(a_pert).sum(axis=1, keepdims=True)
            a_pert_norm = np.divide(a_pert, norm_pert, out=np.zeros_like(a_pert), where=norm_pert > 0)

            # Normalized L1 distance D(a, b) = 0.5 * sum_i |a_i - b_i| in [0, 1]
            dist = 0.5 * np.abs(a_orig_norm[idx_el] - a_pert_norm).sum(axis=1)
            ei_dist_sum[idx_el] += dist

    pd = disagree_count / float(K)
    ei_trans = 1.0 - (eligible_count / float(K))

    is_defined = eligible_count >= min_eligible
    ei_within = np.full(n, np.nan)
    ei_within[is_defined] = ei_dist_sum[is_defined] / eligible_count[is_defined]

    return DiagnosticSignals(
        pd=pd,
        u=u,
        ei_within=ei_within,
        ei_transition=ei_trans,
        eligible_count=eligible_count,
        is_defined=is_defined,
    )


@dataclass
class HypothesisTestResult:
    auroc_base: float
    auroc_ext: float
    delta_auroc: float
    auroc_base_ci: tuple[float, float]
    auroc_ext_ci: tuple[float, float]
    delta_auroc_ci: tuple[float, float]
    p_value_auroc_bootstrap: float
    auprc_base: float
    auprc_ext: float
    delta_auprc: float
    auprc_base_ci: tuple[float, float]
    auprc_ext_ci: tuple[float, float]
    delta_auprc_ci: tuple[float, float]
    p_value_auprc_bootstrap: float
    n_test: int
    n_errors: int
    n_defined_ei: int


def test_central_hypothesis(
    sig_train: DiagnosticSignals,
    y_train_err: np.ndarray,
    sig_test: DiagnosticSignals,
    y_test_err: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> HypothesisTestResult:
    """Test Hypothesis H2: Does PD + U + EI_within improve held-out error detection over PD + U?

    Trains risk models on training fold signals and evaluates on held-out test fold signals.

    Parameters
    ----------
    sig_train : DiagnosticSignals
        Diagnostics on training/validation set.
    y_train_err : np.ndarray
        Binary indicator: 1 if model predicted incorrectly on train sample, else 0.
    sig_test : DiagnosticSignals
        Diagnostics on test set.
    y_test_err : np.ndarray
        Binary indicator: 1 if model predicted incorrectly on test sample, else 0.
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals and paired tests.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)

    # Impute missing EI_within with median from training fold, plus a missingness indicator
    train_defined = sig_train.is_defined
    median_ei = np.nanmedian(sig_train.ei_within) if np.any(train_defined) else 0.0

    X_train_base = np.column_stack([sig_train.pd, sig_train.u])
    X_test_base = np.column_stack([sig_test.pd, sig_test.u])

    # Extended features: PD, U, imputed EI_within, missing_indicator
    train_ei_imp = np.where(train_defined, sig_train.ei_within, median_ei)
    train_ei_miss = (~train_defined).astype(float)
    X_train_ext = np.column_stack([sig_train.pd, sig_train.u, train_ei_imp, train_ei_miss])

    test_defined = sig_test.is_defined
    test_ei_imp = np.where(test_defined, sig_test.ei_within, median_ei)
    test_ei_miss = (~test_defined).astype(float)
    X_test_ext = np.column_stack([sig_test.pd, sig_test.u, test_ei_imp, test_ei_miss])

    # Fit LogisticRegression models
    # If error rate is degenerate (all 0 or all 1), fallback gracefully
    if len(np.unique(y_train_err)) < 2:
        # Fallback to simple linear composite
        scores_base = sig_test.pd + sig_test.u
        scores_ext = sig_test.pd + sig_test.u + test_ei_imp
    else:
        clf_base = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        clf_base.fit(X_train_base, y_train_err)
        scores_base = clf_base.predict_proba(X_test_base)[:, 1]

        clf_ext = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        clf_ext.fit(X_train_ext, y_train_err)
        scores_ext = clf_ext.predict_proba(X_test_ext)[:, 1]


    n_test = len(y_test_err)
    n_errors = int(np.sum(y_test_err))

    # Evaluate point metrics
    if n_errors == 0 or n_errors == n_test:
        return HypothesisTestResult(
            auroc_base=np.nan, auroc_ext=np.nan, delta_auroc=np.nan,
            auroc_base_ci=(np.nan, np.nan), auroc_ext_ci=(np.nan, np.nan), delta_auroc_ci=(np.nan, np.nan),
            p_value_auroc_bootstrap=np.nan,
            auprc_base=np.nan, auprc_ext=np.nan, delta_auprc=np.nan,
            auprc_base_ci=(np.nan, np.nan), auprc_ext_ci=(np.nan, np.nan), delta_auprc_ci=(np.nan, np.nan),
            p_value_auprc_bootstrap=np.nan,
            n_test=n_test, n_errors=n_errors, n_defined_ei=int(test_defined.sum()),
        )

    auroc_base = float(roc_auc_score(y_test_err, scores_base))
    auroc_ext = float(roc_auc_score(y_test_err, scores_ext))
    delta_auroc = auroc_ext - auroc_base

    auprc_base = float(average_precision_score(y_test_err, scores_base))
    auprc_ext = float(average_precision_score(y_test_err, scores_ext))
    delta_auprc = auprc_ext - auprc_base

    # Paired Bootstrap for 95% Confidence Intervals & Hypotheses
    bs_auroc_base = []
    bs_auroc_ext = []
    bs_delta_auroc = []
    bs_auprc_base = []
    bs_auprc_ext = []
    bs_delta_auprc = []

    for _ in range(n_bootstrap):
        b_idx = rng.choice(n_test, size=n_test, replace=True)
        yb = y_test_err[b_idx]
        if len(np.unique(yb)) < 2:
            continue
        sb_b = scores_base[b_idx]
        se_b = scores_ext[b_idx]

        auc_b = roc_auc_score(yb, sb_b)
        auc_e = roc_auc_score(yb, se_b)
        d_auc = auc_e - auc_b

        prc_b = average_precision_score(yb, sb_b)
        prc_e = average_precision_score(yb, se_b)
        d_prc = prc_e - prc_b

        bs_auroc_base.append(auc_b)
        bs_auroc_ext.append(auc_e)
        bs_delta_auroc.append(d_auc)
        bs_auprc_base.append(prc_b)
        bs_auprc_ext.append(prc_e)
        bs_delta_auprc.append(d_prc)

    bs_delta_auroc = np.array(bs_delta_auroc)
    bs_delta_auprc = np.array(bs_delta_auprc)

    ci_auc_base = (float(np.percentile(bs_auroc_base, 2.5)), float(np.percentile(bs_auroc_base, 97.5)))
    ci_auc_ext = (float(np.percentile(bs_auroc_ext, 2.5)), float(np.percentile(bs_auroc_ext, 97.5)))
    ci_delta_auc = (float(np.percentile(bs_delta_auroc, 2.5)), float(np.percentile(bs_delta_auroc, 97.5)))
    p_val_auc = float(np.mean(bs_delta_auroc <= 0.0))

    ci_prc_base = (float(np.percentile(bs_auprc_base, 2.5)), float(np.percentile(bs_auprc_base, 97.5)))
    ci_prc_ext = (float(np.percentile(bs_auprc_ext, 2.5)), float(np.percentile(bs_auprc_ext, 97.5)))
    ci_delta_prc = (float(np.percentile(bs_delta_auprc, 2.5)), float(np.percentile(bs_delta_auprc, 97.5)))
    p_val_prc = float(np.mean(bs_delta_auprc <= 0.0))

    return HypothesisTestResult(
        auroc_base=auroc_base,
        auroc_ext=auroc_ext,
        delta_auroc=delta_auroc,
        auroc_base_ci=ci_auc_base,
        auroc_ext_ci=ci_auc_ext,
        delta_auroc_ci=ci_delta_auc,
        p_value_auroc_bootstrap=p_val_auc,
        auprc_base=auprc_base,
        auprc_ext=auprc_ext,
        delta_auprc=delta_auprc,
        auprc_base_ci=ci_prc_base,
        auprc_ext_ci=ci_prc_ext,
        delta_auprc_ci=ci_delta_prc,
        p_value_auprc_bootstrap=p_val_prc,
        n_test=n_test,
        n_errors=n_errors,
        n_defined_ei=int(test_defined.sum()),
    )
