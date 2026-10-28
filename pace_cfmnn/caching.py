"""Exact Structural Caching, Approximate Caching, and Mathematical Verification.

Implements:
1. Exact Lipschitz bounds for CFMNN membership functions (Appendix B).
2. Exact structural cache sufficient-condition verifier:
   m_{j*}(x) - max_{j != j*} m_j(x) > gamma * rho
3. Structural cache vs approximate cache vs full recomputation.
4. Mathematical verification suite for continuity (Appendix A) and caching (Appendix B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cfmnn_core import CFMNN, feature_terms
from .explain import Decomposition, decomposition


@dataclass
class CacheEvaluationResult:
    mode: str
    structural_mismatch_rate: float
    max_attribution_error: float
    mean_attribution_error: float
    q95_attribution_error: float
    runtime_s: float
    speedup_vs_recompute: float
    exactness_proof_status: str


def compute_exact_lipschitz_constant(gamma: float) -> float:
    """Exact Lipschitz constant L_j for CFMNN box membership under L_inf norm.

    Theorem (Appendix B & Protocol Section 7):
    Let f(t, gamma) = max(0, 1 - max(0, gamma * min(1, t))).
    f is piecewise linear with slope in {-gamma, 0}.
    m_{j, i}(x_i) = 0.5 * [f(x_i - w_{ji}, gamma) + f(v_{ji} - x_i, gamma)].
    Since v_{ji} <= w_{ji}, at most one of (x_i - w_{ji}) and (v_{ji} - x_i) can be > 0.
    Hence |m_{j, i}(x'_i) - m_{j, i}(x_i)| <= (gamma / 2) * |x'_i - x_i|.

    For box membership m_j(x) = (1 / d) * sum_{i=1}^d m_{j, i}(x_i):
    |m_j(x + delta) - m_j(x)| <= (1 / d) * sum_{i=1}^d (gamma / 2) * |delta_i|
                              <= (gamma / 2) * ||delta||_inf.

    Therefore, L_j = gamma / 2 for all hyperboxes j.
    """
    return 0.5 * float(gamma)


def verify_winning_box_invariance_condition(
    model: CFMNN, X: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check Appendix B exact-cache sufficient condition.

    Condition:
        m_{j*}(x) - m_j(x) > (L_{j*} + L_j) * rho = gamma * rho  for all j != j*

    Returns
    -------
    is_provably_invariant : np.ndarray of bool, shape (n,)
        True for samples where the winning box is mathematically guaranteed
        to remain invariant for ALL perturbations delta with ||delta||_inf <= rho.
    winner : np.ndarray of int, shape (n,)
        The winning box j* at x.
    margin : np.ndarray of float, shape (n,)
        The margin m_{j*}(x) - max_{j != j*} m_j(x).
    """
    X = model._check_X(X)
    n = len(X)
    M = model.memberships(X)  # shape (n, H)
    H = M.shape[1]

    if H <= 1:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=int), np.full(n, 1.0)

    # Top two memberships
    sorted_idx = np.argsort(-M, axis=1, kind="stable")
    j_star = sorted_idx[:, 0]
    j_second = sorted_idx[:, 1]

    m_top = M[np.arange(n), j_star]
    m_second = M[np.arange(n), j_second]
    margin = m_top - m_second

    threshold = model.gamma * rho
    is_provably_invariant = margin > threshold

    return is_provably_invariant, j_star, margin


def get_active_competitor_boxes(model: CFMNN, x: np.ndarray, rho: float) -> tuple[np.ndarray, int]:
    """Candidate Bounding Theorem (Appendix B Corollary).

    Any hyperbox j satisfying m_j(x) < m_{j*}(x) - gamma * rho is mathematically
    guaranteed to never win for any perturbation in B_inf(x, rho).
    """
    m_x = model.memberships(x[None, :])[0]
    j_star = int(np.argmax(m_x))
    cutoff = m_x[j_star] - model.gamma * rho
    candidates = np.flatnonzero(m_x >= cutoff)
    return candidates, j_star


class ExactStructuralCache:
    """Exact structural cache for local perturbation explanations.

    For inputs x, verifies Appendix B sufficient condition:
    if margin > gamma * rho, the winning box j* is provably invariant across
    the entire perturbation ball B_inf(x, rho).
    Under invariance, computing explanations for perturbations only evaluates
    terms for j*, bypassing re-ranking all H hyperboxes.
    If the condition fails, it applies candidate bounding to evaluate only
    the active competitors C(x), maintaining 0.00% error while speeding up execution.
    """

    def __init__(self, model: CFMNN, background: np.ndarray, rho: float):
        self.model = model
        self.background = background
        self.rho = rho
        self.L = compute_exact_lipschitz_constant(model.gamma)
        # Precompute expected terms for all boxes
        self.d = background.shape[1]
        self.box_means = np.array([
            feature_terms(background, model.V_[j], model.W_[j], model.gamma).mean(axis=0)
            for j in range(model.n_boxes_)
        ])  # shape (H, d)

    def explain_perturbed_batch(self, x: np.ndarray, perturbations: np.ndarray) -> tuple[np.ndarray, bool]:
        """Explain perturbations of a single point x.

        Returns (attributions, was_exact_cached).
        """
        K, d = perturbations.shape
        cand, j_star = get_active_competitor_boxes(self.model, x, self.rho)

        if len(cand) == 1:
            # Exact cache hit: winning box is guaranteed to be j_star
            terms = feature_terms(perturbations, self.model.V_[j_star], self.model.W_[j_star], self.model.gamma)
            phi = (terms - self.box_means[j_star]) / self.d
            return phi, True
        else:
            # Near boundary: evaluate ONLY the active candidate competitors C(x)
            cand_V = self.model.V_[cand]
            cand_W = self.model.W_[cand]
            # (K, len(cand), d)
            terms = feature_terms(perturbations[:, None, :], cand_V, cand_W, self.model.gamma)
            cand_memberships = terms.mean(axis=2)  # shape (K, len(cand))
            best_cand_idx = np.argmax(cand_memberships, axis=1)
            actual_winners = cand[best_cand_idx]

            best_terms = terms[np.arange(K), best_cand_idx, :]
            phi = (best_terms - self.box_means[actual_winners]) / self.d
            return phi, False


class ApproximateCache:
    """Approximate cache: reuses base winning box j* without verifying condition."""

    def __init__(self, model: CFMNN, background: np.ndarray):
        self.model = model
        self.background = background
        self.d = background.shape[1]
        self.box_means = np.array([
            feature_terms(background, model.V_[j], model.W_[j], model.gamma).mean(axis=0)
            for j in range(model.n_boxes_)
        ])

    def explain_perturbed_batch(self, x: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        det = self.model.predict_details(x[None, :])
        j_star = det.winner[0]
        terms = feature_terms(perturbations, self.model.V_[j_star], self.model.W_[j_star], self.model.gamma)
        return (terms - self.box_means[j_star]) / self.d


def benchmark_caching_modes(
    model: CFMNN, X_val: np.ndarray, background: np.ndarray,
    rho: float = 0.03, K: int = 30, seed: int = 42
) -> list[CacheEvaluationResult]:
    """Benchmark full recompute vs exact structural cache vs approximate cache."""
    rng = np.random.default_rng(seed)
    n, d = X_val.shape

    # Generate test workload: K perturbations for each validation point
    all_perts = []
    for i in range(n):
        delta = rng.uniform(-rho, rho, size=(K, d))
        perts = np.clip(X_val[i] + delta, 0.0, 1.0)
        all_perts.append(perts)

    # 1. Full Recomputation (Ground Truth)
    t0 = time.perf_counter()
    full_phis = []
    full_winners = []
    for i in range(n):
        dec = decomposition(model, all_perts[i], background)
        full_phis.append(dec.phi)
        full_winners.append(dec.box)
    t_full = time.perf_counter() - t0

    res_full = CacheEvaluationResult(
        mode="Full Recompute (No Cache)",
        structural_mismatch_rate=0.0,
        max_attribution_error=0.0,
        mean_attribution_error=0.0,
        q95_attribution_error=0.0,
        runtime_s=t_full,
        speedup_vs_recompute=1.0,
        exactness_proof_status="Exact (Direct Evaluation)",
    )

    # 2. Exact Structural Cache
    exact_cache = ExactStructuralCache(model, background, rho)
    t0 = time.perf_counter()
    exact_phis = []
    exact_mismatches = 0
    total_checks = 0

    for i in range(n):
        phi_batch, was_cached = exact_cache.explain_perturbed_batch(X_val[i], all_perts[i])
        exact_phis.append(phi_batch)
        if was_cached:
            # Verify that winning box really was invariant
            det_perts = model.predict_details(all_perts[i])
            mismatch = (det_perts.winner != det_perts.winner[0]).sum()
            exact_mismatches += mismatch
            total_checks += K
    t_exact = time.perf_counter() - t0

    exact_errors = np.concatenate([
        np.abs(e - f).ravel() for e, f in zip(exact_phis, full_phis)
    ])

    res_exact = CacheEvaluationResult(
        mode="Exact Structural Cache",
        structural_mismatch_rate=float(exact_mismatches / max(1, total_checks)),
        max_attribution_error=float(np.max(exact_errors)),
        mean_attribution_error=float(np.mean(exact_errors)),
        q95_attribution_error=float(np.percentile(exact_errors, 95)),
        runtime_s=t_exact,
        speedup_vs_recompute=float(t_full / max(1e-6, t_exact)),
        exactness_proof_status="Mathematically Proven (Appendix B)",
    )

    # 3. Approximate Cache
    approx_cache = ApproximateCache(model, background)
    t0 = time.perf_counter()
    approx_phis = []
    approx_mismatches = 0

    for i in range(n):
        phi_batch = approx_cache.explain_perturbed_batch(X_val[i], all_perts[i])
        approx_phis.append(phi_batch)
        # Check mismatches against actual full winners
        det_perts = model.predict_details(all_perts[i])
        det_orig = model.predict_details(X_val[i:i+1])
        mismatches = (det_perts.winner != det_orig.winner[0]).sum()
        approx_mismatches += mismatches
    t_approx = time.perf_counter() - t0

    approx_errors = np.concatenate([
        np.abs(a - f).ravel() for a, f in zip(approx_phis, full_phis)
    ])

    res_approx = CacheEvaluationResult(
        mode="Approximate Cache",
        structural_mismatch_rate=float(approx_mismatches / (n * K)),
        max_attribution_error=float(np.max(approx_errors)),
        mean_attribution_error=float(np.mean(approx_errors)),
        q95_attribution_error=float(np.percentile(approx_errors, 95)),
        runtime_s=t_approx,
        speedup_vs_recompute=float(t_full / max(1e-6, t_approx)),
        exactness_proof_status="Unproven / Approximate",
    )

    return [res_full, res_exact, res_approx]
