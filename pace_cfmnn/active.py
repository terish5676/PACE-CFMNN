"""Active learning and selective refinement loop for CFMNN (Phase 4 / Phase 5).

Strictly isolates unlabelled pool features from labels, preventing test-set leakage.
Supports acquisition strategies:
- random
- uncertainty (u)
- prediction disagreement (pd)
- explanation instability (ei)
- pd_u
- prs (PACE Refinement Score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd

from .cfmnn_core import CFMNN
from .evaluate import classification_metrics
from .pace_score import compute_prs
from .refine import RefinementConfig, RefinementResult, refine_sample
from .signals import compute_uncertainty, compute_prediction_disagreement, compute_explanation_instability
from .neighborhood import generate_perturbations


from sklearn.cluster import MiniBatchKMeans

ACQUISITION_STRATEGIES = ("random", "uncertainty", "pd", "ei", "pd_u", "prs", "cluster_margin")


class LabelOracle:
    """Simulation oracle protecting pool ground-truth labels from acquisition logic."""

    def __init__(self, y_pool: np.ndarray):
        self._y = np.asarray(y_pool)
        self.queries = 0

    def reveal(self, idx: int) -> object:
        """Reveal ground truth label only upon selection."""
        self.queries += 1
        return self._y[idx]

    def __len__(self) -> int:
        return len(self._y)


def compute_hyperbox_volume_sum(model: CFMNN) -> float:
    """Measure total hyperbox volume sum: sum_j prod_i (W_{j,i} - V_{j,i})."""
    if model.n_boxes_ == 0:
        return 0.0
    widths = np.clip(model.W_ - model.V_, 0.0, 1.0)
    # Use product for volume, or sum if widths are 0
    return float(np.prod(widths, axis=1).sum())


def compute_hyperbox_perimeter_sum(model: CFMNN) -> float:
    """Measure total hyperbox perimeter/diameter sum: sum_j sum_i (W_{j,i} - V_{j,i})."""
    if model.n_boxes_ == 0:
        return 0.0
    widths = np.clip(model.W_ - model.V_, 0.0, 1.0)
    return float(widths.sum())


@dataclass
class ActiveLearningResult:
    """Results and trajectory from an active refinement run."""

    strategy: str
    budget: int
    labels_queried: int
    history: pd.DataFrame
    initial_metrics: dict
    final_metrics: dict
    initial_boxes: int
    final_boxes: int
    accepted_edits: int
    rejected_edits: int
    total_time_s: float


def run_active_refinement(
    model: CFMNN,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_accept: np.ndarray,
    y_accept: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str = "prs",
    budget: int = 50,
    batch_size: int = 1,
    refinement_config: RefinementConfig | None = None,
    pool_prefilter_size: int = 500,
    batch_sample_size: int | None = None,
    n_clusters: int | None = None,
    seed: int = 0,
) -> ActiveLearningResult:
    """Execute active refinement simulation under a fixed labeling budget.

    Supports scalable large-batch active learning:
    - Pre-clustering (cluster_margin): Clusters unlabelled pool up front, sampling
      diverse boundary instances per cluster.
    - Mini-batch subsampling (batch_sample_size): Subsamples massive pools prior
      to neighborhood evaluation for fast O(1) query time.
    - Perturbation and model output caching invalidated on model revision.

    Parameters
    ----------
    model : CFMNN
        Initial trained model.
    X_pool : (n_pool, d)
        Unlabelled pool features.
    y_pool : (n_pool,)
        Ground-truth labels for pool, protected by simulation oracle.
    X_accept, y_accept :
        Acceptance gate validation set.
    X_test, y_test :
        Held-out evaluation set (never exposed to selection or refinement).
    strategy : str
        Acquisition strategy in ACQUISITION_STRATEGIES.
    budget : int
        Maximum number of samples to label and refine.
    batch_size : int
        Number of samples selected per acquisition cycle.
    refinement_config : RefinementConfig
    pool_prefilter_size : int, default 500
        Maximum candidate pool size evaluated with neighborhood perturbations.
    batch_sample_size : int, optional
        If set and pool exceeds this size, randomly subsamples candidates per step.
    n_clusters : int, optional
        Number of clusters for pre-clustering (default 8 if cluster_margin used).
    seed : int

    Returns
    -------
    ActiveLearningResult
    """
    if strategy not in ACQUISITION_STRATEGIES:
        raise ValueError(f"strategy must be one of {ACQUISITION_STRATEGIES}, got {strategy!r}")

    if refinement_config is None:
        refinement_config = RefinementConfig()

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    oracle = LabelOracle(y_pool)

    n_pool = len(X_pool)
    remaining_indices = list(range(n_pool))
    budget = min(budget, n_pool)

    # 1. Pre-cluster unlabelled pool if cluster_margin strategy requested
    pool_clusters = None
    if strategy == "cluster_margin" or n_clusters is not None:
        k_clusters = min(n_clusters or 8, n_pool)
        mbk = MiniBatchKMeans(n_clusters=k_clusters, random_state=seed, n_init="auto")
        pool_clusters = mbk.fit_predict(X_pool)
    cluster_pointer = 0

    # 2. Perturbation & model revision cache
    model_revision = 0
    pert_cache: dict[tuple[int, int], np.ndarray] = {}

    # Initial evaluation
    initial_preds = model.predict(X_test)
    initial_metrics = classification_metrics(y_test, initial_preds)
    initial_boxes = model.n_boxes_

    history_rows = []
    accepted_count = 0
    rejected_count = 0

    iteration = 0
    while len(remaining_indices) > 0 and oracle.queries < budget:
        iteration += 1
        curr_rem = np.array(remaining_indices, dtype=np.int64)

        # Mini-batch subsampling for massive pools (e.g. >10k points)
        if batch_sample_size is not None and len(curr_rem) > batch_sample_size:
            sub_sample_indices = rng.choice(len(curr_rem), size=batch_sample_size, replace=False)
            curr_rem = curr_rem[sub_sample_indices]

        X_cand = X_pool[curr_rem]

        # Two-stage hierarchical triage filter for large candidate sets
        if len(curr_rem) > pool_prefilter_size and strategy in ("pd", "ei", "pd_u", "prs"):
            u_all, _ = compute_uncertainty(model, X_cand)
            top_sub = np.argsort(-u_all)[:pool_prefilter_size]
            X_eval = X_cand[top_sub]
            sub_map = top_sub
        else:
            X_eval = X_cand
            sub_map = np.arange(len(curr_rem))

        # Acquisition scoring
        if strategy == "random":
            chosen_rel = int(rng.integers(0, len(curr_rem)))
        elif strategy == "cluster_margin":
            # Round-robin across clusters: select candidate from target cluster with highest uncertainty
            k_clust = len(np.unique(pool_clusters))
            target_cluster = cluster_pointer % k_clust
            cluster_pointer += 1

            in_clust_mask = (pool_clusters[curr_rem] == target_cluster)
            if np.any(in_clust_mask):
                sub_indices = np.where(in_clust_mask)[0]
                u_clust, _ = compute_uncertainty(model, X_cand[sub_indices])
                chosen_rel = int(sub_indices[np.argmax(u_clust)])
            else:
                u_all, _ = compute_uncertainty(model, X_cand)
                chosen_rel = int(np.argmax(u_all))
        elif strategy == "uncertainty":
            u, _ = compute_uncertainty(model, X_cand)
            chosen_rel = int(np.argmax(u))
        elif strategy == "pd":
            pert = generate_perturbations(X_eval, n_perturb=15, sigma=0.05, seed=seed + iteration)
            pd_vals = compute_prediction_disagreement(model, X_eval, pert)
            chosen_rel = int(sub_map[np.argmax(pd_vals)])
        elif strategy == "ei":
            pert = generate_perturbations(X_eval, n_perturb=15, sigma=0.05, seed=seed + iteration)
            ei_vals, _ = compute_explanation_instability(model, X_eval, pert)
            chosen_rel = int(sub_map[np.argmax(ei_vals)])
        elif strategy == "pd_u":
            pert = generate_perturbations(X_eval, n_perturb=15, sigma=0.05, seed=seed + iteration)
            pd_vals = compute_prediction_disagreement(model, X_eval, pert)
            u_vals, _ = compute_uncertainty(model, X_eval)
            score = 0.5 * (pd_vals + u_vals)
            chosen_rel = int(sub_map[np.argmax(score)])
        elif strategy == "prs":
            # Composite PRS with validated weights (0.5*PD + 0.5*U, EI_within=0.0)
            prs_res = compute_prs(model, X_eval, n_perturb=15, sigma=0.05, seed=seed + iteration)
            chosen_rel = int(sub_map[np.flatnonzero(prs_res.rank == 0)[0]])

        pool_idx = curr_rem[chosen_rel]
        remaining_indices.remove(pool_idx)

        # Reveal true label via oracle
        y_true = oracle.reveal(pool_idx)
        x_selected = X_pool[pool_idx]

        # Apply selective refinement
        ref_res: RefinementResult = refine_sample(
            model, x_selected, y_true,
            X_accept=X_accept, y_accept=y_accept,
            config=refinement_config,
        )

        if ref_res.accepted:
            accepted_count += 1
            model_revision += 1  # Invalidate perturbation cache on accepted model edit
        elif ref_res.attempted and ref_res.rolled_back:
            rejected_count += 1

        # Evaluate updated model on held-out test set
        test_preds = model.predict(X_test)
        curr_metrics = classification_metrics(y_test, test_preds)

        history_rows.append({
            "iteration": iteration,
            "queries": oracle.queries,
            "pool_idx": pool_idx,
            "operation": ref_res.operation,
            "accepted": ref_res.accepted,
            "rolled_back": ref_res.rolled_back,
            "box_modified": ref_res.box_modified,
            "feature_modified": ref_res.feature_modified,
            "n_boxes": model.n_boxes_,
            "perimeter_sum": compute_hyperbox_perimeter_sum(model),
            "test_accuracy": curr_metrics["accuracy"],
            "test_macro_f1": curr_metrics["macro_f1"],
            "test_balanced_acc": curr_metrics["balanced_accuracy"],
            "test_min_recall": curr_metrics["min_class_recall"],
            "reason": ref_res.reason,
        })

    total_time = time.perf_counter() - t0
    final_preds = model.predict(X_test)
    final_metrics = classification_metrics(y_test, final_preds)

    return ActiveLearningResult(
        strategy=strategy,
        budget=budget,
        labels_queried=oracle.queries,
        history=pd.DataFrame(history_rows),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        initial_boxes=initial_boxes,
        final_boxes=model.n_boxes_,
        accepted_edits=accepted_count,
        rejected_edits=rejected_count,
        total_time_s=total_time,
    )
