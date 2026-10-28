"""PACE-CFMNN: exact explanations and explanation-guided refinement for CFMNN.

Modules so far:

* ``cfmnn_core``: the reference CFMNN classifier.
* ``data``: datasets, splits and scaling.
* ``evaluate``: classification metrics.
* ``explain``: PACE-flip (``PaceFlip``), PACE-multi, contrastive and
  decomposition explanations.
* ``counterfactual``: box-directed counterfactuals, verified on the model.
* ``global_importance``: data-based and model-only feature importance.
* ``xai_baselines``: KernelSHAP, LIME, DiCE, nearest unlike neighbour and
  brute-force occlusion, for comparison.

Planned (later phases): trust signals (``neighborhood``, ``signals``,
``pace_score``) and explanation-guided refinement (``active``, ``refine``,
``audit``).
"""

from .cfmnn_core import CFMNN, EnsembleCFMNN, Prediction, feature_terms, ramp
from .auto import AutoCFMNN
from .contraction import apply_fmm_contraction, safe_apply_fmm_contraction
from .trust_diagnostics import (
    DiagnosticSignals,
    HypothesisTestResult,
    compute_diagnostics,
    compute_error_risk_score,
    compute_pd,
    compute_u,
    test_central_hypothesis,
)
from .caching import (
    ApproximateCache,
    CacheEvaluationResult,
    ExactStructuralCache,
    benchmark_caching_modes,
    compute_exact_lipschitz_constant,
    verify_winning_box_invariance_condition,
)
from .refinement import (
    AcceptanceGateResult,
    NestedRefinementEvaluation,
    apply_explanation_guided_shrink,
    apply_random_shrink_control,
    check_acceptance_gate,
    run_nested_cross_fitted_refinement,
)
from .data import (
    clean_label_conflicts,
    detect_label_conflicts,
)
from .neighborhood import detect_binary_columns, generate_perturbations
from .signals import (
    compute_explanation_instability,
    compute_prediction_disagreement,
    compute_uncertainty,
)
from .pace_score import PrsResult, compute_prs, compute_risk
from .refine import RefinementConfig, RefinementResult, refine_sample
from .active import (
    ActiveLearningResult,
    LabelOracle,
    compute_hyperbox_perimeter_sum,
    compute_hyperbox_volume_sum,
    run_active_refinement,
)
from .oblique import ObliqueCFMNN, RotatedHyperbox
from .embedding import EmbeddedCFMNN

__all__ = [
    "CFMNN",
    "AutoCFMNN",
    "EnsembleCFMNN",
    "ObliqueCFMNN",
    "RotatedHyperbox",
    "EmbeddedCFMNN",
    "Prediction",
    "feature_terms",
    "ramp",
    "apply_fmm_contraction",
    "safe_apply_fmm_contraction",
    "compute_u",
    "compute_pd",
    "compute_error_risk_score",
    "compute_risk",
    "compute_diagnostics",
    "test_central_hypothesis",
    "ExactStructuralCache",
    "ApproximateCache",
    "benchmark_caching_modes",
    "apply_explanation_guided_shrink",
    "apply_random_shrink_control",
    "check_acceptance_gate",
    "run_nested_cross_fitted_refinement",
    "detect_label_conflicts",
    "clean_label_conflicts",
    "detect_binary_columns",
    "generate_perturbations",
    "compute_uncertainty",
    "compute_prediction_disagreement",
    "compute_explanation_instability",
    "compute_prs",
    "PrsResult",
    "RefinementConfig",
    "RefinementResult",
    "refine_sample",
    "ActiveLearningResult",
    "LabelOracle",
    "compute_hyperbox_perimeter_sum",
    "compute_hyperbox_volume_sum",
    "run_active_refinement",
]
__version__ = "0.3.0"



