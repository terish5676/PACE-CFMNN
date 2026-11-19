"""PACE-CFMNN: exact explanations and explanation-guided refinement for CFMNN.

Modules so far:

* ``cfmnn_core``: the reference CFMNN classifier.
* ``data``: datasets, splits and scaling.
* ``evaluate``: classification and error-detection metrics.
* ``explain``: PACE-flip (``PaceFlip``) and its conditional version, PACE-SHAP
  (``PaceShap``: interventional or conditional game, interaction values),
  PACE-multi, contrastive (also for a given class pair) and decomposition
  explanations.
* ``counterfactual``: box-directed counterfactuals (PACE) and the sparsest
  ones (PACE-sparse), verified on the model.
* ``global_importance``: data-based and model-only feature importance, and
  normalised averaging of local scores.
* ``xai_baselines``: KernelSHAP (shap's, and one on a chosen game), LIME, DiCE,
  nearest unlike neighbour and brute-force occlusion, for comparison.
* ``realism``: how often an explainer's replaced inputs lie off the data.
* ``neighborhood``: feature-aware perturbations for the trust signals.
* ``signals``: trust signals PD, EI and U (plus the corrected U*).
* ``pace_score``: the PACE Refinement Score (PRS) from percentile ranks.
* ``trust_baselines``: Trust Score, order-ensemble and kNN disagreement.
* ``refine``: box edits for refinement (explanation-guided and control
  shrinks) and the acceptance gate.
* ``active``: the budgeted active-learning loop (label oracle, selectors,
  rollback).
* ``api``: a small front end, ``PaceClassifier`` and the functions ``explain``,
  ``repair`` and ``certify``.
"""

from .cfmnn_core import CFMNN, Prediction, feature_terms, ramp

__all__ = ["CFMNN", "PaceClassifier", "Prediction", "feature_terms", "ramp"]
__version__ = "0.1.0"


def __getattr__(name):
    # The front end imports the repair loop and its dependencies, so it loads only when asked for.
    if name == "PaceClassifier":
        from .api import PaceClassifier
        return PaceClassifier
    raise AttributeError(f"module 'pace_cfmnn' has no attribute {name!r}")
