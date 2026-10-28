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

from .cfmnn_core import CFMNN, Prediction, feature_terms, ramp

__all__ = ["CFMNN", "Prediction", "feature_terms", "ramp"]
__version__ = "0.1.0"
