"""A small front end: fit, explain, repair and certify in a few lines (tough review B13).

Features must be scaled to [0, 1] first, as everywhere in this package::

    from pace_cfmnn import PaceClassifier

    model = PaceClassifier(theta=0.3).fit(X_train, y_train)
    phi = model.explain(X_test[:5])                    # exact PACE-SHAP values, up to 12 features
    rank = model.explain(X_test[:5], method="flip")    # PACE-flip scores (higher = more important)
    result = model.repair(X_train, y_train, certify_on=X_test)   # no labels beyond the training rows
    result.model          # the repaired model (the original is unchanged)
    result.certificate    # True where a test prediction may have changed; False: certainly the same

The same steps are plain functions (``explain``, ``repair``, ``certify``) for any fitted ``CFMNN``.
They call the classes the paper's runs use, with the paper's settings, and change nothing in them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active import run_self_repair
from .audit import impact_set
from .cfmnn_core import CFMNN
from .explain import PaceFlip, PaceShap

__all__ = ["PaceClassifier", "Repair", "explain", "repair", "certify"]


def explain(model: CFMNN, X, background, method: str = "shap", seed: int = 0) -> np.ndarray:
    """Explain each row of X: an (n, d) array.

    * ``method="shap"``: PACE-SHAP, the Shapley values of the margin between the predicted class
      and the best other class, with ``background`` filling in removed features. Exact for up to
      12 features; above that it samples with a fixed seed. The values add up to the margin at x
      minus its mean over the background rows.
    * ``method="flip"``: PACE-flip, how often replacing the feature by a background value (its 32
      quantiles) changes the predicted class, with the drop in membership breaking ties.
    """
    if method == "shap":
        return PaceShap(model, background, seed=seed).explain(X)
    if method == "flip":
        return PaceFlip(model, background).explain(X)
    raise ValueError("method must be 'shap' or 'flip'")


def certify(before: CFMNN, after: CFMNN, X, removed: int | None = None,
            tight: bool = True) -> np.ndarray:
    """Rows of X whose prediction ONE edit ``before -> after`` can have changed (Proposition 4).

    ``removed`` is the index in ``before`` of a box the edit deleted, if any. False means the
    prediction is certainly the same; True means it may differ. The certificate is sound (every
    changed prediction is marked) but loose (many marked rows do not change). ``tight=False``
    gives the plain version, which also marks rows whose tied boxes are all of one class. For a
    run of edits, use ``repair(..., certify_on=X)``, which takes the union over its edits.
    """
    return impact_set(before, after, X, removed=removed, tight=tight)


@dataclass
class Repair:
    """What ``repair`` returns."""
    model: CFMNN               # the repaired model; the input model is not changed
    edits: list                # the accepted edits, in order
    certificate: np.ndarray | None  # over ``certify_on``: union of the edits' certificates


def repair(model: CFMNN, X_train, y_train, certify_on=None, budget: int = 300,
           n_rounds: int = 10, seed: int = 0, tight: bool = True) -> Repair:
    """Self-repair (Section 5): shrink a wrong-class box at the model's own training errors.

    Each edit is kept only if macro-F1 on the training rows drops by at most 0.5 points and no
    class recall by more than 1 point (the gate). With ``certify_on``, the result also marks the
    rows of ``certify_on`` whose prediction the repair can have changed: the union of every
    accepted edit's certificate (Corollary 1, Appendix F). Rows left unmarked keep their
    prediction for certain.
    """
    cert = None
    on_accept = None
    if certify_on is not None:
        Xc = model._check_X(certify_on)
        cert = np.zeros(len(Xc), dtype=bool)

        def on_accept(before, after, removed):
            cert[:] |= impact_set(before, after, Xc, removed=removed, tight=tight)

    fixed, _, edits = run_self_repair(model, X_train, y_train, X_train, y_train, chooser="volume",
                                      budget=budget, n_rounds=n_rounds, seed=seed,
                                      gate_labels_are_new=False, on_accept=on_accept)
    return Repair(fixed, edits, cert)


class PaceClassifier(CFMNN):
    """A CFMNN that also explains, repairs and certifies itself.

    ``fit`` keeps up to ``n_background`` training rows (chosen with ``background_seed``) as the
    default background for ``explain``. Everything else is CFMNN's, so it works with
    scikit-learn's tools (``clone``, ``cross_val_score``, pipelines).
    """

    n_background = 100
    background_seed = 0

    def fit(self, X, y):
        super().fit(X, y)
        Xa = self._check_X(X)
        rng = np.random.default_rng(self.background_seed)
        k = min(len(Xa), self.n_background)
        self.background_ = Xa[np.sort(rng.choice(len(Xa), size=k, replace=False))]
        return self

    def explain(self, X, method: str = "shap", background=None, seed: int = 0) -> np.ndarray:
        bg = self.background_ if background is None else background
        return explain(self, X, bg, method=method, seed=seed)

    def repair(self, X_train, y_train, certify_on=None, budget: int = 300, n_rounds: int = 10,
               seed: int = 0, tight: bool = True) -> Repair:
        return repair(self, X_train, y_train, certify_on=certify_on, budget=budget,
                      n_rounds=n_rounds, seed=seed, tight=tight)
