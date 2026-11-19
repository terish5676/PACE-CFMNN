"""Tuned fuzzy min-max baselines from hyperbox-brain 0.1.6 (tough review D4).

Fuzzy min-max networks that handle overlaps by design:

* EFMNN (enhanced FMNN): new overlap test and contraction cases;
* KNEFMNN: EFMNN choosing among the k nearest boxes when expanding;
* RFMNN (refined FMNN): new expansion, overlap test, contraction and
  prediction rules;
* IOL-GFMM (improved online GFMM): GFMM's min membership, expansion without
  contraction, distance-based tie-breaking.

Every model, CFMNN included, chooses theta by the same stratified 3-fold
cross-validation on the training rows, over the same grid. The CFMNN and
readable-baselines studies picked by mean validation accuracy, with an exact
tie going to the largest theta (the fewest boxes); this module uses that rule
too (``scripts/run_readable_baselines.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .cfmnn_core import CFMNN

THETAS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]       # larger theta first: ties go to fewer boxes
GAMMA = 4.0
FMM_MODELS = ["EFMNN", "KNEFMNN", "RFMNN", "IOL-GFMM"]


def make_fmm(name: str, theta: float, gamma: float = GAMMA):
    """A fresh, unfitted model."""
    if name == "CFMNN":
        return CFMNN(theta=theta, gamma=gamma)
    if name == "EFMNN":
        from hbbrain.numerical_data.incremental_learner.efmnn import EFMNNClassifier
        return EFMNNClassifier(theta=theta, gamma=gamma)
    if name == "KNEFMNN":
        from hbbrain.numerical_data.incremental_learner.knefmnn import KNEFMNNClassifier
        return KNEFMNNClassifier(theta=theta, gamma=gamma, k_neighbors=5)
    if name == "RFMNN":
        from hbbrain.numerical_data.incremental_learner.rfmnn import RFMNNClassifier
        return RFMNNClassifier(theta=theta, gamma=gamma)
    if name == "IOL-GFMM":
        from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
        return ImprovedOnlineGFMM(theta=theta, gamma=gamma)
    raise ValueError(f"unknown model {name!r}")


def n_boxes(model) -> int:
    if isinstance(model, CFMNN):
        return int(model.n_boxes_)
    return int(len(model.V))


def cv_folds(y, seed: int, k: int = 3):
    """Stratified folds, index lists sorted so each fold keeps the row order."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(va)) for tr, va in skf.split(np.zeros(len(y)), y)]


def _quiet_fit_predict(model, Xtr, ytr, Xva):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
        return np.asarray(model.predict(Xva))


def tune_theta(name: str, X, y, seed: int, grid=THETAS, folds=None, gamma: float = GAMMA):
    """(chosen theta, mean CV accuracy per theta). Exact ties go to the first (largest) theta."""
    folds = folds if folds is not None else cv_folds(y, seed)
    scores = []
    for t in grid:
        acc = [np.mean(_quiet_fit_predict(make_fmm(name, t, gamma), X[tr], y[tr], X[va]) == y[va])
               for tr, va in folds]
        scores.append(float(np.mean(acc)))
    s = np.round(np.asarray(scores), 12)
    return grid[int(np.flatnonzero(s == s.max())[0])], scores


def fit_tuned(name: str, X, y, seed: int, grid=THETAS, gamma: float = GAMMA):
    """Tune theta by CV, refit on all of X. Returns (model, theta, cv scores)."""
    theta, scores = tune_theta(name, X, y, seed, grid, gamma=gamma)
    model = make_fmm(name, theta, gamma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model, theta, scores
