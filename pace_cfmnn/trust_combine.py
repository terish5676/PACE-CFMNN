"""Learned combination of trust signals (RQ2 v2).

Each signal sees CFMNN's possible errors from a different side:

* **overlap** (model geometry): U*, the plain margin, and whether the top two
  classes tie;
* **empty space** (model geometry): ``box_evidence`` (distance to the nearest
  training row the winning box holds) and the winning box's member count;
* **order dependence** (the model's training): disagreement of CFMNNs trained
  on the same rows in other orders;
* **the data** (standard baselines): kNN disagreement and Trust Score.

``LearnedCombiner`` is a logistic regression on standardised signals, fitted
on a labelled calibration set (the inner rows) to predict the model's errors.
The same combiner is used for every group of signals, so groups are compared
on equal terms.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import trust_baselines as tb
from .cfmnn_core import CFMNN
from .signals import SignalComputer, box_evidence, uncertainty

NATIVE = ("U*", "Margin", "Tie", "Box evidence", "Box count", "Order ensemble")
DATA = ("kNN", "Trust Score")
ALL = NATIVE + DATA


def order_members(model: CFMNN, X_train, y_train, n_models: int = 5, seed: int = 0) -> list:
    """The order-ensemble members, trained once (same orders as trust_baselines)."""
    X_train, y_train = np.asarray(X_train, float), np.asarray(y_train)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_models):
        order = rng.permutation(len(X_train))
        out.append(clone(model).fit(X_train[order], y_train[order]))
    return out


# Generic ensembles for the review's Test 3 (results/review_v1/PREREGISTRATION.md): is the
# order ensemble's gain specific to CFMNN, or does any ensemble's disagreement give it?
ENSEMBLES = ("Bagged ensemble", "SGD order ensemble")


def bagged_members(model: CFMNN, X_train, y_train, n_models: int = 5, seed: int = 0) -> list:
    """A generic ensemble: copies of the model, each trained on a bootstrap sample of the rows.

    The sampled rows keep the model's own order (sorted indices), so the members
    differ only in which rows they see, not in the order that the order ensemble varies.
    """
    X_train, y_train = np.asarray(X_train, float), np.asarray(y_train)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_models):
        idx = np.sort(rng.integers(len(X_train), size=len(X_train)))
        out.append(clone(model).fit(X_train[idx], y_train[idx]))
    return out


def sgd_order_members(X_train, y_train, n_models: int = 5, seed: int = 0) -> list:
    """An order ensemble of another learner: logistic regressions trained by stochastic
    gradient descent in one pass, each over the rows in a different random order."""
    from sklearn.linear_model import SGDClassifier
    X_train, y_train = np.asarray(X_train, float), np.asarray(y_train)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_models):
        order = rng.permutation(len(X_train))
        clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, shuffle=False, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # one pass on purpose (ConvergenceWarning)
            out.append(clf.fit(X_train[order], y_train[order]))
    return out


def ensemble_features(X, pred, bagged, sgd) -> dict:
    """Share of each ensemble's members that disagree with the model's prediction."""
    X = np.asarray(X, float)
    return {"Bagged ensemble": np.mean([m.predict(X) != pred for m in bagged], axis=0),
            "SGD order ensemble": np.mean([m.predict(X) != pred for m in sgd], axis=0)}


def trust_features(model: CFMNN, X_train, y_train, X, members, pd_computer=None, pd_seed=0):
    """All signals for the points X (higher = more suspicious), and the predictions."""
    X = np.asarray(X, float)
    det = model.predict_details(X)
    pred = model.classes_[det.cls]
    _, u_star, margin, _, tie = uncertainty(model, det)
    F = {"U*": u_star, "Margin": margin, "Tie": tie.astype(float),
         "Box evidence": box_evidence(model, X_train, X, det),
         "Box count": -np.log1p(model.count_[det.winner]),        # fewer members = less evidence
         "Order ensemble": np.mean([m.predict(X) != pred for m in members], axis=0),
         "kNN": tb.knn_disagreement(X_train, y_train, X, pred),
         "Trust Score": tb.trust_score(X_train, y_train, X, pred)}
    if pd_computer is not None:
        F["PD"] = pd_computer.compute(X, seed=pd_seed).pd
    return F, pred


class LearnedCombiner:
    """Logistic regression from signals to 'the model is wrong here', fitted on labelled rows."""

    def __init__(self, features, C: float = 1.0):
        self.features = tuple(features)
        self.C = C

    def _matrix(self, F) -> np.ndarray:
        return np.column_stack([np.asarray(F[f], float) for f in self.features])

    def fit(self, F, is_error) -> "LearnedCombiner":
        y = np.asarray(is_error, bool)
        self.constant_ = None
        if y.all() or not y.any():              # nothing to learn: every point scores the same
            self.constant_ = float(y.mean())
            return self
        self.model_ = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=self.C, class_weight="balanced",
                                                       max_iter=5000))
        self.model_.fit(self._matrix(F), y)
        return self

    def score(self, F) -> np.ndarray:
        n = len(np.asarray(F[self.features[0]]))
        if self.constant_ is not None:
            return np.full(n, self.constant_)
        return self.model_.predict_proba(self._matrix(F))[:, 1]


def pd_computer(model: CFMNN, perturber) -> SignalComputer:
    """PD only (EI skipped), for the optional PD feature."""
    return SignalComputer(model, perturber, ei=False)
