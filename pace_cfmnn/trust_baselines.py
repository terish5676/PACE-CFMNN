"""Standard trust baselines for the RQ2 comparison.

Each returns a suspicion score per point (higher = more likely wrong), like
the PACE signals. Unlike PD, EI and U, all three read the training labels.

* ``trust_score``: Jiang, Kim, Guan and Gupta, "To Trust or Not to Trust a
  Classifier", NeurIPS 2018. Distance to the predicted class's training data
  compared with distance to the nearest other class, after removing each
  class's least dense training points.
* ``order_ensemble_disagreement``: CFMNN learns in one pass, so its boxes
  depend on the order of the training rows. Models trained on the same rows
  in other orders form a natural committee; the score is the share that
  disagrees with the main model.
* ``knn_disagreement``: share of the nearest training rows whose label
  differs from the prediction.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

from .cfmnn_core import CFMNN


def trust_score(X_train, y_train, X, pred, k: int = 10, alpha: float = 0.1) -> np.ndarray:
    """Suspicion d_pred / (d_pred + d_other) in [0, 1]; 0.5 = equally close to both.

    It ranks points exactly as minus the trust score d_other / d_pred of Jiang et al.
    """
    X_train, X = np.asarray(X_train, float), np.asarray(X, float)
    y_train, pred = np.asarray(y_train), np.asarray(pred)
    classes = np.unique(y_train)
    dist = np.full((len(X), len(classes)), np.inf)
    for c_i, c in enumerate(classes):
        pts = X_train[y_train == c]
        if len(pts) == 0:
            continue
        if alpha > 0 and len(pts) > k + 1:
            # k-NN radius within the class (first neighbour is the point itself).
            r = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts)[0][:, -1]
            pts = pts[r <= np.quantile(r, 1.0 - alpha)]
        dist[:, c_i] = NearestNeighbors(n_neighbors=1).fit(pts).kneighbors(X)[0][:, 0]
    col = np.searchsorted(classes, pred)
    rows = np.arange(len(X))
    d_pred = dist[rows, col]
    other = dist.copy()
    other[rows, col] = np.inf
    d_other = other.min(axis=1)
    tot = d_pred + d_other
    return np.divide(d_pred, tot, out=np.full(len(X), 0.5), where=np.isfinite(tot) & (tot > 0))


def order_ensemble_disagreement(model: CFMNN, X_train, y_train, X, pred, n_models: int = 5,
                                seed: int = 0) -> np.ndarray:
    """Share of n_models CFMNNs, trained on the same rows in other random orders, that disagree."""
    X_train, y_train = np.asarray(X_train, float), np.asarray(y_train)
    rng = np.random.default_rng(seed)
    votes = np.zeros(len(X))
    for _ in range(n_models):
        order = rng.permutation(len(X_train))
        member = clone(model).fit(X_train[order], y_train[order])
        votes += member.predict(X) != np.asarray(pred)
    return votes / n_models


def knn_disagreement(X_train, y_train, X, pred, k: int = 10) -> np.ndarray:
    """Share of the k nearest training rows whose label differs from the prediction."""
    nn = NearestNeighbors(n_neighbors=k).fit(np.asarray(X_train, float))
    idx = nn.kneighbors(np.asarray(X, float))[1]
    return (np.asarray(y_train)[idx] != np.asarray(pred)[:, None]).mean(axis=1)
