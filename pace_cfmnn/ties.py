"""Tie counts for a fitted model (Runs I and K).

Two counts, both read from ``predict_details``:

* **Training errors that are ties** (the paper's Proposition 1). An error is a *tie* when the
  row's own class reaches the winner's membership (within ``tie_tol``), so the membership did
  not decide and the nearest-centroid rule picked the wrong class. It is a tie *inside both
  boxes* when both memberships are 1: the row lies inside a box of its own class and inside a
  box of the winning class. Proposition 1 says that after ``fit`` every training error is of
  the second kind. After a contraction or a shrink it need not hold.
* **Predictions decided by the nearest centroid.** The best box of another class has the
  winner's membership (the rule of ``scripts/peer_review_reanalysis.tie_rows``), so the class
  comes from the tie-break.
"""

from __future__ import annotations

import numpy as np

from .cfmnn_core import CFMNN


def training_error_ties(model: CFMNN, X, y) -> dict:
    """Training errors, and how many of them are ties (see the module docstring)."""
    det = model.predict_details(X)
    t = np.searchsorted(model.classes_, np.asarray(y))
    err = np.flatnonzero(det.cls != t)
    own = np.nan_to_num(det.support[err, t[err]], nan=-np.inf)   # nan: the class has no box left
    win = det.support[err, det.cls[err]]
    tied = own >= win - float(model.tie_tol)
    both = (own >= 1.0) & (win >= 1.0)
    return {"train_errors": int(len(err)), "train_error_pct": 100 * len(err) / len(t),
            "train_errors_tied": int(tied.sum()), "train_errors_tied_inside_both": int(both.sum())}


def centroid_decided(model: CFMNN, X, y) -> dict:
    """Rows whose class comes from the nearest-centroid tie-break, and the errors among them."""
    det = model.predict_details(X)
    s = np.where(det.best_box >= 0, det.support_rescaled, -np.inf)
    rows = np.arange(len(det.cls))
    own = s[rows, det.cls].copy()
    s[rows, det.cls] = -np.inf
    tied = own - s.max(axis=1) <= 2 * float(model.tie_tol)
    wrong = det.y_pred != np.asarray(y)
    return {"n": int(len(rows)), "n_centroid_decided": int(tied.sum()), "n_errors": int(wrong.sum()),
            "n_errors_centroid_decided": int((wrong & tied).sum())}
