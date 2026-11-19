"""Trust signals for a single CFMNN (blueprint 6.1-6.3).

* PD, prediction disagreement: share of K nearby inputs whose predicted class
  differs from the prediction at x.
* EI, explanation instability: how much the contrastive explanation for the
  original class pair (A, B) changes across the same K inputs.
* U, membership-based uncertainty: ambiguity between the top two classes plus
  low support for the top class, on the rescaled supports s' = 2s - 1.

None of these is a probability of error; they only rank cases. No function
here takes labels. The model is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN
from .explain import contrastive_pair
from .neighborhood import Perturber


@dataclass
class Signals:
    pd: np.ndarray            # (n,) prediction disagreement in [0, 1]
    ei: np.ndarray | None     # (n,) explanation instability in [0, 1]; nan where unavailable;
                              # None when computed with ei=False
    u: np.ndarray             # (n,) uncertainty in [0, 1], as in the blueprint
    box_switch: np.ndarray    # (n,) share of perturbations whose winning box changes (logged only)
    mode_switch: np.ndarray   # (n,) share of perturbations whose contrastive mode changes
    margin_unc: np.ndarray    # (n,) baseline: 1 - (s'_(1) - s'_(2)), no tie grading
    entropy: np.ndarray       # (n,) baseline: normalised entropy of s' (secondary)
    u_star: np.ndarray | None = None  # (n,) corrected U (ties first); a post-hoc variant
    tie: np.ndarray | None = None     # (n,) top two classes tie (both boxes contain x)

    @property
    def ei_coverage(self) -> float:
        if self.ei is None or len(self.ei) == 0:
            return 0.0
        return float(np.isfinite(self.ei).mean())


def box_evidence(model: CFMNN, X_train, X, det=None) -> np.ndarray:
    """Distance from each point to the nearest training row its winning box actually holds.

    CFMNN boxes grow up to the size limit, so a point can sit inside a box far
    from any training point the box absorbed (empty space). Higher = less
    evidence. Needs the training rows in the order ``fit`` saw them (the ids in
    ``members_``); it reads no labels.
    """
    from sklearn.neighbors import NearestNeighbors
    X_train, X = np.asarray(X_train, float), np.asarray(X, float)
    det = det if det is not None else model.predict_details(X)
    out = np.empty(len(X))
    for j in np.unique(det.winner):
        rows = det.winner == j
        nn = NearestNeighbors(n_neighbors=1).fit(X_train[model.members_[j]])
        out[rows] = nn.kneighbors(X[rows])[0][:, 0]
    return out


def uncertainty(model: CFMNN, det, eta: float = 0.5):
    """U = eta * Amb + (1 - eta) * (1 - s'_(1)) (blueprint 6.3).

    Amb = 1 - (s'_(1) - s'_(2)); when the top two classes tie, Amb is the
    centroid-distance ratio ||x - mu_A|| / ||x - mu_B|| instead, so ties are
    graded.

    Gate G1 found that this places tied points below most untied ones,
    although ties are where CFMNN errs most (Thyroid: U's AUROC < 0.5). The
    corrected variant U* ranks every tie above every untied point and grades
    ties by the same ratio: U* = (1 + ratio) / 2 for ties, and
    (1 - (s'_(1) - s'_(2))) / 2 otherwise. It was added after the check and is
    reported separately from the blueprint's U.

    Returns (u, u_star, plain margin uncertainty, normalised entropy of s', tie).
    """
    n = len(det.cls)
    rows = np.arange(n)
    s = det.support_rescaled
    s1 = s[rows, det.cls]
    has_b = det.runner_up >= 0
    b = np.maximum(det.runner_up, 0)
    s2 = np.where(has_b, s[rows, b], 0.0)
    plain = 1.0 - (s1 - s2)
    tie = has_b & (np.abs(s1 - s2) <= 2 * float(model.tie_tol))
    da = det.dist2_best[rows, det.cls]
    db = np.where(has_b, det.dist2_best[rows, b], np.inf)
    ratio = np.sqrt(np.divide(da, db, out=np.ones(n), where=np.isfinite(db) & (db > 0)))
    ratio = np.minimum(ratio, 1.0)
    amb = np.where(tie, ratio, plain)
    u = eta * amb + (1.0 - eta) * (1.0 - s1)
    u_star = np.where(tie, (1.0 + ratio) / 2.0, np.clip(plain, 0.0, 1.0) / 2.0)

    C = s.shape[1]
    tot = s.sum(axis=1, keepdims=True)
    p = np.divide(s, tot, out=np.full_like(s, 1.0 / C), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    ent = h / np.log(C) if C > 1 else np.zeros(n)
    return np.clip(u, 0.0, 1.0), u_star, np.clip(plain, 0.0, 1.0), ent, tie


def _ei_terms(e0, eZ, mode0, modeZ, lam, q):
    """Per-perturbation disagreement between e(x) (repeated) and e(x^(k))."""
    n0 = np.linalg.norm(e0, axis=1)
    nZ = np.linalg.norm(eZ, axis=1)
    den = n0 * nZ
    cos = np.divide((e0 * eZ).sum(axis=1), den, out=np.zeros(len(e0)), where=den > 0)
    term = lam * (1.0 - cos) / 2.0
    if lam < 1.0:
        d = e0.shape[1]
        top0 = np.argsort(-np.abs(e0), axis=1, kind="stable")[:, :q]
        topZ = np.argsort(-np.abs(eZ), axis=1, kind="stable")[:, :q]
        m0 = np.zeros((len(e0), d), bool)
        mZ = np.zeros((len(e0), d), bool)
        r = np.arange(len(e0))[:, None]
        m0[r, top0] = True
        mZ[r, topZ] = True
        jac = (m0 & mZ).sum(axis=1) / np.maximum((m0 | mZ).sum(axis=1), 1)
        term = term + (1.0 - lam) * (1.0 - jac)
    return np.where(mode0 != modeZ, 1.0, term)


class SignalComputer:
    """Compute PD, EI and U for a fitted CFMNN and a fitted Perturber.

    Parameters
    ----------
    lam : float
        Weight of the cosine term in EI (blueprint 0.5); forced to 1 when d <= 3,
        where the top-q ranking carries no information.
    q : int or None
        Top-q features compared in EI; default min(5, d - 1).
    eta : float
        Weight of ambiguity in U (blueprint 0.5).
    max_rows : int
        Perturbed rows evaluated at once (a memory bound). It decides which
        random draws each point gets, so keep it fixed when comparing runs.
    ei : bool
        False skips EI (and the mode-switch log), which is most of the cost.
        PD and U are unchanged: EI uses no random draws. EI is then None and
        the mode-switch share is nan.
    """

    def __init__(self, model: CFMNN, perturber: Perturber, lam: float = 0.5,
                 q: int | None = None, eta: float = 0.5, max_rows: int = 200_000,
                 ei: bool = True):
        self.model = model
        self.perturber = perturber
        d = model.n_features_in_
        self.lam = 1.0 if d <= 3 else lam
        self.q = q if q is not None else max(1, min(5, d - 1))
        self.eta = eta
        self.max_rows = max_rows
        self.ei = ei

    def compute(self, X, seed=0) -> Signals:
        m = self.model
        X = m._check_X(X)
        n = len(X)
        K = self.perturber.k
        rng = np.random.default_rng(seed)
        out = {k: np.empty(n) for k in ("pd", "ei", "box_switch", "mode_switch")}
        step = max(1, self.max_rows // K)
        det_all = m.predict_details(X)
        for s in range(0, n, step):
            Xc = X[s:s + step]
            c = len(Xc)
            det0 = m.predict_details(Xc)
            A, B = det0.cls, det0.runner_up
            Z = self.perturber.sample(Xc, rng).reshape(c * K, -1)
            detZ = m.predict_details(Z)
            rep = lambda v: np.repeat(v, K)
            out["pd"][s:s + c] = (detZ.cls != rep(A)).reshape(c, K).mean(axis=1)
            out["box_switch"][s:s + c] = (detZ.winner != rep(det0.winner)).reshape(c, K).mean(axis=1)
            if not self.ei:
                continue

            e0 = contrastive_pair(m, Xc, A, B, det=det0)
            eZ = contrastive_pair(m, Z, rep(A), rep(B), det=detZ)
            avail = (e0.mode != "none") & (np.abs(e0.e).sum(axis=1) > 0)
            terms = _ei_terms(np.repeat(e0.e, K, axis=0), eZ.e, rep(e0.mode), eZ.mode,
                              self.lam, self.q).reshape(c, K)
            out["ei"][s:s + c] = np.where(avail, terms.mean(axis=1), np.nan)
            out["mode_switch"][s:s + c] = (eZ.mode != rep(e0.mode)).reshape(c, K).mean(axis=1)
        u, u_star, plain, ent, tie = uncertainty(m, det_all, self.eta)
        if not self.ei:
            out["ei"], out["mode_switch"] = None, np.full(n, np.nan)
        return Signals(out["pd"], out["ei"], u, out["box_switch"], out["mode_switch"], plain, ent,
                       u_star, tie)
