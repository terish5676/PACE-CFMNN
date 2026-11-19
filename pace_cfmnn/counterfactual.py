"""Box-directed counterfactuals, verified on the model (blueprint 5.3).

* ``counterfactual`` (PACE): the smallest move of x into a box of the target
  class, then greedy resetting of changed features.
* ``sparse_counterfactual`` (PACE-sparse, RQ1 extension B1): the fewest
  changed features, found by scoring subsets of features exactly from the
  model's own terms.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .cfmnn_core import CFMNN, feature_terms


@dataclass
class Counterfactual:
    found: bool
    x_cf: np.ndarray | None
    target: object               # target label
    box: int                     # target box used (-1 if none)
    l1: float = np.nan           # L1 distance from x
    n_changed: int = 0           # features changed
    plausibility: float = np.nan  # L1 distance to nearest training point of the target class
    model_calls: int = 0
    how: str = ""                # PACE-sparse only: "exact", "greedy" or "pace" (fallback)


def _sparsify(model: CFMNN, x, c, t):
    """Greedily reset changed features to x (smallest change first) while the
    prediction stays t. Every kept step is verified on the model."""
    c = c.copy()
    calls = 0
    changed = np.flatnonzero(~np.isclose(c, x))
    for i in changed[np.argsort(np.abs(c - x)[changed])]:
        trial = c.copy()
        trial[i] = x[i]
        calls += 1
        if model.predict_details(trial[None]).cls[0] == t:
            c = trial
    return c, calls


def counterfactual(model: CFMNN, x, target, locked=None, X_train=None, y_train=None,
                   max_boxes: int = 10, steps=(0.25, 0.5, 0.75, 1.0),
                   sparse: bool = True) -> Counterfactual:
    """Smallest move of x into a box of the target class that the model confirms.

    1. Rank target-class boxes by L1 distance on mutable features, skipping boxes
       whose range excludes a locked feature's current value.
    2. Clip x into the box on mutable features and re-run the model.
    3. If it loses a tie, move towards that box's centroid (staying in the box).
    """
    x = np.asarray(x, float).ravel()
    d = len(x)
    locked = np.zeros(d, bool) if locked is None else np.asarray(locked, bool)
    mut = ~locked
    t = int(np.flatnonzero(model.classes_ == target)[0])
    boxes = model.class_boxes_[t]
    V, W = model.V_[boxes], model.W_[boxes]
    ok = np.all((V[:, locked] <= x[locked]) & (x[locked] <= W[:, locked]), axis=1)
    clipped = np.clip(x, V, W)
    clipped[:, locked] = x[locked]
    dist = np.abs(clipped - x)[:, mut].sum(axis=1)
    calls = 0
    for k in np.argsort(np.where(ok, dist, np.inf), kind="stable")[:max_boxes]:
        if not ok[k]:
            break
        j = boxes[k]
        cand = clipped[k].copy()
        candidates = [cand] + [np.where(mut, np.clip(cand + s * (model.centroid_[j] - cand),
                                                     model.V_[j], model.W_[j]), x)
                               for s in steps]
        for c in candidates:
            calls += 1
            if model.predict_details(c[None]).cls[0] == t:
                if sparse:
                    c, extra = _sparsify(model, x, c, t)
                    calls += extra
                cf = Counterfactual(True, c, target, int(j), float(np.abs(c - x).sum()),
                                    int(np.sum(~np.isclose(c, x))), model_calls=calls)
                if X_train is not None and y_train is not None:
                    pts = np.asarray(X_train)[np.asarray(y_train) == target]
                    if len(pts):
                        cf.plausibility = float(np.abs(pts - c).sum(axis=1).min())
                return cf
    return Counterfactual(False, None, target, -1, model_calls=calls)


def _n_changed(c, x) -> int:
    return int(np.sum(np.abs(np.asarray(c) - x) > 1e-9))


def replacement_values(X_train, n_values: int = 16) -> list[np.ndarray]:
    """For each feature, up to ``n_values`` values that occur in the training data: the
    nearest-rank quantiles at (k + 0.5) / n_values, without repeats."""
    X = np.asarray(X_train, dtype=float)
    q = (np.arange(n_values) + 0.5) / n_values
    return [np.unique(np.quantile(X[:, i], q, method="nearest")) for i in range(X.shape[1])]


def brute_force_counterfactual(model: CFMNN, x, target, values, locked=None, max_features: int = 2,
                               chunk: int = 20_000) -> Counterfactual:
    """The smallest-L1 valid counterfactual that changes at most ``max_features`` (1 or 2)
    features, each to one of ``values[i]`` (see ``replacement_values``); not found if none is valid.

    Every candidate is checked on the model, so a counterfactual is valid by construction. Ties in
    L1 go to the candidate met first: single features in order, then pairs in lexicographic order.
    """
    x = np.asarray(x, float).ravel()
    d = len(x)
    locked = np.zeros(d, bool) if locked is None else np.asarray(locked, bool)
    mut = np.flatnonzero(~locked)
    best, best_l1, calls = None, np.inf, 0

    def consider(C):
        nonlocal best, best_l1, calls
        calls += len(C)
        ok = np.flatnonzero(model.predict(C) == target)
        if len(ok):
            l1 = np.abs(C[ok] - x).sum(axis=1)
            k = int(np.argmin(l1))
            if l1[k] < best_l1:
                best, best_l1 = C[ok[k]].copy(), float(l1[k])

    for i in mut:
        C = np.repeat(x[None, :], len(values[i]), axis=0)
        C[:, i] = values[i]
        consider(C)
    if max_features >= 2:
        for a, b in itertools.combinations(mut, 2):
            ga, gb = np.meshgrid(values[a], values[b], indexing="ij")
            ga, gb = ga.ravel(), gb.ravel()
            for s in range(0, len(ga), chunk):
                C = np.repeat(x[None, :], len(ga[s:s + chunk]), axis=0)
                C[:, a], C[:, b] = ga[s:s + chunk], gb[s:s + chunk]
                consider(C)
    if best is None:
        return Counterfactual(False, None, target, -1, model_calls=calls)
    return Counterfactual(True, best, target, -1, best_l1, _n_changed(best, x), model_calls=calls)


class _Moves:
    """Moving features ``F`` of x to ``vals[F]`` (one target box, one step).

    Each moved feature changes every box's membership by a fixed amount
    (its term at the new value minus its term at x, over d) and every box's
    squared centroid distance by a fixed amount. So the model's decision for
    any subset of F (highest membership, ties within tie_tol to the nearest
    centroid, then the lowest box index) is computed exactly from sums,
    without calling the model. Boxes whose highest possible membership is
    below the best lowest one (minus tie_tol) can never win or tie and are
    left out; the target box itself is always kept.
    """

    def __init__(self, x, vals, F, b0, D0, bc, T0, model, j):
        d = len(x)
        mu = model.centroid_
        dM = (feature_terms(vals[F], model.V_[:, F], model.W_[:, F], model.gamma) - T0[:, F]) / d
        dD = (vals[F] - mu[:, F]) ** 2 - (x[F] - mu[:, F]) ** 2
        ceil = b0 + np.maximum(dM, 0).sum(axis=1)
        floor = b0 + np.minimum(dM, 0).sum(axis=1)
        live = np.union1d(np.flatnonzero(ceil >= floor.max() - model.tie_tol - 1e-12), [j])
        self.vals, self.F, self.m = vals, F, len(F)
        self.b, self.D, self.cls = b0[live], D0[live], bc[live]
        self.dM, self.dD = dM[live], dD[live]
        self.l1 = np.abs(vals[F] - x[F])
        self.jl = int(np.searchsorted(live, j))
        self.tol = float(model.tie_tol)

    def decide(self, combos):
        """Predicted class index for each subset (rows of ``combos``, indices into F)."""
        M = self.b[:, None] + self.dM[:, combos].sum(axis=2)
        D = self.D[:, None] + self.dD[:, combos].sum(axis=2)
        top = M.max(axis=0)
        win = np.argmin(np.where(M >= top - self.tol, D, np.inf), axis=0)
        return self.cls[win], M, D

    def chunks(self, size, budget=2_000_000):
        step = max(1, budget // max(1, len(self.b) * size))
        it = itertools.combinations(range(self.m), size)
        while True:
            block = list(itertools.islice(it, step))
            if not block:
                return
            yield np.array(block, dtype=np.intp)

    def greedy(self, t):
        """Forward selection until the target wins, then drop-one; None if it never wins.

        Each step adds the feature that gives a winning subset with the
        smallest L1 change, else the one that most raises the target class's
        best membership over the other classes' (ties: nearest to the target
        box's centroid)."""
        S, rest = [], list(range(self.m))
        isT = self.cls == t
        while rest:
            combos = np.array([S + [f] for f in rest], dtype=np.intp)
            cls, M, D = self.decide(combos)
            hit = np.flatnonzero(cls == t)
            if len(hit):
                S = [int(f) for f in combos[hit[np.argmin(self.l1[combos[hit]].sum(axis=1))]]]
                break
            other = M[~isT].max(axis=0) if (~isT).any() else np.full(len(rest), -np.inf)
            gap = M[isT].max(axis=0) - other
            S.append(rest.pop(int(np.lexsort((D[self.jl], -gap))[0])))
        else:
            return None
        for f in list(S):
            trial = [g for g in S if g != f]
            if trial and self.decide(np.array([trial], dtype=np.intp))[0][0] == t:
                S = trial
        return S


def sparse_counterfactual(model: CFMNN, x, target, locked=None, X_train=None, y_train=None,
                          max_boxes: int = 10, steps=(0.25, 0.5, 0.75, 1.0),
                          max_subsets: int = 65_536, max_checks: int = 20) -> Counterfactual:
    """PACE-sparse: the fewest changed features that make the model predict the target.

    1. Candidate target-class boxes: the ``max_boxes`` nearest by L1 (as in
       PACE) and the ``max_boxes`` with the fewest features outside their
       intervals. Boxes whose range excludes a locked feature's value are
       skipped.
    2. For each box and each step s in (0,) + ``steps``, a mutable feature's
       new value is x clipped into the box, then moved a share s towards the
       box's centroid (s = 0: just clipped, which changes only the features
       outside the box's intervals). Subsets of the moved features are
       scored exactly from the model's own terms (``_Moves``), smallest
       first; among winning subsets of the smallest size, the smallest L1
       change wins. The search is exhaustive while a size has at most
       ``max_subsets`` subsets; beyond that, greedy forward selection then
       drop-one is used.
    3. Every answer is verified with the model (up to ``max_checks`` per box).
    4. PACE's own counterfactual is also computed; if it changes fewer
       features (or as many with a smaller L1), it is returned instead
       (``how="pace"``). So PACE-sparse never changes more features than PACE.
    """
    x = np.asarray(x, float).ravel()
    d = len(x)
    locked = np.zeros(d, bool) if locked is None else np.asarray(locked, bool)
    mut = ~locked
    t = int(np.flatnonzero(model.classes_ == target)[0])
    boxes = model.class_boxes_[t]
    V, W = model.V_[boxes], model.W_[boxes]
    ok = np.all((V[:, locked] <= x[locked]) & (x[locked] <= W[:, locked]), axis=1)
    clipped = np.clip(x, V, W)
    clipped[:, locked] = x[locked]
    dist = np.abs(clipped - x)[:, mut].sum(axis=1)
    n_out = (np.abs(clipped - x) > 1e-12).sum(axis=1)
    cand = []
    for order in (np.argsort(dist, kind="stable"), np.lexsort((dist, n_out))):
        cand += [int(k) for k in order if ok[k]][:max_boxes]
    cand = list(dict.fromkeys(cand))

    T0 = feature_terms(x, model.V_, model.W_, model.gamma)          # (H, d)
    b0 = T0.mean(axis=1)
    D0 = ((x - model.centroid_) ** 2).sum(axis=1)
    bc = model.box_class_
    calls = 0
    best = None                                                      # (n changed, l1, cf, box, how)
    for k in cand:
        j = int(boxes[k])
        opts = []
        for s in (0.0,) + tuple(steps):
            vals = np.where(mut, np.clip(clipped[k] + s * (model.centroid_[j] - clipped[k]),
                                         V[k], W[k]), x)
            F = np.flatnonzero(np.abs(vals - x) > 1e-12)
            if len(F):
                opts.append(_Moves(x, vals, F, b0, D0, bc, T0, model, j))
        limit = max((o.m for o in opts), default=0)
        if best is not None:
            limit = min(limit, best[0])
        hits, greedy, size = [], set(), 0
        for size in range(1, limit + 1):
            searched = False
            for oi, o in enumerate(opts):
                if size > o.m:
                    continue
                if math.comb(o.m, size) > max_subsets:
                    greedy.add(oi)
                    continue
                searched = True
                for combos in o.chunks(size):
                    win = combos[o.decide(combos)[0] == t]
                    hits += [(float(l), oi, tuple(int(f) for f in c))
                             for l, c in zip(o.l1[win].sum(axis=1), win)]
            if hits or not searched:
                break
        found = []
        if hits:
            hits.sort()
            found = [(size, l1, oi, list(c), "exact") for l1, oi, c in hits[:max_checks]]
        for oi in sorted(greedy):
            S = opts[oi].greedy(t)
            if S is not None and (not hits or len(S) < size):
                found.append((len(S), float(opts[oi].l1[S].sum()), oi, S, "greedy"))
        found.sort(key=lambda r: (r[0], r[1]))
        for _, _, oi, S, how in found[:max_checks]:
            o = opts[oi]
            c = x.copy()
            idx = o.F[S]
            c[idx] = o.vals[idx]
            calls += 1
            if model.predict_details(c[None]).cls[0] == t:
                key = (_n_changed(c, x), float(np.abs(c - x).sum()))
                if best is None or key < best[:2]:
                    best = key + (c, j, how)
                break

    pace = counterfactual(model, x, target, locked=locked, X_train=X_train, y_train=y_train,
                          max_boxes=max_boxes, steps=steps, sparse=True)
    calls += pace.model_calls
    if pace.found and (best is None or (_n_changed(pace.x_cf, x), pace.l1) < best[:2]):
        pace.how, pace.model_calls = "pace", calls
        return pace
    if best is None:
        return Counterfactual(False, None, target, -1, model_calls=calls, how="none")
    c = best[2]
    cf = Counterfactual(True, c, target, best[3], float(np.abs(c - x).sum()),
                        int(np.sum(~np.isclose(c, x))), model_calls=calls, how=best[4])
    if X_train is not None and y_train is not None:
        pts = np.asarray(X_train)[np.asarray(y_train) == target]
        if len(pts):
            cf.plausibility = float(np.abs(pts - c).sum(axis=1).min())
    return cf
