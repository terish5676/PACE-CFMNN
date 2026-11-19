"""Simpson's overlap test and contraction, applied to a trained CFMNN.

CFMNN trains by expansion only, so boxes of different classes may overlap. Simpson's fuzzy
min-max classifier (IEEE Trans. Neural Networks 3(5), 1992) removes such overlaps: after a box
expands, it is tested against every box of another class and, where the two overlap, both are
contracted along the one feature with the smallest overlap. This module applies that test and
that contraction to a model that is already trained, as the standard remedy to compare
self-repair with (``refine.shrink_box`` and ``active.run_self_repair``).

The overlap test, for boxes j and k and each feature i (Simpson's four cases, as published,
with his strict and non-strict inequalities):

1. ``v_j < v_k < w_j < w_k``: j's upper part overlaps k's lower part; overlap ``w_j - v_k``.
2. ``v_k < v_j < w_k < w_j``: the mirror image; overlap ``w_k - v_j``.
3. ``v_j < v_k <= w_k < w_j``: k lies inside j; overlap ``min(w_k - v_j, w_j - v_k)``.
4. ``v_k < v_j <= w_j < w_k``: j lies inside k; overlap ``min(w_j - v_k, w_k - v_j)``.

The boxes overlap only if one case holds on every feature. The feature with the smallest
overlap is contracted (the lowest index on a tie):

1. ``w_j = v_k = (w_j + v_k) / 2``.
2. ``w_k = v_j = (w_k + v_j) / 2``.
3. if ``w_k - v_j < w_j - v_k``: ``v_j = w_k``; otherwise ``w_j = v_k``.
4. if ``w_k - v_j < w_j - v_k``: ``w_k = v_j``; otherwise ``v_k = w_j``.

Two limits of the published test are kept on purpose, because this is the comparator:

* A feature where the two boxes share a face (for example ``v_j = v_k``), or are equal, fits
  no case, so the pair counts as not overlapping even if the boxes intersect. Later variants of
  fuzzy min-max add cases for this; Simpson's test does not.
* After a contraction the two boxes still touch on the new face.

The contraction does not look at labels or at which rows a box holds. Afterwards each changed
box keeps only the training members still inside it, and its Sum, Count and centroid are
recomputed from them, as ``refine.shrink_box`` does; a box left with no member is removed. The
rows dropped this way belong to no box ("evicted"). Only the fixed centroid rule is supported.

A second rule, ``rule="gfmm"``, uses GFMM's overlap test and contraction as implemented in
hyperbox-brain (Khuat and Gabrys 2021): closed intervals, so shared faces count as overlap,
four more shared-face cases, and a gap of 1e-5 after each contraction. It is applied to the
same trained CFMNN; it is not the same as training GFMM. A face moved past the other face of
its own box empties that box, which is then removed.

Two entry points, each with ``rule`` "simpson" (default) or "gfmm":

* ``contract_all``: every pair of boxes of different classes, in box order, in sweeps until a
  sweep finds nothing (a pair is contracted at most once, so this ends).
* ``contract_at_errors``: only at the model's own training errors, one error at a time: the
  winning wrong-class box against the row's own-class box. No gate: every contraction is kept.
"""

from __future__ import annotations

import copy

import numpy as np

from .cfmnn_core import CFMNN
from .refine import _require_fixed, contains

GFMM_ALPHA = 1e-5   # hyperbox-brain's default gap after a GFMM contraction


# ------------------------------------------------------------------ the test
def overlap_case(vj: float, wj: float, vk: float, wk: float) -> tuple[int, float]:
    """Simpson's case (1-4) for one feature and the overlap there; (0, inf) if no case holds."""
    if vj < vk < wj < wk:
        return 1, wj - vk
    if vk < vj < wk < wj:
        return 2, wk - vj
    if vj < vk <= wk < wj:
        return 3, min(wk - vj, wj - vk)
    if vk < vj <= wj < wk:
        return 4, min(wj - vk, wk - vj)
    return 0, float("inf")


def overlap_test(vj, wj, vk, wk) -> tuple[int, int, float]:
    """Simpson's overlap test for two boxes: (feature to contract, its case, its overlap).

    The feature is -1 (case 0) when some feature fits no case: no overlap by this test.
    """
    best, best_case, smallest = -1, 0, float("inf")
    for i in range(len(vj)):
        case, delta = overlap_case(vj[i], wj[i], vk[i], wk[i])
        if case == 0:
            return -1, 0, float("inf")
        if delta < smallest:               # strict: the lowest feature wins a tie
            best, best_case, smallest = i, case, delta
    return best, best_case, float(smallest)


def _candidates(V, W, j: int, ks: np.ndarray) -> np.ndarray:
    """Those of boxes ``ks`` that Simpson's test says overlap box j (one case on every feature)."""
    vj, wj, vk, wk = V[j], W[j], V[ks], W[ks]
    c1 = (vj < vk) & (vk < wj) & (wj < wk)
    c2 = (vk < vj) & (vj < wk) & (wk < wj)
    c3 = (vj < vk) & (wk < wj)             # v_k <= w_k always holds
    c4 = (vk < vj) & (wj < wk)
    return ks[(c1 | c2 | c3 | c4).all(axis=1)]


# ----------------------------------------------------------- the contraction
def contract_pair(V, W, j: int, k: int) -> dict | None:
    """Test boxes j and k and contract them in place. Returns what changed, or None.

    ``V`` and ``W`` are the arrays of lower and upper corners. Only one feature changes: in
    cases 1 and 2 both boxes move to the middle of the overlap, in cases 3 and 4 one box does.
    """
    i, case, delta = overlap_test(V[j], W[j], V[k], W[k])
    if i < 0:
        return None
    if case == 1:
        V[k, i] = W[j, i] = 0.5 * (W[j, i] + V[k, i])
        changed = (j, k)
    elif case == 2:
        V[j, i] = W[k, i] = 0.5 * (W[k, i] + V[j, i])
        changed = (j, k)
    elif case == 3:
        if W[k, i] - V[j, i] < W[j, i] - V[k, i]:
            V[j, i] = W[k, i]
        else:
            W[j, i] = V[k, i]
        changed = (j,)
    else:
        if W[k, i] - V[j, i] < W[j, i] - V[k, i]:
            W[k, i] = V[j, i]
        else:
            V[k, i] = W[j, i]
        changed = (k,)
    return {"box_j": int(j), "box_k": int(k), "feature": int(i), "case": int(case),
            "overlap": float(delta), "changed": changed}


def _candidates_closed(V, W, j: int, ks: np.ndarray) -> np.ndarray:
    """Those of boxes ``ks`` whose closed intervals meet box j's on every feature (GFMM's test)."""
    return ks[(np.maximum(V[j], V[ks]) <= np.minimum(W[j], W[ks])).all(axis=1)]


def contract_pair_gfmm(V, W, j: int, k: int, alpha: float = GFMM_ALPHA) -> dict | None:
    """GFMM's overlap test and contraction on boxes j and k, in place (hyperbox-brain).

    The test (``is_two_hyperboxes_overlap_num_data_general``) counts closed intervals, so
    shared faces overlap; the contraction (``overlap_resolving_num_data``, Khuat and Gabrys
    2021) has Simpson's four cases with non-strict inequalities plus four shared-face cases,
    and leaves a gap of ``alpha``. Returns None if the test finds no overlap or no case moves a
    face (for example when the two intervals are equal on every feature).
    """
    from hbbrain.utils.adjust_hyperbox import (is_two_hyperboxes_overlap_num_data_general,
                                               overlap_resolving_num_data)
    if not is_two_hyperboxes_overlap_num_data_general(V[j], W[j], V[k], W[k]):
        return None
    vj, wj, vk, wk = V[j].copy(), W[j].copy(), V[k].copy(), W[k].copy()
    vj, wj, vk, wk = overlap_resolving_num_data(vj, wj, 0, vk, wk, 1, alpha)
    dj = (vj != V[j]) | (wj != W[j])
    dk = (vk != V[k]) | (wk != W[k])
    if not (dj.any() or dk.any()):
        return None
    i = int(np.flatnonzero(dj | dk)[0])
    V[j], W[j], V[k], W[k] = vj, wj, vk, wk
    changed = tuple(b for b, d in ((j, dj), (k, dk)) if d.any())
    return {"box_j": int(j), "box_k": int(k), "feature": i, "case": 0,
            "overlap": float(min(W[j, i], W[k, i]) - max(V[j, i], V[k, i])), "changed": changed}


RULES = {"simpson": (_candidates, contract_pair), "gfmm": (_candidates_closed, contract_pair_gfmm)}


def refresh_box(model: CFMNN, k: int, X_lab: np.ndarray) -> tuple[int, bool]:
    """Box k keeps the members still inside it (as ``refine.shrink_box``).

    Returns (rows evicted, box removed). A box with no member left is removed, so later boxes
    move up one index.
    """
    members = model.members_[k]
    P = X_lab[members]
    keep = members[((P >= model.V_[k]) & (P <= model.W_[k])).all(axis=1)]
    cut = int(len(members) - len(keep))
    if len(keep) == 0:
        model._remove_box(k)
        return cut, True
    if cut:
        model.members_[k] = keep
        model.sum_[k] = X_lab[keep].sum(axis=0)
        model.count_[k] = len(keep)
        model.centroid_[k] = model.sum_[k] / len(keep)
    return cut, False


def _stats(n_boxes: int, model: CFMNN, log: list, evicted: int, removed: int, **extra) -> dict:
    cases = np.bincount([e["case"] for e in log], minlength=5)
    return {"boxes_start": int(n_boxes), "boxes_end": int(model.n_boxes_),
            "contractions": len(log), "rows_evicted": int(evicted), "boxes_removed": int(removed),
            **{f"case_{c}": int(cases[c]) for c in (1, 2, 3, 4)}, **extra}


def contract_all(model: CFMNN, X_lab, max_sweeps: int = 100,
                 rule: str = "simpson") -> tuple[CFMNN, dict, list]:
    """Contract every overlap between boxes of different classes. Returns a new model.

    Pairs are taken in box order (j ascending, then k > j ascending), each tested on the
    corners as they are at that moment. A contraction can turn a shared face into a strict
    overlap, so the sweep is repeated until it changes nothing. Members, Sum, Count and
    centroids are recomputed once at the end, and empty boxes are removed then.

    Returns (model, counts, log): one log entry per contraction.
    """
    _require_fixed(model)
    find, contract = RULES[rule]
    model = copy.deepcopy(model)
    X_lab = np.asarray(X_lab, dtype=float)
    V, W, cls = model.V_, model.W_, model.box_class_
    H = model.n_boxes_
    log, touched, sweeps = [], set(), 0
    for sweeps in range(1, max_sweeps + 1):
        n_before = len(log)
        for j in range(H - 1):
            later = np.arange(j + 1, H)
            for k in find(V, W, j, later[cls[later] != cls[j]]):
                info = contract(V, W, j, int(k))   # re-tested: j may have moved already
                if info is not None:
                    touched.update(info["changed"])
                    log.append({**info, "sweep": sweeps})
        if len(log) == n_before:
            break
    else:
        raise RuntimeError(f"contraction did not settle in {max_sweeps} sweeps")
    evicted = removed = 0
    for k in sorted(touched, reverse=True):             # highest first: removal shifts later boxes
        cut, gone = refresh_box(model, k, X_lab)
        evicted += cut
        removed += gone
    return model, _stats(H, model, log, evicted, removed, sweeps=sweeps), log


def contract_at_errors(model: CFMNN, X_lab, y_lab, budget: int | None = None, seed: int = 0,
                       rule: str = "simpson") -> tuple[CFMNN, dict, list]:
    """Contract only where the model gets a training row wrong. No gate. Returns a new model.

    The training errors of the starting model are taken in a seeded random order, the same
    order as ``active.run_self_repair`` with the same seed; ``budget`` keeps the first that
    many (None: all). For each row, on the model as it is by then:

    * skipped if the model now gets it right;
    * skipped if the winning box does not contain it, or no box of its own class does (this
      can happen once earlier contractions have evicted rows);
    * otherwise the rule's overlap test and contraction on the pair (winning wrong-class box, the best
      own-class box containing the row). The pair is skipped if the test finds no overlap.

    Every contraction is kept. The changed boxes are refreshed at once, so the next row sees
    the new centroids. Returns (model, counts, log): one log entry per row tried.
    """
    _require_fixed(model)
    contract = RULES[rule][1]
    model = copy.deepcopy(model)
    X_lab = np.asarray(X_lab, dtype=float)
    H = model.n_boxes_
    t_all = np.searchsorted(model.classes_, np.asarray(y_lab))
    errors = np.flatnonzero(model.predict_details(X_lab).cls != t_all)
    order = np.random.default_rng(seed).permutation(errors)
    if budget is not None:
        order = order[:budget]
    log, done, evicted, removed = [], [], 0, 0
    for p in order:
        x, t = X_lab[p], int(t_all[p])
        det = model.predict_details(x[None])
        rec = {"train_index": int(p), "contracted": False}
        if int(det.cls[0]) == t:
            log.append({**rec, "reason": "already right"})
            continue
        k, j = int(det.winner[0]), int(det.best_box[0, t])
        if not contains(model, k, x):
            log.append({**rec, "reason": "not inside the winning box"})
            continue
        if j < 0 or not contains(model, j, x):
            log.append({**rec, "reason": "no own-class box contains the row"})
            continue
        info = contract(model.V_, model.W_, j, k)
        if info is None:
            log.append({**rec, "reason": "no overlap by the test"})
            continue
        cut_here = 0
        for b in sorted(info["changed"], reverse=True):
            cut, gone = refresh_box(model, b, X_lab)
            cut_here += cut
            removed += gone
        evicted += cut_here
        done.append(info)
        log.append({**rec, **info, "contracted": True, "reason": "", "rows_evicted": cut_here,
                    "x_correct_after": bool(model.predict_details(x[None]).cls[0] == t)})
    reasons = [e["reason"] for e in log]
    return model, _stats(H, model, done, evicted, removed, errors_start=int(len(errors)),
                         errors_tried=len(log),
                         skipped_already_right=reasons.count("already right"),
                         skipped_no_overlap=reasons.count("no overlap by the test"),
                         skipped_other=sum(r.startswith(("not inside", "no own")) for r in reasons)), log


# ------------------------------------------------------------------ measures
def rows_without_box(model: CFMNN, n_rows: int) -> int:
    """Training rows that are a member of no box (evicted by a shrink or a contraction)."""
    return int(n_rows - sum(len(m) for m in model.members_))


def rows_in_wrong_class_box(model: CFMNN, X, y) -> int:
    """Rows that lie inside (or on the face of) at least one box of another class."""
    X = np.asarray(X, dtype=float)
    t = np.searchsorted(model.classes_, np.asarray(y))
    hit = np.zeros(len(X), dtype=bool)
    step = model._chunk_size()
    for s in range(0, len(X), step):
        Xc = X[s:s + step, None, :]
        inside = ((Xc >= model.V_) & (Xc <= model.W_)).all(axis=2)
        hit[s:s + step] = (inside & (model.box_class_[None, :] != t[s:s + step, None])).any(axis=1)
    return int(hit.sum())
