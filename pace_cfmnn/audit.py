"""Auditing box edits: which predictions an edit can change, and what did change.

CFMNN predicts with the box of highest membership; boxes within ``tie_tol`` of
the top are tied and the nearest centroid wins (``CFMNN.predict_details``). An
edit changes only some boxes (a shrink changes one box or removes it; an add
grows one box or appends a new one). So the prediction at x can change only if
a changed box is in x's tied top group before the edit or after it: otherwise
the top value, the tied group and the centroids that decide it are all the
same. ``impact_set`` returns that set of rows: a certificate, computed from the
two models alone, that contains every prediction the edit changed.

The tight certificate (``tight=True``) also drops the rows whose top groups,
before and after, hold boxes of one class only: the prediction is that class
both times. It is just as safe and usually much smaller.
"""

from __future__ import annotations

import numpy as np

from .cfmnn_core import CFMNN, feature_terms


def changed_boxes(before: CFMNN, after: CFMNN, removed: int | None = None):
    """Indices of changed boxes, in ``before`` and in ``after``.

    ``removed`` is the index (in ``before``) of a box the edit deleted, if any.
    Boxes keep their order, removal shifts later boxes up by one, and new boxes
    are appended (``CFMNN._remove_box``, ``CFMNN.add_points``).
    """
    kept = [i for i in range(before.n_boxes_) if i != removed]
    if after.n_boxes_ < len(kept):
        raise ValueError("after has fewer boxes than the edit can explain")
    ch_before = [] if removed is None else [int(removed)]
    ch_after = []
    for a, b in enumerate(kept):
        same = (np.array_equal(before.V_[b], after.V_[a]) and np.array_equal(before.W_[b], after.W_[a])
                and np.array_equal(before.centroid_[b], after.centroid_[a])
                and before.box_class_[b] == after.box_class_[a])
        if not same:
            ch_before.append(b)
            ch_after.append(a)
    ch_after += list(range(len(kept), after.n_boxes_))
    return np.array(ch_before, dtype=np.int64), np.array(ch_after, dtype=np.int64)


def top_groups(model: CFMNN, X):
    """The tied top group of every row, shape (n, H), and the predicted class index.

    The same computation as ``CFMNN.predict_details``, in one pass.
    """
    X = model._check_X(X)
    n, H = len(X), model.n_boxes_
    tied = np.zeros((n, H), dtype=bool)
    cls = np.full(n, -1, dtype=np.int64)
    if H == 0:
        return tied, cls
    tol = float(model.tie_tol)
    step = model._chunk_size()
    for s in range(0, n, step):
        Xc = X[s:s + step]
        M = feature_terms(Xc[:, None, :], model.V_, model.W_, model.gamma).mean(axis=2)
        D = ((Xc[:, None, :] - model.centroid_[None, :, :]) ** 2).sum(axis=2)
        t = M >= M.max(axis=1, keepdims=True) - tol
        tied[s:s + step] = t
        cls[s:s + step] = model.box_class_[np.argmin(np.where(t, D, np.inf), axis=1)]
    return tied, cls


def in_top_group(model: CFMNN, X, boxes) -> np.ndarray:
    """True where any of ``boxes`` is within ``tie_tol`` of the highest membership at x."""
    boxes = np.asarray(boxes, dtype=np.int64)
    tied, _ = top_groups(model, X)
    return tied[:, boxes].any(axis=1) if len(boxes) else np.zeros(len(tied), dtype=bool)


def certify(before: CFMNN, after: CFMNN, tied_before, tied_after, removed: int | None = None):
    """(certificate, tight certificate) from the top groups of both models."""
    ch_b, ch_a = changed_boxes(before, after, removed)
    hit = tied_before[:, ch_b].any(axis=1) | tied_after[:, ch_a].any(axis=1)
    classes = np.zeros((len(hit), len(before.classes_)), dtype=bool)
    for c in range(len(before.classes_)):
        classes[:, c] = (tied_before[:, before.box_class_ == c].any(axis=1)
                         | tied_after[:, after.box_class_ == c].any(axis=1))
    return hit, hit & (classes.sum(axis=1) > 1)


def impact_set(before: CFMNN, after: CFMNN, X, removed: int | None = None,
               tight: bool = False) -> np.ndarray:
    """Rows of X whose prediction the edit before -> after can have changed (a superset)."""
    loose, strict = certify(before, after, top_groups(before, X)[0], top_groups(after, X)[0], removed)
    return strict if tight else loose


def _tables(model: CFMNN, X):
    """Memberships, squared centroid distances and tied top groups, each (n, H)."""
    X = model._check_X(X)
    n, H = len(X), model.n_boxes_
    M, D = np.empty((n, H)), np.empty((n, H))
    step = model._chunk_size()
    for s in range(0, n, step):
        Xc = X[s:s + step]
        M[s:s + step] = feature_terms(Xc[:, None, :], model.V_, model.W_, model.gamma).mean(axis=2)
        D[s:s + step] = ((Xc[:, None, :] - model.centroid_[None, :, :]) ** 2).sum(axis=2)
    tied = M >= M.max(axis=1, keepdims=True) - float(model.tie_tol)
    return M, D, tied


def _unchanged_winner(before: CFMNN, after: CFMNN, X, removed: int | None = None):
    """Rows where one unchanged box is the nearest-centroid winner of the tied group both before
    and after, and which changed boxes could still beat it.

    Returns (certain, beaters): ``certain`` is True where the tied groups agree on every unchanged
    box and hold at least one; ``beaters`` is (n, c) True where a changed box (c of them: those of
    ``before`` first, then those of ``after``) is in its model's tied group at a squared centroid
    distance not above the unchanged winner's, so it can win; ``cls`` is the class of each changed
    box and ``u_cls`` the class of the unchanged winner.
    """
    _, Db, Tb = _tables(before, X)
    _, Da, Ta = _tables(after, X)
    ch_b, ch_a = changed_boxes(before, after, removed)
    ub = np.setdiff1d(np.arange(before.n_boxes_), ch_b)
    ua = np.setdiff1d(np.arange(after.n_boxes_), ch_a)
    Sb, Sa = Tb[:, ub], Ta[:, ua]
    certain = (Sb == Sa).all(axis=1) & Sb.any(axis=1)
    rows = np.arange(len(Sb))
    Du = np.where(Sb, Db[:, ub], np.inf)
    iu = np.argmin(Du, axis=1) if Du.shape[1] else np.zeros(len(Sb), dtype=np.int64)
    du = Du[rows, iu] if Du.shape[1] else np.full(len(Sb), np.inf)
    u_cls = before.box_class_[ub][iu] if Du.shape[1] else np.full(len(Sb), -1)
    dist = np.concatenate([np.where(Tb[:, ch_b], Db[:, ch_b], np.inf),
                           np.where(Ta[:, ch_a], Da[:, ch_a], np.inf)], axis=1)
    cls = np.concatenate([before.box_class_[ch_b], after.box_class_[ch_a]])
    return certain, dist <= du[:, None], cls, u_cls


def certify_centroid(before: CFMNN, after: CFMNN, X, removed: int | None = None) -> np.ndarray:
    """A tighter certificate that uses the centroid tie-break inside the tied group.

    Starts from the tight certificate (``certify``). A row is also dropped when the unchanged
    boxes of its tied group are the same before and after, and every changed box in a tied group
    that is at least as near as the unchanged winner is of the winner's class: the predicted
    class is then that class both times. Sound by construction (the winner is the nearest tied
    box, ties to the lowest index); the tests check it by brute force, and the run counts misses.
    """
    tied_b, tied_a = _tables(before, X)[2], _tables(after, X)[2]
    _, tight = certify(before, after, tied_b, tied_a, removed)
    certain, beaters, cls, u_cls = _unchanged_winner(before, after, X, removed)
    safe = certain & ~(beaters & (cls[None, :] != u_cls[:, None])).any(axis=1)
    return tight & ~safe


def _flip_exposed(model: CFMNN, X, values, boxes) -> np.ndarray:
    """Rows where any of ``boxes`` can matter to PACE-flip's output at x.

    PACE-flip leaves a box out when its ceiling (the largest membership it can reach by replacing
    one feature, from ``values``) is below the floor of its group's strongest box (the group is
    the predicted class A, or all other classes), minus tie_tol and 1e-12 (``PaceFlip._chunk``).
    Such a box can never be its group's maximum, the winner or tied with it, under any
    replacement. The output also reads the model at x itself: the tied group (the predicted
    class) and the two group maxima. So a box is exposed if some feature keeps it, if it is in
    the tied top group at x, or if no other box of its group reaches its membership at x.
    """
    from .explain import PaceFlip

    X = model._check_X(X)
    n, d = X.shape
    out = np.zeros(n, dtype=bool)
    boxes = np.asarray(boxes, dtype=np.int64)
    if len(boxes) == 0:
        return out
    pf = PaceFlip(model, values, n_quantiles=None)
    tol, bc = float(model.tie_tol), model.box_class_
    T = feature_terms(X[:, None, :], model.V_, model.W_, model.gamma)          # (n, H, d)
    M = T.mean(axis=2)
    D = ((X[:, None, :] - model.centroid_[None, :, :]) ** 2).sum(axis=2)
    tied = M >= M.max(axis=1, keepdims=True) - tol
    a = bc[np.argmin(np.where(tied, D, np.inf), axis=1)]
    isA = bc[None, :] == a[:, None]
    c = M[:, :, None] - T / d
    floor = c + pf._floor
    floor_group = np.where(isA[:, :, None],
                           np.where(isA[:, :, None], floor, -np.inf).max(axis=1)[:, None, :],
                           np.where(isA[:, :, None], -np.inf, floor).max(axis=1)[:, None, :])
    kept = ((c + pf._ceil) >= floor_group - (tol + 1e-12)).any(axis=2)           # (n, H)
    for k in boxes:
        same = isA == isA[:, k][:, None]
        others = np.where(same, M, -np.inf)
        others[:, k] = -np.inf
        covered = others.max(axis=1) >= M[:, k]
        out |= kept[:, k] | tied[:, k] | ~covered
    return out


def explanation_certificate(before: CFMNN, after: CFMNN, X, values, removed: int | None = None):
    """Rows whose PACE-flip and PACE-decomposition explanations an edit can have changed.

    Returns (flip, decomposition), two boolean arrays; False means certainly unchanged.
    ``values`` are PACE-flip's replacement values (the quantiles of its background,
    ``PaceFlip.values_``). PACE-SHAP has its own: ``pace_shap_certificate`` (Proposition 7).

    * PACE-flip: a row is certified unchanged if no changed box, before or after the edit, can
      matter to the output (``_flip_exposed``): the unchanged boxes then give every group maximum,
      tie and prediction, so the output is the same.
    * PACE-decomposition reads only the winning box's terms, so it is also unchanged when the
      same unchanged box wins before and after. The decomposition set is cut from the flip set,
      so it is a subset of it.
    """
    ch_b, ch_a = changed_boxes(before, after, removed)
    flip = _flip_exposed(before, X, values, ch_b) | _flip_exposed(after, X, values, ch_a)
    certain, beaters, _, _ = _unchanged_winner(before, after, X, removed)
    return flip, flip & ~(certain & ~beaters.any(axis=1))


def prediction_churn(y_true, pred_before, pred_after) -> dict:
    """Share of rows whose prediction changed, and how many were fixed or broken."""
    y_true, pb, pa = (np.asarray(a) for a in (y_true, pred_before, pred_after))
    changed = pb != pa
    return {"changed_share": float(changed.mean()),
            "fixed": int(np.sum((pb != y_true) & (pa == y_true))),
            "broken": int(np.sum((pb == y_true) & (pa != y_true))),
            "changed": int(changed.sum())}


def top_k_sets(scores: np.ndarray, k: int = 3) -> list[frozenset]:
    """The k highest-scoring features of each row (stable order on ties)."""
    order = np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]
    return [frozenset(r.tolist()) for r in order]


def explanation_churn(scores_before: np.ndarray, scores_after: np.ndarray, k: int = 3) -> dict:
    """Share of rows whose attribution changed at all, and whose top-k feature set changed."""
    sb, sa = np.asarray(scores_before), np.asarray(scores_after)
    if len(sb) == 0:
        return {"any_change_share": float("nan"), "topk_change_share": float("nan"), "n": 0}
    any_change = np.abs(sb - sa).max(axis=1) > 1e-12
    topk = np.array([a != b for a, b in zip(top_k_sets(sb, k), top_k_sets(sa, k))])
    return {"any_change_share": float(any_change.mean()), "topk_change_share": float(topk.mean()),
            "n": int(len(sb))}


# ------------------------------------------------- tough review: B6 and B7
def _shap_exposed(model: CFMNN, X, background, boxes) -> np.ndarray:
    """Rows where any of ``boxes`` survives PACE-SHAP's pruning for some background row.

    A pruned box is never its group's maximum under any coalition (``PaceShap._prune``), so the
    margin of every coalition, and hence every Shapley value, does not depend on it.
    """
    from .explain import PaceShap

    X = model._check_X(X)
    out = np.zeros(len(X), dtype=bool)
    boxes = np.asarray(boxes, dtype=np.int64)
    if len(boxes) == 0 or model.n_boxes_ == 0:
        return out
    ps = PaceShap(model, background, exact_max_d=0, prune=True)
    cls = model.predict_details(X).cls
    for p in range(len(X)):
        j_i = ps._prepare(X[p], int(cls[p]))[2]
        out[p] = bool(np.isin(j_i, boxes).any())
    return out


def pace_shap_certificate(before: CFMNN, after: CFMNN, X, background,
                          removed: int | None = None) -> np.ndarray:
    """Rows whose PACE-SHAP values an edit can have changed (a superset; B7).

    False means the values are certainly the same: the predicted class A is the same before and
    after, and no changed box survives pruning, before the edit or after it, for any background
    row. Then every coalition's margin is the maximum over the same unchanged boxes, so exact
    values are identical, and sampled values are identical too when computed with the same seed
    on the same points in the same order (the feature orders do not depend on the boxes). The
    same background rows must be used before and after.
    """
    X = before._check_X(X)
    ch_b, ch_a = changed_boxes(before, after, removed)
    out = before.predict_details(X).cls != after.predict_details(X).cls
    out |= _shap_exposed(before, X, background, ch_b)
    out |= _shap_exposed(after, X, background, ch_a)
    return out


def churn_bound(certified, alpha: float = 0.05) -> dict:
    """A label-free upper bound on how many predictions an edit can change (B6, part 1).

    By Proposition 4, a prediction can change only inside the certificate, so on any data
    distribution the share of changed predictions, the negative flip rate and the change in
    accuracy are all at most the probability mass of the certificate. ``certified`` marks which of
    m unlabelled rows fall in it (for several edits, the union of their certificates); the
    one-sided Clopper-Pearson bound then holds with probability at least 1 - alpha over the draw
    of the rows, provided they were not used to choose the edit.
    """
    from scipy.stats import beta

    c = np.asarray(certified, dtype=bool)
    m, k = int(len(c)), int(c.sum())
    if m == 0:
        raise ValueError("need at least one unlabelled row")
    upper = 1.0 if k == m else float(beta.ppf(1.0 - alpha, k + 1, m - k))
    return {"m": m, "k": k, "share": k / m, "upper": upper, "alpha": float(alpha)}


def training_safe_shrink(before: CFMNN, after: CFMNN, X_train, y_idx, box: int,
                         removed: bool = False) -> dict:
    """Condition S: a shrink of ``box`` that cannot break a correct training prediction (B6).

    S holds when the edit changes only this box and keeps its centroid, and:

    (i) no training row of the box's class lies in the part of the old box the shrink cut away;
    (ii) no training row of the box's class outside the old box, whose membership in the box fell,
         was won by the box before the edit;
    (iii) no training row predicted correctly before and won by another box has a higher
         membership in this box than in its winner.

    Under S only this box's membership falls. Take a row predicted correctly before. If the box
    won it, the row is of the box's class, and (i) and (ii) leave it inside the new box (where its
    membership stays 1) or with its membership unchanged. If another box won it, (iii) says the
    top membership is reached by a box other than this one, so it stays the same after the edit;
    the tied top group can only lose this box, and the winner, the nearest centroid in the old
    group, is still the nearest in the new one. So no training row predicted correctly before is
    wrong after. Clause (iii) matters only for ties within ``tie_tol``: with ``tie_tol = 0`` the
    winner always has the top membership and (iii) always holds. The proof needs no other
    property of the model (Proposition 1 is not assumed).
    """
    X = before._check_X(X_train)
    y = np.asarray(y_idx)
    c = int(before.box_class_[box])
    own = y == c
    ch_b, ch_a = changed_boxes(before, after, box if removed else None)
    only_box = (list(ch_b) == [box] or len(ch_b) == 0) and len(ch_a) <= (0 if removed else 1)
    inside_old = (before.V_[box] <= X).all(axis=1) & (X <= before.W_[box]).all(axis=1)
    m_old = feature_terms(X, before.V_[box], before.W_[box], before.gamma).mean(axis=1)
    if removed:
        inside_new = np.zeros(len(X), dtype=bool)
        m_new = np.full(len(X), -np.inf)
        same_centroid = True
    else:
        inside_new = (after.V_[box] <= X).all(axis=1) & (X <= after.W_[box]).all(axis=1)
        m_new = feature_terms(X, after.V_[box], after.W_[box], after.gamma).mean(axis=1)
        same_centroid = bool(np.array_equal(before.centroid_[box], after.centroid_[box]))
    slab = own & inside_old & ~inside_new
    det = before.predict_details(X)
    dropped = own & ~inside_old & (m_new < m_old)
    won = dropped & (det.winner == box)
    other = (det.cls == y) & (det.winner != box)                   # clause (iii)
    above = np.zeros(len(X), dtype=bool)
    if other.any():
        wk = det.winner[other]
        m_win = feature_terms(X[other], before.V_[wk], before.W_[wk], before.gamma).mean(axis=1)
        above[other] = m_old[other] > m_win
    safe = bool(only_box and same_centroid and not slab.any() and not won.any() and not above.any())
    return {"safe": safe, "n_slab": int(slab.sum()), "n_won_dropped": int(won.sum()),
            "n_above_winner": int(above.sum()), "same_centroid": same_centroid,
            "only_this_box": bool(only_box)}
